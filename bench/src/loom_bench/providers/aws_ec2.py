"""`aws_ec2` provider: one tagged EC2 GPU VM per host, engine in Docker, driven over SSM.

Safety rails, in the order they act:
1. the runner's budget guard accrues `Host.hourly_micros` (spot price x a safety
   multiplier, rounded up, plus the root EBS volume) and calls `teardown`. The
   multiplier stays inside accrual: `Host.as_run_micros`, the cost price recorded in
   provenance, is the observed spot (or on-demand) price plus the same volume;
2. user-data schedules `shutdown -h` at the TTL and the instance is launched with
   shutdown behaviour `terminate`, so it ends itself if the runner dies;
3. the reaper Lambda (`aws_reaper`) terminates anything managed past its TTL;
4. `bench reap` does the same from a laptop.

Nothing here handles a secret: the HF token is read on the host from Secrets
Manager by the start script and never passes through SSM parameters, user-data,
tags or logs.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from loom_bench.engines import docker_run_argv
from loom_bench.jobs import EvalJob, EvalJobResult, LoadJob, LoadJobResult, TokenizerSpec
from loom_bench.money import MICROS_PER_USD, Micros
from loom_bench.prices import HOURS_PER_MONTH, PriceBook, load_prices
from loom_bench.provenance import PriceBasis
from loom_bench.providers import aws_reaper, export_requirements
from loom_bench.providers.aws_ssm import parse_markers, render_script, run_script, stage_offsets
from loom_bench.providers.base import (
    Endpoint,
    EngineLaunch,
    Host,
    HostLost,
    HostRequest,
    SpotInterrupted,
)
from loom_bench.providers.weights import HostWeights, WeightsPlan, flat
from loom_bench.records import Market
from loom_bench.registry import PinnedImage, read_yaml
from loom_bench.tokenize import hf_cache_folder
from loom_bench.workloads.profiles import DatasetDownload

PROVIDER_NAME = "aws_ec2"
DLAMI_PARAMETER = (
    "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
)
MANAGED_TAG = aws_reaper.MANAGED_TAG
TTL_TAG = aws_reaper.TTL_TAG
EXPERIMENT_TAG = "loom:experiment"
OWNER_TAG = "loom:owner"
RESERVED_TAGS = frozenset({MANAGED_TAG, TTL_TAG})
SETTINGS_PATH_ENV = "LOOM_AWS_CONFIG"
SETTINGS_ENV_PREFIX = "LOOM_AWS_"

ENGINE_CONTAINER = "loom-engine"
STAGE_FILE = "/var/lib/loom/stages"
LOG_DIR = "/var/log/loom"
JOBS_DIR = "/var/lib/loom/jobs"
CLIENT_ENV_ROOT = "/var/lib/loom/clientenv"
CLIENT_UID = 10001
# Where the client container sees the model's HF cache folder (read-only).
CLIENT_MODEL_CACHE = "/models"
# Where the client container sees the host's pinned workload datasets (read-only).
CLIENT_DATA_MOUNT = "/data"
# run_job.sh's dataset variables for a job that reads no pinned dataset.
NO_DATASET: dict[str, str] = {
    "DATA_URL": "",
    "DATA_SHA256": "",
    "DATA_PATH": "",
    "DATA_DIR": "",
    "DATA_MOUNT": "",
}
# Weight download allowance on top of the engine's ready timeout (Llama 70B is ~141 GB).
DOWNLOAD_ALLOWANCE_S = 3600
# Run-time allowance for a job on top of its own time budget (wheel install, drain).
JOB_ALLOWANCE_S = 900
LOAD_JOB_CMD = ("job", "run")
EVAL_JOB_CMD = ("quality", "job")
# Package extra the eval client needs when a job runs lm-evaluation-harness tasks.
LMEVAL_EXTRA = "lmeval"
# Client container image, pinned by digest: python:3.12.15-slim-trixie (also tagged
# 3.12-slim), the multi-arch image index. Docker-Content-Digest of
# registry-1.docker.io/v2/library/python/manifests/3.12.15-slim-trixie, read 2026-10-05.
CLIENT_IMAGE = "python@sha256:02108f5d322dd89f1c9e552442c25acb0543dfdbc455693a5599624f20d9155d"
BOOT_TIMEOUT_S = 900
SSM_ONLINE_TIMEOUT_S = 900
CAPACITY_ERRORS = frozenset(
    {
        "InsufficientInstanceCapacity",
        "InsufficientHostCapacity",
        "InsufficientCapacity",
        "SpotMaxPriceTooLow",
        "Unsupported",
    }
)
SPOT_INTERRUPTION_CODES = frozenset(
    {"Server.SpotInstanceTermination", "Server.SpotInstanceShutdown"}
)
GONE_STATES = frozenset({"shutting-down", "terminated", "stopping", "stopped"})

_SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SafeId = Annotated[str, StringConstraints(pattern=_SAFE_ID)]
_SAFE_ID_RE = re.compile(_SAFE_ID)
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def utcnow() -> datetime:
    return datetime.now(UTC)


def ttl_tag_value(ttl_at: datetime) -> str:
    return ttl_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class AwsSettings(BaseModel):
    """Account-specific settings; `terraform output -raw aws_settings_yaml` fills them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region: str = "us-east-1"
    bucket: Annotated[str, StringConstraints(min_length=3)]
    instance_profile_name: Annotated[str, StringConstraints(min_length=1)]
    security_group_id: Annotated[str, StringConstraints(pattern=r"^sg-[0-9a-f]+$")]
    subnet_ids: Annotated[
        list[Annotated[str, StringConstraints(pattern=r"^subnet-[0-9a-f]+$")]],
        Field(min_length=1),
    ]
    hf_token_secret_name: Annotated[str, StringConstraints(min_length=1)]
    owner: SafeId
    name_prefix: SafeId = "loom-bench"
    dlami_ssm_parameter: str = DLAMI_PARAMETER
    # Pin one AMI (e.g. the one a smoke ran on) instead of the parameter's latest: the
    # DLAMI parameter moves about weekly, so a smoke and its real run can otherwise differ.
    ami_id: Annotated[str, StringConstraints(pattern=r"^ami-[0-9a-f]{8,17}$")] | None = None
    root_volume_gb: Annotated[int, Field(ge=50, le=2000)] = 200
    weights_dir: str = "/opt/dlami/nvme/loom-hf"
    # Root-owned, world-readable copies of pinned workload datasets: <sha256>/<file name>.
    data_dir: str = "/opt/dlami/nvme/loom-data"
    spot_price_multiplier: Annotated[Decimal, Field(ge=1)] = Decimal("1.25")
    max_ttl_s: Annotated[int, Field(gt=0)] = 8 * 3600
    wheel_path: Path | None = None
    client_image: PinnedImage = CLIENT_IMAGE  # repo@sha256:<digest>; tags are rejected
    poll_interval_s: Annotated[float, Field(gt=0)] = 5.0
    job_timeout_s: Annotated[int, Field(gt=0)] = 7200
    presign_expiry_s: Annotated[int, Field(gt=0, le=7 * 24 * 3600)] = 6 * 3600


def load_aws_settings(
    path: Path | str | None = None, env: Mapping[str, str] | None = None
) -> AwsSettings:
    """Settings from a YAML file (`path` or `$LOOM_AWS_CONFIG`), overridden by
    `LOOM_AWS_<FIELD>` environment variables (`LOOM_AWS_SUBNET_IDS` is comma-separated)."""
    env = os.environ if env is None else env
    path = path or env.get(SETTINGS_PATH_ENV)
    data: dict[str, Any] = dict(read_yaml(Path(path)) or {}) if path else {}
    for field in AwsSettings.model_fields:
        value = env.get(SETTINGS_ENV_PREFIX + field.upper())
        if value is None:
            continue
        data[field] = (
            [s.strip() for s in value.split(",") if s.strip()] if field == "subnet_ids" else value
        )
    return AwsSettings.model_validate(data)


def host_loss(
    instance: Mapping[str, Any] | None, host_id: str, launched_at: datetime, now: datetime
) -> HostLost | None:
    """The exception describing why `instance` can no longer serve, or None if it can."""
    elapsed = (now - launched_at).total_seconds()
    if instance is None:
        return HostLost(
            host_id,
            state="not-found",
            reason_code=None,
            reason_message=None,
            detected_at=now,
            seconds_since_launch=elapsed,
        )
    state = instance["State"]["Name"]
    if state not in GONE_STATES:
        return None
    reason = instance.get("StateReason") or {}
    code, message = reason.get("Code"), reason.get("Message")
    cls = SpotInterrupted if code in SPOT_INTERRUPTION_CODES else HostLost
    return cls(
        host_id,
        state=state,
        reason_code=code,
        reason_message=message,
        detected_at=now,
        seconds_since_launch=elapsed,
    )


def spot_hourly_micros(spot_price_usd: str, multiplier: Decimal) -> Micros:
    """Budget accrual rate for a spot price: price x multiplier, rounded up to a micro."""
    return math.ceil(Fraction(Decimal(spot_price_usd)) * MICROS_PER_USD * Fraction(multiplier))


def _tag_list(tags: Mapping[str, str]) -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def _error_code(e: ClientError) -> str:
    return str(e.response.get("Error", {}).get("Code", ""))


class AwsEc2Provider:
    name = PROVIDER_NAME

    def __init__(
        self,
        settings: AwsSettings,
        *,
        prices: PriceBook | None = None,
        ec2: Any = None,
        ssm: Any = None,
        s3: Any = None,
        wheel_path: Path | None = None,
        requirements: Callable[[str], str] = export_requirements,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.prices = prices if prices is not None else load_prices()
        self.ec2 = ec2 or boto3.client("ec2", region_name=settings.region)
        self.ssm = ssm or boto3.client("ssm", region_name=settings.region)
        self.s3 = s3 or boto3.client("s3", region_name=settings.region)
        self.wheel_path = wheel_path or settings.wheel_path
        self.requirements = requirements
        self._requirements: dict[str, str] = {}  # extra -> exported requirements.txt
        self.clock = clock
        self.sleep = sleep
        self.weights = HostWeights()  # what each instance's HF cache holds

    # -- provisioning -------------------------------------------------------

    def _validate_request(self, req: HostRequest) -> str:
        if req.cloud != "aws":
            raise ValueError(f"aws_ec2 needs cloud 'aws', got {req.cloud!r}")
        if req.region not in (None, self.settings.region):
            raise ValueError(
                f"request region {req.region} != settings region {self.settings.region}"
            )
        if req.market not in (Market.SPOT, Market.ON_DEMAND):
            raise ValueError(f"aws_ec2 supports spot and on_demand, not {req.market}")
        if not req.instance_type:
            raise ValueError("instance_type is required")
        if not 0 < req.ttl_s <= self.settings.max_ttl_s:
            raise ValueError(f"ttl_s must be in (0, {self.settings.max_ttl_s}], got {req.ttl_s}")
        experiment = req.tags.get(EXPERIMENT_TAG)
        if not experiment or not _SAFE_ID_RE.match(experiment):
            raise ValueError(f"tags[{EXPERIMENT_TAG!r}] must match {_SAFE_ID}")
        reserved = RESERVED_TAGS & req.tags.keys()
        if reserved:
            raise ValueError(f"tags {sorted(reserved)} are set by the provider")
        return experiment

    def _on_demand_micros(self, instance_type: str) -> Micros:
        quote = self.prices.instance_price(
            "aws", self.settings.region, instance_type, Market.ON_DEMAND, allow_unverified=True
        )
        return quote.per_hour

    def _ebs_micros(self, volume_gb: int) -> Micros:
        storage = self.prices.region("aws", self.settings.region).storage
        if storage is None:
            raise KeyError(f"no storage price for aws/{self.settings.region}")
        return math.ceil(Fraction(storage.per_gb_month * volume_gb, HOURS_PER_MONTH))

    def _volume_gb(self, req: HostRequest) -> int:
        return max(req.disk_gb, self.settings.root_volume_gb)

    def _as_run(
        self, req: HostRequest, rec: Mapping[str, Any] | None, az: str
    ) -> tuple[Micros | None, PriceBasis]:
        """The host's cost price (no safety multiplier) plus its root volume, rounded
        once (`PriceBook.with_storage`), and its basis. A spot host in an AZ with no
        observed spot price has no known as-run price."""
        assert req.instance_type is not None
        volume = self._volume_gb(req)
        region = self.settings.region
        if rec is not None:
            spot = Fraction(Decimal(rec["SpotPrice"])) * MICROS_PER_USD
            basis = PriceBasis(
                market=Market.SPOT,
                source="observed_spot",
                spot_price_usd=rec["SpotPrice"],
                observed_at=rec["Timestamp"],
                availability_zone=az,
                storage_gb=volume,
            )
            price = self.prices.with_storage("aws", region, spot, volume, allow_unverified=True)
            return price, basis
        if req.market is Market.SPOT:
            return None, PriceBasis(
                market=Market.SPOT, source="unobserved", availability_zone=az, storage_gb=volume
            )
        price = self.prices.replica_hourly_cost(
            "aws", region, req.instance_type, Market.ON_DEMAND, volume, allow_unverified=True
        ).per_hour
        basis = PriceBasis(
            market=Market.ON_DEMAND, source="prices_yaml", availability_zone=az, storage_gb=volume
        )
        return price, basis

    def _candidates(self, req: HostRequest, now: datetime) -> list[dict[str, Any]]:
        """Subnets to try in order, each with its AZ, accrual price and as-run price.

        Accrual = instance rate + root EBS rate, both rounded up. On-demand keeps the
        configured subnet order at the prices.yaml rate. Spot tries the cheapest AZ
        first at its current spot price x multiplier; an AZ with no spot price is
        tried last, accrued at the on-demand rate.
        """
        assert req.instance_type is not None
        on_demand = self._on_demand_micros(req.instance_type)
        ebs = self._ebs_micros(self._volume_gb(req))
        subnets = self.ec2.describe_subnets(SubnetIds=self.settings.subnet_ids)["Subnets"]
        az_of = {s["SubnetId"]: s["AvailabilityZone"] for s in subnets}
        ordered = [sid for sid in self.settings.subnet_ids if sid in az_of]
        latest: dict[str, dict[str, Any]] = {}
        if req.market is Market.SPOT:
            history = self.ec2.describe_spot_price_history(
                InstanceTypes=[req.instance_type],
                ProductDescriptions=["Linux/UNIX"],
                StartTime=now,
            )["SpotPriceHistory"]
            for rec in history:
                az = rec["AvailabilityZone"]
                if az not in latest or rec["Timestamp"] > latest[az]["Timestamp"]:
                    latest[az] = rec
        multiplier = self.settings.spot_price_multiplier
        spot: list[dict[str, Any]] = []
        fallback: list[dict[str, Any]] = []
        for sid in ordered:
            rec = latest.get(az_of[sid])
            if rec is None:
                market = "on_demand" if req.market is Market.ON_DEMAND else "on_demand_fallback"
                instance = on_demand
                basis: dict[str, Any] = {"source": "prices.yaml", "market": market}
            else:
                instance = spot_hourly_micros(rec["SpotPrice"], multiplier)
                basis = {
                    "source": "describe_spot_price_history",
                    "market": "spot",
                    "spot_price_usd": rec["SpotPrice"],
                    "price_timestamp": rec["Timestamp"].isoformat(),
                    "multiplier": str(multiplier),
                }
            basis.update(instance_micros_per_hour=instance, ebs_micros_per_hour=ebs)
            as_run, price_basis = self._as_run(req, rec, az_of[sid])
            cand = {
                "subnet_id": sid,
                "az": az_of[sid],
                "hourly_micros": instance + ebs,
                "accrual_basis": basis,
                "as_run_micros": as_run,
                "price_basis": price_basis,
            }
            (spot if rec is not None else fallback).append(cand)
        return sorted(spot, key=lambda c: c["hourly_micros"]) + fallback

    def _ami(self) -> tuple[str, str]:
        ami = (
            self.settings.ami_id
            or self.ssm.get_parameter(Name=self.settings.dlami_ssm_parameter)["Parameter"]["Value"]
        )
        image = self.ec2.describe_images(ImageIds=[ami])["Images"][0]
        return ami, image["RootDeviceName"]

    def user_data(self, ttl_at: datetime) -> str:
        return render_script(
            "user_data",
            TTL_EPOCH=int(ttl_at.timestamp()),
            STAGE_FILE=STAGE_FILE,
            CLIENT_UID=CLIENT_UID,
        )

    def _tags(self, req: HostRequest, experiment: str, ttl_at: datetime) -> dict[str, str]:
        return {
            "Name": f"{self.settings.name_prefix}-{experiment}",
            OWNER_TAG: self.settings.owner,
            **req.tags,
            EXPERIMENT_TAG: experiment,
            MANAGED_TAG: "true",
            TTL_TAG: ttl_tag_value(ttl_at),
        }

    def _run_instances_kwargs(
        self,
        req: HostRequest,
        *,
        ami: str,
        root_device: str,
        subnet_id: str,
        tags: dict[str, str],
        ttl_at: datetime,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "ImageId": ami,
            "InstanceType": req.instance_type,
            "MinCount": 1,
            "MaxCount": 1,
            "SubnetId": subnet_id,
            "SecurityGroupIds": [self.settings.security_group_id],
            "IamInstanceProfile": {"Name": self.settings.instance_profile_name},
            "InstanceInitiatedShutdownBehavior": "terminate",
            "MetadataOptions": {
                "HttpTokens": "required",
                "HttpPutResponseHopLimit": 1,
                "HttpEndpoint": "enabled",
                "InstanceMetadataTags": "disabled",
            },
            "BlockDeviceMappings": [
                {
                    "DeviceName": root_device,
                    "Ebs": {
                        "VolumeSize": self._volume_gb(req),
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                        "Encrypted": True,
                    },
                }
            ],
            "TagSpecifications": [
                {"ResourceType": "instance", "Tags": _tag_list(tags)},
                {"ResourceType": "volume", "Tags": _tag_list(tags)},
            ],
            "UserData": self.user_data(ttl_at),
            "ClientToken": uuid.uuid4().hex,
        }
        if req.market is Market.SPOT:
            kwargs["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            }
        return kwargs

    def _provision(self, req: HostRequest) -> Host:
        experiment = self._validate_request(req)
        now = self.clock()
        ttl_at = now + timedelta(seconds=req.ttl_s)
        tags = self._tags(req, experiment, ttl_at)
        ami, root_device = self._ami()
        candidates = self._candidates(req, now)
        if not candidates:
            raise ValueError("none of the configured subnets exist")
        errors: list[str] = []
        for cand in candidates:
            kwargs = self._run_instances_kwargs(
                req,
                ami=ami,
                root_device=root_device,
                subnet_id=cand["subnet_id"],
                tags=tags,
                ttl_at=ttl_at,
            )
            try:
                resp = self.ec2.run_instances(**kwargs)
            except ClientError as e:
                if _error_code(e) not in CAPACITY_ERRORS:
                    raise
                errors.append(f"{cand['az']}: {_error_code(e)}")
                continue
            inst = resp["Instances"][0]
            return Host(
                provider=PROVIDER_NAME,
                host_id=inst["InstanceId"],
                request=req,
                hourly_micros=cand["hourly_micros"],
                launched_at=now,
                ttl_at=ttl_at,
                as_run_micros=cand["as_run_micros"],
                price_basis=cand["price_basis"],
                info={
                    "region": self.settings.region,
                    "az": cand["az"],
                    "subnet_id": cand["subnet_id"],
                    "ami": ami,
                    "instance_type": req.instance_type,
                    "market": req.market.value,
                    "accrual_basis": cand["accrual_basis"],
                    "tags": tags,
                },
            )
        raise RuntimeError(f"no capacity for {req.instance_type}: {'; '.join(errors)}")

    async def provision(self, req: HostRequest) -> Host:
        return await asyncio.to_thread(self._provision, req)

    # -- host state -----------------------------------------------------------

    def _describe(self, host_id: str) -> dict[str, Any] | None:
        try:
            resp = self.ec2.describe_instances(InstanceIds=[host_id])
        except ClientError as e:
            if _error_code(e) == "InvalidInstanceID.NotFound":
                return None
            raise
        for reservation in resp["Reservations"]:
            for inst in reservation["Instances"]:
                return dict(inst)
        return None

    async def check_host(self, host: Host) -> dict[str, Any]:
        """Describe the instance; raise `SpotInterrupted`/`HostLost` if it is gone."""
        inst = await asyncio.to_thread(self._describe, host.host_id)
        lost = host_loss(inst, host.host_id, host.launched_at, self.clock())
        if lost is not None:
            raise lost
        assert inst is not None
        return inst

    async def _wait_until(
        self, host: Host, ready: Callable[[], Awaitable[bool]], what: str, timeout_s: float
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while not await ready():
            if loop.time() > deadline:
                raise TimeoutError(f"{host.host_id}: {what} not reached in {timeout_s:.0f}s")
            await self.sleep(self.settings.poll_interval_s)

    async def _running(self, host: Host) -> bool:
        inst = await self.check_host(host)
        return bool(inst["State"]["Name"] == "running")

    async def _ssm_online(self, host: Host) -> bool:
        await self.check_host(host)
        resp = await asyncio.to_thread(
            self.ssm.describe_instance_information,
            Filters=[{"Key": "InstanceIds", "Values": [host.host_id]}],
        )
        return any(i.get("PingStatus") == "Online" for i in resp["InstanceInformationList"])

    async def _run(self, host: Host, script: str, *, timeout_s: int, comment: str) -> str:
        async def check() -> None:
            await self.check_host(host)

        experiment = host.request.tags[EXPERIMENT_TAG]
        return await run_script(
            self.ssm,
            host.host_id,
            script,
            timeout_s=timeout_s,
            comment=comment,
            poll_s=self.settings.poll_interval_s,
            check_host=check,
            output_bucket=self.settings.bucket,
            output_prefix=f"ssm/{experiment}/{host.host_id}",
            sleep=self.sleep,
        )

    # -- engine -----------------------------------------------------------------

    def engine_script(self, launch: EngineLaunch, *, warm: bool, weights: WeightsPlan) -> str:
        cmd = docker_run_argv(
            launch, weights_dir=self.settings.weights_dir, container_name=ENGINE_CONTAINER
        )
        return render_script(
            "start_engine",
            WARM=int(warm),
            REGION=self.settings.region,
            HF_SECRET_ID=self.settings.hf_token_secret_name,
            IMAGE=launch.image,
            FETCH_WEIGHTS=flat(weights.fetch),
            CACHED_WEIGHTS=flat(weights.cached),
            WEIGHTS_DIR=self.settings.weights_dir,
            CONTAINER=ENGINE_CONTAINER,
            PORT=launch.port,
            SERVED_MODEL=launch.served_model,
            READY_TIMEOUT_S=math.ceil(launch.ready_timeout_s),
            LOG_DIR=LOG_DIR,
            STAGE_FILE=STAGE_FILE,
            ENGINE_ENV=[f"{k}={v}" for k, v in sorted(launch.env.items())],
            ENGINE_CMD=cmd,
        )

    async def start_engine(self, host: Host, launch: EngineLaunch, *, warm: bool) -> Endpoint:
        t0 = host.launched_at if not warm else self.clock()
        controller_stages: dict[str, float] = {}
        if not warm:
            await self._wait_until(host, lambda: self._running(host), "running", BOOT_TIMEOUT_S)
            controller_stages["instance_running"] = (self.clock() - t0).total_seconds()
            await self._wait_until(
                host, lambda: self._ssm_online(host), "SSM online", SSM_ONLINE_TIMEOUT_S
            )
            controller_stages["ssm_online"] = (self.clock() - t0).total_seconds()
        # Every checkpoint the engine loads must be on the host before it starts offline:
        # a warm restart onto a checkpoint no earlier start downloaded fetches it too.
        weights = self.weights.plan(host.host_id, launch, warm=warm)
        timeout = math.ceil(launch.ready_timeout_s) + (DOWNLOAD_ALLOWANCE_S if weights.fetch else 0)
        stdout = await self._run(
            host,
            self.engine_script(launch, warm=warm, weights=weights),
            timeout_s=timeout,
            comment=f"loom start {launch.engine} {launch.served_model}",
        )
        host_stages, system = parse_markers(stdout)
        self.weights.downloaded(host.host_id, weights)
        stages = {k: round(v, 3) for k, v in controller_stages.items()}
        stages.update(stage_offsets(host_stages, t0.timestamp()))
        info: dict[str, Any] = {
            **system,
            "instance_type": host.info.get("instance_type"),
            "az": host.info.get("az"),
            "ami": host.info.get("ami"),
        }
        if "gpus" in info:
            info["gpus"] = [g.strip() for g in str(info["gpus"]).split(",") if g.strip()]
            info["gpu_count"] = len(info["gpus"])
        usec = info.pop("ttl_shutdown_usec", None)
        if usec is not None and str(usec).isdigit():
            # The TTL backstop as armed on the host, and how long before the runner's TTL
            # it fires (user-data schedules it in whole minutes from its own start, so 0-60 s).
            shutdown_at = datetime.fromtimestamp(int(usec) / 1e6, UTC)
            info["ttl_shutdown_at"] = shutdown_at.isoformat()
            info["ttl_shutdown_lead_s"] = round((host.ttl_at - shutdown_at).total_seconds(), 1)
        base = f"http://127.0.0.1:{launch.port}"
        return Endpoint(
            base_url=f"{base}/v1",
            metrics_url=f"{base}/metrics",
            engine=launch.engine,
            served_model=launch.served_model,
            start_stages=stages,
            warm=warm,
            system=info,
        )

    async def stop_engine(self, host: Host) -> None:
        await self._run(
            host,
            render_script("stop_engine", CONTAINER=ENGINE_CONTAINER),
            timeout_s=300,
            comment="loom stop engine",
        )

    # -- jobs -------------------------------------------------------------------

    def _job_timeout_s(self, job: LoadJob) -> int:
        if job.duration_s is None:
            return self.settings.job_timeout_s
        budget = job.duration_s + job.warmup_s + job.request_timeout_s + job.drain_timeout_s
        return math.ceil(budget) + JOB_ALLOWANCE_S

    def _presign(self, method: str, key: str) -> str:
        return str(
            self.s3.generate_presigned_url(
                method,
                Params={"Bucket": self.settings.bucket, "Key": key},
                ExpiresIn=self.settings.presign_expiry_s,
            )
        )

    def job_prefix(self, host: Host, run_id: str) -> str:
        if not _SAFE_ID_RE.match(run_id):
            raise ValueError(f"run_id must match {_SAFE_ID}")
        return f"runs/{host.request.tags[EXPERIMENT_TAG]}/{run_id}/"

    def _stage(
        self,
        host: Host,
        run_id: str,
        body: str,
        *,
        command: tuple[str, ...],
        model_cache: Mapping[str, str],
        extras: str = "",
        sample_gpu: bool = False,
        data: Mapping[str, str] | None = None,
    ) -> tuple[str, str]:
        """Upload the job JSON, the wheel and its locked requirements; return (key prefix,
        rendered script)."""
        if self.wheel_path is None:
            raise ValueError("no bench wheel: set AwsSettings.wheel_path or pass wheel_path")
        wheel = Path(self.wheel_path)
        prefix = self.job_prefix(host, run_id)
        if extras not in self._requirements:
            self._requirements[extras] = self.requirements(extras)
        reqs = self._requirements[extras].encode()
        self.s3.put_object(
            Bucket=self.settings.bucket,
            Key=prefix + "job.json",
            Body=body.encode(),
            ContentType="application/json",
        )
        self.s3.put_object(
            Bucket=self.settings.bucket,
            Key=prefix + "requirements.txt",
            Body=reqs,
            ContentType="text/plain",
        )
        self.s3.upload_file(str(wheel), self.settings.bucket, prefix + wheel.name)
        script = render_script(
            "run_job",
            WORK_DIR=f"{JOBS_DIR}/{run_id}",
            ENV_ROOT=CLIENT_ENV_ROOT,
            CLIENT_IMAGE=self.settings.client_image,
            CLIENT_UID=CLIENT_UID,
            JOB_URL=self._presign("get_object", prefix + "job.json"),
            WHEEL_URL=self._presign("get_object", prefix + wheel.name),
            WHEEL_NAME=wheel.name,
            WHEEL_SHA256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            REQS_URL=self._presign("get_object", prefix + "requirements.txt"),
            REQS_SHA256=hashlib.sha256(reqs).hexdigest(),
            BENCH_CMD=command,
            RESULT_URL=self._presign("put_object", prefix + "result.json"),
            GPU_CSV_URL=self._presign("put_object", prefix + "gpu.csv") if sample_gpu else "",
            SAMPLE_GPU=int(sample_gpu),
            **model_cache,
            **(data or NO_DATASET),
        )
        return prefix, script

    def _host_dataset(self, workload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """The job's workload reading its pinned dataset where the client container sees
        it (`CLIENT_DATA_MOUNT/<sha256>/<file name>`), and the run_job.sh variables that
        fetch it once per host into `data_dir`, check it and mount it read-only. A
        workload without a `download` block is returned as is."""
        download = workload.get("download")
        if not download:
            return dict(workload), dict(NO_DATASET)
        spec = DatasetDownload.model_validate(download)
        rel = f"{spec.sha256}/{PurePosixPath(spec.filename).name}"
        return {**workload, "path": f"{CLIENT_DATA_MOUNT}/{rel}"}, {
            "DATA_URL": spec.url,
            "DATA_SHA256": spec.sha256,
            "DATA_PATH": f"{self.settings.data_dir}/{rel}",
            "DATA_DIR": self.settings.data_dir,
            "DATA_MOUNT": CLIENT_DATA_MOUNT,
        }

    def _host_tokenizer(
        self, spec: TokenizerSpec | None
    ) -> tuple[TokenizerSpec | None, dict[str, str]]:
        """`spec` pointed at the model snapshot the engine start downloaded on the host,
        and the run_job.sh variables that mount its cache folder read-only into the
        client container.

        The client container has no Hugging Face token, so an `hf` tokenizer (gated or
        not) always loads from that snapshot; the script fails if the host lacks it.
        """
        if spec is None or spec.kind != "hf":
            return spec, {"MODEL_CACHE_DIR": "", "MODEL_CACHE_MOUNT": "", "MODEL_REVISION": ""}
        repo, revision = spec.repo, spec.revision
        if repo is None or not _HF_REPO_RE.match(repo):
            raise ValueError(f"hf tokenizer repo must be org/name, got {repo!r}")
        if revision is None or not _COMMIT_RE.match(revision):
            raise ValueError(f"hf tokenizer {repo} needs a pinned commit, got {revision!r}")
        folder = hf_cache_folder(repo)
        mount = f"{CLIENT_MODEL_CACHE}/{folder}"
        local = spec.model_copy(update={"local_dir": f"{mount}/snapshots/{revision}"})
        return local, {
            # The engine start downloads with HF_HOME=weights_dir: the hub cache is hub/.
            "MODEL_CACHE_DIR": f"{self.settings.weights_dir}/hub/{folder}",
            "MODEL_CACHE_MOUNT": mount,
            "MODEL_REVISION": revision,
        }

    def _stage_job(self, host: Host, job: LoadJob) -> tuple[str, str]:
        tokenizer, model_cache = self._host_tokenizer(job.tokenizer)
        workload, data = self._host_dataset(job.workload)
        return self._stage(
            host,
            job.run_id,
            job.model_copy(update={"tokenizer": tokenizer, "workload": workload}).model_dump_json(),
            command=LOAD_JOB_CMD,
            model_cache=model_cache,
            sample_gpu=job.sample_gpu,
            data=data,
        )

    def _stage_eval(self, host: Host, job: EvalJob) -> tuple[str, str]:
        selected = [t for t in job.suite.tasks if job.tasks is None or t.name in job.tasks]
        lmeval = any(t.kind == "lm_eval" for t in selected)
        tokenizer, model_cache = self._host_tokenizer(job.tokenizer)
        return self._stage(
            host,
            job.run_id,
            job.model_copy(update={"tokenizer": tokenizer}).model_dump_json(),
            command=EVAL_JOB_CMD,
            model_cache=model_cache,
            extras=LMEVAL_EXTRA if lmeval else "",
        )

    def _fetch_result(self, prefix: str, job: LoadJob) -> LoadJobResult:
        body = self.s3.get_object(Bucket=self.settings.bucket, Key=prefix + "result.json")["Body"]
        result = LoadJobResult.model_validate_json(body.read())
        if job.sample_gpu:
            csv = self.s3.get_object(Bucket=self.settings.bucket, Key=prefix + "gpu.csv")["Body"]
            result = result.model_copy(update={"nvidia_smi_csv": csv.read().decode()})
        return result

    async def run_job(self, host: Host, job: LoadJob) -> LoadJobResult:
        prefix, script = await asyncio.to_thread(self._stage_job, host, job)
        await self._run(
            host,
            script,
            timeout_s=self._job_timeout_s(job),
            comment=f"loom job {job.run_id}",
        )
        return await asyncio.to_thread(self._fetch_result, prefix, job)

    def _fetch_eval_result(self, prefix: str) -> EvalJobResult:
        body = self.s3.get_object(Bucket=self.settings.bucket, Key=prefix + "result.json")["Body"]
        return EvalJobResult.model_validate_json(body.read())

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        """Run the eval in the client container on the host, where the engine listens.
        Bounded by `job_timeout_s` like a request-count load job."""
        prefix, script = await asyncio.to_thread(self._stage_eval, host, job)
        await self._run(
            host,
            script,
            timeout_s=self.settings.job_timeout_s + JOB_ALLOWANCE_S,
            comment=f"loom eval {job.run_id}",
        )
        return await asyncio.to_thread(self._fetch_eval_result, prefix)

    # -- teardown ------------------------------------------------------------------

    def _terminate(self, host_id: str) -> None:
        try:
            self.ec2.terminate_instances(InstanceIds=[host_id])
        except ClientError as e:
            if _error_code(e) != "InvalidInstanceID.NotFound":
                raise

    async def teardown(self, host: Host) -> None:
        await asyncio.to_thread(self._terminate, host.host_id)
        self.weights.forget(host.host_id)

    async def reap(self, now: datetime) -> list[str]:
        return await asyncio.to_thread(aws_reaper.reap, self.ec2, now)
