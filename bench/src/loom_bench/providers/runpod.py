"""`runpod` provider: one RunPod Secure Cloud pod per host, the engine image as the pod's
container, driven over direct SSH.

The pod is the host and its engine image is part of the host (`HostRequest.image`), so
cells on different images get different pods, each with an honest cold start; a warm
restart stops and restarts the engine process inside the same container.

Safety rails, in the order they act:
1. the runner's budget guard accrues `Host.hourly_micros`: the larger of the pod's API
   `costPerHr` and the prices.yaml on-demand price, plus its container disk. A pod whose
   API price is above prices.yaml x `max_price_ratio` is terminated before any work;
2. the pod's start command arms a TTL watchdog first, at an absolute epoch (a container
   restart cannot extend it), which terminates the pod with the pod-scoped key RunPod
   injects;
3. `bench reap` terminates managed pods past their TTL (there is no scheduled reaper).
Pods are only ever terminated, never stopped: a stopped pod keeps billing for its disk.

Secrets: the pod env carries the HF token only as the RunPod secret reference
`{{ RUNPOD_SECRET_<hf_secret_name> }}`, plus `LOOM_*` values and the runner's SSH
public key; never AWS credentials and never a URL. Job inputs and outputs move through
presigned S3 URLs that reach the pod only inside scripts on SSH stdin. Jobs and evals
run as an unprivileged uid that cannot read PID 1's environment; the engine start and
every job check that (`loom-sys job_isolation ok`).
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import itertools
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any

import boto3  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from loom_bench.engines import engine_process_argv
from loom_bench.experiment import RUNPOD_GPU_TYPE_IDS, RunpodProviderSpec
from loom_bench.jobs import EvalJob, EvalJobResult, LoadJob, LoadJobResult, TokenizerSpec
from loom_bench.money import Micros
from loom_bench.prices import HOURS_PER_MONTH, PriceBook, load_prices
from loom_bench.provenance import PriceBasis
from loom_bench.providers import export_requirements, runpod_layout, runpod_reaper
from loom_bench.providers.aws_ssm import parse_markers, render_script, stage_offsets
from loom_bench.providers.base import (
    Endpoint,
    EngineLaunch,
    EngineStartFailed,
    Host,
    HostLost,
    HostRequest,
)
from loom_bench.providers.runpod_api import (
    GRAPHQL_URL,
    REST_URL,
    RunpodApi,
    RunpodApiError,
    public_ssh,
    usd_to_micros,
)
from loom_bench.providers.runpod_ssh import (
    SSH_UNREACHABLE,
    OpenSshExec,
    RemoteCommandError,
    SshExec,
    SshTarget,
    generate_keypair,
    run_detached,
)
from loom_bench.records import Market
from loom_bench.registry import read_yaml
from loom_bench.tokenize import hf_cache_folder

PROVIDER_NAME = "runpod"
TEMPLATE_DIR = "runpod_scripts"
EXPERIMENT_TAG = "loom:experiment"
POD_NAME_TAG = "loom:pod-name"
SETTINGS_PATH_ENV = "LOOM_RUNPOD_CONFIG"
SETTINGS_ENV_PREFIX = "LOOM_RUNPOD_"
MANAGED_ENV = runpod_reaper.MANAGED_ENV
TTL_ENV = runpod_reaper.TTL_ENV
SSH_PUBKEY_ENV = "LOOM_SSH_PUBKEY"
GONE_STATES = frozenset({"TERMINATED", "EXITED"})
ENGINE_HOST = "127.0.0.1"
STOP_TIMEOUT_S = 60
# Weight download allowance on top of the engine's ready timeout (Llama 70B is ~141 GB).
DOWNLOAD_ALLOWANCE_S = 3600
# Run-time allowance for a job on top of its own time budget (Python + wheel install, drain).
JOB_ALLOWANCE_S = 900
LOAD_JOB_CMD = ("job", "run")
EVAL_JOB_CMD = ("quality", "job")
LMEVAL_EXTRA = "lmeval"

_SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
SafeId = Annotated[str, StringConstraints(pattern=_SAFE_ID)]
_SAFE_ID_RE = re.compile(_SAFE_ID)
_PREFIX = r"^[a-z0-9][a-z0-9-]{0,63}$"
_SECRET_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PINNED_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*[a-z0-9]@sha256:[0-9a-f]{64}$")
# RunPod documents no limit on dockerStartCmd; keep the rendered start command small.
MAX_START_CMD_BYTES = 8192
POD_NAME_MAX = 191  # RunPod's documented limit on `name`
# Waits between pod creates that RunPod refuses for lack of stock; the last repeats.
CAPACITY_BACKOFF_S = (30.0, 60.0, 120.0, 240.0)

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(UTC)


def _capacity_backoff() -> Iterator[float]:
    yield from CAPACITY_BACKOFF_S
    yield from itertools.repeat(CAPACITY_BACKOFF_S[-1])


def check_url(url: str) -> str:
    """A URL safe to place inside a double-quoted curl config line (`url = "..."`)."""
    if not url.startswith("https://") or any(c in url for c in '"\\\n\r'):
        raise ValueError("URL must be https and contain no quote, backslash or newline")
    return url


class RunpodSettings(BaseModel):
    """Laptop-side settings: `$LOOM_RUNPOD_CONFIG` (YAML), overridden by `LOOM_RUNPOD_*`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket: Annotated[str, StringConstraints(min_length=3)]
    s3_region: str = "us-east-1"
    owner: SafeId
    name_prefix: Annotated[str, StringConstraints(pattern=_PREFIX)] = "loom-bench"
    # RunPod secret holding the HF token; pods reference it, never its value.
    hf_secret_name: str = "hf_token"
    rest_url: str = REST_URL
    graphql_url: str = GRAPHQL_URL
    max_ttl_s: Annotated[int, Field(gt=0)] = 8 * 3600
    wheel_path: Path | None = None
    client_python_url: str = runpod_layout.CLIENT_PYTHON_URL
    client_python_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] = (
        runpod_layout.CLIENT_PYTHON_SHA256
    )
    # Also authorize the account's registered SSH key (RunPod's PUBLIC_KEY), for debugging.
    authorize_account_key: bool = True
    poll_interval_s: Annotated[float, Field(gt=0)] = 5.0
    job_timeout_s: Annotated[int, Field(gt=0)] = 7200
    presign_expiry_s: Annotated[int, Field(gt=0, le=7 * 24 * 3600)] = 6 * 3600
    pod_ready_timeout_s: Annotated[int, Field(gt=0)] = 900
    ssh_online_timeout_s: Annotated[int, Field(gt=0)] = 600
    # Refuse a pod whose API price is above prices.yaml x this.
    max_price_ratio: Annotated[Decimal, Field(ge=1)] = Decimal("1.25")
    # How long to keep retrying a pod create that RunPod refuses for lack of stock.
    # Nothing is billed while waiting; 0 fails on the first refusal.
    capacity_wait_s: Annotated[int, Field(ge=0)] = 1800
    # Fail an engine start whose log has not grown for this long before it is healthy
    # (874110b6 hung silently after NCCL init until the 1800 s ready timeout).
    engine_stall_s: Annotated[int, Field(gt=0)] = 600

    @field_validator("hf_secret_name")
    @classmethod
    def _not_aws(cls, v: str) -> str:
        if not _SECRET_NAME_RE.match(v):
            raise ValueError("hf_secret_name must be letters, digits and underscores")
        if v.lower().startswith("aws"):
            raise ValueError("pods must never reference the aws_* RunPod secrets")
        return v

    @field_validator("rest_url", "graphql_url", "client_python_url")
    @classmethod
    def _https(cls, v: str) -> str:
        return check_url(v)


def load_runpod_settings(
    path: Path | str | None = None, env: Mapping[str, str] | None = None
) -> RunpodSettings:
    """Settings from a YAML file (`path` or `$LOOM_RUNPOD_CONFIG`), overridden by
    `LOOM_RUNPOD_<FIELD>` environment variables."""
    env = os.environ if env is None else env
    path = path or env.get(SETTINGS_PATH_ENV)
    data: dict[str, Any] = dict(read_yaml(Path(path)) or {}) if path else {}
    for field in RunpodSettings.model_fields:
        value = env.get(SETTINGS_ENV_PREFIX + field.upper())
        if value is not None:
            data[field] = value
    return RunpodSettings.model_validate(data)


def pod_loss(
    pod: Mapping[str, Any] | None, host_id: str, launched_at: datetime, now: datetime
) -> HostLost | None:
    """Why `pod` can no longer serve, or None if it can."""
    state = "not-found" if pod is None else str(pod.get("desiredStatus") or "")
    if pod is not None and state not in GONE_STATES:
        return None
    return HostLost(
        host_id,
        state=state.lower(),
        reason_code=None,
        reason_message=None if pod is None else str(pod.get("lastStatusChange") or "") or None,
        detected_at=now,
        seconds_since_launch=(now - launched_at).total_seconds(),
    )


def _data_center(machine: object) -> tuple[str | None, str | None]:
    """(datacenter id, location) from a pod's `machine`. RunPod sometimes returns an
    empty or missing `dataCenterId` with only `location` (a country code such as "SE")."""
    if not isinstance(machine, Mapping):
        return None, None
    data_center = str(machine.get("dataCenterId") or "").strip() or None
    location = str(machine.get("location") or "").strip() or None
    return data_center, location


def _rm_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


class RunpodProvider:
    name = PROVIDER_NAME

    def __init__(
        self,
        settings: RunpodSettings,
        *,
        spec: RunpodProviderSpec,
        api: RunpodApi,
        prices: PriceBook | None = None,
        s3: Any = None,
        ssh: SshExec | None = None,
        wheel_path: Path | None = None,
        requirements: Callable[[str], str] = export_requirements,
        clock: Callable[[], datetime] = utcnow,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        ssh_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.spec = spec
        self.api = api
        self.prices = prices if prices is not None else load_prices()
        self.s3 = s3 or boto3.client("s3", region_name=settings.s3_region)
        self.ssh = ssh or OpenSshExec()
        self.wheel_path = wheel_path or settings.wheel_path
        self.requirements = requirements
        self._requirements: dict[str, str] = {}  # extra -> exported requirements.txt
        self.clock = clock
        self.sleep = sleep
        self._ssh_dir = ssh_dir
        self._key: tuple[Path, str] | None = None

    # -- SSH identity -----------------------------------------------------------

    def _ssh_root(self) -> Path:
        if self._ssh_dir is None:
            # Short path under /tmp: ControlMaster socket paths are limited to ~104 bytes.
            base = "/tmp" if os.path.isdir("/tmp") else None
            self._ssh_dir = Path(tempfile.mkdtemp(prefix="loom-rp-", dir=base))
            atexit.register(_rm_tree, self._ssh_dir)
        self._ssh_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._ssh_dir, 0o700)
        return self._ssh_dir

    def _keypair(self) -> tuple[Path, str]:
        """This runner's SSH key (one per provider instance, never written to results)."""
        if self._key is None:
            self._key = generate_keypair(self._ssh_root() / "key")
        return self._key

    def _target(self, host: Host, pod: Mapping[str, Any]) -> SshTarget:
        addr = public_ssh(pod)
        if addr is None:
            raise RuntimeError(f"{host.host_id}: no public SSH address yet")
        root = self._ssh_root()
        control = root / "cm"
        control.mkdir(mode=0o700, exist_ok=True)
        return SshTarget(
            host=addr[0],
            port=addr[1],
            key_path=self._keypair()[0],
            known_hosts=root / f"known_hosts-{host.host_id}",
            control_dir=control,
        )

    # -- provisioning -------------------------------------------------------------

    def _validate_request(self, req: HostRequest) -> str:
        if req.cloud != "runpod":
            raise ValueError(f"runpod needs cloud 'runpod', got {req.cloud!r}")
        if req.region not in (None, self.spec.cloud_type):
            raise ValueError(f"request region {req.region} != cloud type {self.spec.cloud_type}")
        if req.market is not Market.ON_DEMAND:
            raise ValueError(f"runpod pods are on_demand, not {req.market}")
        if not req.instance_type:
            raise ValueError("instance_type is required")
        if not req.image or not _PINNED_RE.match(req.image):
            raise ValueError(f"image must be pinned as repo@sha256:<digest>, got {req.image!r}")
        if req.disk_gb <= 0:
            raise ValueError("disk_gb (the container disk) must be positive")
        if not 0 < req.ttl_s <= self.settings.max_ttl_s:
            raise ValueError(f"ttl_s must be in (0, {self.settings.max_ttl_s}], got {req.ttl_s}")
        experiment = req.tags.get(EXPERIMENT_TAG)
        if not experiment or not _SAFE_ID_RE.match(experiment):
            raise ValueError(f"tags[{EXPERIMENT_TAG!r}] must match {_SAFE_ID}")
        if POD_NAME_TAG in req.tags:
            raise ValueError(f"tag {POD_NAME_TAG!r} is set by the provider")
        it = self.prices.instance("runpod", self.spec.cloud_type, req.instance_type)
        if it.gpu_count != req.gpus:
            raise ValueError(
                f"{req.instance_type} is priced for {it.gpu_count} GPUs, request has {req.gpus}"
            )
        return experiment

    def _gpu_type_id(self, instance_type: str) -> str:
        if self.spec.gpu_type_id:
            return self.spec.gpu_type_id
        gpu = self.prices.instance("runpod", self.spec.cloud_type, instance_type).gpu
        try:
            return RUNPOD_GPU_TYPE_IDS[gpu]
        except KeyError:
            raise ValueError(f"no RunPod GPU type id for {gpu}; set provider.gpu_type_id") from None

    def _disk_micros(self, disk_gb: int) -> Micros:
        storage = self.prices.region("runpod", self.spec.cloud_type).storage
        if storage is None:
            raise KeyError(f"no storage price for runpod/{self.spec.cloud_type}")
        return math.ceil(Fraction(storage.per_gb_month * disk_gb, HOURS_PER_MONTH))

    def pod_name(self, experiment: str, ttl_at: datetime) -> str:
        """`{prefix}-{experiment[:8]}-{ttl_epoch}-{nonce}`: the reaper reads the TTL back."""
        nonce = secrets.token_hex(3)
        name = f"{self.settings.name_prefix}-{experiment[:8]}-{int(ttl_at.timestamp())}-{nonce}"
        return name[:POD_NAME_MAX]

    def start_command(self, ttl_at: datetime) -> str:
        script = render_script(
            "pod_start",
            template_dir=TEMPLATE_DIR,
            TTL_EPOCH=int(ttl_at.timestamp()),
            STAGE_FILE=runpod_layout.STAGE_FILE,
            LOOM_ROOT=runpod_layout.LOOM_ROOT,
            SSH_DIR=runpod_layout.SSH_DIR,
            SSHD_CONFIG=runpod_layout.SSHD_CONFIG,
            CTL_ROOT=runpod_layout.CTL_ROOT,
            JOB_UID=runpod_layout.JOB_UID,
            JOB_USER=runpod_layout.JOB_USER,
            JOB_HOME=runpod_layout.JOB_HOME,
            AUTHORIZE_ACCOUNT_KEY=int(self.settings.authorize_account_key),
            GRAPHQL_URL=self.settings.graphql_url,
        )
        if len(script.encode()) > MAX_START_CMD_BYTES:
            raise ValueError(f"pod start command is over {MAX_START_CMD_BYTES} bytes")
        return script

    def pod_env(self, experiment: str, ttl_at: datetime, pubkey: str) -> dict[str, str]:
        """The pod's environment: readable by anyone with the account key, so it holds
        the HF secret only as RunPod's reference, Loom's own values and a public key."""
        return {
            "HF_TOKEN": "{{ RUNPOD_SECRET_" + self.settings.hf_secret_name + " }}",
            MANAGED_ENV: "true",
            TTL_ENV: str(int(ttl_at.timestamp())),
            "LOOM_EXPERIMENT": experiment,
            "LOOM_OWNER": self.settings.owner,
            SSH_PUBKEY_ENV: pubkey,
        }

    def pod_body(
        self, req: HostRequest, *, name: str, experiment: str, ttl_at: datetime, pubkey: str
    ) -> dict[str, Any]:
        assert req.instance_type is not None
        body: dict[str, Any] = {
            "name": name,
            "imageName": req.image,
            "cloudType": self.spec.cloud_type.upper(),
            "computeType": "GPU",
            "interruptible": False,
            "gpuTypeIds": [self._gpu_type_id(req.instance_type)],
            "gpuCount": req.gpus,
            "allowedCudaVersions": list(self.spec.allowed_cuda_versions),
            "containerDiskInGb": req.disk_gb,
            # Explicit: the API's default is a 20 GB pod volume, billed while stopped too.
            # A per-sweep network volume would attach here (networkVolumeId) later.
            "volumeInGb": 0,
            "ports": ["22/tcp"],
            "dockerEntrypoint": ["bash", "-c"],
            "dockerStartCmd": [self.start_command(ttl_at)],
            "env": self.pod_env(experiment, ttl_at, pubkey),
        }
        if self.spec.data_center_ids:
            body["dataCenterIds"] = list(self.spec.data_center_ids)
        return body

    def _create(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """POST /pods; when the request fails without a clear refusal, terminate any pod
        it created anyway (found by its unique name) before re-raising."""
        try:
            return self.api.create_pod(body)
        except RunpodApiError as e:
            if e.status is None or e.status >= 500:
                for pod in self.api.find_pods_by_name(str(body["name"])):
                    self.api.terminate_pod(str(pod["id"]))
            raise

    def _provision(self, req: HostRequest) -> Host:
        experiment = self._validate_request(req)
        assert req.instance_type is not None
        region = self.spec.cloud_type
        listed = self.prices.instance_price(
            "runpod", region, req.instance_type, Market.ON_DEMAND, allow_unverified=True
        ).per_hour
        disk = self._disk_micros(req.disk_gb)
        now = self.clock()
        ttl_at = now + timedelta(seconds=req.ttl_s)
        name = self.pod_name(experiment, ttl_at)
        pubkey = self._keypair()[1]
        body = self.pod_body(req, name=name, experiment=experiment, ttl_at=ttl_at, pubkey=pubkey)
        pod = self._create(body)
        pod_id = str(pod["id"])
        created_s = (self.clock() - now).total_seconds()
        try:
            if pod.get("costPerHr") in (None, 0, ""):
                pod = self.api.get_pod(pod_id) or pod
            cost = pod.get("costPerHr")
            observed = usd_to_micros(cost) if cost not in (None, 0, "") else None
            if observed is not None and Fraction(observed) > Fraction(listed) * Fraction(
                self.settings.max_price_ratio
            ):
                raise RuntimeError(
                    f"pod {pod_id} costs {observed / 1e6:.4f} $/h, above prices.yaml "
                    f"{listed / 1e6:.4f} x {self.settings.max_price_ratio}: terminated; "
                    "update bench/prices.yaml or raise max_price_ratio"
                )
        except BaseException:
            self.api.terminate_pod(pod_id)
            raise
        machine = pod.get("machine") if isinstance(pod.get("machine"), Mapping) else {}
        data_center, location = _data_center(machine)
        # A bare location (e.g. "SE") is a country, not a datacenter: prefixed so it is
        # never read as one.
        zone = data_center or (f"location:{location}" if location else None)
        if observed is not None:
            as_run = self.prices.with_storage(
                "runpod", region, Fraction(observed), req.disk_gb, allow_unverified=True
            )
            basis = PriceBasis(
                market=Market.ON_DEMAND,
                source="observed_api",
                observed_at=now,
                availability_zone=zone,
                storage_gb=req.disk_gb,
            )
        else:
            as_run = self.prices.with_storage(
                "runpod", region, listed, req.disk_gb, allow_unverified=True
            )
            basis = PriceBasis(
                market=Market.ON_DEMAND,
                source="prices_yaml",
                availability_zone=zone,
                storage_gb=req.disk_gb,
            )
        instance = max(observed or 0, listed)
        tags = {**req.tags, POD_NAME_TAG: name}
        return Host(
            provider=PROVIDER_NAME,
            host_id=pod_id,
            request=req,
            hourly_micros=instance + disk,
            launched_at=now,
            ttl_at=ttl_at,
            as_run_micros=as_run,
            price_basis=basis,
            info={
                "pod_name": name,
                "cloud_type": region,
                "instance_type": req.instance_type,
                "gpu_type_id": body["gpuTypeIds"][0],
                "data_center": data_center,
                "data_center_location": location,
                "pod_created_s": round(created_s, 3),
                "accrual_basis": {
                    "api_cost_per_hr": None if cost in (None, 0, "") else str(cost),
                    "api_micros_per_hour": observed,
                    "prices_yaml_micros_per_hour": listed,
                    "disk_micros_per_hour": disk,
                },
                "tags": tags,
            },
        )

    async def provision(self, req: HostRequest) -> Host:
        """Create the pod. While RunPod has no matching GPU in stock, retry with backoff
        for up to `capacity_wait_s`; each attempt gets a fresh name and TTL, so time
        spent waiting never shortens the pod's life."""
        deadline = self.clock() + timedelta(seconds=self.settings.capacity_wait_s)
        for delay in _capacity_backoff():
            try:
                return await asyncio.to_thread(self._provision, req)
            except RunpodApiError as e:
                if not e.no_capacity:
                    raise
                left = (deadline - self.clock()).total_seconds()
                if left <= 0:
                    raise
                wait = min(delay, left)
                log.warning(
                    "RunPod has no %s in stock; retrying in %.0f s (%.0f s left)",
                    req.instance_type,
                    wait,
                    left,
                )
                await self.sleep(wait)
        raise AssertionError("unreachable")

    # -- host state ---------------------------------------------------------------

    async def check_host(self, host: Host) -> dict[str, Any]:
        """The pod; raise `HostLost` if it is gone, exited or terminated."""
        pod = await asyncio.to_thread(self.api.get_pod, host.host_id)
        lost = pod_loss(pod, host.host_id, host.launched_at, self.clock())
        if lost is not None:
            raise lost
        assert pod is not None
        return pod

    async def _wait_for_ssh_address(self, host: Host) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.pod_ready_timeout_s
        while True:
            pod = await self.check_host(host)
            if pod.get("desiredStatus") == "RUNNING" and public_ssh(pod) is not None:
                return pod
            if loop.time() > deadline:
                raise TimeoutError(
                    f"{host.host_id}: no public SSH address after "
                    f"{self.settings.pod_ready_timeout_s}s"
                )
            await self.sleep(self.settings.poll_interval_s)

    async def _wait_for_ssh(self, host: Host, target: SshTarget) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.ssh_online_timeout_s
        while True:
            res = await self.ssh.run(target, "echo loom-online\n", timeout_s=60)
            if res.returncode == 0 and "loom-online" in res.stdout:
                return
            if res.returncode != SSH_UNREACHABLE:
                raise RuntimeError(f"{host.host_id}: SSH probe exited {res.returncode}")
            await self.check_host(host)
            if loop.time() > deadline:
                raise TimeoutError(
                    f"{host.host_id}: SSH not reachable after {self.settings.ssh_online_timeout_s}s"
                )
            await self.sleep(self.settings.poll_interval_s)

    async def _run(self, host: Host, script: str, *, timeout_s: float, what: str) -> str:
        pod = await self.check_host(host)

        async def check() -> None:
            await self.check_host(host)

        return await run_detached(
            self.ssh,
            self._target(host, pod),
            script,
            what=f"{host.host_id}: {what}",
            timeout_s=timeout_s,
            poll_s=self.settings.poll_interval_s,
            check_host=check,
            ctl_root=runpod_layout.CTL_ROOT,
            sleep=self.sleep,
        )

    # -- engine -------------------------------------------------------------------

    def engine_script(self, launch: EngineLaunch, *, warm: bool) -> str:
        return render_script(
            "start_engine",
            template_dir=TEMPLATE_DIR,
            WARM=int(warm),
            MODEL_REPO=launch.model_repo,
            MODEL_REVISION=launch.model_revision,
            WEIGHTS_DIR=runpod_layout.WEIGHTS_DIR,
            PORT=launch.port,
            SERVED_MODEL=launch.served_model,
            READY_TIMEOUT_S=math.ceil(launch.ready_timeout_s),
            ENGINE_STALL_S=self.settings.engine_stall_s,
            LOG_DIR=runpod_layout.LOG_DIR,
            STAGE_FILE=runpod_layout.STAGE_FILE,
            ENGINE_ENV=[f"{k}={v}" for k, v in sorted(launch.env.items())],
            ENGINE_CMD=engine_process_argv(launch, host=ENGINE_HOST),
            ENGINE_PIDFILE=runpod_layout.ENGINE_PIDFILE,
            PROC1_ENVIRON=runpod_layout.PROC1_ENVIRON,
            JOB_UID=runpod_layout.JOB_UID,
            JOB_HOME=runpod_layout.JOB_HOME,
            SCAN_DIRS=list(runpod_layout.SECRET_SCAN_DIRS),
        )

    async def start_engine(self, host: Host, launch: EngineLaunch, *, warm: bool) -> Endpoint:
        if launch.image != host.request.image:
            raise ValueError(
                f"{host.host_id} was created for {host.request.image}, not {launch.image}: "
                "a pod serves one engine image"
            )
        t0 = host.launched_at if not warm else self.clock()
        controller: dict[str, float] = {}
        pod: dict[str, Any] | None = None
        if not warm:
            controller["pod_created"] = float(host.info.get("pod_created_s", 0.0))
            pod = await self._wait_for_ssh_address(host)
            controller["pod_running"] = (self.clock() - t0).total_seconds()
            await self._wait_for_ssh(host, self._target(host, pod))
            controller["ssh_online"] = (self.clock() - t0).total_seconds()
        timeout = math.ceil(launch.ready_timeout_s) + (0 if warm else DOWNLOAD_ALLOWANCE_S)
        try:
            stdout = await self._run(
                host,
                self.engine_script(launch, warm=warm),
                timeout_s=timeout,
                what=f"start {launch.engine} {launch.served_model}",
            )
        except RemoteCommandError as e:
            # The script prints the host's facts and stages as it goes: keep what it
            # reported before failing (the GPU topology, for a hang at NCCL init).
            failed_stages, failed_system = parse_markers(e.stdout)
            stages = {k: round(v, 3) for k, v in controller.items()}
            stages.update(stage_offsets(failed_stages, t0.timestamp()))
            raise EngineStartFailed(
                str(e), stages=stages, system={**failed_system, "pod_id": host.host_id}
            ) from e
        pod_stages, system = parse_markers(stdout)
        if system.get("job_isolation") != "ok":
            raise RuntimeError(f"{host.host_id}: engine start did not prove job isolation")
        stages = {k: round(v, 3) for k, v in controller.items()}
        stages.update(stage_offsets(pod_stages, t0.timestamp()))
        stages = dict(sorted(stages.items(), key=lambda kv: kv[1]))
        info: dict[str, Any] = {
            **system,
            "pod_id": host.host_id,
            "cloud_type": host.info.get("cloud_type"),
            "instance_type": host.info.get("instance_type"),
            "gpu_type_id": host.info.get("gpu_type_id"),
            "data_center": host.info.get("data_center") or system.get("data_center"),
        }
        location = host.info.get("data_center_location")
        if pod is not None:
            pod_dc, pod_location = _data_center(pod.get("machine"))
            info["data_center"] = pod_dc or info["data_center"]
            location = pod_location or location
        if not info["data_center"] and location:
            # Neither the API nor the pod's RUNPOD_DC_ID named the datacenter: record the
            # machine's location under its own key, never as a datacenter id.
            info["data_center_location"] = location
        if "gpus" in info:
            info["gpus"] = [g.strip() for g in str(info["gpus"]).split(",") if g.strip()]
        if "gpu_count" in info:
            try:
                info["gpu_count"] = int(info["gpu_count"])
            except ValueError:
                info["gpu_count"] = len(info.get("gpus", []))
        base = f"http://{ENGINE_HOST}:{launch.port}"
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
        script = render_script(
            "stop_engine",
            template_dir=TEMPLATE_DIR,
            ENGINE_PIDFILE=runpod_layout.ENGINE_PIDFILE,
            STOP_TIMEOUT_S=STOP_TIMEOUT_S,
        )
        await self._run(host, script, timeout_s=STOP_TIMEOUT_S * 3 + 60, what="stop engine")

    # -- jobs -----------------------------------------------------------------------

    def _job_timeout_s(self, job: LoadJob) -> int:
        if job.duration_s is None:
            return self.settings.job_timeout_s + JOB_ALLOWANCE_S
        budget = job.duration_s + job.warmup_s + job.request_timeout_s + job.drain_timeout_s
        return math.ceil(budget) + JOB_ALLOWANCE_S

    def _presign(self, method: str, key: str) -> str:
        url = self.s3.generate_presigned_url(
            method,
            Params={"Bucket": self.settings.bucket, "Key": key},
            ExpiresIn=self.settings.presign_expiry_s,
        )
        return check_url(str(url))

    def job_prefix(self, host: Host, run_id: str) -> str:
        if not _SAFE_ID_RE.match(run_id):
            raise ValueError(f"run_id must match {_SAFE_ID}")
        return f"runs/{host.request.tags[EXPERIMENT_TAG]}/{run_id}/"

    def _pod_tokenizer(
        self, spec: TokenizerSpec | None
    ) -> tuple[TokenizerSpec | None, dict[str, str]]:
        """`spec` pointed at the snapshot the engine start downloaded in the pod (read in
        place: the job has no HF token), and the run_job.sh variables that check it."""
        if spec is None or spec.kind != "hf":
            return spec, {"MODEL_CACHE_DIR": "", "MODEL_REVISION": ""}
        repo, revision = spec.repo, spec.revision
        if repo is None or not _HF_REPO_RE.match(repo):
            raise ValueError(f"hf tokenizer repo must be org/name, got {repo!r}")
        if revision is None or not _COMMIT_RE.match(revision):
            raise ValueError(f"hf tokenizer {repo} needs a pinned commit, got {revision!r}")
        cache = f"{runpod_layout.WEIGHTS_DIR}/hub/{hf_cache_folder(repo)}"
        local = spec.model_copy(update={"local_dir": f"{cache}/snapshots/{revision}"})
        return local, {"MODEL_CACHE_DIR": cache, "MODEL_REVISION": revision}

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
    ) -> tuple[str, str]:
        """Upload the job JSON, the wheel and its locked requirements; return (key prefix,
        rendered script). The presigned URLs exist only inside the returned script."""
        if self.wheel_path is None:
            raise ValueError("no bench wheel: set RunpodSettings.wheel_path or pass wheel_path")
        wheel = Path(self.wheel_path)
        prefix = self.job_prefix(host, run_id)
        if extras not in self._requirements:
            self._requirements[extras] = self.requirements(extras)
        reqs = self._requirements[extras].encode()
        bucket = self.settings.bucket
        self.s3.put_object(
            Bucket=bucket,
            Key=prefix + "job.json",
            Body=body.encode(),
            ContentType="application/json",
        )
        self.s3.put_object(
            Bucket=bucket, Key=prefix + "requirements.txt", Body=reqs, ContentType="text/plain"
        )
        self.s3.upload_file(str(wheel), bucket, prefix + wheel.name)
        script = render_script(
            "run_job",
            template_dir=TEMPLATE_DIR,
            WORK_DIR=f"{runpod_layout.JOBS_DIR}/{run_id}",
            ENV_ROOT=runpod_layout.CLIENT_ENV_ROOT,
            PYTHON_DIR=runpod_layout.PYTHON_DIR,
            PYTHON_URL=check_url(self.settings.client_python_url),
            PYTHON_SHA256=self.settings.client_python_sha256,
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
            PROC1_ENVIRON=runpod_layout.PROC1_ENVIRON,
            JOB_UID=runpod_layout.JOB_UID,
            JOB_HOME=runpod_layout.JOB_HOME,
            PROC_ROOT=runpod_layout.PROC_ROOT,
            **model_cache,
        )
        return prefix, script

    def _stage_job(self, host: Host, job: LoadJob) -> tuple[str, str]:
        tokenizer, model_cache = self._pod_tokenizer(job.tokenizer)
        return self._stage(
            host,
            job.run_id,
            job.model_copy(update={"tokenizer": tokenizer}).model_dump_json(),
            command=LOAD_JOB_CMD,
            model_cache=model_cache,
            sample_gpu=job.sample_gpu,
        )

    def _stage_eval(self, host: Host, job: EvalJob) -> tuple[str, str]:
        selected = [t for t in job.suite.tasks if job.tasks is None or t.name in job.tasks]
        lmeval = any(t.kind == "lm_eval" for t in selected)
        tokenizer, model_cache = self._pod_tokenizer(job.tokenizer)
        return self._stage(
            host,
            job.run_id,
            job.model_copy(update={"tokenizer": tokenizer}).model_dump_json(),
            command=EVAL_JOB_CMD,
            model_cache=model_cache,
            extras=LMEVAL_EXTRA if lmeval else "",
        )

    def _read(self, key: str) -> bytes:
        body = self.s3.get_object(Bucket=self.settings.bucket, Key=key)["Body"]
        return bytes(body.read())

    def _fetch_result(self, prefix: str, job: LoadJob) -> LoadJobResult:
        result = LoadJobResult.model_validate_json(self._read(prefix + "result.json"))
        if job.sample_gpu:
            csv = self._read(prefix + "gpu.csv").decode()
            result = result.model_copy(update={"nvidia_smi_csv": csv})
        return result

    @staticmethod
    def _check_job_output(host: Host, stdout: str) -> None:
        _, system = parse_markers(stdout)
        if system.get("job_isolation") != "ok" or "loom-job-done" not in stdout:
            raise RuntimeError(f"{host.host_id}: job finished without its isolation check")

    async def run_job(self, host: Host, job: LoadJob) -> LoadJobResult:
        prefix, script = await asyncio.to_thread(self._stage_job, host, job)
        stdout = await self._run(
            host, script, timeout_s=self._job_timeout_s(job), what=f"job {job.run_id}"
        )
        self._check_job_output(host, stdout)
        return await asyncio.to_thread(self._fetch_result, prefix, job)

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        """Run the eval in the pod, where the engine listens. Bounded by `job_timeout_s`."""
        prefix, script = await asyncio.to_thread(self._stage_eval, host, job)
        stdout = await self._run(
            host,
            script,
            timeout_s=self.settings.job_timeout_s + JOB_ALLOWANCE_S,
            what=f"eval {job.run_id}",
        )
        self._check_job_output(host, stdout)
        raw = await asyncio.to_thread(self._read, prefix + "result.json")
        return EvalJobResult.model_validate_json(raw)

    # -- teardown -------------------------------------------------------------------

    async def teardown(self, host: Host) -> None:
        """Terminate (never stop) the pod; a pod that is already gone is fine."""
        await asyncio.to_thread(self.api.terminate_pod, host.host_id)

    async def reap(self, now: datetime) -> list[str]:
        return await asyncio.to_thread(
            runpod_reaper.reap, self.api, now, prefix=self.settings.name_prefix
        )
