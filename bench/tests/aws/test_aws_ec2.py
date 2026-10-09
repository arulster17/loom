import base64
import hashlib
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import boto3
import pytest

from loom_bench.engines import render_launch
from loom_bench.experiment import expand, load_experiment
from loom_bench.jobs import (
    EvalJob,
    EvalJobResult,
    EvalTaskResult,
    LoadJob,
    LoadJobResult,
    TokenizerSpec,
)
from loom_bench.prices import load_prices
from loom_bench.provenance import GitInfo, PriceBasis, build_provenance
from loom_bench.providers import export_requirements
from loom_bench.providers.aws_ec2 import (
    CLIENT_IMAGE,
    DOWNLOAD_ALLOWANCE_S,
    AwsEc2Provider,
    AwsSettings,
    HostLost,
    SpotInterrupted,
    host_loss,
    load_aws_settings,
    spot_hourly_micros,
)
from loom_bench.providers.base import Host, HostRequest
from loom_bench.quality.divergence import ReferenceLogprobs
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.records import LoadMode, Market
from loom_bench.registry import REPO_ROOT, load_registry
from loom_bench.runner import _Executor

from .conftest import REGION, amazon_ami, state
from .fakes import FakeSsm, Proxy, client_error, no_sleep

DLAMI_PARAM = "/loom/test/dlami"
QWEN_EXPERIMENT = REPO_ROOT / "bench/experiments/qwen3-8b-vllm-vs-sglang.yaml"
G6E_XLARGE_ON_DEMAND = 1_861_000
# 200 GB gp3 at $0.08/GB-month over 730 h = 21_917.8 micros/h, rounded up.
EBS_200GB = 21_918


@pytest.fixture
def aws(moto_aws: None) -> dict[str, Any]:
    ec2 = boto3.client("ec2", region_name=REGION)
    iam = boto3.client("iam", region_name=REGION)
    ssm = boto3.client("ssm", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    iam.create_role(RoleName="loom-bench-instance", AssumeRolePolicyDocument="{}")
    iam.create_instance_profile(InstanceProfileName="loom-bench-instance")
    iam.add_role_to_instance_profile(
        InstanceProfileName="loom-bench-instance", RoleName="loom-bench-instance"
    )
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]
    sg = ec2.create_security_group(GroupName="loom", Description="x", VpcId=vpc["VpcId"])
    subnets = sorted(ec2.describe_subnets()["Subnets"], key=lambda s: s["AvailabilityZone"])
    ssm.put_parameter(Name=DLAMI_PARAM, Value=amazon_ami(ec2), Type="String")
    s3.create_bucket(Bucket="loom-bench-test")
    settings = AwsSettings(
        bucket="loom-bench-test",
        instance_profile_name="loom-bench-instance",
        security_group_id=sg["GroupId"],
        subnet_ids=[subnets[0]["SubnetId"], subnets[1]["SubnetId"]],
        hf_token_secret_name="loom/hf-token",
        owner="arul",
        dlami_ssm_parameter=DLAMI_PARAM,
        poll_interval_s=0.01,
    )
    return {"ec2": ec2, "ssm": ssm, "s3": s3, "settings": settings, "subnets": subnets}


def locked(extra: str) -> str:
    """Stands in for `export_requirements` (a `uv export` of uv.lock)."""
    return f"pkg-{extra or 'base'}==1.0 --hash=sha256:{'0' * 64}\n"


def provider(aws: dict[str, Any], **kw: Any) -> AwsEc2Provider:
    clients = {"ec2": aws["ec2"], "ssm": aws["ssm"], "s3": aws["s3"], "requirements": locked, **kw}
    if isinstance(clients["ssm"], FakeSsm):
        clients["ssm"].params = aws["ssm"]
    return AwsEc2Provider(aws["settings"], prices=load_prices(), sleep=no_sleep, **clients)


def request(**kw: Any) -> HostRequest:
    base: dict[str, Any] = {
        "cloud": "aws",
        "region": REGION,
        "instance_type": "g6e.xlarge",
        "market": Market.ON_DEMAND,
        "ttl_s": 3600,
        "tags": {"loom:experiment": "exp-1", "team": "bench"},
    }
    return HostRequest(**{**base, **kw})


def describe(ec2: Any, iid: str) -> dict[str, Any]:
    return dict(ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0])


def tags_of(resource: dict[str, Any]) -> dict[str, str]:
    return {t["Key"]: t["Value"] for t in resource.get("Tags", [])}


async def test_provision_on_demand_sets_safety_rails(aws: dict[str, Any]) -> None:
    ec2 = aws["ec2"]
    before = datetime.now(UTC)
    host = await provider(aws).provision(request())
    inst = describe(ec2, host.host_id)

    assert host.hourly_micros == G6E_XLARGE_ON_DEMAND + EBS_200GB
    assert host.info["accrual_basis"] == {
        "source": "prices.yaml",
        "market": "on_demand",
        "instance_micros_per_hour": G6E_XLARGE_ON_DEMAND,
        "ebs_micros_per_hour": EBS_200GB,
    }
    # as run: the same list price and volume, rounded once (21_917.8 micros of EBS)
    assert host.as_run_micros == 1_882_918
    assert host.price_basis == PriceBasis(
        market=Market.ON_DEMAND,
        source="prices_yaml",
        availability_zone=host.info["az"],
        storage_gb=200,
    )
    assert host.info["subnet_id"] == aws["settings"].subnet_ids[0]
    assert "InstanceLifecycle" not in inst
    assert timedelta(seconds=3599) <= host.ttl_at - before <= timedelta(seconds=3601)

    ttl = host.ttl_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    expected = {
        "loom:managed": "true",
        "loom:experiment": "exp-1",
        "loom:owner": "arul",
        "loom:ttl": ttl,
        "team": "bench",
        "Name": "loom-bench-exp-1",
    }
    assert tags_of(inst) == expected
    vols = ec2.describe_volumes(
        Filters=[{"Name": "attachment.instance-id", "Values": [host.host_id]}]
    )["Volumes"]
    assert len(vols) == 1
    assert tags_of(vols[0]) == expected
    assert vols[0]["VolumeType"] == "gp3"
    assert vols[0]["Size"] == 200
    assert vols[0]["Encrypted"] is True
    assert inst["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True

    meta = inst["MetadataOptions"]
    assert meta["HttpTokens"] == "required"
    assert meta["HttpPutResponseHopLimit"] == 1
    assert [g["GroupId"] for g in inst["SecurityGroups"]] == [aws["settings"].security_group_id]
    assert inst["IamInstanceProfile"]["Arn"].endswith("/loom-bench-instance")

    shutdown = ec2.describe_instance_attribute(
        InstanceId=host.host_id, Attribute="instanceInitiatedShutdownBehavior"
    )
    assert shutdown["InstanceInitiatedShutdownBehavior"]["Value"] == "terminate"
    raw = ec2.describe_instance_attribute(InstanceId=host.host_id, Attribute="userData")
    user_data = base64.b64decode(raw["UserData"]["Value"]).decode()
    assert user_data.startswith("#!/bin/bash\n")
    assert f"TTL_EPOCH={int(host.ttl_at.timestamp())}" in user_data
    assert 'shutdown -h "+$remaining_min"' in user_data
    assert "loom/hf-token" not in user_data


async def test_provision_spot_accrues_spot_price_times_multiplier(aws: dict[str, Any]) -> None:
    host = await provider(aws).provision(request(market=Market.SPOT, disk_gb=500))
    inst = describe(aws["ec2"], host.host_id)
    assert inst["InstanceLifecycle"] == "spot"
    # moto quotes every spot price as $0.00001/h: 10 micros x 1.25 = 12.5, rounded up;
    # 500 GB gp3 = 54_794.5 micros/h, rounded up.
    assert host.hourly_micros == 13 + 54_795
    basis = host.info["accrual_basis"]
    assert basis["source"] == "describe_spot_price_history"
    assert basis["spot_price_usd"] == "0.00001"
    assert basis["multiplier"] == "1.25"
    # as run: the observed price without the multiplier, plus the volume, rounded once
    assert host.as_run_micros == 10 + 54_795  # 10 + 54_794.5, half up
    assert host.price_basis is not None
    assert host.price_basis.source == "observed_spot"
    assert host.price_basis.spot_price_usd == "0.00001"
    assert host.price_basis.storage_gb == 500
    assert host.price_basis.availability_zone == host.info["az"]
    vols = aws["ec2"].describe_volumes(
        Filters=[{"Name": "attachment.instance-id", "Values": [host.host_id]}]
    )["Volumes"]
    assert vols[0]["Size"] == 500


def test_spot_hourly_micros_rounds_up() -> None:
    assert spot_hourly_micros("1.8386", Decimal("1.25")) == 2_298_250
    assert spot_hourly_micros("0.4831234", Decimal("1.25")) == 603_905
    assert spot_hourly_micros("0.5", Decimal("1")) == 500_000


async def test_spot_picks_cheapest_az_and_falls_back_to_on_demand_price(
    aws: dict[str, Any],
) -> None:
    a, b = (s["AvailabilityZone"] for s in aws["subnets"][:2])
    now = datetime.now(UTC)

    def history(**_: Any) -> dict[str, Any]:
        return {
            "SpotPriceHistory": [
                {"AvailabilityZone": a, "SpotPrice": "2.0", "Timestamp": now},
                {"AvailabilityZone": b, "SpotPrice": "9.0", "Timestamp": now - timedelta(hours=1)},
                {"AvailabilityZone": b, "SpotPrice": "1.0", "Timestamp": now},
            ]
        }

    ec2 = Proxy(aws["ec2"], describe_spot_price_history=history)
    host = await provider(aws, ec2=ec2).provision(request(market=Market.SPOT))
    assert host.info["az"] == b
    assert host.hourly_micros == 1_250_000 + EBS_200GB
    assert host.as_run_micros == 1_021_918  # $1.00 observed + 21_917.8 EBS
    assert host.price_basis is not None and host.price_basis.observed_at == now

    empty = Proxy(aws["ec2"], describe_spot_price_history=lambda **_: {"SpotPriceHistory": []})
    host = await provider(aws, ec2=empty).provision(request(market=Market.SPOT))
    assert host.hourly_micros == G6E_XLARGE_ON_DEMAND + EBS_200GB
    assert host.info["accrual_basis"]["market"] == "on_demand_fallback"
    # nothing observed: the as-run price is unknown, never guessed
    assert host.as_run_micros is None
    assert host.price_basis is not None and host.price_basis.source == "unobserved"


async def test_provision_tries_next_az_on_capacity_error(aws: dict[str, Any]) -> None:
    first = aws["settings"].subnet_ids[0]
    calls: list[str] = []

    def run_instances(**kw: Any) -> Any:
        calls.append(kw["SubnetId"])
        if kw["SubnetId"] == first:
            raise client_error("InsufficientInstanceCapacity")
        return aws["ec2"].run_instances(**kw)

    host = await provider(aws, ec2=Proxy(aws["ec2"], run_instances=run_instances)).provision(
        request()
    )
    assert calls == aws["settings"].subnet_ids
    assert host.info["subnet_id"] == aws["settings"].subnet_ids[1]


async def test_provision_quota_error_is_not_retried(aws: dict[str, Any]) -> None:
    def run_instances(**kw: Any) -> Any:
        raise client_error("VcpuLimitExceeded")

    p = provider(aws, ec2=Proxy(aws["ec2"], run_instances=run_instances))
    with pytest.raises(Exception, match="VcpuLimitExceeded"):
        await p.provision(request())


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"cloud": "gcp"}, "cloud 'aws'"),
        ({"market": Market.LOCAL}, "spot and on_demand"),
        ({"instance_type": None}, "instance_type"),
        ({"ttl_s": 9 * 3600}, "ttl_s"),
        ({"region": "eu-west-1"}, "region"),
        ({"tags": {}}, "loom:experiment"),
        ({"tags": {"loom:experiment": "has space"}}, "loom:experiment"),
        ({"tags": {"loom:experiment": "e", "loom:ttl": "x"}}, "set by the provider"),
    ],
)
async def test_provision_rejects_bad_requests(
    aws: dict[str, Any], changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        await provider(aws).provision(request(**changes))


async def test_provision_rejects_unpriced_instance_type(aws: dict[str, Any]) -> None:
    with pytest.raises(KeyError, match="no prices"):
        await provider(aws).provision(request(instance_type="g4dn.xlarge"))


async def test_teardown_is_idempotent(aws: dict[str, Any]) -> None:
    p = provider(aws)
    host = await p.provision(request())
    await p.teardown(host)
    await p.teardown(host)
    assert state(aws["ec2"], host.host_id) == "terminated"
    ghost = host.model_copy(update={"host_id": "i-0123456789abcdef0"})
    await p.teardown(ghost)


def test_host_loss_classification() -> None:
    launched = datetime(2026, 10, 4, 12, tzinfo=UTC)
    now = launched + timedelta(minutes=30)
    running = {"State": {"Name": "running"}}
    assert host_loss(running, "i-1", launched, now) is None

    spot = {
        "State": {"Name": "shutting-down"},
        "StateReason": {
            "Code": "Server.SpotInstanceTermination",
            "Message": "Server.SpotInstanceTermination: Spot instance termination",
        },
    }
    lost = host_loss(spot, "i-1", launched, now)
    assert isinstance(lost, SpotInterrupted)
    assert lost.seconds_since_launch == 1800
    assert lost.detected_at == now

    ttl = {
        "State": {"Name": "terminated"},
        "StateReason": {"Code": "Client.InstanceInitiatedShutdown"},
    }
    lost = host_loss(ttl, "i-1", launched, now)
    assert type(lost) is HostLost
    assert lost.reason_code == "Client.InstanceInitiatedShutdown"

    missing = host_loss(None, "i-1", launched, now)
    assert type(missing) is HostLost
    assert missing.state == "not-found"


async def test_check_host_after_termination_raises(aws: dict[str, Any]) -> None:
    p = provider(aws)
    host = await p.provision(request())
    await p.check_host(host)
    await p.teardown(host)
    with pytest.raises(HostLost) as e:
        await p.check_host(host)
    assert not isinstance(e.value, SpotInterrupted)


def engine_stdout(t0: float) -> str:
    lines = [
        f"loom-stage user_data_started {t0 + 20}",
        f"loom-stage user_data_done {t0 + 21}",
        f"loom-stage image_pulled {t0 + 95.5}",
        f"loom-stage weights_ready {t0 + 160}",
        f"loom-stage engine_started {t0 + 161}",
        f"loom-stage engine_healthy {t0 + 250}",
        f"loom-stage first_token {t0 + 250.4}",
        "loom-sys gpus NVIDIA L40S",
        "loom-sys driver_version 595.91.07",
        "loom-sys cuda_version 13.2",
        "loom-sys image_digest vllm/vllm-openai@sha256:" + "a" * 64,
    ]
    return "\n".join(lines) + "\n"


async def test_start_engine_cold_reports_stages_and_system(aws: dict[str, Any]) -> None:
    holder: dict[str, Host] = {}
    ssm = FakeSsm(
        lambda script: [
            {"Status": "InProgress"},
            {
                "Status": "Success",
                "StandardOutputContent": engine_stdout(holder["host"].launched_at.timestamp()),
            },
        ]
    )
    p = provider(aws, ssm=ssm)
    host = holder["host"] = await p.provision(request())
    launch = render_launch(load_registry().get("qwen3-8b"))
    ep = await p.start_engine(host, launch, warm=False)

    assert ep.base_url == "http://127.0.0.1:8000/v1"
    assert ep.metrics_url == "http://127.0.0.1:8000/metrics"
    assert ep.engine == "vllm"
    assert ep.served_model == "qwen3-8b"
    assert not ep.warm
    assert ep.start_stages["image_pulled"] == 95.5
    assert ep.start_stages["first_token"] == 250.4
    assert {"instance_running", "ssm_online"} <= ep.start_stages.keys()
    assert list(ep.start_stages)[-1] == "first_token"
    assert ep.system["gpus"] == ["NVIDIA L40S"]
    assert ep.system["driver_version"] == "595.91.07"
    assert ep.system["cuda_version"] == "13.2"
    assert ep.system["gpu_count"] == 1
    assert ep.system["image_digest"].endswith("a" * 64)
    assert ep.system["instance_type"] == "g6e.xlarge"

    script = ssm.scripts[0]
    assert "WARM=0" in script
    assert "HF_SECRET_ID=loom/hf-token" in script
    assert "--publish 127.0.0.1:8000:8000" in script
    sent = ssm.sent[0]
    assert sent["InstanceIds"] == [host.host_id]
    assert sent["Parameters"]["executionTimeout"] == [str(1800 + 3600)]
    assert sent["OutputS3KeyPrefix"] == f"ssm/exp-1/{host.host_id}"


async def test_cold_start_system_facts_reach_provenance(aws: dict[str, Any]) -> None:
    """The keys start_engine.sh emits are the ones the runner reads into provenance."""
    holder: dict[str, Host] = {}
    ssm = FakeSsm(
        lambda script: [
            {
                "Status": "Success",
                "StandardOutputContent": engine_stdout(holder["host"].launched_at.timestamp()),
            }
        ]
    )
    p = provider(aws, ssm=ssm)
    host = holder["host"] = await p.provision(request())
    cell = next(
        c for c in expand(load_experiment(QWEN_EXPERIMENT), load_registry()) if c.launch.image
    )
    ep = await p.start_engine(host, cell.launch, warm=False)
    sections = _Executor._serving_sections(
        SimpleNamespace(git=GitInfo()),  # type: ignore[arg-type]
        host,
        cell,
        ep,
    )
    prov = build_provenance(cell.config, **sections)
    assert prov.cuda_version == "13.2"
    assert prov.driver_version == "595.91.07"
    assert prov.hardware.gpu_count == 1
    assert prov.hardware.gpu_type == "L40S"


def _fp8_qwen_launch():
    """Qwen3-8B served from another checkpoint (the FP8 smokes' hf override)."""
    spec = load_registry().get("qwen3-8b")
    hf = spec.hf.model_copy(update={"repo": "RedHatAI/Qwen3-8B-FP8-dynamic", "revision": "0" * 40})
    return render_launch(spec.model_copy(update={"hf": hf}))


def _engine_ssm(holder: dict[str, Host]) -> FakeSsm:
    return FakeSsm(
        lambda script: [
            {
                "Status": "Success",
                "StandardOutputContent": engine_stdout(holder["host"].launched_at.timestamp()),
            }
        ]
    )


async def test_warm_start_and_stop(aws: dict[str, Any]) -> None:
    holder: dict[str, Host] = {}
    ssm = _engine_ssm(holder)
    p = provider(aws, ssm=ssm)
    host = holder["host"] = await p.provision(request())
    launch = render_launch(load_registry().get("qwen3-8b"))
    await p.start_engine(host, launch, warm=False)
    assert "FETCH_WEIGHTS=(Qwen/Qwen3-8B " in ssm.scripts[0]
    assert "CACHED_WEIGHTS=()" in ssm.scripts[0]
    await p.stop_engine(host)
    ep = await p.start_engine(host, launch, warm=True)
    assert ep.warm
    assert "loom-engine" in ssm.scripts[1]
    warm = ssm.scripts[2]
    assert "WARM=1" in warm
    # Its checkpoint is on the host already: checked offline, nothing downloaded.
    assert "FETCH_WEIGHTS=()" in warm and "CACHED_WEIGHTS=(Qwen/Qwen3-8B " in warm
    assert ssm.sent[2]["Parameters"]["executionTimeout"] == ["1800"]


async def test_a_warm_start_onto_another_checkpoint_downloads_it(aws: dict[str, Any]) -> None:
    # runpod-smoke-h100 (4e50b5a5): the FP8 cell's warm restart downloaded nothing and
    # its offline engine found the FP8 checkpoint "not cached".
    holder: dict[str, Host] = {}
    ssm = _engine_ssm(holder)
    p = provider(aws, ssm=ssm)
    host = holder["host"] = await p.provision(request())
    await p.start_engine(host, render_launch(load_registry().get("qwen3-8b")), warm=False)
    await p.stop_engine(host)
    await p.start_engine(host, _fp8_qwen_launch(), warm=True)
    warm = ssm.scripts[2]
    assert "FETCH_WEIGHTS=(RedHatAI/Qwen3-8B-FP8-dynamic " + "0" * 40 + ")" in warm
    assert "CACHED_WEIGHTS=()" in warm
    # The download gets the same allowance as a cold start's.
    assert ssm.sent[2]["Parameters"]["executionTimeout"] == [str(1800 + DOWNLOAD_ALLOWANCE_S)]
    await p.stop_engine(host)
    # Back onto BF16: both checkpoints are on the host now.
    await p.start_engine(host, render_launch(load_registry().get("qwen3-8b")), warm=True)
    assert "FETCH_WEIGHTS=()" in ssm.scripts[4]


async def test_spot_interruption_during_command_raises(aws: dict[str, Any]) -> None:
    ssm = FakeSsm(lambda script: [{"Status": "InProgress"}])
    interrupted = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-x",
                        "State": {"Name": "terminated"},
                        "StateReason": {"Code": "Server.SpotInstanceTermination"},
                    }
                ]
            }
        ]
    }
    p = provider(aws, ssm=ssm)
    host = await p.provision(request(market=Market.SPOT))
    p.ec2 = Proxy(aws["ec2"], describe_instances=lambda **_: interrupted)
    with pytest.raises(SpotInterrupted) as e:
        await p.stop_engine(host)
    assert e.value.host_id == host.host_id
    assert e.value.seconds_since_launch >= 0


def load_job(**kw: Any) -> LoadJob:
    base: dict[str, Any] = {
        "run_id": "run-7",
        "base_url": "http://127.0.0.1:8000",
        "metrics_url": "http://127.0.0.1:8000/metrics",
        "engine": "vllm",
        "served_model": "qwen3-8b",
        "workload": {"kind": "fixed", "input_len": 128, "output_len": 128},
        "tokenizer": TokenizerSpec(kind="simple"),
        "mode": LoadMode.CLOSED_LOOP,
        "load_value": 8,
        "duration_s": 60,
        "sample_gpu": True,
    }
    return LoadJob(**{**base, **kw})


async def test_run_job_round_trip_through_s3(aws: dict[str, Any], tmp_path: Path) -> None:
    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    s3 = aws["s3"]
    job = load_job()
    result = LoadJobResult(
        run_id="run-7",
        mode=LoadMode.CLOSED_LOOP,
        load_value=8,
        t_measure_start_s=0,
        t_measure_end_s=60,
        records=[],
        started_at="2026-10-04T00:00:00Z",
        finished_at="2026-10-04T00:01:00Z",
    )
    prefix = "runs/exp-1/run-7/"

    def host_side(script: str) -> list[dict[str, Any]]:
        s3.put_object(
            Bucket="loom-bench-test", Key=prefix + "result.json", Body=result.model_dump_json()
        )
        s3.put_object(
            Bucket="loom-bench-test", Key=prefix + "gpu.csv", Body=b"2026/10/04, 0, 97, 1, 2, 300\n"
        )
        return [{"Status": "Success", "StandardOutputContent": "loom-job-done\n"}]

    ssm = FakeSsm(host_side)
    p = provider(aws, ssm=ssm, wheel_path=wheel)
    host = await p.provision(request())
    got = await p.run_job(host, job)

    assert got.run_id == "run-7"
    assert got.nvidia_smi_csv == "2026/10/04, 0, 97, 1, 2, 300\n"
    stored = s3.get_object(Bucket="loom-bench-test", Key=prefix + "job.json")["Body"].read()
    assert LoadJob.model_validate_json(stored) == job
    assert (
        s3.get_object(Bucket="loom-bench-test", Key=prefix + wheel.name)["Body"].read()
        == b"wheel-bytes"
    )

    reqs = s3.get_object(Bucket="loom-bench-test", Key=prefix + "requirements.txt")["Body"]
    assert reqs.read().decode() == locked("")

    script = ssm.scripts[0]
    for var in ("JOB_URL", "WHEEL_URL", "REQS_URL", "RESULT_URL", "GPU_CSV_URL"):
        m = re.search(rf"^{var}='(https://[^']+)'$", script, re.MULTILINE)
        assert m, var
        assert "Signature" in m.group(1)
    assert "SAMPLE_GPU=1" in script
    assert "WHEEL_SHA256=" in script
    assert f"REQS_SHA256={hashlib.sha256(locked('').encode()).hexdigest()}" in script
    assert f"CLIENT_IMAGE={aws['settings'].client_image}" in script
    assert ssm.sent[0]["Parameters"]["executionTimeout"] == [str(60 + 600 + 60 + 900)]


LLAMA_REPO = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_REV = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
LLAMA_CACHE = "/opt/dlami/nvme/loom-hf/hub/models--meta-llama--Llama-3.3-70B-Instruct"
LLAMA_MOUNT = "/models/models--meta-llama--Llama-3.3-70B-Instruct"


def host_writes(s3: Any, prefix: str, result_json: str) -> Any:
    """FakeSsm `on_command` for a job that succeeds and uploads `result_json`."""

    def host_side(script: str) -> list[dict[str, Any]]:
        s3.put_object(Bucket="loom-bench-test", Key=prefix + "result.json", Body=result_json)
        return [{"Status": "Success", "StandardOutputContent": "loom-job-done\n"}]

    return host_side


def assert_mounts_the_snapshot(script: str, stored_tokenizer: TokenizerSpec) -> None:
    assert f"MODEL_CACHE_DIR={LLAMA_CACHE}" in script
    assert f"MODEL_CACHE_MOUNT={LLAMA_MOUNT}" in script
    assert f"MODEL_REVISION={LLAMA_REV}" in script
    assert '--volume "$WORK_DIR:/work" ${MODEL_MOUNT:+--volume "$MODEL_MOUNT"}' in script
    assert 'MODEL_MOUNT="$MODEL_CACHE_DIR:$MODEL_CACHE_MOUNT:ro"' in script
    assert stored_tokenizer == TokenizerSpec(
        kind="hf",
        repo=LLAMA_REPO,
        revision=LLAMA_REV,
        local_dir=f"{LLAMA_MOUNT}/snapshots/{LLAMA_REV}",
    )
    assert "HF_TOKEN" not in script and "loom/hf-token" not in script


async def test_run_job_loads_a_gated_tokenizer_from_the_host_snapshot(
    aws: dict[str, Any], tmp_path: Path
) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    result = LoadJobResult(
        run_id="run-7",
        mode=LoadMode.CLOSED_LOOP,
        load_value=8,
        t_measure_start_s=0,
        t_measure_end_s=60,
        records=[],
        started_at="2026-10-04T00:00:00Z",
        finished_at="2026-10-04T00:01:00Z",
    )
    ssm = FakeSsm(host_writes(aws["s3"], "runs/exp-1/run-7/", result.model_dump_json()))
    p = provider(aws, ssm=ssm, wheel_path=wheel)
    host = await p.provision(request())
    tok = TokenizerSpec(kind="hf", repo=LLAMA_REPO, revision=LLAMA_REV)
    await p.run_job(host, load_job(tokenizer=tok, sample_gpu=False))
    stored = aws["s3"].get_object(Bucket="loom-bench-test", Key="runs/exp-1/run-7/job.json")
    job = LoadJob.model_validate_json(stored["Body"].read())
    assert_mounts_the_snapshot(ssm.scripts[0], job.tokenizer)

    _, simple = p._stage_job(host, load_job())
    assert "MODEL_CACHE_DIR=''" in simple


@pytest.mark.parametrize(
    ("repo", "revision", "match"),
    [(LLAMA_REPO, "main", "pinned commit"), ("../x", LLAMA_REV, "org/name")],
)
def test_host_tokenizer_needs_a_pinned_commit(repo: str, revision: str, match: str) -> None:
    settings = AwsSettings(
        bucket="loom-bench-test",
        instance_profile_name="loom-bench-instance",
        security_group_id="sg-0123",
        subnet_ids=["subnet-0123"],
        hf_token_secret_name="loom/hf-token",
        owner="arul",
    )
    p = AwsEc2Provider(settings, prices=load_prices(), ec2=object(), ssm=object(), s3=object())
    with pytest.raises(ValueError, match=match):
        p._host_tokenizer(TokenizerSpec(kind="hf", repo=repo, revision=revision))


async def test_run_job_requires_wheel_and_safe_run_id(aws: dict[str, Any], tmp_path: Path) -> None:
    p = provider(aws, ssm=FakeSsm())
    host = await p.provision(request())
    with pytest.raises(ValueError, match="wheel"):
        await p.run_job(host, load_job())
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    p.wheel_path = wheel
    with pytest.raises(ValueError, match="run_id"):
        await p.run_job(host, load_job(run_id="../escape"))


def eval_job(**kw: Any) -> EvalJob:
    suite = Suite.model_validate(
        {
            "suite": "s",
            "model": "qwen3-8b",
            "tasks": [
                {"name": "json_schema", "kind": "json_schema"},
                {
                    "name": "gsm8k",
                    "kind": "lm_eval",
                    "params": {"tasks": ["gsm8k"], "metric": "exact_match"},
                },
            ],
            "divergence": {},
        }
    )
    base: dict[str, Any] = {
        "run_id": "eval-3",
        "suite": suite,
        "base_url": "http://127.0.0.1:8000/v1",
        "served_model": "qwen3-8b",
        "divergence": "capture",
    }
    return EvalJob(**{**base, **kw})


async def test_run_eval_round_trip_through_s3(aws: dict[str, Any], tmp_path: Path) -> None:
    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    s3 = aws["s3"]
    job = eval_job()
    result = EvalJobResult(
        run_id="eval-3",
        suite="s",
        model="qwen3-8b",
        tasks={
            "json_schema": EvalTaskResult(
                kind="json_schema",
                version="1",
                items=[ItemResult(item_id="a", score=1.0, content_hash="h")],
                seconds=2.0,
            )
        },
        sanity=SanityResult(n=1, counts={"empty": 0}),
        reference=ReferenceLogprobs(model="qwen3-8b", top_k=5, max_new_tokens=64, prompts=[]),
        started_at="2026-10-04T00:00:00+00:00",
        finished_at="2026-10-04T00:01:00+00:00",
    )
    prefix = "runs/exp-1/eval-3/"

    def host_side(script: str) -> list[dict[str, Any]]:
        s3.put_object(
            Bucket="loom-bench-test", Key=prefix + "result.json", Body=result.model_dump_json()
        )
        return [{"Status": "Success", "StandardOutputContent": "loom-job-done\n"}]

    ssm = FakeSsm(host_side)
    p = provider(aws, ssm=ssm, wheel_path=wheel)
    host = await p.provision(request())
    got = await p.run_eval(host, job)

    assert got == result
    stored = s3.get_object(Bucket="loom-bench-test", Key=prefix + "job.json")["Body"].read()
    assert EvalJob.model_validate_json(stored) == job
    script = ssm.scripts[0]
    assert "BENCH_CMD=(quality job)" in script
    reqs = s3.get_object(Bucket="loom-bench-test", Key=prefix + "requirements.txt")["Body"]
    assert reqs.read().decode() == locked("lmeval")  # harness tasks: the lmeval extra
    assert "SAMPLE_GPU=0" in script and "GPU_CSV_URL=''" in script
    for var in ("JOB_URL", "WHEEL_URL", "RESULT_URL"):
        m = re.search(rf"^{var}='(https://[^']+)'$", script, re.MULTILINE)
        assert m and "Signature" in m.group(1), var
    assert ssm.sent[0]["Parameters"]["executionTimeout"] == [str(7200 + 900)]
    assert ssm.sent[0]["Comment"] == "loom eval eval-3"


async def test_run_eval_without_harness_tasks_skips_the_lmeval_extra(
    aws: dict[str, Any], tmp_path: Path
) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    p = provider(aws, ssm=FakeSsm(), wheel_path=wheel)
    host = await p.provision(request())
    prefix, script = p._stage_eval(host, eval_job(tasks=["json_schema"]))
    reqs = aws["s3"].get_object(Bucket="loom-bench-test", Key=prefix + "requirements.txt")
    assert reqs["Body"].read().decode() == locked("")
    assert "MODEL_CACHE_DIR=''" in script
    with pytest.raises(ValueError, match="run_id"):
        p._stage_eval(host, eval_job(run_id="../x"))


async def test_run_eval_loads_a_gated_tokenizer_from_the_host_snapshot(
    aws: dict[str, Any], tmp_path: Path
) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    result = EvalJobResult(
        run_id="eval-3",
        suite="s",
        model="qwen3-8b",
        tasks={},
        sanity=SanityResult(n=0, counts={}),
        started_at="2026-10-04T00:00:00+00:00",
        finished_at="2026-10-04T00:01:00+00:00",
    )
    ssm = FakeSsm(host_writes(aws["s3"], "runs/exp-1/eval-3/", result.model_dump_json()))
    p = provider(aws, ssm=ssm, wheel_path=wheel)
    host = await p.provision(request())
    tok = TokenizerSpec(kind="hf", repo=LLAMA_REPO, revision=LLAMA_REV)
    assert await p.run_eval(host, eval_job(tokenizer=tok)) == result
    stored = aws["s3"].get_object(Bucket="loom-bench-test", Key="runs/exp-1/eval-3/job.json")
    job = EvalJob.model_validate_json(stored["Body"].read())
    assert job.tokenizer is not None
    assert_mounts_the_snapshot(ssm.scripts[0], job.tokenizer)


def test_load_settings_yaml_then_env(tmp_path: Path) -> None:
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(
        "bucket: b-1\ninstance_profile_name: p\nsecurity_group_id: sg-0abc\n"
        "subnet_ids: [subnet-0a]\nhf_token_secret_name: loom/hf\nowner: arul\n"
    )
    env = {
        "LOOM_AWS_CONFIG": str(cfg),
        "LOOM_AWS_SUBNET_IDS": "subnet-0b, subnet-0c",
        "LOOM_AWS_SPOT_PRICE_MULTIPLIER": "1.5",
    }
    s = load_aws_settings(env=env)
    assert s.bucket == "b-1"
    assert s.subnet_ids == ["subnet-0b", "subnet-0c"]
    assert s.spot_price_multiplier == Decimal("1.5")
    assert s.region == "us-east-1"
    with pytest.raises(ValueError):
        load_aws_settings(env={**env, "LOOM_AWS_SPOT_PRICE_MULTIPLIER": "0.9"})


@pytest.mark.parametrize(
    "image",
    ["python:3.12-slim", "python:3.12-slim@sha256:abc", "python@sha256:" + "A" * 64],
)
def test_settings_reject_an_unpinned_client_image(tmp_path: Path, image: str) -> None:
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(
        "bucket: b-1\ninstance_profile_name: p\nsecurity_group_id: sg-0abc\n"
        "subnet_ids: [subnet-0a]\nhf_token_secret_name: loom/hf\nowner: arul\n"
    )
    env = {"LOOM_AWS_CONFIG": str(cfg)}
    assert load_aws_settings(env=env).client_image == CLIENT_IMAGE
    with pytest.raises(ValueError, match="client_image"):
        load_aws_settings(env={**env, "LOOM_AWS_CLIENT_IMAGE": image})


async def test_requirements_are_exported_once_per_extra(
    aws: dict[str, Any], tmp_path: Path
) -> None:
    wheel = tmp_path / "w.whl"
    wheel.write_bytes(b"x")
    exported: list[str] = []

    def export(extra: str) -> str:
        exported.append(extra)
        return locked(extra)

    p = provider(aws, ssm=FakeSsm(), wheel_path=wheel, requirements=export)
    host = await p.provision(request())
    for run_id in ("a", "b"):
        p._stage_job(host, load_job(run_id=run_id))
    p._stage_eval(host, eval_job())
    p._stage_eval(host, eval_job(run_id="eval-4"))
    assert exported == ["", "lmeval"]


def requirement_blocks(text: str) -> dict[str, str]:
    """`name==version` -> its block (the pin line and its hash lines)."""
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text)
    return {b.split(" ", 1)[0]: b for b in blocks if "==" in b.split("\n", 1)[0]}


def test_exported_requirements_are_the_locked_hash_pinned_set() -> None:
    base, lmeval = export_requirements(""), export_requirements("lmeval")
    for text in (base, lmeval):
        blocks = requirement_blocks(text)
        assert blocks and all("--hash=sha256:" in b for b in blocks.values())
        assert not any(pin.startswith(("loom-bench==", "loom_bench==")) for pin in blocks)
    base_names = {pin.split("==")[0] for pin in requirement_blocks(base)}
    lmeval_names = {pin.split("==")[0] for pin in requirement_blocks(lmeval)}
    assert {"pydantic", "httpx", "numpy"} <= base_names < lmeval_names
    assert "lm-eval" in lmeval_names and "lm-eval" not in base_names
