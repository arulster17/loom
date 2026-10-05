import base64
import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import pytest

from loom_bench.engines import render_launch
from loom_bench.jobs import LoadJob, LoadJobResult, TokenizerSpec
from loom_bench.prices import load_prices
from loom_bench.providers.aws_ec2 import (
    AwsEc2Provider,
    AwsSettings,
    HostLost,
    SpotInterrupted,
    host_loss,
    load_aws_settings,
    spot_hourly_micros,
)
from loom_bench.providers.base import Host, HostRequest
from loom_bench.records import LoadMode, Market
from loom_bench.registry import load_registry

from .conftest import REGION, amazon_ami, state
from .fakes import FakeSsm, Proxy, client_error, no_sleep

DLAMI_PARAM = "/loom/test/dlami"
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


def provider(aws: dict[str, Any], **kw: Any) -> AwsEc2Provider:
    clients = {"ec2": aws["ec2"], "ssm": aws["ssm"], "s3": aws["s3"], **kw}
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
    assert host.info["price_basis"] == {
        "source": "prices.yaml",
        "market": "on_demand",
        "instance_micros_per_hour": G6E_XLARGE_ON_DEMAND,
        "ebs_micros_per_hour": EBS_200GB,
    }
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
    basis = host.info["price_basis"]
    assert basis["source"] == "describe_spot_price_history"
    assert basis["spot_price_usd"] == "0.00001"
    assert basis["multiplier"] == "1.25"
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

    empty = Proxy(aws["ec2"], describe_spot_price_history=lambda **_: {"SpotPriceHistory": []})
    host = await provider(aws, ec2=empty).provision(request(market=Market.SPOT))
    assert host.hourly_micros == G6E_XLARGE_ON_DEMAND + EBS_200GB
    assert host.info["price_basis"]["market"] == "on_demand_fallback"


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
        "loom-sys driver 595.91.07",
        "loom-sys cuda 13.2",
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

    assert ep.base_url == "http://127.0.0.1:8000"
    assert ep.metrics_url == "http://127.0.0.1:8000/metrics"
    assert ep.engine == "vllm"
    assert ep.served_model == "qwen3-8b"
    assert not ep.warm
    assert ep.start_stages["image_pulled"] == 95.5
    assert ep.start_stages["first_token"] == 250.4
    assert {"instance_running", "ssm_online"} <= ep.start_stages.keys()
    assert list(ep.start_stages)[-1] == "first_token"
    assert ep.system["gpus"] == ["NVIDIA L40S"]
    assert ep.system["driver"] == "595.91.07"
    assert ep.system["cuda"] == "13.2"
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


async def test_warm_start_and_stop(aws: dict[str, Any]) -> None:
    ssm = FakeSsm(lambda script: [{"Status": "Success", "StandardOutputContent": ""}])
    p = provider(aws, ssm=ssm)
    host = await p.provision(request())
    await p.stop_engine(host)
    ep = await p.start_engine(host, render_launch(load_registry().get("qwen3-8b")), warm=True)
    assert ep.warm
    assert "loom-engine" in ssm.scripts[0]
    assert "WARM=1" in ssm.scripts[1]
    assert ssm.sent[1]["Parameters"]["executionTimeout"] == ["1800"]


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

    script = ssm.scripts[0]
    for var in ("JOB_URL", "WHEEL_URL", "RESULT_URL", "GPU_CSV_URL"):
        m = re.search(rf"^{var}='(https://[^']+)'$", script, re.MULTILINE)
        assert m, var
        assert "Signature" in m.group(1)
    assert "SAMPLE_GPU=1" in script
    assert "WHEEL_SHA256=" in script
    assert ssm.sent[0]["Parameters"]["executionTimeout"] == [str(60 + 600 + 60 + 900)]


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
