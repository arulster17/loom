"""Hosts in more than one region (`AwsSettings.extra_regions`).

us-east-1 had no g6e capacity for hours on 2026-11-03 while the G/VT on-demand quota was
4 vCPU (one g6e.xlarge) in each of us-east-1, us-east-2 and us-west-2. One capacity sweep
now tries every AZ of every configured region in turn; the vCPU quota is per region, so
a quota refusal skips that region for the sweep instead of ending it. Everything after
the launch (SSM, describe, teardown, the ClientToken cleanup, the reaper) goes to the
region the host runs in. Moto runs all three regions in one account here."""

import logging
from pathlib import Path
from typing import Any

import boto3
import pytest

from loom_bench.engines import render_launch
from loom_bench.experiment import expand, load_experiment
from loom_bench.plan import host_price
from loom_bench.prices import load_prices
from loom_bench.providers import make_provider
from loom_bench.providers.aws_ec2 import (
    QUOTA_WAIT_S,
    AwsEc2Provider,
    AwsRegion,
    AwsSettings,
    Ec2Unreachable,
    NoCapacity,
    VcpuQuotaHeld,
    load_aws_settings,
)
from loom_bench.providers.base import Host
from loom_bench.registry import load_registry

from .conftest import amazon_ami, state
from .fakes import FakeSsm, Proxy, client_error
from .test_aws_ec2 import (
    AWS_LOG,
    IIC,
    QWEN_EXPERIMENT,
    VCPU,
    FakeTime,
    engine_stdout,
    locked,
    request,
)
from .test_aws_ec2_network import TokenEc2

PRIMARY, EAST2, WEST2 = "us-east-1", "us-east-2", "us-west-2"
DLAMI_NAME = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261006"


def _region(ec2: Any, region: str, ami: str) -> dict[str, Any]:
    vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]
    sg = ec2.create_security_group(GroupName="loom", Description="x", VpcId=vpc["VpcId"])
    subnets = sorted(ec2.describe_subnets()["Subnets"], key=lambda s: s["AvailabilityZone"])
    return {
        "ec2": ec2,
        "region": region,
        "ami": ami,
        "sg": sg["GroupId"],
        "subnets": [s["SubnetId"] for s in subnets[:2]],
        "azs": [s["AvailabilityZone"] for s in subnets[:2]],
    }


def regions_env(*, pin_extra: bool = False) -> dict[str, Any]:
    """Three regions, each with its own SG, two subnets and its copy of one AMI release
    (same name, different id), as the DLAMI is published. Call inside `mock_aws`."""
    iam = boto3.client("iam", region_name=PRIMARY)
    iam.create_role(RoleName="loom-bench-instance", AssumeRolePolicyDocument="{}")
    iam.create_instance_profile(InstanceProfileName="loom-bench-instance")
    iam.add_role_to_instance_profile(
        InstanceProfileName="loom-bench-instance", RoleName="loom-bench-instance"
    )
    boto3.client("s3", region_name=PRIMARY).create_bucket(Bucket="loom-bench-test")
    source = amazon_ami(boto3.client("ec2", region_name=PRIMARY))
    out: dict[str, Any] = {}
    for region in (PRIMARY, EAST2, WEST2):
        ec2 = boto3.client("ec2", region_name=region)
        ami = ec2.copy_image(SourceImageId=source, SourceRegion=PRIMARY, Name=DLAMI_NAME)
        out[region] = _region(ec2, region, ami["ImageId"])
    p = out[PRIMARY]
    settings = AwsSettings(
        bucket="loom-bench-test",
        instance_profile_name="loom-bench-instance",
        security_group_id=p["sg"],
        subnet_ids=p["subnets"],
        hf_token_secret_name="loom/hf-token",
        owner="arul",
        ami_id=p["ami"],
        poll_interval_s=0.01,
        extra_regions=[
            AwsRegion(
                region=r,
                security_group_id=out[r]["sg"],
                subnet_ids=out[r]["subnets"],
                ami_id=out[r]["ami"] if pin_extra else None,
            )
            for r in (EAST2, WEST2)
        ],
    )
    return {"regions": out, "settings": settings}


@pytest.fixture
def env(moto_aws: None) -> dict[str, Any]:
    return regions_env()


class Ec2s:
    """Each region's EC2 client, RunInstances refused per AZ from `codes` (AZ -> list of
    codes, one per attempt), launched by moto once an AZ's list is empty."""

    def __init__(self, env: dict[str, Any], codes: dict[str, list[str]] | None = None) -> None:
        self.env = env
        self.codes = {az: list(c) for az, c in (codes or {}).items()}
        self.calls: list[str] = []  # AZ of each RunInstances
        self.az_of = {
            sid: az
            for r in env["regions"].values()
            for sid, az in zip(r["subnets"], r["azs"], strict=True)
        }
        self.clients = {
            region: Proxy(r["ec2"], run_instances=self._run(r["ec2"]))
            for region, r in env["regions"].items()
        }

    def _run(self, inner: Any) -> Any:
        def run_instances(**kw: Any) -> Any:
            az = self.az_of[kw["SubnetId"]]
            self.calls.append(az)
            if self.codes.get(az):
                raise client_error(self.codes[az].pop(0))
            return inner.run_instances(**kw)

        return run_instances


def make(
    env: dict[str, Any],
    ec2: dict[str, Any] | None = None,
    ssm: dict[str, Any] | None = None,
    **settings: Any,
) -> tuple[AwsEc2Provider, FakeTime]:
    t = FakeTime()
    ec2 = ec2 if ec2 is not None else {r: v["ec2"] for r, v in env["regions"].items()}
    ssm = ssm or {}

    def clients(service: str, region: str) -> Any:
        if service == "ec2":
            return ec2[region]
        if service == "ssm":
            return ssm.setdefault(region, FakeSsm())
        raise AssertionError(f"unexpected client {service} in {region}")

    p = AwsEc2Provider(
        env["settings"].model_copy(update=settings),
        prices=load_prices(),
        s3=boto3.client("s3", region_name=PRIMARY),
        requirements=locked,
        clock=t.clock,
        sleep=t.sleep,
        clients=clients,
    )
    return p, t


def instances(ec2: Any) -> list[dict[str, Any]]:
    out = [i for r in ec2.describe_instances()["Reservations"] for i in r["Instances"]]
    return [i for i in out if i["State"]["Name"] != "terminated"]


# --- settings ------------------------------------------------------------------------------

BASE_YAML = (
    "bucket: b-1\ninstance_profile_name: p\nsecurity_group_id: sg-0abc\n"
    "subnet_ids: [subnet-0a]\nhf_token_secret_name: loom/hf\nowner: arul\n"
)
EXTRA_YAML = (
    "extra_regions:\n"
    "  - {region: us-east-2, security_group_id: sg-0e2, subnet_ids: [subnet-0e2a]}\n"
    "  - {region: us-west-2, security_group_id: sg-0f2, subnet_ids: [subnet-0f2a, subnet-0f2b],"
    " ami_id: ami-0fec8acfc603e6e16}\n"
)


def test_a_single_region_config_is_unchanged(tmp_path: Path) -> None:
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(BASE_YAML)
    s = load_aws_settings(cfg, env={"LOOM_AWS_AMI_ID": "ami-028357be7d5b15c53"})
    assert s.extra_regions == [] and s.region_names == ["us-east-1"]
    (only,) = s.host_regions
    assert only == AwsRegion(
        region="us-east-1",
        security_group_id="sg-0abc",
        subnet_ids=["subnet-0a"],
        ami_id="ami-028357be7d5b15c53",
    )


def test_extra_regions_load_in_order_and_the_env_selects_among_them(tmp_path: Path) -> None:
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(BASE_YAML + EXTRA_YAML)
    s = load_aws_settings(cfg, env={})
    assert s.region_names == ["us-east-1", "us-east-2", "us-west-2"]
    assert s.host_regions[2].ami_id == "ami-0fec8acfc603e6e16"
    assert s.host_regions[2].subnet_ids == ["subnet-0f2a", "subnet-0f2b"]
    only_west = load_aws_settings(cfg, env={"LOOM_AWS_EXTRA_REGIONS": "us-west-2"})
    assert only_west.region_names == ["us-east-1", "us-west-2"]
    reordered = load_aws_settings(cfg, env={"LOOM_AWS_EXTRA_REGIONS": "us-west-2, us-east-2"})
    assert reordered.region_names == ["us-east-1", "us-west-2", "us-east-2"]
    assert load_aws_settings(cfg, env={"LOOM_AWS_EXTRA_REGIONS": ""}).region_names == ["us-east-1"]
    with pytest.raises(ValueError, match="names eu-west-1, which the AWS settings file"):
        load_aws_settings(cfg, env={"LOOM_AWS_EXTRA_REGIONS": "eu-west-1"})


def test_a_region_listed_twice_is_refused(tmp_path: Path) -> None:
    cfg = tmp_path / "aws.yaml"
    twice = "  - {region: us-east-1, security_group_id: sg-0e, subnet_ids: [subnet-0e]}\n"
    cfg.write_text(BASE_YAML + "extra_regions:\n" + twice)
    with pytest.raises(ValueError, match="regions listed more than once: us-east-1"):
        load_aws_settings(cfg, env={})


def test_make_provider_accepts_an_experiment_in_any_configured_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(BASE_YAML + EXTRA_YAML + f"wheel_path: {tmp_path / 'w.whl'}\n")
    monkeypatch.setenv("LOOM_AWS_CONFIG", str(cfg))
    exp = load_experiment(QWEN_EXPERIMENT)
    spec = exp.provider.model_copy(update={"region": "us-west-2"})
    p = make_provider(spec, prices=load_prices(), work_dir=tmp_path)
    assert isinstance(p, AwsEc2Provider)
    with pytest.raises(ValueError, match=r"eu-west-1 is not one of the AWS settings' regions"):
        make_provider(
            exp.provider.model_copy(update={"region": "eu-west-1"}),
            prices=load_prices(),
            work_dir=tmp_path,
        )


# --- one sweep, every region ---------------------------------------------------------------


async def test_no_capacity_in_the_primary_launches_in_the_next_region(
    env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=AWS_LOG)
    r = env["regions"]
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    p, t = make(env, ec2.clients)
    host = await p.provision(request())

    assert ec2.calls == r[PRIMARY]["azs"] + r[EAST2]["azs"] + r[WEST2]["azs"][:1]
    assert t.slept == []  # one sweep: no wait
    (inst,) = instances(r[WEST2]["ec2"])
    assert inst["InstanceId"] == host.host_id
    assert instances(r[PRIMARY]["ec2"]) == instances(r[EAST2]["ec2"]) == []
    # The region's own network and its copy of the primary's AMI release.
    assert inst["ImageId"] == r[WEST2]["ami"]
    assert [g["GroupId"] for g in inst["SecurityGroups"]] == [r[WEST2]["sg"]]
    assert inst["SubnetId"] == r[WEST2]["subnets"][0]
    assert host.info["region"] == WEST2 and host.info["az"] == "us-west-2a"
    assert host.info["ami"] == r[WEST2]["ami"]
    # Priced at us-west-2's prices (equal to us-east-1's today).
    assert host.hourly_micros == 1_861_000 + 21_918
    assert host.price_basis is not None and host.price_basis.availability_zone == "us-west-2a"
    tags = {t["Key"]: t["Value"] for t in inst["Tags"]}
    assert tags["loom:managed"] == "true" and "loom:ttl" in tags


async def test_the_primary_is_tried_first_and_other_regions_untouched_when_it_launches(
    env: dict[str, Any],
) -> None:
    ec2 = Ec2s(env)
    p, _ = make(env, ec2.clients)
    host = await p.provision(request())
    assert ec2.calls == ["us-east-1a"]
    assert host.info["region"] == PRIMARY
    assert host.info["ami"] == env["regions"][PRIMARY]["ami"]


async def test_a_request_for_an_extra_region_sweeps_it_first(env: dict[str, Any]) -> None:
    ec2 = Ec2s(env, {"us-west-2a": [IIC]})
    p, _ = make(env, ec2.clients)
    host = await p.provision(request(region=WEST2))
    assert ec2.calls == ["us-west-2a", "us-west-2b"]
    assert host.info["region"] == WEST2
    with pytest.raises(ValueError, match="eu-west-1 is not one of the settings' regions"):
        await p.provision(request(region="eu-west-1"))


async def test_a_full_sweep_of_every_region_is_retried_on_the_backoff(
    env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=AWS_LOG)
    r = env["regions"]
    every = r[PRIMARY]["azs"] + r[EAST2]["azs"] + r[WEST2]["azs"]
    ec2 = Ec2s(env, {az: [IIC, IIC] for az in every} | {"us-west-2b": [IIC, IIC, IIC]})
    p, t = make(env, ec2.clients)
    host = await p.provision(request())
    assert ec2.calls == every * 2 + r[PRIMARY]["azs"][:1]
    assert t.slept == [30, 60]
    assert host.info["region"] == PRIMARY
    first = next(rec.getMessage() for rec in caplog.records if rec.name == AWS_LOG)
    assert first == (
        "AWS g6e.xlarge capacity: none in us-east-1a, us-east-1b, us-east-2a, us-east-2b, "
        "us-west-2a, us-west-2b (InsufficientInstanceCapacity) (waited 0 s, 1800 s left); "
        "next sweep in 30 s"
    )


# --- the vCPU quota is per region ----------------------------------------------------------


async def test_a_quota_refusal_skips_only_that_region(env: dict[str, Any]) -> None:
    # us-east-1 out of capacity, us-east-2's quota held (say by another host): the sweep
    # skips us-east-2's other AZ and launches in us-west-2, at once.
    r = env["regions"]
    ec2 = Ec2s(env, {**{az: [IIC] for az in r[PRIMARY]["azs"]}, "us-east-2a": [VCPU]})
    p, t = make(env, ec2.clients)
    host = await p.provision(request())
    assert ec2.calls == r[PRIMARY]["azs"] + ["us-east-2a", "us-west-2a"]
    assert host.info["region"] == WEST2 and t.slept == []


async def test_quota_in_one_region_and_capacity_elsewhere_waits_for_capacity(
    env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=AWS_LOG)
    r = env["regions"]
    codes = {az: [IIC] * 30 for az in r[PRIMARY]["azs"] + r[WEST2]["azs"]}
    ec2 = Ec2s(env, codes | {"us-east-2a": [VCPU] * 30})
    p, t = make(env, ec2.clients, capacity_wait_s=1000)
    with pytest.raises(NoCapacity) as err:
        await p.provision(request())
    # Not the 600 s quota limit: capacity elsewhere is what was waited for, to the deadline.
    assert sum(t.slept) == 1000
    assert "us-east-2a: VcpuLimitExceeded" in str(err.value)
    first = next(rec.getMessage() for rec in caplog.records if rec.name == AWS_LOG)
    assert first == (
        "AWS g6e.xlarge capacity: none in us-east-1a, us-east-1b, us-west-2a, us-west-2b "
        "(InsufficientInstanceCapacity); vCPU quota held in us-east-2 (VcpuLimitExceeded) "
        "(waited 0 s, 1000 s left); next sweep in 30 s"
    )
    assert all(instances(v["ec2"]) == [] for v in r.values())


async def test_quota_held_in_every_region_is_waited_out_then_fails_naming_them(
    env: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=AWS_LOG)
    r = env["regions"]
    ec2 = Ec2s(env, {r[x]["azs"][0]: [VCPU] * 30 for x in r})
    p, t = make(env, ec2.clients)
    with pytest.raises(VcpuQuotaHeld) as err:
        await p.provision(request())
    assert QUOTA_WAIT_S == 600
    assert t.slept == [30, 60, 120, 240, 150]
    assert ec2.calls[:3] == ["us-east-1a", "us-east-2a", "us-west-2a"]  # one AZ per region
    msg = str(err.value)
    assert "(VcpuLimitExceeded) in us-east-1, us-east-2, us-west-2 for 600 s" in msg
    assert "describe-instances --region <region>" in msg
    lines = [rec.getMessage() for rec in caplog.records if rec.name == AWS_LOG]
    assert lines[0].startswith(
        "AWS g6e.xlarge: vCPU quota still held (VcpuLimitExceeded) in us-east-1, us-east-2, "
        "us-west-2, most likely by the host just terminated"
    )


async def test_quota_that_clears_in_another_region_launches_there(env: dict[str, Any]) -> None:
    r = env["regions"]
    ec2 = Ec2s(
        env,
        {"us-east-1a": [VCPU] * 5, "us-east-2a": [VCPU] * 5, "us-west-2a": [VCPU]},
    )
    p, t = make(env, ec2.clients)
    host = await p.provision(request())
    assert t.slept == [30]
    assert host.info["region"] == WEST2
    assert instances(r[WEST2]["ec2"])[0]["InstanceId"] == host.host_id


# --- unpriced regions ----------------------------------------------------------------------


async def test_a_region_where_the_type_is_unpriced_is_not_tried(env: dict[str, Any]) -> None:
    # prices.yaml prices only g6e.xlarge in us-east-2 and us-west-2.
    r = env["regions"]
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"]})
    p, _ = make(env, ec2.clients, capacity_wait_s=0)
    with pytest.raises(NoCapacity):
        await p.provision(request(instance_type="g6e.2xlarge"))
    assert ec2.calls == r[PRIMARY]["azs"]


# --- AMI per region ------------------------------------------------------------------------


async def test_a_pinned_extra_region_ami_is_used(moto_aws: None) -> None:
    env = regions_env(pin_extra=True)
    r = env["regions"]
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"]})
    p, _ = make(env, ec2.clients)
    host = await p.provision(request())
    assert host.info["region"] == EAST2 and host.info["ami"] == r[EAST2]["ami"]


async def test_a_pinned_ami_of_another_release_fails_before_any_launch(moto_aws: None) -> None:
    env = regions_env()
    r = env["regions"]
    west = r[WEST2]["ec2"]
    other = west.copy_image(
        SourceImageId=r[WEST2]["ami"], SourceRegion=WEST2, Name=DLAMI_NAME.replace("1006", "1027")
    )["ImageId"]
    settings: AwsSettings = env["settings"]
    east2, west2 = settings.extra_regions
    extra = [east2, west2.model_copy(update={"ami_id": other})]
    ec2 = Ec2s(env)
    p, _ = make(env, ec2.clients, extra_regions=extra)
    wrong = r"us-west-2: ami_id ami-\w+ is '.*20261027', not '.*20261006'"
    with pytest.raises(ValueError, match=wrong):
        await p.provision(request())
    assert ec2.calls == []


async def test_a_region_without_the_release_fails_before_any_launch(moto_aws: None) -> None:
    env = regions_env()
    r = env["regions"]
    r[EAST2]["ec2"].deregister_image(ImageId=r[EAST2]["ami"])
    ec2 = Ec2s(env)
    p, _ = make(env, ec2.clients)
    with pytest.raises(ValueError, match=r"us-east-2: 0 images named .*20261006.*set that region"):
        await p.provision(request())
    assert ec2.calls == []


async def test_an_unpinned_primary_takes_the_release_from_its_parameter(moto_aws: None) -> None:
    env = regions_env()
    r = env["regions"]
    ssm = boto3.client("ssm", region_name=PRIMARY)
    ssm.put_parameter(Name="/loom/test/dlami", Value=r[PRIMARY]["ami"], Type="String")
    fake = FakeSsm()
    fake.params = ssm
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    p, _ = make(
        env, ec2.clients, {PRIMARY: fake}, ami_id=None, dlami_ssm_parameter="/loom/test/dlami"
    )
    host = await p.provision(request())
    assert host.info["ami"] == r[WEST2]["ami"]


# --- after the launch: the host's own region -----------------------------------------------


async def test_ssm_describe_and_teardown_go_to_the_hosts_region(env: dict[str, Any]) -> None:
    r = env["regions"]
    holder: dict[str, Host] = {}
    west_ssm = FakeSsm(
        lambda script: [
            {
                "Status": "Success",
                "StandardOutputContent": engine_stdout(holder["host"].launched_at.timestamp()),
            }
        ]
    )
    primary_ssm = FakeSsm(lambda script: pytest.fail("SSM sent to the primary region"))
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    p, _ = make(env, ec2.clients, {PRIMARY: primary_ssm, WEST2: west_ssm})
    host = holder["host"] = await p.provision(request())

    # The primary region does not know the instance: had any call gone there, the host
    # would read as lost.
    assert (await p.check_host(host))["InstanceId"] == host.host_id
    ep = await p.start_engine(host, render_launch(load_registry().get("qwen3-8b")), warm=False)
    assert ep.system["region"] == WEST2 and ep.system["az"] == "us-west-2a"
    assert west_ssm.sent[0]["InstanceIds"] == [host.host_id]
    assert west_ssm.sent[0]["OutputS3BucketName"] == "loom-bench-test"  # the primary's bucket
    # The HF token is read from the primary region's secret, wherever the host runs.
    assert "REGION=us-east-1" in west_ssm.scripts[0]
    await p.stop_engine(host)
    assert primary_ssm.sent == []

    await p.teardown(host)
    assert state(r[WEST2]["ec2"], host.host_id) == "terminated"


async def test_provenance_records_the_launch_region_and_the_config_hash_ignores_it(
    env: dict[str, Any],
) -> None:
    from types import SimpleNamespace

    from loom_bench.provenance import GitInfo, build_provenance
    from loom_bench.runner import _Executor

    r = env["regions"]
    holder: dict[str, Host] = {}
    west_ssm = FakeSsm(
        lambda script: [
            {
                "Status": "Success",
                "StandardOutputContent": engine_stdout(holder["host"].launched_at.timestamp()),
            }
        ]
    )
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    p, _ = make(env, ec2.clients, {WEST2: west_ssm})
    host = holder["host"] = await p.provision(request())
    cells = expand(load_experiment(QWEN_EXPERIMENT), load_registry())
    cell = next(c for c in cells if c.launch.image)
    ep = await p.start_engine(host, cell.launch, warm=False)
    sections = _Executor._serving_sections(
        SimpleNamespace(git=GitInfo()),  # type: ignore[arg-type]
        host,
        cell,
        ep,
    )
    prov = build_provenance(cell.config, **sections)
    assert prov.region == WEST2  # where it ran, which prices its cost
    assert cell.config["hardware"]["region"] == PRIMARY  # the experiment's, so the hash
    assert prov.config_hash == cell.config_hash  # compares with us-east-1 runs
    assert prov.price_basis is not None and prov.price_basis.availability_zone == "us-west-2a"


async def test_reap_covers_every_configured_region(env: dict[str, Any]) -> None:
    from datetime import timedelta

    r = env["regions"]
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    p, t = make(env, ec2.clients)
    host = await p.provision(request())
    assert await p.reap(t.now) == []
    assert await p.reap(t.now + timedelta(hours=2)) == [host.host_id]
    assert state(r[WEST2]["ec2"], host.host_id) == "terminated"


async def test_bench_reap_takes_one_client_per_region(env: dict[str, Any], tmp_path: Path) -> None:
    from datetime import timedelta

    from loom_bench.runner import reap
    from loom_bench.store.db import upgrade

    r = env["regions"]
    ec2 = Ec2s(env, {az: [IIC] for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    p, t = make(env, ec2.clients)
    host = await p.provision(request())
    db = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(db)
    clients = [r[x]["ec2"] for x in (PRIMARY, EAST2, WEST2)]  # as `bench reap` builds them
    out = await reap(db, now=t.now + timedelta(hours=2), ec2=clients)
    assert [(x.resource_id, x.action) for x in out] == [(host.host_id, "terminated")]
    assert state(r[WEST2]["ec2"], host.host_id) == "terminated"


# --- idempotent RunInstances in another region ---------------------------------------------


def _token_clients(env: dict[str, Any], west_script: list[str]) -> tuple[dict[str, Any], TokenEc2]:
    r = env["regions"]
    refuse = Ec2s(env, {az: [IIC] * 10 for az in r[PRIMARY]["azs"] + r[EAST2]["azs"]})
    west = TokenEc2(r[WEST2]["ec2"], west_script)
    return {**refuse.clients, WEST2: west}, west


async def test_an_unanswered_launch_in_another_region_is_adopted_there(
    env: dict[str, Any],
) -> None:
    clients, west = _token_clients(env, ["launch+timeout"])
    p, t = make(env, clients)
    host = await p.provision(request())
    (inst,) = instances(env["regions"][WEST2]["ec2"])
    assert host.host_id == inst["InstanceId"] and host.info["region"] == WEST2
    assert west.calls[0] == west.calls[1] and len(west.calls) == 2  # resent, same token
    assert t.slept == [30]


async def test_giving_up_terminates_what_the_token_launched_in_its_region(
    env: dict[str, Any],
) -> None:
    clients, west = _token_clients(env, ["launch+timeout"])
    west.lose_replies = True
    p, _ = make(env, clients)
    with pytest.raises(Ec2Unreachable, match="terminated i-"):
        await p.provision(request())
    assert instances(env["regions"][WEST2]["ec2"]) == []


# --- the plan's cap uses the worst region --------------------------------------------------


def test_the_plan_prices_a_host_at_the_worst_configured_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(BASE_YAML + EXTRA_YAML)
    exp = load_experiment(QWEN_EXPERIMENT)
    exp = exp.model_copy(
        update={"provider": exp.provider.model_copy(update={"market": "on_demand"})}
    )
    cell = expand(exp, load_registry())[0]
    prices = load_prices()

    monkeypatch.delenv("LOOM_AWS_CONFIG", raising=False)
    single, _, notes = host_price(exp, cell, prices)
    assert single == 1_861_000 + 21_918 and not any("worst" in n for n in notes)

    monkeypatch.setenv("LOOM_AWS_CONFIG", str(cfg))
    same, _, notes = host_price(exp, cell, prices)
    assert same == single  # the three regions' prices are equal today
    assert any("us-east-1 $1.8829/h, us-east-2 $1.8829/h, us-west-2 $1.8829/h" in n for n in notes)

    dear = prices.model_copy(deep=True)
    west = dear.clouds["aws"]["us-west-2"].instances["g6e.xlarge"]
    dear.clouds["aws"]["us-west-2"].instances["g6e.xlarge"] = west.model_copy(
        update={"on_demand_per_hour": 2_000_000}
    )
    worst, _, notes = host_price(exp, cell, dear)
    assert worst == 2_000_000 + 21_918
    assert notes[-1].endswith(": us-west-2")
