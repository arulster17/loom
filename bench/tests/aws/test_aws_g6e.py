"""The Qwen3-8B AWS run (qwen3-8b-aws-g6e) and its smoke (aws-smoke-g6e), before either
spends anything. No aws_ec2 GPU run has happened yet, so everything the EC2 path does is
first run here, offline:

- the two specs agree on everything but scale (model, provider block, variants and so
  model selection and config hashes, workload profile, load mode, repetitions, SLO,
  quality section), and both plan under their caps even if the host lives to its TTL;
- the smoke's search runs every failure branch (a failing first point, descent, a pass,
  bisection) whatever the host's knee, and the real search repeats RunPod's 7a8237d0
  points where AWS has RunPod's knees, so `bench compare` meets equal loads;
- both specs end to end on the aws_ec2 provider (moto EC2 and S3, HostSim behind SSM):
  one on-demand host accrued at the prices.yaml rate plus its root volume, a cold start
  with user-data's TTL backstop armed at the host's TTL, a warm restart onto the FP8
  checkpoint the host downloads then, the chat workload's pinned dataset fetched once
  and read through the client container's read-only mount, every eval pass with the
  divergence half on the first only, fp8-kv8 gated against bf16 in-run, the host
  terminated at the end and nothing left for the reaper."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import pytest
import yaml
from moto import mock_aws
from sqlalchemy import select

from loom_bench.budget import caps_for, load_budget
from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.plan import build_plan
from loom_bench.prices import load_prices
from loom_bench.providers import aws_reaper
from loom_bench.providers.aws_ec2 import AwsEc2Provider, AwsSettings, load_aws_settings
from loom_bench.registry import load_registry
from loom_bench.runner import RunnerContext, next_search_load, run_experiment
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import BenchColdStart, BenchEvalRun, BenchRun, BenchSpend
from loom_bench.workloads import load_profile

from .conftest import REGION, amazon_ami
from .hostsim import BUCKET, HostSim, pairs, var

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("aws-e2e")]

REAL = EXPERIMENTS_DIR / "qwen3-8b-aws-g6e.yaml"
SMOKE = EXPERIMENTS_DIR / "aws-smoke-g6e.yaml"
REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()
DLAMI_PARAM = "/loom/test/dlami"
# g6e.xlarge on-demand ($1.861/h) + 200 GB gp3 ($0.08/GB-month over 730 h, rounded up).
ON_DEMAND_ACCRUAL = 1_861_000 + 21_918
# RunPod 7a8237d0 (Secure 1x L40S), chat-sharegpt: each cell's search points in order.
RUNPOD_CHAT_POINTS = {
    "bf16": [4.5, 3.0, 2.0, 2.449, 2.213],
    "fp8-kv8": [4.5, 6.75, 5.511, 4.98, 5.239],
}


def _plan(exp: Experiment):
    return build_plan(
        exp, expand(exp, REGISTRY), prices=PRICES, caps=caps_for(exp.budget.max_spend, BUDGET, 0)
    )


def _trajectory(load: Any, knee: float) -> list[tuple[float, bool]]:
    """The search's points in order, against a host that meets the SLO up to `knee`."""
    history: list[tuple[float, bool]] = []
    while (nxt := next_search_load(load, history)) is not None:
        history.append((nxt, nxt <= knee))
    return history


def _branches(history: list[tuple[float, bool]], lo: float, step: float) -> dict[str, bool]:
    descended = (
        len(history) >= 2
        and history[0] == (lo, False)
        and math.isclose(history[1][0], lo / step, rel_tol=1e-9)
    )
    bisected = False
    for i, (load, _) in enumerate(history):
        passes = [x for x, met in history[:i] if met]
        fails = [x for x, met in history[:i] if not met]
        if passes and fails:
            below = [x for x in passes if x < min(fails)]
            if below and math.isclose(load, math.sqrt(max(below) * min(fails)), rel_tol=1e-9):
                bisected = True
    return {"descent": descended, "bisection": bisected, "passed": any(m for _, m in history)}


# --- the two specs ----------------------------------------------------------------------


def test_the_smoke_mirrors_the_real_run_except_scale():
    smoke, real = load_experiment(SMOKE), load_experiment(REAL)
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model == "qwen3-8b"
    assert smoke.provider == real.provider
    assert real.provider.kind == "aws_ec2" and real.provider.market == "on_demand"
    assert real.provider.instance_type == "g6e.xlarge"
    assert smoke.variants == real.variants
    # The workload itself is the same (profile, no overrides: dataset download and
    # full-length chat); load mode, search scale and descent too.
    assert (
        [(w.profile, w.overrides) for w in smoke.workloads]
        == [(w.profile, w.overrides) for w in real.workloads]
        == [("chat-sharegpt", {})]
    )
    for s, r in zip(smoke.workloads, real.workloads, strict=True):
        assert s.load.mode == r.load.mode
        assert s.load.search is not None and r.load.search is not None
        assert s.load.search.scale == r.load.search.scale == "geometric"
        assert s.load.search.descend and r.load.search.descend
        assert s.load.drain_timeout_s == r.load.drain_timeout_s
    assert smoke.repetitions == real.repetitions == 3
    assert smoke.slo == real.slo
    assert smoke.quality is not None and real.quality is not None
    assert smoke.quality.limit is not None and real.quality.limit is None
    assert smoke.quality.model_copy(update={"limit": None}) == real.quality
    assert smoke.loadgen == real.loadgen == "native"


def test_the_smoke_selects_models_exactly_as_the_real_run_does():
    smoke = expand(load_experiment(SMOKE), REGISTRY)
    real = expand(load_experiment(REAL), REGISTRY)
    assert [c.key for c in smoke] == [c.key for c in real] == ["bf16", "fp8-kv8"]
    assert [c.config_hash for c in smoke] == [c.config_hash for c in real]
    assert [c.host_key for c in smoke] == [c.host_key for c in real]
    assert len({c.host_key for c in real}) == 1  # one host: the quota fits one g6e.xlarge
    bf16, fp8 = real
    assert (bf16.spec.id, fp8.spec.id) == ("qwen3-8b", "qwen3-8b-fp8")
    assert (bf16.spec.kv_cache_dtype, fp8.spec.kv_cache_dtype) == ("auto", "fp8")
    assert bf16.launch.image == fp8.launch.image
    # FP8 by its registry row (`Variant.model`), never an `hf:` override.
    doc = yaml.safe_load(REAL.read_text())
    assert not any("hf" in v or "quantization" in v for v in doc["variants"])


def test_the_aws_cells_are_the_runpod_sweeps_cells_on_other_hardware():
    sweep = {
        c.key: c
        for c in expand(
            load_experiment(EXPERIMENTS_DIR / "qwen3-8b-config-sweep-runpod.yaml"), REGISTRY
        )
    }
    for cell in expand(load_experiment(REAL), REGISTRY):
        other = sweep[cell.key]
        assert cell.spec.id == other.spec.id
        assert cell.spec.hf == other.spec.hf
        assert cell.spec.kv_cache_dtype == other.spec.kv_cache_dtype
        assert cell.launch.image == other.launch.image and cell.launch.args == other.launch.args
        assert cell.config_hash != other.config_hash  # the hardware is part of the hash


@pytest.mark.parametrize(("path", "cap"), [(SMOKE, "$5"), (REAL, "$12.50")], ids=["smoke", "real"])
def test_both_plan_under_their_caps_even_to_ttl(path, cap):
    from loom_bench.money import parse_usd

    exp = load_experiment(path)
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd(cap)
    assert plan.ttl_worst_micros <= plan.caps.effective
    (host,) = plan.hosts
    assert host.market == "on_demand" and host.hourly_micros == ON_DEMAND_ACCRUAL
    assert host.seconds < 0.9 * host.ttl_s
    kinds = [s.kind for s in host.steps]
    assert kinds[:3] == ["cold_start", "eval_setup", "dataset"]
    assert kinds.count("warm_start") == 1 and kinds.count("dataset") == 1


def test_the_smoke_search_runs_every_failure_branch_whatever_the_knee():
    (w,) = load_experiment(SMOKE).workloads
    search = w.load.search
    assert search is not None
    floor = search.lo / search.step**search.descend
    # From just above the floor to just below lo: every knee the host could have.
    for knee in (floor * 1.01, 0.5, 1.0, 1.5, 2.2, 3.0, 3.99, 4.5, 5.2, 6.0, 8.0, 11.9):
        b = _branches(_trajectory(w.load, knee), search.lo, search.step)
        assert b == {
            "descent": True,
            "bisection": knee >= search.lo / search.step**2,
            "passed": True,
        }, knee
    # The RunPod knees (bf16 2.2, fp8-kv8 5.2 req/s) both descend, pass and bisect.
    for knee in (2.2, 5.2):
        assert _branches(_trajectory(w.load, knee), search.lo, search.step)["bisection"]


@pytest.mark.parametrize("cell", ["bf16", "fp8-kv8"])
def test_the_real_search_repeats_the_runpod_points_at_the_runpod_knees(cell):
    # At a knee between RunPod's goodput and its first failing load, the AWS search
    # visits exactly 7a8237d0's loads: the cross-check compares them at equal load.
    (w,) = load_experiment(REAL).workloads
    knee = {"bf16": 2.3, "fp8-kv8": 5.3}[cell]
    points = [round(x, 3) for x, _ in _trajectory(w.load, knee)]
    assert points == RUNPOD_CHAT_POINTS[cell]


# --- end to end on the aws_ec2 provider ---------------------------------------------------


def _sharegpt(n: int = 40) -> bytes:
    convs = [
        {
            "id": f"c{i}",
            "conversations": [
                {"from": "human", "value": f"question {i} " + "word " * (5 + i % 7)},
                {"from": "gpt", "value": "answer " * (6 + i % 5)},
                {"from": "human", "value": f"follow up {i}"},
                {"from": "gpt", "value": "reply " * (4 + i % 3)},
            ],
        }
        for i in range(n)
    ]
    return json.dumps(convs).encode()


def _suite(also: list[str]) -> dict[str, Any]:
    return {
        "suite": "aws-g6e",
        "model": "qwen3-8b",
        "also_models": also,
        "seed": 1234,
        "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 500},
        "tasks": [
            {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
            {"name": "json_schema", "kind": "json_schema"},
        ],
        "divergence": {
            "prompts": 8,
            "top_k": 5,
            "max_new_tokens": 8,
            "noise_multiple": 2.0,
            "ceiling_top1": 0.6,
        },
    }


def at_test_scale(path: Path, tmp: Path, body: bytes) -> Experiment:
    """The shipped spec's model, provider, variants and replicate count; the chat
    workload with a tiny dataset pinned in its `download` block, short requests and
    one load point; a tiny suite gating the same cells."""
    doc = yaml.safe_load(path.read_text())
    suite = tmp / "suite.yaml"
    suite.write_text(yaml.safe_dump(_suite(["qwen3-8b-fp8"])))
    (w,) = doc["workloads"]
    w["overrides"] = {
        "download": {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)},
        "max_output_len": 8,
        "min_output_len": 1,
    }
    w["load"] = {
        "mode": "closed_loop",
        "values": [2],
        "num_requests": 6,
        "warmup_requests": 1,
        "scrape_interval_s": 0.05,
    }
    q = doc["quality"]
    doc.update(
        repetitions=1,
        allow_single_run=True,
        slo={"max_error_rate": 0.01},
        quality={
            "suite": str(suite),
            "baseline_variant": q["baseline_variant"],
            "replicates": q["replicates"],
        },
        budget={"max_spend": "$5", "ttl_minutes": 120, "accrual_interval_s": 0.2},
    )
    doc.pop("smoke", None)
    return Experiment.model_validate(doc)


def run_on_host_sim(path: Path, tmp: Path):
    mp = pytest.MonkeyPatch()
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": REGION,
    }.items():
        mp.setenv(k, v)
    mp.delenv("LOOM_AWS_CONFIG", raising=False)
    with mock_aws():
        ec2 = boto3.client("ec2", region_name=REGION)
        s3 = boto3.client("s3", region_name=REGION)
        ssm = boto3.client("ssm", region_name=REGION)
        iam = boto3.client("iam", region_name=REGION)
        iam.create_role(RoleName="loom-bench-instance", AssumeRolePolicyDocument="{}")
        iam.create_instance_profile(InstanceProfileName="loom-bench-instance")
        iam.add_role_to_instance_profile(
            InstanceProfileName="loom-bench-instance", RoleName="loom-bench-instance"
        )
        vpc = ec2.describe_vpcs(Filters=[{"Name": "is-default", "Values": ["true"]}])["Vpcs"][0]
        sg = ec2.create_security_group(GroupName="loom", Description="x", VpcId=vpc["VpcId"])
        subnets = sorted(ec2.describe_subnets()["Subnets"], key=lambda s: s["AvailabilityZone"])
        ssm.put_parameter(Name=DLAMI_PARAM, Value=amazon_ami(ec2), Type="String")
        s3.create_bucket(Bucket=BUCKET)
        db = f"sqlite:///{tmp / 'loom.db'}"
        upgrade(db)
        ctx = RunnerContext(
            db_url=db, out_dir=tmp / "results", registry=REGISTRY, prices=PRICES, budget=BUDGET
        )
        body = _sharegpt()
        exp = at_test_scale(path, tmp, body)
        cells = expand(exp, REGISTRY)
        url = load_profile("chat-sharegpt").download.url  # type: ignore[union-attr]
        sim = HostSim(ec2, s3, tmp / "hosts", served={url: body})
        sim.params = ssm
        wheel = tmp / "loom_bench-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        settings = AwsSettings(
            bucket=BUCKET,
            instance_profile_name="loom-bench-instance",
            security_group_id=sg["GroupId"],
            subnet_ids=[subnets[0]["SubnetId"], subnets[1]["SubnetId"]],
            hf_token_secret_name="loom/hf-token",
            owner="arul",
            dlami_ssm_parameter=DLAMI_PARAM,
            poll_interval_s=0.01,
        )

        async def no_sleep(_: float) -> None:
            await asyncio.sleep(0)

        ctx.provider = AwsEc2Provider(
            settings,
            prices=PRICES,
            ec2=ec2,
            ssm=sim,
            s3=s3,
            wheel_path=wheel,
            requirements=lambda extra: f"pkg-{extra or 'base'}==1.0 --hash=sha256:{'0' * 64}\n",
            sleep=no_sleep,
        )
        try:
            outcome = asyncio.run(run_experiment(exp, ctx))
            instances = [
                i for r in ec2.describe_instances()["Reservations"] for i in r["Instances"]
            ]
            far = datetime.now(UTC) + timedelta(days=2)
            reaped = aws_reaper.reap(ec2, far)
        finally:
            sim.close()
            mp.undo()
    return outcome, sim, ctx, cells, exp, instances, reaped


@pytest.fixture(scope="module", params=[SMOKE, REAL], ids=["smoke", "real"])
def aws_run(request, tmp_path_factory):
    return run_on_host_sim(request.param, tmp_path_factory.mktemp(request.param.stem))


def test_it_completes_on_one_on_demand_host_that_is_terminated(aws_run):
    outcome, sim, _, _, _, instances, reaped = aws_run
    assert outcome.status.value == "completed", outcome.reason
    (inst,) = instances
    assert inst["State"]["Name"] == "terminated"
    assert "InstanceLifecycle" not in inst  # on-demand, not spot
    assert {t["Key"]: t["Value"] for t in inst["Tags"]}["loom:managed"] == "true"
    assert reaped == []  # nothing left for the reaper
    assert {i for i, _, _ in sim.launched} == {inst["InstanceId"]}


def test_spend_accrues_at_the_on_demand_rate_plus_the_root_volume(aws_run):
    _, _, ctx, _, _, _, _ = aws_run
    with session_scope(ctx.db_url) as s:
        rows = list(s.scalars(select(BenchSpend)))
    assert rows
    for row in rows:
        text = json.dumps(row.basis)
        assert "on_demand" in text and str(ON_DEMAND_ACCRUAL) in text, row.basis


def test_cold_start_arms_the_ttl_backstop_and_warm_restart_fetches_the_fp8_checkpoint(aws_run):
    outcome, sim, ctx, cells, _, _, _ = aws_run
    bf16, fp8 = cells
    starts = sim.scripts_of("start_engine")
    assert [var(s, "WARM") for s in starts] == ["0", "1"]
    assert [var(s, "SERVED_MODEL") for s in starts] == ["qwen3-8b", "qwen3-8b-fp8"]
    bf16_ckpt = (bf16.spec.hf.repo, bf16.spec.hf.revision)
    fp8_ckpt = (fp8.spec.hf.repo, fp8.spec.hf.revision)
    assert pairs(starts[0], "FETCH_WEIGHTS") == [bf16_ckpt]
    assert pairs(starts[1], "FETCH_WEIGHTS") == [fp8_ckpt]
    (downloads,) = sim.downloads.values()
    assert downloads == [bf16_ckpt, fp8_ckpt]
    path = ctx.out_dir / str(outcome.experiment_id) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    started = [e for e in events if e["kind"] == "engine_started"]
    assert [e["cell"] for e in started] == ["bf16", "fp8-kv8"]
    cold = started[0]["system"]
    # Armed at the runner's TTL (the sim's systemd fires at TTL_EPOCH, a whole second).
    assert 0 <= cold["ttl_shutdown_lead_s"] <= 1  # rounded to 0.1 s
    assert "ttl_shutdown_at" not in started[1]["system"]  # a warm restart re-reads nothing
    with session_scope(ctx.db_url) as s:
        assert sorted(c.kind for c in s.scalars(select(BenchColdStart))) == ["cold", "warm"]


def test_the_chat_dataset_is_fetched_once_and_read_through_the_mount(aws_run):
    _, sim, ctx, cells, _, _, _ = aws_run
    (fetches,) = sim.dataset_fetches.values()
    assert len(fetches) == 1
    loads = [j for j in sim.jobs if isinstance(j, LoadJob)]
    assert len(loads) == len(cells)  # one point, one repetition per cell at test scale
    for job in loads:
        assert job.workload["path"].startswith("/data/")
        assert job.workload["path"].endswith("/ShareGPT_V3_unfiltered_cleaned_split.json")
    for script in sim.scripts_of("run_job"):
        if var(script, "DATA_URL"):
            assert var(script, "DATA_DIR") == "/opt/dlami/nvme/loom-data"
            assert var(script, "DATA_MOUNT") == "/data"
            assert "Signature=" in var(script, "RESULT_URL")  # presigned
    with session_scope(ctx.db_url) as s:
        runs = list(s.scalars(select(BenchRun)))
    assert {r.status for r in runs} == {"completed"}
    assert {r.workload for r in runs} == {"chat-sharegpt"}


def test_every_cell_runs_its_passes_and_fp8_kv8_is_gated_against_bf16(aws_run):
    outcome, sim, ctx, cells, exp, _, _ = aws_run
    replicates = exp.quality.replicates  # type: ignore[union-attr]
    assert replicates == 3
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert len(evals) == replicates * len(cells)
    modes = [j.divergence for j in evals]
    assert modes[:replicates] == ["capture_and_floor"] + [None] * (replicates - 1)
    assert modes[replicates:] == ["score"] + [None] * (replicates - 1)
    assert evals[replicates].reference is not None
    assert evals[replicates].reference.config_hash == cells[0].config_hash
    assert [j.served_model for j in evals] == ["qwen3-8b"] * replicates + [
        "qwen3-8b-fp8"
    ] * replicates
    (gate,) = outcome.gates
    assert (gate.cell, gate.baseline) == ("fp8-kv8", "bf16")
    with session_scope(ctx.db_url) as s:
        evs = list(s.scalars(select(BenchEvalRun)))
    assert {e.config_hash for e in evs} == {c.config_hash for c in cells}


# --- provider settings ------------------------------------------------------------------


def test_a_pinned_ami_skips_the_dlami_parameter(tmp_path):
    cfg = tmp_path / "aws.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "bucket": "b-123",
                "instance_profile_name": "p",
                "security_group_id": "sg-0123",
                "subnet_ids": ["subnet-0123"],
                "hf_token_secret_name": "loom/hf-token",
                "owner": "arul",
            }
        )
    )
    s = load_aws_settings(cfg, env={"LOOM_AWS_AMI_ID": "ami-028357be7d5b15c53"})
    assert s.ami_id == "ami-028357be7d5b15c53"
    assert load_aws_settings(cfg, env={}).ami_id is None
    with pytest.raises(ValueError):
        load_aws_settings(cfg, env={"LOOM_AWS_AMI_ID": "latest"})

    class NoParam:
        def get_parameter(self, **_: Any) -> Any:
            raise AssertionError("the DLAMI parameter must not be read when an AMI is pinned")

    class Images:
        def describe_images(self, ImageIds: list[str]) -> Any:
            return {"Images": [{"ImageId": ImageIds[0], "RootDeviceName": "/dev/sda1"}]}

    p = AwsEc2Provider(s, prices=PRICES, ec2=Images(), ssm=NoParam(), s3=object())
    assert p._ami() == ("ami-028357be7d5b15c53", "/dev/sda1")
