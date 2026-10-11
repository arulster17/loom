"""SGLang on AWS (qwen3-8b-sglang-aws-g6e) and its smoke (aws-smoke-g6e-sglang), before
either spends anything: the vLLM half of the AWS engine comparison ran as 95cde129
(qwen3-8b-aws-g6e's bf16 cell), and SGLang has never run on the aws_ec2 path.

- The SGLang run is 95cde129's bf16 cell with the engine switched: same provider block,
  chat workload, windows, search, repetitions, SLO and quality subset and passes, gated
  against that cell's stored evaluation (its config hash is the vLLM spec's bf16 hash),
  and its SGLang launch is RunPod 565b8d3f's. At a vLLM-like knee its search visits
  vLLM's loads exactly, so the leaderboard compares the engines at equal load.
- The smoke mirrors it except scale, and gates against aws-smoke-g6e 80ff5648's bf16
  cell, stored at the same item limit.
- Both plan under their caps even to their TTLs, and together under the approved $8.
- End to end on the aws_ec2 provider (moto EC2 and S3, HostSim behind SSM): a vLLM BF16
  baseline run first stores its evaluation, then the SGLang spec runs on its own
  on-demand host: the SGLang image and `python3 -m sglang.launch_server` cold start
  (offline, its checkpoint fetched once, the TTL backstop armed), the chat dataset read
  through the client container's mount, load jobs tagged for SGLang's metric names, every
  eval pass scored on the stored reference (SGLang's completions logprob form) and gated
  against the stored baseline, the host terminated, and one leaderboard ranking both
  engines with latency at equal load."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import select

from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.money import parse_usd
from loom_bench.report.leaderboard import build_leaderboard
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchEvalRun, BenchSpend

from .hostsim import array, pairs, var
from .test_aws_g6e import (
    ON_DEMAND_ACCRUAL,
    REGISTRY,
    RUNPOD_CHAT_POINTS,
    _branches,
    _plan,
    _suite,
    _trajectory,
    at_test_scale,
    run_on_host_sim,
)

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("aws-e2e")]

REAL = EXPERIMENTS_DIR / "qwen3-8b-sglang-aws-g6e.yaml"
SMOKE = EXPERIMENTS_DIR / "aws-smoke-g6e-sglang.yaml"
VLLM_REAL = EXPERIMENTS_DIR / "qwen3-8b-aws-g6e.yaml"
VLLM_SMOKE = EXPERIMENTS_DIR / "aws-smoke-g6e.yaml"
RUNPOD_ENGINES = EXPERIMENTS_DIR / "qwen3-8b-vllm-vs-sglang-runpod.yaml"
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
)
# The stored vLLM BF16 evaluations: 95cde129 (qwen3-8b-aws-g6e, full size) and 80ff5648
# (aws-smoke-g6e, 4 items per task), both 3 phase0-strict passes on the bf16 cell.
VLLM_RUN = uuid.UUID("95cde129-e626-4eda-bad6-1421ed4d8a7b")
VLLM_SMOKE_RUN = uuid.UUID("80ff5648-50a1-49af-9fbe-97f8fa3dd1cb")
# The user-approved total for the smoke and the real run together (2026-10-11).
APPROVED_TOTAL = parse_usd("$8")


def _vllm_bf16(path: Path = VLLM_REAL):
    return {c.key: c for c in expand(load_experiment(path), REGISTRY)}["bf16"]


# --- the two specs ----------------------------------------------------------------------


def test_the_sglang_run_is_the_vllm_runs_bf16_cell_on_sglang():
    sg, vl = load_experiment(REAL), load_experiment(VLLM_REAL)
    assert not sg.smoke
    assert sg.model == vl.model == "qwen3-8b"
    assert sg.provider == vl.provider
    assert sg.provider.kind == "aws_ec2" and sg.provider.market == "on_demand"
    # Same workload entry: profile, windows, drain and search, so equal loads meet.
    assert sg.workloads == vl.workloads
    assert sg.repetitions == vl.repetitions == 3
    assert sg.slo == vl.slo
    assert sg.cost_allocation == vl.cost_allocation
    assert sg.loadgen == vl.loadgen == "native"
    assert sg.seed == vl.seed  # the same seeded arrivals and prompts per load point
    q, vq = sg.quality, vl.quality
    assert q is not None and vq is not None
    assert (q.suite, q.subset, q.replicates) == (vq.suite, vq.subset, vq.replicates)
    assert q.limit is None and q.baseline_variant is None
    # Gated against 95cde129's bf16 cell: the config the vLLM spec serves as bf16.
    assert q.baseline is not None
    assert q.baseline.experiment == VLLM_RUN
    assert q.baseline.config_hash == _vllm_bf16().config_hash
    assert vq.baseline_variant == "bf16"


def test_the_sglang_cell_differs_from_vllm_bf16_only_by_engine():
    (sg,) = expand(load_experiment(REAL), REGISTRY)
    vl = _vllm_bf16()
    assert sg.key == "sglang"
    assert sg.spec.engine.name == "sglang" and sg.spec.engine.version == "0.5.21"
    assert sg.launch.image == SGLANG_IMAGE
    assert sg.spec.model_dump(exclude={"engine"}) == vl.spec.model_dump(exclude={"engine"})
    assert sg.spec.engine.chat_template_kwargs == {"enable_thinking": False}
    assert sg.hardware == vl.hardware and sg.host_key == vl.host_key
    assert sg.config_hash != vl.config_hash
    # RunPod 565b8d3f's SGLang launch, unchanged: only the hardware differs.
    runpod = {c.key: c for c in expand(load_experiment(RUNPOD_ENGINES), REGISTRY)}["sglang"]
    assert sg.launch.args == runpod.launch.args and sg.launch.image == runpod.launch.image
    assert sg.launch.args[:2] == ["--model-path", "Qwen/Qwen3-8B"]
    assert "--enable-metrics" in sg.launch.args
    assert sg.launch.args[sg.launch.args.index("--tool-call-parser") + 1] == "qwen25"


def test_the_smoke_mirrors_the_real_run_except_scale():
    smoke, real = load_experiment(SMOKE), load_experiment(REAL)
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model
    assert smoke.provider == real.provider
    assert smoke.variants == real.variants
    assert [c.config_hash for c in expand(smoke, REGISTRY)] == [
        c.config_hash for c in expand(real, REGISTRY)
    ]
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
    assert smoke.repetitions == real.repetitions
    assert smoke.slo == real.slo
    sq, rq = smoke.quality, real.quality
    assert sq is not None and rq is not None and sq.baseline and rq.baseline
    assert sq.limit is not None and rq.limit is None
    # The same config as baseline, stored at the smoke's own item limit by the vLLM smoke.
    assert sq.baseline.config_hash == rq.baseline.config_hash
    assert sq.baseline.experiment == VLLM_SMOKE_RUN
    assert _vllm_bf16(VLLM_SMOKE).config_hash == sq.baseline.config_hash
    vllm_smoke = load_experiment(VLLM_SMOKE).quality
    assert vllm_smoke is not None
    assert (vllm_smoke.limit, vllm_smoke.replicates) == (sq.limit, sq.replicates)
    unscaled = sq.model_copy(update={"limit": None, "baseline": rq.baseline})
    assert unscaled == rq


def test_both_plan_under_their_caps_even_to_ttl_and_within_the_approved_total():
    worst = 0
    for path, cap in ((SMOKE, "$2.50"), (REAL, "$5.50")):
        plan = _plan(load_experiment(path))
        assert plan.ok, plan.refusals
        assert plan.caps.effective == parse_usd(cap)
        assert plan.ttl_worst_micros <= plan.caps.effective
        (host,) = plan.hosts
        assert host.market == "on_demand" and host.hourly_micros == ON_DEMAND_ACCRUAL
        assert host.seconds < 0.92 * host.ttl_s
        kinds = [s.kind for s in host.steps]
        assert kinds[:3] == ["cold_start", "eval_setup", "dataset"]
        assert "warm_start" not in kinds  # one engine, one cold start
        assert plan.n_cells == 1
        worst += plan.ttl_worst_micros
    assert worst <= APPROVED_TOTAL


def test_the_smoke_search_runs_every_failure_branch_whatever_the_knee():
    (w,) = load_experiment(SMOKE).workloads
    search = w.load.search
    assert search is not None
    floor = search.lo / search.step**search.descend
    for knee in (floor * 1.01, 0.5, 1.0, 1.5, 2.2, 3.0, 3.99, 4.5, 5.2, 6.0, 8.0, 11.9):
        b = _branches(_trajectory(w.load, knee), search.lo, search.step)
        assert b == {
            "descent": True,
            "bisection": knee >= search.lo / search.step**2,
            "passed": True,
        }, knee


def test_at_a_vllm_like_knee_the_search_visits_vllms_loads():
    # 95cde129's bf16 search visited RunPod 7a8237d0's loads exactly; an SGLang knee in
    # the same bracket repeats them, so every point is compared at equal load.
    (w,) = load_experiment(REAL).workloads
    points = [round(x, 3) for x, _ in _trajectory(w.load, 2.3)]
    assert points == RUNPOD_CHAT_POINTS["bf16"]


# --- end to end on the aws_ec2 provider ---------------------------------------------------


def vllm_baseline_at_test_scale(path: Path, tmp: Path, body: bytes) -> Experiment:
    """The vLLM AWS spec's bf16 cell alone at test scale: the stored baseline."""
    exp = at_test_scale(path, tmp, body)
    return Experiment.model_validate(
        {**exp.model_dump(mode="json"), "variants": [{"name": "bf16"}]}
    )


def sglang_at_test_scale(baseline: dict[str, str]):
    def build(path: Path, tmp: Path, body: bytes) -> Experiment:
        """The SGLang spec as at_test_scale shrinks the vLLM one (the same tiny suite and
        chat load), gated against `baseline` as stored by the test-scale vLLM run."""
        doc = yaml.safe_load(path.read_text())
        suite = tmp / "suite.yaml"
        suite.write_text(yaml.safe_dump(_suite(["qwen3-8b-fp8"])))
        shrunk = at_test_scale(VLLM_REAL, tmp, body).model_dump(mode="json")
        q = doc["quality"]
        shrunk.update(
            name=doc["name"],
            description=doc["description"],
            variants=doc["variants"],
            quality={"suite": str(suite), "baseline": baseline, "replicates": q["replicates"]},
        )
        return Experiment.model_validate(shrunk)

    return build


@pytest.fixture(scope="module")
def stored_vllm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vllm-baseline")
    outcome, *_rest = run_on_host_sim(VLLM_REAL, tmp, build=vllm_baseline_at_test_scale)
    assert outcome.status.value == "completed", outcome.reason
    bf16 = _vllm_bf16()
    return (
        tmp / "loom.db",
        outcome,
        {
            "experiment": str(outcome.experiment_id),
            "config_hash": bf16.config_hash,
        },
    )


@pytest.fixture(scope="module", params=[SMOKE, REAL], ids=["smoke", "real"])
def sglang_run(request, stored_vllm, tmp_path_factory):
    db, vllm_outcome, baseline = stored_vllm
    tmp = tmp_path_factory.mktemp(request.param.stem)
    result = run_on_host_sim(request.param, tmp, build=sglang_at_test_scale(baseline), db=db)
    return (*result, baseline, vllm_outcome)


def test_it_completes_on_one_on_demand_host_that_is_terminated(sglang_run):
    outcome, sim, ctx, _, _, instances, reaped, _, _ = sglang_run
    assert outcome.status.value == "completed", outcome.reason
    (inst,) = instances
    assert inst["State"]["Name"] == "terminated"
    assert "InstanceLifecycle" not in inst  # on-demand
    assert reaped == []
    assert {i for i, _, _ in sim.launched} == {inst["InstanceId"]}
    with session_scope(ctx.db_url) as s:
        rows = list(
            s.scalars(select(BenchSpend).where(BenchSpend.experiment_id == outcome.experiment_id))
        )
    assert rows
    for row in rows:
        text = json.dumps(row.basis)
        assert "on_demand" in text and str(ON_DEMAND_ACCRUAL) in text, row.basis


def test_the_sglang_image_cold_starts_offline_with_the_ttl_backstop(sglang_run):
    outcome, sim, ctx, cells, _, _, _, _, _ = sglang_run
    (cell,) = cells
    (start,) = sim.scripts_of("start_engine")
    assert var(start, "WARM") == "0"
    assert var(start, "IMAGE") == SGLANG_IMAGE
    assert var(start, "SERVED_MODEL") == "qwen3-8b"
    ckpt = (cell.spec.hf.repo, cell.spec.hf.revision)
    assert pairs(start, "FETCH_WEIGHTS") == [ckpt] and pairs(start, "CACHED_WEIGHTS") == []
    cmd = array(start, "ENGINE_CMD")
    # The image's python3 (/opt/sglang/bin first on its PATH) runs the server, offline,
    # against the hub cache mounted where the image's HF_HOME default points.
    at = cmd.index(SGLANG_IMAGE)
    assert cmd[at - 2 : at] == ["--entrypoint", "python3"]
    assert cmd[at + 1 : at + 3] == ["-m", "sglang.launch_server"]
    assert cmd[at + 3 :] == cell.launch.args
    assert "HF_HUB_OFFLINE=1" in cmd and "/opt/dlami/nvme/loom-hf:/root/.cache/huggingface" in cmd
    assert "127.0.0.1:8000:8000" in cmd
    path = ctx.out_dir / str(outcome.experiment_id) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    (started,) = [e for e in events if e["kind"] == "engine_started"]
    assert started["cell"] == "sglang"
    assert started["system"]["image_digest"] == SGLANG_IMAGE
    assert 0 <= started["system"]["ttl_shutdown_lead_s"] <= 1


def test_load_jobs_are_sglang_chat_through_the_container_mounts(sglang_run):
    _, sim, _, cells, _, _, _, _, _ = sglang_run
    (cell,) = cells
    (fetches,) = sim.dataset_fetches.values()
    assert len(fetches) == 1
    loads = [j for j in sim.jobs if isinstance(j, LoadJob)]
    assert loads
    for job in loads:
        assert job.engine == "sglang"  # selects SGLang's Prometheus names
        assert job.served_model == "qwen3-8b"
        assert job.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert job.workload["path"].startswith("/data/")
        assert job.tokenizer is not None and job.tokenizer.local_dir == (
            f"/models/models--Qwen--Qwen3-8B/snapshots/{cell.spec.hf.revision}"
        )


def test_every_pass_is_scored_on_the_stored_reference_and_gated_against_it(sglang_run):
    outcome, sim, ctx, cells, exp, _, _, baseline, _ = sglang_run
    (cell,) = cells
    replicates = exp.quality.replicates  # type: ignore[union-attr]
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert len(evals) == replicates == 3
    assert [j.divergence for j in evals] == ["score", None, None]
    ref = evals[0].reference
    assert ref is not None and ref.config_hash == baseline["config_hash"]
    assert ref.self_divergence is not None  # the stored noise floor
    assert {j.served_model for j in evals} == {"qwen3-8b"}
    (event,) = [e for e in outcome.events if e["kind"] == "stored_baseline"]
    assert event["config_hash"] == baseline["config_hash"]
    (gate,) = outcome.gates
    label = f"{baseline['experiment'][:8]}/{baseline['config_hash'][:12]}"
    assert (gate.cell, gate.baseline) == ("sglang", label)
    with session_scope(ctx.db_url) as s:
        mine = set(
            s.scalars(
                select(BenchEvalRun.config_hash).where(
                    BenchEvalRun.experiment_id == outcome.experiment_id
                )
            )
        )
    assert mine == {cell.config_hash}  # vLLM is not served again


def test_one_leaderboard_ranks_both_engines_at_equal_load(sglang_run):
    from loom_bench.cli import _analyze_experiments

    outcome, _, ctx, _, exp, _, _, baseline, vllm_outcome = sglang_run
    assert exp.slo is not None
    with session_scope(ctx.db_url) as s:
        results = _analyze_experiments(
            s,
            [vllm_outcome.experiment_id, outcome.experiment_id],
            exp.slo,
            exp.cost_allocation.allocation(),
            ctx.prices,
        )
    report = build_leaderboard(results)
    (board,) = report.boards
    assert board.model == "Qwen/Qwen3-8B" and board.workload == "chat-sharegpt"
    by_name: dict[str, Any] = {row.result.name: row.result for row in board.rows}
    assert set(by_name) == {"bf16", "sglang"}
    assert by_name["bf16"].label.engine == "vllm" and by_name["sglang"].label.engine == "sglang"
    assert by_name["sglang"].quality is not None
    assert by_name["sglang"].quality.baseline_config_hash == baseline["config_hash"]
    assert board.equal_load is not None
    assert set(board.equal_load.names) == {"bf16", "sglang"}
    assert board.equal_load.points  # the load both engines ran


def test_bench_compare_pairs_the_engines_by_workload(sglang_run):
    from typer.testing import CliRunner

    from loom_bench.cli import EXIT_INVALID, app

    outcome, _, ctx, _, _, _, _, _, vllm_outcome = sglang_run
    args = [str(vllm_outcome.experiment_id), str(outcome.experiment_id), "--db", ctx.db_url]
    result = CliRunner().invoke(
        app, ["compare", *args, "--match-by", "workload", "--cell-a", "bf16"]
    )
    assert result.exit_code != EXIT_INVALID, result.output  # 0 or 6: the metrics decide
    out = " ".join(result.output.split())
    assert "load point" in out and "matched" in out
    assert "engine" in out and "sglang" in out  # listed under config differences
