"""vLLM vs SGLang on one AWS g6e.xlarge (qwen3-8b-sglang-aws-g6e) and its smoke
(aws-smoke-g6e-sglang), before either spends anything. SGLang has never run on the
aws_ec2 path, and no AWS host has switched engine image.

- The run serves 95cde129's vLLM bf16 cell again (the same variant and config hash) and
  then SGLang v0.5.21 BF16 (RunPod 565b8d3f's launch) on the same host, with 95cde129's
  provider block, chat workload, windows, search, seed, repetitions, SLO and quality
  subset and passes; SGLang is gated against the in-run vLLM BF16. At a vLLM-like knee
  the search visits vLLM's loads exactly, so the engines meet at equal load.
- The smoke mirrors it except scale.
- Both plan under their caps even to their TTLs.
- End to end on the aws_ec2 provider (moto EC2 and S3, HostSim behind SSM), after a
  test-scale 95cde129 is recorded: one on-demand host, vLLM's cold start, then a warm
  restart that switches image (the engine stopped, SGLang's image started with
  `python3 -m sglang.launch_server`, the checkpoint vLLM downloaded checked offline with
  SGLang's python3 and not fetched again), the chat dataset fetched once, load jobs tagged
  with each engine's metric names, SGLang's passes scored on vLLM's in-run reference
  (SGLang's completions logprob form) and gated against it, the host terminated, one
  leaderboard with both engines at equal load, and `bench compare` against the earlier
  run's bf16 by config hash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.money import parse_usd
from loom_bench.report.leaderboard import build_leaderboard
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchSpend

from .hostsim import array, pairs, var
from .test_aws_g6e import (
    ON_DEMAND_ACCRUAL,
    REGISTRY,
    RUNPOD_CHAT_POINTS,
    _branches,
    _plan,
    _trajectory,
    at_test_scale,
    run_on_host_sim,
)

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("aws-e2e")]

REAL = EXPERIMENTS_DIR / "qwen3-8b-sglang-aws-g6e.yaml"
SMOKE = EXPERIMENTS_DIR / "aws-smoke-g6e-sglang.yaml"
VLLM_REAL = EXPERIMENTS_DIR / "qwen3-8b-aws-g6e.yaml"
RUNPOD_ENGINES = EXPERIMENTS_DIR / "qwen3-8b-vllm-vs-sglang-runpod.yaml"
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
)
# 95cde129's bf16 cell (vLLM v0.30.0 BF16 on g6e.xlarge on-demand), as recorded.
VLLM_BF16_HASH = "48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375"


def _cells(path: Path) -> dict[str, Any]:
    return {c.key: c for c in expand(load_experiment(path), REGISTRY)}


# --- the two specs ----------------------------------------------------------------------


def test_the_run_is_95cde129s_bf16_cell_and_sglang_on_one_host():
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
    assert q.limit is None and q.baseline is None
    assert q.baseline_variant == vq.baseline_variant == "bf16"
    cells = _cells(REAL)
    assert list(cells) == ["bf16", "sglang"]  # the baseline first
    # The vLLM cell is 95cde129's: same variant, same config hash.
    assert sg.variants[0] == vl.variants[0]
    assert cells["bf16"].config_hash == _cells(VLLM_REAL)["bf16"].config_hash == VLLM_BF16_HASH
    assert len({c.host_key for c in cells.values()}) == 1  # one host, one quota slot


def test_the_sglang_cell_differs_from_vllm_bf16_only_by_engine():
    cells = _cells(REAL)
    sg, vl = cells["sglang"], cells["bf16"]
    assert sg.spec.engine.name == "sglang" and sg.spec.engine.version == "0.5.21"
    assert sg.launch.image == SGLANG_IMAGE != vl.launch.image
    assert sg.spec.model_dump(exclude={"engine"}) == vl.spec.model_dump(exclude={"engine"})
    assert sg.spec.engine.chat_template_kwargs == vl.spec.engine.chat_template_kwargs
    assert sg.spec.engine.chat_template_kwargs == {"enable_thinking": False}
    assert sg.hardware == vl.hardware
    assert (sg.launch.model_repo, sg.launch.model_revision) == (
        vl.launch.model_repo,
        vl.launch.model_revision,
    )
    # RunPod 565b8d3f's SGLang launch, unchanged: only the hardware differs.
    runpod = _cells(RUNPOD_ENGINES)["sglang"]
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
    s_cells, r_cells = expand(smoke, REGISTRY), expand(real, REGISTRY)
    assert [c.key for c in s_cells] == [c.key for c in r_cells]
    assert [c.config_hash for c in s_cells] == [c.config_hash for c in r_cells]
    assert [c.host_key for c in s_cells] == [c.host_key for c in r_cells]
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
    assert smoke.quality is not None and real.quality is not None
    assert smoke.quality.limit is not None and real.quality.limit is None
    assert smoke.quality.model_copy(update={"limit": None}) == real.quality


@pytest.mark.parametrize(
    ("path", "cap"), [(SMOKE, "$4.00"), (REAL, "$9.75")], ids=["smoke", "real"]
)
def test_both_plan_under_their_caps_even_to_ttl(path, cap):
    plan = _plan(load_experiment(path))
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd(cap)
    assert plan.ttl_worst_micros <= plan.caps.effective
    (host,) = plan.hosts
    assert host.market == "on_demand" and host.hourly_micros == ON_DEMAND_ACCRUAL
    assert host.seconds < host.ttl_s
    kinds = [s.kind for s in host.steps]
    assert kinds[:3] == ["cold_start", "eval_setup", "dataset"]
    assert kinds.count("warm_start") == 1  # the engine switch, no second cold start
    assert plan.n_cells == 2


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
    # 95cde129's bf16 search visited RunPod 7a8237d0's loads exactly; a knee in the same
    # bracket repeats them, so both engines (and 95cde129) meet at every load.
    (w,) = load_experiment(REAL).workloads
    points = [round(x, 3) for x, _ in _trajectory(w.load, 2.3)]
    assert points == RUNPOD_CHAT_POINTS["bf16"]


# --- end to end on the aws_ec2 provider ---------------------------------------------------


def vllm_bf16_at_test_scale(path: Path, tmp: Path, body: bytes) -> Experiment:
    """95cde129 at test scale, its bf16 cell alone: the earlier run to cross-check."""
    exp = at_test_scale(path, tmp, body)
    return Experiment.model_validate(
        {**exp.model_dump(mode="json"), "variants": [{"name": "bf16"}]}
    )


@pytest.fixture(scope="module")
def earlier_vllm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("vllm-95cde129")
    outcome, *_ = run_on_host_sim(VLLM_REAL, tmp, build=vllm_bf16_at_test_scale)
    assert outcome.status.value == "completed", outcome.reason
    return tmp / "loom.db", outcome


@pytest.fixture(scope="module", params=[SMOKE, REAL], ids=["smoke", "real"])
def engines_run(request, earlier_vllm, tmp_path_factory):
    db, vllm_outcome = earlier_vllm
    tmp = tmp_path_factory.mktemp(request.param.stem)
    return (*run_on_host_sim(request.param, tmp, db=db), vllm_outcome)


def test_it_completes_on_one_on_demand_host_that_is_terminated(engines_run):
    outcome, sim, ctx, _, _, instances, reaped, _ = engines_run
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


def test_vllm_cold_starts_then_a_warm_restart_switches_to_sglang(engines_run):
    outcome, sim, ctx, cells, _, _, _, _ = engines_run
    vllm, sglang = cells
    kinds = [k for _, k, _ in sim.launched if k != "run_job"]
    # The vLLM engine is stopped before SGLang's starts (the same container name).
    assert kinds == ["start_engine", "stop_engine", "start_engine"]
    cold, warm = sim.scripts_of("start_engine")
    assert [var(s, "WARM") for s in (cold, warm)] == ["0", "1"]
    assert var(cold, "IMAGE") == vllm.launch.image and var(warm, "IMAGE") == SGLANG_IMAGE
    ckpt = (vllm.spec.hf.repo, vllm.spec.hf.revision)
    assert ckpt == (sglang.spec.hf.repo, sglang.spec.hf.revision)
    # Downloaded once by vLLM's start; SGLang's start only checks it offline (with its own
    # image's python3, start_engine.sh) and loads it from the same cache.
    assert pairs(cold, "FETCH_WEIGHTS") == [ckpt] and pairs(cold, "CACHED_WEIGHTS") == []
    assert pairs(warm, "FETCH_WEIGHTS") == [] and pairs(warm, "CACHED_WEIGHTS") == [ckpt]
    (downloads,) = sim.downloads.values()
    assert downloads == [ckpt]
    cmd = array(warm, "ENGINE_CMD")
    at = cmd.index(SGLANG_IMAGE)
    assert cmd[at - 2 : at] == ["--entrypoint", "python3"]
    assert cmd[at + 1 : at + 3] == ["-m", "sglang.launch_server"]
    assert cmd[at + 3 :] == sglang.launch.args
    assert "HF_HUB_OFFLINE=1" in cmd and "/opt/dlami/nvme/loom-hf:/root/.cache/huggingface" in cmd
    assert "127.0.0.1:8000:8000" in cmd
    assert var(warm, "WEIGHTS_DIR") == var(cold, "WEIGHTS_DIR")
    path = ctx.out_dir / str(outcome.experiment_id) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    started = [e for e in events if e["kind"] == "engine_started"]
    assert [(e["cell"], e["warm"]) for e in started] == [("bf16", False), ("sglang", True)]
    assert started[1]["system"]["image_digest"] == SGLANG_IMAGE
    assert 0 <= started[0]["system"]["ttl_shutdown_lead_s"] <= 1


def test_load_jobs_carry_each_engines_metric_names_through_the_mounts(engines_run):
    _, sim, _, cells, _, _, _, _ = engines_run
    (fetches,) = sim.dataset_fetches.values()
    assert len(fetches) == 1  # the dataset outlives the engine switch
    loads = [j for j in sim.jobs if isinstance(j, LoadJob)]
    assert [j.engine for j in loads] == ["vllm", "sglang"]  # one point, one rep per cell
    rev = cells[0].spec.hf.revision
    for job in loads:
        assert job.served_model == "qwen3-8b"
        assert job.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
        assert job.workload["path"].startswith("/data/")
        assert job.tokenizer is not None
        assert job.tokenizer.local_dir == f"/models/models--Qwen--Qwen3-8B/snapshots/{rev}"


def test_sglang_is_scored_on_vllms_in_run_reference_and_gated_against_it(engines_run):
    outcome, sim, _, cells, exp, _, _, _ = engines_run
    vllm, _ = cells
    replicates = exp.quality.replicates  # type: ignore[union-attr]
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert len(evals) == 2 * replicates == 6
    assert [j.divergence for j in evals] == ["capture_and_floor", None, None, "score", None, None]
    ref = evals[replicates].reference
    assert ref is not None and ref.config_hash == vllm.config_hash
    assert not [e for e in outcome.events if e["kind"] == "stored_baseline"]
    (gate,) = outcome.gates
    assert (gate.cell, gate.baseline) == ("sglang", "bf16")


def test_one_leaderboard_ranks_both_engines_at_equal_load(engines_run):
    from loom_bench.cli import _analyze_experiments

    outcome, _, ctx, cells, exp, _, _, _ = engines_run
    assert exp.slo is not None
    with session_scope(ctx.db_url) as s:
        results = _analyze_experiments(
            s, [outcome.experiment_id], exp.slo, exp.cost_allocation.allocation(), ctx.prices
        )
    (board,) = build_leaderboard(results).boards
    assert board.model == "Qwen/Qwen3-8B" and board.workload == "chat-sharegpt"
    by_name: dict[str, Any] = {row.result.name: row.result for row in board.rows}
    assert set(by_name) == {"bf16", "sglang"}
    assert by_name["bf16"].label.engine == "vllm" and by_name["sglang"].label.engine == "sglang"
    assert by_name["sglang"].quality is not None
    assert by_name["sglang"].quality.baseline_config_hash == cells[0].config_hash
    assert board.equal_load is not None
    assert set(board.equal_load.names) == {"bf16", "sglang"}
    assert board.equal_load.points


def test_bench_compare_cross_checks_the_earlier_runs_bf16_by_config_hash(engines_run):
    from loom_bench.cli import EXIT_INVALID, app

    outcome, _, ctx, _, _, _, _, earlier = engines_run
    args = [str(earlier.experiment_id), str(outcome.experiment_id), "--db", ctx.db_url]
    result = CliRunner().invoke(app, ["compare", *args])
    assert result.exit_code != EXIT_INVALID, result.output  # 0 or 6: the metrics decide
    out = " ".join(result.output.split())
    assert "matched" in out and "sglang" in out  # sglang listed as only in the new run
    # Both engines of the new run against each other, by workload.
    same = str(outcome.experiment_id)
    cells = ["--cell-a", "bf16", "--cell-b", "sglang"]
    pair = CliRunner().invoke(
        app, ["compare", same, same, "--match-by", "workload", *cells, "--db", ctx.db_url]
    )
    assert pair.exit_code != EXIT_INVALID, pair.output
    assert "vllm → sglang" in " ".join(pair.output.split())  # a config difference
