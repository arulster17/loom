"""The Qwen3-8B config sweep's pod plan, end to end on the runpod provider: the shipped
sweep's model, provider and five variants (host groups, checkpoints by `Variant.model`,
KV-cache dtypes, the max_num_batched_tokens knob) with replicated passes and the in-run
gate against bf16, at test scale, against FakeRunpod and the pod simulator.

No earlier real run had more than one pod per image (`host_group`) or gated four
candidates, three of them on pods other than the reference's, on three replicated
passes. Its smoke (runpod-smoke-8b-sweep) runs this on real pods; this file runs it for
free first: three pods one after the other, each with its own cold start and checkpoint,
warm restarts onto the KV-FP8 cells, every candidate scored on the reference pod a
captured, and four gates against bf16."""

import json

import pytest
import yaml
from sqlalchemy import select

from loom_bench.experiment import EXPERIMENTS_DIR
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchEvalRun

from . import test_runpod_two_checkpoints as two
from .fakes import script_array, script_pairs, script_var

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("runpod-e2e")]

SWEEP = EXPERIMENTS_DIR / "qwen3-8b-config-sweep-runpod.yaml"
SMOKE = EXPERIMENTS_DIR / "runpod-smoke-8b-sweep.yaml"
REPLICATES = 3


@pytest.fixture(scope="module", params=[SWEEP, SMOKE], ids=lambda p: p.stem)
def sweep_run(request, tmp_path_factory):
    path = request.param
    return two.run_on_pod_sim(path, tmp_path_factory.mktemp(path.stem), replicates=True)


def _pods(sim) -> list[str]:
    """Pods in the order they first ran a script."""
    seen: list[str] = []
    for target, _, _ in sim.launched:
        pod = sim._pod(target)
        if pod not in seen:
            seen.append(pod)
    return seen


def test_the_shipped_sweep_keeps_three_replicated_passes():
    for path in (SWEEP, SMOKE):
        assert yaml.safe_load(path.read_text())["quality"]["replicates"] == REPLICATES


def test_three_pods_run_one_after_the_other(sweep_run):
    outcome, sim, _, cells, _ = sweep_run
    assert outcome.status.value == "completed", outcome.reason
    groups = {c.key: c.host_key.rsplit("/group=", 1)[1] for c in cells}
    assert groups == {
        "bf16": "a",
        "bf16-kv8": "a",
        "fp8": "b",
        "fp8-kv8": "b",
        "fp8-kv8-mbt1024": "c",
    }
    assert len(_pods(sim)) == 3
    # Each pod's last script runs before the next pod's first.
    order = [(sim._pod(target), name) for target, name, _ in sim.launched]
    first = {p: min(i for i, (q, _) in enumerate(order) if q == p) for p in _pods(sim)}
    last = {p: max(i for i, (q, _) in enumerate(order) if q == p) for p in _pods(sim)}
    a, b, c = _pods(sim)
    assert last[a] < first[b] and last[b] < first[c]


def test_each_pod_downloads_only_its_checkpoint_and_warm_restarts_offline(sweep_run):
    _, sim, _, cells, _ = sweep_run
    by_key = {c.key: c for c in cells}
    bf16 = (by_key["bf16"].spec.hf.repo, by_key["bf16"].spec.hf.revision)
    fp8 = (by_key["fp8"].spec.hf.repo, by_key["fp8"].spec.hf.revision)
    a, b, c = _pods(sim)
    assert sim.downloads == {a: [bf16], b: [fp8], c: [fp8]}
    starts = sim.scripts("start_engine")
    assert [script_var(s, "WARM") for s in starts] == ["0", "1", "0", "1", "0"]
    for warm in (starts[1], starts[3]):
        assert script_pairs(warm, "FETCH_WEIGHTS") == []
    assert [script_var(s, "SERVED_MODEL") for s in starts] == [
        "qwen3-8b",
        "qwen3-8b",
        "qwen3-8b-fp8",
        "qwen3-8b-fp8",
        "qwen3-8b-fp8",
    ]
    cmds = [script_array(s, "ENGINE_CMD") for s in starts]
    kv = [cmd[cmd.index("--kv-cache-dtype") + 1] for cmd in cmds]
    assert kv == ["auto", "fp8", "auto", "fp8", "fp8"]
    flag = "--max-num-batched-tokens"
    mbt = [cmd[cmd.index(flag) + 1] if flag in cmd else None for cmd in cmds]
    assert mbt == [None, None, None, None, "1024"]


def test_every_cell_runs_three_passes_and_only_the_first_scores_divergence(sweep_run):
    _, sim, _, cells, _ = sweep_run
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert len(evals) == REPLICATES * len(cells)
    bf16 = next(c for c in cells if c.key == "bf16")
    modes = [j.divergence for j in evals]
    per_cell = [modes[i : i + REPLICATES] for i in range(0, len(modes), REPLICATES)]
    assert per_cell[0] == ["capture_and_floor", None, None]
    for passes in per_cell[1:]:
        assert passes == ["score", None, None]
    # Candidates on pods b and c score against the reference captured on pod a.
    for job in evals:
        if job.divergence == "score":
            assert job.reference is not None
            assert job.reference.config_hash == bf16.config_hash
    assert all(isinstance(j, LoadJob) for j in sim.jobs if not isinstance(j, EvalJob))


def test_every_candidate_is_gated_against_bf16_on_pooled_passes(sweep_run):
    outcome, _, ctx, cells, _ = sweep_run
    candidates = {c.key for c in cells} - {"bf16"}
    assert {g.cell for g in outcome.gates} == candidates
    assert {g.baseline for g in outcome.gates} == {"bf16"}
    with session_scope(ctx.db_url) as s:
        evals = list(s.scalars(select(BenchEvalRun)))
    assert {e.config_hash for e in evals} == {c.config_hash for c in cells}


def test_each_cells_quality_event_pools_three_passes(sweep_run):
    outcome, _, ctx, cells, _ = sweep_run
    path = ctx.out_dir / str(outcome.experiment_id) / "events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    quality = {e["cell"]: e for e in events if e["kind"] == "quality"}
    assert set(quality) == {c.key for c in cells}
    assert {e["replicates"] for e in quality.values()} == {REPLICATES}
    assert not [e for e in events if e["kind"] in ("quality_failed", "divergence_failed")]
