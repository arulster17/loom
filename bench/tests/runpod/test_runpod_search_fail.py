"""The forced-failure smoke (runpod-smoke-8b-search-fail): the rate search's failure
branches, which the sweep smoke (8bc65cfa) never reached on a pod because every point it
tried met the SLO. Descent after a failing first point, bisection between the highest pass
and the lowest failure, and the overloaded drain (the client waits drain_timeout_s for the
requests in flight after the window, cancels the stragglers, and the engine has dropped
them by the next run).

Before it runs on a pod: the spec is the sweep's pod a with only the search start and
depth and the window lengths changed; every knee between its floor and its start sends the
search through all three branches; and the spec at test scale reaches them end to end on
the runpod provider (FakeRunpod, moto S3, the pod simulator), against a mock engine that
cannot keep up with the spec's first two points."""

import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import select

from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.jobs import LoadJob
from loom_bench.registry import load_registry
from loom_bench.runner import next_search_load
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchRun

from . import test_runpod_two_checkpoints as two
from .fakes import script_var
from .test_runpod_e2e import BUCKET, s3_key

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("runpod-e2e")]

FAIL = EXPERIMENTS_DIR / "runpod-smoke-8b-search-fail.yaml"
SWEEP = EXPERIMENTS_DIR / "qwen3-8b-config-sweep-runpod.yaml"
REGISTRY = load_registry()

# 565b8d3f, vLLM BF16 on a RunPod 1x L40S, fixed-1k-1k: at 2.0 req/s offered it served
# 1.34-1.42 req/s and aborted 83-101 requests per run at the drain timeout; 1.0 req/s
# passed (TPOT p95 43-45 ms).
BF16_1K_SATURATION_REQ_S = 1.42
BF16_1K_PASSED_REQ_S = 1.0


def _fail_and_sweep() -> tuple[Experiment, Experiment]:
    return load_experiment(FAIL), load_experiment(SWEEP)


def _trajectory(load, knee: float) -> list[tuple[float, bool]]:
    """The search's points in order, against an engine that meets the SLO up to `knee`."""
    history: list[tuple[float, bool]] = []
    while (nxt := next_search_load(load, history)) is not None:
        history.append((nxt, nxt <= knee))
    return history


def branches(history: list[tuple[float, bool]], lo: float, step: float) -> dict[str, bool]:
    """Which failure branches a search ran, from its (load, met) points in order. Used on
    the offline runs here and on the paid run's database rows."""
    descended = (
        len(history) >= 2
        and history[0] == (lo, False)
        and math.isclose(history[1][0], lo / step, rel_tol=1e-9)
    )
    bisected = False
    for i, (load, _) in enumerate(history):
        seen = history[:i]
        passes = [x for x, met in seen if met]
        fails = [x for x, met in seen if not met]
        if not passes or not fails:
            continue
        fail = min(fails)
        below = [x for x in passes if x < fail]
        if below and math.isclose(load, math.sqrt(max(below) * fail), rel_tol=1e-9):
            bisected = True
    return {"descent": descended, "bisection": bisected, "passed": any(m for _, m in history)}


# --- the spec ----------------------------------------------------------------------------


def test_it_is_the_sweeps_pod_a_with_only_the_search_start_and_windows_changed():
    smoke, sweep = _fail_and_sweep()
    assert smoke.smoke and not sweep.smoke
    assert smoke.model == sweep.model and smoke.provider == sweep.provider
    pod_a = [v for v in sweep.variants if v.host_group == "a"]
    assert smoke.variants == pod_a
    cells = expand(smoke, REGISTRY)
    sweep_cells = {c.key: c for c in expand(sweep, REGISTRY)}
    for c in cells:  # same configs, launches and pod
        ref = sweep_cells[c.key]
        assert (c.config_hash, c.launch, c.host_key) == (ref.config_hash, ref.launch, ref.host_key)
    assert {c.spec.kv_cache_dtype for c in cells} == {"auto", "fp8"}
    assert smoke.repetitions == sweep.repetitions and smoke.slo == sweep.slo
    assert smoke.quality is None  # the sweep smoke ran the evals and gates on these cells

    (w,) = smoke.workloads
    ref = next(x for x in sweep.workloads if x.profile == w.profile == "fixed-1k-1k")
    assert w.overrides == ref.overrides  # full-length requests
    for k in ("mode", "drain_timeout_s", "request_timeout_s", "max_inflight", "arrival"):
        assert getattr(w.load, k) == getattr(ref.load, k), k
    s, r = w.load.search, ref.load.search
    assert s is not None and r is not None
    for k in ("scale", "step", "rel_tol", "hi"):
        assert getattr(s, k) == getattr(r, k), k
    # The sweep's deepest search (chat's), so the descent spans overload to a pass.
    deepest = max((x.load.search for x in sweep.workloads), key=lambda x: x.descend)  # type: ignore[union-attr]
    assert (s.descend, s.max_points) == (deepest.descend, deepest.max_points)
    # Short windows; the warmup still covers the ramp.
    assert w.load.duration_s is not None and ref.load.duration_s is not None
    assert w.load.duration_s < ref.load.duration_s / 4
    assert w.load.warmup_s and w.load.warmup_s < w.load.duration_s / 2


def test_it_starts_in_certain_overload_and_descends_to_a_load_bf16_passed():
    smoke, _ = _fail_and_sweep()
    (w,) = smoke.workloads
    s = w.load.search
    assert s is not None
    assert s.lo >= 2 * BF16_1K_SATURATION_REQ_S
    assert s.lo / s.step**s.descend <= BF16_1K_PASSED_REQ_S * (1 + 1e-9)


@pytest.mark.parametrize("knee", [1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 2.25, 2.6, 3.0, 3.3])
def test_any_knee_between_its_floor_and_its_start_runs_every_branch(knee):
    smoke, _ = _fail_and_sweep()
    (w,) = smoke.workloads
    s = w.load.search
    assert s is not None
    history = _trajectory(w.load, knee)
    assert len(history) <= s.max_points
    assert branches(history, s.lo, s.step) == {"descent": True, "bisection": True, "passed": True}


def test_a_knee_above_its_start_still_bisects_and_one_below_its_floor_only_descends():
    # bf16-kv8 could pass 3.375 (then it climbs and bisects); a bf16 engine failing even
    # 1.0 req/s ends the search at the floor with no goodput, as the sweep would.
    smoke, _ = _fail_and_sweep()
    (w,) = smoke.workloads
    s = w.load.search
    assert s is not None
    above = branches(_trajectory(w.load, 4.0), s.lo, s.step)
    assert above == {"descent": False, "bisection": True, "passed": True}
    below = _trajectory(w.load, 0.9)
    assert [x for x, _ in below] == pytest.approx([3.375, 2.25, 1.5, 1.0])
    assert branches(below, s.lo, s.step) == {"descent": True, "bisection": False, "passed": False}


# --- end to end on the pod simulator -----------------------------------------------------

# The pod's mock engine keeps up with any offered load up to KNEE req/s and not above it:
# before each load job the pod simulator sets its step cost from the job's rate. Below the
# knee a step takes ~11 ms (TPOT well under the SLO's 50 ms, a request lives ~1 s, inside
# the drain timeout); above it 100 ms, so TPOT fails and a request needs ~8 s, so the ones
# in flight at the window's end outlive the drain and are cancelled. A hard knee keeps the
# verdicts deterministic at a few requests per window (a batch-dependent step cost gives a
# soft knee, and 3 repetitions of ~4 requests then fail the CI upper bound at random).
# 1.8 sits between the spec's floor and start: 3.375 and 2.25 fail, 1.5 passes, then it
# bisects.
KNEE = 1.8
FAST = {"step_base_ms": 10.0, "decode_ms_per_seq": 0.5, "prefill_ms_per_token": 0.02}
SLOW = {"step_base_ms": 100.0, "decode_ms_per_seq": 0.5, "prefill_ms_per_token": 0.02}
OUTPUT_LEN = 80
WINDOW = {"duration_s": 3.0, "warmup_s": 0.5, "drain_timeout_s": 1.5, "scrape_interval_s": 0.2}
EVAL_MARK = "BENCH_CMD=(quality job)"


class KneePodSim(two.RecordingPodSim):
    """Pods whose engine has a hard capacity knee (above), and which record how many
    sequences the engine still held when each load job started."""

    def __init__(self, s3: Any) -> None:
        super().__init__(s3, {"time_scale": 1.0, **FAST})
        self.held_at_start: list[tuple[float, int]] = []  # (job rate, sequences held)

    async def job(self, target: Any, script: str) -> Any:
        if EVAL_MARK not in script:
            raw = self.s3.get_object(Bucket=BUCKET, Key=s3_key(script_var(script, "JOB_URL")))
            job = LoadJob.model_validate_json(raw["Body"].read())
            sim = self.servers[self._pod(target)].server.config.app.state.mock.engine.sim
            self.held_at_start.append((job.load_value, len(sim._running) + len(sim._waiting)))
            cost = SLOW if job.load_value > KNEE else FAST
            sim.config = sim.config.model_copy(update=cost)
        return await super().job(target, script)


def at_test_scale(path: Path, tmp: Path) -> Experiment:
    """The spec's model, provider, variants, search, repetitions and SLO; short requests
    and windows so a point takes seconds."""
    doc = yaml.safe_load(path.read_text())
    for w in doc["workloads"]:
        w["overrides"] = {"input_len": 32, "output_len": OUTPUT_LEN}
        w["load"].update(WINDOW)
    doc["budget"] = {"max_spend": "$2", "ttl_minutes": 100, "accrual_interval_s": 0.2}
    return Experiment.model_validate(doc)


@pytest.fixture(scope="module")
def fail_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp(FAIL.stem)
    return two.run_on_pod_sim(FAIL, tmp, scale=at_test_scale, pod_sim=KneePodSim)


def _runs(ctx, cell: str) -> list[BenchRun]:
    with session_scope(ctx.db_url) as s:
        rows = s.scalars(select(BenchRun).where(BenchRun.cell_key == cell))
        return sorted(rows, key=lambda r: r.started_at)


def _points(runs: list[BenchRun]) -> list[float]:
    out: list[float] = []
    for r in runs:
        if not out or out[-1] != r.load_value:
            out.append(r.load_value)
    return out


def _goodput(ctx, outcome, cell: str) -> dict:
    path = ctx.out_dir / str(outcome.experiment_id) / "goodput.json"
    (row,) = [g for g in json.loads(path.read_text()) if g["cell"] == cell]
    return row


def test_the_run_completes_with_every_run_recorded(fail_run):
    outcome, _, ctx, cells, _ = fail_run
    assert outcome.status.value == "completed", outcome.reason
    assert [c.key for c in cells] == ["bf16", "bf16-kv8"]
    for c in cells:
        assert {r.status for r in _runs(ctx, c.key)} == {"completed"}


def test_each_cell_descends_from_its_failing_start_then_bisects(fail_run):
    outcome, _, ctx, cells, _ = fail_run
    load = load_experiment(FAIL).workloads[0].load
    s = load.search
    assert s is not None
    for c in cells:
        order = _points(_runs(ctx, c.key))
        met = {x: m for x, m in _goodput(ctx, outcome, c.key)["points"]}
        history = [(x, met[x]) for x in order]
        assert history == _trajectory(load, KNEE)
        assert branches(history, s.lo, s.step) == {
            "descent": True,
            "bisection": True,
            "passed": True,
        }, history


def test_an_overloaded_point_drains_then_cancels_its_stragglers(fail_run):
    _, _, ctx, cells, _ = fail_run
    for c in cells:
        runs = _runs(ctx, c.key)
        assert {r.load_value for r in runs if r.summary["counts"]["aborted"]} == {
            r.load_value for r in runs if r.load_value > KNEE
        }
        for r in runs:
            assert r.summary["client"]["unsent"] == 0  # every arrival was sent
            spent = (r.finished_at - r.started_at).total_seconds()
            # The client waits out the drain at an overload, not the requests' ~8 s.
            assert spent < WINDOW["duration_s"] + WINDOW["drain_timeout_s"] + 3, spent


def test_every_run_starts_on_an_engine_holding_nothing(fail_run):
    # The engine drops the cancelled requests (vLLM aborts a request whose client
    # disconnected), so a run after an overloaded one is not served behind its backlog.
    _, sim, _, cells, _ = fail_run
    assert len(sim.held_at_start) == 5 * 3 * len(cells)
    assert sum(1 for rate, _ in sim.held_at_start if rate > KNEE) == 3 * 3 * len(cells)
    assert {held for _, held in sim.held_at_start} == {0}
