"""A candidate gated against a baseline evaluated in an earlier experiment
(`quality.baseline`), as the 70B FP8 run is gated against BF16 run cf4d1614: the stored
samples, divergence reference and noise floor stand in for a baseline cell, checked at
plan time, and tasks only one side ran are reported, not gated."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from loom_bench.budget import load_budget
from loom_bench.jobs import EvalJob
from loom_bench.prices import load_prices
from loom_bench.providers.mock import MockProvider
from loom_bench.quality.gate import GateDecision
from loom_bench.quality.runner import gate_against_baseline
from loom_bench.quality.suite import load_suite
from loom_bench.registry import load_registry
from loom_bench.runner import (
    RunnerContext,
    gate_stored,
    load_stored_baseline,
    plan_experiment,
    read_samples,
    run_experiment,
)
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import BenchEvalRun, BenchGateDecision

from .conftest import mock_experiment, write_yaml

pytestmark = [pytest.mark.timeout(240), pytest.mark.xdist_group("stored-baseline")]

SUITE = {
    "suite": "stored-baseline",
    "model": "qwen3-8b",
    "seed": 1234,
    "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 1000},
    "tasks": [
        {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
        {"name": "json_schema", "kind": "json_schema"},
        # Added after the baseline ran (as tool_calling_strict is for the 70B FP8 run).
        {"name": "arithmetic_new", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 8}},
    ],
    "divergence": {
        "prompts": 12,
        "top_k": 5,
        "max_new_tokens": 16,
        "noise_multiple": 2.0,
        "ceiling_top1": 0.6,
    },
    "subsets": {
        "then": ["arithmetic", "json_schema"],
        "now": ["arithmetic", "json_schema", "arithmetic_new"],
        "new_only": ["arithmetic_new"],
    },
}
JITTER = {"logprob_jitter": 0.25}


class RecordingProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.eval_jobs: list[EvalJob] = []

    async def run_eval(self, host, job):
        self.eval_jobs.append(EvalJob.model_validate_json(job.model_dump_json()))
        return await super().run_eval(host, job)


def _ctx(tmp: Path) -> RunnerContext:
    db = f"sqlite:///{tmp / 'loom.db'}"
    upgrade(db)
    return RunnerContext(
        db_url=db,
        out_dir=tmp / "results",
        registry=load_registry(),
        prices=load_prices(),
        budget=load_budget(),
    )


def _baseline_ref(ctx: RunnerContext, experiment_id) -> dict[str, str]:
    with session_scope(ctx.db_url) as s:
        (config,) = set(
            s.scalars(
                select(BenchEvalRun.config_hash).where(BenchEvalRun.experiment_id == experiment_id)
            )
        )
    return {"experiment": str(experiment_id), "config_hash": config}


def _candidate(suite: Path, baseline: dict[str, str], subset: str = "now", **over):
    doc = {
        "name": "fp8-like",
        "variants": [
            {"name": "noisy-kernels", "mock": {**JITTER, "logprob_noise": 0.3}},
            {"name": "broken", "mock": {**JITTER, "degrade": 0.3, "logprob_noise": 3.0}},
        ],
        "workloads": [],
        "quality": {"suite": str(suite), "subset": subset, "baseline": baseline},
        **over,
    }
    return mock_experiment(**doc)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    """A BF16-like baseline experiment, run once: quality only, the older task list."""
    tmp = tmp_path_factory.mktemp("stored-baseline")
    ctx = _ctx(tmp)
    suite = write_yaml(tmp / "suite.yaml", SUITE)
    base = mock_experiment(
        name="bf16-like",
        variants=[{"name": "base", "mock": JITTER}],
        workloads=[],
        quality={"suite": str(suite), "subset": "then", "baseline_variant": "base"},
    )
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    outcome = asyncio.run(run_experiment(base, ctx))
    assert outcome.status.value == "completed", outcome.reason
    return ctx, suite, _baseline_ref(ctx, outcome.experiment_id)


@pytest.fixture(scope="module")
def gated(stored):
    """The candidate experiment, gated against the stored baseline."""
    ctx, suite, ref = stored
    ctx.provider = RecordingProvider()
    outcome = asyncio.run(run_experiment(_candidate(suite, ref), ctx))
    return outcome, ctx.provider, ref


def test_quality_takes_exactly_one_baseline():
    ref = {"experiment": "cf4d1614-2453-4ad4-b345-afd6b769a52d", "config_hash": "a" * 64}
    with pytest.raises(ValidationError, match="exactly one of baseline_variant or baseline"):
        mock_experiment(quality={"suite": "x", "baseline_variant": "a", "baseline": ref})
    with pytest.raises(ValidationError, match="exactly one of baseline_variant or baseline"):
        mock_experiment(quality={"suite": "x"})
    with pytest.raises(ValidationError, match="config_hash"):
        mock_experiment(quality={"suite": "x", "baseline": {**ref, "config_hash": "eff0371d"}})
    with pytest.raises(ValidationError, match="smoke"):
        mock_experiment(smoke=True, quality={"suite": "x", "baseline": ref, "limit": 2})
    exp = mock_experiment(quality={"suite": "x", "baseline": ref})
    assert exp.quality is not None and exp.quality.baseline_variant is None


def test_every_cell_is_scored_on_the_stored_reference(gated):
    outcome, provider, ref = gated
    assert outcome.status.value == "completed", outcome.reason
    assert [j.divergence for j in provider.eval_jobs] == ["score", "score"]
    for job in provider.eval_jobs:
        assert job.reference is not None
        assert job.reference.config_hash == ref["config_hash"]
        assert job.reference.self_divergence is not None  # the baseline's noise floor
        assert job.tasks == ["arithmetic", "json_schema", "arithmetic_new"]
    (event,) = [e for e in outcome.events if e["kind"] == "stored_baseline"]
    assert event["config_hash"] == ref["config_hash"] and event["reference"]
    assert event["tasks"] == ["arithmetic", "json_schema"]


def test_the_gate_pairs_the_shared_tasks_and_lists_the_new_one(gated, stored):
    outcome, _, ref = gated
    label = f"{ref['experiment'][:8]}/{ref['config_hash'][:12]}"
    gates = {g.cell: g for g in outcome.gates}
    assert {g.baseline for g in gates.values()} == {label}
    assert gates["noisy-kernels"].decision == "pass" and not gates["noisy-kernels"].blocked
    assert gates["broken"].decision == "fail" and gates["broken"].blocked
    with session_scope(stored[0].db_url) as s:
        rows = list(
            s.scalars(
                select(BenchGateDecision).where(
                    BenchGateDecision.experiment_id == outcome.experiment_id
                )
            )
        )
    assert len(rows) == 2
    for row in rows:
        assert row.baseline_config_hash == ref["config_hash"]
        details = row.details
        assert [t["task"] for t in details["tasks"]] == ["arithmetic", "json_schema"]
        assert details["ungated"] == {"arithmetic_new": "the baseline did not run it"}
        # Divergence measured against the stored reference and calibrated on its floor.
        assert details["divergence"]["result"] is not None
        assert details["divergence"]["limits"]["calibrated"]
    ok = GateDecision.model_validate(next(r for r in rows if r.decision == "pass").details)
    assert "arithmetic_new: not gated: the baseline did not run it" in ok.reasons


def test_bench_quality_gate_re_decides_across_experiments(gated, stored):
    outcome, _, ref = gated
    ctx = stored[0]
    with session_scope(ctx.db_url) as s:
        cand = s.scalars(
            select(BenchGateDecision.candidate_config_hash).where(
                BenchGateDecision.experiment_id == outcome.experiment_id,
                BenchGateDecision.decision == "pass",
            )
        ).one()
    decision, base_hash, cand_hash = gate_stored(ctx.db_url, ref["config_hash"], cand)
    assert (base_hash, cand_hash) == (ref["config_hash"], cand)
    assert decision.decision.value == "pass"
    assert decision.ungated == {"arithmetic_new": "the baseline did not run it"}
    assert decision.divergence.result is not None  # scored on that baseline's reference


def test_the_plan_names_the_stored_baseline_and_the_ungated_tasks(stored):
    ctx, suite, ref = stored
    _, plan = plan_experiment(_candidate(suite, ref), ctx)
    assert plan.ok, plan.refusals
    (note,) = [n for n in plan.notes if "stored baseline" in n]
    assert "on arithmetic, json_schema" in note and "not gated" in note
    assert "arithmetic_new" in note
    # No baseline cell: no noise-floor pass is planned on the candidates.
    assert plan.n_cells == 2


def _refusal(ctx, exp) -> str:
    _, plan = plan_experiment(exp, ctx)
    assert not plan.ok
    (refusal,) = [r for r in plan.refusals if r.startswith("quality.baseline:")]
    return refusal


def test_a_missing_baseline_is_refused_at_plan_time(stored):
    ctx, suite, ref = stored
    missing = {**ref, "experiment": "00000000-0000-0000-0000-000000000000"}
    assert "has no eval runs" in _refusal(ctx, _candidate(suite, missing))
    other = {**ref, "config_hash": "0" * 64}
    assert "has no eval runs" in _refusal(ctx, _candidate(suite, other))


def test_a_baseline_sharing_no_task_is_refused(stored):
    ctx, suite, ref = stored
    assert "none of the tasks" in _refusal(ctx, _candidate(suite, ref, subset="new_only"))


def test_a_baseline_of_another_suite_is_refused(stored, tmp_path):
    ctx, _, ref = stored
    other = write_yaml(tmp_path / "other.yaml", {**SUITE, "suite": "another-suite"})
    assert "ran suite stored-baseline" in _refusal(ctx, _candidate(other, ref))


@pytest.fixture
def stored_files(stored, tmp_path):
    """The stored baseline's samples directory, restored after the test edits it."""
    ctx, _, ref = stored
    with session_scope(ctx.db_url) as s:
        uri = s.scalars(
            select(BenchEvalRun.samples_uri).where(BenchEvalRun.config_hash == ref["config_hash"])
        ).first()
    folder = Path(uri).parent
    backup = tmp_path / "backup"
    shutil.copytree(folder, backup)
    yield folder
    shutil.rmtree(folder)
    shutil.copytree(backup, folder)


def test_a_baseline_without_its_reference_is_refused(stored, stored_files):
    ctx, suite, ref = stored
    (stored_files / "reference.json").unlink()
    assert "no divergence reference" in _refusal(ctx, _candidate(suite, ref))


def test_a_reference_of_another_config_is_refused(stored, stored_files):
    ctx, suite, ref = stored
    path = stored_files / "reference.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps({**doc, "config_hash": "f" * 64}))
    assert "was captured on config" in _refusal(ctx, _candidate(suite, ref))


def test_a_changed_task_version_or_item_count_is_refused(stored, stored_files):
    ctx, suite, ref = stored
    path = stored_files / "samples.json"
    doc = json.loads(path.read_text())
    doc["tasks"]["json_schema"]["version"] = "0+data.2"
    path.write_text(json.dumps(doc))
    assert "rerun the baseline" in _refusal(ctx, _candidate(suite, ref))
    doc["tasks"]["json_schema"]["version"] = "1+data.2"
    doc["tasks"]["arithmetic"]["items"] = doc["tasks"]["arithmetic"]["items"][:-1]
    path.write_text(json.dumps(doc))
    assert "would not pair" in _refusal(ctx, _candidate(suite, ref))


def test_a_failed_baseline_capture_is_refused(stored, stored_files):
    ctx, suite, ref = stored
    path = stored_files / "samples.json"
    doc = json.loads(path.read_text())
    path.write_text(json.dumps({**doc, "divergence_error": "RuntimeError: boom"}))
    assert "divergence capture failed" in _refusal(ctx, _candidate(suite, ref))


def test_load_stored_baseline_returns_the_stored_side(stored):
    ctx, suite, ref = stored
    exp = _candidate(suite, ref)
    assert exp.quality is not None and exp.quality.baseline is not None
    base = load_stored_baseline(ctx.db_url, exp.quality.baseline, exp.quality.load(), "now")
    assert base.stored and base.config_hash == ref["config_hash"]
    assert set(base.result.tasks) == {"arithmetic", "json_schema"}
    assert base.reference is not None and base.reference.self_divergence is not None


def test_only_shared_tasks_is_opt_in(stored):
    ctx, _, ref = stored
    with session_scope(ctx.db_url) as s:
        uri = s.scalars(
            select(BenchEvalRun.samples_uri).where(BenchEvalRun.config_hash == ref["config_hash"])
        ).first()
    _, base = read_samples(uri)
    policy = load_suite(stored[1])
    fewer = type(base)(**{**_fields(base), "tasks": {"arithmetic": base.tasks["arithmetic"]}})
    with pytest.raises(ValueError, match="task sets differ"):
        gate_against_baseline(base, fewer, policy)
    decision = gate_against_baseline(base, fewer, policy, only_shared_tasks=True)
    assert [t.task for t in decision.tasks] == ["arithmetic"]
    assert decision.ungated == {"json_schema": "the candidate did not run it"}
    nothing = type(base)(**{**_fields(base), "tasks": {}})
    with pytest.raises(ValueError, match="no task in common"):
        gate_against_baseline(base, nothing, policy, only_shared_tasks=True)


def _fields(result) -> dict:
    return {name: getattr(result, name) for name in result.__slots__}
