"""A failed divergence keeps the eval's task scores: only the divergence part is lost, the
gate says why its divergence check is inconclusive (blocked) while still reporting the
per-task deltas, and the experiment ends `completed` with a reason and exit 8.

In 565b8d3f SGLang's divergence raised inside its eval job and took the job's task scores
down with it."""

from collections import Counter
from pathlib import Path

import pytest

from loom_bench.jobs import EvalJob, EvalJobResult
from loom_bench.providers.base import Host
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import (
    EXIT_OK,
    EXIT_RUNS_FAILED,
    RUN_COMPLETED,
    gate_stored,
    judge_runs,
    run_experiment,
)
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchEvalRun, BenchGateDecision, ExperimentStatus

from .conftest import mock_experiment

SUITE = Path("bench/experiments/suites/mock-smoke.yaml")
SWEEP_ERROR = (
    "ValueError: server returned no text_offset and its echoed tokens do not spell the scored text"
)


class DivergenceFails(MockProvider):
    """Mock provider whose eval jobs on the listed call numbers (0-based) lose their
    divergence half: the task scores come back, the divergence does not."""

    def __init__(self, fail_calls: set[int]) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.fail_calls = fail_calls
        self.calls = 0
        self.jobs: list[EvalJob] = []

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        n, self.calls = self.calls, self.calls + 1
        self.jobs.append(job)
        out = await super().run_eval(host, job)
        if n in self.fail_calls:
            lost = {"reference": None, "divergence": None, "divergence_error": SWEEP_ERROR}
            return out.model_copy(update=lost)
        return out


@pytest.fixture
def suite(tmp_path: Path) -> str:
    doc = SUITE.read_text().replace("suite: mock-smoke", "suite: mock-divergence")
    doc += "divergence:\n  prompts: 4\n  top_k: 5\n  max_new_tokens: 8\n  floor_concurrency: 1\n"
    path = tmp_path / "mock-divergence.yaml"
    path.write_text(doc)
    return str(path)


def _two_engines(suite: str):
    """Baseline and candidate on one mock host; the baseline is evaluated first."""
    return mock_experiment(
        variants=[{"name": "base"}, {"name": "cand", "mock": {"max_num_seqs": 8}}],
        quality={"suite": suite, "baseline_variant": "base"},
    )


def _eval_rows(db: str, experiment_id) -> Counter[str]:
    with session_scope(db) as s:
        rows = s.query(BenchEvalRun).filter(BenchEvalRun.experiment_id == experiment_id).all()
        return Counter(r.config_hash for r in rows)


def _gate(db: str, experiment_id) -> BenchGateDecision:
    with session_scope(db) as s:
        gate = (
            s.query(BenchGateDecision)
            .filter(BenchGateDecision.experiment_id == experiment_id)
            .one()
        )
        s.expunge(gate)
        return gate


def test_judge_runs_counts_failed_divergences():
    done = Counter({RUN_COMPLETED: 2})
    assert judge_runs(done, done, 1) == (
        ExperimentStatus.COMPLETED,
        "divergence failed in 1 of 2 quality evals",
        EXIT_RUNS_FAILED,
    )
    assert judge_runs(done, done, 0) == (ExperimentStatus.COMPLETED, None, EXIT_OK)


async def test_a_failed_candidate_divergence_keeps_its_task_scores(ctx, suite):
    ctx.provider = DivergenceFails(fail_calls={1})
    outcome = await run_experiment(_two_engines(suite), ctx)

    assert (outcome.status, outcome.exit_code) == (ExperimentStatus.COMPLETED, EXIT_RUNS_FAILED)
    assert outcome.reason == "divergence failed in 1 of 2 quality evals"
    assert sorted(_eval_rows(ctx.db_url, outcome.experiment_id).values()) == [2, 2]
    gate = _gate(ctx.db_url, outcome.experiment_id)
    assert gate.decision == "inconclusive" and gate.details["blocked"] is True
    div = gate.details["divergence"]
    assert div["verdict"] == "inconclusive"
    assert div["reason"] == f"not measured: candidate divergence failed: {SWEEP_ERROR}"
    assert {t["task"] for t in gate.details["tasks"]} == {"arithmetic", "json_schema"}
    failed = [e for e in outcome.events if e["kind"] == "divergence_failed"]
    assert len(failed) == 1 and failed[0]["mode"] == "score"
    assert not [e for e in outcome.events if e["kind"] == "quality_failed"]


async def test_a_failed_baseline_capture_leaves_the_candidate_unscored(ctx, suite):
    provider = DivergenceFails(fail_calls={0})
    ctx.provider = provider
    outcome = await run_experiment(_two_engines(suite), ctx)

    assert outcome.reason == "divergence failed in 1 of 2 quality evals"
    assert provider.jobs[0].divergence == "capture_and_floor"
    assert provider.jobs[1].divergence is None and provider.jobs[1].reference is None
    gate = _gate(ctx.db_url, outcome.experiment_id)
    assert gate.decision == "inconclusive"
    assert gate.details["divergence"]["reason"] == (
        f"not measured: baseline divergence capture failed: {SWEEP_ERROR}"
    )
    assert {t["task"] for t in gate.details["tasks"]} == {"arithmetic", "json_schema"}


async def test_a_stored_gate_keeps_the_divergence_failure(ctx, suite):
    ctx.provider = DivergenceFails(fail_calls={1})
    outcome = await run_experiment(_two_engines(suite), ctx)
    gate = _gate(ctx.db_url, outcome.experiment_id)
    decision, _, _ = gate_stored(
        ctx.db_url, gate.baseline_config_hash, gate.candidate_config_hash, suite=suite
    )
    assert decision.divergence.verdict.value == "inconclusive"
    assert decision.divergence.reason.startswith("not measured: candidate divergence failed")
    assert decision.blocked and len(decision.tasks) == 2


async def test_every_divergence_measured_is_unchanged(ctx, suite):
    ctx.provider = DivergenceFails(fail_calls=set())
    outcome = await run_experiment(_two_engines(suite), ctx)
    assert (outcome.reason, outcome.exit_code) == (None, EXIT_OK)
    gate = _gate(ctx.db_url, outcome.experiment_id)
    assert gate.details["divergence"]["verdict"] != "inconclusive"
    assert "KL" in gate.details["divergence"]["reason"]
