"""A failed quality eval is recorded and the experiment goes on: the other engines still
run their load sweeps, a gate missing either side is inconclusive and blocked, and the
experiment ends `completed` with a reason and exit 8. Budget outcomes and a lost host
still end the experiment."""

from collections import Counter
from datetime import UTC, datetime

import pytest

from loom_bench.budget import BudgetAbort
from loom_bench.jobs import EvalJob, EvalJobResult
from loom_bench.providers.base import Host, HostLost
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import (
    EXIT_BUDGET_ABORT,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_RUNS_FAILED,
    RUN_COMPLETED,
    RUN_FAILED,
    judge_runs,
    run_experiment,
)
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment, BenchGateDecision, BenchRun, ExperimentStatus

from .conftest import mock_experiment

SUITE = "bench/experiments/suites/mock-smoke.yaml"


class FlakyEval(MockProvider):
    """Mock provider whose run_eval raises `error` on the listed call numbers (0-based)."""

    def __init__(self, fail_calls: set[int], error: Exception | None = None) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.fail_calls = fail_calls
        self.error = error or RuntimeError("eval exited 1: No module named 'langdetect'")
        self.calls = 0
        self.jobs: list[EvalJob] = []

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        n, self.calls = self.calls, self.calls + 1
        self.jobs.append(job)
        if n in self.fail_calls:
            raise self.error
        return await super().run_eval(host, job)


def _two_engines():
    """Baseline and candidate on one mock host; the baseline is evaluated first."""
    return mock_experiment(
        variants=[{"name": "base"}, {"name": "cand", "mock": {"max_num_seqs": 8}}],
        quality={"suite": SUITE, "baseline_variant": "base"},
    )


def _state(db: str, experiment_id) -> tuple[str, str | None, Counter[str], list[tuple[str, dict]]]:
    with session_scope(db) as s:
        exp = s.get(BenchExperiment, experiment_id)
        assert exp is not None
        runs = s.query(BenchRun).filter(BenchRun.experiment_id == experiment_id).all()
        gates = (
            s.query(BenchGateDecision)
            .filter(BenchGateDecision.experiment_id == experiment_id)
            .all()
        )
        return (
            exp.status,
            exp.abort_reason,
            Counter(r.status for r in runs),
            [(g.decision, g.details) for g in gates],
        )


@pytest.mark.parametrize(
    ("runs", "evals", "status", "code", "reason"),
    [
        ({RUN_COMPLETED: 2}, {RUN_COMPLETED: 2}, ExperimentStatus.COMPLETED, EXIT_OK, None),
        (
            {RUN_COMPLETED: 2},
            {RUN_COMPLETED: 1, RUN_FAILED: 1},
            ExperimentStatus.COMPLETED,
            EXIT_RUNS_FAILED,
            "1 of 2 quality evals failed",
        ),
        (
            {RUN_COMPLETED: 1, RUN_FAILED: 1},
            {RUN_FAILED: 1},
            ExperimentStatus.COMPLETED,
            EXIT_RUNS_FAILED,
            "1 of 2 runs failed (1 failed); 1 of 1 quality evals failed",
        ),
        (
            {RUN_FAILED: 2},
            {RUN_FAILED: 1},
            ExperimentStatus.FAILED,
            EXIT_FAILED,
            "all 2 runs failed (2 failed); 1 of 1 quality evals failed",
        ),
    ],
)
def test_judge_runs_counts_failed_evals(runs, evals, status, code, reason):
    assert judge_runs(Counter(runs), Counter(evals)) == (status, reason, code)


async def test_a_failed_baseline_eval_still_runs_the_candidate_and_gates_inconclusive(ctx):
    provider = FlakyEval(fail_calls={0})
    ctx.provider = provider
    outcome = await run_experiment(_two_engines(), ctx)

    assert outcome.status is ExperimentStatus.COMPLETED
    assert outcome.exit_code == EXIT_RUNS_FAILED
    assert outcome.reason == "1 of 2 quality evals failed"
    status, reason, runs, gates = _state(ctx.db_url, outcome.experiment_id)
    assert (status, reason) == ("completed", "1 of 2 quality evals failed")
    assert runs == Counter({"completed": 4})  # both engines ran their load
    # the candidate's eval ran, with nothing to score divergence against
    assert provider.calls == 2 and provider.jobs[1].reference is None
    assert [d for d, _ in gates] == ["inconclusive"]
    assert gates[0][1]["blocked"] is True
    assert "baseline eval failed" in gates[0][1]["reasons"][0]
    assert "langdetect" in gates[0][1]["reasons"][0]
    assert [g.decision for g in outcome.gates] == ["inconclusive"]
    kinds = [e["kind"] for e in outcome.events]
    assert kinds.count("quality_failed") == 1 and kinds.count("quality") == 1


async def test_a_failed_candidate_eval_gates_inconclusive(ctx):
    ctx.provider = FlakyEval(fail_calls={1})
    outcome = await run_experiment(_two_engines(), ctx)

    assert (outcome.status, outcome.exit_code) == (ExperimentStatus.COMPLETED, EXIT_RUNS_FAILED)
    _, _, runs, gates = _state(ctx.db_url, outcome.experiment_id)
    assert runs == Counter({"completed": 4})
    assert [d for d, _ in gates] == ["inconclusive"]
    assert gates[0][1]["reasons"][0].startswith("candidate eval failed: RuntimeError")
    failed = [e for e in outcome.events if e["kind"] == "quality_failed"]
    assert len(failed) == 1 and "langdetect" in failed[0]["error"]


async def test_every_eval_passing_is_unchanged(ctx):
    ctx.provider = FlakyEval(fail_calls=set())
    outcome = await run_experiment(_two_engines(), ctx)
    assert (outcome.status, outcome.reason, outcome.exit_code) == (
        ExperimentStatus.COMPLETED,
        None,
        EXIT_OK,
    )
    assert [g.decision for g in outcome.gates] != ["inconclusive"]


async def test_a_budget_abort_during_an_eval_still_ends_the_experiment(ctx):
    ctx.provider = FlakyEval(fail_calls={0}, error=BudgetAbort("spend reached the cap"))
    outcome = await run_experiment(_two_engines(), ctx)
    assert (outcome.status, outcome.exit_code) == (ExperimentStatus.ABORTED, EXIT_BUDGET_ABORT)
    assert outcome.reason == "spend reached the cap"
    _, _, runs, gates = _state(ctx.db_url, outcome.experiment_id)
    assert runs == Counter({"completed": 2}) and not gates  # the candidate never ran


async def test_a_lost_host_during_an_eval_still_ends_the_experiment(ctx):
    lost = HostLost(
        "h",
        state="TERMINATED",
        reason_code=None,
        reason_message="pod gone",
        detected_at=datetime.now(UTC),
        seconds_since_launch=60.0,
    )
    ctx.provider = FlakyEval(fail_calls={0}, error=lost)
    outcome = await run_experiment(_two_engines(), ctx)
    assert (outcome.status, outcome.exit_code) == (ExperimentStatus.FAILED, EXIT_FAILED)
    assert (outcome.reason or "").startswith("HostLost")
    _, _, runs, gates = _state(ctx.db_url, outcome.experiment_id)
    assert runs == Counter({"completed": 2}) and not gates  # the candidate never ran
