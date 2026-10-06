"""An experiment's final status reflects its load runs: all failed is `failed`, some failed
is `completed` with a reason and exit 8, none failed is `completed` and exit 0."""

from collections import Counter

import pytest
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.jobs import LoadJob, LoadJobResult
from loom_bench.providers.base import Host
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import (
    EXIT_FAILED,
    EXIT_OK,
    EXIT_RUNS_FAILED,
    RUN_COMPLETED,
    RUN_FAILED,
    judge_runs,
    run_experiment,
)
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment, BenchRun, ExperimentStatus

from .conftest import mock_doc, mock_experiment, write_yaml


class FlakyMock(MockProvider):
    """Mock provider whose run_job fails on the listed call numbers (0-based)."""

    def __init__(self, fail_calls: set[int] | None = None) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.fail_calls = fail_calls
        self.calls = 0

    async def run_job(self, host: Host, job: LoadJob) -> LoadJobResult:
        n, self.calls = self.calls, self.calls + 1
        if self.fail_calls is None or n in self.fail_calls:
            raise RuntimeError("loom-error client Python checksum mismatch")
        return await super().run_job(host, job)


def _statuses(db: str, experiment_id) -> tuple[str, str | None, list[str]]:
    with session_scope(db) as s:
        exp = s.get(BenchExperiment, experiment_id)
        assert exp is not None
        runs = s.query(BenchRun).filter(BenchRun.experiment_id == experiment_id).all()
        return exp.status, exp.abort_reason, sorted(r.status for r in runs)


@pytest.mark.parametrize(
    ("statuses", "status", "code", "reason"),
    [
        ({RUN_COMPLETED: 3}, ExperimentStatus.COMPLETED, EXIT_OK, None),
        ({}, ExperimentStatus.COMPLETED, EXIT_OK, None),
        ({RUN_FAILED: 2}, ExperimentStatus.FAILED, EXIT_FAILED, "all 2 runs failed (2 failed)"),
        (
            {RUN_COMPLETED: 2, RUN_FAILED: 1},
            ExperimentStatus.COMPLETED,
            EXIT_RUNS_FAILED,
            "1 of 3 runs failed (1 failed)",
        ),
    ],
)
def test_judge_runs(statuses, status, code, reason):
    assert judge_runs(Counter(statuses)) == (status, reason, code)


async def test_every_run_failing_marks_the_experiment_failed(ctx):
    ctx.provider = FlakyMock()
    outcome = await run_experiment(mock_experiment(), ctx)
    assert outcome.status is ExperimentStatus.FAILED
    assert outcome.exit_code == EXIT_FAILED
    assert outcome.reason == "all 2 runs failed (2 failed)"
    assert _statuses(ctx.db_url, outcome.experiment_id) == (
        "failed",
        "all 2 runs failed (2 failed)",
        ["failed", "failed"],
    )


async def test_some_runs_failing_completes_with_a_reason_and_nonzero_exit(ctx):
    ctx.provider = FlakyMock(fail_calls={0})
    outcome = await run_experiment(mock_experiment(), ctx)
    assert outcome.status is ExperimentStatus.COMPLETED
    assert outcome.exit_code == EXIT_RUNS_FAILED
    assert _statuses(ctx.db_url, outcome.experiment_id) == (
        "completed",
        "1 of 2 runs failed (1 failed)",
        ["completed", "failed"],
    )


async def test_no_failed_runs_is_completed_and_exits_zero(ctx):
    ctx.provider = FlakyMock(fail_calls=set())
    outcome = await run_experiment(mock_experiment(), ctx)
    assert (outcome.status, outcome.reason, outcome.exit_code) == (
        ExperimentStatus.COMPLETED,
        None,
        EXIT_OK,
    )


@pytest.mark.parametrize(
    ("fail_calls", "code", "text"),
    [(None, EXIT_FAILED, "all 2 runs failed"), ({1}, EXIT_RUNS_FAILED, "1 of 2 runs failed")],
)
def test_bench_run_exits_nonzero_and_says_runs_failed(
    db, tmp_path, monkeypatch, fail_calls, code, text
):
    real_run_job = MockProvider.run_job
    calls = Counter[str]()

    async def run_job(self, host, job):
        n = calls["n"]
        calls["n"] += 1
        if fail_calls is None or n in fail_calls:
            raise RuntimeError("loom-error client Python checksum mismatch")
        return await real_run_job(self, host, job)

    monkeypatch.setattr(MockProvider, "run_job", run_job)
    path = write_yaml(tmp_path / "exp.yaml", mock_doc())
    result = CliRunner().invoke(
        app, ["run", str(path), "--db", db, "--out", str(tmp_path / "results")]
    )
    assert result.exit_code == code, result.output
    assert text in result.output
