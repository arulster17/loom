"""Smoke experiments (`smoke: true`) run the real code paths at minimal scale: their eval
jobs carry the limited suite, and they are left out of default reports and the site
because their cells share config hashes with the real experiment's."""

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.jobs import EvalJob, EvalJobResult
from loom_bench.providers.base import Host
from loom_bench.providers.mock import MockProvider
from loom_bench.runner import run_experiment
from loom_bench.site.snapshot import _experiments
from loom_bench.store.db import session_scope

from .conftest import mock_experiment

SUITE = "bench/experiments/suites/mock-smoke.yaml"


class RecordingMock(MockProvider):
    def __init__(self) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.jobs: list[EvalJob] = []

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        self.jobs.append(job)
        return await super().run_eval(host, job)


def _smoke(**over):
    return mock_experiment(
        name="smoke-unit",
        smoke=True,
        quality={"suite": SUITE, "baseline_variant": "a", "limit": 2},
        **over,
    )


def test_quality_limit_is_for_smoke_experiments_only():
    with pytest.raises(ValidationError, match="smoke experiments only"):
        mock_experiment(quality={"suite": SUITE, "baseline_variant": "a", "limit": 2})
    assert _smoke().smoke


async def test_a_smoke_eval_job_carries_the_limited_suite(ctx):
    provider = RecordingMock()
    ctx.provider = provider
    outcome = await run_experiment(_smoke(), ctx)
    assert outcome.status.value == "completed", outcome.reason
    (job,) = provider.jobs
    assert job.suite.item_limit == 2
    assert all((t.planned_items() or 0) <= 2 for t in job.suite.tasks)


async def test_smoke_experiments_stay_out_of_default_reports_and_the_site(ctx, tmp_path):
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    smoke = await run_experiment(_smoke(), ctx)
    run = CliRunner().invoke(app, ["report", "--db", ctx.db_url, "--out", str(tmp_path)])
    assert run.exit_code != 0 and "no matching experiments" in run.output
    picked = CliRunner().invoke(
        app,
        ["report", "-e", str(smoke.experiment_id), "--db", ctx.db_url, "--out", str(tmp_path)],
    )
    assert picked.exit_code == 0, picked.output  # still reportable when asked for by id

    real = await run_experiment(mock_experiment(name="real-unit"), ctx)
    with session_scope(ctx.db_url) as s:
        assert [e.id for e in _experiments(s, "latest")] == [real.experiment_id]
