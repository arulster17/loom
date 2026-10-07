"""Smoke experiments (`smoke: true`) run the real code paths at minimal scale: their eval
jobs carry the limited suite, and they are left out of default reports and the site
because their cells share config hashes with the real experiment's."""

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.experiment import is_smoke, smoke_spec_names
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


def test_a_smoke_is_flagged_or_named_after_a_shipped_smoke_spec(tmp_path):
    (tmp_path / "a-smoke.yaml").write_text("name: a-smoke\nsmoke: true\n")
    (tmp_path / "real.yaml").write_text("name: real\n")
    (tmp_path / "mock-smoke.yaml").write_text("name: mock-smoke\n")  # a name is not a flag
    names = smoke_spec_names(tmp_path)
    assert names == frozenset({"a-smoke"})
    assert is_smoke({"name": "anything", "smoke": True}, names)
    assert is_smoke({"name": "a-smoke"}, names)  # recorded before the flag existed
    assert not is_smoke({"name": "real"}, names)
    assert not is_smoke({"name": "mock-smoke", "smoke": False}, names)
    assert not is_smoke(None, names)
    assert not is_smoke({}, names)


def test_the_shipped_runpod_smoke_spec_is_a_smoke_and_mock_smoke_is_not():
    names = smoke_spec_names()
    assert "runpod-smoke" in names
    assert "mock-smoke" not in names


async def test_unflagged_runs_of_a_smoke_spec_stay_out_too(ctx, tmp_path):
    """runpod-smoke runs 32b9a262 and 17d0cb33 predate `smoke: true`: same name, no flag."""
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    legacy = await run_experiment(mock_experiment(name="runpod-smoke"), ctx)
    assert legacy.status.value == "completed", legacy.reason
    run = CliRunner().invoke(app, ["report", "--db", ctx.db_url, "--out", str(tmp_path)])
    assert run.exit_code != 0 and "no matching experiments" in run.output
    with session_scope(ctx.db_url) as s:
        assert _experiments(s, "latest") == []
