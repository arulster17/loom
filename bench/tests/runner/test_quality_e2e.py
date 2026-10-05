"""Quality inside an experiment on the mock provider: eval jobs per cell, the divergence
reference and its noise floor captured once on the baseline and reused, the gate, and the
report.

Every variant has the same per-request logit jitter (batch-variant numerics), which the
baseline's noise floor measures. "noisy-kernels" adds a little static logit noise (another
engine's kernels) and passes against the floor; "drifted" adds more, stays below the hard
ceiling and answers the same, so it needs review; "broken" fails.
"""

import asyncio
import csv
import shutil
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from loom_bench.budget import load_budget
from loom_bench.cli import app
from loom_bench.jobs import EvalJob
from loom_bench.prices import load_prices
from loom_bench.providers.mock import MockProvider
from loom_bench.quality.divergence import ReferenceLogprobs
from loom_bench.registry import load_registry
from loom_bench.runner import RunnerContext, run_experiment
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import BenchEvalRun, BenchGateDecision, BenchRun

from .conftest import mock_experiment, write_yaml

# The module-scoped experiment runs once: all its tests go to one xdist worker.
pytestmark = [pytest.mark.timeout(240), pytest.mark.xdist_group("quality-e2e")]

SUITE = {
    "suite": "quality-e2e",
    "model": "qwen3-8b",
    "seed": 1234,
    "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 1000},
    "tasks": [
        {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
        {"name": "json_schema", "kind": "json_schema"},
    ],
    # The mock's flat next-token distributions flip their top-1 far more easily than a real
    # model's, so the top-1 ceiling sits lower than in the pinned suites; a noise multiple
    # of 2 puts "drifted" between the calibrated limits and the ceiling.
    "divergence": {
        "prompts": 12,
        "top_k": 5,
        "max_new_tokens": 16,
        "noise_multiple": 2.0,
        "ceiling_top1": 0.6,
    },
}
JITTER = {"logprob_jitter": 0.25}


class RecordingProvider(MockProvider):
    def __init__(self) -> None:
        super().__init__(hourly_micros=1_000_000)
        self.eval_jobs: list[EvalJob] = []
        self.load_run_ids: list[str] = []

    async def run_job(self, host, job):
        self.load_run_ids.append(job.run_id)
        return await super().run_job(host, job)

    async def run_eval(self, host, job):
        self.eval_jobs.append(EvalJob.model_validate_json(job.model_dump_json()))
        return await super().run_eval(host, job)


@pytest.fixture(scope="module")
def quality_run(tmp_path_factory):
    """One experiment for the whole module: every test only reads what it recorded."""
    tmp = tmp_path_factory.mktemp("quality-e2e")
    db = f"sqlite:///{tmp / 'loom.db'}"
    upgrade(db)
    ctx = RunnerContext(
        db_url=db,
        out_dir=tmp / "results",
        registry=load_registry(),
        prices=load_prices(),
        budget=load_budget(),
    )
    suite = write_yaml(tmp / "suite.yaml", SUITE)
    exp = mock_experiment(
        variants=[
            {"name": "base", "mock": JITTER},
            {"name": "broken", "mock": {**JITTER, "degrade": 0.3, "logprob_noise": 3.0}},
            {"name": "noisy-kernels", "mock": {**JITTER, "logprob_noise": 0.3}},
            {"name": "drifted", "mock": {**JITTER, "logprob_noise": 0.5}},
        ],
        quality={"suite": str(suite), "baseline_variant": "base"},
        # No latency target: whether a config has a cost at SLO must not depend on how
        # busy the machine is (a missed SLO would list it as "no cost at SLO" instead).
        slo={"max_error_rate": 0.01},
    )
    ctx.provider = RecordingProvider()
    return asyncio.run(run_experiment(exp, ctx)), ctx.provider, ctx


@pytest.fixture
def outcome(quality_run):
    result, provider, _ = quality_run
    return result, provider


@pytest.fixture
def ctx(quality_run) -> RunnerContext:
    return quality_run[2]


@pytest.fixture
def db(ctx) -> str:
    return ctx.db_url


def test_gate_against_the_measured_noise_floor(outcome, db):
    result, _ = outcome
    assert result.status.value == "completed", result.reason
    gates = {g.cell: g for g in result.gates}
    assert gates["broken"].decision == "fail" and gates["broken"].blocked
    assert gates["noisy-kernels"].decision == "pass" and not gates["noisy-kernels"].blocked
    assert gates["drifted"].decision == "review" and not gates["drifted"].blocked
    with session_scope(db) as s:
        decisions = {d.decision: d.details for d in s.scalars(select(BenchGateDecision))}
        evals = list(s.scalars(select(BenchEvalRun)))
    assert set(decisions) == {"pass", "review", "fail"}
    fail, review, ok = decisions["fail"], decisions["review"], decisions["pass"]
    assert fail["divergence"]["verdict"] == "fail"
    assert "hard ceiling" in fail["divergence"]["reason"]
    for d in (fail, review, ok):
        limits = d["divergence"]["limits"]
        assert limits["calibrated"] and limits["noise_multiple"] == 2.0
        assert d["divergence"]["self_divergence"]["kl"]["point"] > 0
    # Beyond the absolute 0.05 nats yet within 2x the floor: the floor widened the limit.
    ok_kl = ok["divergence"]["result"]["kl"]["point"]
    assert 0.05 < ok_kl < ok["divergence"]["limits"]["max_kl"]
    assert [t["verdict"] for t in review["tasks"]] == ["pass", "pass"]
    assert review["divergence"]["reason"].startswith("needs review: ")
    assert len(evals) == 8 and {e.task for e in evals} == {"arithmetic", "json_schema"}


def test_reference_captured_once_and_reused(outcome, ctx):
    result, provider = outcome
    modes = [j.divergence for j in provider.eval_jobs]
    assert modes == ["capture_and_floor", "score", "score", "score"]
    captured = [e for e in result.events if e["kind"] == "reference_captured"]
    assert len(captured) == 1
    stored = ReferenceLogprobs.model_validate_json(Path(captured[0]["path"]).read_text())
    assert stored.config_hash == captured[0]["config_hash"]
    assert stored.self_divergence is not None and stored.self_divergence.kl.point > 0
    assert captured[0]["self_divergence"] == stored.self_divergence.model_dump(mode="json")
    assert stored.provenance["config_hash"] == stored.config_hash
    assert len(stored.prompts) == 12
    assert all(j.reference == stored for j in provider.eval_jobs[1:])
    assert all(j.suite.suite == "quality-e2e" and j.seed == 1234 for j in provider.eval_jobs)


def test_run_rows_carry_the_load_job_ids(outcome, db):
    result, provider = outcome
    with session_scope(db) as s:
        rows = {str(r.id): r for r in s.scalars(select(BenchRun))}
    assert set(rows) == set(provider.load_run_ids) == {str(i) for i in result.run_ids}
    for rid, row in rows.items():
        assert row.requests_uri.endswith(f"runs/{rid}/requests.parquet")


def test_report_ranks_the_gate_failed_config_last(outcome, db, tmp_path):
    result, _ = outcome
    out = tmp_path / "report"
    rep = CliRunner().invoke(
        app, ["report", "-e", str(result.experiment_id), "--db", db, "--out", str(out)]
    )
    assert rep.exit_code == 0, rep.output
    with (out / "leaderboard.csv").open() as f:
        rows = list(csv.DictReader(f))
    with session_scope(db) as s:
        broken = s.scalars(
            select(BenchGateDecision.candidate_config_hash).where(
                BenchGateDecision.decision == "fail"
            )
        ).one()
    by_board: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_board.setdefault(row["workload"], []).append(row)
    assert by_board
    for board in by_board.values():
        (failed,) = [i for i, r in enumerate(board) if r["config_hash"] == broken]
        assert board[failed]["status"] == "quality gate failed" and not board[failed]["rank"]
        assert all(not r["rank"] for r in board[failed:])
        assert all(r["config_hash"] != broken for r in board if r["rank"])


def _gated(db: str, decision: str) -> tuple[str, str]:
    with session_scope(db) as s:
        row = s.scalars(
            select(BenchGateDecision).where(BenchGateDecision.decision == decision)
        ).one()
        return row.baseline_config_hash, row.candidate_config_hash


def _regate(db: str, decision: str):
    base, cand = _gated(db, decision)
    return CliRunner().invoke(
        app, ["quality", "gate", "--baseline", base, "--candidate", cand, "--db", db]
    )


def test_quality_gate_cli_redecides_with_the_stored_floor(outcome, db, tmp_path):
    # Re-deciding records a new decision: work on a copy so the other tests see one each.
    copy = tmp_path / "regate.db"
    shutil.copy(db.removeprefix("sqlite:///"), copy)
    db = f"sqlite:///{copy}"
    review = _regate(db, "review")
    assert review.exit_code == 0, review.output
    assert "gate REVIEW (needs review, not blocking)" in review.output
    fail = _regate(db, "fail")
    assert fail.exit_code == 7, fail.output
    assert "hard ceiling" in fail.output


def test_report_ranks_the_review_config_and_flags_it(outcome, db, tmp_path):
    result, _ = outcome
    out = tmp_path / "report"
    rep = CliRunner().invoke(
        app, ["report", "-e", str(result.experiment_id), "--db", db, "--out", str(out)]
    )
    assert rep.exit_code == 0, rep.output
    _, drifted = _gated(db, "review")
    with (out / "leaderboard.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["config_hash"] == drifted]
    assert rows
    for row in rows:
        assert row["quality_gate"] == "review" and row["status"] != "quality gate failed"
        assert "noise floor" in row["quality_divergence"]
        if row["status"] == "ranked":  # an untrusted row's recommendation is to rerun
            assert "quality needs review vs base" in row["recommendation"]
