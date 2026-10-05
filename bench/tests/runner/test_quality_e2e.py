"""Quality inside an experiment on the mock provider: eval jobs per cell, the divergence
reference captured once on the baseline and reused, the gate, and the report."""

import asyncio
import csv
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.jobs import EvalJob
from loom_bench.providers.mock import MockProvider
from loom_bench.quality.divergence import ReferenceLogprobs
from loom_bench.runner import run_experiment
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchEvalRun, BenchGateDecision, BenchRun

from .conftest import mock_experiment, write_yaml

pytestmark = pytest.mark.timeout(240)

SUITE = {
    "suite": "quality-e2e",
    "model": "qwen3-8b",
    "seed": 1234,
    "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 1000},
    "tasks": [
        {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
        {"name": "json_schema", "kind": "json_schema"},
    ],
    "divergence": {"prompts": 12, "top_k": 5, "max_new_tokens": 16},
}


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


@pytest.fixture
def outcome(ctx, tmp_path):
    suite = write_yaml(tmp_path / "suite.yaml", SUITE)
    exp = mock_experiment(
        variants=[
            {"name": "base"},
            {"name": "broken", "mock": {"degrade": 0.3, "logprob_noise": 3.0}},
            {"name": "small-batch", "mock": {"max_num_seqs": 8}},
        ],
        quality={"suite": str(suite), "baseline_variant": "base"},
    )
    ctx.provider = RecordingProvider()
    return asyncio.run(run_experiment(exp, ctx)), ctx.provider


def test_gate_blocks_the_degraded_candidate(outcome, db):
    result, _ = outcome
    assert result.status.value == "completed", result.reason
    gates = {g.cell: g for g in result.gates}
    assert gates["broken"].decision == "fail" and gates["broken"].blocked
    assert gates["small-batch"].decision == "pass" and not gates["small-batch"].blocked
    with session_scope(db) as s:
        decisions = {d.decision: d.details for d in s.scalars(select(BenchGateDecision))}
        evals = list(s.scalars(select(BenchEvalRun)))
    assert set(decisions) == {"pass", "fail"}
    assert decisions["fail"]["divergence"]["verdict"] == "fail"
    assert decisions["fail"]["divergence"]["result"]["kl"]["point"] > 0.05
    assert decisions["pass"]["divergence"]["result"]["top1"]["point"] == 1.0
    assert len(evals) == 6 and {e.task for e in evals} == {"arithmetic", "json_schema"}


def test_reference_captured_once_and_reused(outcome, ctx):
    result, provider = outcome
    modes = [j.divergence for j in provider.eval_jobs]
    assert modes == ["capture", "score", "score"]
    captured = [e for e in result.events if e["kind"] == "reference_captured"]
    assert len(captured) == 1
    stored = ReferenceLogprobs.model_validate_json(Path(captured[0]["path"]).read_text())
    assert stored.config_hash == captured[0]["config_hash"]
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
