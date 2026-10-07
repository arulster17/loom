"""A quality-only experiment: no workloads, only each engine's eval job, the divergence
reference and scoring, and the gate. It finishes a gate whose load sweep is already done
(565b8d3f: every load run passed, SGLang's divergence did not) without re-running load."""

from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.experiment import Experiment
from loom_bench.runner import EXIT_OK, plan_experiment, run_experiment
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchEvalRun, BenchGateDecision, BenchRun, ExperimentStatus

from .conftest import mock_doc, write_yaml

SUITE = Path("bench/experiments/suites/mock-smoke.yaml")


@pytest.fixture
def suite(tmp_path: Path) -> str:
    doc = SUITE.read_text().replace("suite: mock-smoke", "suite: mock-divergence")
    doc += "divergence:\n  prompts: 4\n  top_k: 5\n  max_new_tokens: 8\n  floor_concurrency: 1\n"
    path = tmp_path / "mock-divergence.yaml"
    path.write_text(doc)
    return str(path)


def quality_only(suite: str) -> Experiment:
    doc = mock_doc(
        variants=[{"name": "base"}, {"name": "cand", "mock": {"max_num_seqs": 8}}],
        quality={"suite": suite, "baseline_variant": "base"},
    )
    doc["workloads"] = []
    doc.pop("slo", None)
    return Experiment.model_validate(doc)


def test_an_experiment_needs_workloads_or_quality():
    doc = mock_doc()
    doc["workloads"] = []
    with pytest.raises(ValueError, match="quality section to be quality-only"):
        Experiment.model_validate(doc)


def test_a_quality_only_plan_has_only_cold_starts_and_evals(ctx, suite):
    _, plan = plan_experiment(quality_only(suite), ctx)
    assert plan.ok and plan.n_runs_max == 0 and plan.n_cells == 2
    kinds = [step.kind for h in plan.hosts for step in h.steps]
    assert kinds.count("eval") == 2
    assert set(kinds) <= {"cold_start", "warm_start", "eval", "teardown"}


async def test_a_quality_only_experiment_evaluates_and_gates(ctx, suite):
    outcome = await run_experiment(quality_only(suite), ctx)
    assert (outcome.status, outcome.reason, outcome.exit_code) == (
        ExperimentStatus.COMPLETED,
        None,
        EXIT_OK,
    )
    with session_scope(ctx.db_url) as s:
        eid = outcome.experiment_id
        runs = s.query(BenchRun).filter(BenchRun.experiment_id == eid).count()
        evals = Counter(
            r.config_hash
            for r in s.query(BenchEvalRun).filter(BenchEvalRun.experiment_id == eid).all()
        )
        gates = s.query(BenchGateDecision).filter(BenchGateDecision.experiment_id == eid).all()
        assert runs == 0
        assert sorted(evals.values()) == [2, 2]
        assert len(gates) == 1 and gates[0].details["divergence"]["verdict"] != "inconclusive"
    kinds = [e["kind"] for e in outcome.events]
    assert kinds.count("reference_captured") == 1 and kinds.count("gate") == 1


def test_a_follow_up_gate_reaches_the_load_experiments_report(db, tmp_path, suite):
    # The load sweep first (its quality lost), then the quality-only follow-up: the
    # default report ranks the sweep's configs and shows the follow-up's gate on them.
    out = tmp_path / "results"
    load = mock_doc(variants=[{"name": "base"}, {"name": "cand", "mock": {"max_num_seqs": 8}}])
    follow = quality_only(suite).model_dump(mode="json", exclude_none=True)
    for name, doc in (("load", load), ("quality", follow)):
        path = write_yaml(tmp_path / f"{name}.yaml", doc)
        result = CliRunner().invoke(app, ["run", str(path), "--db", db, "--out", str(out)])
        assert result.exit_code == EXIT_OK, result.output
    report = CliRunner().invoke(app, ["report", "--db", db, "--out", str(tmp_path / "rep")])
    assert report.exit_code == 0, report.output
    csv = (tmp_path / "rep" / "leaderboard.csv").read_text()
    assert csv.count("\n") >= 3  # header and both configs
    only = CliRunner().invoke(
        app, ["report", "--db", db, "--out", str(tmp_path / "rep2"), "--experiment", _last(db)]
    )
    assert only.exit_code != 0 and "ran no load" in only.output


def _last(db: str) -> str:
    from loom_bench.store.models import BenchExperiment

    with session_scope(db) as s:
        exp = s.query(BenchExperiment).order_by(BenchExperiment.created_at.desc()).first()
        assert exp is not None
        return str(exp.id)
