import pytest
from typer.testing import CliRunner

from loom_bench.budget import BudgetConfig, caps_for, load_budget
from loom_bench.cli import app
from loom_bench.experiment import expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.plan import AWS_TIMING, Estimator, build_plan
from loom_bench.prices import load_prices
from loom_bench.provenance import GitInfo
from loom_bench.registry import load_registry
from loom_bench.runner import EXIT_REFUSED, plan_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment

from .conftest import LLAMA, QWEN, SMOKE, mock_doc, mock_experiment, write_yaml

REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()
CAP_50 = parse_usd("$50")


def _plan(exp, overall_spent=0):
    return build_plan(
        exp,
        expand(exp, REGISTRY),
        prices=PRICES,
        caps=caps_for(exp.budget.max_spend, BUDGET, overall_spent),
    )


def test_budget_yaml_caps():
    assert BudgetConfig(overall_cap=parse_usd("$150"), per_experiment_cap=CAP_50) == BUDGET


@pytest.mark.parametrize("path", [QWEN, LLAMA], ids=lambda p: p.stem)
def test_real_experiments_fit_their_caps_with_margin(path):
    plan = _plan(load_experiment(path))
    assert plan.ok, plan.refusals
    assert plan.total_micros < CAP_50 * 0.6
    assert plan.ttl_worst_micros <= plan.caps.effective
    assert len(plan.hosts) == 1  # one host per experiment; warm restarts between configs
    assert plan.hosts[0].steps[0].kind == "cold_start"


@pytest.mark.parametrize("path", [QWEN, LLAMA, SMOKE], ids=lambda p: p.stem)
def test_bench_plan_cli_accepts_shipped_experiments(path, db):
    result = CliRunner().invoke(app, ["plan", str(path), "--db", db])
    assert result.exit_code == 0, result.output
    assert "estimated spend" in result.output


def test_qwen_plan_warm_starts_sglang_without_redownloading():
    plan = _plan(load_experiment(QWEN))
    steps = {s.kind: s for s in plan.hosts[0].steps}
    assert steps["warm_start"].seconds < steps["cold_start"].seconds
    # Different image (pull) but the same checkpoint: no second download.
    size = REGISTRY.get("qwen3-8b").hf.size_bytes
    expected = (
        AWS_TIMING.image_pull_s + size / AWS_TIMING.load_bytes_per_s + AWS_TIMING.engine_init_s
    )
    assert steps["warm_start"].seconds == pytest.approx(expected)


def test_open_loop_runs_are_priced_with_their_drain_timeout():
    exp = load_experiment(QWEN)
    cell = expand(exp, REGISTRY)[0]
    entry = exp.workloads[0]
    run_s = Estimator(exp).run_s(cell, entry.load, entry.resolve(), 1.0)
    assert run_s == entry.load.duration_s + entry.load.drain_timeout_s + AWS_TIMING.run_overhead_s


def test_planner_refuses_over_budget():
    exp = mock_experiment(provider={"kind": "mock", "hourly_price": "$100000", "time_scale": 0.01})
    plan = _plan(exp)
    assert not plan.ok
    assert any("exceeds the effective cap" in r for r in plan.refusals)


def test_max_spend_above_per_experiment_cap_is_refused():
    plan = _plan(mock_experiment(budget={"max_spend": "$60", "ttl_minutes": 1}))
    assert any("per-experiment cap" in r for r in plan.refusals)


def test_ttl_worst_case_and_ttl_too_short_are_refused():
    pricey = mock_experiment(
        provider={"kind": "mock", "hourly_price": "$100", "time_scale": 0.01},
        budget={"max_spend": "$1", "ttl_minutes": 60},
    )
    assert any("lives to its TTL" in r for r in _plan(pricey).refusals)
    short = mock_experiment(budget={"max_spend": "$1", "ttl_minutes": 0.001})
    assert any("exceeds the host TTL" in r for r in _plan(short).refusals)


def test_aws_ttl_above_the_provider_limit_is_refused():
    exp = load_experiment(QWEN)
    exp = exp.model_copy(update={"budget": exp.budget.model_copy(update={"ttl_minutes": 600})})
    assert any("AWS host limit" in r for r in _plan(exp).refusals)


def _prior(db, kind, spent):
    with session_scope(db) as s:
        e = repo.create_experiment(
            s,
            name=f"prior-{kind}",
            spec={"provider": {"kind": kind}},
            git=GitInfo(),
            budget_micros=CAP_50,
            status="completed",
        )
        s.get(BenchExperiment, e.id).spent_micros = spent


def test_overall_cap_counts_prior_billable_spend(ctx, db):
    exp = mock_experiment(
        provider={"kind": "mock", "hourly_price": "$3600", "time_scale": 0.01},
        budget={"max_spend": "$40", "ttl_minutes": 0.1},
    )
    _, plan = plan_experiment(exp, ctx)
    assert plan.ok, plan.refusals

    _prior(db, "mock", parse_usd("$149"))  # simulated: never counts
    _, plan = plan_experiment(exp, ctx)
    assert plan.ok and plan.caps.overall_spent == 0

    _prior(db, "aws_ec2", parse_usd("$146"))
    _, plan = plan_experiment(exp, ctx)
    assert plan.caps.overall_spent == parse_usd("$146")
    assert plan.caps.effective == parse_usd("$4")
    assert not plan.ok


def test_run_refuses_before_creating_anything(db, tmp_path):
    doc = mock_doc(provider={"kind": "mock", "hourly_price": "$100000", "time_scale": 0.01})
    path = write_yaml(tmp_path / "pricey.yaml", doc)
    result = CliRunner().invoke(app, ["run", str(path), "--db", db, "--out", str(tmp_path)])
    assert result.exit_code == EXIT_REFUSED
    assert "REFUSED" in result.output
    with session_scope(db) as s:
        assert s.query(BenchExperiment).count() == 0


def test_dry_run_prints_the_plan_and_stops(db, tmp_path):
    result = CliRunner().invoke(
        app, ["run", str(SMOKE), "--dry-run", "--db", db, "--out", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "Plan: mock-smoke" in result.output
    with session_scope(db) as s:
        assert s.query(BenchExperiment).count() == 0
