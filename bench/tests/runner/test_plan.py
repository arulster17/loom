import pytest
from typer.testing import CliRunner

from loom_bench.budget import BudgetConfig, caps_for, load_budget
from loom_bench.cli import app
from loom_bench.experiment import ExpansionError, Experiment, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.plan import (
    AWS_TIMING,
    EVAL_CONCURRENCY,
    EVAL_DIVERGENCE_PROMPT_S,
    EVAL_HARNESS_TASK_S,
    EVAL_ITEM_S,
    MOCK_EVAL_JOB_S,
    Estimator,
    build_plan,
)
from loom_bench.prices import load_prices
from loom_bench.provenance import GitInfo
from loom_bench.quality.tasks import TASKS
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


def _quality_suite(tmp_path, tasks, **over):
    doc = {"suite": "plan-test", "model": "qwen3-8b", "tasks": tasks, **over}
    return str(write_yaml(tmp_path / "suite.yaml", doc))


JSON_TASK = {"name": "json_schema", "kind": "json_schema"}
GSM8K = {
    "name": "gsm8k",
    "kind": "lm_eval",
    "params": {"tasks": ["gsm8k"], "metric": "exact_match", "num_concurrent": 32},
}


def _with_quality(exp, suite_path, **quality):
    q = {"suite": suite_path, "baseline_variant": exp.variants[0].name, **quality}
    return Experiment.model_validate({**exp.model_dump(mode="json"), "quality": q})


def test_aws_eval_time_comes_from_the_suite(tmp_path):
    path = _quality_suite(
        tmp_path, [JSON_TASK, {**GSM8K, "items": 100}], divergence={"prompts": 10}
    )
    exp = _with_quality(load_experiment(QWEN), path)
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    steps = plan.hosts[0].steps
    evals = [s for s in steps if s.kind == "eval"]
    expected = (
        60 * EVAL_ITEM_S["json_schema"] / EVAL_CONCURRENCY
        + 100 * EVAL_ITEM_S["lm_eval"] / 32
        + 10 * EVAL_DIVERGENCE_PROMPT_S / EVAL_CONCURRENCY
        + EVAL_HARNESS_TASK_S
        + AWS_TIMING.eval_job_s
    )
    assert [s.seconds for s in evals] == [pytest.approx(expected)] * 2
    assert [s.kind for s in steps].count("eval_setup") == 1
    assert steps[1].kind == "eval_setup" and steps[1].seconds == AWS_TIMING.eval_setup_s
    without = _plan(load_experiment(QWEN))
    assert plan.total_seconds - without.total_seconds == pytest.approx(
        2 * expected + AWS_TIMING.eval_setup_s
    )
    assert plan.total_micros > without.total_micros


def test_eval_subset_and_mock_time_scale(tmp_path):
    path = _quality_suite(
        tmp_path, [JSON_TASK, {**GSM8K, "items": 100}], subsets={"q": ["json_schema"]}
    )
    exp = mock_experiment(quality={"suite": path, "baseline_variant": "a", "subset": "q"})
    (step,) = [s for s in _plan(exp).hosts[0].steps if s.kind == "eval"]
    scale = exp.provider.time_scale
    assert step.label.endswith("[q]")
    assert step.seconds == pytest.approx(
        60 * EVAL_ITEM_S["json_schema"] / EVAL_CONCURRENCY * scale + MOCK_EVAL_JOB_S
    )


def test_every_task_kind_has_an_eval_time_constant():
    assert set(EVAL_ITEM_S) == set(TASKS)


def test_uncounted_lm_eval_task_is_refused(tmp_path):
    exp = _with_quality(load_experiment(QWEN), _quality_suite(tmp_path, [GSM8K]))
    assert any("no item count" in r for r in _plan(exp).refusals)


def test_code_exec_tasks_need_the_opt_in(tmp_path):
    code = {"name": "code", "kind": "code_exec", "params": {"datasets": ["mbpp"]}}
    path = _quality_suite(tmp_path, [JSON_TASK, code], subsets={"safe": ["json_schema"]})
    with pytest.raises(ExpansionError, match="allow_code_exec"):
        expand(_with_quality(load_experiment(QWEN), path), REGISTRY)
    assert expand(_with_quality(load_experiment(QWEN), path, subset="safe"), REGISTRY)
    assert expand(_with_quality(load_experiment(QWEN), path, allow_code_exec=True), REGISTRY)
    with pytest.raises(ExpansionError, match="no subset"):
        expand(_with_quality(load_experiment(QWEN), path, subset="nope"), REGISTRY)
