import math
from fractions import Fraction

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
    EVAL_FLOOR_PROMPT_S,
    EVAL_HARNESS_TASK_S,
    EVAL_ITEM_S,
    MOCK_EVAL_JOB_S,
    RUNPOD_TIMING,
    Estimator,
    PlanError,
    build_plan,
    runpod_accrual_terms,
)
from loom_bench.prices import HOURS_PER_MONTH, load_prices
from loom_bench.provenance import GitInfo
from loom_bench.quality.divergence import load_prompts
from loom_bench.quality.tasks import TASKS
from loom_bench.quality.tasks.json_schema import load_items
from loom_bench.registry import load_registry, read_yaml
from loom_bench.runner import EXIT_REFUSED, plan_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment

from .conftest import (
    LLAMA,
    LLAMA_RUNPOD,
    QWEN,
    QWEN_QUALITY_RUNPOD,
    QWEN_RUNPOD,
    RUNPOD_SMOKE,
    SMOKE,
    mock_doc,
    mock_experiment,
    write_yaml,
)

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
    # Properties, not exact values: overall_cap is lowered by hand for off-DB spend.
    assert isinstance(BUDGET, BudgetConfig)
    assert BUDGET.per_experiment_cap == CAP_50
    assert 0 < BUDGET.per_experiment_cap <= BUDGET.overall_cap <= parse_usd("$150")


@pytest.mark.parametrize("path", [QWEN, LLAMA], ids=lambda p: p.stem)
def test_real_experiments_fit_their_caps_with_margin(path):
    plan = _plan(load_experiment(path))
    assert plan.ok, plan.refusals
    assert plan.total_micros < CAP_50 * 0.6
    assert plan.ttl_worst_micros <= plan.caps.effective
    assert len(plan.hosts) == 1  # one host per experiment; warm restarts between configs
    assert plan.hosts[0].steps[0].kind == "cold_start"
    evals = [s for s in plan.hosts[0].steps if s.kind == "eval"]
    assert len(evals) == plan.n_cells and all(s.label.endswith("[phase0]") for s in evals)
    assert plan.hosts[0].seconds < 0.95 * plan.hosts[0].ttl_s


@pytest.mark.parametrize(("path", "pods"), [(QWEN_RUNPOD, 2), (LLAMA_RUNPOD, 1)], ids=str)
def test_runpod_experiments_fit_their_caps_with_ttl_margin(path, pods):
    exp = load_experiment(path)
    assert exp.provider.kind == "runpod"
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.ttl_worst_micros <= plan.caps.effective
    assert plan.total_micros < plan.caps.effective * 0.6
    assert len(plan.hosts) == pods  # one pod per engine image
    for host in plan.hosts:
        assert host.provider == "runpod" and host.market == "on_demand"
        assert host.steps[0].kind == "cold_start"
        assert host.seconds < 0.8 * host.ttl_s  # each pod finishes well inside its TTL


def test_the_quality_only_follow_up_fits_its_cap_and_runs_no_load():
    plan = _plan(load_experiment(QWEN_QUALITY_RUNPOD))
    assert plan.ok, plan.refusals
    assert plan.ttl_worst_micros <= plan.caps.effective
    # Three eval passes per pod: the (conservative) estimate is most of the TTL-bound cap.
    assert plan.total_micros < plan.caps.effective * 0.8
    assert plan.n_runs_max == 0 and len(plan.hosts) == 2  # one pod per engine image
    for host in plan.hosts:
        kinds = [s.kind for s in host.steps]
        assert kinds[0] == "cold_start" and kinds.count("eval") == 1
        assert "workload" not in kinds
        assert host.seconds < 0.8 * host.ttl_s


def test_the_quality_only_follow_up_evaluates_the_real_sweeps_configs():
    # It finishes 565b8d3f's gate: same engines, images, model, provider and quality
    # section, so its evals and gate land on that sweep's config hashes.
    follow, real = load_experiment(QWEN_QUALITY_RUNPOD), load_experiment(QWEN_RUNPOD)
    assert not follow.workloads and not follow.smoke
    assert follow.model == real.model and follow.provider == real.provider
    assert follow.variants == real.variants
    # The same quality section, except that the follow-up replicates its passes to
    # settle the gate (replicates change the estimator, not the config hashes).
    assert follow.quality is not None and real.quality is not None
    assert follow.quality.replicates > real.quality.replicates == 1
    assert follow.quality.model_copy(update={"replicates": 1}) == real.quality
    assert [c.config_hash for c in expand(follow, REGISTRY)] == [
        c.config_hash for c in expand(real, REGISTRY)
    ]


def test_runpod_smoke_stays_under_two_twenty_five_even_to_ttl():
    plan = _plan(load_experiment(RUNPOD_SMOKE))
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$2.25")
    assert plan.ttl_worst_micros <= plan.caps.effective  # both pods to their TTL
    assert plan.n_cells == 2 and len(plan.hosts) == 2  # one pod per engine image
    for host in plan.hosts:
        assert host.steps[0].kind == "cold_start"
        assert host.seconds < 0.8 * host.ttl_s


def _eval_paths(exp: Experiment) -> set[tuple[str, str]]:
    """(kind, harness task) of every eval task the experiment runs: what decides which
    code and which dependencies the pod's eval job needs."""
    assert exp.quality is not None
    suite = exp.quality.load()
    tasks = suite.select(exp.quality.subset)
    out = set()
    for t in tasks:
        harness = t.params.get("tasks") if t.kind == "lm_eval" else None
        for name in harness or [t.name]:
            out.add((t.kind, name))
    return out


def _load_shape(exp: Experiment) -> dict[str, tuple[str, str | None, bool]]:
    """Profile -> (load mode, search scale, whether the search can descend below lo)."""
    return {
        w.profile: (
            w.load.mode.value,
            w.load.search.scale if w.load.search else None,
            bool(w.load.search and w.load.search.descend),
        )
        for w in exp.workloads
    }


def test_runpod_smoke_covers_every_code_path_of_the_real_qwen_sweep():
    # The first 8B sweep (058128e9) died on an eval extra the old vLLM-only, eval-less
    # smoke never touched. The smoke must run what the sweep runs, at smoke scale.
    smoke, real = load_experiment(RUNPOD_SMOKE), load_experiment(QWEN_RUNPOD)
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model and smoke.provider == real.provider
    assert smoke.variants == real.variants  # both engines, same images by digest
    assert _load_shape(smoke) == _load_shape(real)  # profiles, load modes, search scale
    # Same repetitions: the pass verdict's CI uses t with reps-1 degrees of freedom, and
    # at 2 reps (t ~ 12.7) every point failed, so smoke 51ad57b0 never ran the climb.
    assert smoke.repetitions == real.repetitions
    assert smoke.slo == real.slo
    assert smoke.quality is not None and real.quality is not None
    assert (smoke.quality.suite, smoke.quality.subset, smoke.quality.baseline_variant) == (
        real.quality.suite,
        real.quality.subset,
        real.quality.baseline_variant,
    )
    assert smoke.quality.allow_code_exec == real.quality.allow_code_exec
    assert _eval_paths(smoke) == _eval_paths(real)
    assert smoke.quality.limit is not None and real.quality.limit is None
    smoke_div, real_div = smoke.quality.load().divergence, real.quality.load().divergence
    assert smoke_div is not None and real_div is not None  # capture, floor and scoring run
    # Divergence on characters split across byte-level tokens: the smoke's 4 ASCII prompts
    # never met them and SGLang's eval in sweep 565b8d3f died on prompts 20 and 40. The
    # smoke's prompts come from the real set and hold at least one of its hard prompts
    # (bench/tests/quality/test_divergence_byte_split.py proves they split characters).
    pinned = len(load_prompts())
    smoke_ids, real_ids = smoke_div.selected_ids(pinned), real_div.selected_ids(pinned)
    assert set(smoke_ids) <= set(real_ids)
    assert real_div.hard_prompts and set(real_div.hard_prompts) & set(smoke_ids)


def test_the_llama_sweep_runs_no_eval_path_or_load_shape_the_qwen_smoke_misses():
    # No separate Llama smoke: a 4x L40S pod is ~$4.40/h and its 141 GB download alone
    # takes most of an hour. Its eval tasks, load modes and engine image are the Qwen
    # smoke's; what only the real Llama run exercises is TP=4, the 250 GB disk and the
    # gated-model token (docs/runbook-runpod.md).
    smoke, llama = load_experiment(RUNPOD_SMOKE), load_experiment(LLAMA_RUNPOD)
    assert _eval_paths(llama) <= _eval_paths(smoke)
    assert set(_load_shape(llama).items()) <= set(_load_shape(smoke).items())
    smoke_images = {v.engine.image for v in smoke.variants if v.engine}
    assert {v.engine.image for v in llama.variants if v.engine} <= smoke_images


def test_runpod_specs_match_their_aws_counterparts_apart_from_the_provider():
    for aws, runpod in ((QWEN, QWEN_RUNPOD), (LLAMA, LLAMA_RUNPOD)):
        a, r = load_experiment(aws), load_experiment(runpod)
        for field in ("model", "variants", "workloads", "repetitions", "slo", "quality"):
            assert getattr(a, field) == getattr(r, field), (runpod.stem, field)


@pytest.mark.parametrize(
    "path", [QWEN, LLAMA, SMOKE, QWEN_RUNPOD, LLAMA_RUNPOD, RUNPOD_SMOKE], ids=lambda p: p.stem
)
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


@pytest.mark.parametrize("loadgen", ["vllm_bench", "sglang_bench"])
def test_external_load_generators_are_refused_on_aws(loadgen, tmp_path, db):
    exp = load_experiment(QWEN).model_copy(update={"loadgen": loadgen})
    (refusal,) = [r for r in _plan(exp).refusals if "loadgen" in r]
    assert f"loadgen {loadgen} runs `" in refusal and "use loadgen native" in refusal
    # the same wrapper is fine where the tool can be installed: mock and local runs
    assert _plan(mock_experiment(loadgen=loadgen)).ok
    path = write_yaml(tmp_path / "exp.yaml", exp.model_dump(mode="json"))
    for command in ("plan", "run"):
        result = CliRunner().invoke(app, [command, str(path), "--db", db])
        assert result.exit_code == EXIT_REFUSED, result.output
        assert "REFUSED" in result.output


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
    assert plan.caps.effective == BUDGET.overall_cap - parse_usd("$146")
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
JSON_ITEMS = len(load_items()[0])  # the pinned json_schema set (300 at data version 2)
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
        JSON_ITEMS * EVAL_ITEM_S["json_schema"] / EVAL_CONCURRENCY
        + 100 * EVAL_ITEM_S["lm_eval"] / 32
        + 10 * EVAL_DIVERGENCE_PROMPT_S / EVAL_CONCURRENCY
        + EVAL_HARNESS_TASK_S
        + AWS_TIMING.eval_job_s
    )
    floor = 10 * EVAL_FLOOR_PROMPT_S / 1  # the baseline's noise-floor pass, floor_concurrency 1
    assert [s.seconds for s in evals] == [pytest.approx(expected + floor), pytest.approx(expected)]
    assert exp.quality.baseline_variant in evals[0].label
    assert [s.kind for s in steps].count("eval_setup") == 1
    assert steps[1].kind == "eval_setup" and steps[1].seconds == AWS_TIMING.eval_setup_s
    without = _plan(load_experiment(QWEN).model_copy(update={"quality": None}))
    assert plan.total_seconds - without.total_seconds == pytest.approx(
        2 * expected + floor + AWS_TIMING.eval_setup_s
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
        JSON_ITEMS * EVAL_ITEM_S["json_schema"] / EVAL_CONCURRENCY * scale + MOCK_EVAL_JOB_S
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


def _runpod(path=QWEN, budget=None, **provider):
    doc = read_yaml(path)
    doc["provider"] = {"kind": "runpod", **provider}
    if budget:
        doc["budget"] = {**doc["budget"], **budget}
    return Experiment.model_validate(doc)


def test_runpod_gives_each_engine_image_its_own_pod():
    plan = _plan(_runpod())
    assert plan.ok, plan.refusals
    assert len(plan.hosts) == 2  # vLLM and SGLang images: one pod each
    for host in plan.hosts:
        kinds = [s.kind for s in host.steps]
        assert kinds[0] == "cold_start" and "warm_start" not in kinds
        assert host.provider == "runpod" and host.market == "on_demand"
        assert host.instance_type == "l40s-x1"
        assert host.steps[-1].kind == "teardown"
        assert host.steps[-1].seconds == RUNPOD_TIMING.teardown_s
    size = REGISTRY.get("qwen3-8b").hf.size_bytes
    t = RUNPOD_TIMING
    cold = t.boot_s + t.image_pull_s + size / t.download_bytes_per_s
    cold += size / t.load_bytes_per_s + t.engine_init_s
    assert plan.hosts[0].steps[0].seconds == pytest.approx(cold)


def test_runpod_hourly_price_is_the_price_book_plus_container_disk():
    host, _ = _plan(_runpod(container_disk_gb=100)).hosts
    it = PRICES.instance("runpod", "secure", "l40s-x1")
    storage = PRICES.region("runpod", "secure").storage
    disk = math.ceil(Fraction(storage.per_gb_month * 100, HOURS_PER_MONTH))
    assert host.hourly_micros == it.on_demand_per_hour + disk
    assert any("costPerHr" in n for n in host.price_notes)


def test_runpod_llama_needs_a_container_disk_that_fits_the_weights():
    p2p_off = {"NCCL_P2P_DISABLE": "1"}  # a multi-GPU pod must decide NCCL P2P
    refusals = _plan(_runpod(LLAMA, engine_env=p2p_off)).refusals
    (refusal,) = [r for r in refusals if "container_disk_gb" in r]
    assert "container_disk_gb 80 is below the 177 GB" in refusal
    plan = _plan(_runpod(LLAMA, container_disk_gb=250, engine_env=p2p_off))
    assert not [r for r in plan.refusals if "container_disk_gb" in r]
    assert plan.hosts[0].instance_type == "l40s-x4"


def test_runpod_gpu_count_must_match_the_priced_instance():
    refusals = _plan(_runpod(instance_type="l40s-x4")).refusals
    assert any("priced for 4 GPUs, the replica uses 1" in r for r in refusals)


def test_runpod_unpriced_instance_is_a_plan_error():
    with pytest.raises(PlanError, match="no price for runpod/secure"):
        _plan(_runpod(instance_type="l40s-x2"))


def test_runpod_ttl_above_the_pod_limit_is_refused():
    limit_min = runpod_accrual_terms().max_ttl_s / 60
    refusals = _plan(_runpod(budget={"ttl_minutes": limit_min + 1})).refusals
    assert any("RunPod pod limit" in r for r in refusals)


@pytest.mark.parametrize("loadgen", ["vllm_bench", "sglang_bench"])
def test_external_load_generators_are_refused_on_runpod(loadgen):
    exp = _runpod().model_copy(update={"loadgen": loadgen})
    (refusal,) = [r for r in _plan(exp).refusals if "loadgen" in r]
    assert "runpod job environment" in refusal and "use loadgen native on runpod" in refusal
