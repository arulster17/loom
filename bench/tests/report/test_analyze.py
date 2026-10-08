import uuid

import pytest

from loom_bench.cost import (
    NO_LOCAL_PRICE,
    NO_PRICE_BASIS,
    CostAllocation,
    PriceColumn,
    cost_from_goodput,
    recorded_hourly_micros,
)
from loom_bench.prices import UnverifiedPriceError
from loom_bench.provenance import Provenance, provenance_hash
from loom_bench.records import LoadMode, Market
from loom_bench.report.analyze import (
    UNTRUSTING,
    WarningKind,
    analyze_runs,
    cold_starts_by_config,
    default_price_resolver,
    label_from_provenance,
    load_points,
    quality_for,
    with_quality,
)
from loom_bench.store.models import BenchColdStart

from .factories import EXPERIMENT, LOADS, SLO, TTFT_SGLANG, eval_row, gate_row, make_runs

# g6e.xlarge us-east-1 from bench/prices.yaml plus a 200 GB gp3 volume
# ($0.08/GB-month x 200 / 730 h = $0.0219178/h), rounded once.
ON_DEMAND = 1_882_918  # 1_861_000 + 21_917.8
SPOT = 1_860_518  # 1_838_600 + 21_917.8
AS_RUN_SPOT = 1_721_918  # observed $1.70/h + 21_917.8


@pytest.fixture(scope="module")
def results(all_runs, price_book):
    return analyze_runs(
        all_runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


def by_name(results):
    return {r.name: r for r in results}


def analyze(runs, price_book, allocation=None):
    return analyze_runs(
        runs,
        slo=SLO,
        allocation=allocation or CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


def kinds(result):
    return {w.kind for w in result.warnings}


def test_groups_by_config_and_load_point(results):
    assert [r.name for r in results] == sorted(
        [r.name for r in results], key=lambda n: by_name(results)[n].config_hash
    )
    assert set(by_name(results)) == {"vllm-bf16", "sglang-bf16", "vllm-awq"}
    for r in results:
        loads = [*LOADS, 10.0] if r.name == "vllm-awq" else list(LOADS)
        assert r.workload == "chat"
        assert r.load_mode is LoadMode.OPEN_LOOP
        assert [p.load for p in r.points] == loads
        assert all(p.aggregate.n_runs == 3 and p.repetitions == [1, 2, 3] for p in r.points)
        assert len(r.run_ids) == 3 * len(loads) == len(set(r.run_ids))
        assert r.experiment_ids == [str(EXPERIMENT)]
        assert r.gpus == 1


def test_goodput_and_latency_at_goodput(results):
    r = by_name(results)
    vllm, sglang = r["vllm-bf16"], r["sglang-bf16"]
    assert (vllm.goodput.max_load, vllm.goodput.first_failing_load) == (4.0, 6.0)
    assert (sglang.goodput.max_load, sglang.goodput.first_failing_load) == (6.0, 8.0)
    assert vllm.goodput.bracketed and vllm.goodput.trusted
    at = vllm.goodput_point
    assert at is not None and at.load == 4.0
    assert vllm.ttft_p95_ms == at.aggregate.get("ttft_ms.p95")
    assert vllm.tpot_p95_ms == at.aggregate.get("tpot_ms.p95")
    assert vllm.goodput.output_tok_s.mean == pytest.approx(400, rel=1e-3)
    assert vllm.goodput_output_tok_s_per_gpu.mean == pytest.approx(400, rel=1e-3)
    # the raw peak ignores the SLO: highest load, failing or not
    assert vllm.peak_load == 8.0
    assert vllm.peak_output_tok_s.mean == pytest.approx(800, rel=1e-3)


def test_cost_uses_price_book_and_goodput(results):
    r = by_name(results)
    vllm, sglang, awq = r["vllm-bf16"], r["sglang-bf16"], r["vllm-awq"]
    # every result ranks at the on-demand list price, whatever market it ran on
    assert (vllm.hourly_micros, sglang.hourly_micros, awq.hourly_micros) == (
        ON_DEMAND,
        ON_DEMAND,
        ON_DEMAND,
    )
    assert vllm.cost == cost_from_goodput(ON_DEMAND, vllm.goodput, CostAllocation.all_output())
    # $1.882918/h at 400 output tok/s = $1.3076 per 1M output tokens
    assert vllm.cost.output_per_mtok.value == pytest.approx(1_307_582, rel=1e-3)
    assert vllm.cost.input_per_mtok.value is None  # all_output: input has no price
    assert vllm.cost.input_per_mtok.na_reason == "all cost allocated to output"
    lo, value, hi = (
        vllm.cost.output_per_mtok.lo,
        vllm.cost.output_per_mtok.value,
        vllm.cost.output_per_mtok.hi,
    )
    assert lo < value < hi
    # same price, 1.5x the goodput
    ratio = vllm.cost.output_per_mtok.value / sglang.cost.output_per_mtok.value
    assert ratio == pytest.approx(1.5, rel=1e-3)
    assert vllm.allocation == "all_output"


def test_weighted_allocation_splits_cost(vllm_runs, price_book):
    (r,) = analyze(vllm_runs, price_book, CostAllocation.weighted(4))
    assert r.allocation == "weighted(output_input_ratio=4)"
    assert r.cost.input_per_mtok.value > 0
    assert r.cost.output_per_mtok.value == pytest.approx(4 * r.cost.input_per_mtok.value, abs=4)


def test_clean_results_are_trusted(results):
    for r in results:
        assert r.trusted, r.warnings
        assert r.warnings == []


def test_single_repetition_is_untrusted(price_book):
    (r,) = analyze(make_runs("one", reps=(1,)), price_book)
    assert WarningKind.SINGLE_REPETITION in kinds(r)
    assert not r.trusted
    assert "load 2 req/s: 1 completed repetition" in r.warnings[0].message


def test_unbracketed_goodput_is_untrusted(price_book):
    (r,) = analyze(make_runs("s", ttft=TTFT_SGLANG, loads=(2.0, 4.0)), price_book)
    assert r.goodput.max_load == 4.0 and not r.goodput.bracketed
    assert WarningKind.UNBRACKETED_GOODPUT in kinds(r)
    assert not r.trusted


def test_high_cv_at_goodput_is_untrusted(price_book):
    (r,) = analyze(make_runs("noisy", ttft=TTFT_SGLANG, rep_jitter=0.15), price_book)
    assert WarningKind.HIGH_CV in kinds(r)
    assert not r.trusted


def test_missing_usage_is_untrusted(price_book):
    (r,) = analyze(make_runs("nousage", missing_usage=3), price_book)
    assert WarningKind.MISSING_USAGE in kinds(r)
    (msg,) = [w.message for w in r.warnings if w.kind is WarningKind.MISSING_USAGE]
    assert msg.startswith("36 successful requests had no usage block")
    assert not r.trusted


def test_no_goodput_means_no_cost(price_book):
    (r,) = analyze(make_runs("slow", latency_scale=10), price_book)
    assert r.goodput.max_load is None
    assert r.cost is None
    assert r.ttft_p95_ms is None and r.goodput_point is None
    assert WarningKind.NO_GOODPUT in kinds(r)


def test_failed_runs_are_excluded_and_counted(price_book):
    runs = make_runs("partial")
    runs[0].status = "failed"
    (r,) = analyze(runs, price_book)
    assert r.points[0].aggregate.n_runs == 2
    assert len(r.run_ids) == 11
    assert any(
        w.kind is WarningKind.EXCLUDED_RUNS and w.message.startswith("1 run") for w in r.warnings
    )
    assert WarningKind.EXCLUDED_RUNS not in UNTRUSTING


def test_every_price_column_comes_from_one_resolver(results, price_book):
    r = by_name(results)
    vllm, awq = r["vllm-bf16"], r["vllm-awq"]
    for result in (vllm, awq):
        assert result.prices == default_price_resolver(price_book)(result.provenance)
        assert (result.prices.on_demand, result.prices.spot) == (ON_DEMAND, SPOT)
        assert result.prices.committed_1y is None and result.committed_1y_cost is None
        assert result.prices.storage_gb == 200
        for column in PriceColumn:
            hourly = result.prices.get(column)
            assert result.cost_at(column) == (
                None
                if hourly is None
                else cost_from_goodput(hourly, result.goodput, CostAllocation.all_output())
            )
    # as run: the on-demand host paid list price; the spot host its observed price,
    # never the budget guard's multiplied accrual rate
    assert vllm.prices.as_run == ON_DEMAND
    assert awq.prices.as_run == AS_RUN_SPOT
    assert awq.as_run_cost.output_per_mtok.value < awq.cost.output_per_mtok.value


def test_cloud_run_without_price_basis_is_not_priced(price_book):
    runs = make_runs("legacy", price_basis=None, hourly_micros=2_323_168, reps=(1, 2))
    (legacy,) = analyze(runs, price_book)
    assert legacy.cost is None and legacy.as_run_cost is None
    assert legacy.prices.missing == NO_PRICE_BASIS
    assert any(NO_PRICE_BASIS in w.message for w in legacy.warnings)


def test_local_market_needs_explicit_hourly_price(price_book):
    resolve = default_price_resolver(price_book)
    local = dict(
        market=Market.LOCAL, instance_type=None, cloud=None, region=None, reps=(1, 2), loads=(2.0,)
    )
    (unpriced,) = analyze(make_runs("local", **local), price_book)
    assert unpriced.hourly_micros is None and unpriced.cost is None
    assert WarningKind.NO_PRICE in kinds(unpriced)
    assert unpriced.prices.missing == NO_LOCAL_PRICE

    (priced,) = analyze(make_runs("mock", hourly_micros=500_000, **local), price_book)
    assert priced.hourly_micros == 500_000 and priced.prices.as_run == 500_000
    assert priced.cost is not None and priced.cost.hourly_micros == 500_000
    assert priced.spot_cost is None and priced.prices.storage_gb is None
    assert resolve(priced.provenance).on_demand == 500_000
    basis = {"market": "local", "source": "experiment"}
    assert recorded_hourly_micros({"hourly_micros": 7, "price_basis": basis}) == 7
    assert recorded_hourly_micros({"hourly_micros": 7}) is None
    with pytest.raises(TypeError):
        recorded_hourly_micros({"hourly_micros": 1.5, "price_basis": basis})


def test_price_resolver_refuses_unverified_and_unknown(price_book):
    resolve = default_price_resolver(price_book)
    gcp = {
        "market": "on_demand",
        "cloud": "gcp",
        "region": "us-central1",
        "hardware": {"instance_type": "g2-standard-8"},
        "price_basis": {"market": "on_demand", "source": "prices_yaml", "storage_gb": 0},
    }
    with pytest.raises(UnverifiedPriceError):
        resolve(gcp)
    assert default_price_resolver(price_book, allow_unverified=True)(gcp).on_demand == 853_600
    with pytest.raises(KeyError):
        resolve({**gcp, "cloud": "aws", "region": "us-east-1"})


def test_label_from_provenance():
    prov = make_runs(
        "tp2",
        engine_args={"tensor_parallel_size": 2, "enable_prefix_caching": True},
        reps=(1,),
        loads=(2.0,),
    )[0].provenance
    label = label_from_provenance(prov)
    assert label.tp == 2
    assert label.engine_args == {"enable_prefix_caching": True}
    assert label.text == (
        "vllm 0.30.0 · unquantized · TP2 · 1×L40S · g6e.xlarge (on_demand) · "
        "enable_prefix_caching=true"
    )


def test_provenance_digest_matches_record_hash(results):
    r = results[0]
    assert len(r.provenance_digests) == 12  # repetition and load differ per run
    prov = Provenance.model_validate(r.provenance)
    assert provenance_hash(prov) in r.provenance_digests
    assert r.git_shas == ["0123456789abcdef0123456789abcdef01234567"]
    assert r.reproduce_run_id() == r.goodput_point.run_ids[0]


def test_load_points_without_slo(vllm_runs):
    points = load_points(vllm_runs)
    ((key, pts),) = points.items()
    assert key.workload == "chat" and key.load_mode is LoadMode.OPEN_LOOP
    assert [p.load for p in pts] == [2.0, 4.0, 6.0, 8.0]


def test_quality_for_candidate_and_baseline(results):
    r = by_name(results)
    base, cand = r["vllm-bf16"].config_hash, r["vllm-awq"].config_hash
    evals = [
        eval_row(base, "gsm8k", 0.80),
        eval_row(base, "ifeval", 0.70),
        eval_row(cand, "gsm8k", 0.71),
        eval_row(cand, "ifeval", 0.69),
    ]
    gates = [gate_row(base, cand, "fail")]
    q = quality_for(cand, evals, gates)
    assert q.gate == "fail" and q.baseline_config_hash == base
    assert [t.task for t in q.tasks] == ["gsm8k", "ifeval"]
    assert q.tasks[0].delta == pytest.approx(-0.09)
    assert q.worst().task == "gsm8k"
    assert quality_for(base, evals, gates).gate == "baseline"
    assert quality_for(r["sglang-bf16"].config_hash, evals, gates) is None

    attached = by_name(with_quality(results, evals, gates))
    assert attached["vllm-awq"].quality.gate == "fail"
    assert attached["sglang-bf16"].quality is None


def test_strict_tool_calling_is_listed_after_tool_calling_and_noted(results):
    r = by_name(results)
    base, cand = r["vllm-bf16"].config_hash, r["vllm-awq"].config_hash
    evals = [
        eval_row(base, "tool_calling_strict", 0.95),
        eval_row(base, "tool_calling", 0.45),
        eval_row(base, "json_schema", 0.97),
        eval_row(cand, "tool_calling_strict", 0.90),
        eval_row(cand, "tool_calling", 0.45),
        eval_row(cand, "json_schema", 0.97),
    ]
    q = quality_for(cand, evals, [gate_row(base, cand, "fail")])
    assert [t.task for t in q.tasks] == ["json_schema", "tool_calling", "tool_calling_strict"]
    assert [t.note is not None for t in q.tasks] == [False, False, True]
    assert "tool_calling is the headline" in q.tasks[2].note
    # Gated like any other task: its own delta, so it can be the worst one.
    assert q.worst().task == "tool_calling_strict"
    assert q.worst().delta == pytest.approx(-0.05)


def test_cold_starts_attributed_only_when_unambiguous(results, vllm_runs, price_book):
    solo_exp = uuid.UUID(int=2)
    (solo,) = analyze(
        make_runs("solo", experiment_id=solo_exp, reps=(1, 2), loads=(2.0,)), price_book
    )
    rows = [
        BenchColdStart(experiment_id=solo_exp, kind="cold", stages={}, total_s=t)
        for t in (100.0, 140.0, 120.0)
    ]
    rows.append(BenchColdStart(experiment_id=solo_exp, kind="warm", stages={}, total_s=5.0))
    rows.append(BenchColdStart(experiment_id=EXPERIMENT, kind="cold", stages={}, total_s=90.0))
    stats = cold_starts_by_config(rows, [*results, solo])
    assert list(stats) == [solo.config_hash]
    assert (stats[solo.config_hash].median_s, stats[solo.config_hash].n) == (120.0, 3)


def test_cold_start_uses_recorded_config_hash():
    import uuid

    from loom_bench.report.analyze import cold_starts_by_config
    from loom_bench.store.models import BenchColdStart

    exp = uuid.uuid4()
    rows = [
        BenchColdStart(experiment_id=exp, config_hash="a", kind="cold", stages={}, total_s=10.0),
        BenchColdStart(experiment_id=exp, config_hash="b", kind="cold", stages={}, total_s=30.0),
        BenchColdStart(experiment_id=exp, config_hash="b", kind="warm", stages={}, total_s=1.0),
    ]
    stats = cold_starts_by_config(rows, [])
    assert stats["a"].median_s == 10.0 and stats["b"].median_s == 30.0 and stats["b"].n == 1


def test_label_prefers_recorded_parallelism():
    from loom_bench.report.analyze import label_from_provenance

    label = label_from_provenance({"parallelism": {"tp": 4}, "engine": {"args": {}}, "config": {}})
    assert label.tp == 4
