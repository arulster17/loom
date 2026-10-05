import json
import uuid

import pytest

from loom_bench.cost import CostAllocation
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.compare import (
    DEFAULT_METRICS,
    compare,
    compare_metric,
    render_json,
    render_markdown,
)
from loom_bench.stats import mean_ci

from .factories import SLO, make_runs

RERUN = uuid.UUID(int=99)


def analyze(runs, price_book):
    return analyze_runs(
        runs,
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


def test_identical_runs_are_within_normal_variance(vllm_runs):
    c = compare(vllm_runs, vllm_runs)
    assert c.within_normal_variance
    assert len(c.points) == 4 and not c.configs
    assert all(p.config_hash_match for p in c.points)
    for p in c.points:
        assert [d.metric for d in p.metrics] == list(DEFAULT_METRICS)
        assert all(d.delta == 0 and d.within_normal_variance for d in p.metrics)


def test_rerun_with_small_jitter_is_within_variance(vllm_runs, price_book):
    rerun = make_runs("vllm-bf16", experiment_id=RERUN, latency_scale=1.03)
    a, b = analyze(vllm_runs, price_book), analyze(rerun, price_book)
    c = compare(a, b, label_a="original", label_b="reproduction")
    assert c.within_normal_variance
    (cfg,) = c.configs
    assert cfg.goodput_load_a == cfg.goodput_load_b == 4.0
    assert {d.metric for d in cfg.cost} == {
        "cost.output_per_mtok",
        "cost.input_per_mtok",
        "cost.total_per_mtok",
    }
    p95 = next(d for d in c.points[0].metrics if d.metric == "ttft_ms.p95")
    assert p95.rel_delta == pytest.approx(0.03, abs=1e-9)
    assert p95.reason.startswith("within tolerance (|Δ| 3.0% vs tolerance 10%)")


def test_shifted_latency_is_flagged(vllm_runs, price_book):
    shifted = make_runs("vllm-bf16", experiment_id=RERUN, latency_scale=1.8)
    c = compare(analyze(vllm_runs, price_book), analyze(shifted, price_book))
    assert not c.within_normal_variance
    point = c.points[0]
    flagged = {d.metric for d in point.metrics if not d.within_normal_variance}
    assert {"ttft_ms.p95", "tpot_ms.p95", "e2e_ms.p95"} <= flagged
    assert "throughput.output_tok_s" not in flagged
    p95 = next(d for d in point.metrics if d.metric == "ttft_ms.p95")
    assert p95.p_value < 0.05
    assert p95.delta_lo > 0  # the CI of the difference excludes zero
    assert "significant (Welch p=" in p95.reason
    # slower latency moves goodput from 4 to 2 req/s, halving throughput: cost doubles
    (cfg,) = c.configs
    assert (cfg.goodput_load_a, cfg.goodput_load_b) == (4.0, 2.0)
    cost = next(d for d in cfg.cost if d.metric == "cost.output_per_mtok")
    assert not cost.within_normal_variance
    assert cost.rel_delta == pytest.approx(1.0, rel=0.01)


def test_large_but_noisy_difference_is_not_significant():
    a = mean_ci([100, 140, 60])
    b = mean_ci([130, 170, 90])
    d = compare_metric("ttft_ms.p95", a, b, rel_tol=0.10, abs_tol={}, alpha=0.05, confidence=0.95)
    assert d.rel_delta == pytest.approx(0.3)
    assert d.within_normal_variance
    assert "not significant" in d.reason


def test_single_repetition_outside_tolerance_is_flagged():
    d = compare_metric(
        "ttft_ms.p95",
        mean_ci([100]),
        mean_ci([120]),
        rel_tol=0.10,
        abs_tol={},
        alpha=0.05,
        confidence=0.95,
    )
    assert not d.within_normal_variance
    assert d.p_value is None and "no test" in d.reason


def test_error_rate_uses_absolute_tolerance():
    d = compare_metric(
        "error_rate",
        mean_ci([0.0, 0.0]),
        mean_ci([0.005, 0.005]),
        rel_tol=0.10,
        abs_tol={"error_rate": 0.01},
        alpha=0.05,
        confidence=0.95,
    )
    assert d.within_normal_variance


def test_cell_key_match_requires_identical_config_hash(vllm_runs):
    changed = make_runs("vllm-bf16", experiment_id=RERUN, engine_args={"max_num_seqs": 128})
    c = compare(vllm_runs, changed, match_by="cell_key")
    assert len(c.points) == 4
    assert not any(p.config_hash_match for p in c.points)
    assert not c.within_normal_variance
    assert all(d.within_normal_variance for p in c.points for d in p.metrics)
    assert not compare(vllm_runs, changed).points  # by config hash nothing matches


def test_two_configs_by_workload(vllm_runs, sglang_runs, price_book):
    c = compare(
        analyze(vllm_runs, price_book), analyze(sglang_runs, price_book), match_by="workload"
    )
    (cfg,) = c.configs
    assert (cfg.name_a, cfg.name_b) == ("vllm-bf16", "sglang-bf16")
    assert not cfg.config_hash_match
    cost = next(d for d in cfg.cost if d.metric == "cost.output_per_mtok")
    assert cost.rel_delta == pytest.approx(-1 / 3, rel=0.01)
    assert not cost.within_normal_variance
    with pytest.raises(ValueError, match="more than one sweep"):
        compare([*vllm_runs, *sglang_runs], vllm_runs, match_by="workload")


def test_unmatched_points_are_reported(vllm_runs):
    partial = make_runs("vllm-bf16", experiment_id=RERUN, loads=(2.0, 4.0))
    c = compare(vllm_runs, partial)
    assert len(c.points) == 2
    assert c.only_in_a == ["vllm-bf16 chat @ 6 req/s", "vllm-bf16 chat @ 8 req/s"]
    assert not c.within_normal_variance


def test_mixed_or_empty_input_is_rejected(vllm_runs, price_book):
    with pytest.raises(ValueError, match="empty"):
        compare([], vllm_runs)
    mixed = [*analyze(vllm_runs, price_book), vllm_runs[0]]
    with pytest.raises(TypeError):
        compare(mixed, vllm_runs)


def test_render_markdown_and_json(vllm_runs, price_book):
    shifted = make_runs("vllm-bf16", experiment_id=RERUN, latency_scale=1.5)
    c = compare(
        analyze(vllm_runs, price_book),
        analyze(shifted, price_book),
        label_a="run 1",
        label_b="run 2",
    )
    md = render_markdown(c)
    assert md.startswith("# Comparison: run 1 vs run 2\n\n**Verdict: outside normal variance**")
    assert "| Metric | A | B | Δ = B − A [CI] (rel) | Welch p | Verdict | Reason |" in md
    assert "## vllm-bf16 vs vllm-bf16: chat @ 2 req/s — OUTSIDE" in md
    assert "Config hashes must match exactly." in md
    assert "| cost.output_per_mtok |" in md
    doc = json.loads(render_json(c))
    assert doc["within_normal_variance"] is False
    assert doc["match_by"] == "config_hash" and doc["rel_tol"] == 0.1
    assert len(doc["points"]) == 4
    assert (
        render_markdown(compare(vllm_runs, vllm_runs)).count("**Verdict: within normal variance**")
        == 1
    )
