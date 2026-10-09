import csv
import io
from fractions import Fraction

import pytest

from loom_bench.competitiveness import price_at_mix
from loom_bench.cost import NA_PREFILL_SATURATED, CostAllocation
from loom_bench.prices import Competitors
from loom_bench.registry import Pricing, Registry, load_registry
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.competitiveness import (
    CSV_COLUMNS,
    build_competitiveness,
    render_csv,
    render_html,
    render_markdown,
)
from loom_bench.slo import Slo

from .factories import SLO, TTFT_SGLANG, assert_self_contained, make_runs

# Prefill-light latencies: requests in prefill (rate x mean TTFT) stay well below 1 at
# goodput, so the measured input/output split applies. sglang passes up to 6 req/s and
# vllm up to 4, as with the default factory latencies.
TTFT_FAST_SGLANG = {2.0: 40.0, 4.0: 50.0, 6.0: 60.0, 8.0: 800.0}
TTFT_FAST_VLLM = {2.0: 40.0, 4.0: 50.0, 6.0: 700.0, 8.0: 900.0}


def _analyze(runs, price_book, slo=SLO):
    return analyze_runs(
        runs,
        slo=slo,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


@pytest.fixture(scope="module")
def results(price_book):
    runs = [
        *make_runs("vllm-bf16", ttft=TTFT_FAST_VLLM),
        *make_runs("sglang-bf16", engine="sglang", ttft=TTFT_FAST_SGLANG),
    ]
    return _analyze(runs, price_book)


@pytest.fixture(scope="module")
def registry():
    base = load_registry()
    qwen = base.get("qwen3-8b").model_copy(
        update={
            "pricing": Pricing(
                input_per_mtok=50_000, output_per_mtok=500_000, cached_input_per_mtok=25_000
            )
        }
    )
    return Registry(models=[qwen, base.get("llama-3.3-70b-instruct")])


@pytest.fixture(scope="module")
def report(results, registry, competitors, price_book):
    return build_competitiveness(
        results, registry, competitors, price_book=price_book, include_aggregators=True
    )


def test_rows_quote_the_headline_config_at_the_measured_split(report, results):
    qwen, llama = report.rows
    assert (qwen.model_id, qwen.workload, qwen.best_config) == ("qwen3-8b", "chat", "sglang-bf16")
    sglang = next(r for r in results if r.name == "sglang-bf16")
    split = sglang.split_cost
    assert qwen.cost_input == split.input_per_mtok and qwen.cost_output == split.output_per_mtok
    assert qwen.cost_blended == split.total_per_mtok
    assert qwen.cost_input.na_reason is None and qwen.cost_output.na_reason is None
    # 200 prompt and 100 output tokens per request
    assert (qwen.tokens_in, qwen.tokens_out) == pytest.approx((200, 100))
    assert qwen.input_share == pytest.approx(Fraction(2, 3))
    # both sides now have a cost of their own, so both have a margin
    assert qwen.margin_input.value == 50_000 - qwen.cost_input.value
    assert qwen.margin_output.value == 500_000 - qwen.cost_output.value
    assert qwen.margin_output.worst == 500_000 - qwen.cost_output.hi
    assert qwen.basis == "leaderboard rank 1"
    assert (llama.model_id, llama.workload, llama.best_config, llama.price) == (
        "llama-3.3-70b-instruct",
        None,
        None,
        None,
    )
    assert llama.basis == "not benchmarked"


def test_flags_compare_cost_and_price_with_the_market(report):
    qwen, llama = report.rows
    # with aggregators: OpenRouter $0.117 in / $0.455 out is the qwen3-8b market
    assert [(f.kind, f.side) for f in qwen.flags] == [
        ("cost_above_market", "input"),
        ("cost_above_market", "output"),
        ("cost_above_market", "blended"),
        ("price_above_market", "output"),
        ("negative_margin", "input"),
        ("negative_margin", "output"),
    ]
    assert [f.kind for f in llama.flags] == ["no_price_set", "no_cost_measurement"]


def test_default_scope_excludes_aggregators_and_unverified(results, registry, competitors):
    qwen = build_competitiveness(results, registry, competitors).rows[0]
    assert [(f.kind, f.side) for f in qwen.flags] == [
        ("no_public_comparison", None),
        ("negative_margin", "input"),
        ("negative_margin", "output"),
    ]
    assert {(c.provider, c.in_comparison) for c in qwen.competitors} == {
        ("Fireworks AI", False),
        ("OpenRouter", False),
    }


def test_public_prices_are_shown_at_the_workload_mix(report):
    qwen = report.rows[0]
    openrouter = next(c for c in qwen.competitors if c.provider == "OpenRouter")
    # 2/3 input: 2/3 x 0.117 + 1/3 x 0.455 = 0.229667
    assert openrouter.blended_at_mix == price_at_mix(117_000, 455_000, Fraction(2, 3)) == 229_667


def test_markdown_states_public_list_prices_only(report):
    md = render_markdown(report)
    qwen = report.rows[0]
    assert "> Public list prices only" in md
    assert "| chat (open loop) | sglang-bf16 (leaderboard rank 1) | 200 / 100 |" in md
    assert "At chat mix (200 / 100)" in md
    assert "| $0.0500 | $0.5000 |" in md
    assert f"${qwen.cost_input.value / 1e6:.4f} (CI high ${qwen.cost_input.hi / 1e6:.4f})" in md
    assert "$0.0000" not in md
    assert "| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 | not disclosed |" in md
    assert "$0.2297 (our cost " in md
    assert "https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |" in md
    assert "`price_above_market`: qwen3-8b output: $0.5000/1M is $0.0450 (9.9%) above" in md
    assert "`cost_above_market`: qwen3-8b blended: our cost at SLO" in md
    assert "`negative_margin`: qwen3-8b output: price $0.5000/1M is" in md
    assert "[not benchmarked] `no_cost_measurement`" in md
    assert "- Groq: Enterprise plans only; contact sales. (checked 2026-10-04)" in md
    assert "## Methodology and provenance" in md
    assert "bench reproduce " in md


def test_csv_long_format(report):
    rows = list(csv.DictReader(io.StringIO(render_csv(report))))
    assert list(rows[0]) == CSV_COLUMNS
    qwen = [r for r in rows if r["model_id"] == "qwen3-8b"]
    assert sorted(r["competitor"] for r in qwen) == ["Fireworks AI", "OpenRouter"]
    assert all(r["public_list_prices_only"] == "True" for r in rows)
    first = qwen[0]
    assert int(first["cost_output_per_mtok_micros"]) == report.rows[0].cost_output.value
    assert int(first["cost_blended_per_mtok_micros"]) == report.rows[0].cost_blended.value
    assert first["price_output_per_mtok_usd"] == "$0.500000"
    assert first["flags"].startswith("cost_above_market(input); cost_above_market(output)")
    assert first["tokens_in_per_request"] == "200.0" and first["alternative_slo"] == "False"
    openrouter = next(r for r in qwen if r["competitor"] == "OpenRouter")
    assert openrouter["competitor_blended_at_mix_micros"] == "229667"
    ratio = report.rows[0].cost_blended.value / 229_667
    assert float(openrouter["our_blended_cost_vs_competitor"]) == pytest.approx(ratio, abs=1e-4)
    assert first["price_basis"] == (
        "aws/us-east-1 g6e.xlarge + 200 GB block storage, from the price book: "
        "on-demand $1.8829/h; spot $1.8605/h; committed 1y n/a"
    )
    llama = [r for r in rows if r["model_id"] == "llama-3.3-70b-instruct"]
    assert len(llama) == 5 and all(r["best_config"] == "" for r in llama)


def test_html_self_contained_with_source_links(report, competitors):
    html = render_html(report)
    sources = tuple(str(e.source) for p in competitors.providers for e in p.entries)
    allowed = (
        *sources,
        "https://b0.p.awsstatic.com/pricing/",
        "https://aws.amazon.com/ebs/pricing/",
        "https://instances.vantage.sh/",
        "https://huggingface.co/datasets/",
    )
    assert_self_contained(html, allowed)
    assert "Public list prices only" in html
    assert '<li class="price_above_market">' in html
    assert '<li class="cost_above_market">' in html
    assert '<a href="https://openrouter.ai/qwen/qwen3-8b">' in html
    assert "Break-even" in html


def test_benchmarked_only_drops_models_without_results(results, registry, competitors):
    rows = build_competitiveness(results, registry, competitors, benchmarked_only=True).rows
    assert [r.model_id for r in rows] == ["qwen3-8b"]


def test_untrusted_config_is_quoted_with_its_reason(price_book, registry, competitors):
    single = make_runs("sglang-1rep", engine="sglang", ttft=TTFT_FAST_SGLANG, reps=(1,))
    (row, _) = build_competitiveness(_analyze(single, price_book), registry, competitors).rows
    assert row.best_config == "sglang-1rep"
    assert row.basis.startswith("untrusted, not ranked: ")
    assert "single repetition" in row.basis
    assert row.cost_blended is not None


def test_no_cost_at_slo_adds_a_labelled_alternative_slo_row(price_book, registry, competitors):
    slow = make_runs("slow", ttft=TTFT_FAST_SGLANG, latency_scale=10)
    loose = Slo(ttft_ms={"p95": 100_000}, tpot_ms={"p95": 1_000}, max_error_rate=0.01)
    main, alt = _analyze(slow, price_book), _analyze(slow, price_book, loose)
    rows = build_competitiveness(main, registry, competitors, alt_results=alt).rows
    declared, alternative = rows[0], rows[1]
    assert declared.best_config is None and declared.basis == "no tested load met the SLO"
    assert declared.cost_blended is None and not declared.alternative_slo
    assert alternative.alternative_slo and alternative.best_config == "slow"
    assert alternative.basis.startswith("alternative SLO TTFT p95 ≤ 100000 ms, TPOT p95 ≤ 1000")
    assert alternative.cost_blended.value is not None
    md = render_markdown(build_competitiveness(main, registry, competitors, alt_results=alt))
    assert "chat (open loop) at the ALTERNATIVE SLO, not the declared one" in md
    # without an alternative analysis there is only the declared row
    assert len(build_competitiveness(main, registry, competitors).rows) == 2  # + llama


FP8_REPO = "RedHatAI/Qwen3-8B-FP8-dynamic"


def _listing(name: str, quantization: str | None) -> dict:
    url = f"https://{name.lower()}.example/pricing"
    entry = {"model_id": "qwen3-8b", "input_per_mtok": 50_000, "output_per_mtok": 150_000}
    if quantization is not None:
        entry["quantization"] = quantization
    return {
        "name": name,
        "pricing_url": url,
        "last_checked": "2026-10-04",
        "entries": [entry | {"source": url}],
    }


QWEN_MARKET = Competitors.model_validate(
    {
        "last_checked": "2026-10-04",
        "providers": [_listing("Eight", "fp8"), _listing("Full", "none"), _listing("Quiet", None)],
    }
)


def test_quantized_entry_is_compared_with_its_base_models_list_prices(price_book):
    base = load_registry()
    registry = Registry(models=[base.get("qwen3-8b"), base.get("qwen3-8b-fp8")])
    fp8 = make_runs("vllm-fp8", quantization="fp8", ttft=TTFT_FAST_VLLM, repo=FP8_REPO)
    report = build_competitiveness(
        _analyze(fp8, price_book), registry, QWEN_MARKET, benchmarked_only=True
    )
    (row,) = report.rows
    assert row.model_id == "qwen3-8b-fp8" and row.market_model_id == "qwen3-8b"
    assert [c.provider for c in row.competitors] == ["Eight", "Full", "Quiet"]
    assert "no_public_comparison" not in [f.kind for f in row.flags]
    md = render_markdown(report)
    assert "No public list price recorded" not in md
    assert "Public list prices are those of qwen3-8b, the model this entry serves" in md
    assert "fp8 (same precision as ours: like for like)" in md
    assert "none (ours fp8: not like for like)" in md
    assert "| not disclosed |" in md
    assert "Public list prices are those of qwen3-8b" in render_html(report)
    (line,) = [r for r in csv.DictReader(io.StringIO(render_csv(report)))][:1]
    assert line["model_id"] == "qwen3-8b-fp8"
    assert line["competitor_listing_model_id"] == "qwen3-8b"


def test_unquantized_entry_marks_a_quantized_listing_not_like_for_like(price_book):
    base = load_registry()
    registry = Registry(models=[base.get("qwen3-8b"), base.get("qwen3-8b-fp8")])
    bf16 = make_runs("vllm-bf16", ttft=TTFT_FAST_VLLM)
    report = build_competitiveness(
        _analyze(bf16, price_book), registry, QWEN_MARKET, benchmarked_only=True
    )
    (row,) = report.rows
    assert row.model_id == "qwen3-8b" and row.market_model_id is None
    md = render_markdown(report)
    assert "fp8 (ours unquantized: not like for like)" in md
    assert "none (same precision as ours: like for like)" in md
    assert "Public list prices are those of" not in md


def test_overlapping_prefills_leave_no_split_but_blended_is_compared(
    price_book, registry, competitors
):
    # The default factory latencies put more than one request in prefill at goodput
    # (6 req/s x ~0.35 s TTFT): the split does not apply, the blended cost still does.
    runs = make_runs("sglang-bf16", engine="sglang", ttft=TTFT_SGLANG)
    qwen = build_competitiveness(
        _analyze(runs, price_book), registry, competitors, include_aggregators=True
    ).rows[0]
    assert qwen.cost_input.na_reason == qwen.cost_output.na_reason == NA_PREFILL_SATURATED
    assert qwen.cost_blended.value is not None
    kinds = [(f.kind, f.side) for f in qwen.flags]
    assert ("cost_above_market", "blended") in kinds
    assert "no_cost_measurement" not in [k for k, _ in kinds]
    assert qwen.margin_input is None and qwen.margin_output is None
