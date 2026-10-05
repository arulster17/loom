import csv
import io

import pytest

from loom_bench.cost import CostAllocation
from loom_bench.registry import Pricing, Registry, load_registry
from loom_bench.report.analyze import analyze_runs, default_price_resolver
from loom_bench.report.competitiveness import (
    CSV_COLUMNS,
    build_competitiveness,
    render_csv,
    render_html,
    render_markdown,
)

from .factories import SLO, assert_self_contained


@pytest.fixture(scope="module")
def results(vllm_runs, sglang_runs, price_book):
    return analyze_runs(
        [*vllm_runs, *sglang_runs],
        slo=SLO,
        allocation=CostAllocation.all_output(),
        price_resolver=default_price_resolver(price_book),
    )


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


def test_rows_use_best_ranked_config_and_margins(report):
    qwen, llama = report.rows
    assert (qwen.model_id, qwen.workload, qwen.best_config) == ("qwen3-8b", "chat", "sglang-bf16")
    assert qwen.cost_output.value == pytest.approx(871_692, rel=1e-3)
    assert qwen.margin_output.value == 500_000 - qwen.cost_output.value
    assert qwen.margin_output.worst == 500_000 - qwen.cost_output.hi
    # all_output allocation: input has no cost of its own, so no input margin
    assert qwen.margin_input is None
    assert qwen.cost_input.na_reason == "all cost allocated to output"
    assert (llama.model_id, llama.workload, llama.best_config, llama.price) == (
        "llama-3.3-70b-instruct",
        None,
        None,
        None,
    )


def test_flags_from_assess(report):
    qwen, llama = report.rows
    assert [(f.kind, f.side) for f in qwen.flags] == [
        ("price_above_market", "output"),
        ("negative_margin", "output"),
    ]
    assert [f.kind for f in llama.flags] == ["no_price_set", "no_cost_measurement"]


def test_default_scope_excludes_aggregators_and_unverified(results, registry, competitors):
    qwen = build_competitiveness(results, registry, competitors).rows[0]
    assert [f.kind for f in qwen.flags] == ["no_public_comparison", "negative_margin"]
    assert {(c.provider, c.in_comparison) for c in qwen.competitors} == {
        ("Fireworks AI", False),
        ("OpenRouter", False),
    }


def test_markdown_states_public_list_prices_only(report):
    md = render_markdown(report)
    assert "> Public list prices only" in md
    assert (
        "| chat (open loop) | sglang-bf16 | n/a (all cost allocated to output) "
        "| $0.8717 [0.8503, 0.8936] | $0.0500 | $0.5000 | n/a (all cost allocated to output) |"
        in md
    )
    assert "$0.0000" not in md
    assert "| OpenRouter (aggregator) | qwen/qwen3-8b | $0.1170 | $0.4550 |" in md
    assert "https://openrouter.ai/qwen/qwen3-8b | 2026-10-04 |" in md
    assert "`price_above_market`: qwen3-8b output: $0.5000/1M is $0.0450 (9.9%) above" in md
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
    assert first["price_output_per_mtok_usd"] == "$0.500000"
    assert first["flags"] == "price_above_market(output); negative_margin(output)"
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
    assert '<a href="https://openrouter.ai/qwen/qwen3-8b">' in html
