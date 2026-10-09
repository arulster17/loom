from fractions import Fraction
from typing import Any

from loom_bench.competitiveness import CostPerMtok, Flag, FlagKind, assess, price_at_mix
from loom_bench.prices import Competitors, load_competitors
from loom_bench.registry import Pricing

URL = "https://example.com/pricing"


def provider(name: str, inp: int, out: int, **kw: Any) -> dict[str, Any]:
    availability = kw.pop("availability", "listed")
    return {
        "name": name,
        "pricing_url": URL,
        "last_checked": "2026-10-04",
        "entries": [
            {
                "model_id": "m",
                "input_per_mtok": inp,
                "output_per_mtok": out,
                "availability": availability,
                "source": URL,
            }
        ],
        **kw,
    }


def market(*providers: dict[str, Any]) -> Competitors:
    return Competitors.model_validate({"last_checked": "2026-10-04", "providers": providers})


MARKET = market(
    provider("A", 100_000, 300_000),
    provider("B", 120_000, 320_000),
    provider("C", 150_000, 400_000),
    provider("Agg", 50_000, 100_000, aggregator=True),
    provider("Unv", 60_000, 60_000, availability="unverified"),
)
COST = CostPerMtok(input=10_000, output=30_000)


def price(inp: int, out: int) -> Pricing:
    return Pricing(input_per_mtok=inp, output_per_mtok=out, cached_input_per_mtok=0)


def kinds(flags: list[Flag]) -> list[tuple[FlagKind, str | None]]:
    return [(f.kind, f.side) for f in flags]


def test_price_above_market_reports_delta_provider_and_median():
    flags = assess("m", price(130_000, 300_000), COST, MARKET)
    assert kinds(flags) == [(FlagKind.PRICE_ABOVE_MARKET, "input")]
    (flag,) = flags
    assert (flag.ours, flag.reference, flag.delta) == (130_000, 100_000, 30_000)
    assert flag.providers == ("A",)
    assert flag.median == 120_000
    assert "$0.0300 (30.0%) above the market minimum $0.1000/1M (A)" in flag.message
    assert "$0.0100 (8.3%) vs the median $0.1200/1M of 3 providers" in flag.message


def test_at_or_below_market_minimum_is_not_flagged():
    assert assess("m", price(100_000, 299_999), COST, MARKET) == []


def test_aggregators_and_unverified_excluded_by_default():
    ours = price(90_000, 250_000)
    assert assess("m", ours, COST, MARKET) == []

    with_agg = assess("m", ours, COST, MARKET, include_aggregators=True)
    assert kinds(with_agg) == [
        (FlagKind.PRICE_ABOVE_MARKET, "input"),
        (FlagKind.PRICE_ABOVE_MARKET, "output"),
    ]
    assert {f.providers for f in with_agg} == {("Agg",)}

    with_unv = assess("m", ours, COST, MARKET, include_unverified=True)
    assert [(f.providers, f.reference) for f in with_unv] == [
        (("Unv",), 60_000),
        (("Unv",), 60_000),
    ]


def test_ties_at_minimum_name_every_provider():
    comp = market(provider("X", 100_000, 1), provider("W", 100_000, 1))
    (flag, _) = assess("m", price(100_001, 2), None, comp)[1:]
    assert flag.providers == ("W", "X")


def test_provider_with_several_entries_counts_once_at_its_cheapest():
    two = provider("X", 300_000, 1)
    two["entries"].append({**two["entries"][0], "input_per_mtok": 200_000})
    comp = market(two, provider("Y", 400_000, 1))
    flag = above_market(assess("m", price(500_000, 1), COST, comp))
    assert (flag.reference, flag.providers, flag.median) == (200_000, ("X",), 300_000)
    assert "of 2 providers" in flag.message


def test_even_median_rounds_half_up_to_the_micro():
    comp = market(provider("X", 100_000, 1), provider("Y", 100_001, 1))
    flag = above_market(assess("m", price(200_000, 1), COST, comp))
    assert flag.median == 100_001  # 100_000.5 rounds half up


def above_market(flags: list[Flag]) -> Flag:
    return next(f for f in flags if f.kind is FlagKind.PRICE_ABOVE_MARKET)


def test_negative_margin():
    flags = assess("m", price(100_000, 300_000), CostPerMtok(110_000, 200_000), MARKET)
    # the input cost is also above the cheapest list price (A, $0.10)
    assert kinds(flags) == [
        (FlagKind.COST_ABOVE_MARKET, "input"),
        (FlagKind.NEGATIVE_MARGIN, "input"),
    ]
    flag = flags[1]
    assert (flag.ours, flag.reference, flag.delta) == (100_000, 110_000, -10_000)
    assert "$0.0100 below measured cost $0.1100/1M" in flag.message


def test_price_equal_to_cost_is_not_negative_margin():
    assert assess("m", price(100_000, 300_000), CostPerMtok(100_000, 300_000), MARKET) == []


def test_missing_inputs():
    flags = assess("m", None, None, MARKET)
    assert kinds(flags) == [(FlagKind.NO_PRICE_SET, None), (FlagKind.NO_COST_MEASUREMENT, None)]

    only_agg = market(provider("Agg", 1, 1, aggregator=True))
    flags = assess("m", price(500_000, 500_000), COST, only_agg)
    assert kinds(flags) == [(FlagKind.NO_PUBLIC_COMPARISON, None)]

    flags = assess("other", price(1, 1), COST, MARKET)
    assert kinds(flags) == [
        (FlagKind.NO_PUBLIC_COMPARISON, None),
        (FlagKind.NEGATIVE_MARGIN, "input"),
        (FlagKind.NEGATIVE_MARGIN, "output"),
    ]


def test_real_competitors_file():
    comp = load_competitors()
    qwen = assess("qwen3-8b", None, None, comp)
    # Fireworks' bucket price is unverified and OpenRouter is an aggregator.
    assert [f.kind for f in qwen] == [
        FlagKind.NO_PRICE_SET,
        FlagKind.NO_COST_MEASUREMENT,
        FlagKind.NO_PUBLIC_COMPARISON,
    ]
    llama = assess("llama-3.3-70b-instruct", price(110_000, 320_000), None, comp)
    assert kinds(llama) == [
        (FlagKind.NO_COST_MEASUREMENT, None),
        (FlagKind.PRICE_ABOVE_MARKET, "input"),
    ]
    assert llama[1].providers == ("DeepInfra",)
    assert llama[1].median == 135_000  # DeepInfra 0.10, Novita 0.135, Together 1.04


def test_side_without_allocated_cost_gets_no_margin_check():
    # all_output: input has no cost of its own; a low input price is not a negative margin
    flags = assess("m", price(1, 300_000), CostPerMtok(input=None, output=310_000), MARKET)
    assert kinds(flags) == [
        (FlagKind.COST_ABOVE_MARKET, "output"),
        (FlagKind.NEGATIVE_MARGIN, "output"),
    ]


def test_cost_above_market_per_side_and_at_the_mix():
    # A lists 0.10 / 0.30. Our input cost 0.05 is below it, output 0.40 above; at a mix
    # of 3 input : 1 output A costs 0.75 x 0.10 + 0.25 x 0.30 = 0.15 per 1M tokens.
    cost = CostPerMtok(input=50_000, output=400_000, blended=160_000, input_share=Fraction(3, 4))
    flags = assess("m", None, cost, MARKET)
    assert kinds(flags) == [
        (FlagKind.NO_PRICE_SET, None),
        (FlagKind.COST_ABOVE_MARKET, "output"),
        (FlagKind.COST_ABOVE_MARKET, "blended"),
    ]
    out, mix = flags[1], flags[2]
    assert (out.reference, out.providers, out.delta) == (300_000, ("A",), 100_000)
    assert out.median == 320_000  # A 0.30, B 0.32, C 0.40; Agg and Unv left out
    assert (mix.reference, mix.providers, mix.ours) == (150_000, ("A",), 160_000)
    assert "above the lowest public list price at the workload's token mix" in mix.message
    # at or below the market on every side: no cost flag
    cheap = CostPerMtok(input=1, output=1, blended=1, input_share=Fraction(1, 2))
    assert kinds(assess("m", None, cheap, MARKET)) == [(FlagKind.NO_PRICE_SET, None)]
    # aggregators widen the market when asked
    with_agg = assess("m", None, cost, MARKET, include_aggregators=True)
    assert [(f.side, f.providers) for f in with_agg[1:]] == [
        ("output", ("Agg",)),
        ("blended", ("Agg",)),
    ]


def test_price_at_mix_rounds_once_half_up():
    assert price_at_mix(100_000, 300_000, Fraction(3, 4)) == 150_000
    assert price_at_mix(1, 2, Fraction(1, 2)) == 2  # 1.5 rounds up
    assert price_at_mix(100_000, 300_000, Fraction(1)) == 100_000
