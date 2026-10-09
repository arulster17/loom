"""Competitiveness check: our per-token price against public list prices and against
our own measured cost at SLO. Pure; all money is integer micro-dollars per 1M tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from statistics import median_high, median_low
from typing import Literal

from loom_bench.money import Micros, format_usd, round_half_up
from loom_bench.prices import Competitors
from loom_bench.registry import Pricing

Side = Literal["input", "output"]
SIDES: tuple[Side, ...] = ("input", "output")
# A flag's side: a price side, or "blended" (the workload's own token mix).
FlagSide = Literal["input", "output", "blended"]


class FlagKind(StrEnum):
    COST_ABOVE_MARKET = "cost_above_market"
    PRICE_ABOVE_MARKET = "price_above_market"
    NEGATIVE_MARGIN = "negative_margin"
    NO_PRICE_SET = "no_price_set"
    NO_COST_MEASUREMENT = "no_cost_measurement"
    NO_PUBLIC_COMPARISON = "no_public_comparison"


@dataclass(frozen=True, slots=True)
class CostPerMtok:
    """Our measured serving cost at SLO, micro-dollars per 1M tokens.

    A side is None when it has no cost of its own: the cost allocation charges the
    whole replica to the other side (input under all_output). No margin is checked
    on such a side.
    """

    input: Micros | None
    output: Micros | None
    # Cost per 1M tokens of the workload's own mix, and the input tokens' share of that
    # mix: compared with each public price taken at the same mix.
    blended: Micros | None = None
    input_share: Fraction | None = None


def like_for_like(theirs: str, ours: str) -> str:
    """Whether a competitor's disclosed precision matches ours (registry quantization
    values; "none" is unquantized)."""
    if theirs == ours:
        return "same precision as ours: like for like"
    mine = "unquantized" if ours == "none" else ours
    return f"ours {mine}: not like for like"


def price_at_mix(input_per_mtok: Micros, output_per_mtok: Micros, input_share: Fraction) -> Micros:
    """A per-side price list as one price per 1M tokens of a mix with this input share."""
    return round_half_up(input_per_mtok * input_share + output_per_mtok * (1 - input_share))


@dataclass(frozen=True, slots=True)
class Flag:
    kind: FlagKind
    message: str
    side: FlagSide | None = None
    ours: Micros | None = None
    # price_above_market and cost_above_market: lowest eligible public price;
    # negative_margin: our cost.
    reference: Micros | None = None
    delta: Micros | None = None  # ours - reference
    providers: tuple[str, ...] = ()  # providers at the reference price
    median: Micros | None = None  # median eligible public price (*_above_market only)


def _median(values: list[Micros]) -> Micros:
    return round_half_up(Fraction(median_low(values) + median_high(values), 2))


def _pct(delta: Micros, reference: Micros) -> str:
    if reference == 0:
        return "n/a"
    return f"{float(Fraction(delta * 100, reference)):.1f}%"


def assess(
    model_id: str,
    our_price: Pricing | None,
    our_cost_per_mtok: CostPerMtok | None,
    competitors: Competitors,
    *,
    include_aggregators: bool = False,
    include_unverified: bool = False,
    market_model_id: str | None = None,
) -> list[Flag]:
    """Flag pricing problems for one model.

    Aggregators and entries whose availability is unverified are left out of the
    market comparison by default: they are not a firm price for this model.
    `market_model_id` names the registry entry whose list prices apply (a quantized
    entry's base model, `Registry.market_model_id`); by default `model_id` itself.
    """
    flags: list[Flag] = []
    market = [
        (provider.name, entry)
        for provider, entry in competitors.entries_for(market_model_id or model_id)
        if (include_aggregators or not provider.aggregator)
        and (include_unverified or entry.availability == "listed")
    ]

    if our_price is None:
        flags.append(Flag(FlagKind.NO_PRICE_SET, f"{model_id}: no price set"))
    if our_cost_per_mtok is None:
        flags.append(Flag(FlagKind.NO_COST_MEASUREMENT, f"{model_id}: no measured cost at SLO"))
    if not market:
        flags.append(
            Flag(FlagKind.NO_PUBLIC_COMPARISON, f"{model_id}: no eligible public list price")
        )

    def market_prices(side: FlagSide) -> dict[str, Micros]:
        """Provider -> its cheapest entry for this model on `side`."""
        prices: dict[str, Micros] = {}
        share = our_cost_per_mtok.input_share if our_cost_per_mtok else None
        for name, entry in market:
            if side == "blended":
                assert share is not None
                p = price_at_mix(entry.input_per_mtok, entry.output_per_mtok, share)
            else:
                p = getattr(entry, f"{side}_per_mtok")
            prices[name] = min(prices.get(name, p), p)
        return prices

    if our_cost_per_mtok is not None and market:
        cost_sides: list[tuple[FlagSide, Micros | None]] = [
            ("input", our_cost_per_mtok.input),
            ("output", our_cost_per_mtok.output),
        ]
        if our_cost_per_mtok.input_share is not None:
            cost_sides.append(("blended", our_cost_per_mtok.blended))
        for cost_side, cost_value in cost_sides:
            if cost_value is None:
                continue
            prices = market_prices(cost_side)
            lowest = min(prices.values())
            if cost_value <= lowest:
                continue
            at_min = tuple(sorted(name for name, p in prices.items() if p == lowest))
            median = _median(sorted(prices.values()))
            what = "list price" + (" at the workload's token mix" if cost_side == "blended" else "")
            flags.append(
                Flag(
                    FlagKind.COST_ABOVE_MARKET,
                    f"{model_id} {cost_side}: our cost at SLO {format_usd(cost_value)}/1M is "
                    f"{format_usd(cost_value - lowest)} ({_pct(cost_value - lowest, lowest)}) "
                    f"above the lowest public {what}, {format_usd(lowest)}/1M "
                    f"({', '.join(at_min)}); median {format_usd(median)}/1M of "
                    f"{len(prices)} providers",
                    side=cost_side,
                    ours=cost_value,
                    reference=lowest,
                    delta=cost_value - lowest,
                    providers=at_min,
                    median=median,
                )
            )

    if our_price is None:
        return flags

    for side in SIDES if market else ():
        ours: Micros = getattr(our_price, f"{side}_per_mtok")
        prices = market_prices(side)
        lowest = min(prices.values())
        if ours <= lowest:
            continue
        at_min = tuple(sorted(name for name, p in prices.items() if p == lowest))
        median = _median(sorted(prices.values()))
        flags.append(
            Flag(
                FlagKind.PRICE_ABOVE_MARKET,
                f"{model_id} {side}: {format_usd(ours)}/1M is "
                f"{format_usd(ours - lowest)} ({_pct(ours - lowest, lowest)}) above the "
                f"market minimum {format_usd(lowest)}/1M ({', '.join(at_min)}); "
                f"{format_usd(ours - median)} ({_pct(ours - median, median)}) vs the "
                f"median {format_usd(median)}/1M of {len(prices)} providers",
                side=side,
                ours=ours,
                reference=lowest,
                delta=ours - lowest,
                providers=at_min,
                median=median,
            )
        )

    if our_cost_per_mtok is not None:
        for side in SIDES:
            ours = getattr(our_price, f"{side}_per_mtok")
            cost: Micros | None = getattr(our_cost_per_mtok, side)
            if cost is None or ours >= cost:
                continue
            flags.append(
                Flag(
                    FlagKind.NEGATIVE_MARGIN,
                    f"{model_id} {side}: price {format_usd(ours)}/1M is "
                    f"{format_usd(cost - ours)} below measured cost {format_usd(cost)}/1M",
                    side=side,
                    ours=ours,
                    reference=cost,
                    delta=ours - cost,
                )
            )
    return flags
