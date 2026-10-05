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


class FlagKind(StrEnum):
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


@dataclass(frozen=True, slots=True)
class Flag:
    kind: FlagKind
    message: str
    side: Side | None = None
    ours: Micros | None = None
    # price_above_market: lowest eligible public price; negative_margin: our cost.
    reference: Micros | None = None
    delta: Micros | None = None  # ours - reference
    providers: tuple[str, ...] = ()  # providers at the reference price
    median: Micros | None = None  # median eligible public price (price_above_market only)


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
) -> list[Flag]:
    """Flag pricing problems for one model.

    Aggregators and entries whose availability is unverified are left out of the
    market comparison by default: they are not a firm price for this model.
    """
    flags: list[Flag] = []
    market = [
        (provider.name, entry)
        for provider, entry in competitors.entries_for(model_id)
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
    if our_price is None:
        return flags

    for side in SIDES if market else ():
        ours: Micros = getattr(our_price, f"{side}_per_mtok")
        prices: dict[str, Micros] = {}  # provider -> its cheapest entry for this model
        for name, entry in market:
            p = getattr(entry, f"{side}_per_mtok")
            prices[name] = min(prices.get(name, p), p)
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
