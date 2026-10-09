"""$/1M tokens at SLO from goodput and the replica's hourly price.

Which hourly price: `replica_prices` gives one per price column, each the instance
price plus the run's block storage amortised per hour (`PriceBook.with_storage`);
network egress and fixed account costs are not included (docs/cost-model.md):

- on_demand: the public on-demand list price in `bench/prices.yaml`. Reproducible
  from the price book alone, so results are ranked by it. A mock or local host has
  no list price; it uses the price its experiment declares, else it has none.
- spot: the indicative `spot_per_hour` in `bench/prices.yaml`.
- committed_1y: `committed_1y_per_hour`, where the price book has one.
- as_run: the price recorded in the run's provenance at launch (the observed spot
  price, or the on-demand price), without the budget guard's safety multiplier.

The replica's hourly price (integer micros) buys `goodput` tokens per second. A GPU
produces input (prefill) and output (decode) tokens at the same time, so splitting
its cost between them is a choice, recorded as a `CostAllocation`:

- all_output: the whole cost is charged to output tokens. This is the headline
  "$ per 1M output tokens at SLO". The input price is not applicable (not $0: the
  allocation assigns input no cost, so there is nothing to price or to margin
  against); its `MicrosRange` carries `na_reason`.
- all_input: the whole cost is charged to input tokens; the output price is not
  applicable in the same way.
- weighted(r): an output token costs r input tokens. With effective throughput
  E = in_tok_s + r · out_tok_s, input price = cost / E and output price = r · that.
  Prices at these rates bill exactly the replica's cost at the measured mix.
- prefill_time: the split measured at the goodput point, the reports' headline split.
  Input tokens pay for the share φ of the replica's time spent prefilling prompts,
  output tokens for the rest (decode steps, and any idle headroom the SLO needs). φ
  is the mean number of requests in their prefill phase, request rate × mean TTFT
  (Little's law), measured per repetition (`RunSummary.prefill_in_flight`) and capped
  at 1. Input price = φ · cost / in_tok_s, output price = (1 - φ) · cost / out_tok_s;
  together they bill exactly the replica's cost at the measured mix
  (docs/cost-model.md, section 2).

All prices are computed exactly with Fraction and rounded once, half-up, to micros.

Confidence intervals: cost falls monotonically as throughput rises, so the low cost
bound comes from the throughput CI's upper bound and vice versa. Where a cost
depends on both input and output throughput, both are taken at the same end of
their CIs. Throughput intervals are computed on the log scale (`Estimate.method`
"log_t"), so their lower bound is positive and both cost bounds are finite; the
cost interval is then exactly the reciprocal of the throughput interval, scaled.
Under prefill_time the share φ has its own CI, and each price bound takes φ and the
throughput at the ends that push that price the same way (input high: φ high and
in_tok_s low; output high: φ low and out_tok_s low). Taking both extremes together is
an outer bound, wider than a joint 95% interval.
Only a throughput bound at or below zero (an arithmetic fallback when a repetition
measured zero) makes the cost bound unbounded (None), and a missing CI (single run)
gives no bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any, Literal, cast

from pydantic import BaseModel

from loom_bench.money import SECONDS_PER_HOUR, TOKENS_PER_MTOK, Micros, round_half_up
from loom_bench.prices import PriceBook, UnverifiedPriceError
from loom_bench.records import Market
from loom_bench.registry import Cloud
from loom_bench.slo import GoodputResult
from loom_bench.stats import Estimate

AllocationMethod = Literal["all_output", "all_input", "weighted", "prefill_time"]


@dataclass(frozen=True)
class CostAllocation:
    method: AllocationMethod
    output_input_ratio: Fraction | None = None  # weighted only

    def __post_init__(self) -> None:
        if (self.method == "weighted") != (self.output_input_ratio is not None):
            raise ValueError("output_input_ratio is required for, and only for, 'weighted'")
        if self.output_input_ratio is not None and self.output_input_ratio <= 0:
            raise ValueError("output_input_ratio must be positive")

    @classmethod
    def all_output(cls) -> CostAllocation:
        return cls("all_output")

    @classmethod
    def all_input(cls) -> CostAllocation:
        return cls("all_input")

    @classmethod
    def weighted(cls, output_input_ratio: int | str | Fraction) -> CostAllocation:
        return cls("weighted", Fraction(output_input_ratio))

    @classmethod
    def prefill_time(cls) -> CostAllocation:
        return cls("prefill_time")

    def describe(self) -> str:
        if self.output_input_ratio is None:
            return self.method
        return f"weighted(output_input_ratio={self.output_input_ratio})"


class MicrosRange(BaseModel):
    """A cost in micros with CI bounds. None means unbounded / not computable.

    `na_reason` is set when the price is not applicable at all (the allocation
    assigns that side no cost); value and bounds are then None.
    """

    value: Micros | None
    lo: Micros | None
    hi: Micros | None
    na_reason: str | None = None

    @classmethod
    def not_applicable(cls, reason: str) -> MicrosRange:
        return cls(value=None, lo=None, hi=None, na_reason=reason)


NA_ALL_OUTPUT = "all cost allocated to output"
NA_ALL_INPUT = "all cost allocated to input"
NA_NO_PREFILL_TIME = "no TTFT measured at goodput, so no prefill-time split"
NA_PREFILL_SATURATED = (
    "prefills overlapped at goodput (request rate × mean TTFT ≥ 1), so prefill time "
    "cannot be separated from decode time"
)


class CostAtSlo(BaseModel):
    hourly_micros: Micros
    allocation: str
    input_per_mtok: MicrosRange
    output_per_mtok: MicrosRange
    total_per_mtok: MicrosRange  # blended: hourly cost / (input + output) tokens
    per_1k_requests: MicrosRange | None  # per 1,000 requests: one request is often < 1 micro
    trusted: bool
    # prefill_time only: φ, the share of the replica's time charged to input (capped at 1).
    input_time_share: Estimate | None = None


End = Literal["mean", "lo", "hi"]
Side = Literal["in", "out"]


def _at(est: Estimate, end: End) -> Fraction | None:
    value = {"mean": est.mean, "lo": est.lo, "hi": est.hi}[end]
    return None if value is None else Fraction(value)


def _per_unit(hourly: Fraction, rate_per_s: Fraction | None, units: int) -> Micros | None:
    """Micros per `units` items produced at `rate_per_s`, rounded once."""
    if rate_per_s is None or rate_per_s <= 0:
        return None
    return round_half_up(hourly * units / (rate_per_s * SECONDS_PER_HOUR))


def _prices(
    hourly: Fraction, tin: Fraction | None, tout: Fraction | None, alloc: CostAllocation
) -> tuple[Micros | None, Micros | None, Micros | None]:
    """(input, output, blended total) micros per 1M tokens at one throughput point."""
    if tin is None or tout is None:
        return None, None, None
    total = _per_unit(hourly, tin + tout, TOKENS_PER_MTOK)
    if alloc.method == "all_output":
        return None, _per_unit(hourly, tout, TOKENS_PER_MTOK), total
    if alloc.method == "all_input":
        return _per_unit(hourly, tin, TOKENS_PER_MTOK), None, total
    if alloc.method == "prefill_time":  # the split needs φ: `_split_prices`
        return None, None, total
    r = alloc.output_input_ratio or Fraction(0)
    effective = tin + r * tout
    return (
        _per_unit(hourly, effective, TOKENS_PER_MTOK),
        _per_unit(hourly * r, effective, TOKENS_PER_MTOK),
        total,
    )


def capped_share(prefill_in_flight: Estimate) -> Estimate:
    """φ: requests in prefill (request rate × mean TTFT), point and bounds capped at 1."""

    def cap(x: float | None) -> float | None:
        return None if x is None else min(x, 1.0)

    return prefill_in_flight.model_copy(
        update={
            "mean": min(prefill_in_flight.mean, 1.0),
            "lo": cap(prefill_in_flight.lo),
            "hi": cap(prefill_in_flight.hi),
        }
    )


def _split_prices(
    hourly: Fraction, input_tok_s: Estimate, output_tok_s: Estimate, share: Estimate
) -> tuple[MicrosRange, MicrosRange]:
    """(input, output) under prefill_time: φ·H over input tokens, (1 - φ)·H over output.

    Each bound takes φ and the throughput at the ends that push that price the same way.
    """

    def price(phi: Fraction | None, tok_s: Fraction | None, side: Side) -> Micros | None:
        if phi is None:
            return None
        return _per_unit(hourly * (phi if side == "in" else 1 - phi), tok_s, TOKENS_PER_MTOK)

    def side_range(tok_s: Estimate, side: Side) -> MicrosRange:
        # input cost rises with φ, output cost falls with it; both fall with throughput
        cheap: End = "lo" if side == "in" else "hi"
        dear: End = "hi" if side == "in" else "lo"
        return MicrosRange(
            value=price(_at(share, "mean"), _at(tok_s, "mean"), side),
            lo=price(_at(share, cheap), _at(tok_s, "hi"), side),
            hi=price(_at(share, dear), _at(tok_s, "lo"), side),
        )

    return side_range(input_tok_s, "in"), side_range(output_tok_s, "out")


def cost_at_slo(
    hourly_micros: Micros,
    *,
    input_tok_s: Estimate,
    output_tok_s: Estimate,
    request_rate: Estimate | None = None,
    allocation: CostAllocation | None = None,
    prefill_in_flight: Estimate | None = None,
) -> CostAtSlo:
    """Prices for one replica at its goodput. Throughputs are per replica.

    `prefill_in_flight` (request rate × mean TTFT at the same load point, with its CI
    across repetitions) is used by the prefill_time allocation only."""
    if isinstance(hourly_micros, bool) or not isinstance(hourly_micros, int):
        raise TypeError("hourly_micros must be integer micros")
    if hourly_micros < 0:
        raise ValueError("hourly_micros must be non-negative")
    alloc = allocation or CostAllocation.all_output()
    hourly = Fraction(hourly_micros)

    def prices(end: End) -> tuple[Micros | None, Micros | None, Micros | None]:
        return _prices(hourly, _at(input_tok_s, end), _at(output_tok_s, end), alloc)

    # The low cost bound comes from the high throughput bound, and the reverse.
    point, low, high = prices("mean"), prices("hi"), prices("lo")
    ranges = [MicrosRange(value=point[i], lo=low[i], hi=high[i]) for i in range(3)]
    share: Estimate | None = None
    if alloc.method == "all_output":
        ranges[0] = MicrosRange.not_applicable(NA_ALL_OUTPUT)
    elif alloc.method == "all_input":
        ranges[1] = MicrosRange.not_applicable(NA_ALL_INPUT)
    elif alloc.method == "prefill_time":
        if prefill_in_flight is None:
            ranges[0] = ranges[1] = MicrosRange.not_applicable(NA_NO_PREFILL_TIME)
        elif prefill_in_flight.mean >= 1:
            ranges[0] = ranges[1] = MicrosRange.not_applicable(NA_PREFILL_SATURATED)
        else:
            share = capped_share(prefill_in_flight)
            ranges[0], ranges[1] = _split_prices(hourly, input_tok_s, output_tok_s, share)

    per_1k: MicrosRange | None = None
    if request_rate is not None:
        per_1k = MicrosRange(
            value=_per_unit(hourly, _at(request_rate, "mean"), 1000),
            lo=_per_unit(hourly, _at(request_rate, "hi"), 1000),
            hi=_per_unit(hourly, _at(request_rate, "lo"), 1000),
        )

    estimates = [input_tok_s, output_tok_s] + ([request_rate] if request_rate else [])
    if share is not None:
        estimates.append(share)
    return CostAtSlo(
        hourly_micros=hourly_micros,
        allocation=alloc.describe(),
        input_per_mtok=ranges[0],
        output_per_mtok=ranges[1],
        total_per_mtok=ranges[2],
        per_1k_requests=per_1k,
        trusted=all(e.trusted for e in estimates),
        input_time_share=share,
    )


def cost_from_goodput(
    hourly_micros: Micros, goodput: GoodputResult, allocation: CostAllocation | None = None
) -> CostAtSlo | None:
    """`cost_at_slo` at a sweep's goodput point; None when no load met the SLO."""
    if goodput.input_tok_s is None or goodput.output_tok_s is None:
        return None
    return cost_at_slo(
        hourly_micros,
        input_tok_s=goodput.input_tok_s,
        output_tok_s=goodput.output_tok_s,
        request_rate=goodput.request_rate,
        allocation=allocation,
        prefill_in_flight=goodput.prefill_in_flight,
    )


class PriceColumn(StrEnum):
    ON_DEMAND = "on_demand"
    SPOT = "spot"
    COMMITTED_1Y = "committed_1y"
    AS_RUN = "as_run"


PRICE_COLUMN_LABELS = {
    PriceColumn.ON_DEMAND: "on-demand",
    PriceColumn.SPOT: "spot",
    PriceColumn.COMMITTED_1Y: "committed 1y",
    PriceColumn.AS_RUN: "as run",
}


class ReplicaPrices(BaseModel):
    """One replica's hourly price per price column (micros, storage included).

    None means no price in that column; `missing` says why there is no on-demand
    price, the one results are ranked by.
    """

    on_demand: Micros | None
    spot: Micros | None
    committed_1y: Micros | None
    as_run: Micros | None
    storage_gb: int | None  # block storage included; None for a mock or local host
    missing: str | None = None

    def get(self, column: PriceColumn) -> Micros | None:
        return cast(Micros | None, getattr(self, column.value))


NO_LOCAL_PRICE = (
    "no hourly price: a mock or local host is priced only when its experiment sets "
    "provider.hourly_price"
)
NO_PRICE_BASIS = (
    "the provenance records no price basis (written before provenance schema 2), so the "
    "storage volume and as-run price are unknown"
)


def recorded_hourly_micros(prov: Mapping[str, Any]) -> Micros | None:
    """The as-run price recorded in a provenance record, when it records its basis."""
    if not isinstance(prov.get("price_basis"), Mapping):
        return None
    value = prov.get("hourly_micros")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"hourly_micros must be integer micros, got {value!r}")
    return value


def replica_prices(
    prov: Mapping[str, Any], book: PriceBook, *, allow_unverified: bool = False
) -> ReplicaPrices:
    """Every price column for the replica a run's provenance describes.

    A price-book entry that is missing raises KeyError and an unverified one raises
    UnverifiedPriceError, so a report never silently drops a price it should have had.
    """
    as_run = recorded_hourly_micros(prov)
    market = prov.get("market")
    if market is None or Market(market) is Market.LOCAL:
        return ReplicaPrices(
            on_demand=as_run,
            spot=None,
            committed_1y=None,
            as_run=as_run,
            storage_gb=None,
            missing=None if as_run is not None else NO_LOCAL_PRICE,
        )
    basis = prov.get("price_basis")
    if not isinstance(basis, Mapping):
        return ReplicaPrices(
            on_demand=None,
            spot=None,
            committed_1y=None,
            as_run=None,
            storage_gb=None,
            missing=NO_PRICE_BASIS,
        )
    cloud = cast(Cloud, prov.get("cloud"))
    region = prov.get("region") or ""
    hardware = prov.get("hardware")
    instance_type = (hardware.get("instance_type") if isinstance(hardware, Mapping) else None) or ""
    entry = book.instance(cloud, region, instance_type)
    if not entry.verified and not allow_unverified:
        raise UnverifiedPriceError(f"{cloud}/{region}/{instance_type}: {entry.note}")
    storage_gb = int(basis.get("storage_gb", 0))

    def priced(per_hour: Micros | None) -> Micros | None:
        if per_hour is None:
            return None
        return book.with_storage(
            cloud, region, per_hour, storage_gb, allow_unverified=allow_unverified
        )

    return ReplicaPrices(
        on_demand=priced(entry.on_demand_per_hour),
        spot=priced(entry.spot_per_hour),
        committed_1y=priced(entry.committed_1y_per_hour),
        as_run=as_run,
        storage_gb=storage_gb,
    )
