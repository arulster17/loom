"""$/1M tokens at SLO from goodput and the replica's hourly price.

The replica's hourly price (integer micros, already including storage, egress etc.)
buys `goodput` tokens per second. A GPU produces input (prefill) and output (decode)
tokens at the same time, so splitting its cost between them is a choice, recorded
as a `CostAllocation`:

- all_output: the whole cost is charged to output tokens. This is the headline
  "$ per 1M output tokens at SLO". The input price is not applicable (not $0: the
  allocation assigns input no cost, so there is nothing to price or to margin
  against); its `MicrosRange` carries `na_reason`.
- all_input: the whole cost is charged to input tokens; the output price is not
  applicable in the same way.
- weighted(r): an output token costs r input tokens. With effective throughput
  E = in_tok_s + r · out_tok_s, input price = cost / E and output price = r · that.
  Prices at these rates bill exactly the replica's cost at the measured mix.

All prices are computed exactly with Fraction and rounded once, half-up, to micros.

Confidence intervals: cost falls monotonically as throughput rises, so the low cost
bound comes from the throughput CI's upper bound and vice versa. Where a cost
depends on both input and output throughput, both are taken at the same end of
their CIs. Throughput intervals are computed on the log scale (`Estimate.method`
"log_t"), so their lower bound is positive and both cost bounds are finite; the
cost interval is then exactly the reciprocal of the throughput interval, scaled.
Only a throughput bound at or below zero (an arithmetic fallback when a repetition
measured zero) makes the cost bound unbounded (None), and a missing CI (single run)
gives no bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel

from loom_bench.money import SECONDS_PER_HOUR, TOKENS_PER_MTOK, Micros, round_half_up
from loom_bench.slo import GoodputResult
from loom_bench.stats import Estimate

AllocationMethod = Literal["all_output", "all_input", "weighted"]


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


class CostAtSlo(BaseModel):
    hourly_micros: Micros
    allocation: str
    input_per_mtok: MicrosRange
    output_per_mtok: MicrosRange
    total_per_mtok: MicrosRange  # blended: hourly cost / (input + output) tokens
    per_1k_requests: MicrosRange | None  # per 1,000 requests: one request is often < 1 micro
    trusted: bool


End = Literal["mean", "lo", "hi"]


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
    r = alloc.output_input_ratio or Fraction(0)
    effective = tin + r * tout
    return (
        _per_unit(hourly, effective, TOKENS_PER_MTOK),
        _per_unit(hourly * r, effective, TOKENS_PER_MTOK),
        total,
    )


def cost_at_slo(
    hourly_micros: Micros,
    *,
    input_tok_s: Estimate,
    output_tok_s: Estimate,
    request_rate: Estimate | None = None,
    allocation: CostAllocation | None = None,
) -> CostAtSlo:
    """Prices for one replica at its goodput. Throughputs are per replica."""
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
    if alloc.method == "all_output":
        ranges[0] = MicrosRange.not_applicable(NA_ALL_OUTPUT)
    elif alloc.method == "all_input":
        ranges[1] = MicrosRange.not_applicable(NA_ALL_INPUT)

    per_1k: MicrosRange | None = None
    if request_rate is not None:
        per_1k = MicrosRange(
            value=_per_unit(hourly, _at(request_rate, "mean"), 1000),
            lo=_per_unit(hourly, _at(request_rate, "hi"), 1000),
            hi=_per_unit(hourly, _at(request_rate, "lo"), 1000),
        )

    estimates = [input_tok_s, output_tok_s] + ([request_rate] if request_rate else [])
    return CostAtSlo(
        hourly_micros=hourly_micros,
        allocation=alloc.describe(),
        input_per_mtok=ranges[0],
        output_per_mtok=ranges[1],
        total_per_mtok=ranges[2],
        per_1k_requests=per_1k,
        trusted=all(e.trusted for e in estimates),
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
    )
