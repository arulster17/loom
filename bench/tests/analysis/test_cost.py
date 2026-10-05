from fractions import Fraction

import pytest

from loom_bench.cost import CostAllocation, cost_at_slo, cost_from_goodput
from loom_bench.money import SECONDS_PER_HOUR, TOKENS_PER_MTOK
from loom_bench.records import LoadMode
from loom_bench.slo import GoodputResult, Slo
from loom_bench.stats import Estimate

HOURLY = 3_600_000  # $3.60 / h


def point(mean: float) -> Estimate:
    return Estimate(mean=mean, lo=None, hi=None, n=1, std=None)


def ci(mean: float, lo: float, hi: float) -> Estimate:
    return Estimate(mean=mean, lo=lo, hi=hi, n=3, std=1.0)


def test_all_output_headline():
    c = cost_at_slo(HOURLY, input_tok_s=point(3000), output_tok_s=point(1000))
    # 1000 tok/s = 3.6M tok/h -> $1.00 per 1M output tokens
    assert c.output_per_mtok.value == 1_000_000
    assert c.input_per_mtok.value == 0
    assert c.total_per_mtok.value == 250_000  # 4000 tok/s blended
    assert c.allocation == "all_output"
    assert not c.trusted


def test_all_input():
    c = cost_at_slo(
        HOURLY,
        input_tok_s=point(3000),
        output_tok_s=point(1000),
        allocation=CostAllocation.all_input(),
    )
    assert c.input_per_mtok.value == 333_333  # 3.6e12 / 1.08e10 = 333_333.33
    assert c.output_per_mtok.value == 0


def test_weighted_prices_reproduce_replica_cost():
    tin, tout, r = 1000, 250, 4
    c = cost_at_slo(
        HOURLY,
        input_tok_s=point(tin),
        output_tok_s=point(tout),
        allocation=CostAllocation.weighted(r),
    )
    # effective 1000 + 4 * 250 = 2000 tok/s = 7.2M tok/h -> $0.50 in, $2.00 out
    assert c.input_per_mtok.value == 500_000
    assert c.output_per_mtok.value == 2_000_000
    assert _billed_per_hour(tin, tout, c) == HOURLY
    assert c.allocation == "weighted(output_input_ratio=4)"


def _billed_per_hour(tin: int, tout: int, c) -> Fraction:
    per_s = tin * c.input_per_mtok.value + tout * c.output_per_mtok.value
    return Fraction(per_s) * SECONDS_PER_HOUR / TOKENS_PER_MTOK


@pytest.mark.parametrize(
    "alloc",
    [
        CostAllocation.all_output(),
        CostAllocation.all_input(),
        CostAllocation.weighted(1),
        CostAllocation.weighted(6),
        CostAllocation.weighted("2.5"),
    ],
    ids=lambda a: a.describe(),
)
def test_every_allocation_bills_the_replica_hour(alloc):
    tin, tout = 2000, 500
    c = cost_at_slo(HOURLY, input_tok_s=point(tin), output_tok_s=point(tout), allocation=alloc)
    # Each price is off by at most half a micro per 1M tokens.
    max_rounding = Fraction(1, 2) * (tin + tout) * SECONDS_PER_HOUR / TOKENS_PER_MTOK
    assert abs(_billed_per_hour(tin, tout, c) - HOURLY) <= max_rounding
    if alloc.output_input_ratio != Fraction(5, 2):  # the others divide exactly
        assert _billed_per_hour(tin, tout, c) == HOURLY


def test_weighted_one_is_the_blended_price():
    c = cost_at_slo(
        HOURLY,
        input_tok_s=point(2000),
        output_tok_s=point(500),
        allocation=CostAllocation.weighted(1),
    )
    assert c.input_per_mtok == c.output_per_mtok == c.total_per_mtok


def test_single_half_up_rounding():
    # 9 micros/h at 1000 tok/s: 9e6 / 3.6e6 = 2.5 micros per 1M -> 3
    c = cost_at_slo(9, input_tok_s=point(0), output_tok_s=point(1000))
    assert c.output_per_mtok.value == 3
    # weighted: E = 600 + 4 * 100 = 1000 tok/s; input 2.5 -> 3, output exactly 10.
    # Rounding the input price first and multiplying would give 12.
    w = cost_at_slo(
        9,
        input_tok_s=point(600),
        output_tok_s=point(100),
        allocation=CostAllocation.weighted(4),
    )
    assert (w.input_per_mtok.value, w.output_per_mtok.value) == (3, 10)


def test_ci_maps_high_throughput_to_low_cost():
    c = cost_at_slo(
        HOURLY,
        input_tok_s=ci(4000, 3200, 5000),
        output_tok_s=ci(1000, 800, 1250),
        request_rate=ci(10, 8, 12.5),
    )
    assert c.output_per_mtok.model_dump() == {"value": 1_000_000, "lo": 800_000, "hi": 1_250_000}
    assert c.total_per_mtok.model_dump() == {"value": 200_000, "lo": 160_000, "hi": 250_000}
    assert c.input_per_mtok.model_dump() == {"value": 0, "lo": 0, "hi": 0}
    # 10 req/s -> 100 micros per request -> 100_000 per 1k requests
    assert c.per_1k_requests.model_dump() == {"value": 100_000, "lo": 80_000, "hi": 125_000}
    assert c.trusted


def test_non_positive_or_missing_bounds_are_unbounded():
    c = cost_at_slo(HOURLY, input_tok_s=ci(10, -2, 22), output_tok_s=ci(5, -1, 11))
    assert c.output_per_mtok.value is not None
    assert c.output_per_mtok.lo is not None
    assert c.output_per_mtok.hi is None  # throughput CI reaches zero

    single = cost_at_slo(HOURLY, input_tok_s=point(10), output_tok_s=point(5))
    assert single.output_per_mtok.lo is None and single.output_per_mtok.hi is None
    assert single.per_1k_requests is None


def test_zero_throughput_is_none():
    c = cost_at_slo(HOURLY, input_tok_s=point(0), output_tok_s=point(0))
    assert c.output_per_mtok.value is None
    assert c.input_per_mtok.value is None
    assert c.total_per_mtok.value is None
    w = cost_at_slo(
        HOURLY,
        input_tok_s=point(0),
        output_tok_s=point(0),
        allocation=CostAllocation.weighted(3),
    )
    assert w.input_per_mtok.value is None and w.output_per_mtok.value is None


def test_allocation_and_price_validation():
    with pytest.raises(ValueError):
        CostAllocation("weighted")
    with pytest.raises(ValueError):
        CostAllocation("all_output", Fraction(2))
    with pytest.raises(ValueError):
        CostAllocation.weighted(0)
    assert CostAllocation.weighted("1.5").output_input_ratio == Fraction(3, 2)
    with pytest.raises(TypeError):
        cost_at_slo(3.6, input_tok_s=point(1), output_tok_s=point(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cost_at_slo(-1, input_tok_s=point(1), output_tok_s=point(1))


def _goodput(tps: Estimate | None) -> GoodputResult:
    return GoodputResult(
        load_mode=LoadMode.OPEN_LOOP,
        slo=Slo(ttft_ms={"p95": 1000}, max_error_rate=0.01),
        max_load=None if tps is None else 4.0,
        first_failing_load=8.0,
        bracketed=True,
        trusted=tps is not None,
        request_rate=None if tps is None else ci(10, 8, 12.5),
        output_tok_s=tps,
        input_tok_s=None if tps is None else ci(4000, 3200, 5000),
        total_tok_s=None,
        max_sustainable_concurrency=None,
        points=[],
    )


def test_cost_from_goodput():
    assert cost_from_goodput(HOURLY, _goodput(None)) is None
    c = cost_from_goodput(HOURLY, _goodput(ci(1000, 800, 1250)))
    assert c is not None
    assert c.output_per_mtok.value == 1_000_000
    assert c.per_1k_requests.value == 100_000
