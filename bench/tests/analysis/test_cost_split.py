"""The prefill_time allocation: input pays for the replica time spent prefilling (φ),
output for the rest; CIs propagate from φ and from throughput."""

from fractions import Fraction

import pytest

from loom_bench.cost import (
    NA_NO_PREFILL_TIME,
    NA_PREFILL_SATURATED,
    CostAllocation,
    capped_share,
    cost_at_slo,
    cost_from_goodput,
)
from loom_bench.experiment import CostAllocationSpec
from loom_bench.metrics.aggregate import aggregate_runs, ci_rule
from loom_bench.metrics.summary import flatten_metrics, summarize_run
from loom_bench.money import SECONDS_PER_HOUR, TOKENS_PER_MTOK
from loom_bench.records import LoadMode, RequestRecord, RequestStatus
from loom_bench.slo import Slo, find_goodput
from loom_bench.stats import Estimate

HOURLY = 3_600_000  # $3.60 / h: $0.001 per second
SPLIT = CostAllocation.prefill_time()


def point(mean: float) -> Estimate:
    return Estimate(mean=mean, lo=None, hi=None, n=1, std=None)


def ci(mean: float, lo: float, hi: float) -> Estimate:
    return Estimate(mean=mean, lo=lo, hi=hi, n=3, std=1.0)


def split(phi: Estimate | None, tin: Estimate, tout: Estimate, rate: Estimate | None = None):
    return cost_at_slo(
        HOURLY,
        input_tok_s=tin,
        output_tok_s=tout,
        request_rate=rate,
        allocation=SPLIT,
        prefill_in_flight=phi,
    )


def test_point_prices_charge_prefill_time_to_input():
    c = split(point(0.25), point(3000), point(1000))
    # input: 0.25 x $0.001/s over 3000 tok/s; output: 0.75 x $0.001/s over 1000 tok/s
    assert c.input_per_mtok.value == 83_333  # 83,333.33 rounds down
    assert c.output_per_mtok.value == 750_000
    assert c.total_per_mtok.value == 250_000  # blended needs no split
    assert c.allocation == "prefill_time"
    assert c.input_time_share == point(0.25)
    assert c.input_per_mtok.na_reason is None and c.output_per_mtok.na_reason is None


@pytest.mark.parametrize("phi", [0.01, 0.2, 0.37, 0.5, 0.93])
@pytest.mark.parametrize(("tin", "tout"), [(2000, 500), (1024, 1024), (1537, 48)])
def test_split_bills_the_replica_hour(phi, tin, tout):
    c = split(point(phi), point(tin), point(tout))
    billed = (
        Fraction(tin * c.input_per_mtok.value + tout * c.output_per_mtok.value)
        * SECONDS_PER_HOUR
        / TOKENS_PER_MTOK
    )
    # each price is rounded once, to half a micro per 1M tokens
    max_rounding = Fraction(1, 2) * (tin + tout) * SECONDS_PER_HOUR / TOKENS_PER_MTOK
    assert abs(billed - HOURLY) <= max_rounding


def test_ci_takes_share_and_throughput_at_the_ends_that_push_each_price_the_same_way():
    c = split(ci(0.25, 0.2, 0.3), ci(3000, 2400, 3750), ci(1000, 800, 1250), ci(2, 1.6, 2.5))
    # input low: φ low, input throughput high; input high: φ high, throughput low
    assert (c.input_per_mtok.lo, c.input_per_mtok.value, c.input_per_mtok.hi) == (
        53_333,
        83_333,
        125_000,
    )
    # output low: φ high (less left for output), output throughput high; and the reverse
    assert (c.output_per_mtok.lo, c.output_per_mtok.value, c.output_per_mtok.hi) == (
        560_000,
        750_000,
        1_000_000,
    )
    # blended and per-request costs are unchanged by the split
    assert (c.total_per_mtok.lo, c.total_per_mtok.hi) == (200_000, 312_500)
    assert c.per_1k_requests.value == 500_000
    assert c.trusted


def test_ci_is_an_outer_bound_of_the_separate_intervals():
    phi, tin, tout = ci(0.3, 0.25, 0.36), ci(1000, 900, 1100), ci(500, 450, 550)
    with_phi = split(phi, tin, tout)
    fixed_phi = split(ci(0.3, 0.3, 0.3), tin, tout)
    assert with_phi.input_per_mtok.lo < fixed_phi.input_per_mtok.lo
    assert with_phi.input_per_mtok.hi > fixed_phi.input_per_mtok.hi
    assert with_phi.output_per_mtok.lo < fixed_phi.output_per_mtok.lo
    assert with_phi.output_per_mtok.hi > fixed_phi.output_per_mtok.hi


def test_single_repetition_has_no_bounds_and_is_untrusted():
    c = split(point(0.25), ci(3000, 2400, 3750), ci(1000, 800, 1250))
    assert c.input_per_mtok.value == 83_333
    assert c.input_per_mtok.lo is None and c.input_per_mtok.hi is None
    assert c.output_per_mtok.lo is None and c.output_per_mtok.hi is None
    assert not c.trusted


def test_no_ttft_means_no_split_but_a_blended_cost():
    c = split(None, point(3000), point(1000))
    assert c.input_per_mtok.na_reason == c.output_per_mtok.na_reason == NA_NO_PREFILL_TIME
    assert c.input_per_mtok.value is None and c.output_per_mtok.value is None
    assert c.total_per_mtok.value == 250_000
    assert c.input_time_share is None


def test_overlapping_prefills_leave_the_split_not_applicable():
    for phi in (point(1.0), ci(2.1, 1.8, 2.5)):
        c = split(phi, point(3000), point(1000))
        assert c.input_per_mtok.na_reason == c.output_per_mtok.na_reason == NA_PREFILL_SATURATED
        assert c.total_per_mtok.value == 250_000


def test_share_bound_above_one_is_capped():
    phi = ci(0.9, 0.7, 1.3)
    assert capped_share(phi) == ci(0.9, 0.7, 1.0)
    c = split(phi, ci(3000, 2400, 3750), ci(1000, 800, 1250))
    # input high bound: all of the replica's time on input at the low throughput
    assert c.input_per_mtok.hi == 416_667  # 1.0 x 1e9 / 2400
    # output low bound: nothing left for output
    assert c.output_per_mtok.lo == 0
    assert c.input_time_share.hi == 1.0


def test_zero_throughput_bound_is_unbounded():
    c = split(ci(0.25, 0.2, 0.3), ci(3000, 0, 3750), ci(1000, 800, 1250))
    assert c.input_per_mtok.hi is None and c.input_per_mtok.lo == 53_333


def test_other_allocations_ignore_the_share():
    phi = point(0.25)
    for alloc in (CostAllocation.all_output(), CostAllocation.weighted(4)):
        with_phi = cost_at_slo(
            HOURLY,
            input_tok_s=point(3000),
            output_tok_s=point(1000),
            allocation=alloc,
            prefill_in_flight=phi,
        )
        without = cost_at_slo(
            HOURLY, input_tok_s=point(3000), output_tok_s=point(1000), allocation=alloc
        )
        assert with_phi == without and with_phi.input_time_share is None


def test_experiments_can_declare_the_split():
    spec = CostAllocationSpec(method="prefill_time")
    assert spec.allocation() == SPLIT
    assert SPLIT.describe() == "prefill_time"
    with pytest.raises(ValueError):
        CostAllocation("prefill_time", Fraction(2))


def _records(rate: float, ttft_s: float, n: int = 20) -> list[RequestRecord]:
    out = []
    for i in range(n):
        sent = i / rate
        out.append(
            RequestRecord(
                request_id=str(i),
                status=RequestStatus.OK,
                sent_at_s=sent,
                first_token_at_s=sent + ttft_s,
                finished_at_s=sent + ttft_s + 2.0,
                prompt_tokens=1000,
                completion_tokens=100,
            )
        )
    return out


def test_runs_measure_requests_in_prefill_by_littles_law():
    s = summarize_run(_records(2.0, 0.15), window_s=10.0, gpus=1)
    # 20 successful requests over 10 s at a mean TTFT of 150 ms
    assert s.prefill_in_flight == pytest.approx(2.0 * 0.15)
    assert flatten_metrics(s)["prefill_in_flight"] == pytest.approx(0.3)
    assert ci_rule("prefill_in_flight").scale == "positive"
    # recomputed from the stored fields: older summaries have it too
    stored = s.model_dump(mode="json")
    stored.pop("prefill_in_flight")
    assert type(s).model_validate(stored).prefill_in_flight == pytest.approx(0.3)


def test_goodput_carries_the_share_into_cost():
    slo = Slo(ttft_ms={"p95": 1000}, max_error_rate=0.01)
    runs = [
        summarize_run(_records(2.0, ttft), window_s=10.0 * f, gpus=1)
        for ttft, f in ((0.14, 1.0), (0.15, 1.02), (0.16, 0.98))
    ]
    agg = aggregate_runs(runs)
    goodput = find_goodput(slo, [(2.0, agg)], LoadMode.OPEN_LOOP)
    assert goodput.prefill_in_flight == agg.get("prefill_in_flight")
    assert goodput.prefill_in_flight.method == "log_t"
    c = cost_from_goodput(HOURLY, goodput, SPLIT)
    assert c is not None and c.input_time_share == goodput.prefill_in_flight
    assert c.input_per_mtok.lo < c.input_per_mtok.value < c.input_per_mtok.hi
    assert c.output_per_mtok.lo < c.output_per_mtok.value < c.output_per_mtok.hi
