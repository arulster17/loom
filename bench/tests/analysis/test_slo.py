import pytest
from pydantic import ValidationError

from loom_bench.metrics.aggregate import AggregateSummary, aggregate_runs
from loom_bench.metrics.summary import summarize_run
from loom_bench.records import LoadMode, RequestRecord, RequestStatus
from loom_bench.slo import Slo, bisect_next_load, find_goodput, slo_met
from loom_bench.stats import Estimate, mean_ci

SLO = Slo(ttft_ms={"p95": 1000}, tpot_ms={"p95": 50}, max_error_rate=0.01)


def est(mean: float, hi: float | None = None, n: int = 3) -> Estimate:
    if hi is None:
        return Estimate(mean=mean, lo=None, hi=None, n=1, std=None)
    return Estimate(mean=mean, lo=2 * mean - hi, hi=hi, n=n, std=1.0)


def agg(ttft_hi=900.0, tpot_hi=40.0, err_hi=0.0, out_tps=100.0, n=3) -> AggregateSummary:
    single = n == 1
    metrics = {
        "ttft_ms.p95": est(ttft_hi - 50, None if single else ttft_hi, n),
        "tpot_ms.p95": est(tpot_hi - 5, None if single else tpot_hi, n),
        "error_rate": est(err_hi, None if single else err_hi, n),
        "throughput.request_rate": est(out_tps / 100, None if single else out_tps / 90, n),
        "throughput.output_tok_s": est(out_tps, None if single else out_tps * 1.1, n),
        "throughput.input_tok_s": est(out_tps * 4, None if single else out_tps * 4.4, n),
        "throughput.total_tok_s": est(out_tps * 5, None if single else out_tps * 5.5, n),
    }
    return AggregateSummary(n_runs=n, confidence=0.95, trusted=n >= 2, warnings=[], metrics=metrics)


def test_slo_from_yaml_and_validation():
    slo = Slo.from_yaml("ttft_ms: {p95: 1000}\ntpot_ms: {p95: 50}\nmax_error_rate: 0.01\n")
    assert slo == SLO
    assert slo.targets() == [("ttft_ms", "p95", 1000), ("tpot_ms", "p95", 50)]
    with pytest.raises(ValidationError):
        Slo.model_validate({"ttft_ms": {"p75": 100}, "max_error_rate": 0.01})
    with pytest.raises(ValidationError):
        Slo.model_validate({"queue_ms": {"p95": 100}, "max_error_rate": 0.01})
    with pytest.raises(ValidationError):
        Slo.model_validate({"ttft_ms": {"p95": 100}})  # error budget is required
    with pytest.raises(ValidationError):
        Slo.model_validate({"ttft_ms": {"p95": -1}, "max_error_rate": 0.01})


def _req(status=RequestStatus.OK, ttft=0.1, e2e=1.0, out=10, itl=()):
    return RequestRecord(
        request_id="r",
        status=status,
        sent_at_s=0.0,
        first_token_at_s=ttft,
        finished_at_s=e2e,
        completion_tokens=out,
        itl_s=list(itl),
    )


def test_request_meets():
    slo = Slo(ttft_ms={"p50": 200, "p99": 500}, tpot_ms={"p95": 100}, max_error_rate=0.0)
    assert slo.request_meets(_req(ttft=0.4))  # loosest TTFT target (500 ms) applies
    assert not slo.request_meets(_req(ttft=0.6))
    assert not slo.request_meets(_req(ttft=0.1, e2e=2.0))  # TPOT = 1.9 / 9 s
    assert slo.request_meets(_req(ttft=0.1, e2e=5.0, out=1))  # TPOT undefined: vacuous
    assert not slo.request_meets(_req(status=RequestStatus.ERROR))
    assert not slo.request_meets(_req(ttft=None))

    itl = Slo(itl_ms={"p50": 30, "p99": 100}, max_error_rate=0.0)
    assert itl.request_meets(_req(itl=[0.02] * 99 + [0.09]))
    assert not itl.request_meets(_req(itl=[0.02] * 90 + [0.2] * 10))  # own p99 > 100 ms
    assert not itl.request_meets(_req(itl=[0.05] * 10))  # own p50 > 30 ms
    assert itl.request_meets(_req(itl=[]))


def test_slo_met_uses_ci_upper_bound():
    passing = slo_met(SLO, agg(ttft_hi=990))
    assert passing.met and passing.trusted
    assert {c.metric for c in passing.checks} == {"ttft_ms.p95", "tpot_ms.p95", "error_rate"}

    # mean 1050 - 50 = 950 is within target, but the CI upper bound 1050 is not
    failing = slo_met(SLO, agg(ttft_hi=1050))
    assert not failing.met
    check = next(c for c in failing.checks if c.metric == "ttft_ms.p95")
    assert check.observed == 1050 and check.used_upper_bound and not check.passed

    assert not slo_met(SLO, agg(err_hi=0.02)).met
    assert not slo_met(SLO, agg(tpot_hi=51)).met


def test_slo_met_single_run_uses_mean_and_is_untrusted():
    verdict = slo_met(SLO, agg(ttft_hi=1040, n=1))  # mean 990
    assert verdict.met
    assert not verdict.trusted
    assert not any(c.used_upper_bound for c in verdict.checks)


def test_slo_met_missing_metric_fails():
    a = agg()
    del a.metrics["tpot_ms.p95"]
    verdict = slo_met(SLO, a)
    assert not verdict.met
    assert next(c for c in verdict.checks if c.metric == "tpot_ms.p95").observed is None


def test_goodput_highest_pass_before_first_failure():
    points = [
        (5.0, agg(ttft_hi=2000, out_tps=500)),
        (1.0, agg(out_tps=100)),
        (4.0, agg(out_tps=400)),  # passes, but above the first failure: ignored
        (2.0, agg(out_tps=200)),
        (3.0, agg(ttft_hi=1200, out_tps=300)),
    ]
    g = find_goodput(SLO, points, LoadMode.OPEN_LOOP)
    assert g.max_load == 2.0
    assert g.first_failing_load == 3.0
    assert g.bracketed and g.trusted
    assert g.output_tok_s == points[3][1].get("throughput.output_tok_s")
    assert g.input_tok_s.mean == 800
    assert g.total_tok_s.mean == 1000
    assert g.request_rate.mean == 2
    assert g.max_sustainable_concurrency is None
    assert [p.met for p in g.points] == [True, True, False, True, False]
    assert g.load_mode == "open_loop"


def test_goodput_closed_loop_and_edge_cases():
    all_pass = find_goodput(SLO, [(8, agg()), (16, agg()), (32, agg())], LoadMode.CLOSED_LOOP)
    assert all_pass.max_load == 32
    assert all_pass.max_sustainable_concurrency == 32
    assert not all_pass.bracketed  # never saw a failure: goodput may be higher

    none = find_goodput(SLO, [(1, agg(ttft_hi=5000)), (2, agg())], LoadMode.OPEN_LOOP)
    assert none.max_load is None and none.output_tok_s is None
    assert none.first_failing_load == 1

    single = find_goodput(SLO, [(1, agg(n=1))], LoadMode.OPEN_LOOP)
    assert single.max_load == 1 and not single.trusted

    with pytest.raises(ValueError):
        find_goodput(SLO, [(1, agg()), (1, agg())], LoadMode.OPEN_LOOP)


@pytest.mark.parametrize("threshold", [1.0, 3.3, 7.25, 15.9])
def test_bisect_converges_to_threshold(threshold):
    history: list[tuple[float, bool]] = []
    while (load := bisect_next_load(history, 1.0, 16.0, rel_tol=0.02)) is not None:
        history.append((load, load <= threshold))
        assert len(history) < 20
    passing = max(load for load, met in history if met)
    failing = [load for load, met in history if not met]
    assert passing <= threshold
    assert min(failing) - passing <= 0.02 * passing


def test_bisect_terminal_cases():
    assert bisect_next_load([], 1, 16) == 1
    assert bisect_next_load([(1, False)], 1, 16) is None
    assert bisect_next_load([(1, True)], 1, 16) == 16
    assert bisect_next_load([(1, True), (16, True)], 1, 16) is None
    # a pass above the lowest failure does not move the bracket
    assert bisect_next_load([(1, True), (16, True), (8, False)], 1, 16) == 4.5


def _run(ttft_s: float, n: int = 20, out_tokens: int = 10, window_s: float = 10.0):
    records = [
        RequestRecord(
            request_id=f"r{i}",
            status=RequestStatus.OK,
            sent_at_s=i * 0.1,
            first_token_at_s=i * 0.1 + ttft_s,
            finished_at_s=i * 0.1 + ttft_s + 0.9,
            completion_tokens=out_tokens,
            prompt_tokens=40,
        )
        for i in range(n)
    ]
    return summarize_run(records, window_s=window_s, gpus=1)


def test_aggregate_runs_t_ci_per_metric():
    runs = [_run(0.1), _run(0.2), _run(0.3)]
    a = aggregate_runs(runs)
    assert a.n_runs == 3 and a.trusted
    ttft = a.get("ttft_ms.p95")
    expected = mean_ci([100.0, 200.0, 300.0])
    assert ttft.mean == pytest.approx(200)
    assert ttft.lo == pytest.approx(expected.lo)
    assert ttft.hi == pytest.approx(expected.hi)
    assert a.get("throughput.output_tok_s").mean == pytest.approx(20)
    assert a.get("throughput.output_tok_s").std == 0
    assert any("ttft_ms.p95" in w and "CV" in w for w in a.warnings)
    assert not any("throughput" in w for w in a.warnings)
    assert AggregateSummary.model_validate_json(a.model_dump_json()) == a


def test_aggregate_single_run_is_flagged():
    a = aggregate_runs([_run(0.1)])
    assert not a.trusted
    assert any("never trusted" in w for w in a.warnings)
    assert a.get("ttft_ms.p50").hi is None
    with pytest.raises(ValueError):
        aggregate_runs([])


def test_aggregate_feeds_slo_and_goodput_end_to_end():
    slo = Slo(ttft_ms={"p95": 250}, max_error_rate=0.0)
    points = [
        (1.0, aggregate_runs([_run(0.10), _run(0.11)])),
        (2.0, aggregate_runs([_run(0.200), _run(0.205), _run(0.210)])),
        # mean 240 ms is within 250 ms, but the CI upper bound is not
        (4.0, aggregate_runs([_run(0.20), _run(0.24), _run(0.28)])),
    ]
    g = find_goodput(slo, points, LoadMode.OPEN_LOOP)
    assert g.max_load == 2.0
    assert g.first_failing_load == 4.0
    assert points[2][1].get("ttft_ms.p95").mean == pytest.approx(240)
