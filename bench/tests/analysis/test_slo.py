import pytest
from pydantic import ValidationError

from loom_bench.metrics.aggregate import AggregateSummary, aggregate_runs, ci_rule, estimate_metric
from loom_bench.metrics.summary import summarize_run
from loom_bench.records import LoadMode, RequestRecord, RequestStatus
from loom_bench.slo import Slo, bisect_next_load, find_goodput, slo_met
from loom_bench.stats import Estimate, log_mean_ci

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


@pytest.mark.parametrize("threshold", [0.5, 1.3, 1.9, 4.4, 6.0, 7.9, 8.0, 20.0])
def test_geometric_search_converges_to_threshold(threshold):
    history: list[tuple[float, bool]] = []
    while (load := bisect_next_load(history, 0.5, 8.0, 0.05, scale="geometric")) is not None:
        history.append((load, load <= threshold))
        assert len(history) < 20
    passing = max(load for load, met in history if met)
    failing = [load for load, met in history if not met]
    if threshold >= 8.0:  # hi passes: the search stops there, unbracketed
        assert passing == 8.0 and not failing
        return
    assert passing <= threshold < min(failing)
    assert min(failing) - passing <= 0.05 * passing


def test_geometric_search_climbs_from_lo_then_bisects_on_a_log_scale():
    assert bisect_next_load([], 1, 8, scale="geometric") == 1
    assert bisect_next_load([(1, False)], 1, 8, scale="geometric") is None
    assert bisect_next_load([(1, True)], 1, 8, scale="geometric") == 2
    assert bisect_next_load([(1, True), (2, True)], 1, 8, scale="geometric") == 4
    assert bisect_next_load([(1, True), (2, True)], 1, 8, scale="geometric", step=3) == 6
    assert bisect_next_load([(1, True), (2, True), (4, True)], 1, 6, scale="geometric") == 6
    assert bisect_next_load([(1, True), (6, True)], 1, 6, scale="geometric") is None
    nxt = bisect_next_load([(1, True), (2, True), (4, False)], 1, 8, scale="geometric")
    assert nxt == pytest.approx(8**0.5)
    with pytest.raises(ValueError):
        bisect_next_load([], 1, 8, scale="geometric", step=1.0)


@pytest.mark.parametrize("scale", ["linear", "geometric"])
def test_a_failing_lo_ends_the_search_without_descend(scale):
    assert bisect_next_load([(1.5, False)], 1.5, 8, scale=scale) is None
    assert bisect_next_load([(1.5, False)], 1.5, 8, scale=scale, descend=0) is None


def test_a_failing_lo_steps_down_then_bisects_the_new_bracket():
    search = {"lo": 1.5, "hi": 8, "rel_tol": 0.1, "scale": "geometric", "descend": 2}
    assert bisect_next_load([(1.5, False)], **search) == 0.75
    assert bisect_next_load([(1.5, False), (0.75, False)], **search) == 0.375
    # The floor is lo / step**descend: nothing below it, and the search gives up.
    assert bisect_next_load([(1.5, False), (0.75, False), (0.375, False)], **search) is None
    # A pass below lo brackets the knee between it and the lowest failure.
    nxt = bisect_next_load([(1.5, False), (0.75, True)], **search)
    assert nxt == pytest.approx((1.5 * 0.75) ** 0.5)
    nxt = bisect_next_load([(1.5, False), (0.75, False), (0.375, True)], **search)
    assert nxt == pytest.approx((0.75 * 0.375) ** 0.5)
    with pytest.raises(ValueError):
        bisect_next_load([], 1, 8, descend=-1)


def test_descend_finds_goodput_the_ci_rule_hid_at_lo():
    # Smoke run 51ad57b0: with a conservative verdict a workload failed its first point;
    # the knee is really at 1.2, so a search from 1.5 found no goodput at all.
    def run(**search) -> list[tuple[float, bool]]:
        history: list[tuple[float, bool]] = []
        while (load := bisect_next_load(history, **search)) is not None and len(history) < 6:
            history.append((load, load <= 1.2))
        return history

    base = {"lo": 1.5, "hi": 8, "rel_tol": 0.1, "scale": "geometric"}
    assert run(**base) == [(1.5, False)]
    with_descent = run(**base, descend=2)
    passing = max(load for load, met in with_descent if met)
    assert 1.2 / 1.1 <= passing <= 1.2


def _points_above(threshold: float, **search) -> int:
    history: list[tuple[float, bool]] = []
    while (load := bisect_next_load(history, **search)) is not None and len(history) < 6:
        history.append((load, load <= threshold))
    return sum(1 for load, _ in history if load > threshold)


def test_geometric_search_spends_fewer_points_in_overload_than_linear():
    # The first RunPod sweep (058128e9): code-completion on 1x L40S failed at 6.88 req/s
    # and passed at 1; the old linear search over [1, 48] spent 4 of 5 points above that.
    linear = _points_above(5.0, lo=1, hi=48, rel_tol=0.1)
    geometric = _points_above(5.0, lo=1, hi=8, rel_tol=0.1, scale="geometric")
    assert linear >= 4
    assert geometric <= 2


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


def test_aggregate_runs_ci_per_metric():
    runs = [_run(0.1), _run(0.2), _run(0.3)]
    a = aggregate_runs(runs)
    assert a.n_runs == 3 and a.trusted
    ttft = a.get("ttft_ms.p95")
    expected = log_mean_ci([100.0, 200.0, 300.0])
    assert ttft.method == "log_t"
    assert ttft.mean == pytest.approx(6_000_000 ** (1 / 3))  # geometric mean
    assert ttft.lo == pytest.approx(expected.lo)
    assert ttft.hi == pytest.approx(expected.hi)
    assert 0 < ttft.lo < ttft.mean < ttft.hi
    assert a.get("error_rate").method == "t_clipped"
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
        # geometric mean 238 ms is within 250 ms, but the CI upper bound is not
        (4.0, aggregate_runs([_run(0.20), _run(0.24), _run(0.28)])),
    ]
    g = find_goodput(slo, points, LoadMode.OPEN_LOOP)
    assert g.max_load == 2.0
    assert g.first_failing_load == 4.0
    at4 = points[2][1].get("ttft_ms.p95")
    assert at4.mean == pytest.approx((200 * 240 * 280) ** (1 / 3))
    assert at4.hi > 250


@pytest.mark.parametrize(
    ("key", "scale"),
    [
        ("ttft_ms.p95", "positive"),
        ("itl_ms.p50", "positive"),
        ("throughput.output_tok_s", "positive"),
        ("goodput.output_tok_s", "positive"),
        ("ttft_ms.n", "non_negative"),
        ("error_rate", "proportion"),
        ("goodput.slo_attainment", "proportion"),
        ("server.prefix_cache_hit_rate", "proportion"),
        ("gpu.overall.utilization_mean_pct", "percent"),
        ("counts.error", "non_negative"),
        ("server.running_mean", "non_negative"),
        ("something.new", "unbounded"),
    ],
)
def test_ci_rule_per_metric(key, scale):
    assert ci_rule(key).scale == scale


def test_estimate_metric_methods_and_zero_fallback():
    assert estimate_metric("ttft_ms.p95", [3.0, 40.0]).method == "log_t"
    # a repetition with zero goodput cannot go on the log scale: clipped arithmetic
    fallback = estimate_metric("goodput.output_tok_s", [0.0, 900.0])
    assert fallback.method == "t_clipped" and fallback.lo == 0.0
    err = estimate_metric("error_rate", [0.0, 0.5])
    assert err.method == "t_clipped" and err.lo == 0.0 and err.hi == 1.0
