import pytest

from loom_bench.metrics.prometheus import summarize_scrapes
from loom_bench.metrics.summary import RunSummary, flatten_metrics, summarize_run
from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.slo import Slo

OK, ERROR, TIMEOUT, ABORTED = (
    RequestStatus.OK,
    RequestStatus.ERROR,
    RequestStatus.TIMEOUT,
    RequestStatus.ABORTED,
)


def rec(rid, status=OK, sent=0.0, first=None, fin=None, out=None, inp=None, **kw):
    return RequestRecord(
        request_id=rid,
        status=status,
        sent_at_s=sent,
        first_token_at_s=first,
        finished_at_s=fin,
        completion_tokens=out,
        prompt_tokens=inp,
        **kw,
    )


RECORDS = [
    # warmup: absurd values that would dominate every metric if counted
    rec("w", sent=-5.0, first=5.0, fin=50.0, out=10_000, inp=10_000, warmup=True),
    rec("r1", sent=0.0, first=0.1, fin=1.1, out=11, inp=100, cached_prompt_tokens=50,
        itl_s=[0.1] * 10, scheduled_at_s=0.0),
    rec("r2", sent=1.0, first=1.2, fin=3.2, out=21, inp=200, cached_prompt_tokens=0,
        itl_s=[0.1] * 20, scheduled_at_s=0.98),
    rec("r3", sent=2.0, first=2.3, fin=2.3, out=1, inp=100, scheduled_at_s=1.96),
    rec("e1", status=ERROR, sent=3.0, http_status=500),
    rec("t1", status=TIMEOUT, sent=4.0),
    rec("a1", status=ABORTED, sent=9.0, first=9.5),
]  # fmt: skip


def test_counts_error_rate_and_warmup_exclusion():
    s = summarize_run(RECORDS, window_s=10.0, gpus=2)
    assert s.n_total == 6
    assert s.n_warmup_excluded == 1
    assert s.counts == {"ok": 3, "error": 1, "timeout": 1, "aborted": 1}
    assert s.error_rate == pytest.approx(2 / 5)  # aborted in neither numerator nor denominator


def test_latency_percentiles_hand_computed():
    s = summarize_run(RECORDS, window_s=10.0, gpus=2)
    # TTFT over ok requests: [100, 200, 300] ms
    assert s.ttft_ms.n == 3
    assert s.ttft_ms.p50 == pytest.approx(200)
    assert s.ttft_ms.p90 == pytest.approx(280)  # 200 + 0.8 * 100
    assert s.ttft_ms.p99 == pytest.approx(298)
    assert s.ttft_ms.mean == pytest.approx(200)
    assert s.ttft_ms.max == pytest.approx(300)
    # TPOT undefined for the 1-token request; both others decode at 100 ms/token
    assert s.tpot_ms.n == 2
    assert s.tpot_ms.p95 == pytest.approx(100)
    # ITL pools all gaps of ok requests
    assert s.itl_ms.n == 30
    assert s.itl_ms.p99 == pytest.approx(100)
    assert s.e2e_ms.p50 == pytest.approx(1100)
    # client queue delay: [0, 20, 40] ms
    assert s.queue_delay_ms.p50 == pytest.approx(20)
    assert s.queue_delay_ms.max == pytest.approx(40)
    # per-request output tok/s = out / e2e: 10, 9.545..., 3.33...
    assert s.per_request_output_tok_s.p50 == pytest.approx(21 / 2.2)
    assert s.per_request_output_tok_s.min == pytest.approx(1 / 0.3)


def test_window_throughput_and_cache_fraction():
    s = summarize_run(RECORDS, window_s=10.0, gpus=2)
    t = s.throughput
    assert t.request_rate == pytest.approx(0.3)
    assert t.output_tok_s == pytest.approx(3.3)  # (11 + 21 + 1) / 10
    assert t.input_tok_s == pytest.approx(40.0)
    assert t.total_tok_s == pytest.approx(43.3)
    assert t.output_tok_s_per_gpu == pytest.approx(1.65)
    # r3 reports no cached count, so only r1 and r2 count: 50 / 300
    assert s.cached_prompt_fraction == pytest.approx(1 / 6)
    assert s.requests_missing_usage == 0


def test_window_defaults_to_measured_span():
    s = summarize_run(RECORDS, window_s=None, gpus=1)
    assert s.window_s == pytest.approx(3.2)  # first send 0.0 to last finish 3.2
    assert s.throughput.output_tok_s == pytest.approx(33 / 3.2)


def test_slo_attainment_and_request_goodput():
    slo = Slo(ttft_ms={"p95": 250}, tpot_ms={"p95": 120}, max_error_rate=0.5)
    s = summarize_run(RECORDS, window_s=10.0, gpus=1, slo=slo)
    assert s.goodput is not None
    assert s.goodput.good_requests == 2  # r3 misses TTFT; errors are never good
    assert s.goodput.slo_attainment == pytest.approx(2 / 5)
    assert s.goodput.request_rate == pytest.approx(0.2)
    assert s.goodput.output_tok_s == pytest.approx(3.2)
    assert summarize_run(RECORDS, window_s=10.0, gpus=1).goodput is None


def test_missing_usage_and_empty_runs():
    s = summarize_run([rec("x", sent=0, first=0.1, fin=0.5)], window_s=1.0, gpus=1)
    assert s.requests_missing_usage == 1
    assert s.throughput.output_tok_s == 0
    assert s.per_request_output_tok_s.n == 0

    empty = summarize_run([RECORDS[0]], window_s=5.0, gpus=1)
    assert empty.n_total == 0
    assert empty.error_rate is None
    assert empty.ttft_ms.p95 is None
    assert empty.throughput.request_rate == 0
    assert empty.cached_prompt_fraction is None
    with pytest.raises(ValueError):
        summarize_run([RECORDS[0]], window_s=None, gpus=1)
    with pytest.raises(ValueError):
        summarize_run(RECORDS, window_s=0, gpus=1)
    with pytest.raises(ValueError):
        summarize_run(RECORDS, window_s=1, gpus=0)


def test_json_round_trip_and_flatten():
    server = summarize_scrapes("vllm", [(0.0, "vllm:num_requests_running 4\n")])
    slo = Slo(ttft_ms={"p95": 250}, max_error_rate=0.5)
    s = summarize_run(RECORDS, window_s=10.0, gpus=2, slo=slo, server=server)
    assert RunSummary.model_validate_json(s.model_dump_json()) == s

    flat = flatten_metrics(s)
    assert flat["ttft_ms.p95"] == pytest.approx(s.ttft_ms.p95)
    assert flat["throughput.output_tok_s"] == pytest.approx(3.3)
    assert flat["counts.error"] == 1
    assert flat["goodput.slo_attainment"] == pytest.approx(0.4)
    assert flat["server.running_mean"] == 4
    assert "server.preemptions" not in flat  # None is skipped
    assert not any(k.startswith("slo") or k == "gpus" for k in flat)
