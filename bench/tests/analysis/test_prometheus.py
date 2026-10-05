import math

import pytest

from loom_bench.metrics.prometheus import parse_prometheus, summarize_scrapes

MODEL = 'engine="0",model_name="Qwen/Qwen3-8B"'


def vllm_scrape(
    running: float,
    waiting: float,
    kv: float,
    preemptions: float,
    queries: float,
    hits: float,
    queue_sum: float,
    queue_count: float,
) -> str:
    """Shaped like vLLM V1 /metrics (prometheus_client output, trimmed)."""
    return f"""\
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{{generation="0"}} 9112.0
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{{MODEL}}} {running}
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{{MODEL}}} {waiting}
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{{{MODEL}}} {kv}
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{{{MODEL}}} {preemptions}
vllm:num_preemptions_created{{{MODEL}}} 1.7596e+09
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{{{MODEL}}} {queries}
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{{{MODEL}}} {hits}
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{{{MODEL}}} 123456.0
# HELP vllm:request_queue_time_seconds Histogram of time spent in WAITING phase for request.
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_sum{{{MODEL}}} {queue_sum}
vllm:request_queue_time_seconds_bucket{{{MODEL},le="0.3"}} {queue_count * 0.8}
vllm:request_queue_time_seconds_bucket{{{MODEL},le="1.0"}} {queue_count * 0.9}
vllm:request_queue_time_seconds_bucket{{{MODEL},le="+Inf"}} {queue_count}
vllm:request_queue_time_seconds_count{{{MODEL}}} {queue_count}
"""


SGLANG = """\
# HELP sglang:num_running_reqs The number of running requests.
# TYPE sglang:num_running_reqs gauge
sglang:num_running_reqs{model_name="Qwen/Qwen3-8B"} %(running)s
# HELP sglang:num_queue_reqs The number of requests in the waiting queue.
# TYPE sglang:num_queue_reqs gauge
sglang:num_queue_reqs{model_name="Qwen/Qwen3-8B"} %(waiting)s
# HELP sglang:token_usage The token usage.
# TYPE sglang:token_usage gauge
sglang:token_usage{model_name="Qwen/Qwen3-8B"} %(usage)s
# HELP sglang:cache_hit_rate The prefix cache hit rate.
# TYPE sglang:cache_hit_rate gauge
sglang:cache_hit_rate{model_name="Qwen/Qwen3-8B"} %(hit)s
# HELP sglang:gen_throughput The generation throughput (token/s).
# TYPE sglang:gen_throughput gauge
sglang:gen_throughput{model_name="Qwen/Qwen3-8B"} 812.3
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name="Qwen/Qwen3-8B"} 40960.0
"""


def sglang_scrape(running, waiting, usage, hit, queue: tuple[float, float] | None = None) -> str:
    text = SGLANG % {"running": running, "waiting": waiting, "usage": usage, "hit": hit}
    if queue is not None:
        text += (
            "# TYPE sglang:queue_time_seconds histogram\n"
            f'sglang:queue_time_seconds_sum{{model_name="Qwen/Qwen3-8B"}} {queue[0]}\n'
            f'sglang:queue_time_seconds_count{{model_name="Qwen/Qwen3-8B"}} {queue[1]}\n'
        )
    return text


def test_parse_types_labels_and_histogram():
    s = parse_prometheus(vllm_scrape(4, 1, 0.5, 0, 0, 0, 2.0, 10))
    assert s.types["vllm:request_queue_time_seconds"] == "histogram"
    assert s.types["vllm:num_preemptions_total"] == "counter"
    labels, value = s.samples["vllm:num_requests_running"][0]
    assert labels == {"engine": "0", "model_name": "Qwen/Qwen3-8B"}
    assert value == 4
    h = s.histogram("vllm:request_queue_time_seconds")
    assert h is not None
    assert h.sum == 2.0 and h.count == 10
    assert h.buckets == [(0.3, 8.0), (1.0, 9.0), (math.inf, 10.0)]


def test_parse_escapes_timestamps_and_junk():
    text = r"""
# EOF-less exposition with odd lines
req{path="a\"b\\c",note="x\ny",} 3 1700000000000
empty{} 1
bare 2.5e3
nan_metric NaN
not a metric line
broken{x="1" 4
"""
    s = parse_prometheus(text)
    labels, value = s.samples["req"][0]
    assert labels == {"path": 'a"b\\c', "note": "x\ny"}
    assert value == 3
    assert s.value("empty") == 1
    assert s.value("bare") == 2500
    assert s.value("nan_metric") is None
    assert "broken" not in s.samples
    assert "not" not in s.samples


def test_series_reduce_sum_and_mean():
    text = (
        'vllm:num_requests_running{engine="0"} 3\n'
        'vllm:num_requests_running{engine="1"} 5\n'
        'vllm:kv_cache_usage_perc{engine="0"} 0.2\n'
        'vllm:kv_cache_usage_perc{engine="1"} 0.6\n'
    )
    m = summarize_scrapes("vllm", [(0.0, text)])
    assert m.running_mean == 8
    assert m.kv_usage_mean == pytest.approx(0.4)


def test_vllm_summary_from_counter_deltas():
    scrapes = [
        (10.0, vllm_scrape(4, 0, 0.20, 2, 1000, 100, 1.0, 10)),
        (0.0, vllm_scrape(2, 0, 0.10, 2, 0, 0, 0.0, 0)),  # out of order on purpose
        (20.0, vllm_scrape(6, 3, 0.60, 7, 5000, 2100, 9.0, 30)),
    ]
    m = summarize_scrapes("vllm", scrapes)
    assert m.n_scrapes == 3
    assert m.duration_s == 20
    assert m.running_mean == 4 and m.running_max == 6
    assert m.waiting_mean == 1 and m.waiting_max == 3
    assert m.kv_usage_mean == pytest.approx(0.3)
    assert m.kv_usage_max == pytest.approx(0.6)
    assert m.preemptions == 5
    assert m.prefix_cache_hit_rate == pytest.approx(2100 / 5000)
    assert m.queue_time_mean_ms == pytest.approx(9.0 / 30 * 1000)
    assert m.missing == []


def test_counter_reset_is_handled():
    scrapes = [
        (0.0, vllm_scrape(1, 0, 0.1, 5, 100, 10, 1.0, 10)),
        (1.0, vllm_scrape(1, 0, 0.1, 8, 200, 30, 2.0, 20)),
        # server restarted: counters start from zero again
        (2.0, vllm_scrape(1, 0, 0.1, 2, 50, 25, 0.5, 5)),
    ]
    m = summarize_scrapes("vllm", scrapes)
    assert m.preemptions == 3 + 2
    assert m.prefix_cache_hit_rate == pytest.approx((20 + 25) / (100 + 50))
    assert m.queue_time_mean_ms == pytest.approx((1.0 + 0.5) / (10 + 5) * 1000)


def test_pre_v1_vllm_names_are_reported_missing_not_guessed():
    old = (
        "vllm:gpu_cache_usage_perc 0.25\n"
        "vllm:gpu_prefix_cache_hit_rate 0.7\n"
        "vllm:num_requests_running 1\n"
    )
    m = summarize_scrapes("vllm", [(0.0, old), (1.0, old)])
    assert m.kv_usage_mean is None
    assert m.prefix_cache_hit_rate is None
    assert {"kv_usage", "waiting", "preemptions", "queue_time"} <= set(m.missing)


def test_mock_uses_vllm_names():
    m = summarize_scrapes("mock", [(0.0, vllm_scrape(3, 1, 0.5, 0, 0, 0, 0, 0))])
    assert m.running_mean == 3


def test_sglang_summary_without_queue_histogram():
    scrapes = [
        (0.0, sglang_scrape(8, 2, 0.30, 0.50)),
        (5.0, sglang_scrape(12, 0, 0.50, 0.60)),
    ]
    m = summarize_scrapes("sglang", scrapes)
    assert m.running_mean == 10 and m.running_max == 12
    assert m.waiting_mean == 1
    assert m.kv_usage_mean == pytest.approx(0.4)
    assert m.prefix_cache_hit_rate == pytest.approx(0.55)
    assert m.preemptions is None
    assert m.queue_time_mean_ms is None
    assert set(m.missing) == {"preemptions", "queue_time"}


def test_sglang_queue_histogram_when_present():
    scrapes = [
        (0.0, sglang_scrape(1, 0, 0.1, 0.0, queue=(1.0, 4))),
        (5.0, sglang_scrape(1, 0, 0.1, 0.0, queue=(3.0, 8))),
    ]
    assert summarize_scrapes("sglang", scrapes).queue_time_mean_ms == pytest.approx(500)


def test_missing_everything_never_crashes():
    for scrapes in ([], [(0.0, "")], [(0.0, "garbage {{{\n\x00")]):
        m = summarize_scrapes("vllm", scrapes)
        assert m.running_mean is None
        assert m.preemptions is None
        assert m.prefix_cache_hit_rate is None
        assert len(m.missing) == 6


def test_counters_need_two_scrapes_and_progress():
    one = summarize_scrapes("vllm", [(0.0, vllm_scrape(1, 0, 0.1, 5, 100, 10, 1, 10))])
    assert one.preemptions is None and one.queue_time_mean_ms is None
    flat = vllm_scrape(1, 0, 0.1, 5, 100, 10, 1, 10)
    idle = summarize_scrapes("vllm", [(0.0, flat), (1.0, flat)])
    assert idle.preemptions == 0
    assert idle.prefix_cache_hit_rate is None  # no queries in the window
    assert idle.queue_time_mean_ms is None


def test_unknown_engine():
    with pytest.raises(ValueError):
        summarize_scrapes("trtllm", [])
