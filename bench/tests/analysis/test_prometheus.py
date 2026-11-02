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
        summarize_scrapes("tgi", [])


# TensorRT-LLM /prometheus/metrics (prometheus_client multiprocess output). Constructed
# from tensorrt_llm/metrics/collector.py at v1.2.1 (request histograms only, labelled
# model_name/engine_type) and v1.4.0rc2 (adds the gauges); not captured from a server.
TRTLLM_LABELS = 'model_name="undefined",engine_type="undefined"'


def trtllm_scrape(queue_sum: float, queue_count: float, gauges: dict[str, float] | None = None):
    text = f"""\
# HELP trtllm_request_success_total Count of successfully processed requests.
# TYPE trtllm_request_success_total counter
trtllm_request_success_total{{{TRTLLM_LABELS},finished_reason="length"}} {queue_count}
# HELP trtllm_time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE trtllm_time_to_first_token_seconds histogram
trtllm_time_to_first_token_seconds_sum{{{TRTLLM_LABELS}}} 1.5
trtllm_time_to_first_token_seconds_bucket{{{TRTLLM_LABELS},le="+Inf"}} {queue_count}
trtllm_time_to_first_token_seconds_count{{{TRTLLM_LABELS}}} {queue_count}
# HELP trtllm_request_queue_time_seconds Histogram of time spent in WAITING phase for request.
# TYPE trtllm_request_queue_time_seconds histogram
trtllm_request_queue_time_seconds_sum{{{TRTLLM_LABELS}}} {queue_sum}
trtllm_request_queue_time_seconds_bucket{{{TRTLLM_LABELS},le="0.3"}} {queue_count}
trtllm_request_queue_time_seconds_bucket{{{TRTLLM_LABELS},le="+Inf"}} {queue_count}
trtllm_request_queue_time_seconds_count{{{TRTLLM_LABELS}}} {queue_count}
"""
    for name, value in (gauges or {}).items():
        text += f"# TYPE {name} gauge\n{name}{{{TRTLLM_LABELS}}} {value}\n"
    return text


def test_trtllm_1_2_1_reports_queue_time_and_lists_the_rest_missing():
    m = summarize_scrapes("trtllm", [(0.0, trtllm_scrape(1.0, 10)), (5.0, trtllm_scrape(3.0, 14))])
    assert m.queue_time_mean_ms == pytest.approx(500)
    assert m.running_mean is None and m.kv_usage_mean is None
    assert set(m.missing) == {
        "running", "waiting", "kv_usage", "preemptions", "prefix_cache_hit_rate"
    }  # fmt: skip


def test_trtllm_gauges_of_later_versions_are_read():
    gauges = {
        "trtllm_num_requests_running": 30,
        "trtllm_num_requests_waiting": 2,
        "trtllm_kv_cache_utilization": 0.6,
        "trtllm_kv_cache_hit_rate": 0.25,
    }
    m = summarize_scrapes("trtllm", [(0.0, trtllm_scrape(0.0, 0, gauges))])
    assert (m.running_mean, m.waiting_mean) == (30, 2)
    assert m.kv_usage_mean == pytest.approx(0.6)
    assert m.prefix_cache_hit_rate == pytest.approx(0.25)
    assert "preemptions" in m.missing  # GUARANTEED_NO_EVICT never preempts


# --- speculative decoding (vLLM v0.30.0 SpecDecodingProm) ----------------------------


def spec_scrape(drafts: float, draft_tokens: float, accepted: float, per_pos: list[float]) -> str:
    """The spec-decode families as vLLM v0.30.0 exposes them (prometheus_client Counters:
    `_total` and `_created` samples, the per-position counter labelled `position`)."""
    pos = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
    lines = [
        "# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.",
        "# TYPE vllm:spec_decode_num_drafts_total counter",
        f"vllm:spec_decode_num_drafts_total{{{MODEL}}} {drafts}",
        f"vllm:spec_decode_num_drafts_created{{{MODEL}}} 1.7596e+09",
        "# TYPE vllm:spec_decode_num_draft_tokens_total counter",
        f"vllm:spec_decode_num_draft_tokens_total{{{MODEL}}} {draft_tokens}",
        "# TYPE vllm:spec_decode_num_accepted_tokens_total counter",
        f"vllm:spec_decode_num_accepted_tokens_total{{{MODEL}}} {accepted}",
        f"# TYPE {pos} counter",
        *(f'{pos}{{{MODEL},position="{i}"}} {v}' for i, v in enumerate(per_pos)),
    ]
    return "\n".join(lines) + "\n"


def test_spec_decode_counters_give_acceptance_rate_length_and_positions():
    base = vllm_scrape(4, 0, 0.2, 0, 0, 0, 0, 0)
    scrapes = [
        (0.0, base + spec_scrape(100, 300, 150, [80, 50, 20])),
        (10.0, base + spec_scrape(1100, 3300, 2550, [980, 1050, 520])),
    ]
    m = summarize_scrapes("vllm", scrapes, speculative=True)
    # The window: 1000 drafts of 3 tokens, 2400 accepted (900 + 1000 + 500 by position).
    assert (m.spec_drafts, m.spec_draft_tokens, m.spec_accepted_tokens) == (1000, 3000, 2400)
    assert m.spec_acceptance_rate == pytest.approx(0.8)
    assert m.spec_mean_accepted_length == pytest.approx(3.4)  # 1 + 2400 / 1000
    assert m.spec_acceptance_by_position == pytest.approx({"0": 0.9, "1": 1.0, "2": 0.5})
    assert list(m.spec_acceptance_by_position or {}) == ["0", "1", "2"]
    assert "spec_decode" not in m.missing


def test_spec_positions_sort_numerically_and_sum_over_engines():
    pos = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

    def per_pos(scale: int) -> str:
        return "".join(
            f'{pos}{{engine="{e}",position="{p}"}} {v * scale}\n'
            for e in (0, 1)
            for p, v in ((10, 1), (2, 2), (0, 3))
        )

    first = spec_scrape(0, 0, 0, []) + per_pos(0)
    last = spec_scrape(10, 30, 12, []) + per_pos(1)
    m = summarize_scrapes("vllm", [(0.0, first), (1.0, last)])
    assert list(m.spec_acceptance_by_position or {}) == ["0", "2", "10"]
    # Two engines: position 0 gained 3 + 3 over 10 drafts.
    assert m.spec_acceptance_by_position == pytest.approx({"0": 0.6, "2": 0.4, "10": 0.2})


def test_no_spec_counters_is_silent_unless_the_engine_speculates():
    flat = vllm_scrape(1, 0, 0.1, 0, 0, 0, 0, 0)
    plain = summarize_scrapes("vllm", [(0.0, flat), (1.0, flat)])
    assert plain.spec_drafts is None and plain.spec_mean_accepted_length is None
    assert "spec_decode" not in plain.missing
    spec = summarize_scrapes("vllm", [(0.0, flat), (1.0, flat)], speculative=True)
    assert "spec_decode" in spec.missing


def test_a_window_with_no_drafts_has_counters_but_no_rates():
    s = spec_scrape(5, 15, 9, [4, 3, 2])
    m = summarize_scrapes("vllm", [(0.0, s), (1.0, s)], speculative=True)
    assert (m.spec_drafts, m.spec_draft_tokens, m.spec_accepted_tokens) == (0, 0, 0)
    assert m.spec_acceptance_rate is None and m.spec_mean_accepted_length is None
    assert m.spec_acceptance_by_position is None
    assert "spec_decode" not in m.missing


def test_spec_counter_reset_counts_from_zero():
    scrapes = [
        (0.0, spec_scrape(10, 30, 15, [8, 5, 2])),
        (1.0, spec_scrape(20, 60, 30, [16, 10, 4])),
        (2.0, spec_scrape(5, 15, 6, [4, 2, 0])),  # the engine restarted
    ]
    m = summarize_scrapes("vllm", scrapes, speculative=True)
    assert m.spec_drafts == 15 and m.spec_accepted_tokens == 21
    assert m.spec_mean_accepted_length == pytest.approx(1 + 21 / 15)
    assert m.spec_acceptance_by_position == pytest.approx({"0": 12 / 15, "1": 7 / 15, "2": 2 / 15})


def test_sglang_has_no_spec_mapping():
    s = sglang_scrape(1, 0, 0.1, 0.0) + spec_scrape(5, 15, 9, [4, 3, 2])
    m = summarize_scrapes("sglang", [(0.0, s), (1.0, s)])
    assert m.spec_drafts is None


def test_runs_stored_before_the_spec_fields_still_load():
    from loom_bench.metrics.prometheus import ServerMetrics

    old = summarize_scrapes("vllm", []).model_dump()
    for key in [k for k in old if k.startswith("spec_")]:
        del old[key]
    assert ServerMetrics.model_validate(old).spec_acceptance_rate is None
