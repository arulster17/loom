"""Prometheus exposition of simulator state under vLLM V1 metric names."""

from __future__ import annotations

from collections.abc import Iterator

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
    Metric,
)
from prometheus_client.registry import Collector

from loom_bench.mock.sim import Simulator

# The only place exposed names live; counters carry their `_total` sample suffix.
METRIC_NAMES = {
    "num_running": "vllm:num_requests_running",
    "num_waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "num_preemptions": "vllm:num_preemptions_total",
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "request_success": "vllm:request_success_total",
    "queue_time": "vllm:request_queue_time_seconds",
}
MODEL_LABEL = "model_name"
FINISH_REASON_LABEL = "finished_reason"

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class _SimCollector(Collector):
    def __init__(self, sim: Simulator, model_name: str) -> None:
        self.sim = sim
        self.model_name = model_name

    def collect(self) -> Iterator[Metric]:
        sim, stats, labels = self.sim, self.sim.stats, [self.model_name]
        for key, doc, value in (
            ("num_running", "Number of requests in model execution batches.", sim.num_running),
            ("num_waiting", "Number of requests waiting to be processed.", sim.num_waiting),
            ("kv_cache_usage", "KV-cache usage. 1 means 100 percent usage.", sim.kv.usage),
        ):
            gauge = GaugeMetricFamily(METRIC_NAMES[key], doc, labels=[MODEL_LABEL])
            gauge.add_metric(labels, value)
            yield gauge

        for key, doc, value in (
            ("num_preemptions", "Cumulative number of preemptions.", stats.num_preemptions),
            (
                "prefix_cache_queries",
                "Prefix cache queries, in tokens.",
                stats.prefix_cache_queries,
            ),
            ("prefix_cache_hits", "Prefix cache hits, in tokens.", stats.prefix_cache_hits),
            ("prompt_tokens", "Number of prefill tokens processed.", stats.prompt_tokens),
            ("generation_tokens", "Number of generation tokens.", stats.generation_tokens),
        ):
            counter = CounterMetricFamily(METRIC_NAMES[key], doc, labels=[MODEL_LABEL])
            counter.add_metric(labels, value)
            yield counter

        success = CounterMetricFamily(
            METRIC_NAMES["request_success"],
            "Count of successfully processed requests.",
            labels=[MODEL_LABEL, FINISH_REASON_LABEL],
        )
        for reason, count in sorted(stats.request_success.items()):
            success.add_metric([self.model_name, reason], count)
        yield success

        hist = stats.queue_time
        cumulative, buckets = 0, []
        for bound, count in zip([*hist.bounds, float("inf")], hist.counts, strict=True):
            cumulative += count
            buckets.append(("+Inf" if bound == float("inf") else repr(bound), cumulative))
        queue = HistogramMetricFamily(
            METRIC_NAMES["queue_time"],
            "Histogram of time spent in WAITING phase for request.",
            labels=[MODEL_LABEL],
        )
        queue.add_metric(labels, buckets, sum_value=hist.total)
        yield queue


def render_metrics(sim: Simulator, model_name: str) -> bytes:
    registry = CollectorRegistry(auto_describe=False)
    registry.register(_SimCollector(sim, model_name))
    return generate_latest(registry)
