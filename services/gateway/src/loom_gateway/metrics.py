"""Prometheus metrics, on their own registry (served on LOOM_GATEWAY_METRICS_PORT, 9100,
never on the public port)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

_FAST = (0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.1, 0.25)
_SLOW = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300, 600)
_RATE = (1, 5, 10, 20, 40, 60, 80, 100, 150, 200, 400, 1000)


class GatewayMetrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.requests = Counter(
            "loom_gateway_requests",
            "Requests by endpoint, model and HTTP status",
            ["endpoint", "model", "status"],
            registry=r,
        )
        self.errors = Counter(
            "loom_gateway_errors", "API errors returned, by error code", ["code"], registry=r
        )
        self.inflight = Gauge(
            "loom_gateway_inflight_requests",
            "Admitted requests not yet settled",
            ["model"],
            registry=r,
        )
        self.duration = Histogram(
            "loom_gateway_request_duration_seconds",
            "Request duration, receipt to settle",
            ["endpoint", "model"],
            buckets=_SLOW,
            registry=r,
        )
        self.ttft = Histogram(
            "loom_gateway_ttft_seconds",
            "Receipt to the first streamed token sent",
            ["model"],
            buckets=_SLOW,
            registry=r,
        )
        self.tokens_per_s = Histogram(
            "loom_gateway_stream_tokens_per_second",
            "Completion tokens per second after the first token, per stream",
            ["model"],
            buckets=_RATE,
            registry=r,
        )
        self.overhead = Histogram(
            "loom_gateway_overhead_seconds",
            "Time the gateway adds: pre_dispatch = receipt to upstream request sent "
            "(auth, validation, pool, reserve); settle = the settle call",
            ["phase"],
            buckets=_FAST,
            registry=r,
        )
        self.tokens = Counter(
            "loom_gateway_tokens",
            "Engine-reported tokens billed",
            ["model", "kind"],
            registry=r,
        )
        self.cost = Counter(
            "loom_gateway_cost_micros", "Micro-dollars billed", ["model"], registry=r
        )
        self.aborts = Counter(
            "loom_gateway_stream_aborts",
            "Streams that ended early",
            ["model", "reason"],
            registry=r,
        )
        self.upstream_failures = Counter(
            "loom_gateway_upstream_failures",
            "Failed upstream attempts before the first byte",
            ["model", "replica"],
            registry=r,
        )
        self.upstream_retries = Counter(
            "loom_gateway_upstream_retries", "Retries on another replica", ["model"], registry=r
        )
        self.reservation_excess = Counter(
            "loom_gateway_reservation_excess",
            "Requests whose cost exceeded their reservation (alert: contracts §6)",
            ["model"],
            registry=r,
        )
        self.usage_missing = Counter(
            "loom_gateway_usage_missing",
            "Requests settled without engine usage (billed only what was reported)",
            ["model"],
            registry=r,
        )
        self.settle_failures = Counter(
            "loom_gateway_settle_failures",
            "Settle calls that failed after retries (usage lost: alert)",
            registry=r,
        )
        self.wakes = Counter(
            "loom_gateway_pool_wakes",
            "Wake requests published for cold models",
            ["model"],
            registry=r,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
