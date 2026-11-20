"""The pool manager's Prometheus metrics (docs/phase1-m2-spec.md §11), on their own
registry, served on `LOOM_POOLMGR_METRICS_PORT` (9102; 0 = none) by `loom-poolmgr run`.

    loom_poolmgr_pods{model,state}                  pods per state (every state but terminated)
    loom_poolmgr_cold_start_seconds{model,stage}    cold-start stage offsets from the create
    loom_poolmgr_creates_total{model,result}        ok | no_capacity | price_too_high |
                                                    spend_cap | error
    loom_poolmgr_replacements_total{model,why}      pods that failed or were lost
    loom_poolmgr_lease_renewals_total{result}       ok | error
    loom_poolmgr_spend_month_micros                 month-to-date serving spend
    loom_poolmgr_spend_cap_micros                   the monthly cap
    loom_poolmgr_reconcile_seconds                  one reconcile tick
    loom_poolmgr_orphans_terminated_total           loom-serve pods with no row, terminated

and, from M3 (docs/phase1-m3-spec.md §5.3, §6.8, §7):

    loom_poolmgr_ca_expiry_seconds                  seconds until the pod CA's notAfter (alarm P3)
    loom_poolmgr_pool_reason{model,reason}          1 for each model's current pool reason, 0 for
                                                    the others (alarm P2b)
    loom_poolmgr_mock_backend                       1 with LOOM_POOLMGR_MOCK_BACKEND, else 0
"""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

from loom_poolmgr.store import POD_STATES

# The values of `loom_poolmgr_creates_total{result}` and of a pool hash's `reason`
# (contracts §10); every series an alarm reads exists at 0 before it first changes.
CREATE_RESULTS = ("ok", "no_capacity", "price_too_high", "spend_cap", "error")
POOL_REASONS = ("crash_loop", "spend_cap", "price_too_high", "not_configured", "no_capacity")

_COLD_START = (15, 30, 60, 120, 180, 240, 300, 450, 600, 900, 1200, 1800)
_TICK = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)


class PoolmgrMetrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.pods = Gauge(
            "loom_poolmgr_pods", "Serving pods by model and state", ["model", "state"], registry=r
        )
        self.cold_start = Histogram(
            "loom_poolmgr_cold_start_seconds",
            "Cold-start stage offsets from the create, per stage",
            ["model", "stage"],
            buckets=_COLD_START,
            registry=r,
        )
        self.creates = Counter(
            "loom_poolmgr_creates", "Create attempts by result", ["model", "result"], registry=r
        )
        self.replacements = Counter(
            "loom_poolmgr_replacements",
            "Pods that failed or were lost (each one counts toward the crash loop)",
            ["model", "why"],
            registry=r,
        )
        self.lease_renewals = Counter(
            "loom_poolmgr_lease_renewals", "Lease renewals by result", ["result"], registry=r
        )
        self.spend_month = Gauge(
            "loom_poolmgr_spend_month_micros", "Month-to-date serving spend", registry=r
        )
        self.spend_cap = Gauge("loom_poolmgr_spend_cap_micros", "Monthly serving cap", registry=r)
        self.reconcile = Histogram(
            "loom_poolmgr_reconcile_seconds", "One reconcile tick", buckets=_TICK, registry=r
        )
        self.orphans = Counter(
            "loom_poolmgr_orphans_terminated",
            "loom-serve pods with no serving_pods row, terminated",
            registry=r,
        )

        self.ca_expiry = Gauge(
            "loom_poolmgr_ca_expiry_seconds", "Seconds until the pod CA expires", registry=r
        )
        self.pool_reason = Gauge(
            "loom_poolmgr_pool_reason",
            "1 for each model's current pool reason, 0 for the others",
            ["model", "reason"],
            registry=r,
        )
        self.mock_backend = Gauge(
            "loom_poolmgr_mock_backend",
            "1 when running with the mock backend outside dev",
            registry=r,
        )

    def init_models(self, models: Iterable[str]) -> None:
        """Create each model's alarm series at 0 (creates by result, pool reasons)."""
        for model in models:
            for result in CREATE_RESULTS:
                self.creates.labels(model, result)
            for reason in POOL_REASONS:
                self.pool_reason.labels(model, reason).set(0)

    def set_pool_reason(self, model: str, reason: str | None) -> None:
        for r in POOL_REASONS:
            self.pool_reason.labels(model, r).set(1 if r == reason else 0)

    def set_pods(self, counts: dict[str, dict[str, int]]) -> None:
        """`counts[model][state]`; every live state of each model is written (0 if none)."""
        for model, by_state in counts.items():
            for state in POD_STATES:
                if state != "terminated":
                    self.pods.labels(model, state).set(by_state.get(state, 0))


def serve(metrics: PoolmgrMetrics, port: int) -> None:
    """Expose `metrics` on `port` (all interfaces, a daemon thread)."""
    start_http_server(port, registry=metrics.registry)
