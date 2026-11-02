"""Prometheus metrics (served on LOOM_BILLING_METRICS_PORT, 9101). No per-org labels:
org ids go to the logs, not to metric cardinality."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

P = "loom_billing_"

USAGE_ENTRIES = Counter(P + "usage_entries_total", "Stream entries handled", ["result"])
# result: inserted | duplicate | dead_letter | deleted (trimmed before it was read)
USAGE_INSERT_ERRORS = Counter(
    P + "usage_insert_errors_total", "Inserts left pending for a retry", ["reason"]
)
USAGE_PENDING = Gauge(P + "usage_pending", "Entries delivered to the group but not acked")
USAGE_LAG = Gauge(P + "usage_lag", "Entries in the stream not yet delivered to the group")
CONSUMER_ERRORS = Counter(P + "consumer_errors_total", "Consumer loop failures")

BATCH_TRANSACTIONS = Counter(P + "batch_transactions_total", "Usage ledger transactions posted")
BATCH_MICROS = Counter(P + "batch_micros_total", "Micro-dollars moved by usage batches")
BATCH_EVENTS = Counter(P + "batch_events_total", "Usage events covered by batches")
BATCH_SKIPPED = Counter(P + "batch_skipped_total", "Org batches skipped", ["reason"])
BATCH_SECONDS = Histogram(P + "batch_seconds", "Duration of one batch run")

RECONCILE_ORGS = Counter(P + "reconcile_orgs_total", "Org credit totals written to Redis")
RECONCILE_SECONDS = Histogram(P + "reconcile_seconds", "Duration of one reconcile run")
DRIFT_ORGS = Gauge(P + "usage_drift_orgs", "Orgs whose Redis usage disagreed with Postgres")
DRIFT_MICROS = Gauge(
    P + "usage_drift_micros", "Total disagreement in the last reconcile", ["direction"]
)
# direction: missing_in_postgres (events lost on the way) | redis_behind (Redis lost usage)
DRIFT_DETECTED = Counter(P + "usage_drift_detected_total", "Drift findings", ["direction"])
DRIFT_CHECK_SKIPPED = Counter(
    P + "usage_drift_check_skipped_total", "Reconciles that skipped the drift check (lagging)"
)
GROUP_RECREATED = Counter(
    P + "usage_group_recreated_total",
    "Consumer group found missing (Redis lost its data) and recreated: alert",
)
GROUP_MISSING = Counter(
    P + "usage_group_missing_total",
    "Reconciles that found the usage stream or its consumer group missing: alert",
)
DRIFT_HEALED_MICROS = Counter(
    P + "usage_drift_healed_micros_total", "Usage added back to Redis for redis_behind drift"
)
CREDIT_NOTIFICATIONS = Counter(
    P + "credit_notifications_total", "loom_credit_changed notifications applied"
)

JOB_ERRORS = Counter(P + "job_errors_total", "Periodic job failures", ["job"])
PARTITIONS_CREATED = Counter(P + "partitions_created_total", "usage_events partitions created")

WEBHOOKS = Counter(P + "stripe_webhooks_total", "Stripe webhooks", ["result"])
CREDITED_MICROS = Counter(P + "stripe_credited_micros_total", "Purchase credit posted")
CLAWED_BACK_MICROS = Counter(
    P + "stripe_clawed_back_micros_total", "Credit taken back for Stripe refunds and disputes"
)
CHECKOUT_SESSIONS = Counter(P + "checkout_sessions_total", "Checkout sessions", ["result"])

NOTIFICATIONS = Counter(P + "notifications_total", "Outbox notifications emitted", ["type"])
