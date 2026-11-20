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
STRIPE_UNAPPLIED = Gauge(
    P + "stripe_unapplied_events",
    "Stored Stripe events not applied yet (e.g. a refund before its purchase): alert if it stays",
)
# S2c: a delivery answered 500 (stored with its error) or 503 (nothing stored).
WEBHOOK_ERRORS = Counter(
    P + "stripe_webhook_errors_total", "Stripe webhook deliveries answered 500 or 503", ["status"]
)
# S4: an attempt on one event raised; counted on the write that makes error_count 1.
HANDLER_FAILURES = Counter(
    P + "stripe_handler_failures_total", "Stripe event attempts that raised", ["path"]
)
# S1a: an event became stale (unapplied for LOOM_STRIPE_STALE_AFTER_S), once per event.
STRIPE_STALE = Counter(
    P + "stripe_unapplied_stale_total", "Stripe events that became stuck", ["type"]
)
# S1b: unapplied events with stale_reported_at set.
STRIPE_STALE_EVENTS = Gauge(
    P + "stripe_unapplied_stale_events", "Unapplied Stripe events already reported stale"
)
# S5: the Unix time of the last completed retry_unapplied run (that run's `now`).
STRIPE_RETRY_LAST_SUCCESS = Gauge(
    P + "stripe_retry_last_success_seconds", "Unix time of the last completed Stripe retry run"
)
CHECKOUT_SESSIONS = Counter(P + "checkout_sessions_total", "Checkout sessions", ["result"])

NOTIFICATIONS = Counter(P + "notifications_total", "Outbox notifications emitted", ["type"])


# Every label value an alarm reads exists from the start, so the series is there before its
# first increment (spec §5.3, §8.6, §8.11). Importing this module is enough.
SUBSCRIBED_TYPES = (
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "checkout.session.async_payment_failed",
    "charge.refunded",
    "charge.dispute.created",
)
for _type in (*SUBSCRIBED_TYPES, "other"):
    STRIPE_STALE.labels(_type)
for _path in ("webhook", "retry", "pending_clawback"):
    HANDLER_FAILURES.labels(_path)
for _status in ("500", "503"):
    WEBHOOK_ERRORS.labels(_status)
for _result in ("bad_signature", "signature_mismatch", "stale_timestamp"):
    WEBHOOKS.labels(_result)
