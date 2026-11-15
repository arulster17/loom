-- Unapplied Stripe events are looked up when a purchase is credited (pending refunds and
-- disputes of its PaymentIntent) and re-run every minute by the billing worker
-- (loom_billing.webhooks.retry_unapplied). Few rows ever match; index only those.
CREATE INDEX stripe_events_unprocessed ON stripe_events (received_at)
    WHERE processed_at IS NULL;
