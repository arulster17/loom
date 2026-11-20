-- M3 Stripe webhooks (docs/phase1-m3-spec.md §8.4, contracts §3).
--
-- stripe_events gains the bookkeeping of the response contract and of stuck events:
--   result            the last outcome (credited, refunded, unapplied, resolved_by_admin, ...)
--   last_error        '<exception class>: <message>' of the last failed attempt, NULL when clean
--   last_error_at     when that attempt failed
--   error_count       failed attempts since the last clean one (S4 counts the write that makes it 1)
--   stale_reported_at when the event was reported stale (alarm S1a), once
-- Every column is nullable or defaulted, so the running M1 code (which names only the M1
-- columns) keeps working. The runtime role already holds SELECT, INSERT, UPDATE on the
-- whole table (0011): the new columns need no further grant. No new table.
--
-- Order matters: stripe_events is altered first and the ledger_transactions index built
-- after, which is the order a running M1 delivery takes its locks (the event row, then the
-- ledger), so this migration cannot deadlock with one.
ALTER TABLE stripe_events
    ADD COLUMN result            text,
    ADD COLUMN last_error        text,
    ADD COLUMN last_error_at     timestamptz,
    ADD COLUMN error_count       integer NOT NULL DEFAULT 0,
    ADD COLUMN stale_reported_at timestamptz;

-- The reopen sweep (loom_billing.webhooks.retry_unapplied, step 1) looks up resolved
-- clawbacks by their PaymentIntent; only resolved rows are indexed. The 0009 index of
-- unprocessed rows is reused for the re-run scan.
CREATE INDEX stripe_events_resolved_payment_intent
    ON stripe_events ((payload -> 'data' -> 'object' ->> 'payment_intent'))
    WHERE result = 'resolved_by_admin';

-- L3: at most one purchase per Checkout Session, in the database. The webhook's advisory
-- lock plus look-up make a violation rare; this index catches a race past them. It reaches
-- callers as loom_platform.ledger.IdempotencyConflict (post_transaction wraps every unique
-- violation), the delivery rolls back, is stored with last_error and answers 500, and the
-- retry finds the purchase (already_credited).
--
-- Fails (and the deploy stops at this step) if a database already holds two purchases with
-- one external_ref. None can: the webhook's look-up refuses a second one. If this raises,
-- fix the data by hand (an admin reversal of the extra purchase), then migrate again.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM ledger_transactions
        WHERE kind = 'purchase' AND external_ref IS NOT NULL
        GROUP BY external_ref HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'migration 0012: two or more purchases share an external_ref; '
            'resolve the duplicate purchases by hand before migrating'
            USING ERRCODE = 'unique_violation';
    END IF;
END $$;

CREATE UNIQUE INDEX ledger_transactions_purchase_external_ref
    ON ledger_transactions (external_ref)
    WHERE kind = 'purchase' AND external_ref IS NOT NULL;
