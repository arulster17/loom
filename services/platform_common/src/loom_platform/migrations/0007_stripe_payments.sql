-- The payment behind each credited Checkout session. Refund and dispute events name the
-- PaymentIntent, not the session: this maps them back to the org and the purchase they
-- claw back, and keeps the running totals that make clawbacks cumulative-safe.
CREATE TABLE stripe_payments (
    payment_intent          text PRIMARY KEY,
    session_id              text NOT NULL UNIQUE,
    org_id                  uuid NOT NULL REFERENCES orgs (id),
    purchase_transaction_id uuid NOT NULL REFERENCES ledger_transactions (id),
    amount_micros           bigint NOT NULL CHECK (amount_micros > 0),
    -- Stripe's cumulative `amount_refunded` of the charge, as last applied.
    refunded_micros         bigint NOT NULL DEFAULT 0 CHECK (refunded_micros >= 0),
    -- Credit taken back by `refund` transactions (refunds and disputes), at most the
    -- purchase.
    clawed_back_micros      bigint NOT NULL DEFAULT 0
        CHECK (clawed_back_micros BETWEEN 0 AND amount_micros),
    created_at              timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX stripe_payments_org ON stripe_payments (org_id);

-- Purchases credited before this table existed: the session's PaymentIntent is in the
-- stored checkout event (purchase idempotency key 'stripe:<event id>').
INSERT INTO stripe_payments
    (payment_intent, session_id, org_id, purchase_transaction_id, amount_micros)
SELECT ev.payload -> 'data' -> 'object' ->> 'payment_intent', t.external_ref, t.org_id, t.id,
       en.amount_micros
FROM ledger_transactions t
JOIN stripe_events ev ON 'stripe:' || ev.id = t.idempotency_key
JOIN ledger_entries en
  ON en.transaction_id = t.id AND en.account_id = loom_org_credit_account(t.org_id)
WHERE t.kind = 'purchase'
  AND t.external_ref IS NOT NULL
  AND ev.payload -> 'data' -> 'object' ->> 'payment_intent' IS NOT NULL
ON CONFLICT DO NOTHING;
