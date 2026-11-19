# Stripe test mode: runbook

Operating the Stripe webhook (`POST https://hooks.<domain>/stripe/webhook`, spec
`docs/phase1-m3-spec.md` §8) on the deployed box. Stripe stays in **test mode** through
Phase 1; live mode needs the prerequisites of spec §8.13 first (notably an Events-API sync,
LM1). Alarms named below are in spec §5.3; the commands run on the box inside the billing
container (`docker compose -p <project> exec billing loom-billing ...`).

Facts the steps rely on:

- Stripe retries a delivery that does not get a 2xx: in live mode for up to three days, **in
  test mode only a few times over a few hours** (B7 records the exact limit; write it here
  when checked: _not yet checked_). Each retry and each resend is signed afresh.
- billing answers 200 only after COMMIT. 500 means "stored unapplied with `last_error`",
  503 means "nothing stored". Both make Stripe retry.
- Resending more than needed is harmless: the event id is applied at most once (`duplicate`).

## Missed deliveries and resend

After any of S2a, S2b, S2c, S5, S1a (a handler failure), H1, H2 or H3 (the table in spec
§8.5), and **once the cause is fixed**:

1. In the Stripe dashboard (test mode, the endpoint's event deliveries) list the events of
   the five subscribed types (`checkout.session.completed`,
   `checkout.session.async_payment_succeeded`, `checkout.session.async_payment_failed`,
   `charge.refunded`, `charge.dispute.created`) whose deliveries did not get a 2xx, from one
   hour before the alarm until now.
2. Resend each to the endpoint: the dashboard's **Resend**, or
   `stripe events resend <event id> --webhook-endpoint <we_...>` with **the operator's own
   key**, never billing's (billing's restricted key has Checkout Sessions write only).
3. Check that each answered 200 and that
   `loom-billing stripe-event list --unapplied` shows nothing new.

An event past Stripe's resend limit is lost to Loom (spec §8.10); say so in the incident
note. Not covered by any alarm (T26): the endpoint disabled or edited in Stripe, a fault only
on Stripe's side, an expired or invalid certificate: found in the dashboard or by a customer,
then resent.

### By alarm

- **S2a** (`signature_mismatch`): the secret on the box is not the endpoint's. Compare
  `STRIPE_WEBHOOK_SECRET` (Secrets Manager `loom/platform/stripe`) with the endpoint's signing
  secret in the dashboard, fix it, redeploy billing, then resend what Stripe gave up on.
- **S2b** (`stale_timestamp`): the box's clock is off by more than 300 s. Check chrony
  (Amazon Time Sync), fix the clock; Stripe's retries succeed afterwards; resend the rest.
- **S2c** (500/503 from the webhook): read billing's logs (`stripe.event_failed`,
  `stripe.webhook_unavailable`). Postgres down or slow also shows as S5. After the fix,
  resend per the steps above.
- **S4** (handler failure): read the event's `last_error` first
  (`loom-billing stripe-event list --unapplied`). A **lock or statement timeout** is a
  transient failure, recorded like a handler bug (spec §8.10); the retry job has usually
  applied the event by the next minute and nothing else is needed. Any other error is a
  handler bug on that payload shape: see "Poison event".
- **S5** (retry job stopped): Postgres is down, or the job fails every minute. Check
  `docker compose -p <project> logs billing | grep stripe.retry_run_ended`.
- **S1a / S1b** (stuck events): see below.

## Stuck events: list, resolve, never hand-adjust

`loom-billing stripe-event list [--unapplied]` shows stored events, newest first, with
`result`, `last_error`, `error_count` and `stale_reported_at`.

`loom-billing stripe-event resolve <event id> --admin-email <admin> --reason "<why>"` closes
one event in a single audited transaction (`admin_audit_log`, action `stripe_event.resolve`,
the reason in `details`). **It posts nothing, ever**: no credit, no clawback. Once resolved
no retry or redelivery applies the event, with one exception (the reopen sweep, below). It
is refused (exit 2, nothing changed) for a non-admin, an unknown or already processed
event, a clawback whose PaymentIntent is not a non-empty string, and a clawback whose
payment is already credited. If the event row is locked (the webhook or the retry job is
working on it) or the database is slow, `resolve` prints `busy, retry: ...` to stderr and
exits **75**, nothing changed: run it again in a minute.

Which case is which:

- **Poison event** (a handler bug: `last_error` names an exception on a payload shape): do
  **not** resolve. Fix forward: a release that fixes the handler lets the retry job apply it
  within a minute of the deploy. For a clawback whose payment exists this is the only exit
  (`resolve` refuses it), so a refund or dispute of a credited payment is never dropped by
  hand.
- **A credit event that cannot be applied** (unknown org, wrong currency, no amount): none
  can come from a Loom checkout on a Loom-only sandbox, so in practice it is an operator's
  own test or a payment made by hand in the dashboard: **resolve** it; nothing is credited.
  If a real customer's payment ever lands here, do **not** resolve it: a release that fixes
  the cause applies it (it stays unapplied until then). M3 has no hand credit for a Stripe
  payment.
- **A clawback with no payment** (a refund of a payment Loom never sold, e.g. made in the
  dashboard, or of a purchase that is itself still stuck): **resolve** it. Nothing is taken
  back now. If its purchase is credited later, the retry job's sweep reopens the clawback
  (its `last_error` reads `reopened: payment credited after resolve`) and the handler takes
  the amount back within about a minute. **Never make a hand adjustment for a resolved
  clawback whose payment may still arrive:** the sweep would take it a second time.
- Other event types are stored as `ignored` (processed) and never stuck.

A stuck event is reported once (S1a) 15 minutes after it arrived; S1b stays above 0 until
every reported event is resolved or applied. With more than 500 unapplied events the oldest
are not re-run until the backlog is worked down (they are all stale).

### Won disputes and refunds after them (T18)

A dispute that Loom wins does not reinstate the credit by itself; an admin credit adjustment
does. That adjustment does not lower the payment's clawed-back total, so if the same charge
is later refunded the refund takes nothing back. If a won-dispute adjustment was made and
the charge is later refunded, the admin **reverses the adjustment by hand**.
A refund that later fails (`charge.refund.updated`, not subscribed) keeps its clawback: the
customer is short until an admin adjustment.

## Rolling the webhook secret (T15)

1. In the Stripe dashboard, roll the endpoint's signing secret and choose an overlap window
   (Stripe signs every delivery with both secrets during it).
2. Put the new `whsec_` into Secrets Manager `loom/platform/stripe`
   (`STRIPE_WEBHOOK_SECRET`).
3. Redeploy billing (`loom-deploy apply` with the current release) **at any point inside
   the overlap window**: every delivery stays verifiable, since any matching `v1` is enough.
4. Check `loom_billing_stripe_webhooks_total{result="signature_mismatch"}` did not move and
   a test event answers 200.

Rolling `STRIPE_SECRET_KEY` (the restricted `rk_test_` key, Checkout Sessions write only)
takes effect at the next billing restart; Checkout creation is the only call that uses it.

## Checks and limits worth knowing

- `GET https://hooks.<domain>/stripe/health` answers
  `{"listener":"stripe_webhook","status":"ok"}`: the Route 53 health check (H3). It proves
  DNS, TLS handshake, Caddy routing and billing's listener, not Postgres, a valid
  certificate or Stripe's endpoint being enabled.
- The sandbox is Loom's alone. A payment made by other means in it yields stuck events to
  resolve (above).
- A payload that Postgres `jsonb` refuses (a NUL character, a lone surrogate) cannot be
  stored: 503 on every attempt (S2c), lost once Stripe gives up (spec §8.10).
- The webhook has no overall deadline; with the default timeouts a delivery is answered
  within about 30 s at worst (`LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S`, `_LOCK_TIMEOUT_MS`,
  `_STATEMENT_TIMEOUT_MS`).
