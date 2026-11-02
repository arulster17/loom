# billing

`loom_billing`: the usage-stream consumer, per-org per-minute ledger batches, the
minute balance reconcile, `usage_events` partition upkeep, Stripe webhooks and Checkout,
and a local Stripe stub. Built on `loom_platform`; the contracts it implements are in
[docs/phase1-contracts.md](../../docs/phase1-contracts.md) (§3 ledger, §6 balance model,
§7 usage stream and batches).

## Processes

One image (`services/billing/Dockerfile`, built from the repo root:
`docker build -f services/billing/Dockerfile .`), entrypoint `loom-billing <command>`:

| Command | What | Ports |
|---|---|---|
| `all` (default) | worker + HTTP API in one process: the contract's `billing` service | 8090, metrics 9101 |
| `worker` | consumer, batches, reconcile, partitions, credit listener | metrics 9101 |
| `api` | HTTP API only | 8090, metrics 9101 |
| `stripe-stub` | local Stripe stand-in; starts only with `LOOM_STRIPE_MODE=stub` | 12111 |
| `batch-once`, `reconcile-once`, `partitions-once` | one run of a job (ops) | |

Run several workers if needed: each gets its own consumer name in the group, batches
and reconcile are safe to race.

| Job | When | What |
|---|---|---|
| Consumer | continuous | `XREADGROUP` on `loom:usage` (group `billing`), insert `usage_events` (`ON CONFLICT DO NOTHING`), commit, then `XACK`; poison entries to `loom:usage:dead` with `dead_reason`; missing partition = left pending and retried; `XTRIM MINID` below the oldest entry still needed |
| Batch | each minute, :05 | per org: lock unposted events, post one `usage` transaction for their exact sum keyed `usage:<org>:<minute>`, link the events |
| Reconcile | each minute, :35 | Redis `credit` <- `loom_org_credit_total`; drift check of Redis `usage` vs Postgres; low-balance and auto-recharge hooks |
| Credit listener | continuous | `LISTEN loom_credit_changed` -> `set_credit` for that org |
| Partitions | start, then daily | `loom_ensure_usage_partitions(today, LOOM_PARTITION_MONTHS_AHEAD)` |

**Drift check.** In one MULTI read each org's Redis `usage` (U1) and the stream's last
id; check the group has acked everything up to that id (else skip: billing is behind,
not wrong); read Postgres usage (P); read Redis `usage` again (U2). A healthy system has
U1 <= P <= U2. `P < U1` (`missing_in_postgres`) means events were lost or
dead-lettered; `P > U2` (`redis_behind`) means Redis lost usage. Both are logged at
error with the org id and counted (`loom_billing_usage_drift_detected_total{direction}`);
reconcile does not rewrite `usage`.

## HTTP API (port 8090)

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /stripe/webhook` | `Stripe-Signature` (HMAC-SHA256 of `t.body`, 300 s tolerance) | Stores the event in `stripe_events` and, for a paid Loom credits Checkout session, posts a `purchase` keyed `stripe:<event id>`. A replay answers 200 `duplicate`; a session is credited at most once even across event ids. `charge.refunded` and `charge.dispute.created` post a `refund` (org credit -X) found through the payment's PaymentIntent (`stripe_payments`): refunds by the charge's cumulative `amount_refunded` minus what was already taken back, disputes once per dispute id, never more than the purchase in total; the balance may go negative. Events that cannot be applied (unknown org or payment, not USD) answer 200 and stay `processed_at IS NULL` for an admin. Live-mode events are refused. |
| `POST /internal/checkout-sessions` | `Authorization: Bearer $LOOM_BILLING_INTERNAL_TOKEN` | Called by the web app's server (never the browser) after it has checked the user belongs to the org. Body `{"org_id", "amount_micros", "success_url", "cancel_url", "customer_email"?}`, whole cents within [`LOOM_CHECKOUT_MIN_MICROS`, `LOOM_CHECKOUT_MAX_MICROS`], org `active`. Answers `{"id", "url", "amount_micros"}`; redirect the user to `url`. `success_url` may contain `{CHECKOUT_SESSION_ID}`. 401 bad token, 503 token unset, 400 bad body/amount, 404 unknown org, 409 org not active, 502 Stripe error. |
| `GET /health` | none | Postgres and Redis reachable; `stripe_mode` |

## Stripe: real or stub

`LOOM_STRIPE_MODE` chooses, and the stub is never a fallback:

- `stripe` (the default): real Stripe. Both `STRIPE_SECRET_KEY` and
  `STRIPE_WEBHOOK_SECRET` are required (`all` and `api` refuse to start without them; the
  `worker` never calls Stripe), and only test-mode keys (`sk_test_`/`rk_test_`) are
  accepted; a live key stops the process at startup.
- `stub`: local dev only (compose sets it). Billing and the stub share a webhook signing
  secret that has no default: `LOOM_STRIPE_STUB_WEBHOOK_SECRET`, or a file named by
  `LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE`; compose generates one per stack with
  `loom-devcerts --secret stripe-stub-webhook` (`.dev/certs/stripe-stub-webhook.secret`).
  Before, a missing key silently meant the stub with a secret written in the code, so
  anyone could sign a `checkout.session.completed` and credit any org. Stripe keys set
  together with `stub` are refused.

In stub mode:

1. web -> `POST /internal/checkout-sessions` -> billing -> stub `POST /v1/checkout/sessions`
   (Stripe's wire format) -> `url` = `http://localhost:12111/checkout/cs_test_...`;
2. the user opens the page and clicks Pay;
3. the stub posts a `checkout.session.completed` event, signed with
   `LOOM_STRIPE_STUB_WEBHOOK_SECRET` exactly as Stripe signs, to
   `LOOM_STRIPE_STUB_WEBHOOK_URL`, then redirects to `success_url`;
4. billing verifies, credits the org, refreshes its Redis credit.

Clicking Pay again re-sends the same event id (exercises replay protection).

The stub also refunds (`POST /v1/refunds`, Stripe's API: `payment_intent`, optional
`amount` in cents) and simulates a dispute (`POST /stub/disputes`, JSON
`{"payment_intent", "amount"?}`, not a Stripe API), each delivering its signed event.

Not built yet: auto-recharge charging a saved card; reinstating credit when a dispute is
won (`charge.dispute.closed`).

## Notifications outbox

Redis stream `loom:notify` (`LOOM_NOTIFY_STREAM`), flat entries `type`, `org_id`,
`balance_micros`, `threshold_micros`, `at`:

- `low_balance`: balance (credit - usage) below `LOOM_LOW_BALANCE_MICROS` ($1). Once per
  crossing (marker `loom:org:<org>:lowbal`, cleared when the balance is back above).
- `auto_recharge_due` (+ `amount_micros`): only with `LOOM_AUTO_RECHARGE_ENABLED=true`, for
  an org whose `auto_recharge` row is enabled and whose balance is below its threshold.
  This is the hook point; charging needs a saved Stripe payment method (later).

The web/email layer consumes the stream with its own group and sends the email. In
local dev nothing consumes it; the entries and the `notification.emitted` log line are
the record.

## Settings

Shared `LOOM_*` variables are in the contracts (§2). Billing adds:

| Variable | Default | |
|---|---|---|
| `LOOM_USAGE_GROUP` | `billing` | consumer group |
| `LOOM_USAGE_DEAD_STREAM` | `loom:usage:dead` | dead letters |
| `LOOM_BILLING_CONSUMER` | `billing-<hostname>` | consumer name; keep it stable across restarts so a restart recovers its own pending entries |
| `LOOM_USAGE_READ_COUNT` / `_BLOCK_MS` / `_CLAIM_IDLE_MS` | 500 / 2000 / 60000 | |
| `LOOM_REDIS_SOCKET_TIMEOUT_S` / `LOOM_REDIS_CONNECT_TIMEOUT_S` | 2 / 1 | Redis client timeouts; the read timeout is raised to `_BLOCK_MS` + 3 s so a blocking read is not mistaken for a frozen Redis |
| `LOOM_BATCH_INTERVAL_S` / `LOOM_RECONCILE_INTERVAL_S` | 60 / 60 | |
| `LOOM_PARTITION_INTERVAL_S` / `LOOM_PARTITION_MONTHS_AHEAD` | 86400 / 3 | |
| `LOOM_BILLING_HOST` / `_PORT` / `_METRICS_PORT` | `0.0.0.0` / 8090 / 9101 | metrics port 0 = off |
| `LOOM_BILLING_INTERNAL_TOKEN` | unset (checkout endpoint answers 503) | shared with the web app |
| `LOOM_NOTIFY_STREAM` / `LOOM_LOW_BALANCE_MICROS` | `loom:notify` / 1000000 | |
| `LOOM_AUTO_RECHARGE_ENABLED` | false | |
| `LOOM_CHECKOUT_MIN_MICROS` / `_MAX_MICROS` | 5000000 / 500000000 | $5 to $500 |
| `LOOM_STRIPE_MODE` | `stripe` | `stripe` or `stub` (compose: `stub`) |
| `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` | unset | real Stripe, test mode only; required in `stripe` mode |
| `LOOM_STRIPE_API_BASE` | `https://api.stripe.com` | |
| `LOOM_STRIPE_WEBHOOK_TOLERANCE_S` | 300 | |
| `LOOM_BILLING_MAX_BODY_BYTES` | 262144 | 413 above, by Content-Length or while streaming, before the signature or token body is read in full |
| `LOOM_STRIPE_STUB_URL` | `http://localhost:12111` | where billing calls the stub (compose: `http://stripe-stub:12111`) |
| `LOOM_STRIPE_STUB_PUBLIC_URL` | `http://localhost:12111` | stub: base of the pay page URL the browser opens |
| `LOOM_STRIPE_STUB_WEBHOOK_URL` | `http://localhost:8090/stripe/webhook` | stub: where it posts (compose: `http://billing:8090/stripe/webhook`) |
| `LOOM_STRIPE_STUB_WEBHOOK_SECRET` / `_FILE` | unset (required in `stub` mode) | shared by stub and billing, at least 24 characters; compose: the file from `loom-devcerts --secret stripe-stub-webhook` |
| `LOOM_STRIPE_STUB_HOST` / `_PORT` | `0.0.0.0` / 12111 | |
| `LOOM_LOG_LEVEL` | `INFO` | |
| `LOOM_ENV` | unset = `prod` (compose: `dev`) | outside dev every command refuses to start with compose's database password or `LOOM_STRIPE_MODE=stub`, and `all`/`api` also without a non-default `LOOM_BILLING_INTERNAL_TOKEN` |

## Observability

Logs are JSON lines on stderr (`event` plus fields; org ids, amounts, stream ids). No
prompts or completions reach billing; webhook payloads go to `stripe_events`, never to
logs; dead-letter reasons carry field names, not values. Prometheus metrics are
`loom_billing_*` (`metrics.py`): entries by result, pending and lag, batch totals,
drift, webhooks by result, checkout sessions, notifications, job errors.

## Tests

```bash
docker compose up -d postgres redis
set -o pipefail; uv run pytest -q services/billing/tests
```
