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
| Stripe retry | at start, then each minute at :50 | `retry_unapplied`: reopen swept clawbacks, re-run unapplied Stripe events newest first (up to 500, no age window), mark events stale after `LOOM_STRIPE_STALE_AFTER_S`; sets `loom_billing_stripe_retry_last_success_seconds` only when a run completes. Calls no Stripe API |
| Partitions | start, then daily | `loom_ensure_usage_partitions(today, LOOM_PARTITION_MONTHS_AHEAD)` |

**Drift check.** In one MULTI read each org's Redis `usage` (U1) and the stream's last
id; check the group has acked everything up to that id, asking again for up to
`LOOM_RECONCILE_DRAIN_WAIT_S` (2 s) while it has not (at most one wait per run, shared by
its chunks of orgs, and none when the group is missing); read Postgres usage (P); in one
MGET read Redis `usage` again (U2) and each org's heal token
(`loom:org:<org>:heal_token`; absent is a value like any other). A healthy system has
U1 <= P <= U2. `P > U2`
(`redis_behind`) means Redis lost usage; it is checked every run, drained stream or not.
`P < U1` (`missing_in_postgres`) means events were lost or dead-lettered; it is checked
only when the group caught up within the wait (else skipped that minute: billing is
behind, not wrong). Both are logged at error with the org id and counted
(`loom_billing_usage_drift_detected_total{direction}`). `redis_behind` is healed, since
an under-count would let the org overspend, by one Lua call that, if the org is loaded
and its heal token is unchanged since the MGET, adds P - U2 to Redis `usage` and sets a
fresh token, and otherwise does nothing (`loom_billing_usage_drift_healed_micros_total`).
The token is a random UUID made by the caller (the reconciler, or `Limiter.init_org`) and
passed to the script, never made by a script. Reconcilers racing each other read the
same token, so only the first heal applies, even when they read different P; every load
of the org from Postgres (`init_org.lua`: the gateway's `noinit` load or any other)
writes a fresh token, so a heal read before a reload does nothing. Settles made since
the MGET stay counted, and a busy org heals at its first run however much settled
between the MGET and the heal (U2 = 900, P = 1000, 150 settled meanwhile: 1150, the true
total). The next run reads afresh. `missing_in_postgres` is not rewritten.

**Redis data loss.** After a FLUSHALL or a failover without AOF the gateway's next
settle recreates `loom:usage` without the consumer group. The consumer answers NOGROUP
by recreating the group at id 0 and carrying on
(`loom_billing_usage_group_recreated_total`, alert); a reconcile that finds the group
missing logs `reconcile.usage_group_missing` (`loom_billing_usage_group_missing_total`).

## HTTP API (port 8090)

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /stripe/webhook` | `Stripe-Signature` (HMAC-SHA256 of `t.body`, 300 s tolerance) | Stores the event in `stripe_events` and applies it in one transaction (spec §8.3): a paid Loom credits Checkout session posts a `purchase` keyed `stripe:<event id>` (one per session, also by a unique index), `charge.refunded` / `charge.dispute.created` post a capped `refund`, other types are `ignored`. Answers 200 only after COMMIT (applied, stored unapplied, or `duplicate`); 400 `bad_signature` (metric result `bad_signature`, `signature_mismatch` or `stale_timestamp`) or `bad_event` (not JSON, no id/type, live mode); 413 over the body limit; **500 `internal_error`** when the Postgres part failed but the event was stored unapplied with `last_error` (a second transaction), **503 `service_unavailable`** when nothing could be stored. Stripe retries any non-2xx. The credit refresh in Redis is awaited for at most 2 s after COMMIT. |
| `GET /stripe/health` | none | `{"listener":"stripe_webhook","status":"ok"}`; no Postgres, Redis or Stripe call. The Route 53 health check (H3) requests it through Caddy. `GET /health` stays internal. |
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

The stub also has `POST /stub/control` (`{"drop_deliveries": bool, "fail_status": int |
null}`: drop or fail its webhook deliveries, for the missed-delivery tests), lists the
events it sent (`GET /stub/events`) and resends one with a fresh signature
(`POST /stub/events/<id>/resend`), as Stripe's resend does.

**The Stripe sandbox is the deployed box's alone** (spec §8.1, D4): no laptop, script or
other stack takes payments through it, so every event its endpoint receives is that
deployment's. Anything else that reaches it (a payment made by hand in the dashboard, say)
stays unapplied, raises alarm S1a and is resolved by an admin (`stripe-event resolve`,
[the runbook](../../docs/runbook-stripe.md)). A laptop uses the stub or a Stripe sandbox
of its own.

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
| `LOOM_RECONCILE_DRAIN_WAIT_S` | 2 | how long reconcile waits for the consumer (once per run; not when the group is missing) before skipping the `missing_in_postgres` check for the minute |
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
| `LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S` | 5 | the pool wait of each Stripe transaction (a delivery, its second transaction, a retry attempt, resolve) |
| `LOOM_STRIPE_LOCK_TIMEOUT_MS` / `LOOM_STRIPE_STATEMENT_TIMEOUT_MS` | 3000 / 5000 | `SET LOCAL lock_timeout` / `statement_timeout` inside each of those transactions; there is no overall deadline (a delivery is answered within about 30 s at worst) |
| `LOOM_STRIPE_STALE_AFTER_S` | 900 | an event unapplied this long since `received_at` is marked stale once (alarm S1a) |
| `LOOM_BRAND_NAME` | unset | the Checkout line item is `<name> credits`, `API credits` when unset |
| `LOOM_STRIPE_STUB_URL` | `http://localhost:12111` | where billing calls the stub (compose: `http://stripe-stub:12111`) |
| `LOOM_STRIPE_STUB_PUBLIC_URL` | `http://localhost:12111` | stub: base of the pay page URL the browser opens |
| `LOOM_STRIPE_STUB_WEBHOOK_URL` | `http://localhost:8090/stripe/webhook` | stub: where it posts (compose: `http://billing:8090/stripe/webhook`) |
| `LOOM_STRIPE_STUB_WEBHOOK_SECRET` / `_FILE` | unset (required in `stub` mode) | shared by stub and billing, at least 24 characters; compose: the file from `loom-devcerts --secret stripe-stub-webhook` |
| `LOOM_STRIPE_STUB_HOST` / `_PORT` | `0.0.0.0` / 12111 | |
| `LOOM_LOG_LEVEL` | `INFO` | |
| `LOOM_ENV` | unset = `prod` (compose: `dev`) | outside dev every command refuses to start with compose's database password, a `LOOM_REDIS_URL` without a password or TLS (`rediss://`), a Redis `maxmemory-policy` other than `noeviction` (checked with `CONFIG GET`; a warning where managed Redis forbids it), or `LOOM_STRIPE_MODE=stub`, and `all`/`api` also without a non-default `LOOM_BILLING_INTERNAL_TOKEN` |

## Observability

Logs are JSON lines on stderr (`event` plus fields; org ids, amounts, stream ids). No
prompts or completions reach billing; webhook payloads go to `stripe_events`, never to
logs; dead-letter reasons carry field names, not values. Prometheus metrics are
`loom_billing_*` (`metrics.py`): entries by result, pending and lag, batch totals,
drift, webhooks by result, checkout sessions, notifications, job errors.

The Stripe alarms read `loom_billing_stripe_webhooks_total{result}` (`signature_mismatch`:
S2a, `stale_timestamp`: S2b), `loom_billing_stripe_webhook_errors_total{status}` (S2c),
`loom_billing_stripe_unapplied_stale_total{type}` (S1a),
`loom_billing_stripe_unapplied_stale_events` (S1b),
`loom_billing_stripe_handler_failures_total{path}` (S4) and
`loom_billing_stripe_retry_last_success_seconds` (S5); every label value starts at 0.

Stuck events: `loom-billing stripe-event list [--unapplied]` and `loom-billing stripe-event
resolve <event id> --admin-email <email> --reason <text>` (audited; posts nothing). See
[docs/runbook-stripe.md](../../docs/runbook-stripe.md).

## Tests

```bash
docker compose up -d postgres redis
set -o pipefail; uv run pytest -q services/billing/tests
```
