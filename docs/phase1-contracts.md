# Phase 1 contracts (M1)

Status: M1, 2026-11-01. The interfaces the gateway, billing worker, web app and mock pod
build against. Changing one means changing this document, the shared code in
`services/platform_common` (`loom_platform`) and its tests together.

Wave 1 wrote this document; wave 2 (gateway, billing, web) added interfaces that were
recorded here at the M1 acceptance ([phase1-m1-acceptance.md](phase1-m1-acceptance.md)),
marked *(wave 2)*: playground keys (§5), the `loom:notify` stream (§6, §7), billing's
internal checkout API (§7), the `/v1/completions` default `max_tokens` (§8), the public
`GET /v1/models` shape (§12), the 502 `upstream_lost` timeout and 503
`balance_unavailable` (§11).

Background: [PLAN.md](PLAN.md) (decisions, the accepted ledger draft) and
[phase1-runpod-plan.md](phase1-runpod-plan.md) (§4 pod security, §7 routing, §8 failure
handling, §13 milestones).

## 1. Layout

| Path | What | Package |
|---|---|---|
| `services/platform_common` | Shared code, schema migrations, Lua scripts, test fixtures | `loom-platform` / `loom_platform` |
| `services/gateway` | Public API (FastAPI + uvloop) | `loom-gateway` / `loom_gateway` |
| `services/billing` | Usage consumer, ledger batches, balance reconcile, Stripe webhooks | `loom-billing` / `loom_billing` |
| `apps/web` | Next.js dashboard, playground, admin (Auth.js, Postgres adapter) | npm |
| `services/podproxy` | The pod proxy: mTLS front of each pod's engine (§9) | `loom-podproxy` / `loom_podproxy` |
| `services/devstack` | Local stack: mock pod launcher, poolsim, dev certificates, load test | `loom-devstack` / `loom_devstack` |
| `bench/` | Benchmark Lab; its mock backend is M1's engine | `loom-bench` |

All Python packages are uv workspace members with their dependencies already locked;
add a dependency only to your own package's `pyproject.toml`. Tests live in
`services/<svc>/tests/` with file names `test_<svc>_*.py` and no `__init__.py` (pytest
imports test modules by base name). Reuse the fixtures:

```python
# services/<svc>/tests/conftest.py
from loom_platform.testing import pg, pg_url, redis_client  # noqa: F401
```

`pg_url` is a fresh, fully migrated database per pytest session; `redis_client` uses db
15. Both skip when `LOOM_TEST_DATABASE_URL` / `LOOM_TEST_REDIS_URL` are unset and the
compose servers are down, and fail when the variable is set. Factories:
`create_user`, `create_org`, `create_api_key`.

## 2. Services, ports, environment

| Service | Port (host) | Notes |
|---|---|---|
| postgres | 5432 | Platform database `loom_platform` (the bench keeps `loom`) |
| redis | 6379 | AOF `everysec`, `maxmemory-policy noeviction` (already in `docker-compose.yml`) |
| migrate | none | One-shot: `loom-migrate --create-database`; others start after it succeeds |
| gateway | 8080 | Public API at `http://localhost:8080/v1`; Prometheus `/metrics` on 9100 |
| billing | 8090 | `POST /stripe/webhook`, `POST /internal/checkout-sessions` (§7), `GET /health`; Prometheus `/metrics` on 9101 |
| web | 3000 | Next.js |
| stripe-stub | 12111 | Local Stripe stand-in (`loom-billing stripe-stub`) |
| mailpit | 8025 (UI), 1025 (SMTP) | Catches the web app's magic-link email |
| mock-pod-1, mock-pod-2 | 8443, not published | One per servable model (`qwen3-8b`, `llama-3.3-70b-instruct-fp8`): mTLS pod proxy; the mock engine listens on `127.0.0.1:8000` inside the container |
| poolsim | none | M1 pool simulator (§10) |
| migrate, devcerts | none | One-shot jobs |

Every published port can be moved with a `LOOM_*_PORT` variable (`docker-compose.yml`).

Settings are `LOOM_*` environment variables, read by `loom_platform.settings.PlatformSettings`
(services subclass it). Defaults are the compose values.

| Variable | Default | Used by |
|---|---|---|
| `LOOM_ENV` | unset = `prod`; compose: `dev` | all services: `dev` or `prod` (anything else refuses to start). Outside dev a service refuses to start on a missing secret or a compose dev default: gateway (DB password, `LOOM_POD_CA_CERT`, client certificate, `LOOM_CORS_ORIGINS`; and it never uses an `http://` replica), billing (DB password, `LOOM_STRIPE_MODE=stub`, internal token missing or the compose value), web (`AUTH_SECRET` missing or the compose value, `AUTH_URL` missing, internal token missing or the compose value) |
| `LOOM_DATABASE_URL` | `postgresql://loom:loom@localhost:5432/loom_platform` | all (web: as `DATABASE_URL`) |
| `LOOM_REDIS_URL` | `redis://localhost:6379/0` | gateway, billing, poolsim |
| `LOOM_MODELS_YAML` | repo `config/models.yaml` | gateway, billing |
| `LOOM_USAGE_STREAM` | `loom:usage` | gateway, billing |
| `LOOM_AUTH_CACHE_TTL_S` | 4 | gateway |
| `LOOM_REQUEST_TIMEOUT_S` / `LOOM_RESERVATION_GRACE_S` | 600 / 60 | gateway (reservation TTL = their sum) |
| `LOOM_MAX_BODY_BYTES` | 1048576 | gateway (413 above) |
| `LOOM_MAX_TOKENS_CAP` | 16384 | gateway |
| `LOOM_DEFAULT_KEY_RPM` / `_TPM` / `_CONCURRENCY` | 60 / 200000 / 8 | gateway |
| `LOOM_DEFAULT_ORG_RPM` / `_TPM` / `_CONCURRENCY` | 120 / 400000 / 16 | gateway |
| `LOOM_SIGNUP_PROMO_MICROS` | 5000000 | web (admin approval) |
| `LOOM_CLOUD` / `LOOM_REGION` | `local` / empty | gateway (usage events) |
| `LOOM_GATEWAY_CLIENT_CERT` / `_KEY`, `LOOM_POD_CA_CERT` | none | gateway (mTLS to pods; required outside `LOOM_ENV=dev`) |
| `LOOM_CORS_ORIGINS` | `http://localhost:3000` | gateway |
| `LOOM_REDIS_SOCKET_TIMEOUT_S` / `LOOM_REDIS_CONNECT_TIMEOUT_S` | 2 / 1 | gateway, billing: a frozen or unreachable Redis fails a call within this (gateway: 503 `service_unavailable`, `/ready` 503); billing raises the read timeout to its blocking read + 3 s |
| `LOOM_SETTLE_SPOOL_PATH` | unset: a per-process temp file; image: `/var/lib/loom-gateway/settle-spool.jsonl` (a volume) | gateway: settles Redis refused, replayed until accepted (§6 "Settle spool"); required outside `LOOM_ENV=dev` |
| `LOOM_SHUTDOWN_GRACE_S` | 10 | gateway: on SIGTERM, how long in-flight requests may finish before open streams are cut (§6 "Shutdown") |
| `AUTH_SECRET`, `AUTH_URL`, `AUTH_GOOGLE_ID` / `_SECRET`, `EMAIL_SERVER` | none | web (Auth.js); `AUTH_URL` is required outside `LOOM_ENV=dev` |
| `LOOM_GOOGLE_EMAIL_LINKING` | `true` | web: a Google sign-in may link to an existing user with the same email; Google sign-in always requires `email_verified` |
| `LOOM_ADMIN_EMAILS` | empty | web: dev bootstrap of the first admin (§4) |
| `LOOM_INTERNAL_TOKEN` / `LOOM_BILLING_INTERNAL_TOKEN` | compose: one shared dev value | web / billing: bearer token of billing's internal API |
| `LOOM_STRIPE_MODE` | `stripe`; compose: `stub` | billing: real Stripe (test keys required) or the local stub (explicit opt-in, never a fallback) |
| `LOOM_STRIPE_STUB_WEBHOOK_SECRET` / `_FILE` | none; compose: `.dev/certs/stripe-stub-webhook.secret` from `loom-devcerts --secret` | billing and stripe-stub in stub mode |
| `LOOM_GATEWAY_URL`, `LOOM_PUBLIC_API_URL`, `LOOM_BILLING_URL` | `http://localhost:8080`, `http://localhost:8080/v1`, `http://localhost:8090` | web: the gateway and billing as the server reaches them; the API base shown to users |
| `LOOM_PLAYGROUND_KEY_SECRET` / `LOOM_PLAYGROUND_KEY_TTL_S` | derived from `AUTH_SECRET` / 3600 | web: playground keys (§5) *(wave 2)* |
| `LOOM_NOTIFY_STREAM` | `loom:notify` | billing: notifications outbox (§7) *(wave 2)* |
| `LOOM_POOLSIM_REPLICAS` | compose: `<model>=<pod>@https://<pod>:8443,...` | poolsim (§10) |
| `LOOM_POOLSIM_TOKEN_DIR`, `_COLD_START_S`, `_IDLE_STOP_S` | `.dev/certs`, 20, 0 (off) | poolsim |
| `LOOM_MOCKPOD_ID`, `LOOM_MOCKPOD_MODELS`, `LOOM_DEV_CERTS` | `mock-pod-1`, `qwen3-8b`, `.dev/certs` | mock pods |
| `LOOM_PODPROXY_ALLOW_IPS` | empty (any source) | pod proxy: allowed source CIDRs |

## 3. Postgres

Migrations are numbered SQL files in `loom_platform/migrations/`, applied in order by
`loom-migrate` (`python -m loom_platform.migrate`), each in one transaction, recorded with
a checksum in `schema_migrations`. Never edit an applied file; add the next number.

Every table of the accepted PLAN.md draft exists as drafted, with these additions:

| Addition | Why |
|---|---|
| Auth.js tables use `uuid` ids (adapter docs use `SERIAL`) | The adapter never supplies an id and returns ids as strings; every platform table references users by uuid. **The web agent must confirm with the adapter version it installs.** |
| `users.role` (`user` / `admin`) | Admin role (§4). |
| `orgs.status` adds `pending_approval` (default) and `rejected`; `approved_at`, `approved_by` | Manual approval of every signup (user decision 2026-10-31). |
| `orgs.rpm_limit`, `tpm_limit`, `concurrency_limit`; `api_keys.concurrency_limit` | Per-org limits and per-key concurrency; NULL = platform default. |
| `usage_events.ledger_transaction_id` | The per-minute usage transaction that debited the event. |
| `usage_events` primary key `(id, created_at)` | A partitioned table's key must hold the partition column. The gateway fixes `created_at` once per request, so `ON CONFLICT (id, created_at) DO NOTHING` de-duplicates redeliveries exactly. |
| `ledger_transactions` is append-only too | It is half of the ledger; corrections are new transactions. |
| `admin_audit_log.actor_user_id` has no foreign key | Deleting a user must not need an UPDATE of the log. |
| `stripe_payments` (migration 0007): PaymentIntent -> session, org, purchase, amount, cumulative refunded and clawed-back micros | Refund and dispute events name the PaymentIntent; this maps them to the org and keeps clawbacks cumulative-safe and capped at the purchase. |

Invariants enforced in the database:

- **Zero-sum**: a deferred constraint trigger checks at commit that each transaction's
  entries sum to 0 and that it has at least 2; entries are non-zero.
- **Append-only**: `UPDATE`, `DELETE` and `TRUNCATE` on `ledger_entries`,
  `ledger_transactions` and `admin_audit_log` raise `insufficient_privilege` for every
  role (only a superuser can disable the triggers).
- **Idempotency**: `ledger_transactions.idempotency_key` is unique.
- **Tenant scope**: `loom_post_transaction` refuses an org account that is not the
  transaction's org.
- Every org gets its `org_credit` ledger account from an insert trigger.

Post only through `loom_post_transaction(kind, idempotency_key, org_id, external_ref,
memo, account_ids uuid[], amounts bigint[]) -> (transaction_id, created)` (Python:
`loom_platform.ledger.post_transaction`). Re-posting a key with the same content returns
the first transaction with `created = false`; other content raises `unique_violation`.

| Kind | org_credit | Counter-account | Idempotency key |
|---|---|---|---|
| purchase | +X | stripe_clearing -X | `stripe:<event id>` |
| usage | -X | revenue +X | `usage:<org id>:<YYYY-MM-DDTHH:MMZ>` (`ledger.usage_batch_key`) |
| promo | +X | promo_expense -X | `promo:signup:<org id>` |
| refund | -X | stripe_clearing +X | `stripe:<event id>` (`charge.refunded`: the new part of the cumulative `amount_refunded`; `charge.dispute.created`: the disputed amount, once per dispute; at most the purchase in total; the balance may go negative) |
| adjustment | +/-X | promo_expense (grant) or revenue (correction) | `admin:<admin_audit_log id>` |

Platform accounts have fixed ids: revenue `...0001`, stripe_clearing `...0002`,
promo_expense `...0003` (`loom_platform.ledger`).

Balance functions: `loom_org_balance(org)` (posted ledger balance),
`loom_org_credit_total(org)` (all org_credit entries except usage),
`loom_org_usage_total(org)` and `loom_key_usage_total(key)` (sum of `usage_events.cost_micros`,
posted or not). Dashboard balance = `credit_total - usage_total` (or the Redis figure, §6).

`usage_events` is partitioned by month (UTC). Migration 0004 creates last month through
12 months ahead; the billing worker calls `loom_ensure_usage_partitions(current_date, 3)`
at start and daily. There is no default partition: an insert outside every partition
fails and the event stays pending in the stream.

NOTIFY channels (delivered on commit):

| Channel | Fires on | Payload | Listener |
|---|---|---|---|
| `loom_auth_changed` | `api_keys` revoked/limits/scopes change; `orgs` status/limits change | `{"kind":"api_key","org_id","key_id","key_hash"}` or `{"kind":"org","org_id","status"}` | gateway: delete `loom:apikey:<key_hash>`; for an org, its keys' entries |
| `loom_credit_changed` | non-usage ledger transaction with an org | `{"org_id","kind"}` | billing: `Limiter.set_credit(org, *ledger.org_credit_state(org))` (total and version) |

Row-level security is not enabled yet: isolation is org-scoped queries plus the
cross-tenant tests M1 requires.

## 4. Signups, approval, admins

1. First sign-in (Auth.js `createUser` event): the web app creates an org (`status =
   pending_approval`) and an `org_members` row with role `owner`, in one transaction.
2. A pending org's members can sign in and see an "awaiting approval" page. They cannot
   create keys (the web app refuses), and the gateway refuses any key of a non-active org
   with 403.
3. An admin approves in the admin page: `SELECT loom_approve_org(org_id, admin_user_id,
   LOOM_SIGNUP_PROMO_MICROS)`. In one transaction it checks the caller is an admin and the
   org is pending, sets `active`, posts the $5 `promo` transaction keyed
   `promo:signup:<org id>`, and writes `admin_audit_log` (`org.approve`). The key makes a
   second signup credit for the same org impossible. Rejecting sets `rejected` and is
   audited the same way.

Why the credit waits for approval: the free $5 is the abuse target the approval step
exists to protect, and posting it on approval means a rejected or never-reviewed signup
costs nothing and leaves no ledger entries to unwind.

Admins are `users.role = 'admin'`. In local dev the web app grants it at sign-in to
emails in `LOOM_ADMIN_EMAILS`. Every grant, suspension, credit adjustment and model
disable is an `admin_audit_log` row written in the same transaction as the change.

## 5. API keys

`loom_sk_<12 base62>_<40 base62>`, 61 characters (`loom_platform.apikeys`). Stored:
`prefix` (first 20 characters, shown in the dashboard) and `key_hash` (SHA-256 of the full
key, 32 bytes). The plaintext is shown once. The web app generates keys with the same
format (crypto-random base62) or calls a shared helper; the gateway only hashes. A
malformed bearer token is a 401 without any lookup.

**Playground keys** *(wave 2, web)*. The browser never holds an API key. The playground
posts to the web app's `POST /api/playground/chat` with its session cookie (a cross-site
`Origin` is refused with 403); the handler checks the session and that the org is
`active`, then calls the gateway's `/v1/chat/completions` (always streaming, with
`include_usage`) with the user's playground key and streams the answer back unchanged:

- one key per (org, user): an ordinary `api_keys` row with `scopes = {playground}`, name
  `Playground`, `created_by` = the user. The gateway needs nothing new: limits, credit,
  suspension, revocation and usage apply as to any key; usage shows it as "Playground";
  the API keys page hides it;
- its plaintext is never stored. The secret half is derived on the server:
  `HMAC-SHA256(server secret, "<prefix>:<n>")`, n = 0, 1, ..., bytes below 248 mapped
  mod 62 to base62 until 40 characters, where the server secret is
  `LOOM_PLAYGROUND_KEY_SECRET` or `HMAC-SHA256(AUTH_SECRET, "loom-playground-key-v1")`.
  The database holds only `prefix` and `key_hash`, as for every key, and the server
  rebuilds the plaintext per request;
- rotated after `LOOM_PLAYGROUND_KEY_TTL_S` (default 3600 s, at least 600): a newer key is
  minted and older ones revoked (the revocation reaches the gateway as in §6). Signing out
  revokes the user's playground keys at once.

## 6. Redis

One Redis node (no Cluster: scripts derive key names). Never evict, never `FLUSHALL`
without also discarding the stream (both are one state).

| Key | Type | Written by | Meaning |
|---|---|---|---|
| `loom:apikey:<sha256 hex>` | string, JSON | gateway | Auth record (`loom_platform.auth.AuthRecord`) or `{"found":false}`; TTL `LOOM_AUTH_CACHE_TTL_S` |
| `loom:org:<org>:credit` | int | init, billing | `loom_org_credit_total` |
| `loom:org:<org>:credit_v` | int | init, billing | Its version, `loom_org_credit_version` (migration 0006): the org's count of non-usage ledger transactions, read in the same snapshot |
| `loom:org:<org>:usage` | int | init, settle | Cumulative settled cost (all time) |
| `loom:org:<org>:rsum` | int | reserve, settle | Sum of live reservations |
| `loom:org:<org>:rsv` | hash | reserve, settle | request id -> `<max cost>|<key id>|<tpm estimate>` |
| `loom:org:<org>:rsv:exp` | zset | reserve, settle | request id -> expiry (ms); its size is the org's in-flight count |
| `loom:org:<org>:done:<request>` | string | settle | Settled marker, 24 h |
| `loom:org:<org>:tb:rpm`, `:tb:tpm` | hash `{t, ts, c}` | reserve, settle | Org token buckets |
| `loom:org:<org>:key:<key>:usage` | int | init, settle | Key's cumulative cost (spend cap) |
| `loom:org:<org>:key:<key>:rsum`, `:inflight` | int | reserve, settle | Key's reservations and in-flight count |
| `loom:org:<org>:key:<key>:tb:rpm`, `:tb:tpm` | hash | reserve, settle | Key token buckets |
| `loom:usage` | stream | settle | Usage events (§7) |
| `loom:notify` | stream | billing reconcile | Notifications outbox (§7) *(wave 2)* |
| `loom:org:<org>:lowbal` | string | billing reconcile | Low-balance notice sent for the current crossing *(wave 2)* |
| `loom:pool:<model>` | hash | pool manager / poolsim | Pool state (§10) |
| `loom:pool:<model>:wake` | string, 600 s | gateway | Start request for a cold model; also `PUBLISH loom:pool:wake <model>` |

Use `loom_platform.limiter` (`Limiter`, `org_key`, `api_key_key`) rather than building
names by hand.

### Balance model

```
available = credit - usage - rsum
```

`credit` comes only from Postgres (purchases, promos, refunds, adjustments); `usage` only
from settle; `rsum` from live reservations. Redis is ahead of Postgres by the events still
in the stream, and `usage` already counts them, so `credit` can be overwritten from
Postgres at any time, but only with a total at least as new as the one Redis holds:
every write carries the total's version (`set_credit.lua`, `init_org.lua`), and a
writer whose snapshot predates another's is refused rather than putting back a stale
total. (The ledger is append-only and a snapshot sees a prefix of the commit order, so
the count of an org's non-usage transactions orders its credit totals.) The per-minute usage batch only moves cost from
unposted `usage_events` to posted ledger entries, both of which are already inside
`usage`, so **batching never touches Redis**.

Loading: `reserve` answers `noinit` when the org's `credit`/`usage` or the key's `usage`
is missing. The gateway then calls `Limiter.load_from_postgres(conn, org_id, key_id)`
(reads `loom_org_credit_total`, `loom_org_usage_total`, `loom_key_usage_total` in one
snapshot; `init_org.lua` sets them only if absent) and retries once.

### Reserve and settle (one Lua call each)

Per request, after auth and model lookup, before the pool check (§8):

1. **Reserve** (`reserve.lua`): purge up to 64 expired reservations of the org, then check
   in order: initialized; request id not seen; `available >= max_cost` (402); key spend
   cap `key usage + key rsum + max_cost <= spend_limit` (402); key and org in-flight under
   their concurrency limits (429); key and org rpm buckets have 1 token, tpm buckets have
   `min(tpm_estimate, limit)` (429 with Retry-After from the refill rate). Only if all
   pass: charge the buckets, store the reservation with `ttl = request timeout + 60 s`.
   `max_cost = pricing.max_cost_micros(price, prompt upper bound, max_tokens,
   max_context)`, the most the request can cost; `tpm_estimate` = body bytes + 1024
   (the body-size bound, corrected at settle; it stays this even when the prompt bound
   is the context, so tool requests are not throttled by a full context of tpm each).
2. **Dispatch** with `max_tokens` always set explicitly (the client's value, or
   `LOOM_MAX_TOKENS_CAP` when it sent none; a value above the cap is a 400 at step 2 of
   §8), so the engine cannot exceed what was reserved.
3. **Settle** (`settle.lua`), exactly once when the response ends (complete, client
   abort, upstream loss, engine error, or no replica answered): refuse a repeat; release the
   reservation; charge tpm buckets `actual tokens - estimate` (refunds an over-estimate,
   debt floored at one minute of tokens); `usage += cost`; key usage += cost; `XADD` the
   usage event with `org_usage_total_micros`. Settling without a live reservation still
   charges. If the org's state is gone from Redis (lost mid-request), settle appends the
   event but creates no counters, so the next reserve loads them from Postgres. Cost is
   `pricing.cost_micros` on the engine's token counts.

Rejected requests (401/402/403/404/413/429, and 503 `model_unavailable`) emit no usage
event; they are metrics only. A 503 `model_unavailable` comes after reserve (§8) and
**releases** its reservation (`release.lua`): the tpm estimate is refunded, the rpm token
stays spent, nothing is charged.

### Worst-case overshoot

**Zero**, given: (a) the prompt bound holds. It is `pricing.prompt_tokens_upper_bound`:
UTF-8 bytes of the JSON body + 1024 template tokens for text the template copies verbatim
(true for the byte-level BPE tokenizers served, Qwen3 and Llama 3); but **the context** for
a chat request with values the template re-serialises with `tojson` (`tools`,
`functions`, `documents`, an assistant message's `tool_calls`) or a request-supplied
`chat_template` / `chat_template_kwargs`. Re-serialised JSON can be much longer than it
was in the body (`9e15` renders as `9000000000000000.0`, one token per digit on Qwen3;
Llama's `tojson(indent=4)` adds indentation per nesting level), and the engine refuses a
prompt over the context. `response_format` is not rendered into the prompt (vLLM applies
it as guided decoding), so it keeps the body bound. (b) the
engine honours `max_tokens`; (c) the gateway settles within the reservation TTL (it
cancels upstream at `LOOM_REQUEST_TIMEOUT_S`); (d) Redis keeps its writes. Then every
admitted request's cost is at most its reservation, and `credit - usage` never goes
below 0, so the posted ledger balance never does either.

When an assumption fails, the balance can go negative by at most:

| Failure | Bound |
|---|---|
| (a)/(b) a request costs more than reserved | its excess over the reservation; the gateway compares cost with its reservation at settle, and an excess is a metric and an alert |
| (c) a settle after its reservation expired | that request's reservation: at most 22,414 micro-dollars ($0.022) for the 70B FP8 (32k context, 16k `max_tokens` cap), 7,291 ($0.0073) for the 8B |
| (d) Redis crash, AOF `everysec` | the cost settled in the last second before the crash (those requests are also never billed), plus events settled after the restart but before the org reloads and not yet consumed by billing |
| An admin refund or negative adjustment | the amount, by the admin's choice |

A gateway crash mid-request (SIGKILL, OOM, host loss) loses that request's settle: the
reservation expires, the tokens go unbilled (under-charge, never overshoot), and the
reservation and its in-flight counts are held until the reservation TTL. A **shutdown**
(SIGTERM) does not lose them, below.

### Settle spool

A settle Redis does not accept (three tries over ~150 ms: down, failing over, or frozen
past `LOOM_REDIS_SOCKET_TIMEOUT_S`) is not lost: the gateway appends the usage event as
one JSON line to `LOOM_SETTLE_SPOOL_PATH` (flushed and fsynced) and a background task
replays the file into `settle.lua` with backoff (0.5 s doubling to 30 s) until Redis
accepts each entry, then drops the replayed lines (rewrite, fsync, rename). The request
ends normally for the client meanwhile; its reservation stays held in Redis until the
replayed settle releases it. Metrics: `loom_gateway_settle_spool_depth` (alert when above
0 for 5 minutes), `loom_gateway_settle_spooled_total`, `loom_gateway_settle_replayed_total
{result}`; `loom_gateway_settle_failures_total` now counts only settles that neither Redis
nor the disk accepted (usage lost: alert at once).

Replay is idempotent: `settle.lua` refuses a request id whose `done:` marker exists, so a
settle that did run although its reply was lost is not charged twice. The marker lives
`LOOM_REQUEST_TIMEOUT_S + LOOM_RESERVATION_GRACE_S + 3600` s (§6 key table); a spooled
entry older than that is first looked up in `usage_events` (dropped if billing already
stored it), and Postgres stores an event at most once anyway (`ON CONFLICT (id,
created_at)`). The remaining gap (an entry that did reach Redis, older than the marker,
not yet consumed by billing) can only double Redis `usage` for it, never the ledger:
reconcile reports it as `missing_in_postgres` drift and the balance shows lower, never
higher.

The file is held with an exclusive `flock` (one gateway process per file) and replayed
at the next start when a process stops with entries left. **M3**: put it on a volume
that outlives the task, or accept that a task stopped during a Redis outage longer than
its shutdown loses the spooled settles (logged at error with the count).

While Redis is unavailable new requests are refused fast: any Redis error during auth,
reserve or the pool check is a 503 `service_unavailable` with `Retry-After: 1` (§11).

### Shutdown

On SIGTERM the gateway (`python -m loom_gateway`, `GatewayServer`):

1. closes its listener (no new connections);
2. gives in-flight requests `LOOM_SHUTDOWN_GRACE_S` (10 s) to finish on their own;
3. cuts every stream still open: the client gets `data: {"error": {... "code":
   "server_shutdown"}}` and `data: [DONE]` (503 `server_error`, §11; bounded at 1 s for a
   client that stopped reading), and the request settles `aborted`, billed for the
   tokens streamed like a client abort; its reservation and in-flight counts are
   released at once;
4. waits up to 2 s more for connections to close, then cancels what is left (a
   non-streaming request still waiting on the engine settles `error`, releasing its
   reservation; its tokens are unbilled, as at the timeout above);
5. waits up to 5 s for the settles it started, then closes Redis and Postgres.

The process exits within about `LOOM_SHUTDOWN_GRACE_S` + 8 s (exit status: uvicorn
re-raises the SIGTERM after a clean shutdown, so 143). **M3 (ECS behind an ALB)**: the
task's `stopTimeout` must exceed that, so set **30 s** (the ECS default; 120 s max) with
the default grace; a SIGKILL before step 5 is the crash case above. Set the target
group's **deregistration delay to 15 s or less** (ECS deregisters a task and waits for the
delay before sending SIGTERM): longer only delays the stop, since streams can run 600 s
and are cut and billed at SIGTERM anyway, and an ALB that closes a connection at the end
of the delay is a client abort, also billed. Keep `LOOM_SHUTDOWN_GRACE_S` +
deregistration delay well under the deploy's health-check grace.

A **non-streaming** request cut at `LOOM_REQUEST_TIMEOUT_S` is also unbilled: the engine
reports usage only in its final body, so the tokens it generated before the cut are
unknown. It settles `aborted` at cost 0 and counts in
`loom_gateway_usage_missing_total` (alert on it). The bound is that request's reservation.
Not fixed by streaming from the engine internally: vLLM's streaming tool-call and
reasoning parsers are a different code path from the non-streaming ones the quality gate
measured, so that switch waits for a user decision and a gate run on the streaming path.

### Rate limits

Token buckets per key and per org for requests/min and tokens/min, refilling
continuously at `limit / 60` per second, capacity one minute. tpm is charged the prompt
upper bound at reserve and corrected to the engine's actual prompt + completion tokens at
settle. Concurrency counts in-flight requests (streaming or not) per key and per org.
Limits come from `api_keys` / `orgs`, else the `LOOM_DEFAULT_*` settings
(`AuthRecord.limits`).

### Revocation within 5 s

The gateway keeps key records in Redis for `LOOM_AUTH_CACHE_TTL_S` = 4 s and may keep an
in-process copy for at most 1 s. A revoked key or suspended org is therefore refused on
every gateway within 5 s of the commit, with no other mechanism. The gateway also
`LISTEN`s on `loom_auth_changed` and deletes the cached record when notified, so in
practice it takes milliseconds. A listener that reconnects needs no resync: the TTL
already bounds anything it missed.

## 7. Usage stream

Stream `loom:usage`, consumer group `billing`, one consumer per billing process
(`billing-<hostname>`). Entries are flat strings (`loom_platform.usage.UsageEvent`):

| Field | Format |
|---|---|
| `v` | `1`; refuse other versions |
| `id` | request uuid (`X-Loom-Request-Id`) |
| `org_id`, `api_key_id` | uuid |
| `model_id` | registry id |
| `price_version` | `<input>:<cached input>:<output>` micro-dollars per 1M (`Price.version`) |
| `cloud`, `region` | `local` / empty in M1 |
| `prompt_tokens`, `completion_tokens`, `cached_prompt_tokens` | ints, from the engine's usage |
| `latency_ms`, `ttft_ms` | ints; `ttft_ms` empty when no token was sent |
| `status` | `ok`, `aborted` (client gone or upstream lost mid-stream), `error` |
| `cost_micros` | int; must equal `cost_micros(Price.from_version(price_version), ...)` |
| `created_at` | ISO 8601 UTC with microseconds: when the gateway received the request |
| `org_usage_total_micros` | added by `settle.lua`: the org's `usage` after this event; empty if the org was not loaded |

The billing worker reads with `XREADGROUP`, validates (`UsageEvent.from_stream_fields`),
inserts with `insert_usage_events` and commits, then `XACK`s; it trims with `XTRIM MINID`
below the oldest pending entry. A malformed entry goes to a dead-letter stream
`loom:usage:dead` and alerts; it is never silently dropped.

Because settle appends in the same atomic call that bumps `usage`, the stream is in
settle order and, for each org, `org_usage_total_micros` of the last event inserted equals
`loom_org_usage_total(org)` once every earlier entry is consumed. A mismatch means lost
events: alert.

**Per-minute batch** (billing): every minute, for each org with unposted events, in one
transaction: lock `usage_events WHERE org_id = $1 AND ledger_transaction_id IS NULL AND
cost_micros > 0` (`FOR UPDATE`), post one `usage` transaction for their sum keyed
`usage:<org>:<minute of the batch run>`, set their `ledger_transaction_id`. A crash rolls
back both; a racing second worker hits the idempotency key.

**Reconcile** (billing, every minute): for each org loaded in Redis
(`SCAN loom:org:*:credit`), `set_credit(org, credit total, version)`; also on every
`loom_credit_changed` notification.

**Notifications outbox** *(wave 2, billing)*: stream `loom:notify`
(`LOOM_NOTIFY_STREAM`), flat string entries `type`, `org_id`, `balance_micros`,
`threshold_micros`, `at` (ISO 8601, seconds), plus `amount_micros` for
`auto_recharge_due`. Reconcile appends:

- `low_balance`: the org's balance (credit - usage) fell below `LOOM_LOW_BALANCE_MICROS`
  ($1). Once per crossing: marker `loom:org:<org>:lowbal`, cleared when the balance is
  back above;
- `auto_recharge_due`: only with `LOOM_AUTO_RECHARGE_ENABLED=true`, for an org whose
  `auto_recharge` row is enabled and whose balance is below its threshold. A hook only:
  charging a saved card is not built.

A consumer reads it with its own consumer group and sends the email. In M1 nothing
consumes it; the entries and billing's `notification.emitted` log line are the record.

**Checkout** *(wave 2, billing's internal API)*: `POST /internal/checkout-sessions` on
billing (8090), `Authorization: Bearer $LOOM_BILLING_INTERNAL_TOKEN` (the web app's
`LOOM_INTERNAL_TOKEN`; unset = 503). Called only by the web app's server after it has
checked the caller may buy credit for the org. Body `{"org_id", "amount_micros",
"success_url", "cancel_url", "customer_email"?}`: whole cents within
[`LOOM_CHECKOUT_MIN_MICROS`, `LOOM_CHECKOUT_MAX_MICROS`] ($5 to $500), org `active`.
Answers `{"id", "url", "amount_micros"}`; the web app redirects the browser to `url`
(Stripe Checkout, or the local stub's pay page). Errors: 401 bad token, 400 bad body or
amount, 404 unknown org, 409 org not active, 502 Stripe error. The credit is posted only
by the signed `checkout.session.completed` webhook, as a `purchase` keyed
`stripe:<event id>`, and at most once per session.

## 8. Gateway request order

1. Body size (413), bearer token format (401), auth record (401 unknown/revoked, 403 org
   not active).
2. Model: registry id, priced and not disabled (`pricing.servable`), else 404. Request
   validation (400), including `max_tokens` above the cap.
   *(wave 2)* `/v1/completions` without `max_tokens` gets OpenAI's default of **16**,
   sent to the engine explicitly (chat completions without one get the cap, §6 step 2).
   At most 16 choices per request (`n` x prompts); `best_of` and `use_beam_search` are
   refused (the engine would generate more than the reservation covers).
3. Reserve (402/429). If the org's balance cannot be loaded from Postgres into Redis
   (`noinit` again after the load), 503 `balance_unavailable` with `Retry-After: 1`
   *(wave 2)*.
4. Pool state (§10): not routable -> release the reservation, 503 `model_unavailable`.
   Reserving first means a cold model is only woken (a GPU start, §10) by a request that
   has the balance for it and passes the rate limits; every 503 spends an rpm token, so
   polling an unavailable model is rate-limited like any other request.
5. Dispatch to a replica (§9); retry once on another replica only before the first byte
   reaches the client.
6. Stream through; settle once at the end; usage event.

## 9. Gateway to pod

The gateway calls a pod's mTLS proxy, which forwards to the OpenAI-compatible engine on
loopback (phase1-runpod-plan.md §4). In M1 the engine is the bench mock
(`python -m loom_bench.mock --host 127.0.0.1 --port 8000 --model <id>`), one mock pod per
servable model: `mock-pod-1` serves `qwen3-8b`, `mock-pod-2` serves
`llama-3.3-70b-instruct-fp8` (`loom-mockpod`). The compose service name is the pod id, so
a replica's URL host and `server_name` are the same (`https://mock-pod-1:8443`).

- **Transport**: `https://<replica url>` (from the pool state), TLS 1.3, HTTP/2 where
  available, pooled connections. Client certificate `LOOM_GATEWAY_CLIENT_CERT` /
  `_KEY`; the server certificate must chain to `LOOM_POD_CA_CERT` and name
  `replica.server_name` (the pod id; `mock-pod-1` in M1).
- **Header** `X-Loom-Pod-Token: <replica.token>`; the proxy refuses a request without
  it (403). No client headers are forwarded except `Content-Type` and `Accept`.
- **Paths** the proxy forwards: `POST /v1/chat/completions`, `POST /v1/completions`,
  `GET /v1/models`, `GET /health`, `GET /metrics`; anything else is 404. Streamed chunks
  are flushed immediately; request bodies are never logged.
- **Request rewriting**: `model` stays the registry id (the engine's served model name);
  `max_tokens` is set explicitly (§6); streaming requests get `stream_options:
  {include_usage: true, continuous_usage_stats: true}`. The gateway takes token counts
  only from the engine's `usage` (never counts itself) and strips per-chunk usage and the
  usage chunk from what the client sees unless the client asked for `include_usage`.
- **Per-chunk usage** (`continuous_usage_stats`, in the mock since M1) is vLLM's: only
  together with `include_usage`; every chunk, the role chunk included (0 completion
  tokens), carries `{prompt_tokens, completion_tokens so far, total_tokens}` and **no**
  `prompt_tokens_details`; cached tokens come only in the final usage chunk. With `n > 1`
  the choices interleave and each chunk's `completion_tokens` is **its own choice's**
  running count (only the final usage chunk sums the choices; the prompt counts once). A
  stream that ends early is billed the prompt plus, per choice index, the highest count
  forwarded (`sse.UsageTracker`), with 0 cached tokens (cached input is priced as input
  in M1, so this is exact today). Non-streaming responses carry the engine's total.
- **Proxy behaviour** (`loom_podproxy`, tests in `services/podproxy/tests`): source-address
  allowlist (`LOOM_PODPROXY_ALLOW_IPS`) checked before the TLS handshake; TLS 1.3 only; a
  client certificate chaining to Loom's CA with `clientAuth` (a pod's own server
  certificate is refused); token checked before the route (403, then 404); HTTP/1.1 with
  keep-alive, ALPN `http/1.1` only (no HTTP/2 in M1). Its own errors are OpenAI-shaped and
  close the connection: 403 `forbidden`, 404 `not_found`, 413 `too_large` (body over
  4 MiB), 502 `upstream_unavailable` (engine unreachable; `GET /health` answers 503
  instead). Engine response headers other than `Content-Type`/`Content-Encoding` are
  dropped. A client that disconnects mid-response closes the engine connection (the engine
  aborts the request); an engine lost mid-response drops the client connection **without**
  the final chunk terminator, so the gateway sees an incomplete body (`upstream_lost`),
  never a clean end.
- **Implementation**: Python (`h11` + `httpx`, already locked) instead of the Caddy binary
  in phase1-runpod-plan.md §4 step 2: it runs in the test suite and in compose unchanged,
  and the proxy tests are the behaviour any replacement must pass. Whether RunPod pods run
  it or Caddy is an M2 decision.
- Dev certificates (`loom-devcerts`: a local CA, a gateway client certificate, a server
  certificate and a token per pod) go into `.dev/certs`, gitignored; never committed.

## 10. Model pools and "model temporarily unavailable"

Decided 2026-10-31: models run on demand, and when no GPU of the model's own config is
available there is no fallback. A request to a model with no ready replica returns
**503 `model_unavailable` with `Retry-After`** at once; the gateway never queues it.

`loom:pool:<model>` (`loom_platform.pool.PoolState`):

| `state` | Gateway | Retry-After |
|---|---|---|
| `ready` with replicas | Route (least in-flight) | |
| `warming` | 503 "starting up" | `retry_after_s` (remaining cold-start estimate) |
| `cold` or no hash | `request_wake(model)`, then 503 "starting up" | `retry_after_s`, else the model's cold-start estimate (8B 240 s, 70B 450 s from the measured cold starts; gateway constants in M1, `config/serving.yaml` from M2) |
| `unavailable` | 503 "no GPU capacity" | `retry_after_s` (default 300) |

`replicas` is a JSON list of `{id, url, server_name, token}`. The gateway reads the hash
per request (or caches it for at most 1 s).

**M1 simulation** (`poolsim`): seeds every servable model as `ready` with the one mock
pod replica. Simulated on-demand behaviour, for the tests and demos: setting a model
`cold` makes the next request return 503 and publish a wake; poolsim moves it to `warming`
(`retry_after_s` counting down) and to `ready` after `LOOM_POOLSIM_COLD_START_S` (default
20 s); setting `unavailable` simulates no capacity. From M2 the pool manager writes the
same hash. Commands: `loom-poolsim set <model> ready|warming|cold|unavailable
[--retry-after-s N]`, `loom-poolsim status`. `cold` carries `retry_after_s` = the
simulated cold start; `ready` carries 0; a wake for an `unavailable` model is ignored (the
wake key expires on its own). Optional `LOOM_POOLSIM_IDLE_STOP_S` moves a ready model with
no usage events for that long to `cold`; poolsim reads `loom:usage` with `XREAD` (no
consumer group), so it never takes events from billing.

## 11. Errors

OpenAI shape: `{"error": {"message", "type", "param", "code"}}` (`loom_platform.errors`).
Clients branch on the status; `code` gives the reason.

| Status | type | code | When |
|---|---|---|---|
| 400 | `invalid_request_error` | `invalid_request`, `context_length_exceeded`, `duplicate_request` | Bad body, over the caps, engine 4xx passed through |
| 401 | `invalid_request_error` | `invalid_api_key` | Missing, malformed, unknown or revoked key |
| 402 | `insufficient_quota` | `insufficient_balance`, `key_spend_limit` | Reservation refused for money |
| 403 | `permission_error` | `org_pending_approval`, `org_suspended` | Org not active |
| 404 | `invalid_request_error` | `model_not_found` (`param: "model"`) | Unknown, unpriced or disabled model |
| 413 | `invalid_request_error` | `request_too_large` | Body over `LOOM_MAX_BODY_BYTES` |
| 429 | `rate_limit_error` | `rate_limit_exceeded`, `concurrency_limit_exceeded` | Limits; `Retry-After` in seconds |
| 502 | `server_error` | `upstream_lost` | Pod lost before any byte reached the client, retry used up; *(wave 2)* also no answer within `LOOM_REQUEST_TIMEOUT_S` of receipt. Billed nothing (usage event `error`) |
| 503 | `server_error` | `model_unavailable` | §10; `Retry-After` in seconds |
| 503 | `server_error` | `balance_unavailable` | *(wave 2)* the org's balance could not be loaded into Redis (§8 step 3); `Retry-After: 1`; no reservation, no usage event |
| 503 | `server_error` | `service_unavailable` | Redis unreachable or frozen past `LOOM_REDIS_SOCKET_TIMEOUT_S` during auth, reserve or the pool check (§6 "Settle spool"); `Retry-After: 1`; nothing reserved |
| 503 | `server_error` | `server_shutdown` | Only as a stream's last error chunk: the gateway is shutting down (§6 "Shutdown"); the tokens streamed are billed; retry |

Mid-stream failures end the stream with `data: {"error": {... "code": "upstream_lost"}}`
and `data: [DONE]` (`ApiError.sse_chunk`).

## 12. Cost in responses

| Header | Value | When |
|---|---|---|
| `X-Loom-Request-Id` | request uuid (= `usage_events.id`) | always |
| `X-Loom-Price-Version` | `<input>:<cached input>:<output>` micro-dollars per 1M | after model lookup |
| `X-Loom-Max-Cost-Micros` | the reservation | after reserve |
| `X-Loom-Cost-Micros` | the charge, integer | non-streaming responses |
| `X-Loom-Cost-USD` | the charge in dollars, 6 decimals (`0.000261`) | non-streaming responses |

Headers of a streamed response leave before the cost is known, so a stream carries the
charge in its final usage chunk as `usage.cost_micros` (when the client asked for usage);
it is always in the usage API. The gateway lists these headers in
`Access-Control-Expose-Headers`.

### Public model list *(wave 2)*

`GET /v1/models` and `GET /v1/models/{id}` need no key (the web app's Models page and
playground read them; the web app falls back to `loom:pool:*` in Redis only when the
gateway is down). They list the servable registry rows (`pricing.servable`) in OpenAI's
list shape, with Loom fields added; OpenAI SDKs ignore the extra fields:

```json
{"object": "list", "data": [{
  "id": "qwen3-8b", "object": "model", "created": 0, "owned_by": "loom",
  "display_name": "Qwen3 8B", "context_length": 32768, "quantization": "none",
  "status": "preview",
  "capabilities": {"tools": true, "json_schema": true, "vision": false, "reasoning": true},
  "pricing": {"currency": "USD", "price_version": "77000:77000:368000",
              "input_per_mtok_micros": 77000, "cached_input_per_mtok_micros": 77000,
              "output_per_mtok_micros": 368000, "input_per_mtok_usd": "0.077000",
              "cached_input_per_mtok_usd": "0.077000", "output_per_mtok_usd": "0.368000"},
  "availability": {"state": "ready", "ready_replicas": 1, "retry_after_s": null}
}]}
```

`status` is the registry status; `availability.state` is the pool state (§10: `ready`,
`warming`, `cold`, `unavailable`; a `ready` hash without replicas shows as `cold`),
`ready_replicas` counts replicas only when `ready`, and `retry_after_s` is null when
routable. Prices are integers in micro-dollars per 1M tokens plus 6-decimal dollar
strings. An unknown id is 404 `model_not_found`.

## 13. Open for the user or the next agents

- **Auth.js with uuid ids**: verify on the installed `@auth/pg-adapter` (web agent).
- **Mock `continuous_usage_stats`**: done (§9). Cached tokens of an aborted stream are
  unknown (as in vLLM); exact while cached input is priced as input.
- **Pod proxy on RunPod**: Python `loom-podproxy` or a pinned Caddy (plan §4); decide in M2.
- **Row-level security**: deferred; org-scoped queries and cross-tenant tests for now.
- **Gateway image size**: `loom-platform` depends on `loom-bench` for the registry,
  which pulls numpy, scipy and pyarrow. Splitting the registry into its own small
  package is a later refactor.
- **Non-streaming requests cut at the timeout** are unbilled (§6). Billing them means
  streaming from the engine internally, which first needs a quality-gate run on vLLM's
  streaming tool-call and reasoning parsers (user decision).
- **Google email linking**: kept on (`LOOM_GOOGLE_EMAIL_LINKING`, verified emails only);
  whether to keep it at all is a user decision.
- **Disputes won** (`charge.dispute.closed`, status `won`) do not reinstate the credit
  taken back yet; an admin adjustment does. Stripe's webhook endpoint must subscribe to
  `charge.refunded` and `charge.dispute.created`.
