# Phase 1 M1 acceptance

M1 is the whole platform running locally on the mock backend, with no cloud, accounts or
paid calls ([phase1-runpod-plan.md](phase1-runpod-plan.md) §13, M1 row; brief §8.4). This
page lists each M1 acceptance item with:

- what is asked;
- the status: **pass** or **fail**;
- the evidence: the numbers measured on the compose stack, and the code and tests behind
  them;
- how to reproduce it on a laptop, for free.

Snapshot: 2026-11-01. Every number on this page is from one run of `loom-m1-acceptance`
at commit `639f412` on a stack brought up from nothing (`docker compose down -v`,
`.dev/certs` deleted, `up --build --wait`) on an Apple Silicon laptop (Docker Desktop).
All ten items passed. The one later code change is in the driver: it read the cross-tenant
error as "RevokeKey not found." (the red Revoke button's text swept in); item h re-run
alone after the fix shows the app's actual message, "Key not found.". Earlier runs the
same day found two defects in the platform, fixed the same day (see
[Defects found](#defects-found)).

Full Python suite on the final tree: 1944 passed, 1 skipped (`pytest -q -n auto` against
the compose Postgres and Redis).

## Summary

| # | Item | Status | Number |
|---|---|---|---|
| 0 | Sign up -> pending -> approve -> $5 -> key -> OpenAI SDK + playground -> usage and balance drop | **pass** | 6 requests, $0.000115; ledger balance $4.999885 exactly |
| a | Usage in the DB = engine-reported tokens, incl. a client abort and a killed pod | **pass** | exact in all 3 cases (engine counters = `usage_events`) |
| b | Revoked key fails within 5 s | **pass** | 3 ms after the revoke commits (31 ms from the click); suspended org 35 ms |
| c | Concurrent load cannot overshoot the balance | **pass** | overshoot **0** micro-dollars (documented bound 0); 64 concurrent, 2,560 attempts on $0.002 |
| d | Gateway p50 added overhead < 10 ms | **pass** | **4.9 ms** p50 (gateway 32.9 ms vs direct 28.0 ms, same mTLS path) |
| e | Load test: TTFT and tokens/s | **pass** | 8B @32: TTFT p50 105 ms, 701 output tok/s; 70B @16: TTFT p50 116 ms, 355 tok/s; 0 errors |
| f | Cold model: 503 "temporarily unavailable" + Retry-After, then ready | **pass** | 503, `Retry-After: 20` counting down, ready after 21.4 s |
| g | Stripe stub add-credits credits once | **pass** | +$10.00 once; replayed event not credited |
| h | Cross-tenant spot check, web + API | **pass** | 7 checks, all refused or isolated |
| i | Logs contain no prompt or completion text | **pass** | 28 strings, 12 containers, 6,524 lines: 0 hits |

Also part of M1 and in place: the stack comes up healthy (all 10 long-running services
`healthy`, `migrate` and `devcerts` exited 0); unit tests for pricing, ledger, rate
limiter, key hashing and reserve/settle; integration tests request -> usage -> ledger
including an aborted stream; cross-tenant tests (see each item). CI now checks the web
app too ([CI](#ci)).

## How to reproduce

```bash
cd /Users/mathurs/Projects/loom
uv sync
LOOM_ADMIN_EMAILS=admin@m1.test docker compose up -d --build --wait
docker compose ps                       # every service healthy; migrate, devcerts exited 0
uv run loom-m1-acceptance --json /tmp/m1.json            # about 4 minutes; exit 0 = all pass
```

The run used here, next to another stack on the default ports:

```bash
export COMPOSE_PROJECT_NAME=loom-m1accept LOOM_ADMIN_EMAILS=admin@m1.test \
  LOOM_PG_PORT=25432 LOOM_REDIS_PORT=26379 LOOM_GATEWAY_PORT=18080 \
  LOOM_GATEWAY_METRICS_PORT=19100 LOOM_BILLING_PORT=18090 LOOM_BILLING_METRICS_PORT=19101 \
  LOOM_STRIPE_STUB_PORT=22111 LOOM_WEB_PORT=13000 LOOM_MAILPIT_UI_PORT=18025 \
  LOOM_MAILPIT_SMTP_PORT=11025
docker compose down -v && rm -rf .dev/certs
docker compose up -d --build --wait
uv run loom-m1-acceptance --json /tmp/m1.json
docker compose down -v                  # afterwards
```

`--only a,f` runs a subset (the sign-up flow, item 0, always runs first: it creates the
tenants). The same run as a pytest test, deselected by default and skipped when the stack
is down: `uv run pytest -m acceptance -s services/devstack/tests/test_devstack_acceptance.py`.

**How it drives the stack.** `loom-m1-acceptance` (`services/devstack/src/loom_devstack/acceptance.py`)
works only from the outside, like a user:

- the web app over plain HTTP, the way a browser without JavaScript uses it: it GETs a
  page and posts its form. React's progressive-enhancement forms carry the server
  action's id in hidden fields, so the post runs the same server action a click does
  (sign-in, approve, create and revoke keys, admin adjustments, add credit);
- magic links read from Mailpit's API (`/api/v1/search`, `/api/v1/message/{id}`);
- the gateway with the unmodified OpenAI Python SDK (`openai` 1.x, `base_url` and
  `api_key` only), plus raw `httpx` streams where it needs to hang up or watch error
  chunks;
- the playground through the web app's `/api/playground/chat` with the session cookie;
- the Stripe stub's pay page (GET, then the Pay form);
- Postgres, Redis and the mock engines' own Prometheus counters (read on the pod with
  `docker exec`) to check every number; `docker kill`/`start` for the pod, `loom-poolsim`
  inside its container for the cold path, `docker logs` for the log check.

It creates its own users (`<who>-<random>@m1.test`), so it can run again on the same
stack. Its helpers are unit-tested on every `pytest` run
(`services/devstack/tests/test_devstack_acceptance.py`).

## Items

### 0. The user flow

> "sign up -> free credits -> create key -> chat in playground -> call API with the OpenAI
> Python SDK -> see usage and balance drop" (brief §8.4)

**Status: pass.**

What the run did and saw:

1. Sign-in form with a new email -> 303 to "check your email"; the magic link arrived in
   Mailpit; following it set the Auth.js session cookie.
2. `/dashboard` showed "Your account is awaiting approval"; the org was
   `pending_approval`; `/dashboard/keys` also showed the approval page (no keys before
   approval).
3. `admin@m1.test` (in `LOOM_ADMIN_EMAILS`) signed in the same way, saw `/admin`, and
   pressed Approve for that org: "Approved; $5.00 credit added."
4. The ledger then held exactly one transaction for the org, `promo` keyed
   `promo:signup:<org>`; credit 5,000,000, usage 0, posted balance 5,000,000 micros; the
   dashboard showed $5.00.
5. Created a key on `/dashboard/keys`; the page showed it once (`loom_sk_...`).
6. With the OpenAI SDK: `models.list()` lists both models; for each of `qwen3-8b` and
   `llama-3.3-70b-instruct-fp8` a non-streaming and a streaming chat completion
   (`stream_options={"include_usage": True}`); a `/v1/completions` call without
   `max_tokens` (16 or fewer tokens: the default). Non-streaming responses carry
   `X-Loom-Cost-Micros` equal to `usage.cost_micros`.
7. A playground chat through the web app (streamed, with a usage chunk). It was billed to
   the user's playground key (`scopes = {playground}`). A post with a foreign `Origin`
   was refused (403).
8. Every request's `usage_events` row equals the usage the API returned (prompt,
   completion, cached tokens, cost, org, status `ok`), and the engines' own counters moved
   by exactly the billed tokens: qwen3-8b prompt 83 = 83, completion 64 = 64;
   llama-3.3-70b prompt 64 = 64, completion 48 = 48.
9. Balance drop: the 6 requests cost 115 micro-dollars. Postgres usage total = 115 = the
   sum of the costs = Redis `usage`. The billing page showed "$0.000115 used" and
   balance $5.00 (cents). The usage CSV had 6 requests and the same completion tokens,
   under the key names `acceptance` and `Playground`. The per-minute batch posted them
   within 4 s of the check (it runs at :05 each minute), and the posted ledger balance was
   **4,999,885** micros = 5,000,000 - 115 exactly.

Code: `Acceptance.flow`. The same chain at unit level: `services/gateway/tests/test_gateway_billing.py`,
`services/billing/tests/test_billing_batch.py::test_batch_sum_equals_usage_events_exactly`,
`apps/web/tests/integration/*.test.ts`.

### a. Usage in the database equals engine-reported tokens

> "usage in the DB matches engine-reported tokens exactly" (brief §8.4), including an
> aborted stream.

**Status: pass**, exact in all three cases.

| Case | Engine | Billed (`usage_events`) |
|---|---|---|
| Completed requests (item 0) | counters: prompt 83 + 64, completion 64 + 48 | same, per model |
| Client hangs up after 10 content chunks (qwen3-8b, `max_tokens` 400) | counters moved by prompt 31, completion 10 | prompt 31, completion 10, status `aborted`, 7 micros |
| `docker kill` of the 70B pod after 10 content chunks | last usage chunk it sent: completion 12 | prompt 33, completion 12, status `aborted` |

How each was checked:

- **Client abort.** The engine's `vllm:prompt_tokens_total` and
  `vllm:generation_tokens_total` were read on the pod before and after, with nothing else
  running. The client saw 10 completion tokens; the gateway billed 10; the engine
  generated 10. The gateway bills the usage of the last chunk it forwarded
  (contracts §9), so tokens the engine generates after that are not billed; here there
  were none: the proxy cancelled the engine request before its next step.
- **Pod killed mid-stream.** The client asked for `continuous_usage_stats`, so it saw the
  engine's own running usage on each chunk. The killed pod's counters die with it, so
  the reference is the last engine usage the client received: 12, billed 12. The client
  got the `upstream_lost` error chunk and `[DONE]` (contracts §11). The pod was started
  again and healthy after 5.6 s, and the next request to the 70B succeeded and was billed
  exactly.

Code: `Acceptance.item_a`. Tests: `services/gateway/tests/test_gateway_billing.py::test_client_abort_bills_exactly_the_streamed_tokens`,
`services/podproxy/tests/test_podproxy_proxy.py::test_client_abort_cancels_the_engine_request`,
`::test_engine_lost_mid_stream_is_not_a_clean_end`.

### b. A revoked key fails within 5 s

**Status: pass.** 3 ms after the revoke committed.

The run created a second key, made 5 requests with it (so the gateway's in-process and
Redis caches held its record), then revoked it on `/dashboard/keys` and polled the
gateway every 10 ms with a body that passes authentication and fails validation (400 for
a live key, 401 for a revoked one, never billed):

| | |
|---|---|
| Revoke form post (click to response) | 28 ms |
| First 401 after the response | **3 ms** (31 ms after the click) |
| Then | 63 polls over the next second, all 401 |

The gateway `LISTEN`s on `loom_auth_changed` and drops the cached record on commit; the
guaranteed bound without that is the cache TTLs, 4 s Redis + at most 1 s in process
(contracts §6). Suspending an org from the admin page refused its key (403
`org_suspended`) **35 ms** after the click.

Code: `Acceptance.item_b`.

### c. Concurrent load cannot overshoot the balance

> "balance cannot go meaningfully negative under concurrent load (document max overshoot)"

**Status: pass.** Worst-case overshoot observed: **0** micro-dollars. Documented bound: 0
(contracts §6, "Worst-case overshoot", under its four assumptions, with the bounds when
one fails).

Set-up: a new org, approved ($5), then an admin correction of -$4.998 through the admin
page, leaving **2,000** micro-dollars ($0.002). Its org and key limits were raised in
Postgres (rpm 100,000, tpm 10^9, concurrency 256) so that only money could refuse a
request. Then 64 concurrent workers, 40 requests each, mixing both models, streaming
and not (`max_tokens` 48); a third of the streams hung up after the first line. A sampler
read Redis `credit - usage - rsum` every 5 ms.

| | |
|---|---|
| Attempts | 2,560 in 21.3 s: 91 admitted (200), 2,469 refused (402) |
| Reservations | 108 to 735 micros each |
| Spent | 1,899 micros (7 aborted streams included) |
| Final balance | 101 micros in Redis, Postgres and (after the batch) the posted ledger |
| Lowest Redis `available` seen | 8 micros |
| Overshoot | **0** |

Code: `Acceptance.item_c`; tests `services/platform_common/tests/test_platform_limiter.py`
(reserve/settle under concurrency).

### d. Gateway p50 added overhead < 10 ms

**Status: pass.** **4.9 ms** p50.

Measured from inside the compose network (`overhead_probe.py` piped to `python -` in the
poolsim container, which has the dev certificates), so both paths start at the same
place: 400 alternating pairs of the same small non-streaming chat request
(`max_tokens` 1) to `qwen3-8b`:

- direct: `https://mock-pod-1:8443` with the gateway's client certificate and the pod
  token: the gateway's own upstream request (mTLS, TLS 1.3);
- through the gateway: `http://gateway:8080` with an API key.

| | p50 | p90 |
|---|---|---|
| Direct to the pod | 28.0 ms | 35.6 ms |
| Through the gateway | 32.9 ms | 40.1 ms |
| Added | **4.9 ms** | 4.5 ms |
| Median of per-pair differences | 4.3 ms | |

The gateway's own histograms over the run: 2.8 ms mean from receipt to dispatch (auth,
pool, reserve), 1.3 ms mean for settle. The rest is the extra HTTP hop. Across the day's
runs the p50 was 3.6 to 4.9 ms. The unit benchmark
(`services/gateway/tests/test_gateway_overhead.py`, one process, zero-latency engine)
measures 2.3 ms.

### e. Load test: TTFT and tokens/s through the gateway

**Status: pass.** `loom-loadtest`'s code path (`loom_devstack.loadtest`, the Lab's
streaming client and summary), `chat-short` workload (~256-token prompts, up to 128-token
answers), 4 keys of one org with raised limits (the defaults allow 16 in flight per org).

| Model | Concurrency | Requests | HTTP | TTFT p50 / p95 | TPOT p50 | Output tok/s per request p50 | Aggregate output tok/s |
|---|---|---|---|---|---|---|---|
| qwen3-8b | 32 | 400 | 400 x 200 | 105 / 351 ms | 41.7 ms | 23.3 | 701 (7.8 req/s) |
| llama-3.3-70b-instruct-fp8 | 16 | 200 | 200 x 200 | 116 / 257 ms | 40.6 ms | 24.0 | 355 (3.8 req/s) |

These are the mock engine's numbers, not a GPU's: the mock is for plumbing and load
shape, never for performance. Its TPOT under load (about 40 ms) is above its low-load
profile (20 to 30 ms) because new requests' prefill steps stall decoding in its
simulated batch, as in an engine without chunked prefill. The gateway is not the cause:
at concurrency 32 on the same model, a streaming probe from inside the compose network
measured TPOT p50 33.4 ms direct to the pod and 30.9 ms through the gateway, and during
the load the gateway container used about 20% of one CPU.

Reproduce: `uv run loom-loadtest --base-url http://localhost:8080/v1 --api-key-file keys.txt
--model qwen3-8b --concurrency 32 --requests 400` with keys whose limits allow it
([local-dev.md](local-dev.md#load-test)).

### f. Cold model path

**Status: pass.**

`loom-poolsim set qwen3-8b cold`, then a request:

- **503** `model_unavailable`, message "The model 'qwen3-8b' is temporarily unavailable:
  it is starting up. Retry in about 20 s.", `Retry-After: 20`;
- poolsim took the wake, moved the model to `warming`, and Retry-After counted down on
  the next requests (20, 19, 18, ...);
- the model was ready and the request succeeded **21.4 s** after the first 503
  (`LOOM_POOLSIM_COLD_START_S` = 20);
- the 503s were decided before any reservation: no usage event.

### g. Stripe stub: add credits credits once

**Status: pass.**

On `/dashboard/billing`, Add credit $10 -> the web app called billing's internal
checkout API -> 303 to the stub's pay page (`/checkout/cs_test_...`) -> Pay -> the stub
posted a signed `checkout.session.completed` to billing and redirected to
`/dashboard/billing?checkout=success`. Credit rose by exactly **10,000,000** micros: one
`purchase` transaction keyed `stripe:evt_test_...`. Pay again (the stub re-sends the same
event id): still one purchase, one `stripe_events` row. The billing page lists the $10.00
purchase.

This failed on the first run: see [Defects found](#defects-found), 1.

Tests: `services/billing/tests/test_billing_webhook.py::test_completed_checkout_credits_once_across_replays`,
`::test_a_session_is_credited_once_across_event_ids`,
`services/billing/tests/test_billing_stub_e2e.py`.

### h. Cross-tenant spot check

**Status: pass.** Two active orgs (Alice, Bob) and a pending one:

| Check | Result |
|---|---|
| Bob posts the revoke form with Alice's key id | "Key not found."; Alice's key not revoked |
| Bob GET `/admin`, `/admin/orgs/<Alice's org>` | 404, 404 |
| Bob's `/dashboard`, `/dashboard/keys`, `/dashboard/usage`, `/api/usage.csv` | no trace of Alice's key prefix or name |
| Bob's API key calls the gateway | billed to Bob's org; Alice's usage total unchanged |
| A key with one character changed | 401 |
| A pending org's user opens `/dashboard/keys` | the approval page |
| Playground post with a foreign `Origin` (item 0) | 403 |

Tests: `services/gateway/tests/test_gateway_billing.py::test_cross_tenant_isolation`,
`services/platform_common/tests/test_platform_ledger.py::test_cannot_post_to_another_orgs_account`,
`services/billing/tests/test_billing_batch.py::test_events_never_post_to_another_orgs_account`,
`services/billing/tests/test_billing_webhook.py::test_credit_goes_to_the_sessions_org_only`,
`apps/web/tests/integration/keys.test.ts` ("cannot revoke another org's key").

### i. No prompts or completions in logs

**Status: pass.**

Every prompt in the run carried a random marker (`m1canary<16 hex>`); the first 48
characters of every completion of 24 characters or more, and every API key created, were
added to the list. At the end, `docker logs` of all 12 containers of the project (6,524
lines, including the gateway's per-request lines, billing, web, the pods and the stub)
contained **none** of the 28 strings.

## Defects found

Fixed, each with a regression test:

1. **Add credit failed on the compose stack** (billing). Compose passes
   `STRIPE_SECRET_KEY: ${STRIPE_SECRET_KEY:-}`, an empty string when unset.
   `stripe_mode` read that as the stub, but `stripe_api()` tested `is not None` and sent
   the Checkout call to `api.stripe.com` with an empty key: a 401 from Stripe, "Payments
   are not available right now" in the web app, and a network call M1 must not make.
   Blank Stripe variables now mean unset (`loom_billing/settings.py`). Test:
   `services/billing/tests/test_billing_stripe_sig.py::test_settings_empty_keys_mean_the_stub`.
2. **`poolsim` and `stripe-stub` had no healthcheck**, so `docker compose up --wait` and
   `ps` could only say "running". Both have one now. Test:
   `services/devstack/tests/test_devstack_compose.py::test_every_long_running_service_has_a_healthcheck`.

Also changed: `services/gateway/tests/test_gateway_overhead.py` measures up to 3 times
and passes on the first attempt under 10 ms. On a shared CI runner, CPU contention can
only add latency, so one clean median shows the gateway meets the target; a real
regression raises every attempt and still fails. The bound stays absolute (10 ms) rather
than relative to a baseline that a slow runner inflates too.

## CI

`.github/workflows/ci.yml` gained a `web` job: `npm ci`, `npm run lint`,
`npm run typecheck`, `npm test` (unit), `npm run test:integration` (a Postgres 16 service,
the database migrated by `uv run loom-migrate`; `LOOM_TEST_DATABASE_URL` is set so a
missing Postgres fails instead of skipping) and `npm run build`. Locally on this commit:
lint clean, typecheck clean, 51 unit and 32 integration tests passed.

## Open questions (policy, not changed)

1. **Default limits are low for real use.** A key may have 8 requests in flight and 60
   per minute, an org 16 and 120. The load test and the overshoot test needed raised
   limits (set per org in Postgres). Fine for a private beta; worth deciding before
   anyone benchmarks us.
2. **Chat requests without `max_tokens` reserve the 16,384-token cap**: up to $0.022 on
   the 70B (contracts §6). An org with less than that cannot make such a request even if
   it would cost a fraction of a cent, and gets a 402. A smaller default reservation (or
   a default `max_tokens`) would change user-facing behaviour.
3. **Aborted streams bill the tokens forwarded, not the tokens generated.** The brief
   says "bill tokens actually generated"; the contracts bill the last chunk forwarded. In
   this run the two were equal; on a real GPU, tokens generated after the last forwarded
   chunk (one engine step at most) would go unbilled.
4. **The balance is shown in cents** ($5.00 while the exact balance is $4.999885), next to
   the exact "used" figure. Clear enough, or show the exact balance?
5. **The cold-start `Retry-After`** in M1 is poolsim's 20 s; the gateway's production
   estimates are 240 s (8B) and 450 s (70B). Unchanged here.
