# Spec: the before-M3 review fixes (plan §13, R1-R6, R8, R9, R11-R14)

Status: approved by the user 2026-11-17. Source: the tables in
[phase1-runpod-plan.md §13](phase1-runpod-plan.md#follow-ups-from-the-m1-pre-merge-review-2026-11-15).
R7 and R10 wait for M5. R15 is done.

User decisions (2026-11-17):
- **R3**: a non-streaming request cut at the timeout is billed an estimate of the work done
  so far (the elapsed-time option). The expensive parameters stay allowed.
- **Admin policy**: `LOOM_ADMIN_EMAILS` grants admin only in dev; admins cannot approve or
  credit their own org.

Three PRs, each on a new branch from `main`, none merged by an agent:
`fix-m3-gateway` (R1-R4), `fix-m3-billing` (R5, R6, R8, R9), `fix-m3-web` (R11-R14 and the
admin policy). Each item gets at least one regression test that fails on `main` and passes
after the fix. Each PR updates the contracts (`docs/phase1-contracts.md`) where its
behaviour is written down, and marks its rows fixed in the plan's §13 table. The gateway
PR also fixes the stale status lines (CLAUDE.md, docs/PLAN.md, the plan's "M2 status").

## Gateway (`fix-m3-gateway`)

**G1 (R1). A stream past its deadline settles even if the client stopped reading.**
When a stream ends at `LOOM_REQUEST_TIMEOUT_S` (or the pod is lost), the request settles
`aborted` for the tokens forwarded before any trailing write (the `upstream_lost` error
chunk, `[DONE]`, the final empty body), and those writes together are bounded at 1 s, like
`_send_shutdown`. Done when: a client that stops reading mid-stream (its socket buffer
full) gets its usage event, `aborted`, with the tokens it was sent, within about 1 s of the
deadline, and the reservation and in-flight counts are released by then.

**G2 (R2a). A reserve whose reply is lost is released.**
When the reserve call raises a Redis error (socket timeout, connection reset), the gateway
still answers 503 `service_unavailable` with `Retry-After: 1`, and it also starts a
background release of that request id (`Limiter.release`), retried with backoff until it
succeeds or `LOOM_RESERVATION_GRACE_S` (60 s) has passed. The release also marks the
request id as done for the reservation TTL (`release.lua` writes the same `:done:<rid>`
marker `reserve.lua` already refuses), so a reserve script that runs *after* the release
(it was queued behind a Redis stall) answers `duplicate_request` and holds nothing.
Done when: (a) a reserve that took effect in Redis but whose reply was lost leaves no
reservation, `rsum` or in-flight count behind once Redis answers again; (b) a release that
runs before the delayed reserve makes that reserve refuse with `duplicate_request` and
write nothing; (c) the existing release after a 503 `model_unavailable` still refunds the
tpm estimate and leaves no usage event.

**G3 (R2b). A Postgres failure while loading the balance is a 503.**
Any Postgres error (`psycopg.Error`, a pool timeout) in `load_from_postgres` after a
`noinit` reserve answers 503 `balance_unavailable` with `Retry-After: 1`, never a 500.
Nothing is reserved.

**G4 (R3). A non-streaming request cut at the timeout is billed an estimate.**
It settles `aborted` with estimated tokens instead of zero:
- completion tokens = `sequences × min(max_tokens, floor(t × R))`, where `t` is seconds
  from the first dispatch to the cut and `R` is a new gateway setting
  `LOOM_TIMEOUT_BILL_TOKENS_PER_S` (tokens per second per sequence, default **10**: below
  any model's measured decode rate at its SLO, so the estimate errs toward the client;
  0 turns estimating off and restores today's zero bill);
- prompt tokens = `prompts × min(prompt_upper, ceil(body bytes / 4))`; cached = 0;
- so the cost never exceeds the reservation (`max_cost_micros`).
The client still gets 502 `upstream_lost`, now with `X-Loom-Cost-Micros` /
`X-Loom-Cost-USD` naming the amount billed. The event counts in a new counter
`loom_gateway_usage_estimated_total{model}` and no longer in `usage_missing`; the request
log line carries `estimated: true`. A request cut before any replica was dispatched, or
by shutdown, keeps today's behaviour (`error`, nothing billed). Streams are unchanged
(they bill the engine's own counts). Contracts §6 and the "unbilled" known limit in §15
are rewritten to match.

**G5 (R4). A settled non-streaming request is never answered 502.**
Once the engine's 200 body has been read and `settle` has started, the request deadline
no longer applies: the client gets the 200 body with its cost, however long settle takes,
and is never charged for a request it was told failed. Done when: a settle that finishes
after the deadline yields 200 with the body and one `ok` usage event, and no 502.

## Billing (`fix-m3-billing`)

**B1 (R5). Concurrent reconcilers heal `redis_behind` once.**
The heal is one atomic Lua call: add the missing amount only if the org's Redis `usage`
is still below the Postgres total `P` that run read (and the org is loaded). Since a
correct heal leaves `usage >= P`, a second reconciler (another worker, or an overlapping
run) finds nothing to do. Done when: two reconcilers that both observed the same
`redis_behind` drift, healing in either order or concurrently, leave Redis `usage` equal
to `P` plus the settles made since, never more.

**B2 (R6). `redis_behind` heals under steady traffic.**
The `redis_behind` check and heal run every reconcile, whether or not the stream is
drained (`P <= U2` holds regardless). The `missing_in_postgres` check still needs the
consumer group to have consumed through the snapshot's last id; when it has not, the
reconciler polls `consumed_through` for up to `LOOM_RECONCILE_DRAIN_WAIT_S` (new, default
2 s) before skipping that direction for the minute. Done when: with usage settling and
being consumed continuously (the stream never empty at the snapshot), an org whose Redis
`usage` lost an amount is healed at the first reconcile run, and a real
`missing_in_postgres` is still reported once the consumer catches up within the wait.

**B3 (R8). Services run as a role that cannot switch off the ledger's guards.**
- Two Postgres roles: the **owner** (runs `loom-migrate`, owns every object) and a
  **runtime** login role, `loom_app`, that every service uses (gateway, billing, pool
  manager, web). `loom_app` is not a superuser, owns nothing, cannot create objects in the
  schema, and has only the privileges the services use: on `ledger_entries`,
  `ledger_transactions` and `admin_audit_log` only `SELECT` and `INSERT`; on other tables
  the DML the services issue; `EXECUTE` on the functions; sequence `USAGE`.
- A new migration creates the grants (through a `NOLOGIN` group role that `loom_app` is a
  member of, so later migrations grant to one name). `loom-migrate` creates or updates
  the `loom_app` login when given its password (`LOOM_APP_DATABASE_PASSWORD`); compose
  sets a dev password and points every service except `migrate` at `loom_app`.
- Outside `LOOM_ENV=dev`, each Python service checks at startup, over its own connection,
  that its role is not a superuser and owns no table in the schema, and refuses to start
  (exit 2, like `production_problems`) otherwise. The dev password for `loom_app` is also
  a production problem.
- A test checks that every table has an explicit grant to the group role, so a later
  migration cannot forget one.
- Roles belong to the whole Postgres server, not one database. The migration, and
  `loom-migrate`'s role setup, must succeed when run at the same time against many
  databases on one server (parallel test workers each migrating a fresh database): no
  "role already exists" or duplicate-key error, and the grants come out the same.
- Contracts §3 is corrected: append-only and zero-sum hold for the runtime role; the
  owner can still disable the triggers, so only migrations run as the owner.
Done when: connected as `loom_app`, `UPDATE`/`DELETE`/`TRUNCATE` on the three append-only
tables, `ALTER TABLE ... DISABLE TRIGGER`, `DROP TRIGGER` and replacing a trigger function
all fail; the M1 and M2 flows still pass on the full compose stack brought up with every
service as `loom_app` (`loom-m1-acceptance`, `loom-m2-acceptance`, the billing and web
integration tests), checked by the verifier, not only unit tests; a service started outside
dev as the owner or a superuser exits 2 naming the problem.

**B4 (R9). A non-UTF-8 stream entry is dead-lettered, not a wedge.**
Decoding moves into `process()`: an entry with a field name or value that is not valid
UTF-8 is copied to the dead-letter stream (the bad bytes kept, or replaced with escapes,
plus `dead_reason`) and acked, and the consumer goes on with the entries after it. Done
when: a stream holding one such entry between good ones ends with the good ones stored,
the bad one in the dead-letter stream, nothing pending, and later rounds still consume.

## Web (`fix-m3-web`)

**W1 (R11). The magic-link limit applies wherever Auth.js sends.**
The per-email and per-IP limit is checked inside the Nodemailer provider's send (a
wrapped `sendVerificationRequest`), so a direct `POST /api/auth/signin/nodemailer` is
limited the same as the form. Each send is counted once (the server action no longer
counts separately). Over the limit no mail is sent, and the sign-in form still shows
"Too many sign-in emails. Try again in a few minutes." It still fails open if Redis is down.

**W2 (R12). An admin credit adjustment posts once per form.**
The adjust-credits form carries a per-render token (a random UUID in a hidden field named `form_token`); the
ledger idempotency key is derived from it (`admin:adjust:<token>`). A resubmit, or the same
form posted from two tabs, posts one ledger transaction and one audit row; the repeat
answers with the same success message and changes nothing. A missing or malformed token
is refused. A newly rendered form gets a new token.

**W3 (R13). Every sign-out revokes the playground keys.**
Auth.js's `events.signOut` revokes the user's playground keys, so signing out through
`POST /api/auth/signout` revokes them as the app's sign-out button does.

**W4 (R14). A key's spend limit has an upper bound.**
A spend limit above **$1,000,000** (10^12 micros, well inside both Postgres `bigint` and
the exact range of the Lua doubles in `reserve.lua`) is refused with a clear message
naming the bound, in the action and in the data layer. No input gives "Something went
wrong".

**W5 (admin policy). `LOOM_ADMIN_EMAILS` only in dev; no self-service admin.**
- Outside `LOOM_ENV=dev`, `LOOM_ADMIN_EMAILS` grants nothing at sign-in, and the app logs
  a warning at startup if it is set. Admins already granted keep their role (granting the
  first production admin is a manual SQL step, written down in the contracts).
- Outside dev, an admin cannot approve, or adjust the credit of, an org they are a member
  of: refused with a message, with no state change, audit row or ledger entry. In dev it
  stays allowed, because the local flow has the bootstrap admin approve their own org
  (user decision: outside dev only).
- Contracts §4 says both.

## Not in scope

R7 and R10 (M5); the `vllm:num_requests_waiting` tie-break; the M3 prerequisites P1-P4;
a CLI for granting admins.
