# Phase 1 M2 acceptance (stack part)

M2 is the pool manager, built and proved offline: against fakes in the test suite, and on
the compose stack with its `local` backend ([phase1-m2-spec.md](phase1-m2-spec.md) §9,
§13). No RunPod or AWS call, no account, no spend. This note records the stack run of
§13 item 25 (integration, wave 3). Items 1-24 are the automated tests under
`services/*/tests` and `bench/tests`; item 26's five runs on 4 CPUs and CI on the final
commit, and the branch review, come after this.

## How to reproduce (free)

```bash
LOOM_ADMIN_EMAILS=admin@m1.test docker compose up -d --build --wait
uv run loom-m1-acceptance --json /tmp/m1.json   # ~10 minutes, the manager in the loop
uv run loom-m2-acceptance --json /tmp/m2.json   # ~20 minutes, most of it the idle wait
docker compose down -v
```

A second stack next to the default one needs moved ports (as
[phase1-m1-acceptance.md](phase1-m1-acceptance.md) shows) **and** another compose subnet:
the network's subnet is fixed (`10.213.0.0/24`, the local pods' allowlist), so two
projects on this file cannot both be up. Use a compose override that changes the
`networks.default.ipam` subnet and `LOOM_POOLMGR_ALLOW_CIDRS` together.

## Run of 2026-11-16 (Windows 11, Docker Desktop, project `loom-m2accept` on moved ports)

`docker compose up -d --build --wait`: every service healthy (poolmgr's healthcheck is its
metrics port, opened once it holds the single-writer lock and has adopted its pods).

**`loom-m1-acceptance`: exit 0, every item PASS** (0, a-m) with the manager writing the
pools. What changed against M1's poolsim run:

| Item | With the manager |
|---|---|
| 0 | Both models warmed first (`loom-poolmgr warm`): 8B ready in 15 s, 70B in 32 s; tokens in Postgres = the engines' own counters |
| a | Client abort billed 10 = engine 10. Killed pod: the 70B engine killed after 10 chunks, client got `upstream_lost` + `[DONE]`, billed 38 = the last usage it saw; the manager failed the pod and its replacement was ready after 185 s (3 failed health checks, the 120 s grace, a cold start) |
| d | Overhead p50 2.19 ms (gateway 11.06 ms vs direct 8.87 ms to the local pod's proxy, same mTLS path) |
| e | 8B: 400/400 at concurrency 32, TTFT p50 59 ms; 70B: 200/200 at 16, TTFT p50 45 ms |
| f | `loom-poolmgr stop`, then 503 "starting up" with Retry-After 240 (the pool's estimate), warming 240, 226, 215, 211, ready 31.5 s after the first 503 |
| m | After FLUSHALL the stack served again within 12.3 s (the manager rewrites the pool hashes on its next tick) |

**`loom-m2-acceptance`: exit 0, every check PASS:**

| Check | Result |
|---|---|
| `cold_wake_ready` | PASS: cold 503 with Retry-After 240 = the pool's, wake key set; states cold, warming (Retry-After 240, 226, 215, 211), ready 41 s later; stages recorded (pod_running 25 s, engine_started 26 s, first_token 31 s) |
| `pod_kill_replaced` | PASS: engine killed mid-stream: `upstream_lost` + `[DONE]`, billed 37 completion tokens = the last usage the client saw; old row `terminated`, failure `unhealthy`; a new pod ready 195 s later and serving |
| `no_capacity` | PASS: `loom-poolmgr local-capacity off`: pool `unavailable` / `no_capacity`, 503 "no GPU capacity", Retry-After 30 (the first backoff step); capacity on: served again 45 s later on the backoff retry |
| `spend_cap` | PASS: a synthetic terminated `serving_pods` row ($1M/h for 60 s) put month-to-date over the $100 cap: the wake was refused before any create (no row written), pool `unavailable` / `spend_cap`, 503 with Retry-After 300; the row and the three alert rows it caused (50/80/100%) deleted, `reset`, served again 45 s later |
| `idle_stop` | PASS: with `LOOM_POOLMGR_IDLE_STOP_S=600` (compose), one request then no demand: out of the pool hash 599 s after the request's answer (the manager's idle clock started at the tick that read its usage event; this run won a race with billing's trim that CI then lost, see spec §11 "Fixed after CI": the clock now starts at the tick that sees the model's `last_request` stamp change), pod terminated by the manager after a drain, pool `cold` |

Exit codes: `loom-m1-acceptance` 0, `loom-m2-acceptance` 0 (35 minutes for both).
