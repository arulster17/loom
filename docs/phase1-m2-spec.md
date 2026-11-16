# Phase 1 M2: the pool manager (spec)

Status: approved by the user 2026-11-15 (§15), branch `phase1-m2-pool`. Nothing here is
built yet.
M2 is free: everything runs against fakes and local mock pods, with no RunPod or AWS
calls. Background: [phase1-runpod-plan.md](phase1-runpod-plan.md) (§5 lifecycle, §7
routing, §8 failures, §10 security, §12 reuse, §13 M2 row), the user decisions in
[PLAN.md](PLAN.md#decisions-so-far) (2026-10-31: on demand, no GPU fallback, separate
RunPod account, US datacenters only), and the fixed interfaces in
[phase1-contracts.md](phase1-contracts.md) (§9 gateway to pod, §10 pools).

## 1. Goal

Replace M1's pool simulator (`poolsim`) with a **pool manager** that creates, bootstraps,
health-checks, drains and terminates real serving pods, and writes `loom:pool:<model>`
exactly as the gateway reads it today. M2 builds it and proves it offline. M4 runs it on
RunPod for the first time.

The M2 row of plan §13: against `FakeRunpod` and `FakePodExec`, a pod is created,
bootstrapped, made ready, drained and replaced; an expired lease terminates its pod; the
reaper ignores bench pods and other projects' pods; a pod killed mid-stream bills exactly
the streamed tokens; a retry happens only before the first byte; and a model with no
replica gets a 503 with `Retry-After`.

**What M1 already covers.** The gateway's half of that row is built and tested: it picks
the replica with the fewest in-flight requests, retries once on another replica before
the first byte, ends a stream cut mid-way with `upstream_lost` and bills the tokens
streamed, and answers 503 `model_unavailable` with `Retry-After`. Items a and f of
[phase1-m1-acceptance.md](phase1-m1-acceptance.md) check these against the mock. M2
re-proves them with the manager, not poolsim, writing the pool state. It changes the
gateway only where §8 says.

## 2. Scope

In scope:

1. `services/poolmgr` (`loom-poolmgr` / `loom_poolmgr`): the manager service, its CLI and
   its Prometheus metrics.
2. `config/serving.yaml` and its loader `loom_platform.serving`. The gateway reads its
   cold-start estimates from this file instead of `COLD_START_ESTIMATE_S`.
3. Migration `0010_serving_pods.sql`: pod records and the spend that accrues on them.
4. Serving scripts for the pod: start command with lease watchdog, bootstrap, engine
   start, lease renewal.
5. Small, backward-compatible changes to the bench RunPod modules the manager reuses
   (§10).
6. A serving rule in the RunPod reaper Lambda that reads S3 lease records.
7. Two pod proxy changes: reload certificates and token on SIGHUP, and run as a pinned
   bundle on the pod.
8. A `local` backend so `docker compose up` runs the real manager on mock pods, plus
   `loom-m2-acceptance`.
9. Contract and doc updates (§12), and the rest of R15 (plan §13): run CI 10x and stop
   the dependent tests in `test_quality_e2e` from cascading.

Out of scope, with the milestone each belongs to:

| Not in M2 | Where |
|---|---|
| Any paid call: RunPod create, real SSH, S3 | M4 (cap $20) |
| Secrets Manager, Terraform, deploying the serving reaper Lambda, the gateway's daily client-certificate rotation | M3 |
| Network volumes, autoscaling on queue depth, warm spare pods | Later (plan §5, §6) |
| Gateway tie-break on `vllm:num_requests_waiting` and skipping replicas at goodput (plan §7) | Matters only with several replicas per model: Phase 3. One replica per model under on-demand makes it moot |
| R1-R14 review follow-ups | M3 / M5 as the table there says |

## 3. Where it runs

One process (`loom-poolmgr run`) next to the gateway. In M3 it runs on the control-plane
instance, so its egress IP is the gateway's and the pod proxy allowlist covers both. It
holds:

- a RunPod API key for the **serving** account (decision 2026-10-31). In M2 this is
  `RUNPOD_API_KEY` in env and is unused offline;
- the Loom pod CA certificate and key, to issue pod server certificates;
- a client certificate signed by the same CA, for its own health checks through each
  pod's proxy (the same path the gateway uses);
- a long-lived SSH key that serving pods authorize. The bench account key is never
  authorized on a serving pod (`authorize_account_key` off).

**Single writer.** At start the manager takes the Postgres session advisory lock
`pg_try_advisory_lock(hashtext('loom-poolmgr'))` and exits non-zero if another manager
holds it. A dropped connection releases the lock. Two managers would create pods twice
and fight over the pool hash. Do not run poolsim and the manager against the same Redis:
compose runs one or the other (§9).

## 4. Configuration: `config/serving.yaml`

`config/models.yaml` stays the source of what is served and at what price. Deployment
settings go here, because the registry's `scaling` field is part of every benchmark config
hash (plan §12). Loaded and validated by `loom_platform.serving.load_serving()`
(Pydantic, `extra="forbid"`). The env var `LOOM_SERVING_YAML` overrides the path.

```yaml
version: 1
provider: runpod                   # runpod | local (compose) | fake (tests)
spend:
  monthly_cap_usd: 100             # plan §15 q1 suggestion for on-demand; confirm before M4
  alert_fractions: [0.5, 0.8, 1.0]
  max_price_ratio: 1.25            # refuse a pod priced above bench/prices.yaml x this
placement:
  data_center_prefixes: ["US-"]    # decision 7: US only; resolved against RunPod's stock list
  allowed_cuda_versions: ["13.0"]
  container_disk_gb: {qwen3-8b: 80, llama-3.3-70b-instruct-fp8: 120}
network:
  gateway_egress_cidrs: []         # pod proxy allowlist; empty only with provider local/fake
  proxy_port: 8443
lifecycle:
  reconcile_every_s: 15
  lease_ttl_s: 3600                # a pod lives at most this long past the last renewal
  lease_renew_every_s: 300
  drain_max_s: 600
  server_cert_days: 7
  server_cert_renew_before_s: 172800   # re-issue over SSH when < 2 days left
  max_replacements_per_hour: 3     # per model; past it: unavailable, alert, admin reset
health:
  pod_state_every_s: 30
  proxy_health_every_s: 10
  proxy_health_failures: 3
  unhealthy_grace_s: 120
  progress_stall_s: 60
  bootstrap_stall_s: 600           # the bench's engine_stall_s
  bootstrap_timeout_s: 1800
capacity:
  backoff_s: [30, 60, 120, 240]    # the bench's capacity_backoff
  unavailable_retry_after_s: 300
models:
  qwen3-8b:
    min_replicas: 0                # 0 = on demand (decision 2026-10-31); >= 1 = always on
    max_replicas: 1                # hard cap on non-terminated pods
    idle_stop_s: 1800
    cold_start_estimate_s: 240
    engine_env: {}
  llama-3.3-70b-instruct-fp8:
    min_replicas: 0
    max_replicas: 1
    idle_stop_s: 1800
    cold_start_estimate_s: 450
    engine_env: {NCCL_P2P_LEVEL: NVL}   # the env 9f0853d7's FP8 cell ran with (pinned by a test)
```

**The committed `config/serving.yaml`** is the compose default: the values above with
`provider: local` (the local backend, §9), so `gateway_egress_cidrs` may stay empty. The
production file with `provider: runpod` and the control plane's egress IP is deployed in
M3 (and the manager refuses `local`/`fake` outside `LOOM_ENV=dev`, §11). An explicit
`load_serving(path)` wins over `LOOM_SERVING_YAML`, which wins over the default path (as
`load_registry` does). Validation errors are `ValueError`s (Pydantic's
`ValidationError` is one) whose message names the field.

**Gateway without the file:** the `Router` loads the default `serving.yaml` itself
(`LOOM_SERVING_YAML` honoured). If that names a missing file, every model gets
`DEFAULT_COLD_START_S` and the gateway still starts. `create_app(cold_start_s=...)`
overrides stay.

Validation at load:

- every model is servable (`pricing.servable`) and has a `runpod` entry in
  `hardware.instance_types`;
- `min_replicas <= max_replicas`;
- `engine_env` has no secret names (the bench's validation rule);
- outside `provider: fake|local`, `gateway_egress_cidrs` is non-empty;
- a servable model missing from `models` is never started. Its pool is written
  `unavailable` with reason `not_configured`.

Also (decided after the wave 1 review; tests and code in wave 3): `lease_renew_every_s <
lease_ttl_s`; `server_cert_renew_before_s < server_cert_days x 86400`;
`bootstrap_stall_s <= bootstrap_timeout_s`; `alert_fractions` unique. `idle_stop_s: 0`
means never idle-stopped (as poolsim's 0).

A test pins each served model's `engine_env` to the one in the benchmark spec that priced
it (`bench/experiments/llama-3.3-70b-h100-tp2-runpod.yaml`, cell `vllm-tp2-fp8`, for the
70B; `qwen3-8b-config-sweep-runpod.yaml`, cell `bf16`, for the 8B), so the served launch
stays the benchmarked one.

GPU type and count come from the registry (`hardware.gpu` through `RUNPOD_GPU_TYPE_IDS`,
`gpus_per_replica`). The listed price comes from `bench/prices.yaml` through the row's
`instance_types.runpod` (`l40s-x1`, `h100-sxm-x2`).

## 5. Pod lifecycle

### States and records

Each pod has a row in `serving_pods`, written **before** RunPod is called. That lets an
ambiguous create be found and cleaned up, and a restart adopt what it owns.

```
creating -> bootstrapping -> ready -> draining -> terminated
     \            \             \________________/
      \____________\______________ failed -> terminated
```

| State | Meaning | In routing |
|---|---|---|
| `creating` | Row written, RunPod create called, waiting for `RUNNING` and the SSH and 8443 port mappings | No |
| `bootstrapping` | SSH up; certificate, token and proxy delivered; weights; engine start; proxy `/health`; one-token probe | No |
| `ready` | Probe passed | **Yes** |
| `draining` | Removed from the pool hash; in-flight requests finish (`vllm:num_requests_running` = 0) or `drain_max_s` passes | No (already-open streams continue) |
| `failed` | A health rule fired or bootstrap failed; terminate now | No |
| `terminated` | `terminate_pod` confirmed (404 counts), or RunPod reports it gone | No |

Always terminate, never stop: a stopped pod keeps billing for its disk.

Migration `0010_serving_pods.sql`:

```sql
serving_pods (
  id uuid PRIMARY KEY, model_id text NOT NULL, pod_name text UNIQUE NOT NULL,
  pod_id text UNIQUE,                         -- null until create answers
  state text NOT NULL,                        -- creating|bootstrapping|ready|draining|failed|terminated
  gpu_type_id text, gpu_count int, data_center text,
  cost_per_hr_micros bigint,                  -- RunPod costPerHr + disk, from the API
  public_ip text, proxy_port int, ssh_port int,
  cert_not_after timestamptz, lease_until timestamptz,
  stages jsonb NOT NULL DEFAULT '{}',         -- pod_created ... first_token, as the bench names them
  failure text, terminated_by text,           -- manager|watchdog|reaper|admin|runpod
  created_at timestamptz NOT NULL, ready_at timestamptz,
  drain_started_at timestamptz, terminated_at timestamptz)
serving_spend_alerts (month date, fraction numeric, sent_at timestamptz,
  PRIMARY KEY (month, fraction))
```

Secrets are never stored: no token, no key, no certificate.

### Reconcile loop (every `reconcile_every_s`)

For each configured model:

1. **Desired count.** `max(min_replicas, 1 if there is demand else 0)`, capped at
   `max_replicas`. Demand means a wake key exists (`loom:pool:<model>:wake`), or the model
   is ready and had a usage event or an in-flight request within `idle_stop_s`. Usage is
   read from `loom:usage` with `XREAD` and no consumer group, as poolsim does, so billing
   loses no events.
2. **Observe.** One `list_pods()` per tick, matched to rows by `pod_id`, applying the
   health rules below.
3. **Act.** Create, drain or terminate to close the gap; replace failed pods.
4. **Publish.** Write the pool hash (§7).

A create runs as its own task and the loop does not wait for it. Bootstraps run in
parallel across models.

### Create

- Name: `loom-serve-<model>-<created epoch>-<6 hex>`. The model id is kept whole (it
  matches `[a-z0-9.-]+`); the reaper rule (§6) parses from the right.
- Body: the bench's `pod_body` shape with `ports: ["22/tcp", "8443/tcp"]`, no HF secret
  (both checkpoints are ungated), `dataCenterIds` = the datacenters with stock
  (`gpu_stock`) whose id starts with a configured prefix, and env `LOOM_MANAGED=true`,
  `LOOM_ROLE=serving`, `LOOM_MODEL`, `LOOM_LEASE_EPOCH` (now + `lease_ttl_s`) and
  `LOOM_SSH_PUBKEY`.
- The ambiguous-create cleanup and the price refusal follow the bench (`_create`,
  `max_price_ratio`). Above the ratio, the pod is terminated, the row goes `failed`
  (`price_too_high`), and the pool goes `unavailable` (reason `price_too_high`).
- **Spend guard**, checked before the row is written:
  `month_to_date + listed_price x lease_ttl_s <= monthly_cap`. `month_to_date` sums
  `cost_per_hr_micros x billed hours` over this UTC month's rows, running pods counted up
  to now. Refused: no pod, pool `unavailable` (reason `spend_cap`), alert. Alerts fire
  once per (month, fraction) through `serving_spend_alerts`. The guard **never**
  terminates a running healthy pod; that is an admin action (plan §5).
- **No capacity** (`RunpodApiError.no_capacity`, or no US datacenter with stock): the row
  is `terminated` with `failure = no_capacity`, and the pool goes `unavailable` (reason
  `no_capacity`, `retry_after_s` = the next backoff step). The next attempt follows
  `capacity.backoff_s` and happens only while demand lasts (a live wake key, or
  `min_replicas >= 1`). There is no fallback to another GPU type (decision 2026-10-31).
- `max_replicas` counts every row not `terminated`, and is checked in the same
  transaction that inserts the row.

### Bootstrap (over SSH, scripts on stdin, `run_detached`)

1. Wait for `RUNNING` and the `22` and `8443` mappings (`public_port(pod, port)`, §10).
2. `serve_bootstrap.sh` creates two uids, `loomproxy` and `loomengine`, and writes
   `/etc/loom/proxy/{server.crt,server.key,ca.crt,token,allow}`, mode 0600, owned by
   `loomproxy`. It installs the pinned proxy bundle (§9, checked against its sha256) and
   starts the proxy as `loomproxy` on `:8443` with upstream `127.0.0.1:8000`. Secrets
   arrive only on SSH stdin: never in the pod env, never on a command line.
3. `serve_engine.sh` is the bench's `start_engine.sh` with the job user and job isolation
   removed. The engine runs as `loomengine`, with `HF_HUB_OFFLINE=1` after the download,
   bound to `127.0.0.1:8000`, from `render_launch`/`engine_process_argv` on the registry
   row plus `--enable-prompt-tokens-details` (plan §9). Stall detection uses
   `bootstrap_stall_s`, and stage markers are recorded as the bench records them. It also
   checks that `loomengine` cannot read `/etc/loom/proxy/*` or `/proc/1/environ`.
4. From the manager: proxy `GET /health` with its client certificate, then a one-token
   `POST /v1/completions`, both through `https://<ip>:<mapped 8443>` with `server_name` =
   pod id.
5. The row goes `ready`, with stages and `ready_at` recorded. The cold-start stage
   durations go to metrics and into `stages`.

Any failure, or no first token within `bootstrap_timeout_s`: `failed`, terminate, and
count toward `max_replacements_per_hour`.

### Health rules (plan §5)

| Check | Every | Fires when | Action |
|---|---|---|---|
| Pod state (`pod_loss` on the listing) | `pod_state_every_s` | `EXITED`, `TERMINATED` or gone | Out of routing now; `terminated_by = runpod`; replace if demand |
| Proxy `GET /health` | `proxy_health_every_s` | `proxy_health_failures` in a row | Out of routing; `failed` if not healthy again within `unhealthy_grace_s` (then replace) |
| Engine progress (`/metrics`) | `proxy_health_every_s` | `vllm:num_requests_running > 0` and `vllm:generation_tokens_total` flat for `progress_stall_s` | `failed`, replace |
| Bootstrap | once | as above | `failed`, replace |
| Replacements | per model | more than `max_replacements_per_hour` | No new pod; pool `unavailable` (reason `crash_loop`); alert; `loom-poolmgr reset <model>` clears it |

The gateway's own per-request penalty (`replica_penalty_s`, 10 s) stays as built. It
covers the seconds between a failure and the manager's next health check. The plan §5
circuit (3 errors, 30 s) is not added: the manager's health rules remove a persistently
failing replica, and the penalty handles the rest.

### Idle stop and drain

A ready pod with no demand for `idle_stop_s` and fewer live pods than `min_replicas`
would leave goes `draining`:

1. The manager rewrites the pool hash without the pod.
2. It waits at least 2 x `pool_cache_ttl_s` (the gateway caches the hash for up to 1 s).
3. It waits for `vllm:num_requests_running` = 0 or `drain_max_s`, then terminates.

A request arriving while the last pod drains sees `cold`, wakes the model, and starts a
new pod. The draining pod is not taken back: adding that would gain little and adds a
race.

### Leases and the watchdog (plan §5)

- `serve_start.sh` is the start command, based on `pod_start.sh`. The watchdog terminates
  the pod (the bench's `loom_terminate`, pod-scoped key) once `now >= lease`, where
  `lease = max(LOOM_LEASE_EPOCH, contents of /run/loom/lease)`, re-read every 30 s. It
  starts sshd (key-only, root-only, no forwarding), authorizes only `LOOM_SSH_PUBKEY`, and
  ends in `sleep infinity`. A setup failure terminates the pod, as in the bench. The
  absolute TTL and the job user are removed.
- Every `lease_renew_every_s` the manager runs `serve_lease` over SSH, rendered with
  `LEASE_FILE` and `LEASE_EPOCH` (§11; an atomic write of `/run/loom/lease`). It then writes the S3 lease record
  `s3://<bucket>/serving/leases/<pod id>` (JSON `{pod_id, pod_name, model, lease_until,
  written_at}`, both times ISO 8601 with a UTC offset as `datetime.isoformat()` writes
  them; the bucket is `LOOM_POOLMGR_LEASE_BUCKET`). A failed SSH renewal is
  retried each tick. The pod stays in routing: the watchdog decides, and the manager marks
  the pod `failed` only when `lease_until` is within 2 renewal periods.
- A dead control plane costs at most `lease_ttl_s` of GPU time per pod: $1.10 for the 8B
  and $8.02 for the 70B at the measured rates.

### Certificates

The manager issues each pod a server certificate (SAN = pod id, `serverAuth`,
`server_cert_days`) and itself a client certificate (`clientAuth`) from the CA it is
given. When a pod's certificate is within `server_cert_renew_before_s` of expiry, the
manager writes a new one over SSH and sends the proxy SIGHUP (§9). Open connections are
kept. The gateway's client certificate rotation is M3.

### Restart and adoption

At start, before the first reconcile:

| RunPod lists | `serving_pods` row | Action |
|---|---|---|
| a `loom-serve-*` pod | not `terminated` | **Adopt.** Re-key over SSH (new token and server certificate, SIGHUP the proxy, renew the lease), run the health check and probe, and put it back in routing. If any step fails: `failed`, terminate |
| a `loom-serve-*` pod | none | Orphan: terminate (`terminated_by = manager`, logged at error). Only pods with `LOOM_MANAGED=true` and an exact name match; the other project's pods do not exist in the serving account and would not match anyway |
| nothing | not `terminated` | `terminated`, `terminated_by = runpod` (gone while the manager was down) |

The pool hash is rewritten from scratch after adoption. Re-keying means tokens never need
to be stored anywhere but the pool hash.

## 6. Reaper serving rule (second backstop)

`runpod_reaper_lambda.py` gains a serving rule, enabled by `LOOM_RUNPOD_REAPER_SERVING=true`
and `LOOM_RUNPOD_REAPER_LEASE_BUCKET` (enabled without a bucket: the run fails closed,
raising and deleting nothing). It is deployed against the serving account in M3, whose
role needs `s3:ListBucket` (without it S3 answers 403 for a missing key, which fails
closed every run). Staleness is judged on `written_at` only, never `lease_until`.
A pod is reaped when **all** of these hold:

- its name matches `^loom-serve-[a-z0-9.-]+-\d{10}-[0-9a-f]{6}$` exactly;
- `LOOM_MANAGED=true` and `LOOM_ROLE=serving` are in its env;
- its id is valid;
- it is older than 2 hours (it covers a pod stuck before its container starts);
- its lease record is missing, or `written_at` is more than 2 hours ago. A failure to read
  S3 (other than a missing key; a missing bucket counts as a failure) fails closed: the
  run raises and deletes nothing, bench pods included. A record that cannot be parsed
  never leads to a reap.

The existing fail-closed rules apply to the union of both rules: nothing is deleted on a
bad listing, at most `max_per_run` per run, and dry run is the default. The bench TTL rule
never matches a `loom-serve-` name, whatever `LOOM_RUNPOD_REAPER_PREFIX` is, and the serving rule never matches a bench name
(`loom-bench-...-<ttl>-<hex>`) or anything else. Both directions are tested. boto3 is
imported only when the serving rule is enabled, so the bench Lambda stays stdlib-only.

## 7. Pool state the manager writes (contracts §10)

`loom:pool:<model>` keeps its fields, plus one: `reason`. The gateway ignores it except
for logs and metrics.

| Condition (evaluated in this order) | `state` | `retry_after_s` | `reason` |
|---|---|---|---|
| >= 1 `ready` pod | `ready`, replicas = the ready pods | 0 | |
| crash loop, spend cap, price too high, not configured | `unavailable` | `unavailable_retry_after_s` | `crash_loop` / `spend_cap` / `price_too_high` / `not_configured` |
| last create answered no capacity, nothing in flight | `unavailable` | the next backoff step | `no_capacity` |
| a pod is `creating` or `bootstrapping` | `warming` | `max(15, cold_start_estimate_s - elapsed)` | |
| otherwise | `cold` | `cold_start_estimate_s` | |

**Contract change: wakes on `unavailable`.** Today the gateway only wakes a `cold` model,
so a model that went `unavailable` for lack of capacity would never be retried after its
wake key expired. From M2, the gateway's `check_routable` also calls `request_wake` on
`unavailable` (one `SET NX`, rate-limited like any 503). The manager treats a live wake
key as demand and keeps retrying capacity on the backoff schedule. It ignores the wake
for `spend_cap`, `price_too_high`, `crash_loop` and `not_configured`, which need an admin.
The client still gets 503 "no GPU capacity" with `Retry-After`.

## 8. Gateway changes

Kept small. Each one updates the contracts doc and tests in the same change.

1. `Router.cold_start_s` comes from `serving.yaml` (`cold_start_estimate_s`). The
   `COLD_START_ESTIMATE_S` constants are removed. When the file is absent (gateway tests),
   the fallback is `DEFAULT_COLD_START_S`.
2. Wake on `unavailable` (§7).
3. Log and count the pool `reason` on 503s
   (`loom_gateway_model_unavailable_total{model,state,reason}`).

Nothing else changes: routing, retry, mid-stream billing and the 503 shape are M1's and
must keep passing unchanged.

## 9. Pod proxy on RunPod, and the local backend

**Decision (contracts §13, "decide in M2"): run the Python `loom-podproxy` on pods, not
Caddy.** The proxy tests are its specification. It already does the token check, the
allowlist before the handshake, `/tokenize`, and error shapes the gateway depends on;
Caddy would have to re-implement and re-prove them. Changes:

- **SIGHUP** reloads certificate, key, CA, token and allowlist from their files (the
  allowlist from a new `--allow-ip-file PATH`, one CIDR or address per line). New
  connections use the new ones and open connections are kept. A reload is all or
  nothing: if any file fails to parse, the whole old configuration stays and an error is
  logged. An empty or blank allowlist file is a parse failure (at start: refuse to
  start), never "any source": a truncated file must not open the proxy.
- **Bundle.** `loom-podproxy-bundle --out DIR [--lock FILE]` (lock default
  `config/serving-artifacts.lock`, committed) builds, from the committed git blobs
  (`git archive HEAD`, so the hash is the same on a CRLF Windows checkout and on Linux),
  one tarball holding the `loom-podproxy` wheel
  and its locked dependency wheels (h11, httpx and their dependencies; nothing from the
  vLLM image's site-packages). It records the tarball's sha256 in
  `config/serving-artifacts.lock`. Bootstrap ships it on SSH stdin, checks the sha256,
  and installs it into `/opt/loom/proxy-venv` with the image's `python3 -m venv
  --without-pip` plus `python3 -m pip install --no-index --find-links`. Whether the
  pinned vLLM image's Python can do this is checked in M4's smoke, before any serving.

**Local backend (`provider: local`, compose).** Each "pod" is a subprocess pair inside
the poolmgr container: the bench mock engine on a loopback port and `loom-podproxy` with
a certificate issued by the manager. The SSH steps are replaced by direct file writes.
`LOOM_POOLMGR_LOCAL_COLD_START_S` (default 20, as poolsim) adds the delay. The create,
health, drain, idle and pool-hash code is the same as for RunPod: only the
`ServingProvider` and `PodExec` implementations differ. Compose swaps `poolsim` for
`poolmgr` (local backend, the committed `serving.yaml` with its `idle_stop_s`, the mock pod containers
kept for M1's fixed-replica checks only if a test still needs them). poolsim stays in the
repo for its own tests and is no longer started by compose. `loom-m1-acceptance` drives
the manager's CLI wherever it called `loom-poolsim set` (§11) and must still pass all ten
items.

## 10. Reuse from the bench (plan §12, as found in the code)

| Piece | How |
|---|---|
| `runpod_api.RunpodApi` | Imported as is (sync; called with `asyncio.to_thread`). Add `user_agent=` (default `loom-bench`; the manager sends `loom-poolmgr`) and `public_port(pod, port)`, generalising `public_ssh` |
| `runpod.pod_loss`, `base.capacity_backoff` / `CAPACITY_BACKOFF_S`, `RUNPOD_GPU_TYPE_IDS` | Imported as is |
| `RunpodProvider._create` cleanup, the price refusal, `_wait_for_ssh` | Logic re-implemented in `loom_poolmgr.runpod` (the class needs an experiment spec, S3 and the job user). The cleanup and refusal are small enough to copy with their tests |
| `runpod_ssh` (`OpenSshExec`, `SshTarget`, `run_detached`, `generate_keypair`) | Imported as is; `generate_keypair` gains a `comment=` argument |
| `engines.render_launch`, `engine_process_argv` | Imported as is |
| `prices.load_prices().instance_price` | Listed price for the ratio check |
| `aws_ssm.render_script`, `parse_markers`, `stage_offsets`, `common.sh` (`stage`, `sysinfo`, `fail`) | Imported for the serving scripts |
| `pod_start.sh`, `start_engine.sh` | Copied to `loom_poolmgr/scripts/serve_*.sh` and changed as §5 says. The bench scripts are not touched |
| `FakeRunpod` (`bench/tests/runpod/fakes.py`) | Moved to `loom_bench.providers.runpod_fake` (with its pod fixture, `runpod_fake_pod.json`) so other packages can import it. `fakes.py` keeps a thin subclass whose pod template goes through its own `create_response_fixture`, because `test_runpod_provider.py` monkeypatches that name; the bench tests are unchanged. `FakePodExec` stays; the manager has its own `FakeServingExec` (it knows the serve scripts) in `loom_poolmgr.testing` |

## 11. Package layout and interfaces

```
services/poolmgr/
  pyproject.toml          # deps: loom-platform, loom-bench, loom-podproxy, cryptography, httpx, boto3
  src/loom_poolmgr/
    __main__.py, cli.py   # loom-poolmgr run|status|warm|drain|stop|reset|issue-ca
    manager.py            # reconcile loop, demand, desired counts, adoption
    provider.py           # ServingProvider protocol; RunpodServing, LocalServing
    runpod.py             # create body, ambiguous-create cleanup, price check, ports
    bootstrap.py          # script rendering and the bootstrap steps
    health.py             # /health, /metrics parsing, progress, probe
    leases.py             # SSH renewal; LeaseRecords protocol: S3LeaseRecords, MemoryLeaseRecords
    spend.py              # month-to-date, guard, alerts
    store.py              # serving_pods queries
    pki.py                # CA load, issue server/client certs (devcerts reuses it)
    metrics.py            # Prometheus on LOOM_POOLMGR_METRICS_PORT (9102)
    testing.py            # FakeServingExec, fake clock, pod network helpers
    scripts/serve_start.sh, serve_bootstrap.sh, serve_engine.sh, serve_lease.sh
  tests/test_poolmgr_*.py
```

`ServingProvider` (async):

```python
class ServingProvider(Protocol):
    async def create(self, model: ServedModel, name: str, env: dict[str, str]) -> PodInfo: ...
    async def list(self) -> list[PodInfo]: ...  # loom-serve-* only
    async def terminate(self, pod_id: str) -> None: ...  # idempotent; 404 is success
    def endpoints(self, pod: PodInfo) -> Endpoints | None: ...  # ip, proxy port, ssh port
```

The clock is injected everywhere (`clock: Callable[[], float]`), so every timing rule is
tested without sleeping.

**Serve scripts** (the seam the script tests use). `loom_poolmgr.scripts.render(name,
**values) -> str` has the bench's `render_script` contract: exactly the variables on the
template's `# requires:` line, values quoted, `set -euo pipefail`, and the package's own
`scripts/common.sh` (the generic `stage`, `sysinfo`, `fail` helpers copied from the bench)
prepended. `required_vars(name)` lists them. Required variables:

| Script | Requires | Reads from the pod env |
|---|---|---|
| `serve_start` (the start command) | `LEASE_FILE` (`/run/loom/lease`), `LEASE_POLL_S` (30), `STAGE_FILE`, `LOOM_ROOT`, `SSH_DIR`, `SSHD_CONFIG`, `GRAPHQL_URL` | `LOOM_LEASE_EPOCH`, `LOOM_SSH_PUBKEY`, and for the terminate call `RUNPOD_API_KEY`, `RUNPOD_POD_ID` |
| `serve_lease` | `LEASE_FILE`, `LEASE_EPOCH` | |
| `serve_bootstrap` | `PROXY_DIR` (`/etc/loom/proxy`), `PROXY_VENV` (`/opt/loom/proxy-venv`), `PROXY_PORT`, `UPSTREAM`, `BUNDLE_SHA256`, `PROXY_UID`, `ENGINE_UID` | secrets arrive on stdin after the script, never as variables |
| `serve_engine` | as the bench's `start_engine` minus `JOB_UID`/`JOB_HOME`, plus `ENGINE_UID`, `PROXY_DIR` | |

The watchdog terminates the pod as the bench's does: `python3 -c <terminate code>
"$GRAPHQL_URL"` with the pod-scoped key from the environment, retried every 30 s. The
script tests stub `python3`, `date`, `sshd` and the other binaries on `PATH`, as
`bench/tests/runpod/test_runpod_scripts.py` does.

**Manager API.** Wave 2a starts by committing the public surface as stubs
(`PoolManager`, its constructor's injected parts, `start`/`reconcile_once`/`stop`, the
`ServingProvider` and `LeaseRecords` protocols, `FakeServingExec`'s expected script
names); the lifecycle tests are written against those stubs.

CLI (admin, against the running manager's Postgres and Redis):

| Command | Does |
|---|---|
| `loom-poolmgr status` | Per model: pool state, reason, pods with state, age, price, lease; month-to-date spend against the cap |
| `loom-poolmgr warm <model>` | `request_wake` (pre-warm) |
| `loom-poolmgr drain <pod id>` | Drain and terminate one pod |
| `loom-poolmgr stop <model>` | Drain and terminate all of a model's pods; the model stays `cold` (wakes still start it) |
| `loom-poolmgr reset <model>` | Clear `crash_loop` / `spend_cap` holds |
| `loom-poolmgr issue-ca --out DIR` | Dev only: create a CA plus manager and gateway client certificates (as `loom-devcerts`) |

Settings (`LOOM_POOLMGR_*`, `PlatformSettings` subclass): `SERVING_YAML`, `CA_CERT`,
`CA_KEY`, `CLIENT_CERT`, `CLIENT_KEY`, `SSH_KEY`, `LEASE_BUCKET`, `METRICS_PORT`,
`LOCAL_COLD_START_S`, plus `RUNPOD_API_KEY`. Outside `LOOM_ENV=dev`, the manager refuses
to start without the CA, the client certificate, the SSH key, the lease bucket and a
non-empty `gateway_egress_cidrs`, and refuses `provider: local|fake`.

Metrics: `loom_poolmgr_pods{model,state}`,
`loom_poolmgr_cold_start_seconds{model,stage}`,
`loom_poolmgr_creates_total{model,result}` (ok, no_capacity, price_too_high, spend_cap,
error), `loom_poolmgr_replacements_total{model,why}`,
`loom_poolmgr_lease_renewals_total{result}`, `loom_poolmgr_spend_month_micros`,
`loom_poolmgr_spend_cap_micros`, `loom_poolmgr_reconcile_seconds`,
`loom_poolmgr_orphans_terminated_total`.

## 12. Contract and doc changes (same change as the code)

- `phase1-contracts.md`: §1 layout row for `services/poolmgr`; §2 the `LOOM_POOLMGR_*`
  and `LOOM_SERVING_YAML` variables and the poolmgr service, port 9102; §3 migration 0010;
  §6 the key table (pool hash `reason`; wake on `unavailable`); §9 the pod proxy decision
  and SIGHUP; §10 the manager as writer, the state table from §7, poolsim as a test tool;
  §13 close "Pod proxy on RunPod".
- `loom_platform.pool`: `PoolState.reason: str | None`, written and read; old hashes
  without it still load. `loom_platform.serving`: loader and schema. Tests for both.
- `phase1-runpod-plan.md` §13: M2 status; the circuit-breaker and waiting-tie-break
  deltas from §5 and §2 noted. `PLAN.md` status line.
- `.github/workflows/ci.yml`: add `services/poolmgr/src` to a mypy line. Tests run under
  the existing `pytest -q -n auto`.
- `services/poolmgr/README.md`: run, CLI, settings, what M4 must verify on a real pod.
- `docs/phase1-m2-acceptance.md`, written at the end like M1's.

## 13. Done when

Each item is an automated test unless it says otherwise. "Fake" means `FakeRunpod` plus
`FakeServingExec` plus a fake clock, with pods whose proxy is a real `loom-podproxy` in
front of the mock engine on localhost (real mTLS, real `/metrics`). Postgres and Redis
tests must **run**, not skip (CI sets both URLs; check the `s` count locally).

**Config and contracts**

1. `serving.yaml` loads. Every validation rule in §4 rejects a bad file, with a message
   naming the field. The 70B `engine_env` equals its benchmark spec's.
2. `PoolState` round-trips `reason`. A hash written by M1 code (no `reason`) still loads.
3. The gateway takes cold-start estimates from `serving.yaml`, and wakes on `unavailable`
   (a 503 for an `unavailable` model sets the wake key and publishes once per TTL).
4. Migration 0010 applies on a fresh database and on one migrated to 0009.

**Lifecycle (fake)**

5. Cold model plus wake: one create with the §5 body (name pattern, ports `22/tcp` and
   `8443/tcp`, no HF secret, `LOOM_MANAGED`/`LOOM_ROLE`/`LOOM_LEASE_EPOCH`, US datacenter
   ids only). Pool goes `warming` with a decreasing `retry_after_s`, then `ready` with
   exactly that replica, after bootstrap, health and probe. Stages are recorded.
6. Secrets delivery: the token, server key and certificate appear only on SSH stdin. They
   are never in the create body or env, any rendered command line, a log line (captured
   logs searched), or Postgres.
7. Idle for `idle_stop_s`: the pod is removed from the hash before termination, terminated
   only after in-flight reaches 0 (or `drain_max_s`), and the pool ends `cold`. A stream
   open on the pod when draining starts completes and is billed in full.
8. `min_replicas: 1`: started without a wake and never idle-stopped.
9. `max_replicas` is never exceeded, including with concurrent wakes, and in the moment a
   failed pod is being replaced.
10. Each health rule in §5 fires on its condition and on nothing else: pod gone or
    `EXITED`; 3 failed health checks, then back within the grace (stays) or not
    (replaced); progress stall; bootstrap stall and timeout. A replacement is created only
    when there is demand.
11. More than `max_replacements_per_hour` replacements: no further create, pool
    `unavailable` with `crash_loop`. `reset` clears it.
12. No capacity: pool `unavailable` with `no_capacity` and the backoff's `retry_after_s`.
    Retries follow `backoff_s` only while the wake key lives and stop after. A later wake
    retries again. No other GPU type is ever requested.
13. Price above `max_price_ratio`: the pod is terminated and the pool goes `unavailable`
    (`price_too_high`). An ambiguous create (timeout or 5xx) finds the pod by name and
    terminates it.
14. Spend guard: a create that would cross the cap is refused before RunPod is called.
    50/80/100% alerts fire once per month each, also across a restart. A running ready
    pod is never terminated by the guard. Month-to-date spend is correct across a month
    boundary.
15. Restart: a second manager cannot start while the first holds the lock. After a
    restart, a ready pod is adopted and re-keyed (old token refused by the proxy, new one
    accepted) without a new create; an orphan `loom-serve-*` pod is terminated; a
    non-serving pod in the listing is never touched.
16. Certificates: issued with SAN = pod id and the right key usages. Renewal before expiry
    reaches the proxy through SIGHUP while a stream stays open.

**Leases, watchdog, reaper**

17. Renewal writes `/run/loom/lease` over SSH and the S3 record (moto) every
    `lease_renew_every_s`. A failing renewal marks the pod `failed` only near lease end.
18. `serve_start.sh`, run under bash with a stub `python3` and `date`: terminates at
    `LOOM_LEASE_EPOCH` with no lease file, later when the lease file moves the deadline,
    never earlier than the larger of the two, and also on setup failure.
19. Reaper serving rule (moto plus `FakeRunpod`): reaps a matching pod older than 2 hours
    with a missing or stale lease record. Keeps one with a fresh record, one younger than
    2 hours, one without `LOOM_MANAGED`/`LOOM_ROLE`, every bench-named pod, and every
    other name. Fails closed on an S3 error and on a bad listing. Respects `max_per_run`.
    With the rule disabled, the Lambda imports no boto3.

**Gateway with the manager in the loop** (real gateway app, real Redis and Postgres,
manager on the fake provider)

20. A pod killed mid-stream (its engine process killed): the client gets `upstream_lost`
    and `[DONE]`, the usage event bills exactly the tokens streamed, and the manager marks
    the pod failed and replaces it.
21. A replica that fails before the first byte: retried once on the second replica (two
    pods, `max_replicas: 2`), billed once. With one replica: 502 `upstream_lost`, billed
    nothing.
22. No ready replica: 503 `model_unavailable` with `Retry-After` from the pool, for each
    of `cold`, `warming` and `unavailable`, with the reservation released.

**Pod proxy**

23. SIGHUP reload: certificate, token and allowlist change for new connections, an open
    stream survives, and a bad file keeps the old configuration. All existing proxy tests
    still pass.
24. The bundle builds reproducibly (same sha256 twice from the lock), installs into an
    empty venv with `--no-index`, and `loom-podproxy --help` runs from it.

**Stack and CI**

25. `docker compose up -d --build --wait` with the local backend: all services healthy.
    `uv run loom-m1-acceptance` passes all ten items against the manager. A new
    `uv run loom-m2-acceptance` runs, from the outside: cold, wake, ready; idle stop;
    kill a pod mid-stream and see it replaced; force no capacity (a `local` backend
    switch) and see 503 "no GPU capacity"; spend cap refusal. Exit 0.
26. All CI checks (CLAUDE.md list plus the new mypy path) pass. The full Python suite
    (`uv run pytest -q -n auto`, Postgres and Redis tests running) passes **5 runs in a
    row on Linux with 4 CPUs**, and CI passes on the final commit (R15's open item). And
    `test_quality_e2e` no longer cascades (a forced failure in one test leaves the
    others' outcomes unchanged).
27. Manual check before M4 (not automated): read `serve_*.sh` against the pinned vLLM
    image's user, Python and paths, and list in the README what M4 must check on a real
    pod (proxy venv install, `loomengine` isolation, the 8443 port mapping, GPU metrics
    fields).

## 14. Order of work

Tests first for each package (the `agent-setup:test-writer` writes them from §13; tests
stay locked while code is written):

1. **Contracts:** `serving.yaml` and loader, `PoolState.reason`, migration 0010, the
   gateway changes in §8, the contracts doc (items 1-4).
2. **Bench reuse:** `RunpodApi.user_agent`, `public_port`, the `generate_keypair`
   comment, the `FakeRunpod` move (no behavior change; the bench suite passes unchanged).
3. **Manager core** on the fake provider: lifecycle, health, spend, adoption, pki
   (items 5-16).
4. **Scripts, leases, reaper** (items 17-19). These can run in parallel with step 3 once
   step 1 is merged.
5. **Proxy:** SIGHUP and the bundle (items 23-24). Parallel with step 3.
6. **Integration:** gateway in the loop, local backend, compose, `loom-m2-acceptance`,
   M1 acceptance re-run (items 20-22, 25).
7. **CI and docs:** the 5 runs on 4 CPUs, the R15 cascade, README, the acceptance note
   (items 26-27).

The `FakeRunpod` move (step 2) is done first, as its own commit, before the tests are
locked: it touches `bench/tests/runpod/fakes.py`, which locking protects.

### Agent plan

| Wave | Agents | Works in | Steps | Items | Starts when |
|---|---|---|---|---|---|
| 0 | `agent-setup:test-writer` | `phase1-m2-pool` | Tests for every item of the wave that follows, from §13 only | | Before each wave; tests then locked |
| 1 | One implementer | `phase1-m2-pool` | 1 (contracts) and the rest of 2 (bench reuse) | 1-4 | Spec committed, tests locked |
| 2a | Implementer: **manager core** | own worktree, branch `m2-manager` | 3, plus the scripts and leases of 4 | 5-18 | Wave 1 merged |
| 2b | Implementer: **proxy and reaper** | own worktree, branch `m2-proxy-reaper` | 5, plus the reaper rule of 4 | 19, 23, 24 | Wave 1 merged; parallel with 2a |
| 3 | One implementer: **integration** | `phase1-m2-pool` | 6 and 7 | 20-22, 25-27 | 2a and 2b merged into `phase1-m2-pool` |

Rules for the parallel wave:

- **2a owns:** `services/poolmgr/**` (except `pki.py`'s use by devcerts, which waits for
  wave 3) and the `serve_*.sh` scripts. **2b owns:** `services/podproxy/**` and
  `bench/src/loom_bench/providers/runpod_reaper_lambda.py`. Neither touches the other's
  paths, `loom_platform`, the contracts doc or compose. A needed change there is reported
  back and made in wave 3.
- The one interface between them: 2a's bootstrap sends SIGHUP to the proxy and installs
  the bundle at the path in §9. 2b implements both exactly as §9 says, so neither waits
  on the other.
- Each wave ends with the stop gate green in its worktree, then a fresh-context
  `agent-setup:reviewer` pass on its diff against this spec before it merges.

After wave 3: a reviewer pass on the whole branch against this spec, and the
`agent-setup:verifier` on §13. M2 ends with a stop for the user before M3.

## 15. Decisions

Approved by the user 2026-11-15, with the agent plan (§14) and item 26's test runs as
amended that day.

1. **Pod proxy on RunPod:** the Python `loom-podproxy` shipped as a pinned bundle
   (recommended, §9), not Caddy.
2. **Pod records and spend in Postgres** (migration 0010), not Redis: spend is money and
   must survive a Redis loss.
3. **Compose runs the real manager** on a local backend instead of poolsim, so M1's
   acceptance exercises it (recommended, §9). It is more work than testing only on fakes,
   but it is the only end-to-end check of the manager before M4 spends money.
4. **Monthly serving cap** in `serving.yaml`: $100 placeholder (plan §15 q1 suggestion).
   It only matters from M4; RunPod billing (q3) is still open.
5. **Wake on `unavailable`** (contract change, §7), so a model that ran out of capacity is
   retried while there is demand.
6. **Deferred:** gateway tie-break on waiting requests and the 3-failure / 30 s circuit
   (§2, §5), both moot at one replica per model.
