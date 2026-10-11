# Phase 1 on RunPod: serving plan (draft for approval)

Status: draft, 2026-10-11. Nothing here is built, deployed or paid for. Phase 1 stays
paused until the user accepts Phase 0 ([PLAN.md](PLAN.md)); this plan is what Phase 1
builds once it resumes.

The user decided on 2026-10-11 that the Phase 1 MVP serves from RunPod and AWS stays the
second cloud ([PLAN.md](PLAN.md), "Pricing and Phase 1 serving cloud"). The build brief
(§8) assumed AWS first: EKS, a Helm chart, one Deployment per model per cloud, KEDA,
retry on the other cloud. This document says how Phase 1 works on RunPod instead, what
that costs, which of the brief's security requirements (§12) it meets differently, and
what needs a decision from the user.

Already decided and not reopened here: FastAPI + uvloop gateway, the PLAN.md ledger
draft, Stripe test mode with a user-created account (local stub until keys exist), and
the prices in `config/models.yaml` (`qwen3-8b` BF16 $0.077 / $0.368 per 1M input /
output on 1x L40S; `llama-3.3-70b-instruct-fp8` $0.571 / $0.797 on 2x H100 SXM, TP=2,
RunPod only).

## 1. The plan in brief

- **Control plane on AWS** (`us-east-1`): one small EC2 instance runs the gateway,
  billing worker, web app, Redis, Prometheus and Grafana under Docker Compose, behind
  Caddy for TLS; Postgres is a small RDS instance. About $50 a month.
- **GPU replicas on RunPod Secure Cloud on-demand pods**, one pod per replica, created
  and replaced by a **pool manager** that runs next to the gateway and reuses the bench
  RunPod provider (API client, pod creation, SSH bootstrap, engine start, watchdog,
  reaper).
- **The gateway reaches pods over the internet with mutual TLS.** RunPod pods cannot
  join an AWS private network, so each pod runs a small TLS proxy on one exposed TCP
  port that accepts only the gateway's client certificate (from Loom's own CA) and
  forwards to vLLM on `127.0.0.1`. The engine is never reachable directly, and RunPod's
  public HTTP proxy is not used.
- **No autoscaling in the MVP.** Each model has a fixed replica count (1 to start).
  Cold starts measured on RunPod are 2.7 to 7 minutes for the 8B and an estimated 5 to
  13.5 minutes for the 70B FP8, too slow to scale on demand per request.
- **Always-on costs real money:** $804 a month for the 8B, $5,851 for the 70B FP8, $6,655
  for both, at our measured hourly rates. At our prices, an always-on replica only
  breaks even when it is busy about 68% (8B) and 80% (70B) of the time. So the plan
  offers three availability modes and recommends starting with both models warmed on
  demand (scale to zero) until there are users (open question 1).
- **Weights** download from Hugging Face at the pinned revision onto the pod's container
  disk, as the bench does today. A network volume is the later option for faster warm
  starts and is measured before it is used.
- **Five milestones**, the first three free: local platform on the mock backend, the pool
  manager against a fake RunPod, the AWS control plane, a paid serving smoke (cap $20),
  and go-live in the chosen mode (cap $35 for the acceptance window).

## 2. What changes from the brief

| Brief (§5.3, §8.2, §8.4) | This plan | Why |
|---|---|---|
| EKS cluster with CPU and GPU node pools, shared Helm chart | Control plane on one EC2 instance with Docker Compose (Terraform); GPU replicas are RunPod pods driven by the pool manager | No GPU nodes on AWS today (P quota 0, one g6e.xlarge of G quota). An EKS control plane alone is $73 a month and adds nothing when GPUs are elsewhere. The Helm chart comes with AWS as the second cloud (Phase 2). |
| One Deployment per model per cloud | One pool per model on RunPod, `min_replicas` pods each | Same idea, run by the pool manager instead of Kubernetes |
| KEDA / HPA autoscaling on queue metrics | Fixed replica count per model; scale-out alert when the queue grows; optional warm-on-demand (scale to zero) | Cold starts of minutes, scarce 2x H100 stock, no traffic yet ([section 5](#5-replica-lifecycle)) |
| GPU pods on private networks, network policies + mTLS | Public pod IP, but only an mTLS proxy port and key-only SSH are exposed; engine on loopback | RunPod has no private link to AWS ([section 4](#4-gateway-to-pods-the-secure-path)) |
| Retry once on the other cloud if no tokens were streamed | Retry once on another replica of the same model, same rule | One cloud in the MVP. The 8B can get an AWS g6e.xlarge fallback pool in Phase 2. |
| Weights on a persistent volume / object storage | Hugging Face download to container disk per pod; network volume later | Measured download times are small next to the engine start ([section 6](#6-weights)) |
| Postgres on RDS, Redis | Same: RDS PostgreSQL; Redis on the control-plane instance (AOF on) | ElastiCache would double the Redis cost for a single small node |
| Secrets in AWS Secrets Manager | Same for the control plane; pods hold only what RunPod injects and a per-pod TLS key | [Section 10](#10-security-deltas-against-12) |
| Deploy to AWS: both models live, Stripe test mode | Control plane on AWS, both models live on RunPod, Stripe test mode | The 2026-10-11 decision |

## 3. Where things run

```
clients (SDK, curl, browser)
        |  HTTPS (Caddy, Let's Encrypt)
        v
AWS us-east-1, one EC2 instance (t4g.medium) ----------------------------+
|  Caddy -> web (Next.js) / gateway (FastAPI) / Grafana (admin only)      |
|  billing worker, pool manager, Redis (AOF), Prometheus                  |
+--------------------------------------------------------------------------+
        |  TLS (rds.force_ssl)              |  mTLS over the internet, HTTP/2,
        v                                   |  pooled connections
RDS PostgreSQL (db.t4g.micro)               v
                               RunPod Secure Cloud pods (US datacenters)
                               +---------------------------------------+
                               | pod proxy :8443 (mTLS, IP allowlist)   |
                               |   -> vLLM on 127.0.0.1:8000            |
                               | sshd (key-only; bootstrap, lease)      |
                               | lease watchdog (self-terminate)        |
                               +---------------------------------------+
```

| Component | Where | Monthly cost (us-east-1 list prices, check at apply time) |
|---|---|---|
| Gateway, billing worker, web app, pool manager, Redis, Prometheus, Grafana, Caddy | One EC2 `t4g.medium` (2 vCPU, 4 GiB, Graviton), Docker Compose, 30 GB gp3 | $24.53 instance + $2.40 disk + $3.65 public IPv4 |
| Postgres | RDS PostgreSQL `db.t4g.micro`, single-AZ, 20 GB gp3, 7-day automated backups | $11.68 + $2.30 storage |
| Secrets (RunPod key, CA key, Stripe keys, Auth.js secret, SSH key) | AWS Secrets Manager, ~6 secrets | $2.40 |
| Logs, alarms, DNS zone, S3 (leases, backups) | CloudWatch, Route 53, S3 | ~$3 |
| **Total** | | **about $50** |

Choices and their trade-offs:

- **One instance, not a cluster.** It is a single point of failure: if it dies, the API
  is down until Terraform and Compose bring a new one up (about 15 minutes, plus the
  RDS data survives). Accepted for an MVP with no customers. Redis keeps an append-only
  file on the instance's disk, so a reboot loses at most a second of usage events; the
  ledger balance is reconciled from Postgres every minute anyway.
- **RDS rather than Postgres on the instance**: the ledger is money, and point-in-time
  recovery is worth $14 a month.
- **Upgrade path**: if memory use passes 75%, move to `t4g.large` (+$24.50). Phase 2
  moves the control plane onto the Helm chart when AWS GPUs arrive.
- **Rejected**: the control plane on RunPod CPU pods (no managed Postgres, no secret
  store, the shared account); Vercel or Fly.io for the web app (new accounts, and the
  database would need a public endpoint).
- **Data transfer is a real cost**: AWS bills $0.09 per GB out to the internet. A
  streamed chat chunk is about 200 bytes per output token, so streaming costs about
  $0.018 per 1M output tokens, 4.9% of the 8B output price and 2.3% of the 70B's. Phase
  0's cost model leaves this out; the Phase 1 cost per 1M served includes it. RunPod's
  pricing lists no data-transfer fees (to verify in M4).

## 4. Gateway to pods: the secure path

RunPod offers two ways in to a pod: its HTTP proxy (`https://<pod id>-<port>.proxy.runpod.net`)
and exposed TCP ports on the pod's public IP (what the bench uses for SSH). Neither is
private.

| Option | Auth | TLS | Problems | Verdict |
|---|---|---|---|---|
| RunPod HTTP proxy + bearer token | Shared token only; anyone who learns the URL can try it | Ends at RunPod's edge, plain HTTP inside RunPod | Public URL; RunPod documents a ~100 s connection timeout on the proxy (to verify), which long non-streamed 70B requests can exceed | Rejected |
| TCP port straight to vLLM with `--api-key` | Bearer token | None | Prompts in clear text over the internet; the key on vLLM's command line is world-readable in the pod (`/proc/<pid>/cmdline`) and changes the benchmarked launch | Rejected |
| **TCP port to an in-pod mTLS proxy** | Client certificate from Loom's CA, plus source-IP allowlist | Ours, end to end, TLS 1.3 | One more process in the pod; Loom runs a small CA | **Chosen** |
| RunPod private networking between pods, gateway on RunPod | Network isolation | Optional | Does not reach AWS; would move Postgres and secrets onto RunPod | Rejected |

How it works:

1. **CA.** A private CA whose key lives in Secrets Manager, readable only by the
   control-plane instance role. The pool manager issues each new pod a server
   certificate (subject alternative name = the pod id, 7-day validity) and the gateway a
   client certificate (7-day validity, rotated daily).
2. **Pod start.** The pod is created with port `8443/tcp` and `22/tcp` exposed. Its start
   command (based on the bench's `pod_start.sh`) arms the lease watchdog, starts sshd
   (key-only, root only, no forwarding) and fetches the proxy binary, a pinned
   single-binary reverse proxy (Caddy) checked against its sha256 as the bench checks its
   client Python today.
3. **Bootstrap over SSH.** The pool manager connects with its own key (from Secrets
   Manager; the account key is not authorized on serving pods) and, on SSH stdin as the
   bench sends presigned URLs, delivers the server key and certificate, the CA
   certificate and the gateway's egress IP. They land in a directory only the proxy user
   (a separate uid) can read; never in the pod environment, which anyone with the
   account API key can read, and never on a command line.
4. **Engine.** The same `start_engine.sh` path as the bench: weights at the pinned
   revision, engine offline (`HF_HUB_OFFLINE=1`), bound to `127.0.0.1:8000`, launched
   from the registry row with the benchmarked flags, so the served config hash is the
   Lab's (plus the one reporting flag in [section 9](#9-observability)).
5. **Proxy.** Listens on 8443, requires a client certificate signed by Loom's CA,
   rejects any source address other than the gateway's Elastic IP, forwards only `POST
   /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`, `GET /health` and
   `GET /metrics`, flushes every streamed chunk immediately, logs no request bodies.
6. **Gateway.** Connects to `<public ip>:<mapped port>` (read from the RunPod API, as
   `runpod_api.public_ssh` reads the SSH mapping), verifies the server certificate
   against the CA **and** that its name is the expected pod id, and keeps a pooled
   HTTP/2 connection per pod, so TLS handshakes stay off the request path.

The gateway also adds a random per-pod request token header that the proxy checks.
It costs nothing and keeps a mis-issued client certificate from being enough on its own.

What stays exposed: a TLS handshake on 8443 and an SSH banner on 22 at each pod's public
IP. Both refuse anyone without our keys; a flood on either can still degrade that pod
(the gateway then routes around it, [section 8](#8-failure-handling)).

**Latency.** A pod in a different region adds its network round trip to time to first
token. The bench's pods landed in Sweden, Iceland and Missouri. Serving pods are pinned
to US datacenters (`dataCenterIds`), which narrows stock (open question 7). The brief's
"p50 gateway overhead < 10 ms" is measured as time spent inside the gateway; the
gateway-to-pod round trip is reported next to it, not hidden in it. Neither has been
measured yet (M4 does).

## 5. Replica lifecycle

### States

`creating` (RunPod create, waits out "no capacity" with the bench's backoff) ->
`bootstrapping` (SSH up, certificates in, weights, engine start) -> `ready` (health and a
one-token probe passed: in routing) -> `draining` (out of routing, in-flight streams
finish, at most 10 minutes) -> `terminated` (always terminate, never stop: a stopped pod
keeps billing for its disk).

The pool manager reconciles every 15 seconds: for each model, desired replicas (from
`config/serving.yaml` and the availability mode) against pods it owns. Pods are cattle:
a broken pod is replaced, not repaired.

### Health checks

| Check | How often | Unhealthy when | Action |
|---|---|---|---|
| Pod state (RunPod API, `pod_loss()` from the bench) | 30 s | `EXITED`, `TERMINATED` or gone | Out of routing at once; replace |
| Proxy + engine `GET /health` | 10 s | 3 failures in a row (30 s) | Out of routing; replace if not back within 2 min |
| Engine progress (`/metrics`) | 10 s | requests running but no new tokens for 60 s | Out of routing; replace |
| Live traffic (gateway) | per request | 3 connect errors or 5xx in a row | Circuit open for 30 s, then one probe |
| Bootstrap | once | no first token within the stall limits the bench uses (`engine_stall_s` 600 s, ready timeout) | Terminate; create a new pod |

### Restarts and runaway spend

- A container restart (RunPod host maintenance, a crashed PID 1) loses the engine. The
  manager sees the health check fail and replaces the pod.
- **Lease watchdog instead of a fixed TTL.** Bench pods die at an absolute TTL; serving
  pods must run indefinitely, but must not bill forever if the control plane dies. Each
  pod starts with a lease 60 minutes ahead (`LOOM_LEASE_EPOCH` in its env); the manager
  renews it over SSH every 5 minutes. A pod whose lease has passed terminates itself
  with the pod-scoped key RunPod injects (the bench's watchdog code). So a dead control
  plane costs at most one hour of GPU time per pod: $1.10 for the 8B, $8.02 for the 70B.
- **Reaper.** The scheduled RunPod reaper Lambda gets a serving rule: terminate a pod
  named `loom-serve-<model>-<created epoch>-<6 hex>` with `LOOM_MANAGED=true` when the
  manager has not written its lease record (`s3://<bucket>/serving/leases/<pod id>`) for
  2 hours. It covers a pod stuck before its container starts, which has no watchdog.
  The existing fail-closed rules (exact names, at most 10 per run, nothing deleted on a
  bad listing) stay.
- **Spend guard.** `config/serving.yaml` sets a monthly serving cap and a hard maximum of
  pods per model. The manager refuses to create a pod that would cross the cap, alerts
  at 50%, 80% and 100%, and refuses any pod whose API price is above `bench/prices.yaml`
  x 1.25 (the bench's `max_price_ratio`). It never terminates a healthy serving pod
  because of the cap; that is an admin action.

### Autoscaling: not in the MVP

| | Why not now | What instead |
|---|---|---|
| Scale out on queue depth | A new 8B replica takes 3 to 7 minutes and a 70B 5 to 13.5; 2x H100 SXM stock was "Low" whenever we looked; there is no traffic to scale for | Alert when `vllm:num_requests_waiting` stays above 0 for 5 minutes or a replica runs at its measured goodput (8B: 2.2 req/s on chat; 70B FP8: 2 req/s on 1k/1k); the admin raises the replica count |
| Scale to zero | Possible, as the warm-on-demand mode below | Per-model idle timeout, opt-in ([section 11](#11-what-it-costs)) |

Real autoscaling (KEDA on the Helm chart, warm pools) is Phase 3's.

### Cold starts we measured on RunPod

From `bench_cold_starts` in the results database (time from the create call to the first
token):

| Model and hardware | Measured | Parts |
|---|---|---|
| Qwen3-8B BF16, 1x L40S, engine image already on the host | 2.7 to 3.7 min (162-221 s, median 179 s, 9 pods) | pod + SSH 14-38 s, weights 20-110 s, engine start 80-150 s |
| Same, image not on the host | 4.1 to 7.0 min (245-421 s, median 353 s, 14 pods) | adds a 2 to 3.5 min image pull (vLLM image 8.7 GB compressed) |
| Llama 3.3 70B FP8, 2x H100 SXM, TP=2 | Not measured directly; estimated **5.1 to 7.5 min**, up to **13.5 min** on a slow host without the image | pod + SSH 15-36 s (images were cached on both H100 hosts), FP8 weights (72.7 GB) 156-277 s measured as warm restarts (9f0853d7, 9e5f7866), engine start 133-139 s |
| Llama 3.3 70B BF16 (reference, not served), same pod | 4.9 min (9e5f7866) and 16.5 min (9f0853d7) | weights 141 GB in 135 s and 826 s: download speed varies 6x between hosts |

These are what a customer would wait for the first request after scale-to-zero, and what
a replacement after pod loss takes.

## 6. Weights

| Option | Time per start (measured) | Running cost | Catch |
|---|---|---|---|
| **Hugging Face download to container disk** (bench today) | 8B (16.4 GB): 20-110 s. 70B FP8 (72.7 GB): 156-277 s | Container disk $0.10 per GB-month while the pod runs (120 GB: $0.016/h). GPU time while downloading: up to $0.04 per 8B start, $0.35-0.62 per 70B start | Hugging Face is a dependency for every new pod; an outage blocks replacements (running pods are unaffected) |
| RunPod network volume | Not measured | $0.07 per GB-month, billed always: 100 GB for both models is about $7 a month | Every pod is tied to the volume's datacenter, which then must stock both 1x L40S and 2x H100 SXM; read speed from it into vLLM unmeasured |
| Our own S3 mirror | Not measured | S3 egress $0.09/GB: $1.48 per 8B start, $6.54 per 70B start | The most expensive; Hugging Face is free |
| Weights baked into the image | The 8.7 GB vLLM image already takes 2-3.5 min to pull when uncached | Registry storage | A ~80 GB 70B image would pull slower than downloading the weights |

**Recommendation:** Hugging Face to container disk for the MVP. It is the tested path and
costs nothing when no pod runs. Both served checkpoints are ungated (`Qwen/Qwen3-8B` and
`RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`), so **serving pods need no Hugging Face
token at all**. Revisit a network volume if warm-on-demand becomes the default mode:
first a small measured run (create a volume in a datacenter with both stocks, time a
cold start from it, delete it), as the runbook's volume rule requires.

## 7. Routing

- **Pool per model**, from `config/models.yaml`. Only `ready` replicas are candidates.
- **Least in-flight**: the gateway counts its own open requests per replica (exact with
  one gateway instance) and picks the fewest, breaking ties on `vllm:num_requests_waiting`
  scraped every 5 s. A replica at its measured goodput with requests waiting is skipped
  while another has room.
- **No replica ready**: 503 (see below). The gateway never queues requests itself.
- Prefix-cache-aware routing (same prefix, same replica) is Phase 3.

With one replica per model, routing is trivial; it is built for N because replacing a pod
briefly runs two.

## 8. Failure handling

The request lifecycle stays the brief's (§5.2): authenticate, reserve credits, route,
stream, settle, usage event. On RunPod:

| What fails | Client sees | Retry | Billing |
|---|---|---|---|
| No ready replica (pool empty, warming, all circuits open) | 503 `model_unavailable`, OpenAI-shaped, `Retry-After` (warming: the model's cold-start estimate) | No | Reservation released; nothing billed |
| Connect error, TLS failure, timeout or 5xx **before the first byte** is sent to the client | Nothing, if the retry works | **Once**, on a different replica of the same model; with no other replica, 503 | Billed once, for the request that succeeded |
| Pod lost **mid-stream** | The stream ends with an OpenAI-shaped error chunk (`server_error`, `upstream_lost`) and `[DONE]` | Never: tokens were already sent | Prompt tokens plus the completion tokens actually streamed |
| Pod lost during a non-streamed request | 502 `upstream_lost` | Once if nothing was returned and it is still within the timeout budget | Nothing (the client got nothing) |
| Client disconnects | (gone) | No | The gateway closes the upstream connection, vLLM aborts the request; billed for tokens generated so far |
| 4xx from the engine (bad schema, context too long) | The engine's error, passed through | Never | Nothing |
| RunPod API down | Nothing while pods are healthy | | Replacements wait; alert |
| Hugging Face down | Nothing while pods are healthy | | New pods cannot start; alert |
| RunPod balance runs out | RunPod stops the pods: 503 for everything | | Prevented by open question 3 |

**Exact token counts on aborts.** The gateway asks vLLM for usage on every chunk
(`stream_options: {include_usage: true, continuous_usage_stats: true}`) and strips it
from what the client receives unless the client asked for usage. Then the token count for
a stream cut at any point is the engine's own, not an estimate, which is how "usage in
the DB matches engine-reported tokens exactly" holds for aborted streams too. M4 checks
that vLLM 0.30.0 honours it.

## 9. Observability

| Metric | Source | Notes |
|---|---|---|
| Requests/s, errors by code, gateway overhead p50/p99, gateway-to-pod RTT | Gateway (Prometheus client) | Overhead and RTT reported separately |
| TTFT, TPOT, end-to-end latency per model and replica | Gateway timings, and vLLM's `vllm:time_to_first_token_seconds` / `vllm:e2e_request_latency_seconds` histograms | Compared with the Lab's measured percentiles at the same load |
| Running and waiting requests, KV-cache use, preemptions, prefix-cache hit rate, prompt and generation token counters | vLLM `/metrics` through the proxy, scraped every 15 s | The same metrics the Lab records server-side |
| GPU utilisation and memory | RunPod API pod runtime (per-GPU utilisation, as the console shows) | No exporter in the pod; to verify the fields in M4 |
| Replica states, cold-start stages, pod hourly price | Pool manager | Stages named as the bench's (`pod_created` ... `first_token`) |
| **Cost per 1M tokens served** per model | (pod-hours x the pod's API `costPerHr` + streaming egress) / tokens billed in `usage_events`, per hour and per day | Next to the Lab's prediction at SLO (8B: $0.1032 per 1M blended on chat; 70B FP8: $0.546 at 1k/1k). Their ratio is the **effective utilisation**, the number that says whether the prices hold |
| Token reconciliation | Hourly: sum of `usage_events` tokens against the vLLM counters' increase | Must match exactly; any gap alerts |
| RunPod spend | Pool manager accrual against the RunPod balance | Only exact with a RunPod account Loom does not share (open question 2) |

Logs are structured JSON, never prompts or completions. vLLM's own log stays on the pod
and is lost with it; M4 checks it holds no prompt text with the served flags. Grafana is
reachable only through Caddy with admin auth.

**Cached tokens.** The API must return `cached_tokens` in `usage` (§8.1). vLLM reports
them only with `--enable-prompt-tokens-details`, which changes the launch and so the
config hash, but not the outputs. The plan adds the flag and records the served config's
own hash; M4's load test confirms the latency matches the Lab's row. Billing is not
affected (cached input is priced as input).

## 10. Security deltas against §12

| §12 requirement | On RunPod | How it is closed | What remains |
|---|---|---|---|
| GPU pods on private networks, only the gateway reaches them, mTLS, no public inference endpoint | Pods have public IPs; no private link to AWS | Engine on loopback; only the mTLS proxy and key-only SSH are exposed; client certificate + IP allowlist + per-pod token; no RunPod HTTP proxy port | Ports answer a TLS handshake or an SSH banner to the internet |
| TLS everywhere | | Client to Caddy (Let's Encrypt), gateway to pods (mTLS, server identity pinned to the pod id), gateway to RDS (`rds.force_ssl`); Redis on loopback | SSH host keys are trusted on first use, as in the bench |
| Secrets only in Secrets Manager; least-privilege IAM | RunPod injects a pod-scoped `RUNPOD_API_KEY` into every pod; RunPod account keys cannot be limited to some pods; the account is shared with another project | Serving pods hold no Hugging Face token (ungated checkpoints) and no AWS credentials; the TLS key arrives over SSH into a file only the proxy user reads; the manager's RunPod key and the CA key in Secrets Manager, readable only by the instance role; a separate RunPod account for production (open question 2) | The pod-scoped key (the watchdog needs it) |
| Pin revisions and image digests; scan images; no `trust_remote_code` | | Already enforced by the registry; the proxy binary is pinned by sha256; add a scan of the pinned vLLM digest (Trivy or Grype) to CI | Upstream image contents |
| No prompt/completion logging | Engine and proxy logs on the pod | Proxy logs no bodies; M4 checks the engine log | RunPod's hosts see prompts in memory: RunPod is a subprocessor (open question 9) |
| Engine isolation (bench: engine runs as root, accepted for Phase 0) | | Run vLLM as a non-root user on serving pods, so an engine compromise cannot read the TLS key or the pod key (the PLAN.md follow-up, required before public signups) | |
| Rate limits, caps, atomic reservation, Stripe webhooks, org-scoped queries, cross-tenant tests, RLS, abuse controls, audit log | Unchanged: all in the gateway, billing worker and Postgres | As the brief says | |

## 11. What it costs

### Always on (24/7)

Measured hourly rates: RunPod's API `costPerHr` plus container disk, as accrued in the
real runs (1x L40S in 7a8237d0, 80 GB disk; 2x H100 SXM in 9f0853d7, 260 GB disk).

| | Per hour | Per day | Per month (730 h) |
|---|---|---|---|
| Qwen3-8B, 1x L40S | $1.10 | $26.42 | $803.70 |
| Llama 3.3 70B FP8, 2x H100 SXM | $8.02 | $192.37 | $5,851.40 |
| **Both models** | **$9.12** | **$218.80** | **$6,655.10** |
| Control plane on AWS | ~$0.07 | ~$1.65 | ~$50 |

The 70B pod only needs about 120 GB of disk for the FP8 checkpoint (to $5,837 a month):
the disk is not where the money is.

**When it pays for itself.** Prices are cost-plus 20% on the cost at SLO assuming the
replica is busy every second. At those prices:

| | Capacity per replica at SLO | Revenue when fully busy | Breaks even at | Tokens a month to break even |
|---|---|---|---|---|
| 8B (chat mix, ~74% input) | ~10.7M tokens/h (2.2 req/s) | ~$1.63/h | ~68% busy | ~5.3 billion |
| 70B FP8 (1k/1k mix) | ~14.7M tokens/h (2 req/s) | ~$10.04/h | ~80% busy | ~8.6 billion |

With no customers, an always-on replica is a pure cost.

### The three availability modes

| Mode | Base cost a month | First request after idle | Fits |
|---|---|---|---|
| **A. Warm on demand** (scale to zero, both models): a request to a cold model returns 503 with `Retry-After` and starts a pod; idle for 30 min, the pod is terminated; the admin can pre-warm | ~$0 + per-use | 8B 3-7 min, 70B 5-13.5 min | Private preview, demos, the M5 acceptance |
| **B. 8B always on, 70B on demand** | $804 | 8B instant; 70B 5-13.5 min | First invited users |
| **C. Both always on** | $6,655 | Instant | Paying traffic near the break-even rates above |
| (B or C on a schedule, e.g. 12 h on weekdays, 264 h) | 8B $291, 70B $2,116 | Instant inside the window | A known user base in one time zone |

**What a warm-up costs** in mode A: the cold start plus the idle timeout, billed at the
pod rate.

| | Cold start only | Each wake with a 30-min idle timeout |
|---|---|---|
| 8B | $0.05-0.13 | ~$0.60-0.68 |
| 70B FP8 | $0.68-1.80 | ~$4.70-5.80 |

Scale-to-zero is cheap for the 8B and costs about $5 per wake for the 70B. A stopped pod
kept as a warm spare is not an option: it keeps billing for its disk and RunPod can hand
its GPU to someone else in the meantime.

### Paid steps in the milestones

| Step | Estimate | Proposed cap |
|---|---|---|
| M4 serving smoke: 8B pod ~1.5 h, then 70B FP8 pod ~1.5 h | $1.65 + $12.02 = $13.67 | $20 |
| M5 acceptance window: both models up ~3 h | $27.35 | $35 |
| After M5 | the chosen mode, plus ~$50 a month for AWS | the serving cap in `config/serving.yaml` |

## 12. Reused from the bench RunPod provider

| Bench piece | Reuse | Change |
|---|---|---|
| `runpod_api.RunpodApi` (REST + GraphQL, key redaction, no-capacity detection, terminate) | As is | |
| `RunpodProvider.pod_body`, naming, `_create` cleanup after an ambiguous create, capacity backoff, the `max_price_ratio` refusal | Pod creation for replicas | Prefix `loom-serve`; ports `22/tcp`, `8443/tcp`; no HF secret; `dataCenterIds` from `config/serving.yaml` |
| `pod_start.sh` (watchdog, sshd, `sleep infinity`) | Start command | Lease file instead of an absolute TTL; fetch and verify the proxy binary |
| `start_engine.sh` (pinned weights, offline engine on loopback, stall detection, isolation checks, GPU topology, NCCL env) | Engine start | No job user; the isolation check covers the proxy user; engine as non-root |
| `runpod_ssh` (`OpenSshExec`, key generation, detached scripts over stdin) | Bootstrap, certificate delivery, lease renewal | A long-lived manager key from Secrets Manager; `authorize_account_key` off |
| `pod_loss()` | Health loop | |
| `engines.py` argv rendering from the registry | Same served flags as the Lab, so the served config is traceable to its benchmark row | Adds `--enable-prompt-tokens-details` |
| `runpod_reaper_lambda.py` | Second backstop | A serving rule keyed on the S3 lease record |
| Mock backend, SSE client (`client/openai_stream.py`), load generator | M1 backend; gateway load test | |
| Test fakes (`bench/tests/runpod/fakes.py`: `FakeRunpod`, `FakePodExec`) | M2's offline tests of the manager | |
| Not reused | Presigned-URL jobs, the job user, eval jobs, the experiment budget guard (replaced by the serving spend guard) | |

`config/models.yaml` stays the source of truth for what is served and at what price.
Deployment knobs (availability mode, replica counts, idle timeouts, datacenters, the
monthly cap) go into a new `config/serving.yaml`, because the registry's `scaling` field
is part of every benchmark config hash and changing it there would move them.

## 13. Milestones and acceptance

Each milestone ends with a stop for the user's go-ahead before the next paid one.

| # | What | Paid? | Acceptance (§8.4 item in brackets) |
|---|---|---|---|
| M1 | Platform locally on the mock: gateway, billing worker, web app, Postgres, Redis, mock backend behind a local mTLS pod proxy, all in `docker compose up` | No | Sign up, free $5, create key, chat in the playground, call with the OpenAI Python SDK, see usage and balance drop [deliverable 1]; mock backend [deliverable 2]; unit tests for pricing, ledger, rate limiter, key hashing, reserve/settle; integration test request to usage to ledger including an aborted stream; cross-tenant tests; load test reporting TTFT and tokens/s [deliverable 4]; SDK unchanged, revoked key fails within 5 s, documented worst-case overshoot under concurrent load, gateway p50 overhead < 10 ms [acceptance] |
| M2 | Pool manager offline: lifecycle, health checks, lease watchdog, reaper serving rule, spend guard, routing, retry and 503 rules | No | Against `FakeRunpod` and `FakePodExec`: create, bootstrap, ready, drain, replace; lease expiry terminates; reaper ignores bench pods and the other project's; pod killed mid-stream bills exactly the streamed tokens; retry happens only before the first byte; no replica gives 503 with `Retry-After` |
| M3 | AWS control plane: Terraform (`infra/aws/platform`: VPC, EC2, RDS, Secrets Manager, IAM, alarms), Compose deploy, domain and TLS, Stripe test webhooks; backend still the mock | ~$50/month from apply | M1's flow on the public URL; Stripe test checkout posts a ledger credit idempotently; revoked key within 5 s and p50 overhead < 10 ms re-measured on the deployed box [acceptance] |
| M4 | Serving smoke on RunPod: one 8B pod, then one 70B FP8 pod, through the real gateway | Cap $20 | mTLS path works and refuses a request without our client certificate and from another IP; usage equals the engine's tokens exactly, including a client abort and a pod terminated mid-stream [acceptance]; cold start, gateway-to-pod RTT and GPU metrics recorded; load test through the gateway at the Lab's goodput reports TTFT and tokens/s within the Lab's intervals plus the RTT [deliverable 4]; engine log holds no prompt text; teardown verified |
| M5 | Go-live in the chosen mode (invite-only preview unless decided otherwise) | Cap $35 for the window, then the mode's cost | Both models live, Stripe test mode [deliverable 3]; dashboards show cost per 1M served next to the Lab prediction; the M4 checks pass again on both models; a written acceptance note like `phase0-acceptance.md` |

## 14. Risks

- **2x H100 SXM stock** has been "Low" each time we checked. If the 70B pod is lost and
  none is available, the 70B returns 503 until stock returns; there is no benchmarked
  fallback hardware (4x L40S FP8 is on hold). The status shows on `/v1/models` and the
  Models page.
- **Cold starts vary 2-6x between hosts** (image cache, download speed), so `Retry-After`
  values are estimates.
- **One control-plane instance** is a single point of failure (accepted for the MVP).
- **Prices assume full utilisation.** Real cost per 1M served will be higher until
  traffic arrives; the dashboard makes that visible rather than hiding it.

## 15. Open questions for the user

Each needs an account, money or a hard-to-reverse choice.

1. **Availability mode** (money). A: both on demand, ~$0 base; B: 8B always on, $804 a
   month; C: both always on, $6,655 a month. **Recommendation: A** through M5 and any
   private preview; switch to B when the first outside user is invited; C only when
   traffic approaches the break-even rates. Set the monthly serving cap with it
   (suggested: $100 in mode A).
2. **A separate RunPod account for production serving** (account). The current account
   is shared with another project, and RunPod API keys cannot be limited to some pods:
   Loom's manager key could terminate the other project's pods and the reaper has to
   rely on naming rules alone; spend cannot be reconciled exactly. **Recommendation:
   yes**, a new RunPod account (or team) used only for serving; the bench keeps the
   current one. The user creates it and puts its key in Secrets Manager.
3. **RunPod billing for production** (money). The bench account is prepaid with
   auto-pay off; when a balance runs out RunPod stops the pods, which for serving means
   an outage. **Recommendation:** prepaid with at least two months of the chosen mode's
   cost, auto-pay on with a monthly limit at the serving cap, and a low-balance alarm at
   two weeks of runway.
4. **AWS control plane** (money, infra). About $50 a month: `t4g.medium`, RDS
   `db.t4g.micro`, in `us-east-1`, in the existing AWS account under its own VPC, tags
   and Terraform state. **Recommendation: approve as described** at M3, not before.
5. **Domain** (account, money). Needed for TLS, OAuth redirects and Stripe. The brand is
   Loom, domain TBD. **Recommendation:** the user registers it (Route 53 is simplest
   with this setup) before M3.
6. **Google OAuth client and email sending** (accounts). Auth.js needs a Google OAuth
   client (the user creates it in Google Cloud) and an email sender for verification
   and low-balance mail. **Recommendation:** Amazon SES in the same AWS account; the user
   requests production access, which AWS reviews (a few days).
7. **US datacenters only for serving pods** (latency, data location). **Recommendation:
   yes**: lower time to first token from `us-east-1` and one place to name in the
   privacy policy, at the cost of fewer GPUs to choose from. Revisit if stock blocks M4.
8. **Who can sign up in Phase 1** (money, abuse). Open signups with $5 free credit invite
   abuse before the abuse controls are proven. **Recommendation:** invite-only from the
   waitlist through M5; open signups after the abuse controls have run for a while.
9. **RunPod as a subprocessor** (legal). Customer prompts are processed on RunPod
   Secure Cloud hosts. **Recommendation:** name RunPod and the US location in the
   preview terms and privacy policy, and get a legal review of RunPod's terms and data
   processing agreement before public signups.
10. **Paid test budget** (money). **Recommendation:** approve M4 at a $20 cap and the M5
    window at $35, each only after the previous milestone passes.
