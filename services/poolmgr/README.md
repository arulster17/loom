# loom-poolmgr: the pool manager

One process next to the gateway that creates, bootstraps, health-checks, drains and
terminates serving pods, and writes `loom:pool:<model>` exactly as the gateway reads it
(docs/phase1-contracts.md §10). Spec: [docs/phase1-m2-spec.md](../../docs/phase1-m2-spec.md).
M2 built and proved it offline (fakes, and the local backend in compose); M4 runs it on
RunPod for the first time.

What it is configured by:

- `config/models.yaml`: what is served (GPU type and count, engine launch, price);
- `config/serving.yaml` (`loom_platform.serving.load_serving`): how (provider, spend cap,
  placement, network, lifecycle, health and capacity timings, per model replicas,
  `idle_stop_s`, `cold_start_estimate_s`, `engine_env`). The committed file is the compose
  default, `provider: local`; production's (`provider: runpod`) is deployed in M3.

## Run

```bash
docker compose up -d --build --wait               # the whole stack: poolmgr on the local backend
docker compose exec poolmgr loom-poolmgr status   # pools, pods, month-to-date spend
docker compose exec poolmgr loom-poolmgr warm qwen3-8b
```

Models start cold: the first request gets 503 `model_unavailable` with `Retry-After` and
wakes the model; the manager starts a pod and publishes `warming`, then `ready` (about 30
to 40 s locally: `LOOM_POOLMGR_LOCAL_COLD_START_S`, 20 s, plus the bootstrap). A model
without demand for its `idle_stop_s` (compose: `LOOM_POOLMGR_IDLE_STOP_S`, 600 s) is
drained back to cold. Demand is a wake key, a request in flight, or a change in
`loom:pool:<model>:last_request`, which the gateway stamps whenever it settles a request
for the model (not the usage stream: billing trims it too soon).

Outside compose: `uv run loom-poolmgr run` with `LOOM_ENV=dev`, `LOOM_DATABASE_URL`
(migrated with `uv run loom-migrate`) and `LOOM_REDIS_URL`. In dev without a CA, SSH key
or lease bucket it generates a CA and key in its state directory and keeps lease records
in memory.

**Single writer.** `run` takes the Postgres advisory lock `hashtext('loom-poolmgr')` and
exits 1 if another manager holds it. Never run poolsim (compose profile `poolsim`) against
the same Redis: both write the pool hashes.

### The local backend (`provider: local`, compose, dev only)

Each "pod" is a subprocess pair inside the poolmgr container: the bench mock engine
(`python -m loom_bench.mock`) on a loopback port and a real `loom-podproxy` in front of
it, with the server certificate, token and allowlist the manager delivers (written as
files instead of over SSH). The create, health, drain, idle and pool-hash code is the
same as for RunPod; only `LocalServing` (the provider) and `LocalPodExec` (instead of
SSH) differ. In compose the proxies listen on all interfaces and are advertised at
`https://poolmgr:<port>` (`LOOM_POOLMGR_LOCAL_HOST`), their certificates signed by the
dev CA in `.dev/certs` (the one the gateway trusts), and admit only the compose subnet
(`LOOM_POOLMGR_ALLOW_CIDRS`). The pods die with the container; a restarted manager marks
their rows terminated and starts over.

## CLI

Admin commands act on the running manager's Postgres and Redis; `drain`, `stop`, `reset`,
`rekey` and `local-capacity` are queued on `loom:poolmgr:admin` and applied on its next tick.

| Command | Does |
|---|---|
| `loom-poolmgr run` | The service |
| `loom-poolmgr status` | Per model: pool state, reason, pods with state, age, price, lease; month-to-date spend against the cap |
| `loom-poolmgr warm <model>` | `request_wake` (pre-warm), as the gateway's 503 does |
| `loom-poolmgr drain <pod id>` | Drain and terminate one pod |
| `loom-poolmgr stop <model>` | Drain and terminate all of a model's pods; it stays `cold` (wakes still start it) |
| `loom-poolmgr reset <model>` | Clear `crash_loop` / `spend_cap` / `price_too_high` holds (and the crash-loop count) |
| `loom-poolmgr rekey [--model M]` | Re-issue every ready pod's server certificate now (all models, or one): the renewal path, run at once. Pods keep their tokens and stay in routing |
| `loom-poolmgr new-ca --out DIR` | A new pod CA, any environment: `ca.crt`, `ca.key` and the manager's client certificate from it (`loom-poolmgr.crt`/`.key`, valid as long as the CA), keys 0600. Refuses to overwrite a CA. See [docs/runbook-pod-ca.md](../../docs/runbook-pod-ca.md) |
| `loom-poolmgr issue-ca --out DIR` | Dev only: a CA plus manager and gateway client certificates |
| `loom-poolmgr local-capacity off\|on` | Dev only, `provider: local` only: `off` makes every create fail for capacity (the acceptance's no-capacity switch) |

## Settings

`LOOM_*` (the platform's `PlatformSettings`, `LOOM_ENV`, `LOOM_DATABASE_URL`,
`LOOM_REDIS_URL`, `LOOM_MODELS_YAML`) plus:

| Variable | Default | What |
|---|---|---|
| `LOOM_POOLMGR_SERVING_YAML` | `LOOM_SERVING_YAML`, else `config/serving.yaml` | The serving config |
| `LOOM_POOLMGR_CA_CERT`, `LOOM_POOLMGR_CA_KEY` | none | The Loom pod CA: issues pod server certificates (SAN = pod id) |
| `LOOM_POOLMGR_CLIENT_CERT`, `LOOM_POOLMGR_CLIENT_KEY` | none | The manager's client certificate (signed by that CA), for health checks and probes |
| `LOOM_POOLMGR_SSH_KEY` | none | The private key serving pods authorize (`<key>.pub` next to it) |
| `LOOM_POOLMGR_LEASE_BUCKET` | none | S3 bucket of the lease records (`serving/leases/<pod id>`), read by the reaper's serving rule |
| `LOOM_POOLMGR_PROXY_BUNDLE` | none | The `loom-podproxy-bundle` tarball; its sha256 must match `config/serving-artifacts.lock` |
| `LOOM_POOLMGR_METRICS_PORT` | 9102 | Prometheus (0 = none); served once the lock is held and adoption is done |
| `LOOM_POOLMGR_STATE_DIR` | dev: a temporary directory | known_hosts per pod (must survive a restart), and in dev a generated SSH key and CA |
| `LOOM_POOLMGR_LOCAL_COLD_START_S` | 20 | `provider: local`: seconds before a local pod is `RUNNING` |
| `LOOM_POOLMGR_LOCAL_HOST` | `127.0.0.1` | `provider: local`: where the gateway reaches the local pods |
| `LOOM_POOLMGR_ALLOW_CIDRS` | empty | Dev and `provider: local` only: CIDRs added to the local pods' allowlist |
| `LOOM_POOLMGR_IDLE_STOP_S` | unset | Dev only: replaces every model's `idle_stop_s` |
| `LOOM_POOLMGR_MOCK_BACKEND` | unset | Outside dev: `true` permits `provider: local` (not `runpod`) and private (RFC 1918) `LOOM_POOLMGR_ALLOW_CIDRS`, and lifts the SSH key, lease bucket and proxy bundle requirements; the CA and client certificate stay required. Warns at start; exports `loom_poolmgr_mock_backend 1`. M4 removes it |
| `LOOM_POOLMGR_GATEWAY_CERT_DIR` | none | Where the manager writes the gateway's client certificate (`gateway-client.pem`: key and certificate, mode 0400, renamed into place): at start unless the file holds a current one (loads, issued by this CA, 6 days or more left), then every 24 h; 7-day validity |
| `RUNPOD_API_KEY` | none | `provider: runpod`: the serving account's key |

Outside `LOOM_ENV=dev` it refuses to start without the CA, the client certificate, the
SSH key, the lease bucket, the proxy bundle, the state directory and a non-empty
`gateway_egress_cidrs`; it refuses `provider: local|fake` and the dev-only variables
(`LOOM_POOLMGR_MOCK_BACKEND` lifts the provider, SSH key, lease bucket, bundle and
private-`ALLOW_CIDRS` rules, below).
With `provider: runpod` the proxy bundle and the lease bucket are required in dev too
(the placeholders are for `local` only, so no paid pod is bootstrapped with them). It
also refuses to start when the CA expires within `server_cert_renew_before_s`, and logs
an error (without re-keying) if the CA gets that close while it runs. Local pods have no
price: no price check and nothing accrues to the month's spend.

Metrics: `loom_poolmgr_pods{model,state}`, `loom_poolmgr_cold_start_seconds{model,stage}`,
`loom_poolmgr_creates_total{model,result}`, `loom_poolmgr_replacements_total{model,why}`,
`loom_poolmgr_lease_renewals_total{result}`, `loom_poolmgr_spend_month_micros`,
`loom_poolmgr_spend_cap_micros`, `loom_poolmgr_reconcile_seconds`,
`loom_poolmgr_orphans_terminated_total`, and from M3:

- `loom_poolmgr_ca_expiry_seconds`: seconds until the pod CA's `notAfter`, set at start and
  each tick (alarm P3 below 30 days; the refusal at `server_cert_renew_before_s` stays);
- `loom_poolmgr_pool_reason{model,reason}`: 1 for each model's current pool `reason`
  (`crash_loop`, `spend_cap`, `price_too_high`, `not_configured`, `no_capacity`), 0 for the
  others, rewritten each tick (alarm P2b reads `spend_cap`);
- `loom_poolmgr_mock_backend`: 1 with `LOOM_POOLMGR_MOCK_BACKEND`, else 0.

Every `creates_total{model,result}` and `pool_reason{model,reason}` series of the models the
manager serves exists at 0 from start, so the alarms never see a missing series.

## The mock backend outside dev (M3)

The M3 box has no GPUs: it runs `provider: local` (the bench mock behind `loom-podproxy`,
as in compose) with `LOOM_ENV` unset, so every production check stays on, and
`LOOM_POOLMGR_MOCK_BACKEND=true` to allow what dev-only rules would refuse. The flag is
refused with `provider: runpod`. Its serving file is `deploy/serving.m3.yaml`. The manager
still requires the CA, its client certificate and `LOOM_POOLMGR_STATE_DIR`.

## Certificates

- **Pod server certificates** (`server_cert_days`) are renewed near expiry over SSH and
  re-issued at once by `rekey`. `loom-poolmgr status` shows, under each ready pod, who issued
  the certificate its proxy presents and whether that is the manager's CA (it connects with
  the manager's client certificate, so run it where `LOOM_POOLMGR_CLIENT_CERT`/`_KEY` and
  `LOOM_POOLMGR_CA_CERT` are set and the pods are reachable).
- **The gateway's client certificate** (`clientAuth`, 7 days) is written to
  `LOOM_POOLMGR_GATEWAY_CERT_DIR/gateway-client.pem`: at start unless the file already holds
  a current one (key and certificate load together, signed by the CA in use, 6 days or more
  left), then every 24 h by the manager's clock. The file is written under a temporary name
  with mode 0400, owned by the directory's owner (so the gateway's uid can read it), and
  renamed into place. A failed write is logged and retried on the next tick; at start it
  stops the manager. The gateway reloads it, and `LOOM_POD_CA_CERT`, every
  `LOOM_GATEWAY_CERT_CHECK_S` (default 60 s).
- **A new CA** is a runbook: [docs/runbook-pod-ca.md](../../docs/runbook-pod-ca.md).

## Tests and acceptance

- `uv run pytest -q -n auto services/poolmgr/tests` (Linux: the gateway loop tests import
  the gateway, which needs `fcntl`; Postgres and Redis must be reachable). They run the
  manager against `FakeRunpod` plus `FakeServingExec` and a fake clock, with real
  `loom-podproxy` processes in front of the mock engine.
- `uv run loom-m2-acceptance --json /tmp/m2.json` against the compose stack: cold start
  and wake, an engine killed mid-stream and the pod replaced, no capacity, the spend cap,
  and an idle stop (about 20 minutes, most of it the 600 s idle wait).
  `uv run loom-m1-acceptance` drives the same stack with the manager in the loop.

## What M4 must check on a real pod

M2 never touched RunPod. The serve scripts were written against the bench's `pod_start.sh`
and `start_engine.sh` and the pinned image (`vllm/vllm-openai:v0.30.0` by digest in
`config/models.yaml`), but these can only be verified on a pod, in M4's smoke, before any
serving:

1. **Proxy venv install.** The image's `python3` can run `python3 -m venv --without-pip
   /opt/loom/proxy-venv` (the `venv` module and `ensurepip`-less venvs present) and
   `python3 -m pip --python /opt/loom/proxy-venv/bin/python install --no-index
   --find-links` from the shipped bundle; the bundle's wheels match the image's Python
   version and platform; `loom-podproxy --help` runs from the venv. **Two different
   `python3`s:** `serve_bootstrap` runs over SSH and finds `python3` on sshd's session
   `PATH`, while `serve_engine` resolves it on PID 1's `PATH` (the image's environment,
   which may put a venv first). Check both on the pinned image: which interpreter each
   finds, that the bootstrap one has `venv` and `pip`, and that they agree or the
   difference is harmless.
2. **Users and isolation.** `useradd` exists and uids for `loomproxy` / `loomengine` are
   free; `/etc/loom/proxy/*` is 0600 `loomproxy`; `serve_engine`'s check that
   `loomengine` cannot read `/etc/loom/proxy/*` or `/proc/1/environ` passes (it prints
   `loom-sys engine_isolation ok`); the engine runs as `loomengine` bound to
   `127.0.0.1:8000` with `HF_HUB_OFFLINE=1` after the download.
3. **Ports.** RunPod maps `8443/tcp` publicly (`public_port(pod, 8443)`) next to `22/tcp`;
   the proxy's allowlist (`gateway_egress_cidrs`) sees the control plane's egress IP as
   the source address (no NAT in between); the gateway and the manager reach
   `https://<ip>:<mapped 8443>` with `server_name` = the pod id.
4. **Watchdog and lease.** `serve_start.sh` as the start command: sshd starts key-only
   with only `LOOM_SSH_PUBKEY` authorized; `/run/loom/lease` renewals move the deadline;
   the pod terminates itself at the lease (pod-scoped `RUNPOD_API_KEY`), and on a setup
   failure.
5. **Metrics fields.** The vLLM `/metrics` names the health and drain rules read
   (`vllm:num_requests_running`, `vllm:generation_tokens_total`) exist under these names
   in the pinned version, through the proxy; `nvidia-smi` fields `serve_engine` records
   (GPU name, driver, CUDA version) parse.
6. **SIGHUP on the pod.** A certificate renewal over SSH reaches the running proxy
   (`loom-proxy reloaded <pid>`) while a stream stays open.
