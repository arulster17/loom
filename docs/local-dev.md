# Local development

Everything in Phase 0 and the Phase 1 platform runs on a laptop without a GPU or a cloud
account: the mock backend stands in for vLLM/SGLang, the `mock` provider runs it
in-process for the Lab, and `docker compose up` runs the whole platform on it
([Platform (Phase 1)](#platform-phase-1)).

## Prerequisites

| Tool | Version | Used for |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | recent | Python toolchain; installs Python 3.12 from `.python-version` |
| Docker with Compose | v2.20+ | Postgres and Redis for the tests; the full platform stack (`docker-compose.yml`) |

Python dependencies are locked in `uv.lock`. The workspace root (`pyproject.toml`) holds
the dev tools (pytest, ruff, moto, ...); the package is `bench/` (`loom-bench`, CLI
`bench`).

## Setup

```bash
uv sync                 # .venv with loom-bench (mock + aws extras) and dev tools
uv run bench --help
```

Optional extra for lm-evaluation-harness tasks (MMLU-Pro, GSM8K, IFEval, RULER):

```bash
uv sync --all-packages --extra lmeval
```

The extra belongs to the `loom-bench` workspace member, so `--all-packages` is needed at
the repo root. A later plain `uv sync` removes it again.

## Tests

```bash
uv run pytest -q        # ~800 tests, about 2 minutes; Postgres tests are skipped
```

Tests never touch the network or a cloud: AWS calls go to moto or fakes, HTTP to
`httpx.MockTransport` or the mock backend. The one exception is deselected by default:

```bash
uv run pytest -m network -s bench/tests/deps   # a few minutes; downloads wheels and datasets
```

It rebuilds the GPU hosts' client environment exactly as RunPod and AWS hosts install it
(a clean CPython 3.12 venv, the hash-locked `uv export` requirements with the `lmeval`
extra via `pip --require-hashes --no-deps`, the wheel, `pip check`) and runs every task of
every eval suite at 2 items, plus divergence, through `bench quality job` against the mock
backend. A missing or broken eval dependency fails here instead of on a paid pod.
`LOOM_POD_REQUIREMENTS=<file>` checks another requirements set; without `HF_TOKEN`, tasks
whose tokenizer is license-gated are skipped when another suite runs the same harness task.

`uv run pytest -m network bench/tests/quality/test_divergence_byte_split.py` (seconds;
downloads Qwen3's `tokenizer.json` once) checks cross-engine divergence on real token
ids. It renders vLLM's captured prompts 20 and 40 from sweep 565b8d3f as SGLang returns
them and scores them against the capture.

Postgres-marked tests (`-m postgres`) need `LOOM_TEST_DATABASE_URL`. Each test creates
and drops its own schema, so the compose database is safe to use:

```bash
docker compose up -d postgres
LOOM_TEST_DATABASE_URL=postgresql+psycopg://loom:loom@localhost:5432/loom \
  uv run pytest -q -m postgres          # or drop -m to run everything
```

Markers (`pyproject.toml`): `postgres`, `redis`, `slow`, `network` and `acceptance` (the last two
deselected unless asked for with `-m network` or `-m acceptance`).
Default per-test timeout is 120 s.

## Platform (Phase 1)

### Tests

```bash
docker compose up -d postgres redis
uv run loom-migrate --create-database        # database loom_platform
uv run pytest -q services                    # platform tests use the compose servers
```

Platform tests use `LOOM_TEST_DATABASE_URL` / `LOOM_TEST_REDIS_URL` when set, else the
compose servers, and skip when neither is reachable. Each session creates and drops its
own database. Contracts: [phase1-contracts.md](phase1-contracts.md).

### The whole stack

```bash
LOOM_ADMIN_EMAILS=you@example.com docker compose up -d --build
docker compose ps                            # all running; migrate and devcerts exited 0
```

The first build takes a few minutes; later `up`s reuse the images. Nothing calls a cloud
or Stripe: billing talks to a local Stripe stub (unless `STRIPE_SECRET_KEY` and
`STRIPE_WEBHOOK_SECRET`, test mode only, are set) and the "GPUs" are mock pods.

| Service | Host port | What |
|---|---|---|
| `web` | [3000](http://localhost:3000) | Dashboard, playground, admin |
| `gateway` | [8080](http://localhost:8080/v1), metrics 9100 | Public OpenAI-compatible API |
| `billing` | 8090, metrics 9101 | `loom-billing all`: usage stream consumer, ledger batches, reconcile, Stripe webhooks, internal checkout API |
| `stripe-stub` | [12111](http://localhost:12111) | Local Stripe Checkout stand-in |
| `mailpit` | [8025](http://localhost:8025) (UI), SMTP 1025 | Catches magic-link email |
| `postgres` | 5432 | Databases `loom_platform` (platform) and `loom` (bench) |
| `redis` | 6379 | Limits, balances, usage stream, pool state |
| `migrate` | | One-shot: `loom-migrate --create-database` |
| `devcerts` | | One-shot: dev CA and certificates into `.dev/certs` |
| `mock-pod-1` | none (8443 on the compose network) | Mock engine for `qwen3-8b` behind the mTLS pod proxy |
| `mock-pod-2` | none (8443 on the compose network) | Same for `llama-3.3-70b-instruct-fp8` |
| `poolsim` | | Pool state in Redis: which pod serves each model, simulated cold starts |

The end-to-end flow:

1. Open <http://localhost:3000>, sign in with your email; the link arrives in Mailpit
   (<http://localhost:8025>). An email in `LOOM_ADMIN_EMAILS` is made admin at sign-in.
2. New signups wait for approval: approve your org in the admin page, which posts the $5
   signup credit.
3. Create an API key in the dashboard and call the API with the unchanged OpenAI SDK:

   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:8080/v1", api_key="loom_sk_...")
   stream = client.chat.completions.create(
       model="qwen3-8b",
       messages=[{"role": "user", "content": "Hello"}],
       stream=True,
       stream_options={"include_usage": True},
   )
   ```

4. Usage and balance update in the dashboard within a second or two, once billing has
   consumed the usage event; the per-minute batch then posts it to the ledger.

Host ports move with variables, e.g. to run a second stack next to the main one:
`LOOM_PG_PORT=15432 LOOM_REDIS_PORT=16379 LOOM_GATEWAY_PORT=18080 ... docker compose -p
loom2 up -d` (the variables are in `docker-compose.yml`: `LOOM_PG_PORT`,
`LOOM_REDIS_PORT`, `LOOM_GATEWAY_PORT`, `LOOM_GATEWAY_METRICS_PORT`, `LOOM_BILLING_PORT`,
`LOOM_BILLING_METRICS_PORT`, `LOOM_STRIPE_STUB_PORT`,
`LOOM_WEB_PORT`, `LOOM_MAILPIT_UI_PORT`, `LOOM_MAILPIT_SMTP_PORT`).

Start over: `docker compose down -v` (drops the databases) and `rm -rf .dev/certs` (a new
CA on the next `up`).

### M1 acceptance

`loom-m1-acceptance` (`services/devstack`) runs the whole M1 acceptance against a running
stack, from the outside, and exits 0 only if every item passes
([phase1-m1-acceptance.md](phase1-m1-acceptance.md) has the items and the recorded
numbers). It signs up through the web app (magic links from Mailpit's API), approves as
the bootstrap admin, calls the gateway with the OpenAI SDK, kills and restarts a mock pod,
sets a pool cold, pays through the Stripe stub and greps every container's logs. It
creates its own users and orgs, so it can run repeatedly against one stack:

```bash
LOOM_ADMIN_EMAILS=admin@m1.test docker compose up -d --build --wait
uv run loom-m1-acceptance --json /tmp/m1.json          # ~5 minutes
uv run loom-m1-acceptance --only b,f                    # a subset (the sign-up flow always runs)
```

On moved ports, export the same `LOOM_*_PORT` variables and `COMPOSE_PROJECT_NAME` for the
run as for `docker compose` (or pass `--web`, `--gateway`, `--database-url`, `--project`,
...). As a test: `uv run pytest -m acceptance -s services/devstack/tests/test_devstack_acceptance.py`
(deselected by default; skips when the stack is down).

### Mock pods and the pod proxy

Each mock pod is one container (`loom-mockpod`, `services/devstack`): the bench mock
engine on `127.0.0.1:8000` inside it, and the pod proxy (`loom-podproxy`,
`services/podproxy`) on 8443 as the only way in, as on a RunPod serving pod
([phase1-contracts.md §9](phase1-contracts.md#9-gateway-to-pod)). The proxy requires TLS
1.3 and a client certificate from the dev CA, the pod's `X-Loom-Pod-Token`, forwards only
the five contract routes with only `Content-Type` and `Accept`, streams byte for byte, and
cancels the engine request when the client goes away. Source-address filtering is off in
compose (the gateway's address is not fixed); set `LOOM_PODPROXY_ALLOW_IPS` on a pod to
try it.

`devcerts` writes into `.dev/certs` (gitignored): `ca.crt`/`ca.key`, the gateway's
`gateway.crt`/`gateway.key`, and per pod `<pod>.crt`/`<pod>.key` (name = the pod id = the
compose service name) and `<pod>.token`. It keeps valid files, so every `up` reuses them;
`uv run loom-devcerts --force` (or deleting the directory) starts over. The gateway mounts
them as `LOOM_GATEWAY_CLIENT_CERT`, `LOOM_GATEWAY_CLIENT_KEY`, `LOOM_POD_CA_CERT`.

The engines stream at roughly the real replicas' low-load pace (about 20 ms between tokens
for the 8B, 25 ms for the 70B; `ENGINE_PROFILES` in `loom_devstack/mockpod.py`). Mock
engine options for a standalone run:

```bash
uv run python -m loom_bench.mock --port 8000 --model qwen3-8b \
  --set step_base_ms=20 --set prefill_ms_per_token=0.1 --set time_scale=1
```

Inter-token latency is about `step_base_ms + decode_ms_per_seq x running sequences`, time
to first token about `step_base_ms + prefill_ms_per_token x uncached prompt tokens`;
`time_scale` scales both. Streams honour `stream_options.continuous_usage_stats` as vLLM
does (a running usage block on every chunk when `include_usage` is on too), which is how
the gateway bills a stream the client aborted.

### Pool states: "model temporarily unavailable"

`poolsim` seeds both models `ready` on their pods. To exercise the gateway's 503 +
`Retry-After` path ([phase1-contracts.md §10](phase1-contracts.md#10-model-pools-and-model-temporarily-unavailable)):

```bash
docker compose exec poolsim loom-poolsim status
docker compose exec poolsim loom-poolsim set qwen3-8b cold
# next request: 503 model_unavailable, the gateway publishes a wake, poolsim moves the
# model to warming (Retry-After counts down) and back to ready after 20 s
docker compose exec poolsim loom-poolsim set qwen3-8b unavailable   # "no GPU capacity", 300 s
docker compose exec poolsim loom-poolsim set qwen3-8b ready
```

`LOOM_POOLSIM_COLD_START_S` (default 20) sets the simulated cold start;
`LOOM_POOLSIM_IDLE_STOP_S` (default 0, off) stops a model after that long without usage,
as an on-demand pool would.

### Load test

`loom-loadtest` (`services/devstack`) drives the gateway with real API keys through the
Lab's streaming client and reports TTFT, TPOT, inter-token latency, per-request and
aggregate tokens/s, and counts per HTTP status:

```bash
uv run loom-loadtest --base-url http://localhost:8080/v1 --api-key loom_sk_... \
  --model qwen3-8b --concurrency 8 --requests 200 --json /tmp/loadtest.json
```

`--workload` takes `chat-short` (default: ~256-token prompts, up to ~128-token answers) or
any Lab profile (`fixed-1k-1k`, `shared-prefix`, ...); engine-only fields such as
`ignore_eos` are dropped unless `--engine-params`. Mind the limits: by default a key may
have 8 requests in flight and 60 per minute, an org 16 and 120, so heavier runs need
several keys (`--api-key` repeated, or `--api-key-file`), keys from several orgs, or raised
limits; otherwise they measure 429s. `--max-error-rate 0.01` makes the exit code a gate.

## Results database

`bench` uses `--db URL`, else `$LOOM_DATABASE_URL`, else
`postgresql+psycopg://loom:loom@localhost:5432/loom` (the compose Postgres). `bench plan`,
`run`, `reproduce`, `report`, `competitiveness` and `reap` apply migrations themselves;
`bench db upgrade` does it explicitly.

```bash
docker compose up -d postgres
uv run bench db upgrade
# or, without Docker:
mkdir -p results && export LOOM_DATABASE_URL=sqlite:///results/loom.db
```

## Mock experiments end to end

```bash
uv run bench plan bench/experiments/mock-smoke.yaml    # hosts, step times, spend vs caps
uv run bench run  bench/experiments/mock-smoke.yaml    # ~15 s
uv run bench report                                    # reports/leaderboard.{md,html,csv}
uv run bench competitiveness                           # reports/competitiveness.{md,html,csv}
```

`mock-smoke` runs 2 configs (`baseline`, `small-batch`) × 2 workloads × 3 load points ×
2 repetitions on one simulated host (cold start, then a warm restart), runs a small
quality suite (`bench/experiments/suites/mock-smoke.yaml`) on both and gates
`small-batch` against `baseline`. The mock's `hourly_price: "$1.00"` is simulated: it
exercises the cost math and the budget guard but never counts toward the overall cap.
`time_scale: 0.01` makes the simulated GPU 100× faster than real time, so the numbers
are plumbing checks, not benchmarks.

Other mock runs:

```bash
uv run bench run bench/experiments/mock-budget-abort.yaml   # budget guard trips: exit 5
uv run bench reproduce <run id>       # re-run one stored run, compare (exit 0 or 6)
uv run bench compare <exp A> <exp B>  # per-metric deltas at the cells and loads both ran:
                                      # exit 0 within variance, 6 outside, 2 nothing matched;
                                      # --match-by cell_key lists config differences, unjudged
uv run bench export csv --out runs.csv
uv run bench reap --dry-run           # expired resources (none for finished mock runs)
```

`bench report` prints the `bench reproduce <run id>` command for each result. With no
`-e/--experiment`, reports cover every completed experiment except reproductions; the
selected experiments must share one `slo` and `cost_allocation`.

Experiment and run ids can be given in full or as a unique prefix of at least 4 hex
digits, like git's short hashes: `bench report -e 7a8237d0`. An ambiguous prefix is
refused with the ids it matches (exit 2).

`results/` and `reports/` are written to the current directory (the defaults of `--out`
on `run`, `reproduce`, `report` and `competitiveness`); at the repo root both are in
`.gitignore`, so the quick start never leaves files to commit.

## The mock backend

An OpenAI-compatible server with a simulated batching GPU (continuous batching, prefill
and decode step costs, KV-cache capacity and preemption, prefix cache) and vLLM-named
Prometheus metrics. Settings: `bench/src/loom_bench/mock/config.py` (`MockConfig`).

```bash
uv run bench mock-server --port 8000            # optional: --config mock.yaml (MockConfig fields)
curl -s localhost:8000/v1/models
curl -sN localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"mock-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"stream":true}'
curl -s localhost:8000/metrics | grep '^vllm:'
```

Routes: `/health`, `/version`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`,
`/metrics`. It serves `mock-model` unless `models` is set in the config.

### Benchmarking a running endpoint (`local` provider)

The `local` provider benchmarks an endpoint you already run (the mock server, or vLLM /
SGLang on your own GPU). Nothing is started, stopped or billed, and an experiment has
exactly one cell. Example, against the mock server above:

```yaml
# local-mock.yaml
name: local-mock
description: Benchmark a running bench mock-server through the local provider.
model: qwen3-8b                      # registry id the results are filed under
provider:
  kind: local
  base_url: http://127.0.0.1:8000/v1
  metrics_url: http://127.0.0.1:8000/metrics
  engine: mock                       # vllm | sglang | mock: selects the metric-name map
  served_model: mock-model
  tokenizer: simple                  # hf: the registry model's pinned tokenizer
variants:
  - name: as-running
workloads:
  - profile: fixed-128-128
    load: {mode: closed_loop, values: [1, 4], num_requests: 8, warmup_requests: 2}
repetitions: 2
slo: {ttft_ms: {p95: 1000}, tpot_ms: {p95: 50}, max_error_rate: 0.01}
budget: {max_spend: "$1", ttl_minutes: 30}
```

```bash
uv run bench run local-mock.yaml
```

### Quality suite against an endpoint

```bash
uv run bench quality run bench/experiments/suites/mock-smoke.yaml \
  --base-url http://127.0.0.1:8000/v1 --model mock-model --out quality/samples.json
```

Code-execution tasks run only with `--allow-code-exec`; see
[security.md](security.md#code-execution-sandbox).

## Results site

```bash
uv run bench site build                         # committed snapshot site/data -> site/_build
python -m http.server 8000 --directory site/_build
```

To preview mock results, export a snapshot of the experiments you name somewhere other
than `site/data` (the default `--out`, which is the committed, published snapshot). Without
`-e` the export takes only the experiments pinned in `site/config.yaml`:

```bash
uv run bench site export -e <experiment id> --out /tmp/loom-site-data
uv run bench site build --data /tmp/loom-site-data --out /tmp/loom-site
```

See [site.md](site.md) for publishing.

## Lint and format

```bash
uv run ruff format .            # format
uv run ruff format --check .    # what CI checks
uv run ruff check .             # lint (rules in pyproject.toml); --fix applies safe fixes
uv run mypy bench/src           # type check; CI fails on any error
```

`mypy` also runs on the platform packages in CI:
`uv run mypy services/platform_common/src services/gateway/src services/billing/src` and
`uv run mypy services/platform_common/src services/podproxy/src services/devstack/src`.

Untyped third-party libraries get stub packages in the dev group (`types-PyYAML`,
`scipy-stubs`, `pyarrow-stubs`); boto3 and botocore, which have none installed, are
imported with a targeted `# type: ignore[import-untyped]`.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests:
`uv sync --frozen`, `ruff check`, `ruff format --check`, `mypy bench/src`,
`pytest -q -n auto` with a Postgres 16
service and `LOOM_TEST_DATABASE_URL` set, so Postgres-marked tests run there. The `web`
job checks `apps/web`: `npm ci`, `npm run lint`, `npm run typecheck`, `npm test`,
`npm run test:integration` (Postgres service, migrated with `uv run loom-migrate`) and
`npm run build`. A third job, `pod-client-env`, runs `pytest -m network bench/tests/deps`
(above).
`.github/workflows/site.yml` builds the results site on pushes to `main` that touch the
site or report code, and deploys it to GitHub Pages once Pages is enabled
([site.md](site.md#deployment)).

Run the CI checks locally before pushing:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy bench/src \
  && uv run pytest -q -n auto
```
