# Local development

Everything in Phase 0 runs on a laptop without a GPU or a cloud account: the mock
backend stands in for vLLM/SGLang, and the `mock` provider runs it in-process.

## Prerequisites

| Tool | Version | Used for |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | recent | Python toolchain; installs Python 3.12 from `.python-version` |
| Docker with Compose | any recent | Optional: the local Postgres (`docker-compose.yml`) |

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

Markers (`pyproject.toml`): `postgres`, `slow`, `network` (deselected unless `-m network`).
Default per-test timeout is 120 s.

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
uv run bench compare <exp A> <exp B>  # per-metric deltas (exit 6 outside variance)
uv run bench export csv --out runs.csv
uv run bench reap --dry-run           # expired resources (none for finished mock runs)
```

`bench report` prints the `bench reproduce <run id>` command for each result. With no
`-e/--experiment`, reports cover every completed experiment except reproductions; the
selected experiments must share one `slo` and `cost_allocation`.

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

To preview mock results, export a snapshot somewhere other than `site/data` (the default
`--out`, which is the committed, published snapshot):

```bash
uv run bench site export --out /tmp/loom-site-data
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

Untyped third-party libraries get stub packages in the dev group (`types-PyYAML`,
`scipy-stubs`, `pyarrow-stubs`); boto3 and botocore, which have none installed, are
imported with a targeted `# type: ignore[import-untyped]`.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests:
`uv sync --frozen`, `ruff check`, `ruff format --check`, `mypy bench/src`,
`pytest -q -n auto` with a Postgres 16
service and `LOOM_TEST_DATABASE_URL` set, so Postgres-marked tests run there. A second
job, `pod-client-env`, runs `pytest -m network bench/tests/deps` (above).
`.github/workflows/site.yml` builds the results site on pushes to `main` that touch the
site or report code, and deploys it to GitHub Pages once Pages is enabled
([site.md](site.md#deployment)).

Run the CI checks locally before pushing:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy bench/src \
  && uv run pytest -q -n auto
```
