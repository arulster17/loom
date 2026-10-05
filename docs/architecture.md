# Architecture (Phase 0: Benchmark Lab)

Phase 0 is one Python package, `loom-bench` (`bench/src/loom_bench/`), with one CLI,
`bench`. It measures a registry model on a serving config (engine, engine args,
parallelism, quantization, GPU, cloud) for latency, throughput, quality and cost per
token at an SLO, and records enough provenance to re-run any number. There is no
server: results go to Postgres (or SQLite) and Parquet on the machine running `bench`,
and are published as a static site.

Inputs are config files, each validated by Pydantic when it is loaded (unknown keys are
errors):

| File | Loader | What it holds |
|---|---|---|
| `config/models.yaml` | `registry.load_registry` | Models: pinned HF revision, engine + image digest, hardware, parallelism, quantization, price |
| `bench/prices.yaml` | `prices.load_prices` | Instance and storage prices per cloud/region, integer micro-dollars, with sources |
| `bench/competitors.yaml` | `prices.load_competitors` | Public list prices of other providers, with sources |
| `bench/budget.yaml` | `budget.load_budget` | `overall_cap` and `per_experiment_cap` |
| `bench/experiments/*.yaml` | `experiment.load_experiment` | One experiment: model, provider, variants, sweep, workloads, load, SLO, budget |
| `bench/workloads/*.yaml` | `workloads.load_profile` | Workload profiles (request shapes) |
| `bench/evals/*.yaml` | `quality.suite.load_suite` | Pinned quality suites per model |
| `site/config.yaml` | `site.config.load_site_config` | Results site settings (waitlist form target) |

## Data flow

```mermaid
flowchart TD
    Y[experiment YAML] --> E[experiment.expand<br/>cells + config_hash]
    R[config/models.yaml] --> E
    E --> P[plan.build_plan<br/>time + spend estimate]
    PR[bench/prices.yaml] --> P
    B[bench/budget.yaml] --> P
    P -->|refusals: exit 3| X[stop, nothing created]
    P --> G[budget.BudgetGuard<br/>accrues spend, trips at cap]
    G --> V[providers.make_provider<br/>mock / local / aws_ec2]
    V --> H[provision Host]
    H --> S[start_engine<br/>engines.render_launch]
    S --> J[LoadJob per load point x repetition]
    J --> X2[jobexec.execute_load_job<br/>loadgen + /metrics scrapes]
    X2 --> RS[LoadJobResult<br/>RequestRecord rows]
    RS --> M[metrics.summarize_run<br/>+ prometheus + gpu]
    M --> DB[(Postgres / SQLite<br/>bench_* tables)]
    RS --> PQ[(requests.parquet<br/>+ provenance.json)]
    S --> Q[quality.run_suite<br/>+ gate_against_baseline]
    Q --> DB
    DB --> A[report.analyze_runs<br/>goodput + cost at SLO]
    A --> L[bench report / competitiveness / compare]
    A --> SX[bench site export<br/>site/data snapshot]
    SX --> SB[bench site build<br/>site/_build -> GitHub Pages]
```

Step by step (`runner.run_experiment`):

1. **Expand** (`experiment.expand`). Each variant's overrides are applied to the
   registry entry, then each sweep point. The patched `ModelSpec` is re-validated and the
   engine launch rendered, so a bad override fails here. Each result is a `Cell`; its
   `config_hash` is the sha256 of the canonical JSON of
   `{model, launch, hardware}` and identifies the setup across experiments.
2. **Plan** (`plan.build_plan`). Cells with the same hardware share one host (cold start
   once, warm restarts between configs). Every step gets a time estimate (assumptions in
   `plan.AWS_TIMING`, `MOCK_*`, `ASSUMED_*`), priced at the host's hourly rate. The plan
   is refused (exit 3, nothing created) when the estimate, or every host living to its
   TTL, exceeds the effective cap, or a host's time exceeds its TTL.
3. **Budget guard** (`budget.BudgetGuard`). Accrues `Σ host.hourly_micros × elapsed` into
   `bench_spend` every `accrual_interval_s`, checks each next step against the cap
   (graceful stop, exit 4) and trips at the experiment's cap or the overall cap across
   experiments (in-flight call cancelled, all hosts torn down, exit 5).
4. **Provider** (`providers/`). Provisions a host, starts and stops the engine, runs
   load jobs where latency is measured correctly, tears down, reaps.
5. **Load** (`jobexec.execute_load_job`). Builds the requests from the workload profile
   (`loadgen.base.prepare_load`), runs the named load generator, scrapes the engine's
   `/metrics` meanwhile and returns a `LoadJobResult`.
6. **Summarise and store.** `metrics.summary.summarize_run` turns records into one
   `RunSummary` (warmup dropped). The runner writes a `bench_runs` row with summary and
   provenance, `requests.parquet` and `provenance.json`.
7. **Quality** (optional). After a cell's workloads, `quality.runner.run_suite` runs the
   pinned suite on the live endpoint; non-baseline cells are gated against the baseline
   cell (`gate_against_baseline`). Stored in `bench_eval_runs`, `bench_gate_decisions`.
   There is no eval job type: suites run from the runner process, which is why the planner
   refuses `quality:` on `aws_ec2` (the endpoint listens on the host's loopback).
8. **Teardown** always runs (`finally`), also on Ctrl-C and budget aborts.
9. **Analyse** (`report.analyze_runs`). Groups completed runs by
   `(config_hash, workload, load_mode)`, aggregates repetitions per load point, finds
   goodput, prices it. The same analysis feeds `bench report`, `bench competitiveness`,
   the table printed after `bench run` and `bench site export`.

## Module map

| Module | Responsibility |
|---|---|
| `cli.py` | Typer commands and exit codes |
| `registry.py` | `ModelSpec` rules: pinned sha, digest-pinned image, GPUs = tp × pp, quant pairs, `trust_remote_code` review |
| `experiment.py` | Experiment schema, variants, sweep, `expand`, `derive_seed` |
| `plan.py` | Dry-run planner and time model |
| `budget.py` | Caps and `BudgetGuard` |
| `runner.py` | Orchestration, provenance assembly, spot retry, `reproduce`, `reap` |
| `engines.py` | `ModelSpec` → `EngineLaunch` (vLLM / SGLang argv) → `docker run` |
| `providers/` | `base.py` contract; `mock.py`, `local.py`, `aws_ec2.py` (+ `aws_ssm.py`, `aws_scripts/*.sh`, `aws_reaper.py`) |
| `jobs.py`, `jobexec.py` | `LoadJob` / `LoadJobResult` and their execution |
| `loadgen/` | `native` driver, arrivals, wrappers for `vllm bench serve` and `sglang.bench_serving` |
| `client/openai_stream.py` | Streaming OpenAI client with per-chunk timestamps |
| `workloads/` | Profiles and request generation |
| `metrics/` | Run summary, repetition aggregates, Prometheus and `nvidia-smi` parsing |
| `slo.py`, `stats.py` | SLO checks, goodput search, confidence intervals, paired bootstrap |
| `cost.py`, `money.py`, `prices.py` | $/1M tokens at SLO, integer micro-dollar math, price book |
| `competitiveness.py` | Price vs market and vs cost flags |
| `quality/` | Eval tasks, suites, gate, logprob divergence, sanity checks, code sandbox |
| `store/` | SQLAlchemy models, Alembic migrations, repo functions, Parquet, CSV/Parquet export |
| `report/` | Analysis, leaderboard, competitiveness, compare, methodology text |
| `site/` | Snapshot export, static site build, waitlist count |
| `mock/` | OpenAI-compatible mock backend with a simulated batching GPU |

## Contracts between modules

**Records** (`records.py`). Every load generator emits `RequestRecord`s: one request as
the client saw it, with times in seconds from the run's t0 (`sent_at_s`,
`scheduled_at_s`, `first_token_at_s`, `finished_at_s`, `itl_s`), token counts from the
server's `usage` block, status (`ok`, `error`, `timeout`, `aborted`) and `warmup`.
TTFT, E2E, TPOT and client queue delay are derived properties. `to_row()` is the Parquet
row. Metrics, storage and reports read only records, so they do not depend on which
generator produced the load.

**Jobs** (`jobs.py`). A `LoadJob` is one load point × one repetition against one
endpoint: base URL, metrics URL, engine name (selects the Prometheus name map), served
model, load generator key, the resolved workload, tokenizer spec, load mode and value,
arrival spec, durations, timeouts, seed, `extra_body`. It round-trips through JSON: the
mock and local providers execute it in-process; `aws_ec2` uploads it to S3 and runs
`bench job run --in job.json --out result.json` on the GPU host, so measured latency has
no WAN hop. A `LoadJobResult` carries the records, the measurement window, the raw
`/metrics` scrapes, optional `nvidia-smi` CSV and `timeline` (`measured`, or
`unavailable` for a tool that reports no send times).

**Providers** (`providers/base.py`). The `Provider` protocol:
`provision(HostRequest) -> Host`, `start_engine(host, EngineLaunch, warm) -> Endpoint`,
`stop_engine`, `run_job(host, LoadJob) -> LoadJobResult`, `teardown` (idempotent),
`reap(now)`. A `Host` reports the spend inputs (`hourly_micros`, the accrual rate, market,
launch time, `ttl_at`); the guard decides when to abort. It also carries its as-run cost
price and basis (`as_run_micros`, `price_basis`, no safety multiplier), which the runner
writes into every run's provenance. Every host gets a TTL; mock and EC2 hosts are
recorded in `bench_resources` so the reaper can find them. Losing a host raises
`HostLost`, or `SpotInterrupted` for a spot reclaim. An `Endpoint` returns the base URL,
metrics URL, cold/warm start stage timings and system info (GPU names, driver, CUDA,
image digest).

| Provider | Host | Engine | Where load runs | Billed |
|---|---|---|---|---|
| `mock` | in-process uvicorn thread | `loom_bench.mock` (simulated GPU) | runner process | simulated (`hourly_price`), never counts toward the overall cap |
| `local` | an endpoint you already run | not started or stopped; one cell only | runner process | no (market `local`) |
| `aws_ec2` | one tagged EC2 GPU VM (DLAMI, Docker) driven over SSM | engine container from the registry image | `python:3.12-slim` container on the host | yes: spot × multiplier or on-demand, plus root EBS |

**Registry** (`registry.py`). `config/models.yaml` is the single source of model
configuration. Experiments never bypass it: a variant patches a registry entry and the
result must pass the same validation. `engines.render_launch` is the only place an engine
command line is built; flags that come from structured fields (`--revision`,
`--tensor-parallel-size`, `--max-model-len`, `--quantization`, `--kv-cache-dtype`,
`--trust-remote-code`, ...) cannot be set through `engine.args`.

**Load generators** (`loadgen/base.py`). `LOAD_GENERATORS` maps a name to a factory; each
generator implements `async run(LoadContext) -> RunResult`. See
[load-generators.md](load-generators.md).

**Eval tasks** (`quality/tasks/`). `TASKS` maps a suite `kind` to a factory; each task
implements `async run(EvalContext) -> TaskOutput` with one `ItemResult` per item. See
[quality-gate.md](quality-gate.md) and [how-to/add-eval-task.md](how-to/add-eval-task.md).

**Store** (`store/`). Tables: `bench_experiments`, `bench_runs`, `bench_cold_starts`,
`bench_eval_runs`, `bench_gate_decisions`, `bench_resources`, `bench_spend`,
`waitlist_signups` (schema in [PLAN.md](PLAN.md#postgres-schema), migrations in
`store/migrations/`). Money is `BIGINT` micro-dollars. Per-request rows are Parquet
files referenced by `bench_runs.requests_uri`. The database is `--db`, else
`$LOOM_DATABASE_URL`, else `postgresql+psycopg://loom:loom@localhost:5432/loom` (the
docker-compose Postgres); SQLite URLs work for local use.

## Where each measurement lives

| Measurement | Code |
|---|---|
| Per-chunk timestamps, TTFT, ITL | `client/openai_stream.py`, `records.RequestRecord` |
| TTFT / TPOT / ITL / E2E percentiles, error rate, client queue delay | `metrics/summary.py` |
| Throughput per replica and per GPU, request rate | `metrics/summary.py` (`Throughput`) |
| SLO attainment, request goodput | `slo.Slo.request_meets`, `metrics/summary.py` |
| Load-point SLO verdict (CI upper bound vs target) | `slo.slo_met` |
| Goodput via load sweep, max sustainable concurrency | `slo.find_goodput`, `slo.bisect_next_load`, `runner._run_workload` |
| Repetitions, CIs, run-to-run CV | `metrics/aggregate.py`, `stats.py` |
| Server queue time, preemptions, KV-cache use, prefix-cache hit rate | `metrics/prometheus.py` (`ENGINE_METRICS`) |
| GPU utilization, memory, power | `metrics/gpu.py`; sampled by `providers/aws_scripts/run_job.sh` |
| Cold and warm start stages | provider `start_engine` → `bench_cold_starts` (`runner._run_cell`) |
| Spot interruptions | `SpotInterrupted` → run status `interrupted`, `events.jsonl` |
| $/1M tokens at SLO with CIs | `cost.py`, `report/analyze.py` ([cost-model.md](cost-model.md)) |
| Quality scores, gate, divergence, sanity | `quality/` ([quality-gate.md](quality-gate.md)) |
| Provenance and config hash | `provenance.py`, `runner._serving_sections` |
| Reproduction and variance verdict | `runner.reproduce`, `report/compare.py` |

## Results on disk

```
results/<experiment id>/
  spec.json            validated experiment spec
  events.jsonl         provisioning, engine starts, spot interruptions, quality, gates
  goodput.json         per cell and workload, every price column, priced as bench report does
  runs/<run id>/requests.parquet, provenance.json
  evals/<config hash>/samples.json   per-item scores for re-gating
reports/               bench report / competitiveness / compare output
site/data/             committed results snapshot (bench site export)
site/_build/           rendered static site (gitignored)
```

`results/` and `reports/` default to the current directory (`--out` changes them).

## Phase 1 (future, not built)

Phase 1 adds the inference platform around the same registry and price code. Proposed
schema and open decisions are in [PLAN.md](PLAN.md). How it attaches to Phase 0:

- **Gateway** (proposed Python FastAPI): reads `config/models.yaml` for routing and
  per-token prices, and `money.py` for micro-dollar math, so serving and benchmark prices
  cannot drift.
- **Billing**: usage events and a double-entry ledger in the same Postgres; Phase 0
  tables are untouched.
- **k8s provider**: a fourth implementation of `providers/base.Provider` that deploys the
  shared Helm chart, so EKS/GKE benchmarks become an experiment `provider.kind`.
  `hardware.nodes_per_replica > 1` is rejected until multi-node serving exists.
- **Production cost validation**: Prometheus cost-per-1M-tokens served, compared with the
  Lab's predictions from `report/analyze.py`.
