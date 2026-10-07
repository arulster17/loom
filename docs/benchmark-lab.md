# Benchmark Lab: running experiments

An experiment benchmarks one registry model across serving configs (engines, engine args,
parallelism, quantization, checkpoints) and workloads, on one provider, under a hard budget.
Every run stores its per-request rows (Parquet), a summary and a full provenance record, so
any number can be traced back and re-run.

Code: `bench/src/loom_bench/{experiment,plan,budget,jobexec,runner,cli}.py` and
`providers/`. Experiments: `bench/experiments/`. Caps: `bench/budget.yaml`.

## Quick start (no GPU, no cloud)

```bash
uv sync
mkdir -p results
export LOOM_DATABASE_URL=sqlite:///results/loom.db   # or the docker-compose Postgres
uv run bench db upgrade
uv run bench plan bench/experiments/mock-smoke.yaml   # what it will do and cost
uv run bench run bench/experiments/mock-smoke.yaml    # ~15 s on the mock backend
uv run bench report                                   # leaderboard into reports/
```

`mock-smoke` runs 2 mock engine configs x 2 workloads x 3 load points x 2 repetitions on one
simulated host (cold start, then a warm restart), runs a small quality suite on both configs
and gates the second against the first. `mock-budget-abort` shows the budget guard killing
an experiment mid-run (exit 5).

To benchmark an endpoint you already run (vLLM, SGLang, or `bench mock-server`), use the
`local` provider (see below). AWS runs need the account settings in `docs/aws-setup.md`.

## Commands

| Command | What it does |
|---|---|
| `bench plan EXP.yaml` | Hosts, per-step time estimates, estimated spend vs caps. Exit 3 if refused. |
| `bench run EXP.yaml [--dry-run] [--db URL] [--out results/]` | Plans, refuses if over a cap, then runs. `--dry-run` stops after the plan. |
| `bench reproduce RUN_ID \| provenance.json [--tolerance 0.25]` | Re-runs one stored run from its provenance and compares it with the original. |
| `bench reap [--dry-run]` | Terminates resources whose TTL passed (DB-recorded; all Loom-tagged EC2 instances when AWS is configured; all Loom-managed RunPod pods when a RunPod API key is found). |
| `bench report [-e EXP]...` | Leaderboard (md, html, csv) by $/1M output tokens at SLO, with goodput brackets and latency at equal load (also `leaderboard.equal_load.csv`; see "Reading the reports"). |
| `bench competitiveness [-e EXP]...` | Our cost at SLO vs competitors' list prices. |
| `bench compare EXP_A EXP_B [--match-by]` | Per-metric deltas with variance verdicts, goodput brackets and latency at equal load. Exit 6 if outside normal variance. |
| `bench quality run SUITE --base-url URL --model NAME` | Runs a pinned eval suite against an endpoint. |
| `bench quality gate --baseline X --candidate Y` | Re-decides the gate from stored per-item samples (X, Y: experiment id or config hash). Exit 7 if blocked. |
| `bench export csv\|parquet --out FILE` | One row per run with summary and provenance flattened. |
| `bench site export [-e EXP]... [--out site/data]` / `bench site build` | Results snapshot and static site (`docs/site.md`). |
| `bench waitlist count [--no-record]` | Signups in `waitlist_signups`, recorded in `docs/waitlist.md`. |
| `bench db upgrade` | Applies schema migrations. |
| `bench job run --in job.json --out result.json` | Executes one load job; cloud providers run this on the GPU host. |
| `bench quality job --in job.json --out result.json` | Executes one eval job (suite tasks plus the divergence capture or score); cloud providers run this on the GPU host. |
| `bench mock-server [--port] [--config mock.yaml]` | The OpenAI-compatible mock backend. |

Exit codes: 0 ok, 1 failed, 2 invalid input, 3 refused by the planner, 4 stopped before a
step that would pass the cap, 5 hard budget abort, 6 reproduction outside normal variance,
7 quality gate blocked, 8 finished but some load runs or quality evals failed. An
experiment whose every load run failed is recorded as `failed` and exits 1; one with some
failed runs or a failed eval stays `completed`, with the counts in its reason, and exits 8.
A failed eval never stops the experiment: the other engines still run, and a gate missing
a side is `inconclusive` (blocked). Invalid input (a malformed or missing file, an unknown model or
suite, an instance type with no price in `bench/prices.yaml`, a malformed experiment id)
is reported as one message, never a traceback.

The database is `--db`, else `$LOOM_DATABASE_URL`, else the local compose Postgres.

## Experiment YAML

Validated by `loom_bench.experiment.Experiment`; unknown keys are errors.

```yaml
name: qwen3-8b-vllm-vs-sglang       # slug
description: ...
model: qwen3-8b                     # registry id in config/models.yaml
provider: {kind: aws_ec2, region: us-east-1, instance_type: g6e.xlarge, market: spot, disk_gb: 200}
variants:                           # named overrides of the registry entry
  - name: vllm
  - name: sglang
    engine: {name: sglang, version: "0.5.21", image: lmsysorg/sglang@sha256:...}
sweep:                              # optional: cartesian product, applied to every variant
  engine.args.max_num_seqs: [128, 256]
  kv_cache_dtype: [auto, fp8]
sample: {random: 3, seed: 0}        # optional: random search over the sweep grid
workloads:
  - profile: fixed-1k-1k            # bench/workloads/<name>.yaml or a path
    label: fixed-1k-1k              # optional, defaults to the profile name; unique
    overrides: {output_len: 512}    # deep-merged into the profile, re-validated
    load:
      mode: open_loop               # or closed_loop
      values: [1, 2, 4]             # req/s (open) or concurrency (closed) ...
      # search: {lo: 0.5, hi: 4, rel_tol: 0.1, max_points: 6, scale: geometric}   # ... or search
      # the SLO: geometric climbs x`step` (default 2) from lo until a load fails, then bisects
      # on a log scale (few points in overload); linear (the default) tests lo, hi, midpoints;
      # descend: N (default 0) steps down from a failing lo by `step` up to N times
      duration_s: 180               # open loop: required; closed loop: this or num_requests
      warmup_s: 30                  # open loop, inside duration_s; closed loop: warmup_requests
      arrival: {kind: gamma, burstiness: 0.5}   # open loop; the rate comes from the load value
      # poisson (default) | constant | gamma | diurnal | trace (replays a trace file's arrival
      # times at the load value's mean rate; docs/how-to/add-workload-profile.md)
      drain_timeout_s: 60
      request_timeout_s: 600
      scrape_interval_s: 1
repetitions: 3                      # 1 only with allow_single_run: true (flagged untrusted)
slo: {ttft_ms: {p95: 1000}, tpot_ms: {p95: 50}, max_error_rate: 0.01}
cost_allocation: {method: all_output}   # all_input | weighted + output_input_ratio
budget: {max_spend: "$40", ttl_minutes: 480, accrual_interval_s: 15}
quality: {suite: qwen3-8b, subset: phase0, baseline_variant: vllm}   # optional
loadgen: native                     # key in LOAD_GENERATORS; aws_ec2 and runpod take native only
seed: 0
```

**Providers.**

- `mock`: every `MockConfig` field (`time_scale`, `max_num_seqs`, `step_base_ms`,
  `startup_delay_s`, ...) plus `hourly_price` (a positive USD string; simulated, so cost
  math and the guard run end to end; without it results have no cost at SLO and are not
  ranked). Variants may carry `mock: {...}` overrides and sweeps may use `mock.<field>`
  knobs.
- `local`: `base_url` (with `/v1`), `metrics_url`, `engine` (vllm | sglang | mock),
  `served_model`, `tokenizer` (hf | simple), and optionally `hourly_price` (what the
  endpoint's hardware costs, as a positive USD string). Nothing is provisioned or billed;
  one cell only. Without `hourly_price` the endpoint has no price: results show "no cost at
  SLO" and are not ranked, never $0.
- `aws_ec2`: `region`, `instance_type` (default: the registry's), `market` (spot |
  on_demand), `disk_gb`. Account settings come from `$LOOM_AWS_CONFIG` / `LOOM_AWS_*`.
- `runpod`: `cloud_type` (secure), `instance_type` (default: the registry's
  `instance_types.runpod`, e.g. `l40s-x1`), `gpu_type_id` (default from the GPU, e.g.
  `NVIDIA L40S`), `container_disk_gb` (default 80; must hold the weights plus 35 GB),
  `allowed_cuda_versions` (default `["13.0"]`), `data_center_ids` (default: any). Always
  on-demand. Each engine image gets its own pod, so a vLLM vs SGLang experiment plans
  two hosts, each with a cold start. Settings come from `$LOOM_RUNPOD_CONFIG` /
  `LOOM_RUNPOD_*`, the API key from `RUNPOD_API_KEY` or the macOS Keychain
  ([runbook-runpod.md](runbook-runpod.md)).

**Variants** may set `engine` (name, version, image, args, chat_template_kwargs), `hf` (repo,
revision, license, gated, size_bytes, quant_method: a pre-quantized checkpoint), `hardware`,
`parallelism`, `quantization`, `kv_cache_dtype`, `max_context`. The patched entry is
re-validated by the registry rules (pinned revision and image digest, GPUs = tp x pp,
quantization servable from the checkpoint, ...) and the engine launch is rendered by
`engines.render_launch`, so a bad override fails at `bench plan`. Switching engine needs its
image and version and drops the other engine's args; an arg (or sweep value) set to `null` is removed; `max_context` cannot exceed
the registry's.

**Cells.** `expand()` produces one cell per variant x sweep point, in declaration order. A
cell's key is `variant[knob=value,...]`; its `config_hash` is the sha256 of the canonical
JSON of `{model: resolved ModelSpec, launch: EngineLaunch, hardware}` and identifies the
setup across experiments.

**Open vs closed loop.** Open loop sends on an arrival schedule regardless of the server
(the honest mode for latency SLOs; `load_value` is req/s). Closed loop keeps a fixed number
of requests in flight (saturation; `load_value` is concurrency). Every run records
`load_mode` in `bench_runs` and in provenance, and reports label the unit (req/s vs
concurrent). Goodput and cost at SLO compare like with like only within one mode.

**Seeds.** Each run's seed is derived from `(seed, workload, load value, repetition)`: the
same across variants (paired comparisons), different across repetitions and load points (no
prefix-cache hits from repeated prompts). Arrivals and prompts both use it.

## Budget rails

Phase 0 has $150 of GPU money in total and $50 per experiment. Five independent rails keep
it there.

1. **Caps** (`bench/budget.yaml`): `per_experiment_cap` $50, `overall_cap` $149.75 (lowered
   by hand for spend made outside the results DB). An experiment's `budget.max_spend` may
   be lower, never higher. The effective cap is
   `min(max_spend, per_experiment_cap, overall_cap - billable spend already in the DB)`.
   Spend from the `mock` provider is recorded but simulated: it never counts toward the
   overall cap.
2. **Planner refusal.** `bench run` prints the plan first and refuses (exit 3, nothing
   created) when the estimate exceeds the effective cap, when the worst case of every host
   living to its TTL exceeds it, when a host's estimated time exceeds its TTL, or when the
   TTL is above the AWS or RunPod provider's limit. There is no override flag. Assumptions
   live in `plan.py` (`AWS_TIMING`, `RUNPOD_TIMING`, `MOCK_*`, `ASSUMED_*`, `EVAL_*`). On
   AWS: boot 180 s, image pull 300 s,
   weights at 150 MB/s download and 400 MB/s load, engine init 240 s, 30 s per run on top
   of its duration and full drain timeout (an overloaded point waits it out), 90 s
   teardown. An eval job takes, per task, items x a per-kind item time / concurrency
   (`EVAL_ITEM_S`: 24 s for lm-eval, 40 s for needle, 16 s for code, 6-8 s for tool calling
   and JSON, i.e. an item's share of a full engine decoding at the 50 ms TPOT SLO), plus
   60 s of harness start-up per lm-eval task, 4 s per divergence prompt over 16 slots (and
   on the baseline 1 s per prompt, one at a time, for the noise-floor pass), 60 s per job and, once per AWS host, 300 s to install the eval harness. Spot
   hosts are priced like the provider accrues them: price x its safety multiplier (1.25),
   plus the root EBS volume. On RunPod: boot 60 s, image pull (and sshd install) 240 s,
   30 s teardown, the rest as on AWS; a pod is priced at prices.yaml's on-demand rate plus
   its container disk, and the provider accrues the larger of that and the pod's API
   `costPerHr`. Warm restarts reload weights from the host's cache and pull an
   image only when it changes.
3. **Budget guard** (`budget.BudgetGuard`). Every `accrual_interval_s` it writes
   `Σ host.hourly_micros x elapsed` to `bench_spend` (computed from launch each time, so no
   rounding drift; the basis records price, market and the provider's accrual basis). Before
   each step (provision, engine start, run, eval) it stops gracefully (exit 4) if spend so
   far plus the step's estimate would pass the cap. When recorded spend reaches the cap, or
   the overall cap across experiments, it trips: the in-flight provider call is cancelled,
   every host is torn down, the experiment is marked `aborted` with the reason, exit 5.
   **Worst-case overshoot** past the cap is one accrual interval of all live hosts' hourly
   price, plus the teardown time (which is still accrued, so recorded spend stays true):
   $2.32/h x 15 s is about one cent for the Qwen run.
4. **Instance self-TTL.** Every host is created with `ttl_s = budget.ttl_minutes`; EC2
   instances schedule their own shutdown (terminate) at that age, and RunPod pods run a
   watchdog that terminates the pod at that absolute time (a container restart cannot
   extend it), so a dead runner cannot leave one billing. The planner's TTL check bounds that worst case below the cap.
5. **Reaper.** `bench reap` terminates expired resources recorded in `bench_resources`
   (marking them `terminated_by: reaper`, or `self-ttl` for EC2 instances that already ended)
   and, with AWS configured, every Loom-tagged instance past its TTL tag; with a RunPod API
   key, every pod named `loom-bench-…` with `LOOM_MANAGED=true` past its TTL or exited. The
   AWS reaper also runs as a scheduled Lambda (`infra/aws/bench`); there is no scheduled
   RunPod reaper yet.

Spot interruptions (`SpotInterrupted` from the provider) are recorded (the in-flight run as
`interrupted`, an event in `events.jsonl`) and the cell is retried once on a new host if the
budget allows; completed repetitions are not re-run. A second interruption fails the
experiment.

## Provenance and reproduce

Each run's provenance (`bench_runs.provenance`, and `results/<experiment>/runs/<run>/provenance.json`)
records: git sha, dirty flag and branch; bench and load-generator versions; engine name,
version, image, image digest and args; CUDA and driver versions (from the host); model repo,
revision and quantization; parallelism; GPU type and count; cloud, region, instance type and
market; the as-run hourly price (`hourly_micros`: instance plus storage, never the budget
guard's multiplied rate) and its `price_basis` (market, source: observed spot price at
launch, `prices.yaml` or the experiment, storage GB; [cost-model.md](cost-model.md#4-the-hourly-price-h));
the config hash and the full resolved config; the workload
name, profile hash and content kind (synthetic or realistic); dataset and license; load
mode, value and seed; the repetition index; the client host. Fields nobody measured stay
null (the mock has no CUDA, image or cloud).

`bench reproduce <run id>` rebuilds the exact cell from the stored config (not from today's
registry), resolves the workload from the experiment's stored spec, checks that the derived
seed matches, warns loudly when the git sha differs or the tree is dirty, re-runs that load
point and repetition as a new experiment named `<name>--reproduce`, and compares it with
`report.compare`: the original point's repetitions against the new run. Exit 0 within normal
variance, 6 outside. A `provenance.json` path works too; the spec is read from the
experiment's `spec.json` next to it (or `--spec`).

## Reading the reports

**Goodput bracket.** Goodput comes from a search over a grid of loads, so it is known only
to a bracket: at least the goodput load, below the next tested load, which failed. The
leaderboard and `bench compare` show both, e.g. `1 req/s (fails at 1.091)`;
`≥8 req/s (none failed)` when every tested load met the SLO (a lower bound, and cost at SLO
is an upper bound); `none met the SLO (fails at 0.5)` when even the lowest failed. Loads
print to four significant figures, since geometric search produces loads like 1.0905.

**Ties within the search resolution.** Below saturation a server delivers what it is
offered, so two configs whose brackets overlap get their goodput, throughput and cost at
SLO from the same grid point: identical numbers that are not a measured equality
(`report/resolution.py`). The leaderboard names the tie in the goodput cell ("tied with
X"), the recommendation says "Tied for cheapest" or "Tied with X" instead of a percentage,
and the CSV has `goodput_bracket`, `goodput_at_least` and `goodput_ties`. Brackets are
half-open: one config passing 6 and another failing at 6 are separated. In the 8B sweep
565b8d3f, vLLM and SGLang tied on fixed-1k-1k (both `1 req/s (fails at 1.091)`) and
shared-prefix, and were separated on code-completion (1.297 vs 1.091 req/s). Narrowing a
tie needs a finer search (`rel_tol`), i.e. more GPU time.

**Latency at equal load** (`report/equal_load.py`). Where goodput cannot separate configs,
latency at the same offered load can. Each leaderboard board with two or more configs, and
each matched sweep in `bench compare`, gets a table of TTFT p50 and p95, TPOT p50 and p95
and E2E p95 at every load all of them ran, with the SLO verdict at that load:

- Loads shown: every common load at or below the lowest goodput, where all configs are
  compared inside the SLO, plus the highest common load, which shows how they degrade past
  it. In `bench compare` from raw runs (no SLO verdict), every common load.
- Values: geometric means across repetitions with 95% log-t CIs, as everywhere else.
- A config is **lower** on a metric only when its CI lies entirely below every other
  config's at that load (shown in bold); otherwise the verdict is "no significant
  difference". A single repetition has no CI and is never significant. CI separation is
  stricter than the Welch test `bench compare` uses for its per-metric verdicts, so a
  difference called here is a clear one.
- `leaderboard.equal_load.csv` has one row per (board, load, config, metric) with the
  estimate, its bounds, the SLO verdict, `significantly_lower` and the verdict text.

## Results layout

```
results/<experiment id>/
  spec.json            the validated experiment spec
  events.jsonl         provisioning, engine starts, spot interruptions, quality, gates
  goodput.json         goodput and cost at SLO per cell and workload, per price column
                       (on-demand, spot, committed 1y, as run), priced as bench report does
  runs/<run id>/requests.parquet, provenance.json
  evals/<config hash>/samples.json   per-item quality scores (for re-gating)
  evals/<config hash>/reference.json the baseline's divergence reference (ReferenceLogprobs)
```

## Quality

With `quality:` set, every cell runs the suite (or its named `subset`) after its
workloads, as an eval job where the engine is reachable: in-process for `mock` and `local`,
in the client container on the GPU host for `aws_ec2` (`bench quality job`, see
`docs/aws-setup.md`). Baseline variant cells run first. Their eval job also captures the
divergence reference (greedy continuations and top-k logprobs of the pinned prompts) and
its noise floor (the baseline scored against its own capture one request at a time),
stored as `evals/<config hash>/reference.json` with its config hash and provenance. Every
other cell's job scores its engine against the reference of the baseline cell at the same
sweep point, so the two engines never need to be up at once, and the cell is gated against
that baseline on task scores, divergence (limits calibrated on the noise floor) and
sanity; eval runs and gate decisions are stored (`bench_eval_runs`,
`bench_gate_decisions`), reports never rank gate-failed configs, and configs whose gate
needs review are ranked with that flag.
Code-executing tasks are refused at `bench plan` unless `allow_code_exec: true`. The suite
is a name in `bench/evals/` or a YAML path from the repo root. See `docs/quality-gate.md`
for the method. A failed eval job is recorded (`quality_failed` event, exit 8) and the
experiment goes on; gates missing a side are `inconclusive`. A task whose requests were
all errors, or more than 10% non-retryable 4xx rejections, fails its eval job this way
instead of scoring zeros. A failed divergence (capture or scoring) keeps the job's task
scores: a `divergence_failed` event, a gate whose divergence check is `inconclusive` with
the error as reason, and exit 8 ("divergence failed in N of M quality evals").

A quality-only experiment (`workloads: []`, with a `quality` section) runs no load: each
engine's cold start, then only its eval job, the divergence reference and scoring, and the
gate. It finishes a gate whose load sweep is already recorded: same model, provider,
images and quality section give the same config hashes, so its evals and gate attach to
that sweep's goodput in reports (the latest eval and gate per config win). `bench report`
checks SLO and cost allocation only across experiments that ran load.

A smoke experiment (`smoke: true`) runs a real experiment's code paths at minimal scale:
`quality.limit: N` (allowed only there) caps every suite task near N items and divergence
at N prompts (`Suite.limited`), the suite's `divergence.hard_prompts` first (prompts whose
continuations split characters across byte-level tokens; Qwen3: 20 and 40). Smoke
experiments share config hashes with the real ones,
so default `bench report` selection and the site's latest snapshot leave them out; pass
`-e <id>` to report one. An experiment counts as a smoke when its spec sets `smoke: true`
or carries the name of a shipped spec that does (`experiment.is_smoke`), so runpod-smoke
runs recorded before the flag existed stay out too.

## Phase 0 experiments

The real runs are on RunPod Secure Cloud on-demand, since the AWS GPU spot quota is 0
(an increase is pending with AWS):

| Experiment | Host | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `qwen3-8b-vllm-vs-sglang-runpod` | 2x RunPod 1x L40S (one pod per engine), 3.6 h each of a 6 h TTL | $7.92 | $13.21 | $15 |
| `llama-3.3-70b-tp4-runpod` | 1x RunPod 4x L40S, 2.7 h of a 4 h TTL | $11.73 | $17.58 | $45 |
| `runpod-smoke` | 2x RunPod 1x L40S (the Qwen sweep at smoke scale), 37 min each of a 60 min TTL | $1.35 | $2.20 | $2.25 |
| `qwen3-8b-quality-runpod` | 2x RunPod 1x L40S, evals and gate only (finishes 565b8d3f's gate), 45 min each of a 75 min TTL | $1.64 | $2.75 | $2.80 |

The AWS specs stay as the secondary path:

| Experiment | Host | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `qwen3-8b-vllm-vs-sglang` | 1x g6e.xlarge spot, 7.1 h of an 8 h TTL | $16.56 | $18.56 | $40 |
| `llama-3.3-70b-tp4` | 1x g6e.12xlarge spot, 2.7 h of a 3.5 h TTL | $28.04 | $35.86 | $45 |

Rates are searched geometrically, with ranges sized for one L40S from the first RunPod
sweep (058128e9): Qwen3-8B fixed-1k-1k passed 1.22 req/s and failed 1.94; shared-prefix
and code-completion passed 1 and failed at 4.88 and 6.88, so those two start at 1 req/s.
The earlier linear search over hi = 12-48 req/s spent most of its points deep in
overload. Every search has `descend: 2`: the pass verdict uses the CI upper bound over 3
repetitions, which can fail a first point whose raw latencies pass easily, so a failing
`lo` steps down to lo/2 and lo/4 before the workload ends with no goodput.

Estimates include cold starts, assume every searched point waits out its drain timeout,
and include each config's eval job (the suites' `phase0` subset: GSM8K, IFEval, tool
calling, JSON schema, divergence and sanity; about 27 min per config) and the eval harness
install. The full suites would take about 1.8 h per config and push the Qwen host past its
TTL. RunPod and AWS run the same variants, workloads, SLO and quality subset on the same
GPUs (L40S).

