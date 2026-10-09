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
| `bench report [-e EXP]... [--alt-slo T=V]` | Leaderboard (md, html, csv): a plain-English summary per model and workload, the boards ranked by $/1M output tokens at SLO with goodput brackets and latency at equal load (also `leaderboard.equal_load.csv`), and a competitiveness section (also `leaderboard.competitiveness.csv`); see "Reading the reports". |
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
  # - name: vllm-fp8
  #   model: <another registry id>    # patch that entry instead (e.g. the model's FP8 row);
  #                                   # the quality suite must cover it (also_models)
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
# or gate every cell against a config evaluated in an earlier experiment:
# quality: {suite: ..., subset: ..., baseline: {experiment: <uuid>, config_hash: <64 hex>}}
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
   image only when it changes. A warm restart onto a checkpoint no earlier start on the
   host downloaded (a variant's `model` or `hf` override, e.g. the FP8 row after BF16)
   downloads it first and is planned with that download; a RunPod pod's container disk
   must hold every checkpoint its cells serve plus 35 GB of headroom.
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

A leaderboard reads top to bottom: summary, boards, competitiveness, methodology.

**Summary.** Per model: the hardware and its on-demand price, a table with every config
on every workload ($/1M input, $/1M output, $/1M blended, goodput bracket, standing,
quality), then a few sentences per workload, all generated from the numbers
(`report/summary.py`):

- the cost at SLO of the quoted config, with 95% CIs. The quoted config is the first of:
  trusted, then quality verified (the gate's reference or a gate pass), then leaderboard
  rank; a config that failed the gate is never quoted. When it is not rank 1 the text
  says why (e.g. rank 1's gate is inconclusive);
- ties within the search resolution;
- what limits it: each SLO target missed at the first failing load, with its value and
  CI (the verdict uses the CI upper bound, and the text says so when the mean is inside
  the target). With no passing load, the targets missed at the lowest load, and engine
  settings that explain a floor (e.g. `NCCL_P2P_DISABLE=1`, from the provenance);
- caveats: why a config is untrusted, with the numbers (e.g. "run-to-run CV 14.7%
  exceeds 10%");
- "No cost at the declared SLO" with the reason. With `--alt-slo`, the alternative
  figure follows, labelled "NOT the declared one, for reference only";
- the public list prices at the workload's mix and our blended cost as a multiple.

Then the quality per config: the reference, or each gate check's verdict with its reason.

**$/1M input, $/1M output and blended.** The headline split charges input tokens the
replica's measured prefill time (request rate × mean TTFT at goodput) and output tokens
the rest; blended is the replica's cost over all tokens at the workload's own mix
(docs/cost-model.md, section 2). The board's "Rank key" column is the cost under the
experiments' declared allocation (`all_output`), and ranking still uses it.

**Workload order.** A model's boards start with the most representative workload, a chat
shape, then fixed-1k-1k, then the others by name (`leaderboard.workload_order`), so
code-completion (short output) comes after them.

**Competitiveness.** Per model: our cost per side and blended, our price ("no price
set" today) and the break-even price, public list prices with source and date, each also
at every workload's mix, competitor quantization where disclosed, and the flags
(`cost_above_market`, `price_above_market`, `negative_margin`, ...). Public list prices
only (docs/cost-model.md, section 6).

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

**Alternative SLO view.** `bench report --alt-slo tpot_ms.p95=100` (repeatable, also
`max_error_rate=0.02`) re-judges the same runs with those SLO targets replaced. It writes
a second report, `leaderboard.alt-slo.{md,html,csv,equal_load.csv}`, whose title names
the SLO it used and the one the experiments declare. The main leaderboard always uses the
declared SLO, and nothing is stored. Where no config has a cost at the declared SLO, the
main leaderboard's summary and competitiveness section quote the alternative figure,
labelled as such. A search runs in place against the declared SLO, so
under a looser alternative the goodput is often only a lower bound, at the highest load
tested (`≥… (none failed)`).

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

**Stored baseline.** `quality.baseline: {experiment, config_hash}` (instead of
`baseline_variant`) gates every cell against a config an earlier experiment evaluated,
without serving it again: the Llama 3.3 70B FP8 row is gated against BF16 run cf4d1614.
`runner.load_stored_baseline` reads that config's `samples.json` and `reference.json`
(through its `bench_eval_runs` rows) and `bench plan` refuses the experiment if they are
missing, come from another suite, have a failed capture, a reference of another config or
shape, no task in common, a different item count, or a native task of another version
(lm-eval versions are only known at run time; the gate checks them, and item ids and
content hashes, again). Every cell's eval job then scores divergence on the stored
reference (`score`), and the gate uses the stored noise floor. Tasks only one side ran (one
added since the baseline, like `tool_calling_strict`) are reported with their scores and
listed as "not gated" in the decision (`GateDecision.ungated`); the shared tasks,
divergence and sanity decide. `bench quality gate` does the same for two configs from
different experiments. A smoke cannot use a stored baseline (its capped items never pair
with a full run). A suite gates other checkpoints of its model listed in `also_models`
(the FP8 registry rows `llama-3.3-70b-instruct-fp8` and `qwen3-8b-fp8` run their BF16
suites; the FP8 smokes serve `qwen3-8b-fp8` by `Variant.model`, as the 70B runs do).

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
| `llama-3.3-70b-tp4-runpod` | 1x RunPod 4x L40S, 2.7 h of a 3.5 h TTL | $11.88 | $15.38 | $20 |
| `runpod-smoke` | 2x RunPod 1x L40S (the Qwen sweep at smoke scale), 37 min each of a 60 min TTL | $1.35 | $2.20 | $2.25 |
| `llama-3.3-70b-fp8-tp4-runpod` (**on hold**, superseded by the H100 run) | 1x RunPod 4x L40S, Llama 3.3 70B FP8 at defaults, gated against BF16 run cf4d1614 (phase0-strict: tool_calling_strict reported, not gated); 2.5 h of a 3.25 h TTL (planner; cf4d1614 took 81 min) | $11.05 | $14.24 | $15 |
| `runpod-smoke-fp8` | 1x RunPod 1x L40S, the FP8 run's FP8 path at smoke scale (Qwen3-8B BF16 vs RedHatAI FP8-dynamic), 51 min of a 65 min TTL | $0.93 | $1.19 | $1.25 |
| `llama-3.3-70b-h100-tp2-runpod` (approved 2026-10-08, after its smoke) | 1x RunPod 2x H100 SXM ($7.98/h), BF16 baseline + FP8 at TP=2 on one pod, FP8 gated in-run; 5.1 h of a 5.5 h TTL (planner) | $40.53 | $44.09 | $45 |
| `runpod-smoke-h100` (approved 2026-10-08, runs first) | 1x RunPod 2x H100 SXM, the H100 run's paths at smoke scale (Qwen3-8B BF16 vs FP8, TP=2, NVLink P2P), 51 min of a 65 min TTL | $6.78 | $8.66 | $9 |
| `qwen3-8b-quality-runpod` | 2x RunPod 1x L40S, evals and gate only, 3 eval passes per engine (finishes 565b8d3f's gate), 105 min each of a 135 min TTL (planned; ~35 min at b1b904dc's measured pass time) | $3.85 | $4.95 | $5.00 |

The AWS specs stay as the secondary path:

| Experiment | Host | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `qwen3-8b-vllm-vs-sglang` | 1x g6e.xlarge spot, 7.1 h of an 8 h TTL | $16.56 | $18.56 | $40 |
| `llama-3.3-70b-tp4` | 1x g6e.12xlarge spot, 2.7 h of a 3.5 h TTL | $28.04 | $35.86 | $45 |
| `llama-3.3-70b-fp8-tp4` | 1x g6e.12xlarge spot, 2.6 h of a 3.25 h TTL | $26.48 | $33.25 | $45 |

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

### 70B tuning

Llama 3.3 70B BF16 at TP=4 on 4x L40S missed the 50 ms TPOT p95 SLO at every load in
cf4d1614 (at least 60.3 ms, even at 0.06 req/s). Before calling that a hardware limit,
a 47-minute tuning pod (2026-10-08, $3.40) measured the levers on the same image, with
the bench client running cf4d1614's own fixed-1k-1k and shared-prefix jobs over shorter
windows:

- **Physics.** One decode step reads each GPU's 35 GB weight shard. At the L40S's
  864 GB/s that's at least 41 ms. vLLM's defaults were already right: CUDA graphs (full
  and piecewise), `torch.compile`, chunked prefill, no eager mode.
- **NCCL.** On these hosts every P2P setting hangs: `NCCL_P2P_LEVEL=PIX`, `=NODE` and
  the default all timed out in a 4-rank all-reduce. Only `NCCL_P2P_DISABLE=1` works.
  With it a 16 KB all-reduce, one decode token's worth, takes 27.6 µs: 4.4 ms over a
  token's 160 all-reduces. `NCCL_PROTO=LL` cuts that to 20.5 µs but doubles 1 MB
  transfers (prefill), and `NCCL_ALGO=Tree` gives 25.1 µs. So NCCL is worth at most
  about 1 ms per token.
- **Speculative decoding** is the lever. EAGLE3 (`RedHatAI/Llama-3.3-70B-Instruct-
  speculator.eagle3`, pinned) drafts tokens that the 70B model verifies, so outputs
  keep its distribution.

| Config | Load | TPOT p50 / p95 (ms) | TTFT p95 (ms) | Accepted per step |
|---|---|---|---|---|
| defaults (`NCCL_P2P_DISABLE=1`) | fixed-1k-1k 0.125 req/s | 62.4 / 63.1 | 619 | 1 |
| defaults | fixed-1k-1k 0.25 req/s | 75.1 / 77.3 | 945 | 1 |
| EAGLE3, 3 draft tokens | fixed-1k-1k 0.125 req/s | 28.7 / 56.8 | 649 | 2.28 |
| EAGLE3, 3 draft tokens | fixed-1k-1k 0.25 req/s | 38.5 / 47.1 | 999 | 2.28 |
| EAGLE3, 3 draft tokens | shared-prefix 0.25 req/s | 30.8 / 35.4 | 1022 | 2.39 |
| EAGLE3, 2 draft tokens | fixed-1k-1k 0.125 req/s | 33.4 / 55.1 | 645 | 2.15 |
| EAGLE3, 2 draft tokens | fixed-1k-1k 0.25 req/s | 40.5 / 46.4 | 986 | 2.15 |
| EAGLE3, 2 draft tokens | shared-prefix 0.25 req/s | 32.3 / 39.5 | 1030 | 2.15 |

(Windows of 60-90 s, so 6-21 requests per row: too short, see the full sweep below.) The
spec runs EAGLE3 with 3 draft tokens.
Limits:

- **Content-dependent tail.** Per-request TPOT ranges from 26 to 47 ms, with a rare
  outlier at 69 ms: a random-word continuation where few drafts are accepted, so it runs
  at the ~60 ms floor plus the verify overhead. A few such requests set the p95.
- **TTFT now binds** at 0.25 req/s: about 1 s for 1k-2k-token prompts, the prefill
  compute plus queueing behind another arrival.
- **Less KV room.** The draft costs KV-cache space: 41k cached tokens instead of 71k.
  Raising `max_num_batched_tokens` to 8192 left too little KV for one 32k request, and
  the engine refused to start.

**The full sweep (55102ddb, $5.77) overturned the short tuning windows.** Under the declared
SLO, no load met it on either workload. The tuning windows ran 60-90 s with 15 s of
warmup, and 1024-token requests live about 40 s, so concurrency never reached steady state
and few requests were sampled. The sweep runs 180 s windows over 3 repetitions:

| Workload | Load (req/s) | TPOT p95 per rep (ms) | TTFT p95 per rep (ms) | Fails on |
|---|---|---|---|---|
| fixed-1k-1k | 0.25 | 61.6, 67.9, 84.2 | 918, 632, 680 | TPOT |
| fixed-1k-1k | 0.125 | 77.6, 49.8, 50.1 | 814, 660, 687 | TPOT |
| fixed-1k-1k | 0.0625 | 68.8, 36.7, 57.8 | 738, 569, 690 | TPOT |
| shared-prefix | 0.5 | 45.1, 54.6, 60.0 | 922, 1255, 1417 | TTFT, TPOT |
| shared-prefix | 0.25 | 42.9, 35.4, 39.9 | 1014, 1012, 989 | TTFT |
| shared-prefix | 0.125 | 39.7, 37.6, 34.4 | 1016, 1008, 1054 | TTFT |

The median request decodes at 29-48 ms per token (BF16 at defaults: 60-75 ms), but the p95
doesn't. In one fixed-1k-1k repetition at 0.25 req/s, 16 of 43 requests ran over 50 ms,
and 4 ran 82-105 ms. A random-word continuation that the draft can't predict decodes at
the ~60 ms floor plus the verify overhead, and with more than 5% of such requests the p95
lands above 50 ms at any load. A smaller load doesn't help, because the floor is per token.

**The measured floors on this hardware, with P2P off (every P2P setting hangs):**

- **TPOT.** At least 41 ms of weight reads per step. In practice about 60 ms at batch 1,
  and about 4.4 ms of that is NCCL. EAGLE3 lowers the median, not the floor of requests the
  draft can't predict.
- **TTFT.** A 2048-token prefill all-reduces 32 MB 160 times. A second diagnostic pod (on
  2026-10-08) measured 2.76 ms per all-reduce at the default settings: 441 ms of pure
  communication per 2k prefill, 222 ms per 1k. The host-memory path runs at about
  11.6 GB/s. No NCCL setting helped by more than 5%: `NCCL_SHM_USE_CUDA_MEMCPY`,
  `NCCL_MIN_NCHANNELS` 8 or 16, `NCCL_BUFFSIZE` 16 MB, `NCCL_PROTO=Simple`, `NCCL_ALGO=Tree`.
  So shared-prefix TTFT p95 sits at about 1.0-1.05 s at any load: compute plus that
  communication for the ~1,190 uncached tokens. Prefix caching worked (42% of prompt tokens
  hit, as at the defaults).
- **Quality is unchanged by EAGLE3.** Against cf4d1614's BF16 reference, the teacher-forced
  top-k logprobs give KL 0.00022 (top-1 99.87%) on the 37 of 48 prompts whose greedy
  continuations are identical. That's below either config's own noise floor (self-KL
  0.00033 and 0.00035). The other 11 prompts flip at a near-tie: with 99.7% per-token
  self-agreement, about 17% of 64-token continuations flip anyway. Scores: gsm8k 0.956 vs
  0.955, ifeval 0.891 vs 0.897, json_schema 0.977 vs 0.973, tool_calling 0.450 vs 0.450.
- **tool_calling 0.450 is Llama 3.3's typing, not the harness.** In the raw outputs, now
  stored, 29 of the 33 failures pass numbers as JSON strings (`"amount": "250"`), 2 pass
  booleans as strings (`"true"`), 1 passes an array as a string (`"[6, 9]"`), and 1 has a
  wrong value. The prompt is the tokenizer's own Llama 3.3 template, Meta's JSON
  tool-calling format, and vLLM's `llama3_json` keeps the model's JSON as it is. vLLM lists
  "parameters in an incorrect format" as a known Llama 3 issue. vLLM can constrain
  arguments to the schema when a tool sets `strict: true`, but the eval sends plain tools,
  as most clients do, so it measures the model's own typing.

A config that meets 50 ms TPOT p95 on this content needs P2P that works (or NVLink), or
fewer bytes per token (FP8 weights, a different config), or a target taken per model.
`bench report --alt-slo tpot_ms.p95=100` shows the 100 ms view.

The draft is a second pinned checkpoint, so `engines.draft_weights` reads it from the
rendered `--speculative-config` (vLLM) or `--speculative-draft-model-path` and
`--speculative-draft-model-revision` (SGLang). Both providers download it next to the
model, since the engine runs offline. A draft must be a Hugging Face repo pinned to a
40-hex commit.

