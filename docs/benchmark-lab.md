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
| `bench compare EXP_A EXP_B [--match-by]` | Per-metric deltas with variance verdicts, goodput brackets and latency at equal load, judged only on the cells and loads both ran (the rest are listed). Exit 0 within normal variance, 6 when a matched metric is outside, 2 if nothing matched. With `--match-by cell_key` or `workload`, paired sweeps on different configs (another cloud, instance type, engine args) are listed key by key under "Config differences" and do not affect the verdict. Ids here and elsewhere in the CLI accept a unique prefix (4+ hex). |
| `bench quality run SUITE --base-url URL --model NAME` | Runs a pinned eval suite against an endpoint. |
| `bench quality gate --baseline X --candidate Y` | Re-decides the gate from stored per-item samples (X, Y: experiment id or config hash). Exit 7 if blocked. |
| `bench export csv\|parquet --out FILE` | One row per run with summary and provenance flattened. |
| `bench site export [-e EXP]... [--out site/data]` / `bench site build [--require-pinned]` | Results snapshot and static site (`docs/site.md`). Without `-e` the export takes exactly the experiments pinned in `site/config.yaml` (`publish.experiments`); `--require-pinned` refuses any other snapshot. |
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
  #   host_group: b                   # optional: a separate host for this variant's cells
  #                                   # (see "Host groups" below)
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

**Host groups.** Cells on the same hardware (on RunPod, the same engine image too) share
one host through warm restarts. A variant's `host_group` puts its cells on a host of
their own: each group gets its own cold start, eval setup and TTL, and the groups run one
after the other, the quality baseline's first. It splits a sweep longer than one host's
TTL limit (8 h on RunPod) without splitting the experiment, so candidates on later hosts
are still gated in-run against the baseline's stored reference. Only the host key
changes, never the config hash.

**Datasets on remote hosts.** A `chat_dataset` profile reads `path` (e.g.
`${LOOM_DATA_DIR}/sharegpt/...`), which exists only where you downloaded it. With a
`download:` block (Hugging Face dataset, 40-hex revision, file, sha256, size), a RunPod
pod fetches the file itself: the first load job on the pod downloads it into
`/opt/loom/data/<sha256>/`, checks the sha256 and leaves it root-owned and world-readable;
every job's workload `path` points there. An `aws_ec2` host does the same (since
2026-10-10): the first load job downloads it into `AwsSettings.data_dir`
(`/opt/dlami/nvme/loom-data/<sha256>/`, on the instance-store NVMe), and the client
container mounts that directory read-only at `/data`, where the job's `path` points.
`bench plan` refuses a file-reading workload (chat dataset or trace) without a `download:`
block on either cloud. `chat-sharegpt` pins ShareGPT_V3_unfiltered_cleaned_split.json at
commit `192ab218` (673 MB).

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
   teardown; a chat-dataset run adds 30 s to read the dataset and tokenize its
   conversations, and a pod downloads a pinned dataset once. An eval job takes, per task,
   items x an item's slot time / concurrency, plus 60 s of harness start-up per lm-eval
   task, 4 s per divergence prompt over 16 slots (and on the baseline 1 s per prompt, one
   at a time, for the noise-floor pass), 60 s per job and, once per host, 300 s to install
   the eval harness. Tasks timed on real engines (GSM8K, IFEval, JSON schema, tool
   calling, strict tool calling) scale with the cell's decode-step floor, the weight bytes
   each GPU reads per step over its memory bandwidth (`EVAL_ITEM_S_PER_STEP_MS`,
   `decode_step_ms`): calibrated on every stored full pass (8B on L40S, 70B on 4x L40S and
   on 2x H100, BF16 and FP8) as the largest measured ratio plus ~20%, so a phase0-strict
   pass plans at ~14 min for Qwen3-8B BF16 on an L40S (measured 7-8), ~26 min for the 70B
   on 4x L40S (17) and ~15 / ~9 min for the 70B BF16 / FP8 on 2x H100 (7.3 / 4.7).
   Untimed tasks (MMLU-Pro, RULER, needle, code) keep a fixed `EVAL_ITEM_S` (24, 40, 16 s:
   an item's share of an engine decoding at the 50 ms TPOT SLO). Spot
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
   key, every pod named `loom-bench-…` with `LOOM_MANAGED=true` past its TTL or exited.
   Both also run as scheduled Lambdas (`infra/aws/bench`), every 15 minutes: the EC2
   reaper, and the RunPod reaper, which only terminates pods in the runner's exact name
   format, with `LOOM_MANAGED=true`, whose TTL has passed
   ([runbook-runpod.md](runbook-runpod.md#orphaned-pods)).

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
`report.compare`: the original point's repetitions against the new run, matched strictly by
config hash. Exit 0 within normal variance, 6 outside, or when the reproduction's config
hash differs from the original's (nothing matches, so it is not a reproduction). A `provenance.json` path works too; the spec is read from the
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
so default `bench report` selection leaves them out (the site exports only the
experiments pinned in `site/config.yaml`); pass `-e <id>` to report one. An experiment counts as a smoke when its spec sets `smoke: true`
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
| `llama-3.3-70b-h100-tp2-runpod` (ran 2026-10-09 as 9f0853d7: $25.18, results below) | 1x RunPod 2x H100 SXM ($7.98/h), BF16 baseline + FP8 at TP=2 on one pod, FP8 gated in-run; 5.1 h of a 5.5 h TTL (planner) | $40.53 | $44.09 | $45 |
| `runpod-smoke-h100` (ran three times: 4e50b5a5, e929eb0c, 4680fa3e, $6.15) | 1x RunPod 2x H100 SXM, the H100 run's paths at smoke scale (Qwen3-8B BF16 vs FP8, TP=2, NVLink P2P), 51 min of a 65 min TTL | $6.78 | $8.66 | $9 |
| `llama-3.3-70b-h100-tp2-quality-runpod` (ran 2026-10-09 as 9e5f7866: $3.29 in 24.6 min, FP8 gate passes, results below) | 1x RunPod 2x H100 SXM, 9f0853d7's BF16 and FP8 cells again, evals and gate only (`workloads: []`), tool-calling data version 2; 1.27 h of a 2.5 h TTL (planner; ~45 min at 9f0853d7's timings) | $10.20 | $20.04 | $22 |
| `qwen3-8b-quality-runpod` | 2x RunPod 1x L40S, evals and gate only, 3 eval passes per engine (finishes 565b8d3f's gate), 105 min each of a 135 min TTL (planned; ~35 min at b1b904dc's measured pass time) | $3.85 | $4.95 | $5.00 |
| `qwen3-8b-config-sweep-runpod` (ran 2026-10-09/10 as 7a8237d0: $14.44, 13.1 pod-hours, [results below](#qwen3-8b-config-sweep-result-7a8237d0-2026-10-10)) | 3x RunPod 1x L40S in sequence (host groups), 5 vLLM cells: BF16, BF16 + FP8 KV, FP8, FP8 + FP8 KV, FP8 + FP8 KV + `max_num_batched_tokens` 1024; chat-sharegpt and fixed-1k-1k; every candidate gated vs BF16 in-run, 3 passes; 7.0 / 6.6 / 3.4 h of a 7.5 h TTL. [Section below](#qwen3-8b-config-sweep-proposed) | $18.71 | $24.77 | $25 |
| `runpod-smoke-8b-sweep` (ran 2026-10-09 as 8bc65cfa: $1.02, passed) | The sweep at smoke scale on the same three pods, 68 / 67 / 39 min of a 75 min TTL | $3.20 | $4.13 | $4.25 |
| `runpod-smoke-8b-search-fail` (ran 2026-10-09 as 55a6222b: $1.12, passed) | The sweep's pod a (bf16, bf16-kv8) on fixed-1k-1k from an overloading 3.375 req/s, so the search descends, bisects and drains an overload; 1.40 h of a 1.67 h TTL | $1.54 | $1.83 | $2 |

The estimates above were planned when eval time was a fixed per-task constant (~30 min a
phase0 pass on any model). Since 2026-10-09 it scales with the decode-step floor (Budget
rails, rail 2), and `bench plan` gives: `qwen3-8b-vllm-vs-sglang-runpod` $7.41,
`llama-3.3-70b-tp4-runpod` $11.58, `qwen3-8b-quality-runpod` $2.10,
`llama-3.3-70b-h100-tp2-runpod` $35.78, `llama-3.3-70b-fp8-tp4-runpod` $9.95, the smokes
unchanged within a cent or two (same worst cases and caps, which are TTL-bound). The 70B
H100 quality-only rerun ([quality-gate.md](quality-gate.md#what-can-decide-a-60-item-task))
plans at $10.20 instead of $14.88 ($10.13 on the 60-item tool-calling set; the 135-item
data version 2 adds about 30 s of eval per cell).

The AWS specs stay as the secondary path:

| Experiment | Host | Estimate | Worst case (TTL) | Cap |
|---|---|---|---|---|
| `qwen3-8b-vllm-vs-sglang` | 1x g6e.xlarge spot, 7.1 h of an 8 h TTL | $16.56 | $18.56 | $40 |
| `llama-3.3-70b-tp4` | 1x g6e.12xlarge spot, 2.7 h of a 3.5 h TTL | $28.04 | $35.86 | $45 |
| `llama-3.3-70b-fp8-tp4` | 1x g6e.12xlarge spot, 2.6 h of a 3.25 h TTL | $26.48 | $33.25 | $45 |
| `qwen3-8b-aws-g6e` (ran 2026-10-10/11 as 95cde129: $6.39, fp8-kv8's gate fails on ifeval; [results below](#qwen3-8b-on-aws-g6exlarge-result-95cde129-2026-10-11)) | 1x g6e.xlarge **on-demand**, bf16 + fp8-kv8 on chat, 3 passes each, 4.7 h of a 6.5 h TTL | $8.90 | $12.24 | $12.50 |
| `aws-smoke-g6e` (passed 2026-10-10 as 80ff5648, $1.67, after a7bc3123 found the mount bug, $0.58) | the same host and cells at smoke scale, 1.85 h of a 2.5 h TTL | $3.48 | $4.71 | $5 |

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
  as most clients do, so it measures the model's own typing. (The strict variant, run on
  H100 below, showed vLLM 0.30 does not constrain Llama 3.3 70B's calls either.)

A config that meets 50 ms TPOT p95 on this content needs P2P that works (or NVLink), or
fewer bytes per token (FP8 weights, a different config), or a target taken per model.
`bench report --alt-slo tpot_ms.p95=100` shows the 100 ms view.

The draft is a second pinned checkpoint, so `engines.draft_weights` reads it from the
rendered `--speculative-config` (vLLM) or `--speculative-draft-model-path` and
`--speculative-draft-model-revision` (SGLang). Both providers download it next to the
model, since the engine runs offline. A draft must be a Hugging Face repo pinned to a
40-hex commit.

### 70B on 2x H100 SXM (9f0853d7, 2026-10-09)

`llama-3.3-70b-h100-tp2-runpod`: one RunPod Secure 2x H100 SXM pod ($8.0156/h with 260 GB
of storage), TP=2 over NVLink, vLLM v0.30.0, BF16 then a warm restart onto the FP8
checkpoint, FP8 gated against the in-run BF16 cell. Spend $25.18 (DB), plus $6.15 for
three smoke attempts (`runpod-smoke-h100`: 4e50b5a5 $1.18, failed on the second
checkpoint's weights not being downloaded before its offline engine started, fixed since;
e929eb0c $2.26; 4680fa3e $2.71): $31.33 in all, under the $45 + $9 caps. Report:
`reports/70b-h100-v2/` (these load results with the quality-only rerun 9e5f7866's evals
and gate, below); `reports/70b-h100-v1/` is the report as first run.

| Row | Workload | Goodput at SLO | Status | p95 TTFT / TPOT at goodput | $/1M blended | vs lowest eligible list price |
|---|---|---|---|---|---|---|
| Llama 3.3 70B (BF16) | fixed-1k-1k | 0.5 req/s (fails at 0.59) | untrusted: TTFT p95 CV 21.8% | 271 ms / 31.1 ms | $2.37 [1.97, 2.84] | 11.27× DeepInfra $0.210 (fp8, not like for like) |
| Llama 3.3 70B (BF16) | shared-prefix | 2 req/s (fails at 2.83) | untrusted: TTFT p95 CV 10.8% | 411 ms / 42.0 ms | $0.502 [0.419, 0.601] | 4.45× DeepInfra $0.113 |
| Llama 3.3 70B FP8 | fixed-1k-1k | 2 req/s (fails at 4) | trusted, rank 1, gate pass (9e5f7866) | 304 ms / 29.4 ms | $0.546 [0.532, 0.561] | 2.60× DeepInfra $0.210 (fp8, like for like) |
| Llama 3.3 70B FP8 | shared-prefix | 4 req/s (fails at 8) | trusted, rank 1, gate pass (9e5f7866) | 417 ms / 38.1 ms | $0.263 [0.239, 0.289] | 2.33× DeepInfra $0.113 (like for like) |

- **BF16 meets the 50 ms TPOT SLO on H100.** On 4x L40S (cf4d1614, 55102ddb) no load met
  it; here a decode step reads its 70 GB shard pair over ~3.35 TB/s and the all-reduce
  runs over NVLink, and BF16 holds 0.5 req/s on fixed-1k-1k with TPOT p95 31 ms. The
  figure is untrusted (TTFT p95 varies 21.8% across the three repetitions at goodput) and
  so indicative only: rerun before relying on it.
- **FP8 is 4x the goodput for the same pod**: 2 req/s on fixed-1k-1k (2,037 output tok/s
  per replica vs 470) and 4 req/s on shared-prefix, both trusted. Blended cost $0.546 per
  1M tokens at 1k/1k and $0.263 on shared-prefix. Against public list prices for Llama
  3.3 70B (the FP8 row is compared with its base model's listings), that is 2.60× and
  2.33× DeepInfra's fp8 Turbo listing, and 0.53× and 0.25× Together AI's $1.04.
- **Quality gate as first run: inconclusive, so FP8 was blocked** (now settled: it passes
  on the 135-item tool-calling set, next section). gsm8k (0.957 vs 0.955), ifeval
  (0.898 vs 0.889), tool_calling (0.450 vs 0.450), json_schema (0.973 vs 0.970),
  divergence (KL 0.0038 nats, top-1 97.99%) and sanity all pass; tool_calling_strict is
  inconclusive at -1.67 pts [-6.67, +3.33] against its 6-point margin (one item of 60, the
  ±3/n floor). Strict mode never engaged on Llama 3.3 70B (both configs still sent
  numbers as strings), and the lever that can decide the task is more items, not
  replicates: [quality-gate.md](quality-gate.md), "Strict tool calling". The set is now
  135 items (data version 2); `llama-3.3-70b-h100-tp2-quality-runpod` reran both
  configs' evals and the gate on it.

### 70B FP8 gate on tool-calling data version 2 (9e5f7866, 2026-10-09)

`llama-3.3-70b-h100-tp2-quality-runpod`: 9f0853d7's BF16 and FP8 cells again (same config
hashes) on one RunPod Secure 2x H100 SXM pod, evals and gate only, every phase0-strict
task re-measured on both, tool-calling at data version 2 (135 items). Spend $3.29 (DB)
for 24.6 min of pod time, against $10.20 planned and a $22 cap: the image was already
cached on the host and the BF16 weights took 2.3 min, so BF16 was healthy after 4.9 min
(16.5 in 9f0853d7); its eval job took 9.0 min with setup, the FP8 warm restart (download
included) 5.2 min and its eval 5.4 min. Exit 0, no `quality_failed` or
`divergence_failed`; pod terminated by the runner (gone from the API), no network volume.

**Gate: PASS on every check, so FP8 is shippable** as its own labelled row, "Llama 3.3
70B FP8", with BF16 the reference row:

| Task | n | BF16 | FP8 | Delta [95% CI] | Margin | Verdict |
|---|---|---|---|---|---|---|
| gsm8k | 1319 | 0.955 | 0.955 | +0.08 pts [-0.61, +0.76] | 1 pt | PASS |
| ifeval | 541 | 0.891 | 0.893 | +0.18 pts [-1.29, +1.66] | 2 pts | PASS |
| tool_calling | 135 | 0.459 (62) | 0.467 (63) | +0.74 pts [-1.48, +3.70] | 6 pts | PASS |
| tool_calling_strict | 135 | 0.444 (60) | 0.459 (62) | +1.48 pts [-0.74, +3.70] | 6 pts | PASS |
| json_schema | 300 | 0.973 | 0.977 | +0.33 pts [-1.00, +2.00] | 3 pts | PASS |

Divergence: KL 0.0040 nats, top-1 98.15% (48 prompts, 3,015 positions; limits KL 0.05,
top-1 95%; BF16's self-KL ≤ 0.0004). Sanity over 2,428 FP8 outputs: empty 0%, truncated
0.21%, repetition 0.16%, language drift 0.78%: pass.

- FP8 and BF16 differ on 3 of 135 plain items (FP8 gains s-alarm-1 and s-products-3, loses
  m-cart-2) and 2 strict ones (FP8 gains s-lights-1 and s-products-3). On the first 60
  items both score 27 on both tasks, so s-books-1, the strict item FP8 lost in 9f0853d7
  (its schema is unchanged in version 2), passed on both this time: the flip was not
  stable across runs. The 75 new items score 35 (BF16) and 36 (FP8) plain, 33 and 35
  strict.
- Strict mode still never engaged: of the 148 strict failures over both configs, 147
  stored replies (`unconstrained_text`) begin `{"type": "function", "name": ...`, the
  prefix xgrammar's Llama trigger misses, and every one of the 140 wrong values is a
  string where the schema asks for a number (120), boolean (16) or array (4); the other 8
  are unexpected arguments. This settles the prefix question in
  [quality-gate.md](quality-gate.md), "Strict tool calling".
- The tool-calling scores stay low on both (0.459 / 0.467 plain) because Llama 3.3 70B
  types numbers as strings, a model trait, not FP8's: the gate compares the two configs,
  and FP8 costs nothing measurable on any task.


### Qwen3-8B config sweep (proposed)

`qwen3-8b-config-sweep-runpod` (proposed 2026-10-09, not run; the spec's comments carry
the full reasoning) looks for the cheapest Qwen3-8B serving config that passes the
quality gate against BF16, priced at the SLO on a realistic chat workload, to fill
`qwen3-8b`'s `pricing` (now null) as described in
[how-to/price-a-model.md](how-to/price-a-model.md).

**Hardware.** RunPod Secure 1x, live from the API on 2026-10-09 16:47 UTC (all "Low"
stock except A40 "Medium"):

| GPU | $/h | Memory | Bandwidth | GB/s per $ | FP8 tensor cores |
|---|---|---|---|---|---|
| L40S (kept) | 1.09 | 48 GB | 864 GB/s | 793 | yes (Ada) |
| RTX 6000 Ada | 0.99 | 48 GB | 960 GB/s | 970 | yes (Ada) |
| L40 | 0.82 | 48 GB | 864 GB/s | 1054 | yes (Ada, half the L40S's tensor throughput) |
| RTX A6000 / A40 | 0.59 | 48 GB | 768 / 696 GB/s | 1302 / 1180 | no (Ampere: W8A16 Marlin) |
| L4 | 0.59 | 24 GB | 300 GB/s | 508 | yes |
| H100 SXM | 3.99 | 80 GB | 3350 GB/s | 840 | yes |
| H100 NVL | 3.19 | 94 GB | 3900 GB/s | 1223 | yes |

An 8B decode at the 50 ms TPOT SLO is bound by memory bandwidth and KV-cache room, so
$/token follows GB/s per dollar where the batch the SLO allows fits in memory. H100 SXM
is no cheaper per token than the L40S and runs an 8B at batch sizes where scheduling
binds; L4 is worse. L40 and RTX 6000 Ada run the same Ada FP8 kernels on 48 GB for 25% /
9% less and are plausibly cheaper per token, but neither has been run (and the L40's
halved compute may cost TTFT under chat prefill); the Ampere cards serve FP8 weights
through a different kernel path. So the sweep stays on the L40S (the prior baseline and
the registry's GPU), and the winning config can be rerun on an L40 or RTX 6000 Ada pod
afterwards for about $3 once those are priced in `bench/prices.yaml`.

**Cells** (vLLM v0.30.0 only: it won or tied SGLang on every workload in 565b8d3f). TPOT
p95 is what fails first (565b8d3f's 1k/1k knee failed TPOT with TTFT p95 at a third of
its target), and a decode step's time is its weight read plus every running sequence's
KV read, so the grid varies bytes per step: BF16 (the reference row and gate baseline),
BF16 with an FP8 KV cache, FP8-dynamic weights (`qwen3-8b-fp8`), FP8 weights with an FP8
KV cache, and that last with `max_num_batched_tokens: 1024` (vLLM's default chunked-prefill
budget on a 48 GB GPU is 2048; smaller chunks keep decode steps memory-bound, spending
some of TTFT's 3x headroom on TPOT). Every candidate is gated against BF16 in-run on
`phase0-strict` with three replicated passes (IFEval's 2-point margin needs them:
quality-gate.md, "Replicated passes").

**Sizing load windows.** 565b8d3f's untrusted figures (run-to-run CV 11-15%) were
sampling, not engine noise. From its stored runs (32 load points x 3 repetitions):

- **Throughput** at a passing open-loop point is the number of Poisson arrivals that land
  in the window, so its CV is 1/sqrt(N) for N measured requests (output tokens:
  sqrt((1 + cv_len²) / N) when lengths vary). The observed run-to-run variance was 1.06x
  (mean; median 0.93x) that prediction. The flagged points measured 100-200 requests (100 s
  windows at 1-2 req/s: expected CV 7-10%).
- **Flag probability.** With 3 repetitions the sample variance is s² = σ² χ²₂ / 2, so a
  point is flagged (sample CV > 10%) with probability exp(-(10% / σ)²): 27% at N = 130,
  5% at N = 300, 1.8% at N = 400, 0.7% at N = 500.
- **Latency quantiles** varied 3x (TTFT p95), 19x (TPOT p95) and 22x (E2E p95) more than
  resampling requests within a run predicts (median variance ratios): run to run, the
  load trajectory a window draws (bursts, how many long requests overlap) moves them, and
  more so near the knee. That too shrinks with longer windows.
- **Edges.** Requests sent in a window's last ~45 s finish after arrivals stop, so they
  see falling concurrency: on fixed-1k-1k near the knee their mean TPOT was 10-25% below
  the rest (vLLM at 1.189 req/s: 46.5, 47.9, 47.7, 48.5, then 42.5 ms by fifth of the
  window), and with 30 s of warmup the first fifth ran ~5% low as well. Short windows
  flatter TPOT; longer ones dilute it (a `cooldown_s` that keeps arrivals going past the
  window would remove it, not built).

So the sweep measures chat-sharegpt for 210 s after 45 s of warmup (~630 requests at a
BF16 knee near 3 req/s: output-token CV ~4.7%) and fixed-1k-1k for 180 s after 60 s
(~220 requests for BF16, CV ~6.8%, kept for comparability; 450+ for the FP8 cells).
Searches step by 1.5x from mid-range (chat from 4.5 req/s, 1k/1k from 1.5) so BF16
descends and FP8 climbs, then bisect to 5%.

**Plan** (`bench plan`, 2026-10-09): three pods in sequence, 7.04 / 6.57 / 3.38 h of a
7.5 h TTL each (a RunPod pod's TTL is capped at 8 h, so the 17 h sweep is split by
`host_group`), estimate $18.71, worst case $24.77, cap $25. The smoke
(`runpod-smoke-8b-sweep`): $3.20, worst $4.13, cap $4.25, run first. Paths it runs that
no earlier smoke did: the chat workload on a pod (the pinned dataset download and
multi-turn chat requests), FP8 KV cache on vLLM 0.30 on Ada with BF16 and FP8 weights,
`--max-num-batched-tokens 1024`, three host groups in one experiment, and three
replicated phase0-strict passes.

**Follow-up on L40 and RTX 6000 Ada** (`qwen3-8b-winner-ada-runpod`, approved with the
sweep; a draft until the sweep has run). It runs the sweep's winning config on one
RunPod Secure 1x L40 pod ($0.82/h) and one 1x RTX 6000 Ada pod ($0.99/h; both "Low"
stock on 2026-10-09), one after the other: chat-sharegpt only, the workload the price is
set from, at the sweep's windows and search step, and three phase0-strict passes per card
gated against the sweep's **stored** BF16 baseline (a new GPU is a new config hash, so it
ships only on its own gate pass). Filled in from the sweep on 2026-10-10, not run: fp8-kv8
in both variants (the sweep's cheapest at the SLO, though its own gate was inconclusive,
so it is the leading candidate rather than a winner), `quality.baseline` = 7a8237d0's
bf16 (`7a203a3f...`), chat `search.lo` 5.239 (fp8-kv8's L40S goodput). `bench plan`
against the results DB accepts it: 1.98 h + 1.95 h, estimate $3.60, worst case $4.58,
cap $5 (four search points). Dropping to three search points brings it to about $3.0 at
a coarser knee. Its stored-baseline gate has never run on a real pod (the 70B FP8 run on 4x L40S that would have
used it is on hold); it is covered offline by `bench/tests/runner/test_stored_baseline.py`.

### Qwen3-8B config sweep result (7a8237d0, 2026-10-10)

Ran 2026-10-09 23:28 to 2026-10-10 12:36 UTC: exit 0, $14.44 recorded against $18.78
planned (three Secure 1x L40S pods at $1.10/h, 5.4, 5.2 and 2.5 h of their 7.5 h TTLs),
135 of 135 load runs completed, five `quality` events with three phase0-strict passes,
four gates, no `quality_failed` or `divergence_failed`. Every pod was terminated by the
runner (`pods: []`, `networkVolumes: []`, `bench reap --dry-run` clean). Report:
[reports/8b-config-sweep-v1](../reports/8b-config-sweep-v1/leaderboard.md). Before it,
`runpod-smoke-8b-search-fail` (55a6222b, $1.12) ran the search's failure branches on a
pod ([runbook-runpod.md](runbook-runpod.md#smoke-test)); the sweep then hit them for real
(bf16 descended on both workloads; fp8 at 2.25 req/s on 1k/1k cancelled 156-209
stragglers per run at the drain timeout).

Cost at the SLO (TTFT p95 1 s, TPOT p95 50 ms), on-demand $1.1010/h, every goodput
trusted; $/1M in and out under the prefill-time split, with 95% CIs in the report:

| Cell | chat goodput (fails at) | chat $/1M in / out / blended | 1k/1k goodput (fails at) | 1k/1k $/1M in / out / blended | Gate vs bf16 |
|---|---|---|---|---|---|
| bf16 | 2.213 (2.449) | 0.0557 / 0.2517 / 0.1032 | 1.0 (1.107) | 0.0572 / 0.2293 / 0.1432 | reference |
| bf16-kv8 | 3.674 (3.865) | 0.0655 / 0.0713 / 0.0670 | 1.66 (1.837) | 0.0593 / 0.1187 / 0.0890 | inconclusive (gsm8k, ifeval) |
| fp8 | 3.865 (4.066) | 0.0525 / 0.0961 / 0.0636 | 1.5 (1.66) | 0.0542 / 0.1464 / 0.1003 | inconclusive (gsm8k) |
| fp8-kv8 | 5.239 (5.511) | 0.0544 / 0.0242 / 0.0467 | 2.25 (2.756) | 0.0495 / 0.0827 / 0.0661 | inconclusive (gsm8k) |
| fp8-kv8-mbt1024 | 4.734 (4.98) | 0.0603 / 0.0278 / 0.0522 | 2.25 (2.756), tied | 0.0637 / 0.0686 / 0.0661 | inconclusive (gsm8k, ifeval) |

Gates (paired deltas in points with 95% CIs; margins gsm8k 1, ifeval 2, json_schema 3,
tool tasks 6):

| Cell | gsm8k | ifeval | tool_calling | tool_calling_strict | json_schema | KL / top-1 | sanity (trunc / rep / drift) |
|---|---|---|---|---|---|---|---|
| bf16-kv8 | -0.58 [-1.29, +0.13] | -0.86 [-2.34, +0.49] | +0.74 [-1.48, +2.96] | +0.25 [-1.98, +2.47] | +0.11 [-1.00, +1.22] | 0.0036 / 97.78% | 1.02 / 1.55 / 0.86% |
| fp8 | -0.51 [-1.29, +0.33] | -0.12 [-1.73, +1.48] | 0.00 [-2.22, +2.22] | 0.00 [-2.22, +2.22] | -0.11 [-1.67, +1.33] | 0.0064 / 97.00% | 1.04 / 1.35 / 0.82% |
| fp8-kv8 | -0.28 [-1.14, +0.61] | +0.43 [-1.60, +2.46] | 0.00 [-2.22, +2.22] | 0.00 [-2.22, +2.22] | -0.33 [-2.00, +1.22] | 0.0109 / 95.80% | 1.07 / 1.50 / 0.82% |
| fp8-kv8-mbt1024 | -0.66 [-1.54, +0.25] | -1.23 [-2.96, +0.49] | 0.00 [-2.22, +2.22] | 0.00 [-2.22, +2.22] | +0.33 [-1.22, +1.89] | 0.0108 / 96.25% | 0.99 / 1.48 / 0.86% |

Divergence passes on every candidate (limits KL 0.05 nats, top-1 95%; the bf16 noise
floor is self-KL 0.0005, self top-1 99.25%), and so does sanity. BF16's own scores:
gsm8k 0.904, ifeval 0.821, json_schema 0.910, tool_calling and tool_calling_strict 0.978.

**No candidate ships.** Under the quality standard a candidate ships only on a gate pass
within every task's margin, and every gate is inconclusive: gsm8k's CI crosses its
1-point margin on all four (none of them is a measured drop: every CI includes 0), and
ifeval's crosses 2 points on bf16-kv8 and fp8-kv8-mbt1024. BF16 stays the only verified
row (chat $0.1032 per 1M blended). The cheapest config, fp8-kv8 (FP8-dynamic weights and
an FP8 KV cache), serves chat at 2.4x bf16's rate for 45% of its blended cost; its gate
misses only on gsm8k (-0.28 [-1.14, +0.61] against -1.00), and its top-1 agreement
(95.80%) sits closest to its limit of the four. What would settle it is a quality-only
rerun of fp8-kv8 against these configs with more replicated passes (as
`qwen3-8b-quality-runpod` did for 565b8d3f); that is not approved, nor run. FP8
weights alone and an FP8 KV cache alone gave about the same gain (chat 3.865 and 3.674
req/s against bf16's 2.213; 1k/1k 1.5 and 1.66 against 1.0), and together they compound
(5.239 and 2.25). The max_num_batched_tokens knob cost chat goodput (4.734 against 5.239
req/s) for no TPOT difference at equal load and a higher TTFT.

Measured knees against the plan's guesses: bf16 chat 2.2 req/s (planned near 3), 1k/1k
1.0 (1.2); the FP8 + FP8 KV chat knee 5.2 (planned near 9). Chat's prefill-time split
puts most of the replica's cost on input tokens (the prefill share at fp8-kv8's goodput
is near 1), so its $/1M output CI reaches $0; the blended figure is the stable one.

**Pricing, information only** (`config/models.yaml` is unchanged; the margin is the
user's call, [how-to/price-a-model.md](how-to/price-a-model.md)). Chat-sharegpt costs at
the CI high bound, per 1M tokens; market is the lowest public list price per side
(OpenRouter $0.117 input, Fireworks $0.20 output; both outside the flag comparison:
aggregator, unverified):

| Row | Basis | u = 100%, +0% | u = 100%, +20% | u = 50%, +0% | u = 50%, +20% |
|---|---|---|---|---|---|
| bf16 (`qwen3-8b`, shippable today) | cost-plus in / out | 0.0638 / 0.3060 | 0.0766 / 0.3672 | 0.1276 / 0.6120 | 0.1531 / 0.7344 |
| bf16 | market floored at that cost-plus | 0.117 / 0.3060 | 0.117 / 0.3672 | 0.1276 / 0.6120 | 0.1531 / 0.7344 |
| fp8-kv8 (`qwen3-8b-fp8`, only after a gate pass) | cost-plus in / out | 0.0710 / 0.0584 | 0.0852 / 0.0701 | 0.1420 / 0.1168 | 0.1704 / 0.1401 |
| fp8-kv8 | market floored at that cost-plus | 0.117 / 0.20 | 0.117 / 0.20 | 0.1420 / 0.20 | 0.1704 / 0.20 |

Market alone is $0.117 / $0.20. BF16 at market loses money on output tokens at any
utilization (cost CI high $0.306 against $0.20); fp8-kv8 at market clears its CI-high
cost on both sides down to about 61% utilization (input binds: 0.0710 / 0.117). Because
chat's split puts nearly all of fp8-kv8's cost on input, its blended cost ($0.0522 CI high, against about $0.20 for both
listings at this mix) is the steadier comparison.

### Qwen3-8B on AWS g6e.xlarge (proposed)

`qwen3-8b-aws-g6e` (proposed 2026-10-10, ran 2026-10-10/11:
[result](#qwen3-8b-on-aws-g6exlarge-result-95cde129-2026-10-11)) is the brief's §7.8
Model A leaderboard on one GPU type on AWS, and it does three things on one host:

1. **AWS leaderboard rows** for Qwen3-8B: BF16 (reference row, gate baseline) and
   fp8-kv8 (FP8-dynamic weights + FP8 KV cache, the cheapest config at the SLO in
   7a8237d0) on chat-sharegpt at the SLO.
2. **fp8-kv8's gate** against BF16 with 3 replicated phase0-strict passes per side, as
   in 7a8237d0 (the user's choice on 2026-10-10: more passes barely move the odds, see
   the arithmetic below).
3. **A RunPod-vs-AWS cross-check** on the same GPU: the sweep's two cells (same
   checkpoints, KV dtype, image and vLLM args) on the same chat profile, windows (255 s
   with 45 s warmup, 60 s drain) and search (lo 4.5, step 1.5, descend 3, 5 points).
   Where AWS has RunPod's knees, the search visits 7a8237d0's loads exactly (bf16 4.5,
   3, 2, 2.449, 2.213; fp8-kv8 4.5, 6.75, 5.511, 4.98, 5.239), so
   `bench compare 7a8237d0 <aws id> --match-by cell_key` compares them at equal load.
   The config hashes differ (cloud, instance type and market are hardware; `compare`
   lists the differences and judges only the metrics), and so does
   the host: a g6e.xlarge has 4 vCPU (2 cores) and 32 GiB against the pod's 16 vCPU and
   94 GiB, with the load client on the same host, so the cross-check measures the AWS SKU
   as a deployment, CPU included.

**Host.** The account's G/VT quotas are 4 vCPU on-demand and 0 spot, which fits exactly
one g6e.xlarge (1x L40S, $1.861/h on-demand, checked against the AWS Price List API on
2026-10-10, + $0.0219/h for the 200 GB gp3 root: accrued at $1.8829/h). Both cells share
it through a warm restart onto the FP8 checkpoint, which the host downloads then. Weights
(16.4 + 9.4 GB) and the dataset (0.7 GB) go on the 250 GB instance-store NVMe, the
images and client virtualenvs on the root volume.

**Replicates: the arithmetic.** Per gsm8k item (n = 1319), the paired difference of the
two configs' pass means is D_i = (p_i^fp8 − p_i^bf16) + run noise, so with R passes per
side

    Var(D_i) = τ² + (σ²_bf16 + σ²_fp8) / R,     CI half-width ≈ 1.96 √(Var(D_i) / n)

From 7a8237d0's stored per-pass scores (`replicate_scores`, 3 passes per side), the
within-item run variance per pass is σ²_bf16 = 0.0040 (16 items unstable across passes)
and σ²_fp8 = 0.0142 (56 unstable), and Var(D_i) = 0.0250, so τ² = 0.0250 − 0.0182/3 =
0.0190: three quarters of the variance is items FP8 flips **consistently** (19 items
right on every pass of one config and wrong on every pass of the other, 11 of them in
FP8's favour). Replicates divide only the run-noise quarter:

| R per side | CI half-width (pts) | P(resolves), true delta = −0.28 | P(resolves), predictive |
|---|---|---|---|
| 3 (7a8237d0; this run) | 0.85 | 27% | 33% |
| 5 | 0.81 | 30% | 37% |
| 10 | 0.78 | 32% | 41% |
| 20 | 0.76 | 32% | 43% |
| → ∞ | 0.74 | 3% | 46% |

"Resolves" is a PASS: lower bound ≥ −1 pt, so the new delta must land at or above
−1 + half-width (−0.15 pts at R = 3, −0.19 at R = 5) against −0.28 measured. The
predictive column allows for the run noise in 7a8237d0's −0.28 itself (sd 0.21 pts); a
FAIL (point delta or upper bound below −1) is under 1% either way. The other three FP8 cells show the same
structure (τ² 0.012-0.021, run noise 21-30% of the variance), while SGLang vs vLLM, both
BF16, has τ² 0.0029 (b03b3c52: half-width 0.39 pts at R = 3). The normal approximation
gives 0.85 pts at R = 3 against the gate's percentile bootstrap's 0.87 ([−1.14, +0.61]).

So no replicate count makes this gate likely to resolve: R = 5 would add 4 points of
probability (0.85 → 0.81 pts) for about 26 min of host time (~$0.80), R = 10 another 4
for ~$2 more; the user chose R = 3. IFEval must pass too (fp8-kv8 +0.43 [−1.60, +2.46]
against 2 pts: predictive 78% at R = 3, 84% at R = 5, 88% at R = 10), and divergence
top-1 (95.80% against the 95% limit) can fall under its limit on a new reference, which
makes the gate REVIEW (allowed, not a PASS). Overall a PASS is about 23% at R = 3 (28%
at R = 5). The lever that can decide it is **more items**: at τ² ≈ 0.019 a non-inferiority bound of 1 pt needs ~3000 items of
comparable math for a ~0.54-pt half-width (P(PASS on gsm8k-like items) ≈ 64%), ~5000
for 0.42 (73%), ~8800 for 0.31 (81%), with the remaining risk being that the true delta
is nearer −1 than −0.28. GSM8K has only 1319 test items; the candidates (its 7473-item
train split, which Qwen3 has likely trained on, biasing both configs toward agreement;
GSM-Plus or GSM-Symbolic variants, which cluster on their source problems and need a
clustered bootstrap) are a suite decision, not made here.

**Plans** (`bench plan` against the results DB, 2026-10-10; $78.74 billable spent):

| Spec | Plan | Estimate | Worst case (TTL) | Cap | Expected |
|---|---|---|---|---|---|
| `aws-smoke-g6e` | 1.85 h of a 2.5 h TTL | $3.48 | $4.71 | $5 | ~1.4 h, ~$2.60 |
| `qwen3-8b-aws-g6e` | 4.73 h of a 6.5 h TTL | $8.90 | $12.24 | $12.50 | ~4.1 h, ~$7.70 |

"Expected" uses 7a8237d0's measured times (chat search 75-76 min per cell, a
phase0-strict pass 7.7 min on bf16 and 5.2 min on fp8-kv8) plus an EC2 cold start of
~15 min; the 4 vCPUs may stretch both. The laptop drives each run for its whole length.

**The smoke** (`aws-smoke-g6e`, run first; `bench/tests/aws/test_aws_g6e.py` fails if it
drifts from the real spec): same provider block, variants, config hashes, chat profile
(the full-size dataset download, full-length requests), repetitions, SLO and quality
section (3 passes, gated in-run) with tasks capped at 4 items; its search starts at 12
req/s with step 3 so both cells fail their first point, drain an overload, descend, pass
and bisect (the test checks every knee from the floor to lo). Offline,
`bench/tests/aws/hostsim.py` runs both specs end to end on the aws_ec2 provider (moto
EC2 and S3, a simulated host behind SSM serving the mock engine).

Paths the smoke runs on AWS for the first time: every one of them, since no aws_ec2 GPU
run has happened: on-demand RunInstances in the configured subnets (us-east-1e and 1f
do not offer g6e.xlarge and answer `Unsupported`; the provider moves on), the DLAMI
(Ubuntu 24.04, driver 595.91.07, CUDA 13.2) with the vLLM v0.30.0 image (CUDA 13.0),
user-data's TTL shutdown (now recorded as `ttl_shutdown_at` and `ttl_shutdown_lead_s` in
the cold start's `engine_started` event), the HF token from Secrets Manager, weights on
the NVMe, the warm restart onto the FP8 checkpoint, the pinned dataset download, the
client container (load and lm-eval jobs, uid 10001, presigned S3 URLs), and on-demand
accrual.

### Qwen3-8B on AWS g6e.xlarge result (95cde129, 2026-10-11)

`qwen3-8b-aws-g6e` ran from 2026-10-10 22:50 to 2026-10-11 02:14 UTC, as planned above with
3 passes:

- **Outcome:** exit 0, **$6.39** recorded (planned $8.90, cap $12.50), 3.39 h of a 6.5 h
  TTL, 30 of 30 load runs completed, 3 phase0-strict passes per cell, one gate. No
  `quality_failed` or `divergence_failed` event.
- **Host:** one g6e.xlarge on-demand host in us-east-1c, AMI ami-028357be7d5b15c53, pinned
  from the smoke (DLAMI 20261006, driver 595.91.07, CUDA 13.2).
- **Teardown:** the runner terminated the host. Afterwards no Loom-managed instance or
  volume remained, and `bench reap --dry-run` (AWS configured) was clean.
- **AWS-side cost:** Cost Explorer is not enabled for this IAM user (`AccessDenied`). From
  the instance's launch and stop times (22:50:18 to 02:13:47) at $1.861/h plus the
  200 GB gp3 volume, AWS bills about $6.39, matching the DB.
- **Report:** [reports/8b-aws-g6e-v1](../reports/8b-aws-g6e-v1/leaderboard.md).

**Smokes.** The first attempt to get a host, `aws-smoke-g6e` a7bc3123 ($0.58), found a
bug: every load job failed in the client container with `No such file or directory (os
error 2)`. The vLLM image's huggingface_hub stores tokenizer.json and safetensors in a
hub-level blob store (`hub/blobs/<xx>/<sha256>`), outside the model folder that was
mounted. The fix mounts the whole hub cache and refuses links that leave the mount
([aws-setup.md](aws-setup.md#load-and-eval-jobs-on-the-host)). RunPod never hit this:
its jobs run on the pod itself, not in a container.

80ff5648 ($1.67) then passed every path:
- **Accrual:** on-demand at 1882918 µ$/h.
- **TTL:** shutdown armed (lead 19.8 s).
- **Engines:** cold start 10.2 min; warm restart onto FP8 3.2 min.
- **Search:** both searches failed 12 req/s (69 stragglers aborted at the drain), then:
  - bf16 failed 4, passed 1.333 and bisected at 2.309.
  - fp8-kv8 passed 4, then bisected at 6.928 and 5.264.
- **Quality:** 3 passes per cell, with divergence captured on bf16 and scored on fp8.
- **Gate:** one gate event.

Smoke spend: $2.25.

**Capacity.** On-demand g6e.xlarge capacity in us-east-1 comes and goes. Four of the
seven launch attempts (4efe27b6, 5d1a39c3, 96e8f54c, f16cefe2) got
`InsufficientInstanceCapacity` in all four AZs that offer the type, before any host
existed ($0). A try 1-11 min later succeeded each time. RunInstances is not retried
inside a run, so a launcher retried only that $0 no-host failure, every 10 min.

Cost at the SLO (TTFT p95 1 s, TPOT p95 50 ms), on-demand $1.8829/h with storage, both
goodputs trusted; $/1M in and out under the prefill-time split, with 95% CIs in the report:

| Cell | chat goodput (fails at) | $/1M in / out / blended | Trusted | Gate vs bf16 |
|---|---|---|---|---|
| bf16 | 2.213 (2.449) | 0.0917 / 0.4417 / 0.1765 | yes | reference |
| fp8-kv8 | 5.239 (5.511) | 0.0937 / 0.0393 / 0.0799 | yes | **FAIL** (ifeval) |

fp8-kv8's gate against bf16 (3 passes a side; margins gsm8k 1, ifeval 2, json_schema 3,
tool tasks 6 pts):

| gsm8k | ifeval | tool_calling | tool_calling_strict | json_schema | KL / top-1 | sanity (trunc / rep / drift) |
|---|---|---|---|---|---|---|
| -0.30 [-1.24, +0.66] | **-2.03** [-4.37, +0.25] | 0.00 [-2.22, +2.22] | 0.00 [-2.22, +2.22] | -1.11 [-2.67, +0.22] | 0.0102 / 95.96% (limits 0.05 / 94.79%) | 1.15 / 1.62 / 0.82% |

**The gate fails.** ifeval's point delta (−2.03 pts) is beyond its 2-point margin, which
the gate counts as a FAIL whatever the CI. The CI still includes 0, so this is not a
significant drop. gsm8k is inconclusive (its CI crosses −1). Divergence, sanity and the
tool and JSON tasks pass. fp8-kv8 is unranked, and BF16 is the only AWS row. Since 8B
ships on BF16 regardless (the user's decision), nothing changes in `config/models.yaml`.

BF16's scores on AWS are gsm8k 0.902, ifeval 0.826, json_schema 0.911, tool_calling and
tool_calling_strict 0.978.

**Why the ifeval verdict moved** (+0.43 on RunPod 7a8237d0, −2.03 here). The two runs score
the same 541 items, so the difference is the environment, not the item sample. Per item:

- **On AWS every pass repeated exactly.** bf16 had 0 items unstable across its 3 passes on
  gsm8k and ifeval, against 16 and 31 on RunPod.
- **Across environments, fp8-kv8's ifeval moved.** 37 items scored differently, a net
  −1.97 pts, while bf16 moved +0.49.
- **gsm8k barely moved.** bf16 −0.23, fp8-kv8 −0.25; paired delta −0.28 on RunPod against
  −0.30 here.

So FP8 + FP8-KV ifeval outputs depend on the host (driver 580 / CUDA 13.0 on RunPod,
595 / 13.2 here, and the batch mix), and in-run replicates cannot see that. They repeat
the same numerics. The replicate arithmetic above assumed run noise that this host
does not have. Pooling both environments informally (6 passes a side) gives ifeval −0.80 ±
1.82 pts and gsm8k −0.29 ± 0.82, which is still inconclusive on both. This is an argument,
not a gate. Settling FP8 + FP8-KV takes more items (above) and more than one environment,
not more passes.

**RunPod 7a8237d0 vs AWS: an AWS-instance deployment comparison**
(`bench compare 7a8237d0 95cde129 --match-by cell_key`). This is not a pure GPU
comparison. The g6e.xlarge has 4 vCPU and 32 GiB, with the load client on the same
host, against the RunPod pod's 16 vCPU and 94 GiB. Both runs used the same image, vLLM
args, checkpoints, profile, windows and seeded schedule.

- **Same search path.** Each search visited exactly 7a8237d0's loads (bf16 4.5, 3, 2,
  2.449, 2.213; fp8-kv8 4.5, 6.75, 5.511, 4.98, 5.239), so the knees are identical:
  bf16 2.213 (fails at 2.449), fp8-kv8 5.239 (fails at 5.511).
- **Latency within normal variance.** This holds at all 10 matched (cell, load) points,
  on every metric (0 outside the 10% tolerance or significant by Welch's t). bf16's TTFT
  p95 is 4-6% lower on AWS, TPOT within ±1%; fp8-kv8's TPOT is 0.4-3.5% higher, TTFT
  −3% to +4%; none of these is significant. Throughput at equal load is identical by
  construction (same seeded arrivals and output lengths).
- **Compare exits 0.** The config hashes differ (hardware is part of the hash): the
  provider, cloud, region, instance type, disk and RunPod's GPU type id and CUDA pin.
  `compare` lists these under "Config differences (not judged)" and its verdict reads
  "within normal variance (10 load points matched; 8 unmatched, not judged; configs differ
  in 2 matched sweeps, not judged)". Before the fix on 2026-10-10 it counted a hash
  difference as "outside" under `--match-by cell_key` and exited 6 with 0 metrics outside.
- **GPU (from the runs' nvidia-smi samples).** Utilisation is the same: bf16 97.0% on
  both at 2-3 req/s, fp8-kv8 94.5% on both. KV-cache use and preemptions match too (bf16
  at 4.5 req/s: 91 preemptions on both). AWS's board draws 45-70 W less at every load
  (bf16 at 2.213 req/s: 246 W against 302 W) for the same latency.
- **4 vCPU is enough for this load.** The load client never saturated (0
  `client_saturated_count` on all 30 runs).
- **Cost differs only through the hourly price:** $1.8829/h against RunPod's $1.1010/h,
  1.71×. bf16 costs $0.1765 per 1M blended against $0.1032, and fp8-kv8 $0.0799 against
  $0.0467. At chat's mix bf16 on AWS costs 0.88× the cheapest public listing ($0.20
  blended; aggregator or unverified, so not flagged).

**Plan against measured.**

| Step | Planned | Measured |
|---|---|---|
| Cold start | 14.5 min | 9.5 min |
| Warm restart onto FP8 | 5.4 min | 3.1 min |
| bf16 chat search | 1.56 h | 74.7 min |
| fp8-kv8 chat search | 1.56 h | 72.1 min |
| bf16 eval (3 passes) | 42.1 min | 26.3 min |
| fp8-kv8 eval (3 passes) | 27.7 min | 17.7 min |
| Total | 4.73 h, $8.90 | 3.39 h, $6.39 |

The planner's eval estimate is again about 1.6x the measured one.
