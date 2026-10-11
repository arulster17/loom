# Phase 0 acceptance checklist

The brief (§7.7) says Phase 0 is complete when you have tested it yourself and approved
it against the acceptance criteria in §7.8. This page lists each §7.8 criterion and
deliverable, plus the §7.7 outputs and the §7.5 optimizer, with:

- what the brief asks;
- the status: **met**, **partly**, **not met**, or **changed** (a decision recorded in
  [PLAN.md](PLAN.md#decisions-so-far));
- the evidence: files, tests (`path::name`), run ids and reports;
- a command you can run on your laptop for free, and what it should print. Where only a
  paid run can show something, this page says so and names the past run that showed it.

Snapshot: 2026-10-10, commit `efaf62a`. Every command below was run on that commit; the
expected output shown (abbreviated) is what it printed. Nothing on this page launches a
RunPod pod or an EC2 instance. Writing this page found three defects, fixed the same day:
`bench compare` counted unmatched cells against its verdict (A1), ids had to be given in
full (A3), and `bench site export` defaulted to the newest experiment of each name (D8).
The D6 report is now committed. The commands in A1, A3, D6 and D8 were re-run after
those fixes.

## Setup

Run from the repo root. `ACC` is a scratch directory for mock results.

```bash
cd /Users/mathurs/Projects/loom
uv sync
ACC=$(mktemp -d); echo $ACC
MOCK=sqlite:///$ACC/mock.db                                  # mock runs go here
REAL=sqlite:////Users/mathurs/Projects/loom/results/loom.db  # the real results DB
```

Which database each command touches:

- `$MOCK`: a throwaway SQLite file. Mock runs never go into the real DB, so they cannot
  mix into its reports.
- `$REAL`: only read here (`bench report`, `export`, `compare`, `competitiveness`,
  `site export`, `reap --dry-run`). These commands run the schema migration first, which
  does nothing because the DB is already at head. The file's checksum was the same
  before and after every command on this page.
- `bench quality gate` saves a new gate decision, and reports use the latest one. So run
  it on a copy: `cp results/loom.db $ACC/copy.db` and `COPY=sqlite:///$ACC/copy.db`.

The whole test suite (about 3.5 minutes):

```bash
uv run pytest -q -n auto
# 1466 passed, 20 skipped   (19 of the 20 skips are the Postgres tests; see D5)
```

## Summary

| # | Item | Status |
|---|---|---|
| A1 | One command reproduces a benchmark and its report from a provenance record | **partly**: works on the mock; never run on a GPU |
| A2 | The quality gate blocks a deliberately broken config, in a test | **met** |
| A3 | The report states $/1M tokens at SLO with confidence intervals | **met** |
| A4 | A hard budget abort works | **met** on the mock; never triggered on a paid run |
| A5 | The TTL reaper removes orphaned resources | **partly**: AWS Lambda live, no scheduled RunPod reaper |
| D1 | `bench` CLI: run, report, compare | **met** |
| D2 | Load generators | **met** (only the native one has run on a GPU) |
| D3 | Workload profiles | **met** as profiles; 4 of 10 have run on a GPU |
| D4 | Quality gate with pinned eval subsets per model | **met**; the long-context, MMLU-Pro and code tasks have never run on a GPU |
| D5 | Provenance in Postgres + CSV export | **met**; the real results live in SQLite, not Postgres |
| D6 | Model A leaderboard: vLLM vs SGLang, one GPU type, AWS | **changed**: engines compared on RunPod 1x L40S (`reports/8b-engines-v1`); the AWS g6e.xlarge leaderboard ran 2026-10-11 (95cde129, `reports/8b-aws-g6e-v1`), vLLM only |
| D7 | Model B leaderboard: one tensor-parallel config | **changed**: RunPod 2x H100 SXM, TP=2 |
| D8 | Public results page + waitlist | **partly**: site built, not published; no waitlist backend |
| D9 | Docs | **met** |
| O1 | Leaderboard report (md + HTML + CSV) | **met** |
| O2 | Competitiveness view | **partly**: costs and market prices shown; no prices or margins yet |
| O3 | Cloud comparison report | deferred to Phase 2 by the brief |
| P1 | §7.5 ranked table by $/1M at SLO, subject to the gate | **met** |
| P2 | §7.5 winner into `config/models.yaml` | **not met** |

## Acceptance criteria (§7.8)

### A1. Reproduce from a provenance record

> "one command reproduces a benchmark and report from a provenance record within normal
> variance"

**Status: partly.** `bench reproduce` works from a run id or from a `provenance.json`, and
it reports whether the re-run is within normal variance. It has only been run on the
mock, never on a GPU run.

Evidence:

- Code: `runner.reproduce` (`bench/src/loom_bench/runner.py`). It rebuilds the cell from
  the stored config (not today's registry), checks the derived seed, warns on a different
  git sha or a dirty tree, re-runs that load point and repetition as a new experiment
  `<name>--reproduce`, and compares it with the original (exit 0 within variance, exit 6
  outside). It plans and enforces the budget caps like `bench run`. Docs:
  [benchmark-lab.md](benchmark-lab.md#provenance-and-reproduce).
- Tests: `bench/tests/runner/test_e2e.py::test_reproduce_a_run_from_its_provenance`
  (by run id and by `provenance.json`), and
  `bench/tests/runner/test_e2e.py::test_reproduce_reports_a_real_difference` (exit 6
  when something really changed).
- On real hardware: the results DB has no `--reproduce` experiment. The closest real
  evidence is the same BF16 config hash (`7a203a3f…`) measured on two pods, two to three days
  apart: 565b8d3f (2026-10-07) and the config sweep 7a8237d0 (2026-10-09/10). At their one
  shared point (fixed-1k-1k, 1 req/s), every metric is within normal variance (command
  below).

Check it yourself (free, about 4 minutes). This uses a one-config mock experiment that runs
the simulated GPU in real time (30 ms decode steps, 30 s windows, 3 repetitions):

```bash
cat > $ACC/mock-reproduce.yaml <<'EOF'
name: mock-reproduce
description: One mock config at real-time speed, with windows long enough to reproduce.
model: qwen3-8b
provider:
  kind: mock
  hourly_price: "$1.00"
  time_scale: 1.0
  step_base_ms: 30
  seed: 0
variants:
  - name: baseline
workloads:
  - profile: fixed-128-128
    load: {mode: open_loop, values: [8], duration_s: 30, warmup_s: 5, drain_timeout_s: 15}
repetitions: 3
slo: {ttft_ms: {p95: 1000}, tpot_ms: {p95: 50}, max_error_rate: 0.01}
budget: {max_spend: "$1", ttl_minutes: 10}
EOF
uv run bench run $ACC/mock-reproduce.yaml --db $MOCK --out $ACC/results
#   baseline / fixed-128-128 @ 8 rep 0: p95 TTFT ~98 ms, ~1,096 out tok/s   (rep 1, rep 2 similar)
#   completed experiment <id>: 3 runs, spent $0.03
EXP=$(ls -t $ACC/results | head -1)      # the experiment just run
RUN=$(ls $ACC/results/$EXP/runs | head -1)
uv run bench reproduce $RUN --db $MOCK --out $ACC/results; echo "exit=$?"
#   completed experiment <new id>: 1 runs ...
#   Comparison: original vs reproduction
#   Verdict: within normal variance
#   ... per-metric table, every row "within" ...
#   reproduced within normal variance
#   exit=0
uv run bench reproduce $ACC/results/$EXP/runs/$RUN/provenance.json --db $MOCK --out $ACC/results
#   same, from the provenance file: "reproduced within normal variance"
REPRO=$(ls -t $ACC/results | head -1)
uv run bench report -e $REPRO --db $MOCK --out $ACC/repro-report
#   wrote .../leaderboard.{md,html,csv,...}: the report of the reproduction
#   (one repetition, so its figures are marked untrusted)
```

All three runs reproduced within normal variance when this was checked. Plain
`bench report` (without `-e`) leaves reproductions out on purpose, so name the
reproduction with `-e`.

Don't use `mock-smoke`'s runs for this. They are plumbing checks: 0.6 s windows of
12-request batches at 100x speed, so latencies are a few milliseconds and laptop
scheduling jitter moves them by 50% or more. In one of two tries, a reproduction of one
of its runs came out "outside normal variance" (exit 6). That is because the original
point was itself unstable, not because of a reproduction fault: its two repetitions
disagreed so much that the 95% CI on its TTFT p95 ran from 0.00003 ms to 12.6 million ms. `test_reproduce_a_run_from_its_provenance` avoids this
by replaying a fixed point.

The real-hardware comparison (read-only):

```bash
uv run bench compare 565b8d3f 7a8237d0 --db $REAL --out $ACC/cmp; echo "exit=$?"
#   Verdict: within normal variance (1 load point matched; 22 unmatched, not judged)
#   vllm vs bf16: fixed-1k-1k @ 1 req/s — within
#     TTFT p95 298 vs 256 ms (-14%, not significant, Welch p=0.093); TPOT p95 44.0 vs 43.7 ms;
#     output tok/s 1,031 vs 1,068 (+3.5%)
#   Only in 565b8d3f-...: sglang code-completion open_loop; ...; vllm fixed-1k-1k @ 2 req/s
#   Only in 7a8237d0-...: fp8-kv8-mbt1024 chat-sharegpt open_loop; ...
#   exit=0
```

The two experiments share only that one point. `compare` judges the cells and loads both
sides ran and lists the other 22 without counting them (before the fix on 2026-10-10 it
counted them and exited 6 here). Exit codes: 0 within normal variance, 6 when a matched
metric is outside, 2 when nothing matched at all (try `--match-by cell_key` or
`workload`). Test: `bench/tests/runner/test_cli_ids.py::test_compare_exit_judges_only_the_cells_both_ran`.

Reproducing a GPU run is paid. `bench reproduce <run id> --db $REAL` launches a pod
straight away: it has no `--dry-run`, although it does refuse to run over the caps. The
cost is one cold start plus one load point.

### A2. The quality gate blocks a broken config

> "the quality gate blocks a deliberately broken config (e.g. over-aggressive
> quantization) in a test"

**Status: met.**

Evidence:

- `bench/tests/quality/test_gate_acceptance.py::test_gate_blocks_over_aggressive_quantization`:
  the mock backend with `degrade: 0.3` (30% of answers wrong or JSON truncated) plus heavy
  logprob noise. The gate FAILs it on arithmetic, JSON validity and divergence, and
  blocks it. The same file also has `test_identical_config_passes`,
  `test_small_sample_is_inconclusive`, `test_engine_noise_passes_against_the_measured_floor`,
  `test_drift_beyond_a_bit_exact_floor_needs_review` and
  `test_results_and_decision_persist`.
- `bench/tests/runner/test_quality_e2e.py`: PASS, REVIEW and FAIL candidates through a
  mock experiment, `bench quality gate` (exit 7 on FAIL) and the report
  (`test_report_ranks_the_gate_failed_config_last`).
- Method: [quality-gate.md](quality-gate.md#decision-rule) and
  [its acceptance test section](quality-gate.md#acceptance-test).
- On real runs, the gate has blocked configs that were unproven, not broken. All four
  8B sweep candidates (7a8237d0) and the 70B FP8 row's first run (9f0853d7) came out
  INCONCLUSIVE and blocked. The 70B FP8 row then passed on the 135-item tool-calling set
  (9e5f7866).

Check it yourself (free):

```bash
uv run pytest -v bench/tests/quality/test_gate_acceptance.py bench/tests/runner/test_quality_e2e.py
#   ...::test_gate_blocks_over_aggressive_quantization PASSED ... 12 passed

cp results/loom.db $ACC/copy.db; COPY=sqlite:///$ACC/copy.db
# The 70B FP8 row against BF16, re-decided from the stored per-item samples of 9e5f7866:
uv run bench quality gate --db $COPY \
  --baseline 2f7fb3f9372ba09550198b14978459e35c68866db852f44d4856c293a5adf8f9 \
  --candidate 0381825d763025fdf1752037fbd925abe0a992e191215227ccb01e48039e5afd; echo "exit=$?"
#   gate PASS
#     gsm8k: delta +0.08 pts [-0.61 pts, +0.76 pts], n=1319: non-inferior at 1.00 pts
#     ... ifeval, tool_calling, tool_calling_strict, json_schema, divergence, sanity
#   exit=0
# The 8B fp8-kv8 candidate against BF16 (7a8237d0):
uv run bench quality gate --db $COPY \
  --baseline 7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a \
  --candidate c5e726712cab37cfff7e0ae8108575dc26ec5c7883c835e53149eddac42a4028; echo "exit=$?"
#   gate INCONCLUSIVE (blocked)
#     gsm8k: delta -0.28 pts [-1.14 pts, +0.61 pts], n=1319: CI crosses -1.00 pts, more samples needed
#   exit=7
```

### A3. $/1M tokens at SLO with confidence intervals

> "the report states $/1M tokens at SLO with confidence intervals"

**Status: met.**

Evidence:

- Reports in git: [reports/8b-config-sweep-v1](../reports/8b-config-sweep-v1/leaderboard.md)
  and [reports/70b-h100-v2](../reports/70b-h100-v2/leaderboard.md). Every cost cell has a
  95% CI, e.g. 8B BF16 on chat-sharegpt: $0.0557 [0.0486, 0.0638] per 1M input,
  $0.2517 [0.2049, 0.3060] per 1M output, $0.1032 [0.0967, 0.1099] blended. The CSVs carry
  `*_lo_micros` / `*_hi_micros` columns.
- Method: [cost-model.md](cost-model.md). Log-scale t-intervals across 3 repetitions;
  input and output split by measured prefill time.
- Tests: `bench/tests/analysis/test_cost.py::test_ci_maps_high_throughput_to_low_cost`,
  `bench/tests/analysis/test_cost_split.py::test_ci_is_an_outer_bound_of_the_separate_intervals`,
  `bench/tests/report/test_leaderboard.py::test_markdown_table_and_footer` (checks cells
  like `$0.2906 [0.2834, 0.2979]`).
- Caveats the reports state themselves. Some figures are "untrusted" (run-to-run CV
  above 10%), e.g. 70B BF16 on H100 fixed-1k-1k (TTFT p95 CV 21.8%) and 8B
  code-completion in 565b8d3f. The 70B on 4x L40S has no cost at SLO, because no tested
  load met the SLO.

Check it yourself (free; regenerates the committed reports from the real DB):

```bash
uv run bench report -e 7a8237d0-9917-47e7-b93e-8cb0230e0059 --db $REAL --out $ACC/8b-sweep
diff -q reports/8b-config-sweep-v1/leaderboard.md $ACC/8b-sweep/leaderboard.md && echo identical
diff -q reports/8b-config-sweep-v1/leaderboard.csv $ACC/8b-sweep/leaderboard.csv && echo identical
#   identical / identical
uv run bench report -e 9f0853d7-e708-43b2-a16f-a4e9e48cf473 -e 9e5f7866-f2f3-4672-bc05-925fe270271c \
  --db $REAL --out $ACC/70b-h100
diff reports/70b-h100-v2/leaderboard.md $ACC/70b-h100/leaderboard.md
#   one line differs: "Price book last checked: 2026-10-08" -> "2026-10-09"
#   (bench/prices.yaml was re-checked after the report was committed; no number changes)
diff -q reports/70b-h100-v2/leaderboard.csv $ACC/70b-h100/leaderboard.csv && echo identical
#   identical
```

Experiment and run ids can be given in full or as a unique prefix of at least 4 hex
digits, like git's short hashes: `bench report -e 7a8237d0` is the same as the full id
above. An ambiguous prefix is refused with the ids it matches, and an unknown one with
"no experiment ..." (exit 2). This holds for every command that takes an id: `report`,
`competitiveness`, `compare`, `export`, `site export`, `reproduce` (run ids) and
`quality gate` (experiment ids or config hashes). Tests:
`bench/tests/runner/test_cli_ids.py`.

### A4. Hard budget abort

> "a hard budget abort works"

**Status: met on the mock; never triggered on a paid run.**

Evidence:

- `bench/experiments/mock-budget-abort.yaml`: a mock host at a simulated $3,600/h ($1/s)
  serving one sequence at a time, so recorded spend reaches the $8 cap mid-run.
- `bench/tests/runner/test_e2e.py::test_hard_budget_abort`: exit 5, experiment
  `aborted` with "budget cap reached", the in-flight run cancelled, the host torn down,
  and the overshoot bounded by one accrual interval plus teardown.
- `bench/tests/runner/test_budget.py` (`test_trips_when_spend_reaches_the_cap`,
  `test_trip_cancels_the_guarded_call`, `test_projected_spend_stops_gracefully`,
  `test_overall_cap_counts_other_billable_experiments`, ...).
- The five rails (caps, planner refusal, guard, host self-TTL, reaper):
  [benchmark-lab.md](benchmark-lab.md#budget-rails).
- Real runs: no paid experiment has hit its cap (none is `aborted` in the DB). Total
  recorded spend is $78.74 against the $149.75 overall cap.

Check it yourself (free):

```bash
uv run bench run bench/experiments/mock-budget-abort.yaml --db $MOCK --out $ACC/results; echo "exit=$?"
#   (plan: estimated spend $7.66, effective cap $8.00)
#   aborted experiment <id>: 1 runs, spent $8.30   (varies by a few cents)
#   reason: budget cap reached: spent $8.10 >= cap $8.0000
#   exit=5
```

The ~$0.30 past the cap is about a quarter-second of accrual plus teardown at the
simulated $1/s, within the documented worst case. At real prices that's about a cent
(benchmark-lab.md, rail 3).

### A5. TTL reaper

> "the TTL reaper removes orphaned resources" (§7.6: "tag every cloud resource; TTL
> reaper that tears down leftover benchmark infrastructure automatically")

**Status: partly.** On AWS, the reaper runs automatically and is live. On RunPod, which
is the provider every real run used, it is a manual command. The automatic backstop there
is the in-pod TTL watchdog, and a pod stuck before its container starts has no watchdog.

Evidence:

- AWS: `loom-bench-reaper` Lambda (`infra/aws/bench/reaper.tf`, code
  `bench/src/loom_bench/providers/aws_reaper.py`) on a 15-minute EventBridge schedule.
  It can only terminate `loom:managed=true` resources. Checked 2026-10-10: `Active`,
  `LOOM_REAPER_DRY_RUN=false`, schedule `ENABLED`, `rate(15 minutes)`. A dry-run invoke
  returned `{"reaped": [], "dry_run": true}`. No AWS GPU host has run yet (the spot
  quota is 0), so it has never had a real orphan to remove.
  Tests: `bench/tests/aws/test_reaper.py::test_reap_terminates_only_expired_managed`,
  `::test_reap_deletes_expired_unattached_volumes`, `::test_lambda_handler` (moto), and
  `bench/tests/aws/test_terraform.py::test_reaper_lambda_points_at_the_module`.
- RunPod: `bench reap` terminates pods both named `loom-bench-…` and carrying
  `LOOM_MANAGED=true` once past their TTL or exited, plus expired pods recorded in the
  DB. Each pod also runs a TTL watchdog that terminates it at an absolute time.
  Tests: `bench/tests/runpod/test_runpod_reaper.py::test_reap_terminates_only_expired_managed_pods`,
  `bench/tests/runpod/test_runpod_e2e.py::test_reap_terminates_recorded_and_unrecorded_managed_pods`.
  Code: `bench/src/loom_bench/providers/runpod_reaper.py`. In real runs, all 31 recorded
  pods were terminated by the runner (`terminated_by: runner` in `bench_resources`). The
  teardown check, a clean `bench reap --dry-run`, is recorded for the smoke runs and the
  70B quality rerun ([runbook-runpod.md](runbook-runpod.md#5-teardown-verification)).
- Any provider, from the DB: `bench/tests/runner/test_e2e.py::test_reaper_removes_expired_resources_of_a_dead_runner`
  (a mock host left by a "dead" runner is listed by `--dry-run`, then killed by `bench
  reap`, and its row is marked `terminated_by: reaper`; an unexpired one is left alone).

Check it yourself (free):

```bash
uv run pytest -v bench/tests/runner/test_e2e.py::test_reaper_removes_expired_resources_of_a_dead_runner \
  bench/tests/aws/test_reaper.py bench/tests/runpod/test_runpod_reaper.py
#   18 passed
uv run bench reap --dry-run --db $REAL
#   AWS not configured: EC2 instances are not reaped     (unless LOOM_AWS_CONFIG is set)
#   no expired resources
# With your RunPod key in the Keychain, that dry run also listed the account's pods: none expired.
aws lambda get-function-configuration --region us-east-1 --function-name loom-bench-reaper \
  --query '{state: State, env: Environment.Variables}'
#   "state": "Active", "LOOM_REAPER_DRY_RUN": "false", "LOOM_REAPER_MAX_AGE_HOURS": "24"
aws events describe-rule --region us-east-1 --name loom-bench-reaper --query '{state: State, schedule: ScheduleExpression}'
#   "state": "ENABLED", "schedule": "rate(15 minutes)"
aws lambda invoke --region us-east-1 --function-name loom-bench-reaper \
  --cli-binary-format raw-in-base64-out --payload '{"dry_run": true}' /dev/stdout
#   {"reaped": [], "dry_run": true} ... "StatusCode": 200
```

To see the reaper remove a real orphan, you would have to leave a paid host running past
its TTL. That has not been done.

## Deliverables (§7.8)

### D1. `bench` CLI: `run`, `report`, `compare`

**Status: met.** `bench run`, `bench report` and `bench compare` exist, plus `plan`,
`reproduce`, `reap`, `export`, `competitiveness`, `quality run|gate|job`,
`site export|build`, `waitlist count`, `db upgrade` and `mock-server`
([benchmark-lab.md](benchmark-lab.md#commands)). Code: `bench/src/loom_bench/cli.py`.
Error paths: `bench/tests/runner/test_cli_errors.py`.

```bash
uv run bench --help                 # lists the commands above
uv run bench plan bench/experiments/mock-smoke.yaml
#   plan table; estimated spend $0.01; effective cap $1.00
# run, report, compare: see A1 and A3.
```

### D2. Load generators

> "wrap, do not reinvent: ... vLLM (`benchmark_serving`) and SGLang (`bench_serving`) ...
> and a thin engine-neutral client (OpenAI-compatible streaming)"

**Status: met.** Only the native client has run on a GPU.

- Native client: `bench/src/loom_bench/loadgen/native.py` (httpx + hand-rolled SSE;
  open loop and closed loop). Every real run used it, because the `aws_ec2` and `runpod`
  providers accept `loadgen: native` only.
- Wrappers: `loadgen/vllm_bench.py` (`vllm bench serve`) and `loadgen/sglang_bench.py`
  (`sglang.bench_serving`). They are tested against fake tools and recorded fixtures
  (`bench/tests/loadgen_external/`), and usable with the `local` provider. Never run on a
  real GPU run.
- Arrivals (`loadgen/arrivals.py`): constant, Poisson, gamma, on/off bursts, ramp,
  diurnal, trace. Real runs used Poisson (open loop).
- Docs: [load-generators.md](load-generators.md).

Check it yourself (free; the native client against the mock server, through the `local`
provider):

```bash
uv run bench mock-server --port 8765 &
cat > $ACC/local-mock.yaml <<'EOF'
name: local-mock
description: Benchmark a running bench mock-server through the local provider.
model: qwen3-8b
provider:
  kind: local
  base_url: http://127.0.0.1:8765/v1
  metrics_url: http://127.0.0.1:8765/metrics
  engine: mock
  served_model: mock-model
  tokenizer: simple
variants:
  - name: as-running
workloads:
  - profile: fixed-128-128
    load: {mode: closed_loop, values: [1, 4], num_requests: 8, warmup_requests: 2}
repetitions: 2
slo: {ttft_ms: {p95: 1000}, tpot_ms: {p95: 50}, max_error_rate: 0.01}
budget: {max_spend: "$1", ttl_minutes: 30}
EOF
uv run bench run $ACC/local-mock.yaml --db sqlite:///$ACC/local.db --out $ACC/results-local
#   as-running / fixed-128-128 @ 1 rep 0: p95 TTFT ~20 ms, ~140 out tok/s ...
#   completed experiment <id>: 4 runs, spent $0.0000
uv run pytest -q bench/tests/loadgen bench/tests/loadgen_external
#   100 passed
kill %1        # stop the mock server
```

### D3. Workload profiles

**Status: met as profiles.** All of the brief's shapes exist in `bench/workloads/`.
Four have run on a GPU.

| Brief | Profile | Run on a GPU |
|---|---|---|
| fixed 128/128, 1k/1k, 8k/1k, 32k/1k | `fixed-128-128`, `fixed-1k-1k`, `fixed-8k-1k`, `fixed-32k-1k` | 1k/1k only (all real runs); 128/128 in the mock smoke |
| chat (ShareGPT-style), license recorded | `chat-sharegpt` (pinned at commit `192ab218`, sha256-checked; license note in the profile and in each run's provenance) | yes (7a8237d0) |
| shared prefix, 0-95% knob | `shared-prefix` (`prefix_share`, default 0.5) | yes (565b8d3f, 9f0853d7, 55102ddb) |
| long context (needle) | `long-context-needle` | no |
| code completion | `code-completion` | yes (565b8d3f) |
| long generation (~2k out) | `long-generation` | no |
| trace replay | `trace-azure-code` | no |
| arrival patterns | Poisson, gamma, on/off bursts, ramp, diurnal, trace (`arrival:` in the experiment) | Poisson only |

The ShareGPT license note says the card's Apache-2.0 does not settle OpenAI's terms for
the scraped conversations. Loom uses them only as request shapes and publishes only
latency and cost; the profile says "Verify before publishing". Tests:
`bench/tests/workloads/test_generate.py` (e.g.
`::test_shared_prefix_honours_share_and_groups`,
`::test_every_generative_shipped_profile_builds`). How to add one:
[how-to/add-workload-profile.md](how-to/add-workload-profile.md).

```bash
uv run pytest -q bench/tests/workloads
#   20 passed
```

### D4. Quality gate with pinned eval subsets per model

**Status: met**, with a scope note.

- Suites: `bench/evals/qwen3-8b.yaml` and `bench/evals/llama-3.3-70b-instruct.yaml`,
  each with pinned items and seed, gate settings (1-point default margin, per-task
  margins where the set is small), divergence limits, sanity limits, and subsets
  `phase0` (GSM8K, IFEval, tool calling, JSON schema) and `phase0-strict` (plus strict
  tool calling). Divergence (KL and top-1 agreement against the BF16 reference) and the
  sanity checks always run.
- Scope note: MMLU-Pro, RULER / needle (long context) and code (HumanEval / MBPP) are in
  the suites but have never run on a GPU. They were left out of the Phase 0 subsets for
  time (about 1.8 h per config for the full suite). Their CIs are unbounded until a
  full-suite run.
- Real gate results: the 70B FP8 row passes against BF16 (9e5f7866). SGLang vs vLLM on 8B
  is inconclusive on IFEval (b03b3c52: -0.80 pts [-2.03, +0.37] against a 2-point
  margin). All four 8B sweep candidates are inconclusive (7a8237d0).
- Docs: [quality-gate.md](quality-gate.md).

```bash
uv run bench mock-server --port 8765 &
uv run bench quality run bench/experiments/suites/mock-smoke.yaml \
  --base-url http://127.0.0.1:8765/v1 --model mock-model --out $ACC/samples.json
#   Suite mock-smoke on mock-model: arithmetic n=60 1.000 [1.000 .. 1.000]; json_schema n=300 0.760 [0.711 .. 0.809]
kill %1
uv run pytest -q bench/tests/quality
#   208 passed, 1 skipped, 8 deselected (network-marked tests are deselected by default)
```

### D5. Provenance records in Postgres + CSV export

> "provenance records in Postgres + CSV export"

**Status: met.** One gap from the letter of the brief: **the real results live in SQLite**
(`results/loom.db`), not Postgres.

- The store is SQLAlchemy 2 + Alembic (`bench/src/loom_bench/store/`) and runs on Postgres
  or SQLite. CI runs the Postgres tests against Postgres 16. Per-request rows are Parquet
  (`results/<experiment>/runs/<run>/requests.parquet`). Schema:
  [PLAN.md](PLAN.md#postgres-schema).
- Each run's provenance (`bench_runs.provenance` and `provenance.json`) has: git sha,
  dirty flag and branch; bench and loadgen versions; engine, version, image and digest;
  CUDA and driver; model repo, revision and quantization; parallelism; GPU type and
  count; cloud, region, instance type, market; the as-run hourly price and its source;
  the config hash and full config; the workload and its content kind; dataset and
  license; load mode, value and seed; repetition; the client host. Example: the
  7a8237d0 runs record CUDA 13.0, driver 580.126.09, vLLM image digest `sha256:8a69ffad…`,
  Qwen3-8B revision `b968826d…`.
- Tests: `bench/tests/store/test_provenance.py`, `bench/tests/store/test_export.py`,
  `bench/tests/store/test_migrations.py::test_migrations_match_models` (runs on SQLite
  and on Postgres),
  `bench/tests/runner/test_e2e.py::test_smoke_runs_have_complete_provenance_and_summaries`.

Check it yourself (free):

```bash
uv run bench export csv --db $REAL --out $ACC/runs.csv
#   wrote 595 runs to .../runs.csv   (columns provenance.git.sha, provenance.engine.image_digest, ...)

# Postgres (needs Docker): the compose Postgres, a throwaway database
docker compose up -d postgres
docker compose exec postgres createdb -U loom loom_acc
PG=postgresql+psycopg://loom:loom@localhost:5432/loom_acc
uv run bench run bench/experiments/mock-smoke.yaml --db $PG --out $ACC/results-pg
#   completed experiment <id>: 24 runs
docker compose exec postgres psql -U loom -d loom_acc -c \
  "select cell_key, workload, load_value, repetition, provenance->'engine'->>'name' as engine, provenance->>'config_hash' as config_hash from bench_runs limit 3"
#   baseline | fixed-128-128 | 10 | 0 | mock | f73977745a71...
uv run bench export csv --db $PG --out $ACC/runs-pg.csv
#   wrote 24 runs
docker compose exec postgres dropdb -U loom loom_acc
LOOM_TEST_DATABASE_URL=postgresql+psycopg://loom:loom@localhost:5432/loom uv run pytest -q -m postgres
#   19 passed   (each test creates and drops its own schema)
```

### D6. Model A leaderboard: vLLM vs SGLang, one GPU type, AWS

**Status: changed** (PLAN.md decision, 2026-10-05: RunPod Secure Cloud, because the AWS
GPU spot quota is 0). The vLLM-vs-SGLang comparison ran on RunPod Secure 1x L40S, the GPU
in AWS g6e.xlarge. A Model A leaderboard on AWS itself ran on 2026-10-10/11 on one
g6e.xlarge on-demand host, with vLLM only (BF16 and fp8-kv8; SGLang was not re-run on
AWS). The spot quota is still 0; the on-demand G/VT quota (4 vCPU) fits that one host.
The user decides whether this meets D6 (open item 4).

Evidence, RunPod (engines):

- Load sweep 565b8d3f (2026-10-07, $5.70; SGLang's eval failed in that run). Quality-only
  reruns b1b904dc (1 pass, $0.51) and b03b3c52 (3 passes per engine, $1.13).
- Results: on fixed-1k-1k and shared-prefix, vLLM and SGLang tie within the search
  resolution (1 req/s and 2 req/s; $0.1483 and $0.0695 per 1M blended). On
  code-completion, vLLM holds 1.297 req/s against SGLang's 1.091, but both are untrusted
  (CV 12-15%). SGLang's gate against vLLM is inconclusive on IFEval, so SGLang is blocked
  and vLLM stays the 8B engine.
- Report: [reports/8b-engines-v1](../reports/8b-engines-v1/leaderboard.md), generated
  2026-10-10 from the real DB with the command below. It replaces the uncommitted
  `reports/8b-final-v4/` (gitignored) under a new name, so it cannot collide with that
  local copy. Against 8b-final-v4 no number changed: the price book date is 2026-10-09
  instead of 2026-10-06, and the competitiveness CSV has the newer
  `competitor_listing_model_id` column.

```bash
uv run bench report --db $REAL --out $ACC/8b-engines -e 565b8d3f -e b1b904dc -e b03b3c52
for f in md csv equal_load.csv competitiveness.csv html; do
  diff -q reports/8b-engines-v1/leaderboard.$f $ACC/8b-engines/leaderboard.$f && echo identical
done
#   identical (x5)
```

Evidence, AWS (`qwen3-8b-aws-g6e`, 95cde129, 2026-10-10 22:50 to 2026-10-11 02:14 UTC):

- Smoke first. `aws-smoke-g6e` 80ff5648 passed ($1.67) after a7bc3123 ($0.58) found a
  bug: the client container's model mount left tokenizer.json dangling (fixed in
  `aws_ec2`, [aws-setup.md](aws-setup.md#load-and-eval-jobs-on-the-host)). Three more
  attempts found no on-demand capacity in any AZ and launched nothing ($0).
- Run: exit 0, $6.39 (DB), 3.39 h of a 6.5 h TTL, 30 of 30 load runs, 3 phase0-strict
  passes per cell, one gate. The runner terminated the host. Afterwards no Loom-managed
  instance or volume remained, and `bench reap --dry-run` was clean. AMI
  ami-028357be7d5b15c53 (DLAMI, driver 595.91.07, CUDA 13.2), us-east-1c.
- Leaderboard at the SLO, $1.8829/h, both trusted: bf16 2.213 req/s, $0.1765 per 1M
  blended. fp8-kv8 5.239 req/s, $0.0799, but its gate against BF16 **fails**: ifeval
  −2.03 pts [−4.37, +0.25] is a point drop beyond the 2-pt margin, and gsm8k is
  inconclusive. So BF16 is the only AWS row, as on RunPod.
- Cross-check against RunPod 7a8237d0: both searches visited the same loads. At all 10
  matched (cell, load) points every latency metric is within normal variance, and GPU
  utilisation is the same. `bench compare` exits 6 only because the config hashes
  differ (the cloud is part of the hash).
- Details: [benchmark-lab.md](benchmark-lab.md#qwen3-8b-on-aws-g6exlarge-result-95cde129-2026-10-11).
  Report: [reports/8b-aws-g6e-v1](../reports/8b-aws-g6e-v1/leaderboard.md), regenerated
  identically by:

```bash
uv run bench report --db $REAL --out $ACC/8b-aws -e 95cde129
for f in md csv equal_load.csv competitiveness.csv html; do
  diff -q reports/8b-aws-g6e-v1/leaderboard.$f $ACC/8b-aws/leaderboard.$f && echo identical
done
#   identical (x5)
uv run bench compare 7a8237d0 95cde129 --match-by cell_key --db $REAL; echo "exit=$?"
#   Verdict: outside normal variance (0 metrics outside; 10 load points matched; 8 unmatched)
#   exit=6: every matched point is flagged only for "Config hash differs"
```

### D7. Model B leaderboard: one tensor-parallel config

**Status: changed** (PLAN.md decisions 2026-10-05 and 2026-10-08). Llama 3.3 70B ran
first at TP=4 on 4x L40S and then at TP=2 on 2x H100 SXM, both on RunPod.

- 4x L40S, TP=4 (cf4d1614, $5.95; EAGLE3 sweep 55102ddb, $5.77): no tested load met the
  50 ms TPOT p95 SLO, so there is no cost at SLO. That host had no working GPU P2P
  ([benchmark-lab.md](benchmark-lab.md#70b-tuning)).
- 2x H100 SXM, TP=2 (9f0853d7, $25.18; quality rerun 9e5f7866, $3.29):
  - BF16 holds 0.5 req/s on fixed-1k-1k, $2.37 [1.97, 2.84] per 1M blended. This figure
    is untrusted (TTFT p95 CV 21.8%).
  - FP8 holds 2 req/s on fixed-1k-1k, $0.546 [0.532, 0.561], and 4 req/s on
    shared-prefix, $0.263 [0.239, 0.289]. Both FP8 figures are trusted.
  - The FP8 gate passes against BF16 on every task.
  - Report: [reports/70b-h100-v2](../reports/70b-h100-v2/leaderboard.md) (regenerate: A3).

```bash
uv run bench report --db $REAL --out $ACC/70b-l40s \
  -e cf4d1614-2453-4ad4-b345-afd6b769a52d -e 55102ddb-ba53-48c2-88c1-bb824577dcd6
#   the 4x L40S board: "No cost at the declared SLO: no tested load met the SLO", with the
#   TPOT p95 at the lowest load (62.2 ms) and the NCCL_P2P_DISABLE=1 note
```

### D8. Public results page + waitlist

**Status: partly.** The site is built and tested, but not published. The waitlist has no
backend, so it shows "Waitlist opens soon" and no form.

- Code: `bench/src/loom_bench/site/`. Pages: index, one per model, methodology, pricing,
  harness ([site.md](site.md)). Workflow `.github/workflows/site.yml` deploys only once
  GitHub Pages is enabled (Settings → Pages, plus `LOOM_PAGES_ENABLED=true`). That is not
  done, and PLAN.md says to hold publishing until you say so.
- The committed snapshot `site/data/` is empty (0 experiments), so the committed build
  says "No published results yet — first runs pending."
- What gets published is pinned by full id in `site/config.yaml` (`publish.experiments`,
  empty today). `bench site export` without `-e` exports exactly those, and the deploy
  workflow builds with `--require-pinned`, which refuses a `site/data` snapshot of
  anything else. There is no "newest experiment of each name" default any more: it picked
  the 70B EAGLE3 sweep 55102ddb over cf4d1614. Tests: `bench/tests/site/test_site_cli.py`,
  `test_site_snapshot.py::test_export_takes_exactly_the_named_experiments`. Docs:
  [site.md](site.md#publishing-results-after-runs).
- Waitlist: `site/config.yaml` has `waitlist.action_url: null`. The backend (Formspree /
  Buttondown, or a Loom endpoint into `waitlist_signups`) is an open question in PLAN.md.
  [waitlist.md](waitlist.md) has an empty count table.
- Tests: `bench/tests/site/` (e.g. `test_site_build.py::test_waitlist_without_endpoint_renders_no_form`,
  `::test_waitlist_form_when_endpoint_set`, `::test_populated_snapshot_renders_results`).

Check it yourself (free; builds a site from the real results into a temp dir, without
touching `site/data`):

```bash
uv run bench site export --db $REAL --out $ACC/site-none
#   wrote .../site-none: 0 experiments, 0 configs, 0 runs
#   no experiments pinned in .../site/config.yaml (publish.experiments): the snapshot has no results
uv run bench site export --db $REAL --out $ACC/site-data \
  -e 565b8d3f -e b03b3c52 -e 7a8237d0 -e cf4d1614 -e 9f0853d7 -e 9e5f7866
#   a table of the six experiments (id, name, status, created), then
#   wrote .../site-data: 6 experiments, 21 configs, 297 runs
#   these are not the experiments pinned in .../site/config.yaml (publish.experiments): use this
#   snapshot for a local preview, or pin them before committing it; ...
uv run bench site build --data $ACC/site-data --out $ACC/site
#   wrote 8 pages
uv run bench site build --data $ACC/site-data --out $ACC/site --require-pinned; echo "exit=$?"
#   cannot build the site: the snapshot does not hold the experiments pinned ... exit=2
python3 -m http.server 8000 --directory $ACC/site      # open http://localhost:8000
uv run bench waitlist count --no-record --db $REAL
#   0 signups
```

The `-e` list above is one possible selection (the 8B engines and quality rerun, the 8B
sweep, the 70B on 4x L40S and on 2x H100 with its quality rerun), not a decision. To
publish, pin your choice in `site/config.yaml` (open item 7).

### D9. Docs

**Status: met.** [docs/README.md](README.md) indexes them. Getting started:
[local-dev.md](local-dev.md). Running experiments: [benchmark-lab.md](benchmark-lab.md).
Method: [cost-model.md](cost-model.md), [quality-gate.md](quality-gate.md),
[load-generators.md](load-generators.md). Operations: [runbook-runpod.md](runbook-runpod.md),
[runbook.md](runbook.md), [aws-setup.md](aws-setup.md), [security.md](security.md).
Publishing: [site.md](site.md), [waitlist.md](waitlist.md). How-tos for adding a model,
engine, GPU type, workload or eval task, and for pricing a model: [how-to/](how-to/).

## §7.7 outputs

### O1. Per-model leaderboard report (markdown + HTML + CSV)

> "ranked configs with $/1M tokens, goodput, p95 latency, quality deltas, one-line
> recommendation"

**Status: met.** `bench report` writes `leaderboard.md`, `.html` and `.csv`, plus
`leaderboard.equal_load.csv` and `leaderboard.competitiveness.csv`. Each board has
ranked configs, $/1M input, output and blended at SLO with CIs, the goodput bracket,
p95 TTFT and TPOT at goodput, the quality delta and gate verdict, cold start, and a
generated recommendation. Above the boards is a plain-English summary per model and
workload. Reading guide: [benchmark-lab.md](benchmark-lab.md#reading-the-reports).
Tests: `bench/tests/report/test_leaderboard.py` (e.g.
`::test_ranking_cheapest_first_then_gate_failed_then_untrusted`,
`::test_recommendations_are_generated_from_numbers`,
`::test_html_is_self_contained_with_dark_mode`). Commands: A3, D6, D7.

### O2. Competitiveness view

> "my cost, my planned price and margin, next to public competitor prices"

**Status: partly.** Our cost at SLO is shown next to public list prices
(`bench/competitors.yaml`, public sources only, with source and date), at each
workload's input:output mix, with like-for-like precision marked. The break-even price
is shown too. **Our planned price and margin are not**: every `pricing` in
`config/models.yaml` is null, so the view says "no price set", and the margin flags
(`negative_margin`, `price_above_market`) cannot fire yet.

```bash
uv run bench competitiveness -e 7a8237d0-9917-47e7-b93e-8cb0230e0059 --db $REAL --out $ACC/comp
#   writes competitiveness.{md,html,csv}; our cost per side and blended, "no price set",
#   public list prices (e.g. Fireworks AI, OpenRouter for Qwen3-8B) and flags
```

Tests: `bench/tests/config/test_competitiveness.py`,
`bench/tests/report/test_competitiveness_report.py`.

### O3. Cloud comparison report

Deferred to Phase 2 by the brief (AWS vs GCP). Not built.

## §7.5 optimizer

### P1. Ranked table by $/1M at SLO, subject to the quality gate

**Status: met.** The 8B config sweep (7a8237d0, five vLLM cells on 1x L40S, chat-sharegpt
and fixed-1k-1k) is ranked by $/1M at SLO in
[reports/8b-config-sweep-v1](../reports/8b-config-sweep-v1/leaderboard.md). How the gate
applies:

- A config that **fails** the gate is never ranked or quoted
  (`test_quality_e2e.py::test_report_ranks_the_gate_failed_config_last`).
- A config whose gate is **inconclusive** is ranked but marked, and is not quoted while a
  verified config exists. On chat-sharegpt, fp8-kv8 is cheapest ($0.0467 [0.0418, 0.0522]
  blended, 5.239 req/s against BF16's 2.213 req/s and $0.1032). Its gate is inconclusive
  (gsm8k's CI crosses the 1-point margin), so the Qwen3-8B section quotes BF16.
- Reading note: the FP8 rows sit under their own model, "RedHatAI/Qwen3-8B-FP8-dynamic".
  Within that section none is verified, so the summary quotes the rank-1 row (fp8-kv8)
  and labels its quality "inconclusive". It is not a gate-verified figure.

Sweep and random search over a declared grid: `sweep:` and `sample:` in the experiment
YAML ([benchmark-lab.md](benchmark-lab.md#experiment-yaml)). Bayesian optimization and the
auto-generated pull request are "later" in the brief.

### P2. Winner into `config/models.yaml`

**Status: not met.**

- 8B: no candidate passed the gate, so `qwen3-8b` (BF16, vLLM 0.30.0, 1x L40S) stays as
  it is, and its `pricing` is null until you choose the margin method (cost-plus,
  market, or market floored at cost-plus; options computed in
  [benchmark-lab.md](benchmark-lab.md#qwen3-8b-config-sweep-result-7a8237d0-2026-10-10),
  method in [how-to/price-a-model.md](how-to/price-a-model.md)). Settling fp8-kv8 needs a
  quality-only rerun with more passes, which is not approved.
- 70B: the measured winner is FP8 on 2x H100 SXM at TP=2 (gate pass). The registry rows
  `llama-3.3-70b-instruct` and `llama-3.3-70b-instruct-fp8` still describe 4x L40S at
  TP=4, and both prices are null.

```bash
grep -n "pricing:" config/models.yaml
#   40:    pricing: null   78:    pricing: null   131:    pricing: null   186:    pricing: null
```

## Open items before approval

These are either not met or wait on a decision from you:

1. **Prices (P2, O2).** Pick the margin method for Qwen3-8B (cost-plus, market, or
   market floored at cost-plus), and decide whether the 70B FP8 row gets a price now.
   Until then every `pricing` is null, the competitiveness view has no margin, and
   Phase 1 billing has no price to charge.
2. **70B registry row (P2).** Decide whether the 70B rows move to the measured winner
   (FP8, 2x H100 SXM, TP=2). Today they describe 4x L40S at TP=4, where no load met the
   SLO.
3. **8B winner (P1/P2).** Accept BF16 as the 8B row, or approve a quality-only rerun to
   settle fp8-kv8's inconclusive gate (it is about 2.2x cheaper on chat if it passes).
4. **Model A on AWS (D6).** The AWS g6e.xlarge leaderboard ran (95cde129, $6.39): BF16
   on vLLM. It matches RunPod's latency at equal load, and fp8-kv8's gate failed on
   ifeval. The engine comparison (vLLM vs SGLang) is RunPod-only. Accept that as D6, or
   ask for SGLang on AWS too.
5. **Reproduce on real hardware (A1).** `bench reproduce` has only run on the mock.
   Accept the mock test plus the cross-pod comparison in A1, or approve one paid
   reproduce of a stored GPU run (one cold start plus one load point).
6. **Scheduled RunPod reaper (A5).** Today the RunPod backstops are the in-pod watchdog
   and a manual `bench reap`. A pod stuck before its container starts has neither. A
   sweep in the AWS reaper Lambda with the RunPod key is listed in PLAN.md as a
   follow-up and is not built.
7. **Results site and waitlist (D8).** Choose a waitlist backend (Formspree / Buttondown,
   or a Loom endpoint), choose which experiments to publish (pinned by full id in
   `site/config.yaml`, `publish.experiments`), and say when to enable GitHub Pages. The
   committed snapshot and the pinned list are empty.
8. **Results store (D5).** The real results are in SQLite. If you want them in Postgres
   for Phase 1, they need loading into it; the code already supports Postgres.
9. **Coverage not exercised on a GPU (D2-D4).** These are built and tested on the mock
   but have never run on real hardware:
   - the `vllm bench` / `sglang` wrappers;
   - closed-loop runs;
   - non-Poisson arrivals;
   - the 8k/1k, 32k/1k, needle, long-generation and trace profiles;
   - the MMLU-Pro, long-context and code eval tasks.

   Say whether any of these must run before approval.
