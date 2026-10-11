# Loom build plan

Status: Phase 0 leaderboards done (8B and 70B on RunPod). The 70B ran on 2x H100 SXM (9f0853d7): BF16 meets the SLO, and FP8 passes the quality gate against BF16 on every task in the quality-only rerun on the 135-item tool-calling set (9e5f7866, $3.29), so FP8 ships as its own labelled row. The Qwen3-8B config sweep ran 2026-10-09/10 (7a8237d0, $14.44, after smokes 8bc65cfa and 55a6222b): fp8-kv8 is cheapest at the SLO (chat $0.0467 per 1M blended against bf16's $0.1032), but all four candidates' gates against BF16 are inconclusive (gsm8k's 1-point margin on every one), so none ships and BF16 stays the only verified 8B row; the 8B price stays unset for the user's decision. Model A on AWS ran 2026-10-10/11 (95cde129, $6.39, one g6e.xlarge on-demand; smokes $2.25): BF16 there matches RunPod's latency at equal load and its knee (chat 2.213 req/s, $0.1765 per 1M blended at AWS's $1.88/h), and fp8-kv8's gate fails on ifeval (-2.03 pts against a 2-pt margin), so BF16 stays the 8B row. Phase 1 is approved but paused until the user accepts the reworked Phase 0. Later phases start only after the previous phase is approved.

## Decisions so far

| Decision | Choice | Why |
|---|---|---|
| Phase 0 GPU provisioning | EC2 spot + Docker, driven over SSM; instance self-terminates at TTL | Cheapest and fastest under the $150 cap; no EKS control plane or cluster spin-up. Providers are pluggable, so a k8s/Helm provider (Phase 1) makes EKS/GKE runs a config change. |
| Phase 0 GPU provider (2026-10-05) | RunPod Secure Cloud on-demand: Qwen3-8B on 1x L40S, Llama 3.3 70B on 4x L40S. The `runpod` provider is built (2026-10-06): one pod per engine image, driven over direct SSH, jobs as an unprivileged user, results through presigned S3 URLs ([runbook-runpod.md](runbook-runpod.md)). EC2 (`aws_ec2`) stays as the secondary path; a spot quota increase is pending with AWS. | The AWS GPU spot quota is 0. Secure Cloud over Community for steadier latency. AWS support is still needed later. |
| Phase 0 region | AWS `us-east-1` | Deepest GPU capacity and spot pools; prices in `bench/prices.yaml`. |
| License | Apache-2.0 | Permissive with patent grant; same as vLLM / SGLang. |
| Quality standard for candidates (2026-10-08) | A candidate config (other precision, engine or serving technique) can ship if the quality gate passes it against the BF16 baseline, within each task's margin. Its precision always appears in the row name, and BF16 always stays the reference row. | Byte-identical output to BF16 is not the bar: BF16 itself is not identical from run to run (measured self-divergence and per-item eval noise, [quality-gate.md](quality-gate.md)). |
| 70B on 4x L40S (2026-10-08) | **FP8 on 4x L40S is on hold** (superseded by the 2x H100 SXM row below; 4x L40S had no stock). As planned: BF16 stays the reference and headline row, with its measured 50 ms TPOT p95 miss. FP8 is tried as its own labeled row ("Llama 3.3 70B FP8"), gated against BF16, with a $15 cap. If the gate fails FP8, record the score cost and accept the miss. No per-model SLO. BF16 on H100/H200 is a possible later run. | The measured floors are the PCIe host's (no working P2P across sockets). FP8 halves the bytes per token on the same hardware. Moving the SLO would hide the result. |
| 70B on 2x H100 SXM (2026-10-08) | Approved: Llama 3.3 70B on one RunPod Secure 2x H100 SXM pod, TP=2, vLLM v0.30.0. BF16 is the reference row and FP8 its own labeled row, gated against that BF16 cell in-run. $45 cap (`llama-3.3-70b-h100-tp2-runpod`, estimate $40.53), with the $9 smoke (`runpod-smoke-h100`, estimate $6.78) run first. Replaces the FP8 run on 4x L40S, which stays in the repo on hold. **Ran 2026-10-09 (9f0853d7, $25.18; smokes 4e50b5a5, e929eb0c, 4680fa3e, $6.15).** BF16 meets the 50 ms SLO at 0.5 req/s on fixed-1k-1k (TPOT p95 31 ms; untrusted, TTFT p95 CV 21.8%), $2.37/1M blended. FP8 holds 2 req/s (trusted), $0.546/1M blended at 1k/1k and $0.263 on shared-prefix. In that run FP8's gate was inconclusive (blocked) on `tool_calling_strict` alone, -1.67 pts [-6.67, +3.33] on 60 items; every other task, divergence and sanity passed. `tool_calling` is now 135 items (data version 2, 2026-10-09). **The quality-only rerun of both configs ran 2026-10-09 (9e5f7866, $3.29 of a $22 cap; planned $10.20): FP8's gate passes on every task**, tool_calling +0.74 pts [-1.48, +3.70] and tool_calling_strict +1.48 pts [-0.74, +3.70] on 135 items against the 6-point margin, gsm8k, ifeval, json_schema, divergence (KL 0.0040 nats, top-1 98.15%) and sanity too. FP8 is shippable as its own row ("Llama 3.3 70B FP8"); BF16 stays the reference row. Report: `reports/70b-h100-v2/`. Results: [benchmark-lab.md](benchmark-lab.md#70b-on-2x-h100-sxm-9f0853d7-2026-10-09) and [the rerun](benchmark-lab.md#70b-fp8-gate-on-tool-calling-data-version-2-9e5f7866-2026-10-09). | 4x L40S had no stock, and its cross-socket host (no working P2P) is the known TPOT floor. On H100 SXM a decode step reads its shard at ~3.35 TB/s and the TP all-reduce runs over NVLink. A same-host BF16 baseline isolates precision from the GPU. |
| Tool calling scores | The current task (plain tools, as most clients send them) stays the headline score. A strict-mode variant (`strict: true`, so vLLM constrains arguments to the schema) is added and reported next to it. Measured 2026-10-09: on Llama 3.3 70B with vLLM 0.30 strict mode never engages (xgrammar's Llama format only triggers on a call starting `{"name": `, and the 70B starts its calls another way), so both tasks score 0.450. Engine behaviour, documented in [quality-gate.md](quality-gate.md); failed strict items now also store the model's raw reply. | Llama 3.3 70B sends numbers as strings (0.450); both behaviours are useful to know, and the headline keeps measuring the default. |
| Phase 1 (2026-10-08) | Approved, then **paused 2026-10-08** until the user accepts the reworked Phase 0 (report rework, 8B config sweep, 70B on H100); it needs model prices from the Lab. Gateway and web app resume after that acceptance. Checklist: [phase0-acceptance.md](phase0-acceptance.md). The results site is not published yet (hold until the user says so). | |
| Gateway language (2026-10-08) | Python FastAPI + uvloop (confirmed) | See the Stack table. Revisit Go only if the p50 overhead test (< 10 ms) fails. |
| Ledger design (2026-10-08) | The Phase 1 draft below is accepted: double-entry, append-only, integer micro-dollars, idempotency keys, usage debits batched per org per minute, balance cached in Redis and reconciled every minute. | Expensive to reverse, so confirmed before building. |
| Stripe (2026-10-08) | Test mode only for Phase 1. The user creates the Stripe account; billing is built against a local Stripe stub until test-mode keys exist. | No accounts are created on the user's behalf. |
| Qwen3-8B config sweep (approved 2026-10-09; ran 2026-10-09/10) | Approved at a $25 cap, smoke first ($4.25 cap); the L40 / RTX 6000 Ada follow-up with the winning config is approved too.  `qwen3-8b-config-sweep-runpod`: vLLM v0.30.0 on RunPod Secure 1x L40S, five cells (BF16 reference; BF16 + FP8 KV cache; FP8-dynamic weights; FP8 + FP8 KV; FP8 + FP8 KV + `max_num_batched_tokens` 1024), each candidate gated vs BF16 in-run on phase0-strict with 3 passes, on chat-sharegpt (realistic, leading) and fixed-1k-1k, with windows sized to the measured variance. Three pods in sequence (`host_group`, a pod's TTL is capped at 8 h): estimate $18.71, worst case $24.77, proposed cap $25; smoke `runpod-smoke-8b-sweep` first ($3.20 estimate, $4.25 cap; **passed 2026-10-09, `8bc65cfa`, $1.02**: every path of the sweep ran on three pods, all four gates measured divergence, see [runbook-runpod.md](runbook-runpod.md#smoke-test)). Then the winner and its price go into `config/models.yaml` ([how-to/price-a-model.md](how-to/price-a-model.md): cost-plus, market, or market floored at cost-plus; the margin is the user's call). L40 / RTX 6000 Ada (25% / 9% cheaper per hour, same Ada FP8 kernels) are a ~$3 follow-up with the winning config: drafted as `qwen3-8b-winner-ada-runpod` (one pod per card, chat-sharegpt at the sweep's windows, gated against the sweep's stored BF16; planned $3.60, worst case $4.58, cap $5, or ~$3.03 at three search points), filled in from the sweep's result on 2026-10-10 with fp8-kv8 and the stored bf16 baseline; `bench plan` accepts it ($3.60, worst case $4.58, cap $5), not run. Design: [benchmark-lab.md](benchmark-lab.md#qwen3-8b-config-sweep-proposed). The planner's eval estimate is now calibrated on stored passes (scaled by each GPU's decode-step floor), which also brings the 70B H100 quality-only rerun from $14.88 to $10.20 planned (with the 135-item tool-calling set). | **Result (7a8237d0, 2026-10-10, $14.44, exit 0; [benchmark-lab.md](benchmark-lab.md#qwen3-8b-config-sweep-result-7a8237d0-2026-10-10), [reports/8b-config-sweep-v1](../reports/8b-config-sweep-v1/leaderboard.md))**: 135/135 runs, every goodput trusted. Chat goodput / $ per 1M blended: bf16 2.213 req/s / $0.1032, bf16-kv8 3.674 / $0.0670, fp8 3.865 / $0.0636, fp8-kv8 5.239 / $0.0467, fp8-kv8-mbt1024 4.734 / $0.0522. Every candidate's gate is INCONCLUSIVE (gsm8k CI crosses -1.00 on all four, none a measured drop; ifeval crosses -2.00 on bf16-kv8 and fp8-kv8-mbt1024); divergence and sanity pass on all. So no candidate ships; BF16 (`qwen3-8b`) is the only verified row, and the 8B price stays null: cost-plus, market and market-floored options are computed in benchmark-lab.md for the user to choose. Settling fp8-kv8 needs a quality-only rerun with more passes (not approved). Before the sweep, `runpod-smoke-8b-search-fail` (55a6222b, $1.12) ran the search's descent, bisection and overloaded drain on a pod. |
| 8B ships as BF16 (2026-10-10) | Qwen3-8B ships on BF16. FP8 + FP8 KV (fp8-kv8) stays unshipped: its gsm8k gate cannot resolve on 1,319 items, because most of its variance is consistent per-item flips that more passes do not average out; a larger math set is a possible later follow-up. | The quality standard: a candidate ships only on a gate pass; margins are not moved. |
| Model A on AWS (2026-10-10) | Approved: `aws-smoke-g6e` ($5 cap) then `qwen3-8b-aws-g6e` ($12.50 cap) on one g6e.xlarge on-demand (the 4 vCPU on-demand G/VT quota; spot is still 0), bf16 + fp8-kv8 on chat-sharegpt with 3 quality passes. AWS spend shares the single $150 Phase 0 pool with RunPod. **Ran 2026-10-10/11**: smoke a7bc3123 ($0.58) found a client-mount bug (fixed), smoke 80ff5648 ($1.67) passed, real run 95cde129 ($6.39, exit 0, 30/30 runs; four $0 launch attempts hit on-demand `InsufficientInstanceCapacity`). bf16 2.213 req/s (fails at 2.449), $0.1765 per 1M blended; fp8-kv8 5.239 req/s, $0.0799, gate FAIL (ifeval -2.03 [-4.37, +0.25]; gsm8k inconclusive). Latency within normal variance of RunPod 7a8237d0 at all 10 matched loads, same knees, same GPU utilisation; cost is 1.71x through the hourly price. Report [reports/8b-aws-g6e-v1](../reports/8b-aws-g6e-v1/leaderboard.md), details in [benchmark-lab.md](benchmark-lab.md#qwen3-8b-on-aws-g6exlarge-result-95cde129-2026-10-11). | Meets §7.8's "one GPU type, AWS" (vLLM only; the engine comparison is RunPod's) and cross-checks RunPod's numbers. |
| Phase 0 acceptance gaps (2026-10-11) | Before approval: one paid `bench reproduce` of a stored 8B RunPod run (A1), a scheduled RunPod reaper (A5), and vLLM vs SGLang on AWS g6e.xlarge (D6). Accepted as-is: results stay in SQLite until Phase 1 (D5); profiles and eval tasks not yet run on a GPU (D2-D4). The 70B registry rows move to FP8 on 2x H100 SXM, TP=2. Results site and waitlist deferred, no backend chosen. Prices: still open, to be discussed. | User decisions 2026-10-11. |
| Same-socket 2x L40S FP8 70B (2026-10-08) | Not now. | Considered (FP8 70B fits on 2x L40S with working P2P); outside the approved hardware config. |

## Stack

| Layer | Choice | One-line justification |
|---|---|---|
| Benchmark Lab | Python 3.12+, `uv` workspace | vLLM, SGLang, lm-eval-harness are Python; wrap them in-process instead of reimplementing. |
| CLI | Typer | Typed commands, little code. |
| Config | YAML + Pydantic v2 | Registry, prices, experiments and workloads are validated at load time; adding a model is a YAML change. |
| HTTP client | `httpx` async, hand-rolled SSE parsing | Per-chunk timestamps with no client-side buffering. |
| Mock backend | FastAPI + uvicorn | OpenAI-compatible streamer with a simulated batching GPU (queueing, prefill/decode costs, prefix cache, `/metrics`), so the whole Lab runs without a GPU. |
| Results store | Postgres (SQLAlchemy 2 + Alembic) for provenance and aggregates; Parquet for per-request rows | Postgres is queryable and shared with the platform later; Parquet keeps millions of request rows cheap. |
| Stats | numpy + scipy | Log-scale t-intervals (geometric means) for positive metrics and clipped t-intervals for fractions across repetitions; paired bootstrap for eval deltas. |
| Quality | lm-evaluation-harness for MMLU-Pro / GSM8K / IFEval; native tasks for code exec (sandboxed), JSON-schema validity, tool calling, needle retrieval, logprob divergence | lm-eval covers the standard tasks; the native ones need control over sandboxing and per-sample scoring. |
| AWS | boto3 (EC2, SSM, S3, Secrets Manager); Terraform for the bucket, IAM role and reaper Lambda | Least-privilege role and the reaper exist independently of any laptop. |
| Reports / site | Jinja2 → markdown, HTML, CSV; static site published to GitHub Pages | No server to run; repo is public. |
| Gateway (Phase 1) | Python FastAPI + uvloop | Shares registry and pricing code with Lab and billing, so prices cannot drift; one Redis Lua call per request keeps p50 overhead well under 10 ms. Revisit Go if the overhead test fails. |
| Web (Phase 1) | Next.js App Router, Tailwind, Auth.js + Postgres adapter | Per spec. |

## Phase 0 plan (increments, each committed with tests)

1. Scaffold: uv workspace, `loom-bench` package, CI, docker-compose Postgres.
2. Config schemas + loaders: `config/models.yaml`, `bench/prices.yaml`, `bench/competitors.yaml`; integer micro-dollar money type.
3. Mock backend with simulated batching GPU and Prometheus metrics.
4. Native load generator: OpenAI-compatible streaming client; open-loop arrivals (Poisson, bursty, ramp, diurnal, trace replay) and closed-loop concurrency; warmup discard.
5. Workload profiles: fixed shapes 128/128, 1k/1k, 8k/1k, 32k/1k; chat (ShareGPT-style); shared prefix with 0-95% knob; long-context needle; code completion; long generation; trace replay.
6. Metrics: TTFT / TPOT / ITL / E2E percentiles, throughput per replica and per GPU, SLO attainment, goodput via load sweep, max sustainable concurrency; server-side queue time, preemptions, KV-cache use, prefix-cache hit rate; GPU utilization.
7. Cost: $/1M input and output tokens at SLO, from goodput and `prices.yaml`, with confidence intervals.
8. Storage + provenance: Postgres tables, Parquet, CSV export; config hashing.
9. Runner: experiment matrix expansion, dry-run plan with cost estimate, budget guard with hard abort, repetitions, teardown. Providers: `mock`, `local`, `aws_ec2`.
10. Reports: leaderboard (md / HTML / CSV), `bench compare`, `bench reproduce`, competitiveness view and margin flags.
11. Quality: pinned eval subsets, gate (paired non-inferiority with CIs), logprob divergence (KL, top-1 agreement), sanity checks; test proving the gate blocks a broken config.
12. AWS provider + TTL reaper + Terraform; wrappers for `vllm bench serve` and `sglang.bench_serving`.
13. Public results page + waitlist form.
14. **Stop and ask** for the AWS account, HF token and GPU quota, then run the first leaderboards for Qwen3-8B (vLLM vs SGLang, 1 GPU type) and Llama 3.3 70B (one TP config).

## Postgres schema

All money is `BIGINT` micro-dollars. Timestamps are `timestamptz`.

### Phase 0 (Benchmark Lab) — being built now

```sql
bench_experiments (
  id uuid PRIMARY KEY, name text, spec jsonb, spec_hash text,
  git_sha text, git_dirty bool, status text,           -- planned|running|completed|aborted|failed
  budget_micros bigint, spent_micros bigint DEFAULT 0, abort_reason text,
  created_at timestamptz, finished_at timestamptz)

bench_runs (                                           -- one load point x one repetition
  id uuid PRIMARY KEY, experiment_id uuid REFERENCES bench_experiments,
  cell_key text, config_hash text, workload text, load_mode text,  -- open_loop|closed_loop
  load_value double precision, repetition int, status text,
  provenance jsonb,                                    -- git sha, engine+version, image digest, CUDA/driver,
                                                       -- model revision, GPU, cloud, region, dataset+license, ...
  summary jsonb,                                       -- percentiles, throughput, SLO attainment, server metrics
  requests_uri text,                                   -- Parquet with per-request rows
  started_at timestamptz, finished_at timestamptz)

bench_cold_starts (id uuid, experiment_id uuid, resource_id text, kind text,   -- cold|warm
  stages jsonb, total_s double precision, created_at timestamptz)

bench_eval_runs (id uuid, experiment_id uuid, config_hash text, task text, task_version text,
  n int, score double precision, ci_low double precision, ci_high double precision,
  provenance jsonb, samples_uri text, created_at timestamptz)

bench_gate_decisions (id uuid, experiment_id uuid, baseline_config_hash text,
  candidate_config_hash text, decision text,           -- pass|fail|inconclusive
  details jsonb, created_at timestamptz)

bench_resources (id uuid, provider text, resource_type text, resource_id text, region text,
  experiment_id uuid, tags jsonb, created_at timestamptz, ttl_at timestamptz,
  terminated_at timestamptz, terminated_by text)       -- runner|reaper|self-ttl

bench_spend (id uuid, experiment_id uuid, resource_id text, amount_micros bigint,
  basis jsonb, recorded_at timestamptz)                -- price used, seconds, market (spot/on-demand)

waitlist_signups (id uuid, email citext UNIQUE, source text, created_at timestamptz)
```

### Phase 1 (platform) — accepted 2026-10-08

Billing and ledger design are expensive to reverse; this draft was reviewed and accepted at the start of Phase 1.

```sql
-- Auth.js Postgres adapter tables: users, accounts, sessions, verification_token
orgs (id uuid PK, name text, status text,          -- active|suspended
  suspended_at timestamptz, suspended_reason text, retention_days int DEFAULT 0, created_at timestamptz)
org_members (org_id uuid, user_id uuid, role text, PRIMARY KEY (org_id, user_id))   -- owner|admin|member
api_keys (id uuid PK, org_id uuid, created_by uuid, name text,
  prefix text UNIQUE, key_hash bytea UNIQUE,       -- SHA-256 of the full key; plaintext shown once
  scopes text[], spend_limit_micros bigint, rpm_limit int, tpm_limit int,
  created_at timestamptz, last_used_at timestamptz, revoked_at timestamptz)
usage_events (id uuid PK,                          -- request id = idempotency key
  org_id uuid, api_key_id uuid, model_id text, price_version text, cloud text, region text,
  prompt_tokens int, completion_tokens int, cached_prompt_tokens int,
  latency_ms int, ttft_ms int, status text,        -- ok|aborted|error
  cost_micros bigint, created_at timestamptz)      -- partitioned by month
ledger_accounts (id uuid PK, org_id uuid NULL, kind text)   -- org_credit|revenue|stripe_clearing|promo_expense
ledger_transactions (id uuid PK, kind text,        -- purchase|usage|refund|promo|adjustment
  idempotency_key text UNIQUE, org_id uuid, external_ref text, memo text, created_at timestamptz)
ledger_entries (id bigserial PK, transaction_id uuid, account_id uuid, amount_micros bigint)
  -- deferred constraint trigger: entries of each transaction sum to zero; table is append-only
stripe_events (id text PK, type text, payload jsonb, received_at timestamptz, processed_at timestamptz)
auto_recharge (org_id uuid PK, enabled bool, threshold_micros bigint, amount_micros bigint)
admin_audit_log (id bigserial PK, actor_user_id uuid, action text, target_type text,
  target_id text, details jsonb, created_at timestamptz)   -- UPDATE/DELETE revoked + trigger
```

Usage debits would post to the ledger in per-org batches (one transaction per org per minute referencing the covered `usage_events`) rather than one transaction per request. Balance = sum of `org_credit` entries, cached in Redis and reconciled every minute.

## Model registry schema (`config/models.yaml`)

Enforced by `loom_bench.registry` (Pydantic). Adding a model is a YAML change.

```yaml
models:
  - id: qwen3-8b                    # public API id
    display_name: Qwen3 8B
    hf:
      repo: Qwen/Qwen3-8B
      revision: <40-char commit sha>  # required, pinned
      license: apache-2.0
      gated: false                    # false | auto | manual
      size_bytes: 16381516776
      quant_method: null              # checkpoint format, from config.json quantization_config:
                                      # fp8|compressed-tensors|awq|gptq|modelopt; null = BF16/FP16
      trust_remote_code: false        # true requires trust_remote_code_review {reviewer, date, notes}
    engine:
      name: vllm                      # vllm | sglang
      version: "x.y.z"
      image: vllm/vllm-openai@sha256:<digest>   # digest required
      args: {}                        # engine-specific flags (prefix caching, max seqs, ...);
                                      # flags set from fields here (quantization, tp, ...) are refused
      chat_template_kwargs: {}        # sent with chat requests in benchmarks and evals
    hardware:
      gpu: L40S
      gpus_per_replica: 1             # must equal tp * pp
      nodes_per_replica: 1            # >1 reserved for LeaderWorkerSet / KubeRay (not built yet)
      instance_types: {aws: g6e.xlarge}
    parallelism: {tp: 1, pp: 1, ep: 1}
    max_context: 32768
    quantization: none                # serving precision: none|fp8|awq|gptq|w4a16|w8a8|fp4|nvfp4
    kv_cache_dtype: auto
    pricing:                          # integer micro-dollars per 1M tokens
      input_per_mtok: 0
      output_per_mtok: 0
      cached_input_per_mtok: 0        # must be <= input
    scaling: {min_replicas: 0, max_replicas: 1}
    clouds: [aws]
    capabilities: {tools: true, json_schema: true, vision: false, reasoning: true}
    routing_tier: 1
    status: preview                   # enabled|preview|disabled
```

`hf.quant_method` describes the checkpoint; `quantization` is the precision served.
Allowed pairs (`registry.CHECKPOINT_PRECISIONS`) and the engine flag they render
(`engines.quantization_flag`, same for vLLM and SGLang):

| `hf.quant_method` | `quantization` | `--quantization` |
|---|---|---|
| null | none | none |
| null | fp8 | `fp8` (quantized on load) |
| fp8 | fp8 | none |
| compressed-tensors | fp8, w8a8, w4a16, nvfp4 | none |
| awq | awq | none |
| gptq | gptq | none |
| modelopt | fp8, nvfp4 | none |

A pre-quantized checkpoint names its method in `config.json`, which the engine reads;
passing a different `--quantization` makes it refuse to start. Any other pair is rejected
when the registry loads.

## Open questions

Asked now:
- ~~Provisioning approach~~ → EC2 spot + Docker; secondary for GPU benches since the AWS GPU spot quota came back 0 (RunPod Secure Cloud on-demand is the primary; a quota increase is pending with AWS).
- ~~License~~ → Apache-2.0.

Needed later (will stop and ask when reached):
1. ~~AWS account access and GPU spot quota~~ → account access is set up and `infra/aws/bench` is applied; the GPU spot quota is 0, so GPU benches move to RunPod (see Decisions).
2. ~~Hugging Face token with the Llama 3.3 license accepted~~ → done. On AWS it is in Secrets Manager (`loom/hf-token`); on RunPod it is a RunPod secret ([security.md](security.md#runpod-pods-runpod)). Never in the repo.
3. Waitlist backend for the public results page: a form service (Formspree / Buttondown) or a small AWS Lambda writing to Postgres.
4. ~~Phase 1: confirm gateway language, ledger design and Stripe account before building~~ → confirmed 2026-10-08 (see Decisions).
5. ~~Approval to build the RunPod provider~~ → approved and built (2026-10-06).

Later (RunPod hardening and follow-ups):
- Scheduled RunPod reaper: built (2026-10-10) as its own Lambda, `loom-bench-runpod-reaper` (`infra/aws/bench/runpod_reaper.tf`), with its RunPod key in Secrets Manager; to deploy, follow [aws-setup.md, step 8](aws-setup.md#8-turn-on-the-runpod-reaper). Until it is live, the backstops are the in-pod TTL watchdog and a manual `bench reap`, and a pod stuck before its container starts has no watchdog.
- Codify the `loom-runpod-bench` IAM user (created with the CLI) in Terraform.
- Run the engine as a non-root user in the pod; today it runs as root and can read the HF token and the pod-scoped key ([security.md](security.md#runpod-pods-runpod)).
- Before every paid sweep: the RunPod smoke (`runpod-smoke`, about $1) runs every code
  path of the Qwen sweep at minimal scale, and CI's `pod-client-env` job rebuilds the
  pods' client environment and runs every eval task offline. The first 8B sweep
  (058128e9, $3.39) failed on an lm-eval extra the old vLLM-only smoke never touched.
- Optionally pin both pods of a vLLM vs SGLang comparison to one datacenter (`provider.data_center_ids`); today RunPod chooses and the datacenter is recorded per run.
- The AWS GPU spot quota increase is pending with AWS; when granted, the `aws_ec2` specs can run as well.
