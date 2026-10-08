# Loom build plan

Status: Phase 0 leaderboards done (8B and 70B on RunPod); the FP8 70B row and strict tool calling are next. Phase 1 is approved (2026-10-08). Later phases start only after the previous phase is approved.

## Decisions so far

| Decision | Choice | Why |
|---|---|---|
| Phase 0 GPU provisioning | EC2 spot + Docker, driven over SSM; instance self-terminates at TTL | Cheapest and fastest under the $150 cap; no EKS control plane or cluster spin-up. Providers are pluggable, so a k8s/Helm provider (Phase 1) makes EKS/GKE runs a config change. |
| Phase 0 GPU provider (2026-10-05) | RunPod Secure Cloud on-demand: Qwen3-8B on 1x L40S, Llama 3.3 70B on 4x L40S. The `runpod` provider is built (2026-10-06): one pod per engine image, driven over direct SSH, jobs as an unprivileged user, results through presigned S3 URLs ([runbook-runpod.md](runbook-runpod.md)). EC2 (`aws_ec2`) stays as the secondary path; a spot quota increase is pending with AWS. | The AWS GPU spot quota is 0. Secure Cloud over Community for steadier latency. AWS support is still needed later. |
| Phase 0 region | AWS `us-east-1` | Deepest GPU capacity and spot pools; prices in `bench/prices.yaml`. |
| License | Apache-2.0 | Permissive with patent grant; same as vLLM / SGLang. |
| Quality standard for candidates (2026-10-08) | A candidate config (other precision, engine or serving technique) can ship if the quality gate passes it against the BF16 baseline, within each task's margin. Its precision always appears in the row name, and BF16 always stays the reference row. | Byte-identical output to BF16 is not the bar: BF16 itself is not identical from run to run (measured self-divergence and per-item eval noise, [quality-gate.md](quality-gate.md)). |
| 70B on 4x L40S (2026-10-08) | BF16 stays the reference and headline row, with its measured 50 ms TPOT p95 miss. FP8 is tried as its own labeled row ("Llama 3.3 70B FP8"), gated against BF16, with a $15 cap. If the gate fails FP8, record the score cost and accept the miss. No per-model SLO. BF16 on H100/H200 is a possible later run. | The measured floors are the PCIe host's (no working P2P across sockets). FP8 halves the bytes per token on the same hardware. Moving the SLO would hide the result. |
| Tool calling scores | The current task (plain tools, as most clients send them) stays the headline score. A strict-mode variant (`strict: true`, so vLLM constrains arguments to the schema) is added and reported next to it. | Llama 3.3 70B sends numbers as strings (0.450); both behaviours are useful to know, and the headline keeps measuring the default. |
| Phase 1 (2026-10-08) | Approved; gateway and web app can start. The results site is not published yet (hold until the user says so). | |

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
| Gateway (Phase 1, proposed) | Python FastAPI + uvloop | Shares registry and pricing code with Lab and billing, so prices cannot drift; one Redis Lua call per request keeps p50 overhead well under 10 ms. Revisit Go if the overhead test fails. |
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

### Phase 1 (platform) — proposal, to confirm before building

Billing and ledger design are expensive to reverse, so this is a draft for review at the start of Phase 1.

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
4. Phase 1: confirm gateway language, ledger design and Stripe account before building.
5. ~~Approval to build the RunPod provider~~ → approved and built (2026-10-06).

Later (RunPod hardening and follow-ups):
- A scheduled RunPod reaper, e.g. a sweep in the AWS reaper Lambda with the RunPod key in Secrets Manager. Today the backstops are the in-pod TTL watchdog and a manual `bench reap`; a pod stuck before its container starts has no watchdog.
- Codify the `loom-runpod-bench` IAM user (created with the CLI) in Terraform.
- Run the engine as a non-root user in the pod; today it runs as root and can read the HF token and the pod-scoped key ([security.md](security.md#runpod-pods-runpod)).
- Before every paid sweep: the RunPod smoke (`runpod-smoke`, about $1) runs every code
  path of the Qwen sweep at minimal scale, and CI's `pod-client-env` job rebuilds the
  pods' client environment and runs every eval task offline. The first 8B sweep
  (058128e9, $3.39) failed on an lm-eval extra the old vLLM-only smoke never touched.
- Optionally pin both pods of a vLLM vs SGLang comparison to one datacenter (`provider.data_center_ids`); today RunPod chooses and the datacenter is recorded per run.
- The AWS GPU spot quota increase is pending with AWS; when granted, the `aws_ec2` specs can run as well.
