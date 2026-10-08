# Runbook: Phase 0 GPU runs on RunPod

Operating the Benchmark Lab on RunPod Secure Cloud on-demand pods (`provider.kind:
runpod`): preflight, running, watching spend, incidents and checking teardown. This is
the primary GPU path for Phase 0, because the AWS GPU spot quota is 0 (an increase is
pending with AWS). The `aws_ec2` path is in [runbook.md](runbook.md); its sections on
reproducing and publishing apply here unchanged. The security model is in
[security.md](security.md#runpod-pods-runpod).

Money rails, for reference: $149.75 overall, $50 per experiment (`bench/budget.yaml`), and
each experiment's own `budget.max_spend`. RunPod is prepaid with auto-pay off, so the
account balance is the hard ceiling behind them. RunPod has no per-pod spend cap: the
rails are the runner's budget guard, the pod's own TTL watchdog and `bench reap`. How the
guard acts is in [benchmark-lab.md](benchmark-lab.md#budget-rails).

Commands below assume the repo root and the default name prefix `loom-bench`. They read
the API key from `RUNPOD_API_KEY`; never print it.

## 1. Preflight checklist

Run through all of it before every real run.

- [ ] **API key** is found (environment first, then the macOS Keychain, service
  `RUNPOD_API_KEY`):

  ```sh
  uv run python -c "from loom_bench.providers.runpod_api import load_runpod_api_key as k; print('key found' if k() else 'NO KEY')"
  ```

- [ ] **Balance, and nothing already running.** An old pod keeps billing, and a network
  volume bills even when idle:

  ```sh
  curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"query { myself { clientBalance pods { id name desiredStatus } networkVolumes { id name size } } }"}'
  ```

  Expect `pods: []` and `networkVolumes: []`. Note `clientBalance` to compare after the
  run.
- [ ] **RunPod secret `hf_token`** exists (console → Secrets; `RunpodSettings.hf_secret_name`),
  and the HF account behind it has accepted the license of gated models (Llama 3.3).
  Pods reference it as `{{ RUNPOD_SECRET_hf_token }}`. The `aws_*` RunPod secrets must
  never be referenced (a validator rejects such a name).
- [ ] **AWS identity for presigning.** The runner uploads job inputs to the bench bucket
  and presigns per-object URLs with the laptop's AWS credentials; pods get no AWS
  credentials. The identity needs `s3:PutObject`/`GetObject` on the bucket. Presigned URLs
  stop working when the signing credentials expire, so short-lived session credentials
  cut them short:

  ```sh
  aws sts get-caller-identity
  ```

- [ ] **Settings** load. Set them as `LOOM_RUNPOD_<FIELD>` environment variables, or in
  a YAML file named by `LOOM_RUNPOD_CONFIG` (environment variables override the file):

  ```sh
  export LOOM_RUNPOD_BUCKET=<bench bucket> LOOM_RUNPOD_OWNER=<your-name>
  uv run python -c "from loom_bench.providers.runpod import load_runpod_settings as s; print(s())"
  ```

  | Field | Default | Notes |
  |---|---|---|
  | `bucket` | required | the bench bucket (`terraform -chdir=infra/aws/bench output -raw bucket`) |
  | `owner` | required | recorded in the pod env as `LOOM_OWNER` |
  | `s3_region` | `us-east-1` | |
  | `name_prefix` | `loom-bench` | pods are named `{prefix}-{experiment[:8]}-{ttl epoch}-{nonce}` |
  | `hf_secret_name` | `hf_token` | names starting with `aws` are rejected |
  | `rest_url`, `graphql_url` | `https://rest.runpod.io/v1`, `https://api.runpod.io/graphql` | https only |
  | `max_ttl_s` | 28800 (8 h) | the planner refuses a longer host TTL |
  | `wheel_path` | built with `uv build` when unset | |
  | `client_python_url`, `client_python_sha256` | pinned python-build-standalone CPython 3.12 | `runpod_layout.CLIENT_PYTHON_*` |
  | `authorize_account_key` | true | also authorize the account's registered SSH key, for debugging |
  | `poll_interval_s` | 5 | |
  | `job_timeout_s` | 7200 | for request-count jobs |
  | `presign_expiry_s` | 21600 (6 h) | at most 7 days |
  | `pod_ready_timeout_s` | 900 | until the pod is running with an SSH address |
  | `ssh_online_timeout_s` | 600 | until sshd answers |
  | `max_price_ratio` | 1.25 | refuse a pod whose API price is above prices.yaml × this |
  | `capacity_wait_s` | 1800 | how long to keep retrying a create refused for lack of stock (0: fail at once) |
  | `engine_stall_s` | 600 | fail an engine start whose log has not grown for this long before it is healthy |

- [ ] **OpenSSH** on the laptop (`ssh`, `ssh-keygen`): the runner drives pods over SSH.
- [ ] **Results database** is up and migrated, the same one as every other real run. It
  holds every recorded cent of spend (the overall cap is computed from it):

  ```sh
  docker compose up -d postgres && uv run bench db upgrade
  ```

- [ ] **Clean git tree.** Provenance records the commit and a dirty flag. Commit first.
- [ ] **The smoke test passed on this commit** (see [Smoke test](#smoke-test)) before
  any real sweep: it runs every code path the Qwen sweep uses for about $1.
- [ ] **Plan is under the cap**:

  ```sh
  uv run bench plan bench/experiments/qwen3-8b-vllm-vs-sglang-runpod.yaml   # exit 0 = accepted
  ```

  Check "estimated spend" and "worst case (all hosts to TTL)" against "effective cap".
  The hourly rate per pod is prices.yaml's on-demand price plus container disk; at run
  time the guard accrues the larger of that and the pod's API `costPerHr`. Expected
  today: `runpod-smoke` $1.35 estimate, $2.20 worst case, $2.25 cap (two pods);
  `qwen3-8b-vllm-vs-sglang-runpod` $7.92, $13.21, $15 (two pods, one per engine);
  `llama-3.3-70b-tp4-runpod` $11.73, $17.58, $45;
  `llama-3.3-70b-fp8-tp4-runpod` $11.05, $14.24, $15 (plan it against the real results
  DB: it loads BF16 run cf4d1614's stored samples and reference, and is refused without
  them); `runpod-smoke-fp8` $0.93, $1.19, $1.25. Exit 3 means refused: lower the load
  points or `budget.ttl_minutes`, never the caps.

## 2. Running an experiment

```sh
caffeinate -i uv run bench run bench/experiments/qwen3-8b-vllm-vs-sglang-runpod.yaml   # macOS: keep awake
```

What happens, per engine image (each image gets its own pod):

1. The runner generates an SSH key for this run and creates a Secure Cloud on-demand pod
   (`interruptible: false`, `volumeInGb: 0`, CUDA 13.0 hosts, port 22/tcp) from the
   engine image pinned by digest. A pod whose API price is above prices.yaml ×
   `max_price_ratio` is terminated before any work.
2. The pod's start command arms the TTL watchdog, then installs and starts sshd and
   creates the job user (uid 10001).
3. Over SSH, the engine start downloads weights at the pinned revision with the HF token,
   starts the engine on `127.0.0.1`, waits for one token and checks job isolation
   (`job_isolation ok`).
4. Each load point × repetition runs as `bench job run` in the pod, as the job user;
   inputs and results move through presigned S3 URLs under `runs/<experiment id>/<run id>/`.
   Quality suites run there too, as `bench quality job`.
5. The pod is terminated at the end, also on errors and Ctrl-C.

Exit codes: 0 ok, 1 failed (including every load run failing), 3 refused, 4 stopped
before a step that would pass the cap, 5 hard budget abort, 8 finished but some load runs
or quality evals failed. Run one experiment at a time, as on AWS.

## 3. Monitoring

Spend: the `bench_experiments` and `bench_spend` queries in
[runbook.md](runbook.md#3-monitoring). Progress: the runner's log lines and
`results/<experiment id>/events.jsonl` (`provisioned`, `engine_started` with its stages
and `system` facts, quality, gates). The pod's API price and the prices.yaml rate it was
compared with are in the host's `accrual_basis`; each run's provenance records the
as-run price with basis `observed_api` and the datacenter. When RunPod names no
datacenter (an empty `machine.dataCenterId` and no `RUNPOD_DC_ID` in the pod), the
machine's location is recorded instead, as `data_center_location` and
`availability_zone: location:<code>` (a country, not a datacenter).

Pod side, while it runs: with `authorize_account_key` on, the account's registered key
can log in as root (`ssh root@<public ip> -p <mapped port>`; both are on the pod's
Connect tab). Logs, all lost when the pod is terminated:

- `/var/lib/loom/`: `stages`, `apt.log`, `sshd.log`, `setup.log`, `terminate.log`
- `/var/log/loom/`: `weights.log`, `engine.log`
- `/var/lib/loom/jobs/<run id>/`: `client.log`, `install.log`

The RunPod console (Pods) shows each pod's state, price and container logs.

## 4. Incidents

### Hard budget abort (exit 5) or graceful stop (exit 4)

The runner has already terminated every pod and marked the experiment `aborted` with the
reason. Confirm nothing is running (the preflight query), read `abort_reason` and
`spent_micros`, and fix the plan before running again, as in
[runbook.md](runbook.md#hard-budget-abort-exit-5-or-graceful-stop-exit-4).

### A quality eval failed (exit 8)

The experiment keeps going: the other engines still run their load sweeps and evals.
`events.jsonl` has a `quality_failed` event with the error, the experiment's reason counts
the failed evals, and any gate missing a side is stored as `inconclusive` (blocked). The
load results stand; fix the cause (the error usually names a missing module or a harness
failure), prove it with the smoke test, and rerun.

A `divergence_failed` event instead means the eval's task scores were recorded but its
divergence half raised: the gate's divergence check is `inconclusive` (blocked) with the
error, and the reason reads "divergence failed in N of M quality evals". When the load
sweep itself is fine, finish the gate without re-running load: fix the cause, prove it
with the smoke test, then run the quality-only spec (`qwen3-8b-quality-runpod.yaml`
for the Qwen sweep). Its evals and gate land on the sweep's config hashes.

The same spec settles a gate that came back INCONCLUSIVE because the CI crossed a task's
margin. It runs three eval passes per engine (`quality.replicates: 3`), which averages
engine nondeterminism out of the paired comparison (docs/quality-gate.md, "Replicated
passes"). If the reason reads "a real drop (CI below 0)", the drop is measured and more
passes only narrow it. Report it as a regression of that size, not as noise.

### Pod refused for its price

`pod … costs … $/h, above prices.yaml … x 1.25: terminated`. RunPod's price moved. Check
the console price, update `bench/prices.yaml` (and its `last_checked`), plan again.

### No capacity, or a pod that never comes up

A failed create surfaces as `RunPod POST /pods failed (...)`. If the request failed
without a clear answer from RunPod, the provider terminates any pod created under that
name anyway. When RunPod refuses for lack of stock (`There are no instances currently
available`), the provider waits 30, 60, 120, then every 240 s and retries, for up to
`capacity_wait_s` (1800 s). Each attempt gets a fresh name and TTL, and nothing is billed
while waiting. Other refusals are not retried. A pod without an SSH address after `pod_ready_timeout_s` (900 s), or whose
sshd does not answer within `ssh_online_timeout_s` (600 s), fails the experiment and is
terminated. L40S Secure stock varies by datacenter and over the day: retry later, or pin
`provider.data_center_ids` to one with stock.

Multi-GPU pods are scarcer: on 2026-10-07 RunPod listed no 4x L40S Secure stock at all
(1x was "Low"). Before launching `llama-3.3-70b-tp4-runpod`, check it read-only:

```graphql
query { gpuTypes(input: {id: "NVIDIA L40S"}) {
  lowestPrice(input: {gpuCount: 4, secureCloud: true}) { stockStatus uninterruptablePrice }
} }
```

A null `stockStatus` means none is listed; wait for "Low" or better. Then run with
`LOOM_RUNPOD_CAPACITY_WAIT_S=7200`, so a refusal waits up to 2 h rather than 30 min
(nothing is billed while waiting).

### A multi-GPU engine hangs at start (NCCL peer-to-peer)

RunPod can place a multi-GPU pod's GPUs on both CPU sockets. On 2026-10-08 the first 70B
attempt (`874110b6`, 4x L40S) had GPU0 on socket 0 and GPUs 1-3 on socket 1
(`nvidia-smi topo -m`: SYS between them). `torch.cuda.can_device_access_peer` still said
yes for every pair, but NCCL hung at init: vLLM's last line was `vLLM is using nccl==…`,
until the 1800 s ready timeout. On a pod with the same topology, a 4-rank NCCL
`all_reduce` timed out by default and with `NCCL_SHM_DISABLE=1`, and passed in under a
second with `NCCL_P2P_DISABLE=1`. vLLM TP=4 then came up in 80 s. vLLM itself disables
its custom all-reduce on more than two PCIe-only GPUs, so NCCL is the only P2P user.

So expansion refuses a multi-GPU RunPod cell unless `provider.engine_env` sets
`NCCL_P2P_DISABLE` or `NCCL_P2P_LEVEL`. The value is part of the launch, so of the
config hash. The shipped 70B spec sets `NCCL_P2P_DISABLE: "1"`. Each engine start also
records the GPU topology (`gpu_topology` in `engine_started.system`, or in
`engine_start_failed` when the start fails), so a hang like this can be read from
`events.jsonl`.

### Runner gone, or a stuck pod

- If the runner hangs, press Ctrl-C once: pods are terminated and the experiment is
  marked `failed`.
- If the runner process is gone (laptop slept, crashed), the pod's watchdog terminates it
  at its TTL. Terminate it now instead, from the console (**Terminate**, never **Stop**:
  a stopped pod keeps billing for its disk) or the API:

  ```sh
  curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' \
    -d '{"query":"mutation { podTerminate(input: {podId: \"<pod id>\"}) }"}'
  ```

- A pod stuck before its container starts has no watchdog yet. There is no scheduled
  RunPod reaper, so check the console or run `bench reap` (below).

### Orphaned pods

`bench reap` terminates pods both named `loom-bench-…` and carrying env
`LOOM_MANAGED=true`, once past their TTL (env `LOOM_TTL`, else the epoch in the name, else
older than 24 h) or `EXITED`, plus expired pods recorded in the database. Other pods on
the account are never touched.

```sh
uv run bench reap --dry-run      # what is expired: DB-recorded hosts + managed pods
uv run bench reap                # terminate them
```

Without an API key, `bench reap` prints `No RunPod API key: RunPod pods are not reaped`.

## 5. Teardown verification

After every real run:

1. The preflight query shows `pods: []` and `networkVolumes: []`.
2. `uv run bench reap --dry-run` lists nothing to terminate.
3. The balance drop matches the experiment's `spent_micros` to within a few cents
   (container disk and rounding). Spend made outside the results database lowers
   `overall_cap` by hand (`bench/budget.yaml`).

No network volume is created today (`volumeInGb: 0`, weights on the container disk).
When a sweep needs one (about 250 GB, about $17 a month, billed even when idle), create it
only when the sweep starts, in a datacenter with L40S stock, and delete it as soon as the
sweep finishes.

## Smoke test

`bench/experiments/runpod-smoke.yaml` is the real Qwen3-8B sweep
(`qwen3-8b-vllm-vs-sglang-runpod`) at smoke scale, so every code path the sweep uses runs
once before the sweep pays for hours: both engine images by the same digests (one Secure
1× L40S pod each), the same three workload profiles under the same geometric rate search
(2 points with a 1-step descent, 10 s windows) and the same 3 repetitions, the same
quality subset with every task capped at 4 items (`quality.limit`), divergence capture,
noise floor and scoring on 4 prompts that include the suite's hard prompts (20 and 40,
whose continuations split characters across tokens), and the SGLang-vs-vLLM gate. `max_spend: "$2.25"`, TTL 60 min per
pod (worst case $2.20; SGLang's image pull took 226 s in b97dea4c, so 47 min left ~2 min of margin). The repetitions must match: the pass verdict's CI upper bound
uses t with reps-1 degrees of freedom, and at 2 reps (t ≈ 12.7) smoke 51ad57b0 failed
every first point, so the climb never ran.
`bench/tests/runner/test_plan.py` fails if the smoke and the real spec drift apart. It is
`smoke: true`: left out of default reports and the results site, since its cells share
config hashes with the real ones. The old smoke ran vLLM only with no eval, and the first
8B sweep (058128e9) found the missing lm-eval extra hours in, after $3.39. Smoke 28d301fa
scored divergence on the first 4 prompts only, all ASCII continuations, so it passed while
the second sweep (565b8d3f) then lost SGLang's eval to characters split across tokens on
prompts 20 and 40; the smoke now scores those two.

Before the 70B FP8 run, `runpod-smoke-fp8.yaml` (one 1x L40S pod, about $0.93) runs its
FP8 path: RedHatAI's FP8-dynamic compressed-tensors checkpoint of Qwen3-8B (made the same
way as the 70B's) loaded by the same vLLM image with no `--quantization` flag, a warm
restart onto it from the BF16 checkpoint, its eval job scoring divergence on the BF16
reference, and its gate, under the 70B FP8 run's load shapes and quality subset
(`phase0-strict`, so `tool_calling_strict` runs on both). Check
both cells' `engine_started` events, `quality` events for both, a `gate` event for
`vllm-fp8` with a measured divergence, and no `quality_failed` or `divergence_failed`.
What it cannot run (TP=4, the stored-baseline gate) is covered by the BF16 70B runs and
offline tests (`bench/tests/runner/test_stored_baseline.py`, `test_fp8_70b.py`).

The 2x H100 SXM option (`llama-3.3-70b-h100-tp2-runpod.yaml`, BF16 baseline and FP8 on one
pod, TP=2) is a draft that is **not approved**. If it is approved, run its smoke
`runpod-smoke-h100.yaml` first (also a draft, about $6.78): it adds the H100 SXM GPU type,
sm_90 FP8 GEMMs, TP=2 with NCCL P2P over NVLink (`NCCL_P2P_LEVEL=NVL`) and
`gpu_memory_utilization: 0.95`. Check its `engine_started.system.gpu_topology` shows
`NV*` links between the two GPUs. H100 SXM Secure 2x was "Low" stock on 2026-10-08.

Dependency bugs are caught for free before that: `uv run pytest -m network` (CI job
`pod-client-env`) rebuilds the pods' client environment from `uv.lock` and runs every task
of every eval suite at 2 items against the mock backend.

```sh
uv run bench plan bench/experiments/runpod-smoke.yaml
uv run bench run bench/experiments/runpod-smoke.yaml
```

Checklist:

- [ ] `events.jsonl` has an `engine_started` event with `system.job_isolation == "ok"`
  for both pods (vLLM and SGLang).
- [ ] Each workload ran a second search point (a climb if `lo` passed, a descent if it
  failed), every load run is ok, and the experiment ended `completed` with exit 0 (exit 8
  means a run or an eval failed: read the reason). Goodput may come out empty at smoke
  scale; the search path running is what counts.
- [ ] Both cells have a `quality` event naming gsm8k, ifeval, tool_calling and
  json_schema (tool_calling scored, not rejected), there is no `quality_failed` or
  `divergence_failed` event, `reference_captured` has a `self_divergence`, and SGLang has
  a `gate` event (inconclusive at 4 items is expected).
- [ ] vLLM's `reference.json` holds prompts 20 and 40 (the hard prompts), and SGLang's gate
  reasons give a measured divergence ("KL ... nats, top-1 ...%"), not "not measured: ...".
- [ ] The cold start stages include `pod_created`, `image_pulled`, `sshd_ready`,
  `ssh_online`, `weights_ready`, `engine_healthy` and `first_token`.
- [ ] The run completed, and its `result.json` and `gpu.csv` are under
  `s3://<bucket>/runs/<experiment id>/<run id>/`.
- [ ] Provenance records `cloud: runpod`, `instance_type: l40s-x1`, basis `observed_api`,
  and CUDA and driver versions.
- [ ] The recorded API `costPerHr` matches prices.yaml (within `max_price_ratio`).
- [ ] The pod shows as terminated (gone from the pod list), not exited.
- [ ] Section 5 passes.

Smoke test result (redesigned smoke with the hard divergence prompts): pending. Smoke
28d301fa (2026-10-07, ASCII-only divergence prompts) passed every item; the byte-split bug
it could not see is fixed and covered offline since.

Earlier smoke test result (the vLLM-only smoke, 2026-10-06): passed on the second attempt, experiment `17d0cb33`, 1x L40S Secure in US-MO-1. `job_isolation ok` was recorded, the pod terminated, `bench reap --dry-run` was clean, and the DB spend ($0.1506 over both attempts) matched the balance drop ($0.1502). The first attempt failed because pods did not follow the 302 redirect on the client Python download (fixed in 0721077).
