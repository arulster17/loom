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

- [ ] **OpenSSH** on the laptop (`ssh`, `ssh-keygen`): the runner drives pods over SSH.
- [ ] **Results database** is up and migrated, the same one as every other real run. It
  holds every recorded cent of spend (the overall cap is computed from it):

  ```sh
  docker compose up -d postgres && uv run bench db upgrade
  ```

- [ ] **Clean git tree.** Provenance records the commit and a dirty flag. Commit first.
- [ ] **Plan is under the cap**:

  ```sh
  uv run bench plan bench/experiments/qwen3-8b-vllm-vs-sglang-runpod.yaml   # exit 0 = accepted
  ```

  Check "estimated spend" and "worst case (all hosts to TTL)" against "effective cap".
  The hourly rate per pod is prices.yaml's on-demand price plus container disk; at run
  time the guard accrues the larger of that and the pod's API `costPerHr`. Expected
  today: `runpod-smoke` $0.25 estimate, $0.73 worst case, $2 cap;
  `qwen3-8b-vllm-vs-sglang-runpod` $7.92, $13.21, $40 (two pods, one per engine);
  `llama-3.3-70b-tp4-runpod` $11.73, $17.58, $45. Exit 3 means refused: lower the load
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

Exit codes: 0 ok, 1 failed, 3 refused, 4 stopped before a step that would pass the cap,
5 hard budget abort. Run one experiment at a time, as on AWS.

## 3. Monitoring

Spend: the `bench_experiments` and `bench_spend` queries in
[runbook.md](runbook.md#3-monitoring). Progress: the runner's log lines and
`results/<experiment id>/events.jsonl` (`provisioned`, `engine_started` with its stages
and `system` facts, quality, gates). The pod's API price and the prices.yaml rate it was
compared with are in the host's `accrual_basis`; each run's provenance records the
as-run price with basis `observed_api` and the datacenter.

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

### Pod refused for its price

`pod … costs … $/h, above prices.yaml … x 1.25: terminated`. RunPod's price moved. Check
the console price, update `bench/prices.yaml` (and its `last_checked`), plan again.

### No capacity, or a pod that never comes up

A failed create surfaces as `RunPod POST /pods failed (...)`. If the request failed
without a clear answer from RunPod, the provider terminates any pod created under that
name anyway. A pod without an SSH address after `pod_ready_timeout_s` (900 s), or whose
sshd does not answer within `ssh_online_timeout_s` (600 s), fails the experiment and is
terminated. L40S Secure stock varies by datacenter and over the day: retry later, or pin
`provider.data_center_ids` to one with stock.

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

`bench/experiments/runpod-smoke.yaml`: Qwen3-8B on vLLM v0.30.0, one Secure 1× L40S pod,
one short closed-loop job, one repetition, `max_spend: "$2"`, TTL 40 min. It is the
cheapest run that exercises the whole provider once; its results are flagged untrusted
and never published.

```sh
uv run bench plan bench/experiments/runpod-smoke.yaml
uv run bench run bench/experiments/runpod-smoke.yaml
```

Checklist:

- [ ] `events.jsonl` has an `engine_started` event with `system.job_isolation == "ok"`.
- [ ] The cold start stages include `pod_created`, `image_pulled`, `sshd_ready`,
  `ssh_online`, `weights_ready`, `engine_healthy` and `first_token`.
- [ ] The run completed, and its `result.json` and `gpu.csv` are under
  `s3://<bucket>/runs/<experiment id>/<run id>/`.
- [ ] Provenance records `cloud: runpod`, `instance_type: l40s-x1`, basis `observed_api`,
  and CUDA and driver versions.
- [ ] The recorded API `costPerHr` matches prices.yaml (within `max_price_ratio`).
- [ ] The pod shows as terminated (gone from the pod list), not exited.
- [ ] Section 5 passes.

Smoke test result: pending
