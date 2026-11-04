# Runbook: Phase 0 GPU runs on AWS

Operating the Benchmark Lab against real GPUs: preflight, running, watching spend,
incidents, reproducing, publishing and tearing everything down. One-time account setup
(quotas, HF token secret, Terraform, runner policy, `AwsSettings`) is in
[aws-setup.md](aws-setup.md); do that first.

Money rails, for reference: $249.75 overall (raised from $149.75 on 2026-11-02), $50 per experiment (`bench/budget.yaml`), and
each experiment's own `budget.max_spend`. How they act is in
[benchmark-lab.md](benchmark-lab.md#budget-rails).

Commands below assume the repo root, region `us-east-1`, the default name prefix
`loom-bench` and an identity with the `loom-bench-runner` policy unless a step says it
needs an admin identity. With `extra_regions` configured
([aws-setup.md](aws-setup.md#hosts-in-more-than-one-region)), a host can run in
us-east-2 or us-west-2 too. Its events (`provisioned`, then the engine start's
`system.region`) and its provenance `region` say which. Run the per-instance commands
there with `--region`, and the "nothing running" checks in each region:
`for r in us-east-1 us-east-2 us-west-2; do aws ec2 describe-instances --region $r …; done`.

This is the `aws_ec2` path, now secondary for GPU benches: the account's GPU spot quota
is 0 (an increase is pending with AWS), and the real runs are on RunPod
([runbook-runpod.md](runbook-runpod.md)). While RunPod has no 1x L40S stock, the
perf-track runs have g6e.xlarge on-demand twins here (section 7b).

## 1. Preflight checklist

Run through all of it before every real run.

- [ ] **GPU quota** covers the host's vCPUs (g6e.xlarge 4, g6e.12xlarge 48):

  ```sh
  aws service-quotas get-service-quota --region us-east-1 --service-code ec2 \
    --quota-code L-3819A6DF --query Quota.Value     # spot G/VT; L-DB2E81BA for on-demand
  ```

- [ ] **Instance type allowed** by the runner policy: it must be in
  `allowed_instance_types` (`infra/aws/bench/variables.tf`, default g5.xlarge, g6.xlarge,
  g6e.xlarge, g6e.2xlarge, g6e.12xlarge). Otherwise `RunInstances` is denied.
- [ ] **HF token secret** exists (admin identity; the runner policy cannot read secrets),
  and the HF account behind it has accepted the license of gated models (Llama 3.3):

  ```sh
  aws secretsmanager describe-secret --region us-east-1 --secret-id loom/hf-token --query Name
  ```

- [ ] **Settings from Terraform outputs** are current and load:

  ```sh
  mkdir -p ~/.config/loom
  terraform -chdir=infra/aws/bench output -raw aws_settings_yaml > ~/.config/loom/aws.yaml
  export LOOM_AWS_CONFIG=~/.config/loom/aws.yaml
  uv run python -c "from loom_bench.providers.aws_ec2 import load_aws_settings as s; print(s())"
  aws sts get-caller-identity      # the identity the runner policy is attached to
  ```

- [ ] **Reaper Lambda is live**, scheduled, and not in dry-run mode (admin identity):

  ```sh
  aws lambda get-function-configuration --region us-east-1 --function-name loom-bench-reaper \
    --query '{state: State, env: Environment.Variables}'   # LOOM_REAPER_DRY_RUN must be "false"
  aws events describe-rule --region us-east-1 --name loom-bench-reaper --query State   # ENABLED
  aws lambda invoke --region us-east-1 --function-name loom-bench-reaper \
    --cli-binary-format raw-in-base64-out --payload '{"dry_run": true}' /dev/stdout
  ```

  With `extra_regions`, the env must also show `LOOM_REAPER_REGIONS` listing them, and the
  invoke's `regions` lists them too. Otherwise the Lambda reaps only us-east-1: apply the
  Terraform first.

- [ ] **Nothing managed is already running** (an old host would share the account's spot
  quota and keep billing):

  ```sh
  aws ec2 describe-instances --region us-east-1 \
    --filters Name=tag:loom:managed,Values=true \
              Name=instance-state-name,Values=pending,running,stopping,stopped \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`loom:ttl`]|[0].Value]'
  ```

- [ ] **Results database** is up and migrated. It holds every recorded cent of spend
  (the overall cap is computed from it), so run against the same database every time:

  ```sh
  docker compose up -d postgres && uv run bench db upgrade
  ```

- [ ] **Clean git tree.** Provenance records the commit and a dirty flag, and
  `bench reproduce` warns when they differ. Commit first.
- [ ] **Plan is under the cap**:

  ```sh
  uv run bench plan bench/experiments/qwen3-8b-vllm-vs-sglang.yaml    # exit 0 = accepted
  ```

  Check "estimated spend" and "worst case (all hosts to TTL)" against "effective cap",
  and read the notes (spot multiplier). The hourly rate per host is the budget accrual
  rate (spot x 1.25 plus EBS), not the cost reports use. Expected today:
  `qwen3-8b-vllm-vs-sglang` $16.56 estimate, $18.56 worst case, $40 cap;
  `llama-3.3-70b-tp4` $28.04, $35.86, $45. Exit 3 means refused: lower the load points or
  `budget.ttl_minutes`, never the caps.

## 2. Running an experiment

```sh
caffeinate -i uv run bench run bench/experiments/qwen3-8b-vllm-vs-sglang.yaml   # macOS: keep awake
```

What happens: the plan prints again and is re-checked; one spot host is launched in the
cheapest AZ; user-data arms the TTL shutdown; the start script pulls the image,
downloads weights at the pinned revision with the HF token, starts the engine and waits
for one token; each load point × repetition runs as a `bench job run` on the host; the
host is terminated at the end, also on errors and Ctrl-C. Exit codes: 0 ok, 1 failed
(including every run failing), 3 refused, 4 stopped before a step that would pass the cap,
5 hard budget abort, 8 finished but some load runs or quality evals failed.

Run one experiment at a time. Each plan assumes the remaining overall budget is its own;
the guard still trips when spend across all experiments reaches the overall cap, but two
concurrent runs would both have been planned against the same remainder. Keep the
results database up for the whole run: if an accrual write fails, the guard trips and
aborts rather than run blind.

## 3. Monitoring

Spend, from the results database (micro-dollars; `bench_spend` gets a row per host every
`accrual_interval_s`):

```sh
docker compose exec postgres psql -U loom -d loom -c \
  "SELECT name, status, spent_micros, budget_micros, abort_reason
     FROM bench_experiments ORDER BY created_at DESC LIMIT 5;"
docker compose exec postgres psql -U loom -d loom -c \
  "SELECT recorded_at, resource_id, amount_micros, basis
     FROM bench_spend ORDER BY recorded_at DESC LIMIT 5;"
```

The `basis` column records the price used (spot price and its timestamp, multiplier,
EBS rate). Progress: the runner's log lines, and `results/<experiment id>/events.jsonl`
(provisioned, engine started, spot interruptions, quality, gates).

Host side:

- SSM command output: `s3://<bucket>/ssm/<experiment id>/<instance id>/`.
- On the host, `/var/log/loom/` (`pull.log`, `weights.log`, `engine-run.log`) and
  `/var/lib/loom/jobs/<run id>/client.log`, through Session Manager
  (`aws ssm start-session --target <instance id>`; needs an admin identity, the runner
  policy does not allow sessions).
- AWS-side cost appears in Cost Explorer with a delay of hours; the database is the live
  number.

## 4. Incidents

### Hard budget abort (exit 5) or graceful stop (exit 4)

The runner has already torn every host down and marked the experiment `aborted` with the
reason. Then:

1. Confirm nothing is running (the `describe-instances` command in the preflight).
2. Read `abort_reason` and `spent_micros`. Recorded spend can exceed the cap by about one
   accrual interval of the live hosts plus the teardown time.
3. Completed runs stay in the database; reports skip runs that did not complete. Fix the
   plan (fewer points, shorter durations) before running again. Spend already recorded
   counts toward the overall cap.

### Spot interruption

Recorded automatically: the in-flight run gets status `interrupted`, `events.jsonl` gets
a `spot_interruption` event, and the cell is retried once on a new host if the budget
allows (completed repetitions are not re-run). A second interruption fails the
experiment (exit 1). Options: run again later, or set `provider.market: on_demand`
(needs the on-demand quota `L-DB2E81BA`; plan it first).

### Stuck instance or hung run

- SSM commands have timeouts (engine start: `ready_timeout_s` 1800 s plus 3600 s for a
  cold weight download; jobs: duration + warmup + request timeout + drain timeout + 900 s,
  or `job_timeout_s` 7200 s for request-count jobs). A timed-out job marks its run
  `failed` and the runner moves on; a timed-out engine start fails the experiment, and
  hosts are torn down either way.
- If the runner itself hangs, press Ctrl-C once: hosts are torn down and the experiment is
  marked `failed`.
- If the runner process is gone (laptop slept, crashed), terminate the instance now
  rather than waiting for its TTL (8 h for the Qwen experiment):

  ```sh
  aws ec2 terminate-instances --region <host region> --instance-ids <instance id>
  ```

  The runner policy allows this for `loom:managed=true` instances. The experiment row
  stays `running`; it is history only.

### No capacity (`InsufficientInstanceCapacity`)

A launch tries each subnet's AZ once (a sweep) and keeps the first instance it gets. When
every AZ refuses for capacity, the provider sweeps again for up to `capacity_wait_s`
(1800 s, as RunPod's; `LOOM_AWS_CAPACITY_WAIT_S` overrides it, 0 fails on the first
sweep). It waits 30, 60, 120, then 240 s between sweeps (RunPod's backoff) and makes one
last sweep at the deadline. Each sweep logs one line with the time waited:

```text
AWS g6e.xlarge capacity: none in us-east-1c, us-east-1a, us-east-1d, us-east-1b (InsufficientInstanceCapacity); not offered in us-east-1e, us-east-1f (waited 0 s, 1800 s left); next sweep in 30 s
AWS g6e.xlarge capacity: none in us-east-1c, us-east-1a, us-east-1d, us-east-1b (InsufficientInstanceCapacity) (waited 30 s, 1770 s left); next sweep in 60 s
AWS g6e.xlarge capacity: launched i-0abc… in us-east-1d (waited 90 s)
```

**With `extra_regions`,** a sweep tries every AZ of us-east-1, then of each extra region
in order, before it waits. The vCPU quota is per region: a `VcpuLimitExceeded` skips
the rest of that region for this sweep only. The 10-minute quota wait below applies only
when every region refused for its quota.

```text
AWS g6e.xlarge capacity: none in us-east-1c, us-east-1a, us-east-1d, us-east-1b, us-west-2a, … (InsufficientInstanceCapacity); vCPU quota held in us-east-2 (VcpuLimitExceeded); not offered in us-east-1e, us-east-1f (waited 0 s, 1800 s left); next sweep in 30 s
AWS g6e.xlarge capacity: launched i-0def… in us-west-2b (waited 30 s)
```

- **Nothing exists or bills while waiting.** No instance exists between sweeps, and the
  budget guard only starts accruing a host once a sweep returns it. The host's launch
  time and TTL are those of the sweep that got it, so the wait never shortens its life.
  A run that gives up (`no capacity for g6e.xlarge after waiting 1800 s: …`, exit 1) has
  spent $0 and has no host.
- **What is waited out:** `InsufficientInstanceCapacity`, `InsufficientHostCapacity`,
  `InsufficientCapacity` and `SpotMaxPriceTooLow`. `Unsupported` means the AZ does not
  offer the type at all (g6e.xlarge is offered in us-east-1a to 1d only, per
  `aws ec2 describe-instance-type-offerings`), so that AZ is dropped from later sweeps,
  and a sweep with only `Unsupported` AZs fails at once.
- **The vCPU quota (`VcpuLimitExceeded`) is waited out for up to 10 min.** AWS keeps
  counting an instance against the quota for a short while after it reports
  `terminated`: aws-smoke-trtllm (d549221f) launched its third host 0.3 s after the
  second's teardown saw `terminated` and was refused. The quota is per account and
  region, so a sweep skips the rest of that region at the first refusal (with no other
  region configured, the sweep ends there), and the next sweep follows on the same
  backoff. If
  the refusal lasts 600 s (`QUOTA_WAIT_S`, or less when `capacity_wait_s` ends first),
  the run fails with `VcpuQuotaHeld`. That means something else holds the quota (another
  instance not yet terminated) or the quota is below one instance's vCPUs. No instance
  exists while it waits.
- **Network errors are retried, and RunInstances is idempotent.** A sweep call that gets no
  answer is retried on the same backoff, at most 4 times in a row. These are
  `ReadTimeoutError`, `ConnectTimeoutError`, `EndpointConnectionError` and
  `ConnectionClosedError`; 014218c9 lost its experiment to one RunInstances read timeout.
  - **RunInstances is resent unchanged,** with the same `ClientToken`. If the lost call did
    launch an instance, EC2 returns that instance, which is adopted, never doubled.
  - **When the retries run out,** the provider looks up the token (`describe-instances`
    filter `client-token`) and terminates anything it launched. The run then fails with
    `Ec2Unreachable`, saying what was found.
  - **Teardown** resends `terminate-instances` after a network error (5 tries over about
    2 min) and polls through `describe` timeouts while it waits for `terminated`.
- **Everything else fails at once,** as before: auth errors, a bad AMI, invalid
  parameters.
- **A longer wait** costs nothing: `LOOM_AWS_CAPACITY_WAIT_S=7200 uv run bench run …`.
  Keep the laptop awake (`caffeinate -i`) for the whole wait.

### Orphaned resources

`bench reap` and the Lambda only act on resources whose `loom:ttl` has passed (or that are
older than 24 h with no readable TTL).

```sh
uv run bench reap --dry-run      # what is expired: DB-recorded hosts + tagged EC2 instances
uv run bench reap                # terminate them; DB rows marked reaper / self-ttl
```

Without `LOOM_AWS_CONFIG`, `bench reap` only handles database rows; with it, it reaps
every region in the settings. Console checks, always by tag and in every host region
(EC2 lists only its own region):

```sh
for r in us-east-1 us-east-2 us-west-2; do
  aws ec2 describe-instances --region "$r" --filters Name=tag:loom:managed,Values=true \
    --query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime,Placement.AvailabilityZone]'
  aws ec2 describe-volumes --region "$r" --filters Name=tag:loom:managed,Values=true \
    --query 'Volumes[].[VolumeId,State,CreateTime,AvailabilityZone]'
done
```

In the console: EC2 → Instances / Volumes, filter `tag:loom:managed = true`; Resource
Groups → Tag Editor with `loom:managed = true` (hosts) or `loom:stack = loom-bench`
(everything Terraform created). Reaper logs: CloudWatch log group
`/aws/lambda/loom-bench-reaper`. Hosts tagged with an experiment id
(`loom:experiment`) can be matched to `bench_experiments.id`.

## 5. Reproducing a run

```sh
uv run bench reproduce <run id>                                   # from the database
uv run bench reproduce results/<exp id>/runs/<run id>/provenance.json   # or a file
```

It rebuilds the cell from the stored config, re-runs that load point and repetition as a
new experiment `<name>--reproduce` on the same provider (so an AWS reproduction provisions
a host, is planned and capped like any run), and compares by config hash: exit 0 within
normal variance, 6 outside or when the config hash no longer matches the original's. `bench report` prints the command for each result.

## 6. Publishing results

1. Back up the results database; it is the only copy of the runs:

   ```sh
   docker compose exec postgres pg_dump -U loom loom > loom-$(date +%F).sql
   ```

2. Pin the experiments to publish by full id in `site/config.yaml`
   (`publish.experiments`), then export exactly those (with `-e` the snapshot is a
   preview; the deploy workflow's `bench site build --require-pinned` refuses it):

   ```sh
   uv run bench site export --out site/data      # the pinned experiments
   uv run bench site build
   python -m http.server 8000 --directory site/_build
   ```

3. Review the numbers, warnings and provenance links, then commit `site/data/` on a
   branch and open a pull request. Merging to `main` runs `.github/workflows/site.yml`,
   which deploys to GitHub Pages once Pages is enabled ([site.md](site.md#deployment)).

Snapshots publish experiment specs and provenance in full: keep secrets out of them.

## 7. The Qwen3-8B g6e.xlarge on-demand run (ran 2026-10-10/11)

`aws-smoke-g6e` then `qwen3-8b-aws-g6e`: one g6e.xlarge on-demand host each, the first
GPU runs on this path ([benchmark-lab.md](benchmark-lab.md#qwen3-8b-on-aws-g6exlarge-proposed)
has the design, the replicate arithmetic and the plans). Preflight as of 2026-10-10, free
read-only calls:

| Check | Found |
|---|---|
| Identity | `arn:aws:iam::735792833179:user/arul-cli`, with `AdministratorAccess` (the narrower `loom-bench-runner` policy is not attached; it would also work) |
| Quotas | on-demand G/VT `L-DB2E81BA` = 4 vCPU (one g6e.xlarge), spot `L-3819A6DF` = 0 |
| g6e.xlarge offerings | us-east-1a, b, c, d; not 1e or 1f (two of the six configured subnets: `Unsupported`, tried and skipped) |
| On-demand price | $1.861/h (Price List API, effective 2026-10-01), as in `bench/prices.yaml` |
| DLAMI | `ami-028357be7d5b15c53`, Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261006: driver 595.91.07, CUDA 13.2, NVMe at `/opt/dlami/nvme`, SSM agent 3.3.4793 |
| Stack | bucket `loom-bench-5791dcdd765e549af6838bf64b` (lifecycle on `runs/`, `ssm/`), instance profile `loom-bench-instance`, SG with no ingress and all egress, secret `loom/hf-token`, reaper Lambda active (`LOOM_REAPER_DRY_RUN=false`, rule `rate(15 minutes)` ENABLED, dry-run invoke returned `{"reaped": [], "dry_run": true}`) |
| Managed instances | none |
| `~/.config/loom/aws.yaml` | **missing**: create it (below) |

The only quota-sized host: if a managed instance is still running or shutting down, the
next RunInstances fails with `VcpuLimitExceeded` (waited out for at most 10 min, section 4),
so run the smoke and the real run one after the other, never together, and check nothing
managed is alive first.

```sh
mkdir -p ~/.config/loom
terraform -chdir=infra/aws/bench output -raw aws_settings_yaml > ~/.config/loom/aws.yaml
export LOOM_AWS_CONFIG=~/.config/loom/aws.yaml
export LOOM_DATABASE_URL=sqlite:///$PWD/results/loom.db     # the main checkout's DB
uv run bench plan bench/experiments/aws-smoke-g6e.yaml       # $3.48, worst $4.71, cap $5
caffeinate -i uv run bench run bench/experiments/aws-smoke-g6e.yaml --out results/
```

After the smoke, before the real run (each from `results/<id>/events.jsonl` and the DB):

- exit 0; one `provisioned` event at `hourly_micros` 1882918 and `bench_spend.basis`
  with market `on_demand` (prices.yaml on-demand + 200 GB gp3);
- two `engine_started` events (bf16 cold, fp8-kv8 warm) with `driver_version`,
  `cuda_version`, `image_digest`; the cold one with `ttl_shutdown_at` and
  `ttl_shutdown_lead_s` between 0 and 60 (user-data schedules it in whole minutes);
  note the `ami` in that event;
- chat-sharegpt runs for both cells: each search's first point (12 req/s) fails with
  aborted stragglers, then 4, then a pass and a bisection point; no failed runs;
- three `quality` passes per cell (`replicates: 3`), no `quality_failed` or
  `divergence_failed`, one `gate` event for fp8-kv8 against bf16 (its verdict means
  nothing at 4 items);
- the instance is `terminated` (the runner did it), `bench reap --dry-run` is clean;
- from the start stages: image pull, weights, engine init and eval-pass times, to check
  the real run's plan (cold start planned at 14.5 min).

Then pin the smoke's AMI so the real run gets the same image
(`export LOOM_AWS_AMI_ID=ami-...`), check the plan and run:

```sh
uv run bench plan bench/experiments/qwen3-8b-aws-g6e.yaml    # $8.90, worst $12.24, cap $12.50
caffeinate -i uv run bench run bench/experiments/qwen3-8b-aws-g6e.yaml --out results/
uv run bench compare 7a8237d0-9917-47e7-b93e-8cb0230e0059 <aws id> --match-by cell_key
uv run bench report -e <aws id>
```

**As run (2026-10-10/11).**

- **Smokes:** a7bc3123 ($0.58) found the client-mount bug (load jobs could not open
  tokenizer.json, [aws-setup.md](aws-setup.md#load-and-eval-jobs-on-the-host)).
  80ff5648 ($1.67) passed every item above.
- **Real run:** 95cde129, $6.39, exit 0
  ([result](benchmark-lab.md#qwen3-8b-on-aws-g6exlarge-result-95cde129-2026-10-11)).
- **Capacity:** on-demand capacity was the operational issue. Four launch attempts got
  `InsufficientInstanceCapacity` in every g6e AZ before any host existed (exit 1, $0, a
  `failed` experiment row each). A retry 1-11 min later got a host each time, so retry
  that failure (and only it) rather than giving up. The provider now does this itself
  for up to `capacity_wait_s` (section 4, No capacity). A
  `VcpuLimitExceeded` instead means the previous host still holds the quota (shutting
  down, or just terminated); the provider waits that out for up to 10 min.
- **Spend:** Cost Explorer is not enabled for the CLI user (`AccessDenied`). Check AWS-side
  spend from the instance's launch and stop times, or in the console's Billing pages.
- **Never edit the checkout a run is using:** the host scripts are read from disk at
  each step. An edit mid-run broke a7bc3123's last eval.

### 7a. vLLM vs SGLang on one host (ran 2026-10-11/12)

`aws-smoke-g6e-sglang`, then `qwen3-8b-sglang-aws-g6e`: one g6e.xlarge on-demand host
each, vLLM v0.30.0 BF16 (95cde129's bf16 cell) and then SGLang v0.5.21 BF16 through a warm
restart that switches image; SGLang is gated against vLLM in-run. The design is in
[benchmark-lab.md](benchmark-lab.md#qwen3-8b-vllm-vs-sglang-on-aws-g6exlarge-proposed).
Preflight as in section 7, free read-only calls (2026-10-11):

- AMI ami-028357be7d5b15c53 is `available` (deprecation 2028-10-07).
- The on-demand G/VT quota is 4 vCPU.
- No managed instances.

Same AMI as 95cde129, so all runs share driver 595.91.07 and CUDA 13.2:

```sh
export LOOM_AWS_CONFIG=~/.config/loom/aws.yaml LOOM_AWS_AMI_ID=ami-028357be7d5b15c53
export LOOM_DATABASE_URL=sqlite:///$PWD/results/loom.db     # the main checkout's DB
uv run bench plan bench/experiments/aws-smoke-g6e-sglang.yaml     # $3.62, worst $3.92, cap $4.00
caffeinate -i uv run bench run bench/experiments/aws-smoke-g6e-sglang.yaml --out results/
```

After the smoke, before the real run:

- exit 0, one `provisioned` event at `hourly_micros` 1882918;
- two `engine_started` events: bf16 cold (vLLM image, driver 595.91.07,
  `ttl_shutdown_lead_s` 0-60), then sglang warm with `image_digest`
  `lmsysorg/sglang@sha256:b1259f3e…`; from the warm start's stages, SGLang's image pull
  and the offline weight check (the planner allows 9.7 min for the warm start);
- chat-sharegpt on both cells: 12 req/s fails with stragglers aborted at the drain, then
  4, then a pass and a bisection point; no failed runs; every run's `server` metrics have
  `missing: []` (SGLang's names scraped) and `client.client_saturated_count` 0 (4 vCPU);
- no engine exit or out-of-memory on the 32 GiB host (`engine_started` reached, no
  `engine_start_failed`, no failed run);
- three `quality` passes per cell, no `quality_failed` or `divergence_failed`, one `gate`
  event for sglang against bf16 (its verdict means nothing at 4 items), and
  tool_calling_strict scored on sglang;
- the instance `terminated`, and `bench reap --dry-run` clean.

Then:

```sh
uv run bench plan bench/experiments/qwen3-8b-sglang-aws-g6e.yaml  # $9.46, worst $9.73, cap $9.75
caffeinate -i uv run bench run bench/experiments/qwen3-8b-sglang-aws-g6e.yaml --out results/
uv run bench report -e <id> --out reports/8b-aws-engines-v1
uv run bench compare <id> <id> --match-by workload --cell-a bf16 --cell-b sglang   # the engines
uv run bench compare 95cde129 <id>                                # vLLM bf16 vs 95cde129's
```

The provider waits out `InsufficientInstanceCapacity` itself, for up to `capacity_wait_s`
(no host, $0; section 4, No capacity). If it gives up, run again later.

**As run.**

- **Launcher:** a script ran `bench run` under `caffeinate` with the AMI pinned and
  retried only the no-host capacity failure every 10 min (cap 4 h). The smoke got a host
  on the fourth attempt (9700c9cb, e5387f30, 26d4453e: `InsufficientInstanceCapacity` in
  1a, 1c, 1d; $0 each), the real run on its first.
- **Smoke:** deb3ba58, $1.96, passed every item above with no code change (SGLang's image
  pull 6.9 min inside a 10.5 min warm restart).
- **Real run:** fccf9ed3, $7.00, exit 0
  ([result](benchmark-lab.md#qwen3-8b-vllm-vs-sglang-on-aws-g6exlarge-result-fccf9ed3-2026-10-12)).
  Wait until the previous host is `terminated` (`aws ec2 wait instance-terminated`)
  before the next launch: the quota fits one g6e.xlarge.
- **SGLang's retraction counter** is absent until SGLang first retracts a request, so a
  run's server metrics list `preemptions` as missing; read it as zero.

### 7b. The perf-track runs on AWS (proposed 2026-11-02, not approved, not run)

RunPod Secure 1x L40S was out of stock all of 2026-11-02 (191 refused creates), so the
three perf-track runs have AWS twins on g6e.xlarge on-demand (the same L40S, pinned, no
fallback). Each twin is its RunPod spec with only the provider block, the budget and, for
TensorRT-LLM, one host group changed (`bench/tests/aws/test_aws_perf_track.py`). The
explainers have the design: [roofline-and-profiling.md](perf/roofline-and-profiling.md),
[speculative-decoding.md](perf/speculative-decoding.md),
[tensorrt-llm.md](perf/tensorrt-llm.md).

Quotes (`bench plan` against a copy of the results DB, $96.57 already spent, $153.18 left
under the overall cap; every real run is under the $50 per-experiment cap):

| Spec | Hosts | Estimate | Worst case (hosts to TTL) | Cap |
|---|---|---|---|---|
| `aws-smoke-profile` | 1, 42.6 min of 55 | $1.34 | $1.73 | $1.75 |
| `qwen3-8b-profile-aws` | 1, 2.29 h of 2.75 | $4.32 | $5.18 | $5.25 |
| `aws-smoke-specdec` | 1, 1.56 h of 1.92 | $2.94 | $3.61 | $3.75 |
| `qwen3-8b-specdec-aws` | 1, 7.46 h of 8 | $14.04 | $15.06 | $15.25 |
| `aws-smoke-trtllm` | 3 in a row, 65 min TTL each | $4.52 | $6.12 | $6.25 |
| `qwen3-8b-trtllm-vs-vllm-aws` | 3 in a row, 4.75 h TTL each | $20.85 | $26.83 | $27 |
| All six | | $48.01 | $58.53 | $59.25 |

New on this path, each covered offline first (moto and the host simulator):

- **Profile traces:** a profiled vLLM container bind-mounts `/var/lib/loom/profiles` at the
  same host path; `collect_profile.sh` archives it as root over SSM and uploads it through
  a presigned PUT.
- **Several hosts in one run** (TensorRT-LLM's three host groups): teardown now waits
  until each instance is `terminated` (up to 10 min), so the next launch does not hit
  `VcpuLimitExceeded`. Runs still go one at a time, and only after `aws ec2 wait
  instance-terminated` on the previous run's host.
- **TensorRT-LLM in Docker:** NVIDIA's `--ulimit memlock=-1 --ulimit stack=67108864`; a
  17.3 GB image (~8 min to pull); `provider.engine_env` now exists on aws_ec2 for the
  `TRTLLM_ENABLE_PDL: "0"` fallback.
- **32 GiB host RAM** (the pod had 94): TensorRT-LLM's model load and evals on it are not
  measured yet. The smoke shows it first.
- **Waiting for capacity:** on 2026-11-02 at 22:20, `aws-smoke-profile` failed at once
  with `InsufficientInstanceCapacity` in us-east-1a to 1d (1e and 1f do not offer
  g6e.xlarge). The provider now sweeps the AZs again for up to `capacity_wait_s`
  (`LOOM_AWS_CAPACITY_WAIT_S`, default 1800 s) on a 30/60/120/240 s backoff, with no
  instance and no spend while waiting (section 4, No capacity). Only one g6e.xlarge fits
  the 4-vCPU quota. `VcpuLimitExceeded` is waited out for at most 10 min (the quota trails
  `terminated`, d549221f), then fails with `VcpuQuotaHeld`.

Run each smoke before its real run, with the same pinned AMI as 95cde129 and fccf9ed3
(driver 595.91.07, CUDA 13.2) and the preflight in section 7a:

```sh
export LOOM_AWS_CONFIG=~/.config/loom/aws.yaml LOOM_AWS_AMI_ID=ami-028357be7d5b15c53
export LOOM_DATABASE_URL=sqlite:///$PWD/results/loom.db     # the main checkout's DB
uv run bench plan bench/experiments/aws-smoke-trtllm.yaml          # one spec at a time
caffeinate -i uv run bench run bench/experiments/aws-smoke-trtllm.yaml --out results/
aws ec2 wait instance-terminated --region us-east-1 --instance-ids <last host>
```

To wait longer than 30 min for a host, set `LOOM_AWS_CAPACITY_WAIT_S` (e.g. `7200`) on
the `bench run`; if the wait gives up (exit 1, no host, $0), run again later. After
each smoke, check its RunPod twin's list (in the RunPod smoke spec's header), plus on AWS:
`engine_started` for every cell, a `profile` event per window and no `profile_failed`
(profile), `spec_decode` absent from every run's `server.missing` (specdec), three
`provisioned` events each after a `torn_down` (TensorRT-LLM), and no engine exit or
out-of-memory on the 32 GiB host.

## 8. Tearing down the whole stack

1. Make sure no managed instance or volume is left (section 4).
2. Detach the runner policy from every user or role it was attached to (IAM refuses to
   delete an attached policy):

   ```sh
   aws iam detach-user-policy --user-name <you> \
     --policy-arn "$(terraform -chdir=infra/aws/bench output -raw runner_policy_arn)"
   ```

3. Empty the bucket (Terraform does not force-delete a non-empty bucket):

   ```sh
   aws s3 rm "s3://$(terraform -chdir=infra/aws/bench output -raw bucket)" --recursive
   ```

4. Destroy (admin identity):

   ```sh
   terraform -chdir=infra/aws/bench destroy -var owner=<your-name>
   ```

5. Delete the HF token secret, which Terraform only reads, after the destroy, then revoke
   the token on huggingface.co:

   ```sh
   aws secretsmanager delete-secret --region us-east-1 --secret-id loom/hf-token \
     --recovery-window-in-days 7
   ```

6. Remove `~/.config/loom/aws.yaml`. The spot service-linked role
   (`AWSServiceRoleForEC2Spot`) stays in the account; it costs nothing.
