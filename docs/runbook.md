# Runbook: Phase 0 GPU runs on AWS

Operating the Benchmark Lab against real GPUs: preflight, running, watching spend,
incidents, reproducing, publishing and tearing everything down. One-time account setup
(quotas, HF token secret, Terraform, runner policy, `AwsSettings`) is in
[aws-setup.md](aws-setup.md); do that first.

Money rails, for reference: $150 overall, $50 per experiment (`bench/budget.yaml`), and
each experiment's own `budget.max_spend`. How they act is in
[benchmark-lab.md](benchmark-lab.md#budget-rails).

Commands below assume the repo root, region `us-east-1`, the default name prefix
`loom-bench` and an identity with the `loom-bench-runner` policy unless a step says it
needs an admin identity.

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
  and read the notes (unverified EBS price, spot multiplier). Expected today:
  `qwen3-8b-vllm-vs-sglang` $14.22 estimate, $18.56 worst case, $40 cap;
  `llama-3.3-70b-tp4` $22.38, $35.86, $45. Exit 3 means refused: lower the load points or
  `budget.ttl_minutes`, never the caps.

## 2. Running an experiment

```sh
caffeinate -i uv run bench run bench/experiments/qwen3-8b-vllm-vs-sglang.yaml   # macOS: keep awake
```

What happens: the plan prints again and is re-checked; one spot host is launched in the
cheapest AZ; user-data arms the TTL shutdown; the start script pulls the image,
downloads weights at the pinned revision with the HF token, starts the engine and waits
for one token; each load point × repetition runs as a `bench job run` on the host; the
host is terminated at the end, also on errors and Ctrl-C. Exit codes: 0 ok, 1 failed,
3 refused, 4 stopped before a step that would pass the cap, 5 hard budget abort.

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
   counts toward the $150 overall cap.

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
  aws ec2 terminate-instances --region us-east-1 --instance-ids <instance id>
  ```

  The runner policy allows this for `loom:managed=true` instances. The experiment row
  stays `running`; it is history only.

### Orphaned resources

`bench reap` and the Lambda only act on resources whose `loom:ttl` has passed (or that are
older than 24 h with no readable TTL).

```sh
uv run bench reap --dry-run      # what is expired: DB-recorded hosts + tagged EC2 instances
uv run bench reap                # terminate them; DB rows marked reaper / self-ttl
```

Without `LOOM_AWS_CONFIG`, `bench reap` only handles database rows. Console checks,
always by tag:

```sh
aws ec2 describe-instances --region us-east-1 --filters Name=tag:loom:managed,Values=true \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]'
aws ec2 describe-volumes --region us-east-1 --filters Name=tag:loom:managed,Values=true \
  --query 'Volumes[].[VolumeId,State,CreateTime]'
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
a host, is planned and capped like any run), and compares: exit 0 within normal variance,
6 outside. `bench report` prints the command for each result.

## 6. Publishing results

1. Back up the results database; it is the only copy of the runs:

   ```sh
   docker compose exec postgres pg_dump -U loom loom > loom-$(date +%F).sql
   ```

2. Export the snapshot (newest completed experiment of each name, or pick with `-e`):

   ```sh
   uv run bench site export --out site/data      # -e <experiment id> ... to choose
   uv run bench site build
   python -m http.server 8000 --directory site/_build
   ```

3. Review the numbers, warnings and provenance links, then commit `site/data/` on a
   branch and open a pull request. Merging to `main` runs `.github/workflows/site.yml`,
   which deploys to GitHub Pages once Pages is enabled ([site.md](site.md#deployment)).

Snapshots publish experiment specs and provenance in full: keep secrets out of them.

## 7. Tearing down the whole stack

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
