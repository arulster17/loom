# AWS setup for the Benchmark Lab

Phase 0 runs each benchmark host as one EC2 GPU VM (spot, or on-demand as a
fallback) from the AWS Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 24.04).
The engine runs in Docker, and the runner drives the host over SSM, so the host
has no inbound ports and no SSH keys. Everything below is done once per account.
The whole Phase 0 budget is $150, at most $50 per experiment, so the steps put
the safety rails up before the first GPU starts.

This account's GPU spot quota is 0 (an increase is pending with AWS), so Phase 0 GPU
benchmarks run on RunPod Secure Cloud on-demand instead
([runbook-runpod.md](runbook-runpod.md)). The EC2 path below stays as the secondary one,
and `infra/aws/bench` is applied: RunPod runs use its S3 bucket through presigned URLs.

Region: `us-east-1` (prices in `bench/prices.yaml`).

## 0. What you need locally

- AWS CLI v2, signed in as an identity that can create IAM roles, S3 buckets and
  Lambda functions. You only need that for `terraform apply`. Day-to-day runs use the
  narrower runner policy from step 4.
- Terraform >= 1.6 (the AWS provider 5.x or 6.x is pulled in by `terraform init`).
- `uv` (the repo's Python toolchain).
- A Hugging Face account that has accepted the Llama 3.3 license on
  <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>.

## 1. Request GPU quotas (do this first: approval can take days)

EC2 GPU quotas are counted in vCPUs and start at 0. Request them in `us-east-1`:

| Quota | Code | Ask for | Why |
|---|---|---|---|
| All G and VT Spot Instance Requests | `L-3819A6DF` | 48 | g6e.12xlarge (4x L40S, 48 vCPU) for Llama 3.3 70B; g6e.xlarge (4 vCPU) for Qwen3-8B |
| Running On-Demand G and VT instances | `L-DB2E81BA` | 48 | Fallback when spot capacity is missing |
| All P Spot Instance Requests | `L-7212CCBC` | optional | Only for A100/H100 runs |
| Running On-Demand P instances | `L-417A185B` | optional | Same |

```sh
aws service-quotas request-service-quota-increase --region us-east-1 \
  --service-code ec2 --quota-code L-3819A6DF --desired-value 48
aws service-quotas request-service-quota-increase --region us-east-1 \
  --service-code ec2 --quota-code L-DB2E81BA --desired-value 48

# Check progress / current values
aws service-quotas get-service-quota --region us-east-1 --service-code ec2 --quota-code L-3819A6DF
```

## 2. Make sure the default VPC exists

The hosts run in the default VPC's subnets. They get a public IP for outbound
traffic only (Docker Hub, Hugging Face, PyPI), and their security group has no
inbound rules.

```sh
aws ec2 describe-vpcs --region us-east-1 --filters Name=is-default,Values=true
# If that prints no VPCs:
aws ec2 create-default-vpc --region us-east-1
```

## 3. Store the Hugging Face token in Secrets Manager

Create a **read-only** token at <https://huggingface.co/settings/tokens>. Store it
as a plain string (not JSON) so it never lands in your shell history or in a file:

```sh
read -rs HF_TOKEN   # paste the token, press Enter
printf %s "$HF_TOKEN" | aws secretsmanager create-secret --region us-east-1 \
  --name loom/hf-token --description "Loom bench: Hugging Face read token" \
  --secret-string file:///dev/stdin
unset HF_TOKEN
```

Only the GPU host's role can read this secret. The host reads it inside the
weight-download step and nowhere else. It is never put in SSM parameters,
user-data, tags or logs, and the engine container never gets it: weights are
downloaded first, and the engine then starts with `HF_HUB_OFFLINE=1`.

## 4. Apply the Terraform

```sh
cd infra/aws/bench
terraform init
terraform apply -var owner=<your-name>      # optional: -var hf_token_secret_name=...
```

This creates:

- an S3 bucket (`loom-bench-<random>`). It is private, encrypted (SSE-S3), refuses
  plain HTTP, and expires `runs/` and `ssm/` objects after 30 days;
- the GPU host role and instance profile: SSM core, read/write on that bucket, and
  `GetSecretValue` on the HF token secret only;
- a security group with no inbound rules, in the default VPC, and one in the default
  VPC of each extra host region (`extra_host_regions`, default `us-east-2` and
  `us-west-2`; see [Hosts in more than one region](#hosts-in-more-than-one-region));
- the TTL reaper Lambda on a 15-minute EventBridge schedule. It can only terminate
  instances and delete volumes tagged `loom:managed=true`, in the primary region and
  every extra host region;
- the RunPod reaper Lambda, also every 15 minutes, its role, and the empty Secrets
  Manager secret `loom/runpod-reaper-api-key` it reads its RunPod key from. It starts in
  dry run and does nothing useful until step 8;
- the managed policy `loom-bench-runner` for the identity that runs `bench`.

The state is local (`terraform.tfstate`, gitignored). Keep it somewhere safe.
`terraform` is not installed in CI, so `terraform validate` runs here, on your
first `init`.

Attach the runner policy to the user or role you run `bench` as:

```sh
aws iam attach-user-policy --user-name <you> \
  --policy-arn "$(terraform output -raw runner_policy_arn)"
# or: aws iam attach-role-policy --role-name <role> --policy-arn ...
```

The runner policy only allows the following:

- launching instances of the allowed types (`allowed_instance_types`) from Amazon
  AMIs, with IMDSv2 required, a metadata hop limit of 1, the bench instance profile
  and all of the `loom:managed`, `loom:ttl`, `loom:experiment` and `loom:owner` tags;
- terminating, and sending SSM commands to, instances tagged `loom:managed=true`;
- reading and writing the bench bucket;
- reading spot prices and the DLAMI SSM parameter.

## 5. Configure `AwsSettings`

```sh
mkdir -p ~/.config/loom
terraform output -raw aws_settings_yaml > ~/.config/loom/aws.yaml
export LOOM_AWS_CONFIG=~/.config/loom/aws.yaml
```

Fields (any field can be overridden with `LOOM_AWS_<FIELD>`, e.g.
`LOOM_AWS_SUBNET_IDS=subnet-a,subnet-b`):

| Field | From Terraform | Default | Meaning |
|---|---|---|---|
| `region` | yes | `us-east-1` | Primary region: the bucket, the HF token secret, and the first region a capacity sweep tries |
| `bucket` | yes | | Bench bucket (job inputs/outputs, SSM output) |
| `instance_profile_name` | yes | | GPU host instance profile (IAM, so it works in every region) |
| `security_group_id` | yes | | No-ingress security group in the primary region |
| `subnet_ids` | yes | | Primary-region default-VPC subnets to try (cheapest spot AZ first) |
| `extra_regions` | yes | `[]` | More regions to launch hosts in, swept after `region` in order: each `{region, security_group_id, subnet_ids, ami_id}` ([below](#hosts-in-more-than-one-region)). `LOOM_AWS_EXTRA_REGIONS=us-west-2` keeps only the named ones; an empty value keeps none |
| `hf_token_secret_name` | yes | | Secret read on the host |
| `owner` | yes | | `loom:owner` tag |
| `name_prefix` | yes | `loom-bench` | `Name` tag prefix |
| `dlami_ssm_parameter` | | DLAMI Ubuntu 24.04 base OSS driver | AMI lookup |
| `ami_id` | | unset | Pin one AMI in the primary region instead of the parameter's latest (it moves about weekly), e.g. the one a smoke ran on. Extra regions run the same release |
| `data_dir` | | `/opt/dlami/nvme/loom-data` | Pinned workload datasets on the host (`<sha256>/<file>`), mounted read-only at `/data` in the client container |
| `root_volume_gb` | | 200 | Root gp3 volume (Docker images) |
| `weights_dir` | | `/opt/dlami/nvme/loom-hf` | HF cache on the instance-store NVMe |
| `spot_price_multiplier` | | 1.25 | Budget accrual = spot price x this, rounded up |
| `max_ttl_s` | | 28800 | Longest host lifetime the provider accepts |
| `wheel_path` | | | Bench wheel installed in the on-host client container; build it from the checkout you run `bench` in (its dependencies come from that checkout's `uv.lock`) |
| `client_image` | | `python:3.12.15-slim-trixie` by digest | Client container image (load and eval jobs); must be `repo@sha256:<digest>` |
| `poll_interval_s` | | 5 | SSM / instance-state polling |
| `job_timeout_s` | | 7200 | Run-time cap for request-count (no `duration_s`) load jobs and, plus 900 s, for eval jobs |
| `presign_expiry_s` | | 21600 | Lifetime of job input/output presigned URLs |
| `capacity_wait_s` | | 1800 | How long to keep sweeping the AZs while every one refuses the instance for capacity; 0 fails on the first sweep ([runbook](runbook.md#no-capacity-insufficientinstancecapacity)) |

Presigned URLs stop working when the credentials that signed them expire. With
short-lived SSO credentials, keep `presign_expiry_s` within the session length.

## 6. Dry run

```sh
uv run bench run <experiment.yaml> --dry-run
```

The dry run expands the matrix and prints the plan and cost estimate. It makes
no AWS calls that create anything. Check that the estimate fits the experiment's
budget (at most $50).

## 7. First real run

Start small: Qwen3-8B on one `g6e.xlarge` spot host with a short TTL.

```sh
uv run bench run <experiment.yaml>
```

While it runs, and always afterwards, check that nothing managed is still alive:

```sh
aws ec2 describe-instances --region us-east-1 \
  --filters Name=tag:loom:managed,Values=true \
            Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`loom:ttl`]|[0].Value]'
```

To see what a host did, open the SSM command output under `s3://<bucket>/ssm/`, or
look at `/var/log/loom/` on the host through Session Manager.

## 8. Turn on the RunPod reaper

The RunPod reaper Lambda terminates Loom's RunPod pods whose TTL has passed, so a pod
stuck before its container starts (no in-pod watchdog yet) or left by a dead runner is
removed without your laptop. What it will and will not touch is in
[security.md](security.md#runpod-pods-runpod); in short, only pods in the runner's exact
`loom-bench-<experiment>-<ttl epoch>-<nonce>` name format with `LOOM_MANAGED=true`, once
that TTL has passed, and it deletes nothing when anything in RunPod's answer looks wrong.

1. **Apply** (from `infra/aws/bench`, as in step 4). On a stack applied before the RunPod
   reaper existed, the plan should read `10 to add, 0 to change, 0 to destroy`:

   ```sh
   terraform plan  -var owner=<your-name>
   terraform apply -var owner=<your-name>
   ```

   Until step 2 the secret is empty, so each scheduled run fails with
   `ResourceNotFoundException` and touches nothing.

2. **Put the RunPod key.** Create a key used only by the reaper (RunPod console →
   Settings → API Keys, read/write, named e.g. `loom-reaper`), so you can revoke it on its
   own. Store it as a plain string, without echoing it:

   ```sh
   read -rs RUNPOD_REAPER_KEY   # paste the key, press Enter
   printf %s "$RUNPOD_REAPER_KEY" | aws secretsmanager put-secret-value --region us-east-1 \
     --secret-id loom/runpod-reaper-api-key --secret-string file:///dev/stdin
   unset RUNPOD_REAPER_KEY
   ```

   Afterwards you cannot read it back (`get-secret-value` is denied to everyone but the
   reaper's role); to change it, put a new value.

3. **Verify with a dry-run invoke:**

   ```sh
   aws lambda invoke --region us-east-1 --function-name loom-bench-runpod-reaper \
     --cli-binary-format raw-in-base64-out --payload '{"dry_run": true}' /tmp/rp-reaper.json
   cat /tmp/rp-reaper.json
   aws logs tail /aws/lambda/loom-bench-runpod-reaper --region us-east-1 --since 15m
   ```

   With no Loom pods running, expect `{"dry_run": true, "reaped": [], "failed": [],
   "kept": 0}` and the log line `N pods and M volumes listed; 0 Loom-named resources
   kept`, where N and M count the whole account, your other project included.
   `"FunctionError"` in the invoke output means the run failed, and a dry run deletes
   nothing either way; the log says why (a wrong key gives `GET /pods failed (401)`).

4. **Go live.** Keep the setting in `terraform.tfvars` (gitignored) so a later apply does
   not put the reaper back into dry run:

   ```sh
   echo 'runpod_reaper_dry_run = false' >> terraform.tfvars
   terraform apply -var owner=<your-name>
   aws lambda get-function-configuration --region us-east-1 \
     --function-name loom-bench-runpod-reaper \
     --query 'Environment.Variables.LOOM_RUNPOD_REAPER_DRY_RUN'   # "false"
   ```

   A scheduled run logs `dry_run=False` next to anything it reaps. To pause it, set
   `runpod_reaper_dry_run = true` again and apply, or disable the EventBridge rule
   `loom-bench-runpod-reaper`.

## How the four safety layers work

Each layer covers the failure of the one before it.

1. **Budget guard (runner).** Each host reports an accrual rate, `hourly_micros`. For
   spot, that is the current spot price in its AZ x 1.25, rounded up. On-demand, and
   spot AZs with no price, use the on-demand price from `prices.yaml`. The root EBS
   volume is added in both cases. The runner records spend every
   `budget.accrual_interval_s` while the host lives. Before each step it stops
   gracefully if spend so far plus the step's estimate would pass the cap (exit 4).
   When recorded spend reaches the cap it trips: the in-flight call is cancelled, every
   host is torn down and the experiment is aborted (exit 5). The cap can therefore be
   overshot by up to one accrual interval of every live host's rate, plus the teardown
   time, which is still recorded ([benchmark-lab.md](benchmark-lab.md)). A host accrues
   at its own region's prices; the plan's estimate and cap check use the dearest region
   it may launch in.
2. **Instance self-shutdown at TTL.** User-data runs `shutdown -h` at the host's
   TTL. Instances launch with shutdown behaviour `terminate`, so this ends the
   instance and deletes its root volume even if the laptop running `bench` sleeps,
   crashes or loses network. The engine start script refuses to continue unless
   this step finished.
3. **Reaper Lambda (every 15 minutes).** It terminates any `loom:managed=true`
   instance whose `loom:ttl` has passed, and deletes unattached managed volumes.
   Managed resources with a missing or garbled TTL are reaped once they are older
   than 24 h. It covers a host whose OS hung before the shutdown fired. EC2 is
   regional, so the Lambda (in the primary region) reaps each region in its
   `LOOM_REAPER_REGIONS` environment variable, which Terraform sets to the primary
   region plus `extra_host_regions`. A deployed Lambda older than that variable reaps
   only its own region. A second Lambda, the RunPod reaper (step 8), does the same for
   RunPod pods.
4. **`bench reap`.** It runs the same `aws_reaper.reap` function from your machine
   with your runner credentials, in every region of your `AwsSettings`. Use it after an
   aborted run, or if the Lambda is disabled.

Layers 1, 2 and 4 cover every region a host can launch in without any setup; layer 3
covers a region once Terraform lists it in `extra_host_regions` and has been applied.

Other rails:

- Hosts have no inbound ports, and IMDSv2 is required with hop limit 1, so the
  engine container cannot reach the instance role.
- The client container uses host networking, so the hop limit does not apply
  to it. It runs as uid 10001, which user-data blocks from the metadata service.
  Its inputs and outputs move through presigned URLs.
- Spot interruptions surface as `SpotInterrupted`, with the time since launch, so
  the runner can record the interruption rate.

## Hosts in more than one region

On 2026-11-03 us-east-1 refused g6e.xlarge (`InsufficientInstanceCapacity`) for hours,
while the G/VT on-demand quota was 4 vCPU, one g6e.xlarge, in each of us-east-1,
us-east-2 and us-west-2. With `extra_regions` set, the provider launches the host
wherever one of those has capacity.

**What happens.**

- **One sweep tries every AZ of every region.** It starts with the experiment's
  `provider.region` (normally the primary, us-east-1), then goes through the others in
  configured order. An extra region is tried only if `bench/prices.yaml` prices the
  instance type there; today that is g6e.xlarge only.
- **The vCPU quota is per region.** A `VcpuLimitExceeded` skips the rest of that region
  for the sweep, and the sweep goes on to the next region. The 600 s quota wait
  (`QUOTA_WAIT_S`) applies only when every region was refused for its quota. If one
  region is quota-blocked and the others are out of capacity, the provider waits for
  capacity up to `capacity_wait_s`.
- **Everything after the launch uses the host's own region** (`Host.info["region"]`):
  SSM commands, describe and `check_host`, teardown and its wait for `terminated`, and
  the ClientToken lookup that adopts or terminates an unanswered RunInstances. The
  ClientToken is resent to the region it was first sent to.
- **What stays in the primary region:** the bucket (job inputs and outputs, SSM output,
  traces) and the HF token secret, which the host reads with `--region <primary>`.
  The instance profile is IAM, which is global, so the same profile and role work in
  every region without changes.
- **The same image everywhere.** AMI ids are regional. Every region runs the primary
  region's release, which is `ami_id` (or the DLAMI parameter's latest):
  - an extra region's own `ami_id` must carry the same image name;
  - with no `ami_id`, the provider looks up the one image of that name and owner there;
  - a region that has neither fails the sweep before anything launches.
- **Comparability.** The cell's config, and so its `config_hash`, keeps the
  experiment's `provider.region`. A g6e.xlarge cell that ran in us-west-2 hashes the
  same as one that ran in us-east-1, so the two compare and reproduce against each
  other. Provenance `region` and the events' `system.region`/`az` record where it
  actually ran. Its as-run price and report prices come from that region's
  `prices.yaml` entry.
- **Prices.** The host accrues at its own region's price. The plan's estimate and cap
  check use the dearest region it may launch in.

DLAMI release `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20261006`
(owner 898082745236, root `/dev/sda1`, x86_64), from `aws ec2 describe-images`
on 2026-11-03:

| Region | AMI id | Default VPC | g6e.xlarge AZs |
|---|---|---|---|
| us-east-1 | `ami-028357be7d5b15c53` | | 1a, 1b, 1c, 1d |
| us-east-2 | `ami-089d4ee74147726da` | `vpc-…` | 2a, 2b, 2c |
| us-west-2 | `ami-0fec8acfc603e6e16` | `vpc-…` | 2a, 2b, 2c, 2d |

Prices from the AWS Price List API (publication 2026-11-03, effective 2026-11-01).
g6e.xlarge is $1.861/h on demand and gp3 is $0.08/GB-month in all three regions, so a
200 GB host accrues $1.8829/h in each.

**Cross-region traffic.** A host outside us-east-1 still uses the us-east-1 bucket. AWS
bills that traffic as inter-region transfer: $0.01/GB between us-east-2 and us-east-1,
and $0.02/GB between us-west-2 and us-east-1. The sending region bills it, so downloads
from the bucket and uploads to it are each billed once.

- Each job's inputs are the job JSON, the bench wheel and `requirements.txt`, a few MB.
- Outputs are `result.json`, `gpu.csv` and the SSM logs: KB to a few MB.
- A profile trace is at most `max_trace_mb` (256 MB).
- So a run with several profile windows moves well under 2 GB, under $0.04. That is
  less than two minutes of the host's price, so the plan does not add it.
- Docker images and weights come from the internet (Docker Hub, Hugging Face), which is
  free inbound in every region.

**Setting it up.**

1. Apply the Terraform (step 4) with `extra_host_regions` listing the regions (the
   default is `["us-east-2", "us-west-2"]`). For an existing stack this adds only the
   following:
   - a no-ingress security group in each extra region's default VPC;
   - `LOOM_REAPER_REGIONS` and the extra regions' instance and volume ARNs for the
     reaper Lambda;
   - the extra regions' ARNs in the runner policy.
2. Add the regions to `AwsSettings` from `terraform output extra_host_networks` (or
   `aws_settings_yaml`), pinning each region's copy of the primary's AMI:

   ```yaml
   extra_regions:
     - region: us-east-2
       security_group_id: sg-…        # terraform output extra_host_networks
       subnet_ids: [subnet-…, subnet-…, subnet-…]
       ami_id: ami-089d4ee74147726da
     - region: us-west-2
       security_group_id: sg-…
       subnet_ids: [subnet-…, subnet-…, subnet-…, subnet-…]
       ami_id: ami-0fec8acfc603e6e16
   ```

   A checkout older than this change refuses an `AwsSettings` file with
   `extra_regions` (unknown field). Use a separate file (`LOOM_AWS_CONFIG`) while an
   older checkout still runs from the shared one.
3. Check: `uv run bench run <experiment.yaml> --dry-run` notes the regions it priced
   (`worst of the regions it may launch in (…)`).

## Load and eval jobs on the host

The engine listens on the host's loopback only (no inbound ports), so everything that
talks to it runs on the host, in the client container:

- **Load jobs** (`bench job run`): the runner uploads `job.json`, the bench wheel and
  `requirements.txt` (its dependencies exported from `uv.lock`, hash-pinned) to
  `s3://<bucket>/runs/<experiment>/<run id>/`, sends `run_job.sh` over SSM with presigned
  GET/PUT URLs, and reads `result.json` (and `gpu.csv`) back. The host installs the
  requirements with `pip --require-hashes --no-deps`, then the wheel, into a virtualenv
  cached per wheel and requirements file.
- **Eval jobs** (`bench quality job`): the same path and script with the eval
  subcommand. The job carries the resolved suite, its task subset, the divergence mode
  and, for candidates, the baseline's captured reference. Jobs with lm-eval tasks install
  the `lmeval` extra's locked requirements (lm-eval, torch, transformers) into a separate cached virtualenv on
  first use, about 5 minutes once per host; datasets come from the Hugging Face Hub inside
  the container.
- **Tokenizers.** The client container has no Hugging Face token. Instead, the hub cache
  the engine start downloaded into (`<weights_dir>/hub`) is mounted read-only at
  `/models`, and jobs load the tokenizer from
  `/models/models--<org>--<name>/snapshots/<revision>`: load generation, external tools
  and lm-eval tasks that name the model repo (RULER). This works for gated models (Llama)
  and pins the RULER tokenizer to the model's revision. The whole hub cache, not the
  model's folder: the vLLM image's huggingface_hub keeps large files (tokenizer.json,
  safetensors) in a hub-level store (`snapshots/<rev>/tokenizer.json` →
  `../../blobs/<etag>` → `hub/blobs/<xx>/<sha256>`), so a folder-only mount left them
  dangling and every load job failed with `No such file or directory (os error 2)` on the
  first AWS smoke (2026-10-10). The job fails before it starts if the host has no snapshot
  at that revision or a snapshot file resolves outside the mount, and the client checks
  that the path is that repo and revision.
- **Isolation.** The container runs as uid 10001, which user-data denies the instance
  metadata service, so it has no instance-role credentials; it gets no secrets or AWS
  settings in its environment; its virtualenv and the model cache are mounted read-only and
  only its job directory is writable; it is removed when the job ends. It does share the
  host network,
  which it needs to reach the engine, so it can reach the internet. Code-executing eval
  tasks run model-written programs in it (inside the sandbox's rlimits) and stay off
  unless the experiment sets `quality.allow_code_exec: true`.

An eval job's SSM command is bounded by `job_timeout_s` + 900 s; raise `job_timeout_s` for
a full-suite run on a large model.

## What it costs besides GPU time

The S3 bucket and the two Lambdas cost cents; reaping three regions instead of one
keeps the EC2 reaper inside the Lambda free tier. Security groups are free. A host
outside the bucket's region pays inter-region transfer on its bucket traffic, under
$0.04 a run ([above](#hosts-in-more-than-one-region)). Secrets Manager is $0.40 per secret per
month (two secrets: the HF token and the RunPod reaper key). CloudWatch Logs for the
reapers are kept 30 days. The root gp3 volume
(200 GB, about $0.022/h) is part of each host's accrued `hourly_micros` and of every
reported cost ([cost-model.md](cost-model.md#4-the-hourly-price-h)).
