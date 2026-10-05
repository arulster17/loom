# AWS setup for the Benchmark Lab

Phase 0 runs each benchmark host as one EC2 GPU VM (spot, or on-demand as a
fallback) from the AWS Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 24.04).
The engine runs in Docker, and the runner drives the host over SSM, so the host
has no inbound ports and no SSH keys. Everything below is done once per account.
The whole Phase 0 budget is $150, at most $50 per experiment, so the steps put
the safety rails up before the first GPU starts.

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
- a security group with no inbound rules, in the default VPC;
- the TTL reaper Lambda on a 15-minute EventBridge schedule. It can only terminate
  instances and delete volumes tagged `loom:managed=true`;
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
| `region` | yes | `us-east-1` | Region for every API call |
| `bucket` | yes | | Bench bucket (job inputs/outputs, SSM output) |
| `instance_profile_name` | yes | | GPU host instance profile |
| `security_group_id` | yes | | No-ingress security group |
| `subnet_ids` | yes | | Default-VPC subnets to try (cheapest spot AZ first) |
| `hf_token_secret_name` | yes | | Secret read on the host |
| `owner` | yes | | `loom:owner` tag |
| `name_prefix` | yes | `loom-bench` | `Name` tag prefix |
| `dlami_ssm_parameter` | | DLAMI Ubuntu 24.04 base OSS driver | AMI lookup |
| `root_volume_gb` | | 200 | Root gp3 volume (Docker images) |
| `weights_dir` | | `/opt/dlami/nvme/loom-hf` | HF cache on the instance-store NVMe |
| `spot_price_multiplier` | | 1.25 | Budget accrual = spot price x this, rounded up |
| `max_ttl_s` | | 28800 | Longest host lifetime the provider accepts |
| `wheel_path` | | | Bench wheel installed in the on-host client container; build it from the checkout you run `bench` in (its dependencies come from that checkout's `uv.lock`) |
| `client_image` | | `python:3.12.15-slim-trixie` by digest | Client container image (load and eval jobs); must be `repo@sha256:<digest>` |
| `poll_interval_s` | | 5 | SSM / instance-state polling |
| `job_timeout_s` | | 7200 | Run-time cap for request-count (no `duration_s`) load jobs and, plus 900 s, for eval jobs |
| `presign_expiry_s` | | 21600 | Lifetime of job input/output presigned URLs |

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
   time, which is still recorded ([benchmark-lab.md](benchmark-lab.md)).
2. **Instance self-shutdown at TTL.** User-data runs `shutdown -h` at the host's
   TTL. Instances launch with shutdown behaviour `terminate`, so this ends the
   instance and deletes its root volume even if the laptop running `bench` sleeps,
   crashes or loses network. The engine start script refuses to continue unless
   this step finished.
3. **Reaper Lambda (every 15 minutes).** It terminates any `loom:managed=true`
   instance whose `loom:ttl` has passed, and deletes unattached managed volumes.
   Managed resources with a missing or garbled TTL are reaped once they are older
   than 24 h. It covers a host whose OS hung before the shutdown fired. It only
   watches the Terraform region.
4. **`bench reap`.** It runs the same `aws_reaper.reap` function from your machine
   with your runner credentials. Use it after an aborted run, or if the Lambda is
   disabled.

Other rails:

- Hosts have no inbound ports, and IMDSv2 is required with hop limit 1, so the
  engine container cannot reach the instance role.
- The client container uses host networking, so the hop limit does not apply
  to it. It runs as uid 10001, which user-data blocks from the metadata service.
  Its inputs and outputs move through presigned URLs.
- Spot interruptions surface as `SpotInterrupted`, with the time since launch, so
  the runner can record the interruption rate.

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
- **Tokenizers.** The client container has no Hugging Face token. Instead, the model's
  cache folder that the engine start downloaded (`<weights_dir>/hub/models--<org>--<name>`)
  is mounted read-only at `/models/models--<org>--<name>`. Jobs load the tokenizer from its
  `snapshots/<revision>` directory: load generation, external tools and lm-eval tasks that
  name the model repo (RULER). This works for gated models (Llama) and pins the RULER
  tokenizer to the model's revision. The job fails before it starts if the host has no
  snapshot at that revision, and the client checks that the path is that repo and revision.
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

The S3 bucket and the Lambda cost cents. Secrets Manager is $0.40 per secret per
month. CloudWatch Logs for the reaper are kept 30 days. The root gp3 volume
(200 GB, about $0.022/h) is part of each host's accrued `hourly_micros` and of every
reported cost ([cost-model.md](cost-model.md#4-the-hourly-price-h)).
