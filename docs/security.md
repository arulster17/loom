# Security notes (Phase 0)

What the Benchmark Lab does today to protect credentials, the AWS account and the
integrity of published numbers, with the known limits of each measure. Phase 0 handles
no customer data: prompts are synthetic or come from public datasets.

## Secrets

- **None in the repository.** `.gitignore` excludes `.env*`, `*.tfstate*` and
  `*.tfvars`; `bench/tests/aws/test_terraform.py` fails on account ids or secrets in the
  Terraform. Terraform state is local to whoever applied it.
- **Hugging Face token** (`aws_ec2`) lives only in AWS Secrets Manager
  (`loom/hf-token`, plain string). Only the GPU host's instance role may read it
  (`GetSecretValue` on that one secret ARN); the runner policy cannot.
  `providers/aws_scripts/start_engine.sh` reads it on the host during a cold start,
  exports it only in that shell, passes it by name to the short-lived download container,
  and unsets it. It is never in user-data, SSM command
  text (the script carries the secret's name, not its value), tags, logs or the engine
  container: weights are downloaded first and the engine runs with `HF_HUB_OFFLINE=1`.
  The host scripts must never use `set -x` (`aws_scripts/common.sh`).
  RunPod differs; see [RunPod pods (`runpod`)](#runpod-pods-runpod).
- **Engine launches carry no secrets**: `engines.docker_run_argv` refuses environment
  variable names containing `TOKEN`, `SECRET`, `PASSWORD`, `CREDENTIAL` or `KEY`.
- **User-data** is readable by anyone who can describe the instance and contains only the
  TTL, the stage file path and the load-generator uid.

## GPU host isolation (`aws_ec2`)

- **No inbound access.** The security group has no ingress rules (tested in
  `test_terraform.py`); there are no SSH keys. The runner drives hosts over SSM. The host
  has a public IP for outbound traffic only (Docker Hub, Hugging Face, PyPI, AWS APIs).
  The engine port is published on `127.0.0.1` only; load is generated on the host.
- **Instance metadata.** Instances launch with IMDSv2 required, a PUT hop limit of 1 and
  instance-metadata tags disabled; the runner policy refuses any other launch. The engine
  container (bridge network) is one hop too far to reach the metadata service, so it
  cannot obtain the instance role. The load-generator container uses host networking, so
  the hop limit does not apply to it; it runs as uid 10001, and user-data adds
  `iptables -I OUTPUT -d 169.254.169.254 -m owner --uid-owner 10001 -j REJECT`. The
  engine start script refuses to run until user-data has finished.
- **Load-generator container**: `python:3.12.15-slim-trixie` pinned by digest
  (`AwsSettings.client_image`, which rejects a tag), non-root (uid 10001), the bench wheel
  and its requirements verified by sha256 before install, the virtualenv mounted
  read-only, inputs and outputs only through presigned URLs. A pinned workload dataset
  (chat-sharegpt) is fetched by the host as root over HTTPS, checked against its sha256,
  and mounted read-only at `/data`.
- **Lifetime**: every instance schedules its own shutdown at its TTL with shutdown
  behaviour `terminate`; the root volume is encrypted and deleted on termination.
- **Shell rendering**: `providers/aws_ssm.render_script` passes every value as a
  `shlex.quote`d assignment; tag values and run ids must match a safe-id pattern.

## IAM (least privilege, tag-conditioned)

Defined in `infra/aws/bench/`:

| Principal | Can |
|---|---|
| GPU host role | SSM agent (`AmazonSSMManagedInstanceCore`); `s3:GetObject`/`PutObject` on the bench bucket's objects; `GetSecretValue` on the HF token secret |
| Runner policy (`loom-bench-runner`) | `RunInstances` only with tags `loom:managed=true` plus `loom:ttl`, `loom:experiment`, `loom:owner`, an allowed instance type, IMDSv2 required, hop limit ≤ 1, the bench instance profile, Amazon-owned AMIs and the bench security group; `CreateTags` only at launch; `TerminateInstances`, `DeleteVolume` and `ssm:SendCommand` only on `loom:managed=true` resources; `PassRole` only for the bench role to EC2; the bench bucket; read-only describes |
| Reaper Lambda | `TerminateInstances` and `DeleteVolume` only on `loom:managed=true`; describes; its own log group |

Limits worth knowing: `ssm:SendCommand` with `AWS-RunShellScript` is root on bench hosts,
so the runner identity can act as the host role (including reading the HF token) on a
managed instance. The launch statement allows any subnet in the region and the describe
and command-tracking actions are not resource-scoped.

## RunPod pods (`runpod`)

Since the AWS GPU spot quota is 0, GPU benchmarks run on RunPod Secure Cloud on-demand
pods ([runbook-runpod.md](runbook-runpod.md)). The pod is the engine image's own
container: there is no Docker inside it, and the engine, the bench client and the
secrets RunPod injects share one container. The decisions below date from 2026-10-05/06
so they are not reopened.

- **No AWS credentials in pods.** As on AWS hosts, the runner presigns per-job S3 URLs
  ([below](#s3-and-presigned-urls)) and pods only GET inputs and PUT results through them.
- **Accepted change: the HF token is in the pod environment.** The pod env sets
  `HF_TOKEN` to the RunPod secret reference `{{ RUNPOD_SECRET_hf_token }}`
  (`RunpodSettings.hf_secret_name`), which RunPod substitutes into PID 1's environment.
  On AWS the token is fetched from Secrets Manager and never reaches the engine
  container. Accepted because fetching it from Secrets Manager would need AWS credentials
  in the pod, a larger exposure than a read token. Mitigations: the token is read-only
  and used only for model downloads; pods hold no AWS credentials; S3 access is through
  presigned URLs only. `providers/runpod_scripts/start_engine.sh` reads it from
  `/proc/1/environ` as root and passes it only to the weight-download process; it is
  never exported, written to disk, logged or given to the engine process.
- **Accepted for now: the engine runs as root.** Its environment is PID 1's minus
  `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `PUBLIC_KEY`, `RUNPOD_*`, `LOOM_*`, `AWS_*` and
  any name matching `TOKEN|SECRET|PASSWORD|CREDENTIAL|KEY`, and it is bound to
  `127.0.0.1` with `HF_HUB_OFFLINE=1`. Being root, it can still read `/proc/1/environ`,
  so the HF token and the pod-scoped `RUNPOD_API_KEY` are within its reach. This stays
  inside the accepted HF-token change; a non-root engine is later hardening
  ([PLAN.md](PLAN.md)).
- **Job isolation.** Load jobs and evals run as uid 10001 (`loom`, no login shell)
  through `setpriv --reuid=10001 --regid=10001 --clear-groups --no-new-privs env -i`
  with an allowlisted environment (`HOME`, `PATH`, `LANG`): no `HF_TOKEN`, `RUNPOD_*`,
  `PUBLIC_KEY` or `AWS_*`. Only PID 1's owner (root) may read `/proc/1/environ`. Every
  engine start checks that:
  1. the job user cannot read `/proc/1/environ`;
  2. the job user's environment carries no secret names (names only are printed);
  3. no secret value (the HF token or the pod key) is stored under `/etc`, `/opt/loom`,
     `/var/lib/loom`, `/tmp`, `/root` or `/var/log/loom` (the weights directory,
     `/opt/loom/hf`, is excluded).

  Any failure fails the engine start. On success it emits `loom-sys job_isolation ok`,
  which lands in `engine_started.system.job_isolation` in the run's `events.jsonl`;
  every job repeats check 1 before `bench` runs. The client runs on a pinned
  python-build-standalone CPython (sha256-checked, `runpod_layout.CLIENT_PYTHON_*`) in
  a virtualenv the job user cannot write; after each job every uid-10001 process is
  killed.
- **Limits of the isolation.** The offline tests (`bench/tests/runpod/`) run the pod
  scripts against stub `setpriv`, `curl` and `nvidia-smi`; only a real pod proves the
  kernel denies uid 10001 `/proc/1/environ`, which is why the smoke checklist requires
  `job_isolation ok` ([runbook-runpod.md](runbook-runpod.md#smoke-test)). uid 10001 has
  unfiltered network egress: the container has no `iptables`, unlike the AWS
  load-generator rule. Model-written code from code evals still needs the sandbox caveats
  [below](#code-execution-sandbox).
- **SSH.** RunPod has no exec API, so the pod start command installs `openssh-server`
  and the runner connects to the pod's mapped TCP 22 (`providers/runpod_ssh.py`). Each
  provider instance (one `bench run`) generates its own ed25519 key in a 0700 temporary
  directory, removed at exit and never written to results; its public key reaches the
  pod as env `LOOM_SSH_PUBKEY`. With `authorize_account_key` (default true) the account's
  registered key (`PUBLIC_KEY`) is also authorized, for debugging. sshd is key-only,
  `AllowUsers root` (the job user cannot log in), no forwarding, and starts with an empty
  environment. Host keys are trusted on first use (`StrictHostKeyChecking=accept-new`)
  and pinned in a per-pod `known_hosts` file: a man-in-the-middle on the very first
  connection would go unnoticed.
- **Nothing secret in the pod spec.** Anyone with the account key can read a pod's env
  and start command through the API, so the env holds only the HF secret reference,
  `LOOM_*` values (`LOOM_MANAGED`, `LOOM_TTL`, experiment, owner) and the SSH public key,
  and the start command no secrets and no URLs besides RunPod's GraphQL endpoint. A
  validator rejects an `hf_secret_name` starting with `aws`. Presigned URLs reach the pod
  only inside scripts sent on SSH stdin (`bash -s`), are written to a root-only control
  directory removed once the output is read, and go to `curl` as a config on stdin, never
  in argv (`/proc/<pid>/cmdline` is world-readable); URLs containing `"`, `\` or a
  newline are refused before rendering.
- **Fallback keys, not for pods.** A scoped IAM user, `loom-runpod-bench` (inline policy
  `loom-bench-bucket-only`: `s3:PutObject`/`GetObject` on the bench bucket's objects and
  `s3:ListBucket` on the bucket), has its keys stored as the RunPod secrets
  `aws_access_key_id` and `aws_secret_access_key`. They are an unused fallback: pod specs
  must not reference them. The user was created with the AWS CLI and is not yet in
  Terraform (to do).
- **Lifetime: terminate, never stop.** A stopped pod keeps billing for its disk, so the
  provider, the in-pod watchdog and the reaper only terminate. The pod start command arms
  a TTL watchdog before anything else, at an absolute epoch (a container restart cannot
  extend it); at the TTL it calls GraphQL `podTerminate` with the pod-scoped
  `RUNPOD_API_KEY` and `RUNPOD_POD_ID` RunPod injects, read from its own environment,
  retrying every 30 s. A pod whose setup fails terminates itself the same way.
  `bench reap` terminates pods both named `{name_prefix}-…` and carrying
  `LOOM_MANAGED=true` once past their TTL, or `EXITED`. There is no scheduled RunPod
  reaper yet: if the runner dies, the backstops are the watchdog and a manual
  `bench reap`, and a pod stuck before its container starts has no watchdog at all.
- **API key.** The account key comes from `RUNPOD_API_KEY` or the macOS Keychain
  (service `RUNPOD_API_KEY`), is sent only in the `Authorization` header and is redacted
  from error text. `bench/tests/conftest.py` clears it (and `LOOM_RUNPOD_*`) for every
  test, so no test reaches the real API.

## S3 and presigned URLs

- The bench bucket is private (all public access blocked, bucket-owner-enforced
  ownership), encrypted (SSE-S3), refuses non-TLS requests, and expires `runs/` and
  `ssm/` objects after 30 days.
- Each load job gets presigned URLs scoped to single objects under
  `runs/<experiment id>/<run id>/`: GET for `job.json` and the wheel, PUT for
  `result.json` and `gpu.csv`. They expire after `presign_expiry_s` (default 6 h, at most
  7 days) or when the signing credentials expire, whichever is first.
- On `aws_ec2` the URLs are part of the script sent with `ssm:SendCommand`, so they are
  visible in SSM Run Command history to anyone who can list commands until they expire.
  On `runpod` they travel only over SSH (see above). Until a URL expires, a holder could
  read the job and wheel or overwrite a result. The runner validates every result against
  the `LoadJobResult` schema.

## Code-execution sandbox

`quality/sandbox.py` runs model-written programs (HumanEval, MBPP) only when
`allow_code_exec` is set (`--allow-code-exec`, or `quality.allow_code_exec: true` in an
experiment). Each program gets a fresh `python -I -S` in its own process group, an empty
environment, a temporary working directory, and rlimits: CPU 10 s, file size 16 MB,
64 open files, no core dumps, no new processes, address space 1024 MB; a 10 s wall-clock
timeout kills the group (defaults in `SandboxLimits`, overridable per task).

Platform caveats: macOS does not enforce `RLIMIT_AS` (memory is unbounded there) and the
process limit is only required on Linux. It is a resource sandbox, not a security
boundary: programs run as the current user, can read that user's files and can open
network connections. Run code evals in a disposable container or VM with no network
egress and no credentials.

## Prompts and outputs

- Loom never logs prompts or completions. `RequestRecord.output_text` is filled only when
  a `LoadJob` sets `keep_output`, which the runner never does, so the Parquet column
  stays empty.
- Eval outputs are kept in memory for the sanity checks only. `samples.json` and
  `bench_eval_runs` store per-item scores, content hashes and short metadata (status,
  error type), not outputs.
- Exception: lm-evaluation-harness tasks keep lm-eval's raw samples and log in the
  suite's working directory (for `bench quality run --out x/samples.json`, that is
  `x/samples-work/`).
- Engine container logs stay on the GPU host and are deleted with it.

## Supply chain and reproducibility

- **Model revisions**: `hf.revision` must be a 40-character commit sha; weights and the
  tokenizer are loaded at that revision.
- **Engine images**: `engine.image` must be `repo@sha256:<64 hex>`; tags are rejected.
  The digest the host actually pulled is recorded in provenance.
- **`trust_remote_code`** is off for every model. Turning it on requires
  `hf.trust_remote_code_review: {reviewer, date, notes}` in `config/models.yaml`;
  `--trust-remote-code` is rendered only from that field and cannot be passed through
  `engine.args`.
- **Load-generator image**: `client_image` must be `repo@sha256:<64 hex>`. The default is
  `python@sha256:02108f5d…155d`, the multi-arch index of `python:3.12.15-slim-trixie`
  (also tagged `3.12-slim`), read from the Docker Hub registry API
  (`registry-1.docker.io/v2/library/python/manifests/3.12.15-slim-trixie`,
  `Docker-Content-Digest`) on 2026-10-05.
- **Client dependencies**: the runner exports `uv.lock` for loom-bench (plus the `lmeval`
  extra for harness evals) with `uv export --frozen --no-dev --no-emit-workspace
  --package loom-bench --format requirements-txt` (`providers.export_requirements`) and
  uploads it with the wheel. The host installs it with `pip install --require-hashes
  --no-deps`, so every package is the locked version with a locked hash and nothing is
  resolved from PyPI; the wheel follows with `--no-deps`, and `pip check` fails the job if
  the wheel's requirements and the lock disagree (a `wheel_path` built from another
  commit). Three lm-eval dependencies (`rouge-score`, `sqlitedict`, `word2number`) ship
  only as source; their sdists are hash-checked, but pip builds them with an isolated,
  unpinned setuptools.
- Not pinned by default: the DLAMI (latest via its SSM parameter; `AwsSettings.ami_id`
  pins one), lm-eval datasets (guarded
  by per-item content hashes instead) and the tokenizer of the wrapped `vllm bench` /
  `sglang` tools.

## Competitor APIs

No Loom code calls, benchmarks or ships credentials for third-party inference APIs.
Competitor data is public list prices entered by hand in `bench/competitors.yaml`.
Benchmarking a named third-party endpoint needs that provider's written permission and a
legal review: [legal/competitor-benchmarking.md](legal/competitor-benchmarking.md).

## Published data

The results site loads no external scripts, fonts or images, and publishes experiment
specs and provenance in full: keep credentials out of experiment files. The waitlist form
has a honeypot field and stays disabled until a backend is configured
([site.md](site.md#waitlist-endpoint)).

## Phase 1 requirements (future, not built)

From the build spec, to apply when the platform is built:

- API keys stored as SHA-256 hash + prefix only, shown once; rotation, per-key limits and
  scopes.
- GPU pods on private networks reachable only from the gateway (network policies, mTLS);
  no public inference endpoints.
- Rate limits on requests and tokens; caps on `max_tokens`, body size and concurrent
  streams.
- Atomic credit reservation with a documented worst-case overshoot; idempotent,
  signature-verified Stripe webhooks.
- No prompt/completion logging by default; per-org retention; secrets redacted in logs.
- Image scanning in addition to pinned revisions and digests; `trust_remote_code` only
  after a recorded review (already enforced by the registry).
- Org-scoped queries everywhere, automated cross-tenant access tests, Postgres row-level
  security where practical.
- Secrets only in AWS Secrets Manager / GCP Secret Manager; least-privilege IAM per
  workload; TLS everywhere; CORS locked to the web app.
- Abuse controls: email verification, free-credit limits per org / IP / card fingerprint,
  spend-spike alerts, instant org suspension.
- Append-only audit log of admin actions.
