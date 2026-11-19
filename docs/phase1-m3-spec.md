# Phase 1 M3: the AWS control plane (spec)

Status: **approved by the user 2026-11-18, with the changes of §15** (a Loom-only Stripe
sandbox, no automated sync in Phase 1, the product name and domain as settings). Nothing
here is built, applied or paid for. The Stripe design (§8) went through 16 fresh
design-review rounds on Fable until one found no blocker or major; the user's decisions then
cut foreign events, the Events-API sync and `park`, and one more round (17) reviewed the
reduced design; the user then took round 17's list of removable mechanisms, and one last
round (18) reviewed the result (Appendix A). M3 has two stages: **Stage A** (free: code, tests, runbooks, Terraform up to
`terraform plan`) and **Stage B** (money: `terraform apply`, DNS, deploy, live acceptance,
teardown), which needs the user's separate yes with a cap (§14). Every work item below is
marked **[A]** or **[B]**.

Background: [phase1-runpod-plan.md](phase1-runpod-plan.md) (§3 where things run, §9
observability, §10 security, §11 costs, §13 the M3 row and "M3 prerequisites from the M2
review" P1-P4, §14 risks, §15 open questions 4-6), the user decisions in
[PLAN.md](PLAN.md#decisions-so-far), the fixed interfaces in
[phase1-contracts.md](phase1-contracts.md), and the M2 spec
([phase1-m2-spec.md](phase1-m2-spec.md)) whose "M3" rows this spec picks up (Secrets
Manager, Terraform, the serving reaper Lambda, the gateway's client-certificate rotation).

## 1. Goal

Run the platform M1 and M2 built (gateway, billing, web, pool manager, Redis, Postgres)
on AWS, on a public URL with TLS, with real Stripe in **test mode**, and the backend
**still the mock** (the pool manager's `local` provider). The M3 row of plan §13:

- M1's flow works on the public URL (sign up, approval, $5, key, SDK, playground, usage
  and balance);
- a Stripe test checkout posts one ledger credit, idempotently;
- a revoked key is refused within 5 s, and the gateway's p50 overhead is under 10 ms,
  both re-measured on the deployed box.

Plus the M2 review's prerequisites (plan §13 P1-P4): the reaper alarms, the spend-cap
alarm, the pod CA expiry gauge with its alarm and runbook, and the deploy-order rule.

## 2. Scope

In scope:

1. `infra/aws/platform`: Terraform for the VPC, the EC2 instance, RDS, Secrets Manager
   containers, IAM, ECR, S3 (lease bucket for M4), DNS and SES identity, the site
   parameter (domain and product name, §6.11), the serving
   reaper Lambda (disabled), CloudWatch alarms with an SNS topic, an AWS budget.
   `infra/aws/state`: the remote-state bucket. [A to write and test, B to apply]
2. `deploy/`: a production compose file, Caddy (TLS, routing, per-IP limit), Redis with
   TLS and a password, Prometheus and Grafana (not public), the CloudWatch agent config,
   and `loom-deploy` (secrets to files, migrations, ordered rollout per P4). [A, B]
3. The pool manager outside dev on its mock backend (an explicit flag), its new gauges
   (CA expiry, pool reason, mock backend), the gateway client-certificate rotation and
   reload, and `docs/runbook-pod-ca.md` (P3). [A]
4. Stripe test mode: the webhook's response contract under failures, alarms, a health
   check on the endpoint and a runbook resend for deliveries Stripe gave up on, a database
   constraint for one purchase per Checkout Session, an alarm per stuck event and an admin
   `resolve`, the runbook to register the endpoint, and the list of live-mode
   prerequisites (§8.13). [A, B]
5. `loom-m3-acceptance` and the runbook for the manual steps. [A to write, B to run]
6. Contract and doc changes (§11, §12) in the same change as the code. [A]
7. The product name and the domain as settings, and today's hard-coded name removed from
   user-facing text (§6.11). [A]

Out of scope:

| Not in M3 | Where |
|---|---|
| Any RunPod call, the real `runpod` provider on the box, turning on the serving reaper | M4 (cap $20) |
| Live Stripe keys, auto-recharge charging a saved card | Later (Phase 1 is test mode only; prerequisites §8.13) |
| An automatic sync of missed Stripe deliveries (Events API) | Before live mode (§8.12, §8.13) |
| Handling Stripe events of other systems on a shared account | Not needed: the sandbox is Loom's alone (§15 D4) |
| R7, R10 (plan §13 follow-ups) | M5 |
| High availability (a second instance, Multi-AZ RDS) | Later; plan §3 accepts one instance |
| The four follow-ups in the handoff (comment over-claims, `free_port()`, signin-send minors) | Separate small change, not M3 |

## 3. What the user must provide

No accounts are created for the user. Each item says which step it blocks.

| Item | What exactly | Blocks | Lead time |
|---|---|---|---|
| **Domain** (+ registrar or DNS access) | A registered domain; Route 53 as registrar or DNS host is simplest (Terraform then writes the records). Otherwise the user creates the records `terraform output` lists at their DNS host | Nothing in Stage A (the domain is a Terraform variable, §6.11). Stage B: DNS, TLS, the Stripe endpoint URL, the Google redirect URI, the SES identity. The user registers it before Stage B | Registration minutes to a day (TLD dependent); a transfer days; DNS propagation minutes to 48 h |
| **AWS access** | The existing account, region `us-east-1`. (1) A **read-only** role or profile for `terraform plan` (AWS `ReadOnlyAccess`, minus `secretsmanager:GetSecretValue`). (2) For apply: the user runs `terraform apply` themselves, or grants a deployer role for that run. (3) An ops email for alarms (SNS subscription to confirm). (4) The monthly budget: $75 (§15 D12) | (1) the Stage A `terraform plan` item (A28). (2)-(4) Stage B | Minutes |
| **SES production access** and domain verification | Verify the domain identity (Easy DKIM: 3 CNAMEs Terraform writes or lists), then request production access in the SES console (use case: transactional sign-in mail) | Sign-in mail to **unverified** addresses, i.e. real users. **Not** the M3 acceptance: in the SES sandbox the acceptance's own addresses are verified identities (§10) | DKIM: minutes to 72 h. Production access: AWS reviews it, typically about a day, can be longer; ask as soon as the domain exists |
| **Google OAuth client** | A Google Cloud project, OAuth consent screen (External, "Testing", the acceptance tester listed as test user), a Web client with redirect URI `https://<domain>/api/auth/callback/google`; client id and secret go into Secrets Manager | Only the Google sign-in step of the acceptance (manual, B24). The web app runs without it (Google sign-in is off when unset) | Minutes |
| **Product name** | The name users see (web app, sign-in mail sender, Checkout product, the Stripe sandbox's public business name); not decided yet | Nothing in Stage A (a Terraform variable, §6.11). Stage B: the first deploy | None |
| **Stripe test keys + webhook secret** | A Stripe **sandbox** (test mode) used only by Loom's deployed box: no laptop, script or other stack on it (§15 D4). A **restricted** test key `rk_test_...` with Checkout Sessions write only. A webhook endpoint `https://hooks.<domain>/stripe/webhook` subscribed to the five event types of §8.2 at the sandbox's default API version; its signing secret `whsec_...`. Both go into Secrets Manager | Stage B deploy: billing refuses to start in `stripe` mode without both (contracts §2). The endpoint can be registered before the box exists (Stripe just fails deliveries until then) | Minutes |
| **Stage B yes and cap** | §14 | All of Stage B | |

Not needed for M3: the RunPod serving account and key (M4), live Stripe keys.

## 4. Where it runs

```
browser / SDK / Stripe
   | HTTPS :443 (Caddy, Let's Encrypt; :80 only redirects and answers ACME)
   v
EC2 t4g.medium (arm64), public subnet, Elastic IP, IMDSv2 hop limit 1, no SSH (SSM only)
   Caddy -> <domain>        -> web:3000
            api.<domain>    -> gateway:8080      (/v1/* only)
            hooks.<domain>  -> billing:8090      (POST /stripe/webhook, GET /stripe/health only)
   gateway, billing (all), web, poolmgr (provider local = mock pods in its container),
   redis (TLS + password, AOF), prometheus, grafana (127.0.0.1 only),
   CloudWatch agent (host; scrapes 127.0.0.1:9100-9102), SSM agent (host)
   data EBS volume /var/lib/loom: redis AOF, settle spool, caddy certs, poolmgr state,
   prometheus TSDB
   |  TLS verify-full (rds.force_ssl=1)
   v
RDS PostgreSQL 16, db.t4g.micro, private subnets (2 AZs), 7-day backups + PITR
```

Only 80 and 443 are open to the internet. `/internal/*`, `/health`, `/metrics` (9100,
9101, 9102), Grafana, Prometheus, Redis and Postgres are not reachable from outside;
Grafana is reached with SSM port forwarding (§15 D13). The box's egress IP (the Elastic
IP) becomes `gateway_egress_cidrs` in M4.

**Containers get no AWS credentials.** IMDSv2 with a hop limit of 1 means only host
processes (`loom-deploy`, the Docker daemon's `awslogs` driver, the CloudWatch and SSM
agents) can use the instance role. Containers get their secrets as files `loom-deploy`
renders (§6.3).

## 5. Terraform

### 5.1 Layout

```
infra/aws/state/          bootstrap: the remote-state bucket (local state, kept by the user)
infra/aws/platform/
  versions.tf             terraform >= 1.10 (S3 native locking), aws provider as bench pins (6.x),
                          plus the alias aws.us_east_1 (Route 53 health-check metrics, §8.5)
  backend.tf              backend "s3" {use_lockfile = true}; bucket/key from -backend-config
  variables.tf            domain, brand_name (both without defaults, §6.11),
                          route53_zone_id (optional), ops_email, budget_usd,
                          serving_reaper_enabled (false), instance_type, db_deletion_protection
  network.tf ec2.tf rds.tf secrets.tf iam.tf ecr.tf s3.tf dns.tf ses.tf
  reaper_serving.tf alarms.tf healthcheck.tf budget.tf site.tf outputs.tf
  tests/*.tftest.hcl      terraform test with mock_provider "aws" (no credentials)
  README.md               how to init, test, plan; what each variable is
```

IAM and bucket policies are written as `jsonencode()` locals, not
`aws_iam_policy_document` data sources, so `terraform test` under `mock_provider` can
assert their exact content (a mocked data source returns random strings).

### 5.2 Resources

| File | What | Notes |
|---|---|---|
| network | VPC `10.42.0.0/16`; one public subnet (AZ a) for the instance; two private subnets (AZ a, b) for the RDS subnet group; internet gateway; route tables | No NAT gateway ($32/month): RDS needs no egress |
| ec2 | `t4g.medium` (§15 D3), Amazon Linux 2023 arm64 AMI from the SSM public parameter, 30 GB gp3 root (encrypted), a separate 20 GB gp3 **data volume** (encrypted, `prevent_destroy`) mounted at `/var/lib/loom`, Elastic IP, `metadata_options {http_tokens = "required", http_put_response_hop_limit = 1}`, `disable_api_termination`, user data installing Docker + compose plugin, the CloudWatch agent, Python 3.12, and `chrony` on Amazon Time Sync | Security group: 80 and 443 from anywhere, nothing else in. No key pair, no port 22 |
| rds | PostgreSQL 16 (compose runs 16), `db.t4g.micro`, 20 GB gp3, `storage_encrypted`, `publicly_accessible = false`, single AZ, backups 7 days, `deletion_protection = var.db_deletion_protection` (true), final snapshot on destroy, parameter group with `rds.force_ssl = 1` and server TCP keepalives (`tcp_keepalives_idle` 60, `tcp_keepalives_interval` 10, `tcp_keepalives_count` 3, §8.3), master user `loom_owner` with `manage_master_user_password = true` (the password lives in an RDS-managed secret, never in Terraform state), auto minor upgrades | Security group: 5432 from the instance's security group only |
| secrets | Secret **containers** only, no values (values never enter Terraform state): `loom/platform/stripe` (`secret_key`, `webhook_secret`), `auth` (`auth_secret`, `google_id`, `google_secret`), `internal-token`, `app-db-password`, `redis` (`password`, `ca_crt`, `server_crt`, `server_key`), `pod-ca` (`crt`, `key`), `poolmgr-client` (`crt`, `key`), `ses-smtp` (`username`, `password`), `serving-runpod-key` (empty until M4). Each has a resource policy allowing `GetSecretValue` only to the instance role (the serving key: only the serving reaper's role), as `infra/aws/bench` does for the reaper key | Values come from `loom-deploy secrets init` (generated ones) and the user (Stripe, Google), §6.3 |
| iam | Instance role: `AmazonSSMManagedInstanceCore`; `cloudwatch:PutMetricData` only with `cloudwatch:namespace = Loom/Platform`; logs to `/loom/platform*`; `secretsmanager:GetSecretValue` on `loom/platform/*` and the RDS master secret; ECR pull on the Loom repositories; S3 read/write on the lease bucket's `serving/leases/*` (M4); `ssm:GetParameter` on `/loom/platform/site` only (§6.11). SES SMTP user `loom-ses-smtp` with `ses:SendRawEmail` on the domain identity only (its access key is created by the runbook, not Terraform, so the secret stays out of state). GitHub OIDC role for the image workflow: ECR push only, only from this repository's `main` branch and `workflow_dispatch` (§15 D1) | |
| ecr | Repositories `loom/gateway`, `billing`, `web`, `poolmgr`, `devstack`, `caddy`: immutable tags, basic scan on push, lifecycle keeps the last 20 images | |
| s3 | Lease bucket (`serving/leases/<pod id>`, M4): private, SSE, TLS-only policy, versioning off | Unused in M3 |
| dns | If `route53_zone_id` is set: A records for the apex, `api` and `hooks` to the Elastic IP; CAA `0 issue "letsencrypt.org"`; SES DKIM CNAMEs, MAIL FROM MX and SPF TXT, DMARC TXT (`p=quarantine`). Otherwise the same records as an output for the user's DNS host | |
| ses | `aws_sesv2_email_identity` for the domain with Easy DKIM, a custom MAIL FROM subdomain, a configuration set | Production access is a console request (§3) |
| reaper_serving | A second instance of `runpod_reaper_lambda.py` (`loom-serving-runpod-reaper`) for the **serving** RunPod account: `LOOM_RUNPOD_REAPER_SERVING=true`, `LOOM_RUNPOD_REAPER_LEASE_BUCKET`, dry run on, key from `serving-runpod-key`; role with `s3:ListBucket` on the bucket and `s3:GetObject` on `serving/leases/*` (P1); EventBridge rule every 15 min, **disabled** while `serving_reaper_enabled = false` | The bench reaper holds the bench account's key and cannot see serving pods (decision 2026-10-31: separate account), so the serving rule needs its own instance. Enabled in M4 |
| site | SSM parameter `/loom/platform/site` (String): `{"domain", "brand_name"}` from the two variables (§6.11) | Standard tier, free |
| alarms | §5.3 | |
| healthcheck | Route 53 health check (`HTTPS_STR_MATCH`) on `https://hooks.<domain>/stripe/health` and alarm H3 on it, created through the provider alias `aws.us_east_1`; a precondition that the stack's region is `us-east-1` (§8.5) | Created whether or not `route53_zone_id` is set (it resolves the public name). Cost §14 |
| budget | AWS Budgets monthly cost budget `budget_usd` (§15 D12), email at 50%, 80%, 100% actual and 100% forecast | First two budgets are free |

### 5.3 Alarms (P1-P3 and the platform's own)

All alarms notify the SNS topic `loom-platform-alerts` (email subscription to `ops_email`)
on ALARM and OK. Metrics from the services reach CloudWatch through the CloudWatch agent's
Prometheus scraping (§6.9), namespace `Loom/Platform`, only the series listed here.

| # | Alarm | Metric and condition | Missing data |
|---|---|---|---|
| P1a | Serving reaper errors | `AWS/Lambda Errors` (FunctionName = serving reaper), Sum >= 1 over 1 h | not breaching |
| P1b | Serving reaper not running | `AWS/Lambda Invocations`, Sum < 1 over 2 consecutive 15-min periods | **breaching** |
| P2a | Spend-cap refusal | `loom_poolmgr_creates_total{result="spend_cap"}` increased (> 0 over 5 min) | not breaching |
| P2b | Model held at the cap | new gauge `loom_poolmgr_pool_reason{model,reason="spend_cap"}` = 1 for 5 consecutive minutes | not breaching |
| P3 | Pod CA near expiry | new gauge `loom_poolmgr_ca_expiry_seconds` < 2,592,000 (30 days), 1-h period | **breaching** (a manager that refuses to start publishes nothing) |
| H1 | Platform heartbeat | `up{job="gateway"}`, `up{job="billing"}`, `up{job="poolmgr"}` (the agent's scrape result) < 1 for 5 min | **breaching** (box or agent down) |
| H2 | Instance health | `AWS/EC2 StatusCheckFailed_System` -> EC2 **recover** action; `StatusCheckFailed_Instance` -> notify | breaching |
| H3 | Webhook endpoint unreachable from the internet | `AWS/Route53 HealthCheckStatus` (dimension `HealthCheckId` = the `hooks.` health check, §8.5) Minimum < 1 for 5 consecutive 1-min periods; the alarm is in us-east-1, created through `aws.us_east_1` (Route 53 publishes health-check metrics only there) | **breaching** |
| G1 | Settle spool | `loom_gateway_settle_spool_depth` > 0 for 5 min (contracts §6) | not breaching |
| G2 | Lost usage | `loom_gateway_settle_failures_total` increased (contracts §6: alert at once) | not breaching |
| G3 | Gateway client certificate | new gauge `loom_gateway_client_cert_expiry_seconds` < 172,800 (rotation broken, §7) | breaching |
| S1a | A Stripe event became stuck | new counter `loom_billing_stripe_unapplied_stale_total{type}` increased (> 0 over 5 min): an event has stayed unapplied for 15 min (§8.6). Each new stuck event alarms again (ALARM, then OK when the window passes), so one old stuck event never hides the next | not breaching |
| S1b | Stuck events outstanding | new gauge `loom_billing_stripe_unapplied_stale_events` (as defined in §8.6) > 0 for 24 h: a daily-scale reminder that an admin must resolve them or a release must fix them (§8.6 "Resolve") | not breaching |
| S2a | Webhook secret mismatch | `loom_billing_stripe_webhooks_total{result="signature_mismatch"}` (a well-formed header with a current `t`, no `v1` matching: the box's secret is not the endpoint's, §8.3) >= 1 in 15 min | not breaching |
| S2b | Clock skew | `loom_billing_stripe_webhooks_total{result="stale_timestamp"}` (HMAC valid, `t` outside the tolerance) >= 1 in 15 min (test-mode retries are few, so any genuine one matters). Junk POSTs count as `bad_signature`, which no alarm reads; a wrong secret shows up as S2a | not breaching |
| S2c | Webhook 5xx | new counter `loom_billing_stripe_webhook_errors_total{status}` (503, 500) >= 1 in 1 h (one event gets only a few attempts over hours in test mode, and a 503 stores nothing, so one is enough; a Postgres outage also shows as S5, a stopped billing as H1) | not breaching |
| S4 | Handler failures | new counter `loom_billing_stripe_handler_failures_total{path}` (`webhook`, `retry`, `pending_clawback`, each initialised at 0: an attempt on one event raised; counted on the event's first failure since its last clean attempt, i.e. the write that makes `error_count` 1, not on each minute's repeat, §8.3) increased | not breaching |
| S5 | Retry job stopped | `loom_billing_stripe_retry_last_success_seconds` older than 5 min (§8.3) | **breaching** |
| D1 | Disk | CloudWatch agent `disk_used_percent` > 80% on `/` or `/var/lib/loom` | not breaching |
| R1 | RDS | `FreeStorageSpace` < 2 GB; `CPUUtilization` > 80% for 15 min | not breaching |

Every Prometheus counter an alarm reads is initialised to 0 for each label value at
process start, so the series exists before its first increment. Whether the agent sends
counters as cumulative values or as deltas decides the alarm's math (cumulative: metric
math `RATE()`/`DIFF()`; delta: `Sum`); the implementer checks the agent's documented
behaviour and a test pins the choice (A9). The P1 alarms keep `actions_enabled =
var.serving_reaper_enabled`, so a disabled reaper in M3 does not page.

Runbook (`docs/runbook-platform.md`, at B2; round 18 n1): from `terraform apply` until the
first deploy is healthy (B6), H1 and H3 are in ALARM (nothing is scraped yet and nothing
answers on `hooks.`), and so are the other alarms whose missing data is breaching and whose
metric only the running stack publishes (P3, G3, S5): expected; the operator acknowledges
those emails and checks that each sends its OK email after B6 (one still in ALARM then is a
real fault).

### 5.4 Remote state

`infra/aws/state` creates one S3 bucket (versioning, SSE, public access block, TLS-only
policy, `prevent_destroy`) with its own local state, which the user keeps (it holds no
secrets). `infra/aws/platform` uses `backend "s3"` with `use_lockfile = true` (no DynamoDB
table) and a partial configuration (`-backend-config=backend.hcl`, gitignored, written
from `backend.hcl.example`). `infra/aws/bench` keeps its local state as today.

### 5.5 What can be checked without AWS credentials, and what cannot

| Check | Credentials | Where |
|---|---|---|
| `terraform fmt -check -recursive` | none | CI and agents |
| `terraform init -backend=false` then `terraform validate` | none (downloads the providers from the registry: network) | CI and agents |
| `terraform test` with `mock_provider "aws"` (and one for the `aws.us_east_1` alias) and `override_data` for the AMI parameter, caller identity, partition, region and AZs | none | CI and agents; this is where the A-items' Terraform assertions live |
| `tflint` with the AWS ruleset, shallow rules only | none (plugin download: network) | CI |
| `trivy config` (or checkov) on the directory | none | CI, findings triaged in the PR |
| `terraform plan` | **read-only AWS credentials**; local backend through a gitignored `backend_override.tf` (from `backend_override.tf.example`), so no state bucket is needed yet | Stage A, last item (A28), run by the user or with their yes |
| Remote state, `terraform apply`, everything live | write credentials, money | Stage B |

Terraform is not installed in the WSL environment today. Agents run a pinned Terraform:
the official `hashicorp/terraform` image pinned by digest (`docker run --rm -v ...`), the
same version as CI's `hashicorp/setup-terraform` step and the committed
`.terraform.lock.hcl`. No agent ever runs `terraform plan` or `apply` against AWS.

## 6. Deploy to the box

### 6.1 Images

All images arm64 (the instance is Graviton). Built by a new workflow
`.github/workflows/images.yml` (manual dispatch on a commit of `main`) on GitHub's arm64
runner, pushed to ECR through the OIDC role, tagged with the git sha, and recorded in a
**release manifest** `releases/<sha>.json` (sha, each image's digest, the migration count,
the compose file's sha256) uploaded to S3 with the images (§15 D1; the alternative is
building on the box). Compose references images only by digest. The Caddy image is built
in the same workflow (§6.7).

### 6.2 `deploy/compose.prod.yml`

The dev `docker-compose.yml` stays as is (CI acceptance, local dev). The production file:

- services `caddy`, `web`, `gateway`, `billing` (`all`), `poolmgr`, `redis`,
  `prometheus`, `grafana`, and `migrate` (profile `migrate`, run by `loom-deploy` only);
  no `mailpit`, `stripe-stub`, `devcerts`, poolsim or mock pods;
- `LOOM_ENV` unset (prod): every service runs its production checks;
- published ports: only Caddy's 80/443; the metrics ports 9100-9102, Prometheus and
  Grafana bound to `127.0.0.1`;
- `gateway` `stop_grace_period: 30s` (contracts §6 "Shutdown": `LOOM_SHUTDOWN_GRACE_S` +
  8 s, with margin); `restart: unless-stopped` everywhere;
- `poolmgr` `depends_on: gateway: condition: service_healthy` (P4; the dev file has the
  opposite edge, `gateway` after `poolmgr` started, and keeps it);
- volumes on the data EBS volume: `/var/lib/loom/{redis,gateway-spool,caddy,poolmgr,
  prometheus}` (the settle spool outlives a container, contracts §6);
- logs: the `awslogs` driver to `/loom/platform` (30-day retention), never prompts
  (M1 item i still holds);
- the RDS CA bundle (`global-bundle.pem`, sha256 pinned in the repo) mounted read-only.

### 6.3 Secrets on the box

`loom-deploy secrets init` (run once, on the box through SSM) generates and puts the values
Loom makes itself, without echoing them: `internal-token`, `app-db-password`, `auth_secret`,
the Redis password and its TLS CA and server certificate (SAN `redis`), the pod CA (2-year
validity) and the manager's client certificate. It refuses to overwrite a secret that has
a value. The user puts Stripe and Google values with the runbook's
`printf %s ... | aws secretsmanager put-secret-value --secret-string file:///dev/stdin`
pattern (as `aws-setup.md` step 8). SES SMTP: the runbook creates an access key for
`loom-ses-smtp` and `loom-deploy ses-smtp-password` derives the SMTP password (AWS's
documented HMAC derivation) and stores it; the access key secret is never printed.

At each deploy `loom-deploy` reads the secrets with the instance role and renders them to
`/run/loom/secrets/<service>/` (tmpfs, `0400`, owned by the container's uid), mounted
read-only into the one service that needs each. Env-only settings point at files
(`*_FILE` variables where a service supports them; otherwise an env file under
`/run/loom`, `0400`). Nothing secret is on a command line, in compose files, in the
release manifest or in logs.

### 6.4 Postgres roles and migrations (R8 on RDS)

- The **owner** is the RDS master user `loom_owner` (member of `rds_superuser`, not a
  superuser). Only `migrate` uses it: `loom-deploy` reads the RDS-managed master secret
  and passes the URL to the one-shot `migrate` container. `loom-migrate --create-database`
  creates `loom_platform` on first deploy; `LOOM_APP_DATABASE_PASSWORD` (from
  `app-db-password`) makes it create or update the runtime login `loom_app` (contracts §3
  "Roles").
- Every service connects as `loom_app` with
  `sslmode=verify-full&sslrootcert=/etc/loom/rds-ca.pem`. Their start-up role check
  (not a superuser, owns no table) runs as today. `loom-deploy` refuses to render a `LOOM_DATABASE_URL` or `DATABASE_URL`
  without `sslmode=verify-full` and the CA file (§15 D10). The services' own start-up
  checks (`production_problems`) are **not** changed: protected tests pin their exact
  output (§16 "Constraints from protected tests").
- Migrations run before any service is updated and must be backward compatible with the
  running release (add, never drop or rename in the same release). Migration 0012 (§8.4)
  is additive.

### 6.5 Redis

`redis:7.4-alpine` pinned by digest, TLS only (`--port 0 --tls-port 6379`,
`--tls-auth-clients no`), `requirepass` from the secret, `appendonly yes`, `appendfsync
everysec`, `maxmemory 1gb`, `maxmemory-policy noeviction` (contracts §2 checks it with
`CONFIG GET`). Python services use
`rediss://:<password>@redis:6379/0?ssl_ca_certs=/run/loom/secrets/<svc>/redis-ca.crt`; the
web app the same URL with `NODE_EXTRA_CA_CERTS` pointing at the CA file.

### 6.6 `loom-deploy` and the deploy order (P4)

`loom-deploy` is one Python module in `services/devstack` that uses **only the standard
library** (a test enforces it), so it runs on the box with the host's Python 3.12 and calls
the `aws` CLI and `docker compose`. Subcommands: `secrets init`, `ses-smtp-password`,
`apply --release <sha>`, `status`, `rollback`. The operator runs it through SSM Run
Command (`aws ssm send-command`, wrapped in the runbook).

`apply` steps, stopping at the first failure:

1. fetch and check the release manifest (sha256 of the compose file it names);
2. render secrets (§6.3); `docker compose pull`;
3. `migrate` (owner URL); wait for exit 0;
4. update services in this order, each `up -d --no-deps <svc>` then wait healthy: `redis`
   (only if its image changed), `billing`, **`gateway`**, **`poolmgr`**, `web`, `caddy`,
   `prometheus`, `grafana`;
5. record the release as current (`/var/lib/loom/release.json`).

**P4 rule.** The gateway is updated before the pool manager in every release, and
`loom-deploy` refuses any plan that would run a `poolmgr` image from a newer release than
the running `gateway` (`apply --only poolmgr` included). The step list is a pure function
(`plan_steps(current, target, only)`), unit-tested for every order. The rule is also in
the runbook (B12) and in the compose `depends_on` edge (§6.2).

Rollback: `loom-deploy rollback` applies the previous manifest with the same order;
migrations are forward-only, so a rollback across a migration relies on the
backward-compatible rule of §6.4. A rollback to a pre-M3 release is a last resort for
billing: M1's `retry_unapplied` catches only `BadEvent`, so a stored poison event fails its
job every minute and starves the rest, and M1 has no sweep that reopens a clawback an
admin resolved when its purchase arrives later (§8.6). Before such a rollback the runbook lists both
kinds: resolved clawbacks whose purchase may still come are noted for a hand check after
any late credit, and each poison event is either resolved or left for M1 only after
checking M1's handler applies it (otherwise stay on M3 and fix forward).

### 6.7 Caddy, domain, TLS, public routes

Caddy image built with the rate-limit module pinned by version (§15 D2; `xcaddy` in the
image workflow, `go.sum` committed). Caddyfile (`deploy/Caddyfile`):

- `<domain>` (`{$LOOM_DOMAIN}`, §6.11): everything to `web:3000`;
- `api.<domain>`: `/v1/*` to `gateway:8080`, `flush_interval -1` (streams), request body
  limit 1 MiB + 1 KiB, no read/write timeout shorter than the gateway's 600 s request
  timeout, `lb_try_duration 15s` (a request that cannot connect while the gateway
  restarts waits instead of failing); anything else 404; per-IP rate limit
  (§15 D2: e.g. 600 requests/min per IP) in front of the gateway (contracts §8 step 1);
- `hooks.<domain>`: only `POST /stripe/webhook` (body limit 256 KiB as billing's own, the
  request body passed through byte for byte: no request encoding or rewriting) and `GET
  /stripe/health` (the H3 health check, §8.5) to `billing:8090`; anything else 404.
  `billing`'s `/internal/*` and `/health` are never routed;
- HTTP redirects to HTTPS; HSTS (`max-age=31536000`) on all three; TLS 1.2+;
- ACME: Let's Encrypt, contact `ops_email`; first issuance against the **staging** CA,
  then production (Let's Encrypt's rate limits); certificates on the data volume;
- client IP: Caddy trusts no proxy, so `X-Forwarded-For` holds only the client's address
  and the right-most entry is the client (the web sign-in limit, contracts §2).

### 6.8 The mock backend outside dev

The manager refuses `provider: local` outside `LOOM_ENV=dev` (M2 spec §11), and running
the box with `LOOM_ENV=dev` would switch off every production check. New, explicit opt-in
(§15 D9): `LOOM_POOLMGR_MOCK_BACKEND=true` outside dev

- permits `provider: local` (refused with `runpod`: M4 removes the flag);
- permits `LOOM_POOLMGR_ALLOW_CIDRS`, only private (RFC 1918) networks;
- does not require the SSH key, lease bucket or proxy bundle (unused by `local`), and still
  requires the CA and the manager's client certificate;
- logs a warning at start and exports `loom_poolmgr_mock_backend 1`.

The box's serving file is `deploy/serving.m3.yaml` (`provider: local`, the committed
`serving.yaml`'s models and timings, `gateway_egress_cidrs` = the Elastic IP). The gateway
needs no change: it reaches the local pods' proxies over mTLS as in compose.

### 6.9 Metrics, dashboards, logs

- Prometheus scrapes gateway 9100, billing 9101, poolmgr 9102 (and itself); 15-day
  retention on the data volume. Grafana with provisioned dashboards (requests and errors
  by code, overhead p50/p99, settle spool, Stripe outcomes, pool states), admin password
  from a secret, bound to `127.0.0.1:3001`, reached by SSM port forwarding.
- The CloudWatch agent (`deploy/cloudwatch-agent.json`) scrapes the same three endpoints
  and sends only the series §5.3 names (an allowlist of metric names and label values,
  so custom-metric cost stays ~15 metrics), plus disk and memory.

### 6.10 Backups and teardown

RDS: automated backups (7 days) and point-in-time recovery; a restore drill is an
optional Stage B step (B21). Data volume: daily EBS snapshots by a Data Lifecycle Manager
policy, 7 kept. Teardown (B22): `loom-deploy` stops the stack; `terraform destroy` with
`db_deletion_protection = false` set first (RDS takes a final snapshot); the data volume
and state bucket have `prevent_destroy` and are removed only with the runbook's explicit
steps; secrets are deleted with the 7-day recovery window.

### 6.11 Product name and domain (settings)

Neither the product name nor the domain is decided (§15 D16). Each is **one setting**: the
Terraform variables `domain` and `brand_name` in `infra/aws/platform/variables.tf`, with no
defaults (`terraform plan` asks for them; the tftests use placeholders such as
`example.test`). Terraform writes both into the SSM parameter `/loom/platform/site`
(String, JSON `{"domain": ..., "brand_name": ...}`, free); the instance role may
`ssm:GetParameter` that one parameter. `loom-deploy apply` reads it at each deploy and
renders the services' environment from it:

| Service | Variable | Value |
|---|---|---|
| caddy | `LOOM_DOMAIN` (new) | the domain; the Caddyfile names `{$LOOM_DOMAIN}`, `api.{$LOOM_DOMAIN}`, `hooks.{$LOOM_DOMAIN}` |
| web | `LOOM_BRAND_NAME` (new) | the name, read on the server at request time (not a `NEXT_PUBLIC_` variable and not at build time, so one image serves any name) |
| web | `AUTH_URL`, `LOOM_PUBLIC_API_URL`, `EMAIL_FROM` (existing) | `https://<domain>`, `https://api.<domain>/v1`, `<name> <no-reply@<domain>>` |
| gateway | `LOOM_CORS_ORIGINS` (existing) | `https://<domain>` |
| billing | `LOOM_BRAND_NAME` (new) | the Checkout line item's product name, `<name> credits` |

- No domain or product name is written in code, Terraform, the Caddyfile,
  `deploy/compose.prod.yml`, an email or the sign-in page. Internal names stay `loom`:
  packages, services, `LOOM_*` variables, AWS resource names and tags, the Secrets Manager
  and SSM paths, the ECR repositories, the internal pod CA's organization.
- Unset `LOOM_BRAND_NAME` (dev, the existing tests): the services show no name (the web app
  uses each page's own title and a neutral header; billing's product is `API credits`).
  `loom-deploy` refuses an apply when the parameter lacks either value (as it refuses a
  database URL without `verify-full`, D10); the services' start-up checks are unchanged
  (protected tests pin them, §16). The dev `docker-compose.yml` is unchanged (its sender
  `login@loom.localhost` is on a reserved local name).
- The magic-link email is Auth.js's default template, which names the site by the host of
  `AUTH_URL`; its sender is `EMAIL_FROM`. No template in the repo holds a name or domain.
- Stripe Checkout also shows the sandbox's public business name, which the user sets in the
  Stripe dashboard (§3).
- Changing the name or the domain: edit `terraform.tfvars`, `terraform apply`, then
  `loom-deploy apply` with the same release. A new domain also needs DNS, SES verification,
  the Stripe endpoint URL and the Google redirect URI (runbook).
- Not changed in M3: wire names in the contracts that API clients see, the key prefix
  `loom_sk_` (contracts §5), the `X-Loom-*` response headers (contracts §12) and the
  `LOOM_API_KEY` variable in the playground's code snippets. Renaming them
  is a contracts change with a transition for issued keys, decided once the name is (§15
  D16).

Hard-coded today and removed in M3 (A36; owner: the deploy worker, except `stripe_api.py`,
the Stripe worker's, §16):

| File | Text |
|---|---|
| `apps/web/src/app/layout.tsx` | page title `Loom`, template `%s · Loom` |
| `apps/web/src/app/signin/page.tsx` | heading `Loom`; error text "Tell the Loom team." |
| `apps/web/src/app/dashboard/layout.tsx`, `apps/web/src/app/admin/layout.tsx` | header `Loom`, `Loom admin` |
| `apps/web/src/app/pending/page.tsx` | "the Loom team" (three messages) |
| `apps/web/src/app/dashboard/page.tsx`, `dashboard/billing/page.tsx`, `dashboard/keys/page.tsx`, `dashboard/playground/playground.tsx`, `app/api/playground/chat/route.ts` | sentences naming Loom ("Loom is prepaid", "Adjustment by Loom", "Loom never sees your card", "Loom stores only a hash", "use a Loom key", "The Loom API is not reachable") |
| `apps/web/src/auth.config.ts`, `apps/web/README.md` | default sender `Loom <no-reply@loom.local>` (no default name; the sender comes from `EMAIL_FROM`) |
| `services/billing/src/loom_billing/stripe_api.py` | Checkout product name `Loom credits` |
| `services/platform_common/src/loom_platform/errors.py` | API error "awaiting approval by a Loom admin." (reworded without a name) |

Each sentence either takes the name from `LOOM_BRAND_NAME` or is reworded without one. Not
user-facing, unchanged: docstrings and comments, the stub's checkout page (a test double),
the bench site, the internal certificates.

## 7. Pod PKI: CA expiry, gateway client-certificate rotation (P3)

- **CA expiry gauge.** The manager exports `loom_poolmgr_ca_expiry_seconds` (seconds until
  the pod CA's `notAfter`), set at start and each tick. Alarm P3 at 30 days; the existing
  refusal at `server_cert_renew_before_s` (2 days) stays.
- **Pool reason gauge** (P2b): `loom_poolmgr_pool_reason{model,reason}` is 1 for each
  model's current `reason` (contracts §10) and 0 for the others, rewritten each tick.
- **Gateway client certificate** (plan §4: 7-day validity, rotated daily; M2 spec
  deferred it to M3). The manager issues the gateway's client certificate (`clientAuth`,
  7 days) every 24 h, or at start when the current one has under 6 days left, into
  `LOOM_POOLMGR_GATEWAY_CERT_DIR` (a volume shared read-only with the gateway): one PEM
  file with key and certificate, written to a temporary name and renamed into place, so a
  reader never sees a key without its certificate. The gateway checks the file and
  `LOOM_POD_CA_CERT` every 60 s; on a change it builds a new TLS context and uses it for
  new upstream connections (open streams keep theirs). A file that fails to load keeps
  the old context and logs an error. It exports
  `loom_gateway_client_cert_expiry_seconds` (alarm G3).
- **CA trust bundle.** `LOOM_POD_CA_CERT` may hold several CA certificates (old and new
  during a rollover); the gateway trusts any of them.
- **Admin re-key.** `loom-poolmgr rekey [--model M]`: re-issue every ready pod's server
  certificate now (the SIGHUP path the manager already uses near expiry).
- **`loom-poolmgr new-ca --out DIR`**: a new CA outside dev (the dev-only `issue-ca`
  stays dev-only).
- **`docs/runbook-pod-ca.md`**: 1) `new-ca` on the operator's machine, put it as
  `pod-ca-next`; 2) deploy the gateway with the bundle old + new; 3) switch the manager to
  the new CA (secret swap, deploy: gateway first, P4) and `rekey`; 4) check every pod's
  certificate chains to the new CA (`loom-poolmgr status` shows the issuer); 5) issue the
  gateway's client certificate from the new CA (the manager does on start); 6) drop the
  old CA from the bundle and deploy. With the mock backend the whole runbook runs on the
  box (B20).

## 8. Stripe test mode: webhooks

### 8.1 What exists (M1) and what M3 adds

Built in M1 and kept: `POST /stripe/webhook` verifies `Stripe-Signature` on the raw body
(HMAC-SHA256 `v1` over `"<t>.<body>"`, any `v1` matches, `|now - t| <= 300 s`,
`stripe_sig.py`), then `webhooks.handle_event` runs in **one** Postgres transaction:
insert `stripe_events (id)` `ON CONFLICT DO NOTHING`, lock the row `FOR UPDATE`, skip it if
`processed_at` is set, otherwise apply it and set `processed_at`, or leave it unapplied
(stored, logged, retried every minute by `retry_unapplied`). Credits are `purchase`
transactions keyed `stripe:<event id>`, at most one per Checkout Session (an advisory lock
on the session id plus a look-up); clawbacks (`charge.refunded` cumulative-safe,
`charge.dispute.created` once per dispute) are `refund` transactions keyed
`stripe:<event id>`, capped at the purchase through `stripe_payments` (row-locked); a
clawback that arrives before its purchase is applied in the purchase's transaction
(contracts §3, §7). Test mode only: a live key or a `livemode: true` event is refused.

M3 adds:

1. a defined **response contract** under every failure: a failing event is stored with its
   error whenever Postgres can store it (§8.3);
2. **alarms for missed deliveries**, an internet **health check** on the endpoint, and a
   runbook resend of what Stripe gave up on (§8.5); no automatic sync in Phase 1 (decision
   record §8.12);
3. a **database constraint** for one purchase per Checkout Session (§8.4);
4. **stuck events**: an alarm for each event that stays unapplied, an admin `resolve`, and
   a sweep that reopens a resolved clawback once its payment is credited (§8.6);
5. the real endpoint, its secret and key permission, and the runbook (§8.2, §8.8).

What Stripe live mode needs on top of this is listed in §8.13.

The Stripe sandbox is used only by the deployed box (§15 D4): no laptop, script or other
stack takes payments through it, so every event the endpoint receives is this
deployment's. Anything else that reaches it (a payment made by hand in the dashboard, say)
stays unapplied, raises S1a and is resolved by an admin (§8.6).

### 8.2 Endpoint and event types

`https://hooks.<domain>/stripe/webhook` (the domain is a setting, §6.11), created by the
user in the Stripe dashboard (test mode) at the sandbox's **default API version**,
subscribed to exactly:

| Event | Handling |
|---|---|
| `checkout.session.completed` | credit if the session is Loom's (`metadata.purpose = loom_credits`) and `payment_status = paid`; a Loom session not yet paid -> `not_paid`; a session that is not Loom's -> `ignored` (as today) |
| `checkout.session.async_payment_succeeded` | credit (delayed payment methods) |
| `checkout.session.async_payment_failed` | stored, `ignored` (nothing was credited) |
| `charge.refunded` | claw back the new part of the cumulative `amount_refunded` |
| `charge.dispute.created` | claw back the disputed `amount`, once per dispute |

Any other type that arrives is stored and marked `ignored`. Snapshot events only (Stripe's
"thin" events are not supported).

### 8.3 Response contract (what Stripe sees)

Stripe retries any delivery that does not get a 2xx: in live mode for up to three days
with exponential backoff; **in test mode only a few times over a few hours** (Stripe's
documentation; checked again at Stage B, B7). Each retry is signed afresh with a new
timestamp. Stripe does not order deliveries. So:

| Situation | Answer | Why |
|---|---|---|
| Body over 256 KiB (`LOOM_BILLING_MAX_BODY_BYTES`, as today; Stripe events are a few KB) | 413 | before reading more; only junk is that large |
| Signature header missing or malformed, or no `v1` matches and `t` is outside 300 s | 400 `bad_signature` (metric result `bad_signature`) | forged or junk; junk POSTs land here and no alarm reads it |
| Header well-formed, `t` within 300 s, but no `v1` matches | 400 `bad_signature` (metric result `signature_mismatch`, alarm S2a) | the secret on the box is not the endpoint's (a wrong or half-rolled secret, T25); a forged request with a well-formed, current header lands here too. `stripe_sig.verify` raises the new subclass `SignatureMismatch` for this case (§8.11) |
| HMAC valid but `t` outside 300 s | 400 `bad_signature` (metric result `stale_timestamp`, alarm S2b) | a replay, or the box's clock is off; a genuine retry carries a fresh signature. `stripe_sig.verify` already checks the HMAC before the time; it raises the new subclass `StaleTimestamp` for this case (§8.11) |
| Signed, but not JSON, or no `id`/`type`, or `livemode: true` | 400 `bad_event` | cannot come from Loom's test endpoint; decided before any transaction |
| Stored and applied, or stored unapplied (§8.7 T11), or a duplicate | **200**, only **after COMMIT returned** | Stripe stops retrying; the row is the record |
| **Anything** raised by the Postgres part of a delivery (the pool wait, `handle_event`, COMMIT): a handler bug on a payload shape, a lost connection, a lock or statement timeout, a deadlock, `IdempotencyConflict` (which includes the L3 index's `unique_violation`, §8.4), a deferred check at COMMIT. No classification by exception class or SQLSTATE | the transaction rolls back; the **second transaction** (below) stores the event unapplied with `last_error`; then **500** `internal_error` (200 `duplicate` if its guard found the event processed meanwhile) (counted, S2c; S4 on the event's first failure) | Stripe retries (harmless: the row is unprocessed, the retry re-runs the handler); the stored row makes the event visible to `retry_unapplied`, the stale counter (S1a) and an admin, so it is never lost silently when Stripe stops retrying |
| The same, and the second transaction fails too (Postgres down or unreachable; a payload `jsonb` refuses, §8.13) | **503** `service_unavailable`, nothing stored (counted, S2c) | Stripe retries; the next delivery finds the row committed (duplicate, 200) or absent (processed then); if Stripe gives up, the alarms tell the operator, who resends it (§8.5) |

**Second transaction.** A new transaction from the pool, under the same timeouts:
`INSERT INTO stripe_events (id, type, payload, received_at, result, last_error,
last_error_at, error_count) VALUES (%(id)s, %(type)s, %(payload)s, %(now)s, 'unapplied',
%(error)s, %(now)s, 1) ON CONFLICT (id) DO UPDATE SET result = 'unapplied', last_error =
EXCLUDED.last_error, last_error_at = %(now)s, error_count = stripe_events.error_count + 1
WHERE stripe_events.processed_at IS NULL RETURNING error_count`. It never touches a row
another delivery has processed meanwhile: then it returns no row and the answer is 200
`duplicate` (no S2c, no S4). It needs the event's row lock, so when the first failure was a
lock timeout on that row itself (held by R, a `resolve` or another delivery, or an
uncommitted insert of the same id) it times out too and the answer is 503 while the holder
lives; a lock wait elsewhere (the session's advisory lock, a `stripe_payments` row, a
pending clawback's row) gives the 500. In R the same case ends the run. When it returns `error_count = 1` (the event's first failure
since its last clean attempt) S4 is counted. `%(error)s` is `<exception class>: <message>`
with NUL characters replaced (a `text` column refuses them) and cut to 1,000 characters.
`retry_unapplied` uses the same statement for its own failures (it updates the existing
row). Every attempt that returns an outcome other than `duplicate` or `skipped` writes
`result` and clears `last_error` and `error_count` in the attempt's transaction, so the
next failure counts S4 again. (Round-18 probe: the statement returns 1, then 2, then no row once
the event is processed; a payload with `\u0000` fails it with `22P05`.)

**Timeouts.** Inside each transaction `SET LOCAL lock_timeout` =
`LOOM_STRIPE_LOCK_TIMEOUT_MS` (3000) and `SET LOCAL statement_timeout` =
`LOOM_STRIPE_STATEMENT_TIMEOUT_MS` (5000); the pool wait is
`LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S` (5 s). Billing merges `keepalives=1 keepalives_idle=10
keepalives_interval=5 keepalives_count=2 tcp_user_timeout=10000` (milliseconds) into its
own conninfo with `psycopg.conninfo.make_conninfo` (URL and keyword forms alike;
`loom_platform.db` is unchanged), so a connection blackholed by an RDS failover fails in
seconds instead of the kernel's ~15 minutes; the RDS parameter group's server keepalives
(§5.2) end a backend whose client vanished and release its locks. There is **no overall
deadline**: with the defaults a delivery is answered within roughly 30 s at worst (pool
wait, lock waits, the second transaction). Stripe counts an answer it stopped waiting for
as a failed delivery and retries; a COMMIT that lands after that is safe by L1 (the retry
answers `duplicate`; the `loom_credit_changed` trigger and reconcile refresh Redis). Tests
scale the three settings down so the timing tests stay inside the flake-check budget.

**Credit refresh.** After COMMIT, the credit refresh in Redis (`_refresh_credit`) is
awaited for at most 2 s (a constant in `loom_billing.api`), then left to finish in the
background: the protected tests that read Redis right after a delivery stay stable under
load, and a frozen Redis delays the answer by at most 2 s (A22). The
`loom_credit_changed` notification and the minute's reconcile cover a refresh that fails
(contracts §6, §7). Outcome metrics (`stripe_webhooks_total{result}`, credited and
clawed-back micros) and S4 are counted **after** COMMIT, so a rolled-back attempt is not
counted twice.

**Pending clawbacks in savepoints.** When a purchase applies the clawbacks that arrived
before it (`_apply_pending_clawbacks`), M1's query is kept as it is (`processed_at IS NULL`,
clawback types of its PaymentIntent, `ORDER BY received_at, id FOR UPDATE`, blocking under
`lock_timeout`), so every pending clawback is applied in the purchase's own transaction,
as contracts §3 says. Each runs in its own **plain savepoint** (a nested
`conn.transaction()`), with no `SET CONSTRAINTS`: `loom_post_transaction` refuses an
unbalanced or one-entry post before inserting anything, so the two deferred ledger checks,
which still run at the purchase's COMMIT, cannot fail for a post made through it, and a
savepoint that is rolled back discards its rows and their queued checks (round-18 probe on
the real schema: a purchase with two pending clawbacks commits both, as the protected M1
test requires; with three, a middle savepoint that posts and then raises is rolled back
alone and the COMMIT succeeds, ledger balanced). A clawback that raises is rolled back to
its savepoint and, in the purchase's transaction, gets `result = 'unapplied'`,
`last_error`, `last_error_at` and `error_count + 1` (S4 `pending_clawback` after COMMIT
when that makes it 1); the purchase and the other clawbacks commit, so a failing clawback
never blocks the credit of its payment (T19), and `retry_unapplied` owns it. If that update
fails too (the transaction itself is gone, e.g. a lost connection), the delivery fails as a
whole and takes the 500/503 path; a wait on the pending rows past `lock_timeout` does the
same (stored, retried by Stripe and R).

**`retry_unapplied` (R).** Runs each minute at :50 and once at worker start (so S5 does not
fire on a deploy). One run, in order:

1. **reopen sweep** (§8.6): one set-based statement;
2. **re-run** unapplied events, newest first (`ORDER BY received_at DESC, id`), up to
   `limit` (500), with no age window (M1's 30-day window is dropped): each in its own
   transaction under the same timeouts, through `process_event(retry=True)`, which takes
   the row `FOR UPDATE SKIP LOCKED` (a row a delivery holds is skipped for this run, T10).
   An event that raises is rolled back and recorded by the second transaction's statement
   (S4 `retry` on its first failure), and the run continues with the next event. There is
   no back-off: every unapplied event is re-run every minute, so a release that fixes the
   handler applies a poison event within a minute of the deploy. Newest first keeps a new
   T4-race clawback or a newly stored purchase ahead of any backlog (T21, T22); with more
   than 500 unapplied events, the oldest are not re-run until an admin works the backlog
   down (they are stale, S1b);
3. **stale marking**, set-based (§8.6: `UPDATE ... SET stale_reported_at = %(now)s WHERE id
   IN (SELECT id ... ORDER BY received_at, id FOR UPDATE SKIP LOCKED) RETURNING type`: the
   purchase's lock order, never waiting).

Any failure outside a per-event attempt (the sweep, the event list, stale marking), and an
attempt whose error store fails too (`StoreFailed`: Postgres is not answering), ends the
run there. Only a run that finishes step 3 sets
`loom_billing_stripe_retry_last_success_seconds` (alarm S5), so a job that fails every
minute (and so never marks anything stale) is itself alarmed.

### 8.4 Idempotency layers

| Layer | Guarantees | Enforced by |
|---|---|---|
| L1 event id | An event is applied at most once, however often it is delivered, replayed or resent | `stripe_events.id` primary key; `INSERT ... ON CONFLICT DO NOTHING` waits for a concurrent uncommitted insert of the same id; `processed_at` checked under `FOR UPDATE` |
| L2 ledger key | Even if L1 were bypassed, one ledger transaction per event | `ledger_transactions.idempotency_key` unique, `stripe:<event id>` (contracts §3) |
| L3 session | One purchase per Checkout Session across event ids (`completed` and `async_payment_succeeded`) | Advisory lock on the session id + look-up (M1; M3 drops the look-up's `org_id` filter so it has the index's predicate, `kind = 'purchase' AND external_ref = <session id>`), and **new**: migration 0012's unique partial index `ledger_transactions_purchase_external_ref` on `ledger_transactions (external_ref) WHERE kind = 'purchase' AND external_ref IS NOT NULL`. A race past the look-up fails with `unique_violation` on that index, which `loom_platform.ledger.post_transaction` raises as `IdempotencyConflict` like any unique violation (`ledger.py` unchanged; round-18 probe: `23505`, `constraint_name` = the index): the delivery rolls back, is stored with `last_error` and answers 500, and the retry (Stripe's or R's) finds `already_credited` |
| L4 clawback total | Refunds and disputes never take back more than the purchase; each cumulative refund amount once; each dispute once | `stripe_payments` row lock; `refunded_micros` high-water mark; `clawed_back_micros <= amount_micros` check constraint; a dispute is refused when a `refund` with its dispute id exists (M1). Only the handler posts clawbacks (`resolve` posts nothing, §8.6) |

Migration 0012 alters `stripe_events` first and builds the `ledger_transactions` index
after (the order a running M1 delivery takes its locks, so the migration cannot deadlock
with one). It adds to `stripe_events`: `result` (the last outcome, e.g. `credited`,
`unapplied`, `resolved_by_admin`); `last_error`, `last_error_at`, `error_count` (default
0); `stale_reported_at` (§8.6). No new table. The unprocessed-row index of 0009 is reused
(it also serves R's newest-first scan); 0012 adds a partial index for the reopen sweep
(§8.6) on `(payload -> 'data' -> 'object' ->> 'payment_intent') WHERE result =
'resolved_by_admin'`. All new columns are nullable or defaulted, so the running M1 code
keeps working; the runtime role's grants cover them (a grant in the migration where
contracts §3 requires one). It fails, and the deploy stops at step 3, if a database
already holds two purchases with the same `external_ref` (none can, by L3's look-up; the
migration says so).

### 8.5 Missed deliveries: alarms, a health check and a manual resend

Stripe gives up on a delivery after a few hours in test mode (§8.3). M3 does not fetch
missed events itself (D15; §8.12 compares this with an Events-API sync). Instead every
failure billing, the heartbeat or the endpoint health check can see raises an alarm, and
the operator resends what Stripe gave up on:

| A delivery fails because | Seen as | Alarm |
|---|---|---|
| Postgres down or slow, a lock wait | 503 (nothing stored) or 500 (stored) from billing; the retry job stops when Postgres is down | S2c, S5 |
| A handler bug | 500 from billing, a stored poison event | S2c, S4, then S1a |
| Billing stopped or starting (Caddy answers 502) | the heartbeat; the endpoint health check | H1, H3 |
| The box or instance down | heartbeat, status checks, the endpoint health check | H1, H2, H3 |
| DNS, TLS or Caddy down or misrouted | the endpoint health check | H3 |
| A wrong webhook secret on the box | `signature_mismatch` | S2a |
| The box's clock off | `stale_timestamp` | S2b |
| Stored, but not applicable | stale event | S1a, then S1b daily |
| A payload `jsonb` refuses (§8.13) | 503 on every attempt | S2c |
| The endpoint disabled or edited in Stripe; a fault only on Stripe's side or on its path to the box; an invalid or expired certificate | nothing Loom sees | none: uncovered (T26, §8.10) |

**Endpoint health check (H3).** A Route 53 health check (`healthcheck.tf`, §5.2) requests
`https://hooks.<domain>/stripe/health` from Route 53's checkers on the internet every 30 s:
type `HTTPS_STR_MATCH`, `fqdn = "hooks.<domain>"` (Route 53 resolves it through public
DNS, with or without `route53_zone_id`), port 443, SNI on, failure threshold 3, search
string `stripe_webhook` (Route 53 looks in the first 5,120 bytes of a 2xx/3xx answer).
Caddy routes `GET /stripe/health` on the `hooks.` host to billing, beside `POST
/stripe/webhook` (§6.7). Billing's API process answers it from the same app and port as
the webhook: 200 `{"listener":"stripe_webhook","status":"ok"}`, with no signature, no
Postgres, Redis or Stripe call and nothing else in the body (no mode, no version); the API
process listens only once its pool is open (T13). A healthy check proves that public DNS
resolves `hooks.`, a TLS handshake with SNI succeeds on 443, Caddy routes the host to
billing, and billing's webhook listener answers. It does not prove the certificate valid
(Route 53 does not validate certificates; Caddy renews 30 days before expiry), Stripe's
endpoint enabled and pointed here, or Postgres up (S2c, S5). Alarm H3 (§5.3) reads
`AWS/Route53 HealthCheckStatus`, which Route 53 publishes only in us-east-1, so it is
created through the provider alias `aws.us_east_1`; its action, `loom-platform-alerts`,
must be in the same region, so `healthcheck.tf` fails the plan unless
`data.aws_region.current.region` (provider 6.x; `name` is deprecated) is `us-east-1`
(where §3 puts the stack). Cost in §14.

**Runbook** (`docs/runbook-stripe.md`, "Missed deliveries"), after any of these alarms and
once the cause is fixed: (1) in the Stripe dashboard (the endpoint's event deliveries), list
the events of the five subscribed types whose deliveries did not get a 2xx from one hour
before the alarm until now; (2) resend each to the endpoint (the dashboard's resend, or
`stripe events resend <event id> --webhook-endpoint <we_...>` with the operator's own key,
never billing's); (3) check that each answered 200 and that `loom-billing stripe-event list
--unapplied` shows nothing new. Resending more than needed is harmless: L1 answers
`duplicate`. A resend carries a fresh signature, so the 300-s tolerance does not stand in
its way. How long Stripe keeps letting an event be resent is checked at Stage B (B7) and
written into the runbook; an event past that is lost to Loom (§8.10). The runbook's S4
entry says to read the event's `last_error` first: a lock or statement timeout there is a
transient failure, recorded like a handler bug (§8.10), and R has usually applied the event
by the next minute.

### 8.6 Stuck events: staleness, `resolve` and the reopen sweep

**Stale events and the alarm.** `retry_unapplied` marks an event stale (set-based, §8.3,
step 3) when it has been unapplied for `LOOM_STRIPE_STALE_AFTER_S` (900 s) since
`received_at` and `stale_reported_at` is NULL. It sets `stale_reported_at` once and
increments `loom_billing_stripe_unapplied_stale_total{type}` once; `type` is one of the five
subscribed types or `other`, every value initialised at 0 at start. Alarm S1a fires for each
new stuck event, so one old stuck event never hides a new one. The gauge
`loom_billing_stripe_unapplied_stale_events` is **the number of unapplied events with
`stale_reported_at` set** (S1b; a restart that loses an unscraped increment still leaves
S1b). The existing gauge `loom_billing_stripe_unapplied_events` stays as information. A
T4-race clawback is applied by R within a minute and never counts. A clawback the sweep
reopens has `stale_reported_at` cleared, so it counts once more if it stays unapplied.

**Resolve.** `loom-billing stripe-event list [--unapplied]` and `loom-billing stripe-event
resolve <event id> --admin-email <email> --reason <text>`. In one transaction, under the
webhook's `lock_timeout` and `statement_timeout`, `resolve`: checks the admin
(`users.role = 'admin'`); locks the event row `FOR UPDATE`; refuses an unknown or already
processed event; for a clawback (`charge.refunded`, `charge.dispute.created`), reads its
PaymentIntent from the payload and refuses when it is not a non-empty string (a payload
shape the handler does not read, fixed forward like a poison event) or has a
`stripe_payments` row (only a release that fixes the handler applies such a clawback); otherwise sets
`processed_at`, `result = 'resolved_by_admin'`, and writes `admin_audit_log`
(`stripe_event.resolve`, the reason in `details`). It **posts nothing, ever**. Once
resolved, no retry or redelivery applies the event (L1), with one exception: the sweep
below.

**Reopen sweep.** A clawback resolved while its payment had no `stripe_payments` row must
still be applied if that purchase is credited later, and the purchase's own pending query
(M1's) does not see it, because it is processed. So R's first step reopens every such
clawback whose payment now exists: `UPDATE stripe_events SET processed_at = NULL, result =
'unapplied', stale_reported_at = NULL, last_error = 'reopened: payment credited after
resolve' WHERE id IN (SELECT e.id FROM stripe_events e JOIN stripe_payments p ON
p.payment_intent = e.payload -> 'data' -> 'object' ->> 'payment_intent' WHERE e.result =
'resolved_by_admin' AND e.type = ANY(<the two clawback types>) ORDER BY e.received_at, e.id
FOR UPDATE OF e SKIP LOCKED) RETURNING id` (each id logged at warning). A reopened clawback
is an ordinary unapplied event: step 2 applies it, or it keeps `last_error`, goes stale
again (S1a) and waits for a fix. Credit events are never reopened. No lock order is
involved: `resolve` and the purchase's query take only unprocessed rows (a row a resolve
committed while the purchase waited on it no longer matches), and the sweep skips held rows
until the next run. So whichever of a resolve and a purchase commits first, the clawback is
applied once, by the purchase or by the first run after both (T24): `resolve` can delay a
clawback of a credited payment by about a minute, as T4's race does, but never drop it.
(Round-18 probe: the sweep skips a held row, reopens it on the next run, R applies it once.)

*Why not an acknowledge-only `resolve`:* it would have to exempt the clawback from
staleness only until its payment appears (the park/unpark rule behind the majors of rounds
12, 14 and 15); the sweep leaves `resolve`, staleness and the purchase's query as they are.

What `resolve` is for (the runbook lists the cases):

- **A poison event** (a handler bug, T17) is not resolved but fixed forward: a release that
  fixes the handler lets R apply it within a minute of the deploy (R re-runs every
  unapplied event each minute). For a clawback whose payment exists this is the only exit
  (`resolve` refuses it), so a refund or dispute of a credited payment is never dropped by
  hand.
- **A credit event that cannot be applied** (unknown org, wrong currency, no amount: none
  can come from a Loom checkout on a Loom-only sandbox, so in practice an operator's own
  test, T11) is resolved; nothing is credited. Should a real customer's payment ever land
  here, the admin does not resolve it: a release that fixes the cause applies it (it stays
  unapplied until then). M3 has no hand credit for a Stripe payment (§8.10).
- **A clawback with no payment** (a refund of a payment Loom never sold, e.g. one made in
  the dashboard; or of a purchase that is itself still stuck) is resolved; nothing is taken
  back now, and if its purchase is credited later, the sweep reopens and applies it. The
  runbook says never to make a hand adjustment for a resolved clawback whose payment may
  still arrive (round 18 n3): once that purchase is credited the sweep reopens the clawback
  and the handler takes the amount back, so a hand adjustment would take it twice.
- Other types are already `ignored` (processed) and never stuck.

### 8.7 Traces

Each trace is a test (A16-A24, A31-A34) unless marked: T12's metric is checked in A22, T15
at Stage B (B23), and T26 is an accepted limitation (§8.10). T20 (a COMMIT landing after
the 6 s deadline) was retired with the deadline in round 18: a COMMIT that lands after
Stripe stopped waiting is T9 (b). `W` = webhook delivery, `R` = `retry_unapplied`; `E1`,
`E2` event ids; `cs` a Checkout Session; `pi` its PaymentIntent.

| # | Trace | Steps | Outcome |
|---|---|---|---|
| T1 | Duplicate delivery | W(E1 completed, paid) commits, answers 200; W(E1) again | Second: row exists, `processed_at` set -> `duplicate`, 200; one `purchase` |
| T2 | Same event, two workers at once | W1(E1) inserts the row (uncommitted); W2(E1)'s `INSERT ... ON CONFLICT` waits on W1's uncommitted row; W1 commits | W2's insert does nothing, `FOR UPDATE` reads the committed row: `duplicate`, 200. If W1 fails instead, W2's insert proceeds and W2 applies it; W1's second transaction then finds the event processed and answers `duplicate` (if W2 commits within `lock_timeout`; otherwise 503, and Stripe's retry finds `duplicate`). One `purchase` either way |
| T3 | Two event ids, one session | E1 `completed` (unpaid, delayed method) -> `not_paid`, processed; E2 `async_payment_succeeded` -> credit. Reversed order: E2 credits, E1 `not_paid`. Both carry `paid` and run concurrently: both take the session's advisory lock in turn; the second's look-up (a new statement after the lock, READ COMMITTED) sees the first's purchase -> `already_credited` | One `purchase` per session; L3's index catches anything that slips past (rollback, stored with `last_error`, 500; the retry -> `already_credited`) |
| T4 | Refund before purchase | W(E2 `charge.refunded`, pi) arrives first: no `stripe_payments` row -> stored unapplied, 200. W(E1 `completed`) credits, inserts `stripe_payments(pi)`, then locks and applies pending clawbacks of `pi` in the same transaction | Credit then clawback, both in E1's commit. **Concurrent variant**: E2's transaction reads `stripe_payments` before E1 commits (no row: unapplied) and commits after E1's pending-clawback query ran (E2 not yet visible to it): both commit, E2 unapplied; R (newest first) applies it within a minute, before it can count as stale (15 min) |
| T5 | Partial refunds out of order | Charge refunded 500 then 1000 cumulative (E2: 500, E3: 1000). Order E3, E2: E3 takes 1000 (high-water 1000); E2: 500 - 1000 <= 0 -> `already_refunded`. Order E2, E3: 500 + 500. Concurrent: both lock the `stripe_payments` row; the second re-reads the updated row after the first commits | 1000 taken back in total in every order |
| T6 | Refund and dispute together | 600 refunded, then a dispute for 1000 on a 1000 purchase | Dispute takes `min(1000, 1000 - 600)` = 400; total 1000, never more than the purchase. Balance may go negative (money returned) |
| T7 | Replay | (a) A captured request re-sent 10 min later: `t` outside 300 s -> 400. (b) Re-sent within 300 s: signature valid -> L1 `duplicate`, 200. (c) Re-sent while the first was stored unapplied: re-runs the same idempotent handling | No double credit; (a) never reaches Postgres |
| T8 | Postgres down | The pool wait times out at 5 s, the second transaction's too: 503 after about 10 s, nothing stored; Stripe retries; R's run ends at its first statement (S5 within 5 min) and the 503s count (S2c). Outage shorter than Stripe's schedule: a retry stores and applies it. Longer (test mode: hours): Stripe gives up; once Postgres is back the operator resends the events of the outage (§8.5) | Applied once, late, after the operator acts; an event past Stripe's resend limit is lost (§8.10) |
| T9 | Crash between ledger write and 2xx | (a) Before COMMIT (process killed, box lost): Postgres rolls back; no 2xx; Stripe retries -> applied. (b) After COMMIT, before the response (or Stripe stopped waiting): Stripe retries -> `duplicate`, 200. (c) COMMIT sent, connection lost before its result: the second transaction runs on another connection: if the COMMIT landed, its guard finds the event processed -> 200 `duplicate`; if not, it stores it with `last_error` -> 500 and a retry applies it; if it cannot connect either -> 503 and a retry finds `duplicate` or applies it | Exactly one ledger transaction in each case (a test injects each failure point) |
| T10 | Webhook and retry at once | W(E1) holds E1's row while R runs | R takes rows `FOR UPDATE SKIP LOCKED`, so it never waits on a delivery's event row: it skips E1 this run; one application |
| T11 | Signed but not applicable | `completed` for an unknown org, wrong currency, no amount | Stored unapplied, logged at error, 200 (Stripe retrying cannot fix it); after 15 min S1a fires once for it; R re-runs it each minute until an admin resolves it (§8.6), audited, nothing credited. A second such event later fires S1a again |
| T12 | Clock skew on the box | The box's clock drifts > 300 s: every delivery 400 | Counted as `stale_timestamp`, alarm S2b; Stripe keeps retrying (new timestamps); once chrony corrects the clock, retries succeed; deliveries Stripe gave up on are resent (§8.5). Prevented by chrony on Amazon Time Sync (§5.2) |
| T13 | Billing starting or role gate waiting | The API process does not listen until its pool is open (deferred role check, contracts §3): Caddy answers 502 | Stripe retries a 502 like any non-2xx; S2c does not count it (billing never saw it); a longer stop shows as H1 and H3; A19's role-gate case uses an injected unopened pool and expects 503 (the second transaction cannot get a connection either) |
| T14 | Redis down after COMMIT | The credit refresh fails | Logged; Postgres has the credit; reconcile writes it within a minute, the `loom_credit_changed` listener earlier |
| T15 | Secret rolled in the dashboard | Stripe signs with old and new secrets during the roll (both `v1` values sent) | Swapping the secret in Secrets Manager and redeploying billing at any point in Stripe's overlap window keeps every delivery verifiable (runbook B23) |
| T16 | Org suspended or rejected after paying | `completed` for a non-active org | Credited anyway (the money was received; the org cannot spend it while not active); an admin refunds if needed |
| T17 | Poison event | E1 makes `handle_event` raise every time (a handler bug on a payload shape) | W: roll back, second transaction stores E1 unapplied with `last_error` (`error_count` 1), 500, S2c, S4 once. Stripe's retries re-run it and fail the same way (`error_count` grows, no new S4). R re-runs it every minute, fails, **continues** with the other events. After 15 min S1a. A release that fixes the handler lets R apply it within a minute of the deploy. Newer events are never blocked. A payload `jsonb` refuses (`\u0000`, a lone surrogate) is not stored at all: 503 on every attempt (S2c), lost once Stripe gives up (accepted in test mode, §8.10; a live-mode prerequisite, §8.13) |
| T18 | Dispute won, then refunded | Dispute of 1000 claws back 1000; Loom wins; an admin adjustment gives 1000 back; later a full refund of the charge | `clawed_back_micros` is still 1000, so the refund takes `min(1000, 0)` = 0: the customer keeps the credit and gets the refund. Accepted limitation (§8.10): the runbook tells the admin to reverse a won-dispute adjustment by hand if the charge is later refunded |
| T19 | Poison clawback before its purchase | E2 (`charge.refunded`, pi) arrives first and raises: stored unapplied with `last_error`. E1 (`completed`, pi) credits, inserts `stripe_payments`, then applies pending clawbacks of pi, each in a savepoint: E2 raises again, rolls back to its savepoint, keeps `last_error` | E1 commits: the customer gets the credit. E2 stays unapplied (S4 on its first failure, then S1a at 15 min); a fixed release applies it (`resolve` refuses it: its payment exists) |
| T21 | Unapplied backlog | 600 unapplied rows (old stuck events not yet resolved) and a new T4-race clawback | R re-runs newest first: the clawback is among the newest 500 and is applied at the next run, never stale; the 100 oldest are not re-run while the backlog lasts (all stale, in S1b) |
| T22 | Backlog and a new stored purchase | 600 unapplied rows, all re-run each minute (e.g. unknown-org checkouts of an operator's test); a purchase is stored unapplied by the 500 path | Newest first: the purchase is re-run at the next run, not after the backlog |
| T23 | Two clawbacks before the purchase | E2 (`charge.refunded`, pi) and E3 (`charge.dispute.created`, pi) stored unapplied; E1 (`completed`, pi) credits, then applies E2 in savepoint 1 and E3 in savepoint 2 | Plain savepoints, no `SET CONSTRAINTS`: the deferred ledger checks run once, at E1's COMMIT, and pass (`loom_post_transaction` refuses an unbalanced post up front): both clawbacks commit in E1's transaction (contracts §3), as the protected M1 test `test_refund_and_dispute_before_the_purchase_are_applied_with_it` requires (round-18 probe) |
| T24 | Clawback resolved, payment credited later | E2 `charge.refunded(pi, 500)` is stuck with no payment because its purchase E1 (1000) is a poison event; the admin resolves E2; a fixed release lets R apply E1, whose pending query skips the resolved E2. Also: E1 in flight while the admin resolves E2 | The next R run's sweep reopens E2 and applies it: 500 taken back about a minute after E1's commit; if that fails, E2 is unapplied with `last_error`, counted stale again (S1a), and R owns it. In flight: if E1 takes E2's row first, E1 applies it and the resolve refuses (processed); if the resolve commits first, E1 skips E2 and the sweep reopens it. Applied once either way (§8.6) |
| T25 | Wrong secret on the box | A deploy renders a `whsec_` that is not the endpoint's: every delivery 400 | Counted as `signature_mismatch`, alarm S2a within 15 min; the operator fixes the secret and redeploys billing; Stripe's later retries succeed; deliveries it gave up on are resent (§8.5) |
| T26 | Failure H3 cannot see | The endpoint disabled or edited in Stripe, a fault only on Stripe's side or on its path to the box, or an invalid or expired certificate, for longer than Stripe's retries | No Loom alarm. Not a test: an accepted limitation (§8.10); found in Stripe's dashboard or from a customer, then resent (§8.5). DNS, TLS, Caddy or billing down as seen from the internet raise H3 (§8.5; B25) |

### 8.8 Checkout creation, keys and secrets

- The web app's checkout path is unchanged (`/dashboard/billing` -> billing's
  `POST /internal/checkout-sessions`, contracts §7). Billing now also sets
  `expires_at` = now + 1 h on each session (unpaid sessions stop being payable sooner; the
  minimum Stripe accepts is 30 minutes) and turns off Adaptive Pricing for the session
  (`adaptive_pricing[enabled]=false`), so the customer always pays in USD (a non-USD
  session would be stored unapplied); B7 checks the parameter on the real sandbox. The
  line item's product name is `<name> credits` with the name from `LOOM_BRAND_NAME`
  (`API credits` when unset, §6.11).
- A double-click creates two sessions; only a paid one credits, once (L3).
- `STRIPE_SECRET_KEY` is the restricted `rk_test_` key with **Checkout Sessions write**
  only (billing reads nothing else from Stripe's API); `STRIPE_WEBHOOK_SECRET` the
  endpoint's `whsec_`. Both from `loom/platform/stripe`; billing refuses
  `sk_live_`/`rk_live_` and live-mode objects as today.
- Refunds, disputes and resends in the acceptance are made by the operator in the Stripe
  dashboard (or with their own key from their own machine), never with billing's key.

### 8.9 The Stripe stub (tests and local dev)

The stub (`loom_billing.stripe_stub`, a test double, protected like the test fakes) gains
`POST /stub/control` (`{"drop_deliveries": bool, "fail_status": int | null}`: drop or fail
its webhook deliveries, for the missed-delivery tests) and `expires_at` on sessions. Tests
resend an event by re-posting the stored event with a fresh signature, which is what
Stripe's resend does. The holdout tests do not use the stub: they drive `process_event` and
`retry_unapplied` against Postgres.

### 8.10 Accepted limitations (documented, not built)

- **No automatic recovery of missed deliveries** (D15, §8.12): a delivery Stripe gave up on
  is applied only when the operator resends it (hours or days late), and the failures H3
  cannot see (the endpoint disabled or edited in Stripe, a fault only on Stripe's side or
  its path to the box, an invalid certificate) raise no Loom alarm (T26). An event Stripe
  no longer lets anyone resend (B7 records how long) is lost to Loom. **An Events-API sync
  (or an equivalent) is required before live mode** (§8.13).
- **An event whose payload `jsonb` refuses** (`\u0000`, a lone surrogate, a non-finite
  number) cannot be stored: 503 on every attempt (S2c) and lost once Stripe gives up. Only
  the operator's own test payments reach the sandbox, so none is expected (T17; §8.13).
- **Transient failures are recorded like handler bugs**: a lock or statement timeout, a
  deadlock or a lost connection while Postgres still answers the second transaction makes a
  500 with `last_error`, S2c on the webhook path and one S4 for the event, although nothing
  is wrong with it; R applies it at its next run. The runbook's S4 entry says so (§8.5).
- **No bound on the answer time** beyond the sum of the timeouts (about 30 s, §8.3): Stripe
  treats an answer it stopped waiting for as a failed delivery and retries; L1 makes a late
  COMMIT safe.
- **The sandbox is Loom's alone** (D4): a payment made in it by other means (the
  dashboard, a laptop in `stripe` mode) yields stuck events that an admin resolves; laptops
  use the stub or a sandbox of their own (billing README).
- **No hand credit or hand clawback for a Stripe event**: `resolve` posts nothing. A payment
  that must be credited waits for a release that fixes its handling; a refund or dispute of
  a credited payment is applied only by the handler. An admin credit adjustment
  (contracts §4) stays possible but is booked against `promo_expense`, not linked to the
  event.
- A dispute that Loom **wins** (`charge.dispute.closed`, `won`) does not reinstate the
  credit; an admin adjustment does (contracts §13; §15 D5). That adjustment does not lower
  `clawed_back_micros`, so a later refund of the same charge takes nothing back (T18): the
  admin reverses the adjustment by hand (runbook).
- A refund that later **fails** (Stripe's `charge.refund.updated` with status `failed`)
  keeps its clawback: the customer is short until an admin adjustment. Not subscribed.

### 8.11 Entry points (fixed for the test writers)

- `loom_billing.webhooks.process_event(pool, event, *, retry=False, settings=None,
  now=None) -> Outcome`: **the one entry point that owns §8.3's Postgres behaviour** (the
  transaction with its `SET LOCAL` timeouts, `handle_event`, the second transaction,
  outcome metrics and S4 after COMMIT). It checks the event's shape first (`BadEvent`,
  before any transaction). Any exception from the Postgres part runs the second
  transaction: an event it stores returns `Outcome("unapplied",
  stored_after_error=True)`, and an event its guard finds processed returns
  `Outcome("duplicate")`; if the second transaction fails too, it raises
  `loom_billing.webhooks.StoreFailed` with the original exception as `__cause__`. The
  webhook route (signature, body, the credit refresh) answers 500 for
  `stored_after_error`, 503 for `StoreFailed`, 200 otherwise; `retry_unapplied` calls it
  per event and ends its run on `StoreFailed`. No exception class or SQLSTATE is
  inspected anywhere.
- `process_event(..., retry=True)` takes the event row `FOR UPDATE SKIP LOCKED` and
  returns `Outcome("skipped")` when a delivery holds it; `Outcome("skipped").processed` is
  False (like `unapplied`). Its failures are counted in S4 with `path="retry"`.
- `loom_billing.webhooks.handle_event(conn, event, *, retry=False, settings=None,
  now=None) -> Outcome`: the in-transaction core, never calls Stripe; the M1 call
  `handle_event(conn, event)` keeps working. It writes `result` and clears `last_error` and
  `error_count` for every outcome except `duplicate` (§8.3). Called directly it counts no
  outcome metric; `process_event` counts them after COMMIT (no protected test reads these
  counters after a direct call).
- `loom_billing.webhooks.retry_unapplied(pool, *, limit=500, settings=None, now=None) ->
  int`: the M1 call `retry_unapplied(pool)` keeps working. It keeps M1's `-> int` (events
  this run marked processed: every result except `unapplied`, `skipped` and `duplicate`)
  and shows a failed run only by not setting its last-success gauge (S5); it never raises.
  Steps and order as in §8.3: sweep, newest-first re-run up to `limit` with no age window,
  stale marking. A stored row that fails the shape check (`BadEvent`; only a test can store
  one) is skipped with no error store, as in M1; it goes stale and an admin resolves it.
- The purchase's pending-clawback query is M1's (`processed_at IS NULL`, clawback types of
  its PaymentIntent, `ORDER BY received_at, id FOR UPDATE`); each clawback in a plain
  savepoint, no `SET CONSTRAINTS` anywhere in billing.
- `last_error` is `<exception class>: <message>`, NUL replaced, at most 1,000 characters;
  the sweep writes `reopened: payment credited after resolve`. S4 counts a write that makes
  `error_count` 1, with `path` `webhook`, `retry` or `pending_clawback` (each initialised
  at 0).
- `stripe_sig.verify` raises `StaleTimestamp` (valid HMAC, `t` outside the tolerance) and
  `SignatureMismatch` (a header with `t` and at least one `v1`, `t` within the tolerance,
  no `v1` matching), both new subclasses of `SignatureError` (existing callers catching
  `SignatureError` are unchanged); the route maps them to the metric results
  `stale_timestamp` and `signature_mismatch`, and answers 400 `bad_signature` in every case.
- `GET /stripe/health` on billing's API app (`loom_billing.api.create_app`): 200
  `{"listener":"stripe_webhook","status":"ok"}`, no authentication, no Postgres, Redis or
  Stripe call (H3, §8.5). The existing `GET /health` is unchanged and never routed
  publicly.
- The credit refresh bound is the constant `REFRESH_WAIT_S = 2.0` in `loom_billing.api`
  (no setting).
- Faults in tests are injected at the connection level (a TCP proxy, or a trigger in a
  scratch database raising a chosen SQLSTATE), never by patching internals, so the hidden
  (holdout) tests and the visible ones need no private seam.
- `settings` is a `BillingSettings`; `None` means the defaults of §11, which is the M1
  behaviour. The timeouts reach the handler only this way, never from the environment at
  call time.
- `now`: every timestamp the M3 code writes or compares (`received_at`, `last_error_at`,
  `stale_reported_at`, `processed_at` of a resolve, staleness) uses one `now` per call, the
  injected one or `datetime.now(UTC)` read once, passed to SQL as a parameter.
- The 15-min staleness is `LOOM_STRIPE_STALE_AFTER_S` (900).
- `loom_billing.resolve.resolve_event(pool, event_id, *, admin_email, reason,
  settings=None, now=None) -> Outcome` behind the CLI (its `Outcome` has `result =
  "resolved_by_admin"`, nothing posted), and `loom_billing.resolve.list_events(pool, *,
  unapplied=False) -> list[dict]` (keys `id`, `type`, `received_at`, `result`,
  `last_error`, `error_count`, `stale_reported_at`). Every refusal (a non-admin, an unknown
  or processed event, a clawback without a non-empty string PaymentIntent or whose payment
  exists) raises
  `loom_billing.resolve.ResolveRefused(reason)` with nothing changed; the CLI prints the
  reason and exits 2.
- The index name `ledger_transactions_purchase_external_ref`; `loom_platform.ledger` is
  unchanged.
- Settings: the `LOOM_STRIPE_*` variables of §11 and `LOOM_BRAND_NAME`, all read through
  `BillingSettings` as `stripe_webhook_db_timeout_s`, `stripe_lock_timeout_ms`,
  `stripe_statement_timeout_ms`, `stripe_stale_after_s` and `brand_name`.
- A clawback applied by its purchase gets the `result` its own outcome names (`refunded`
  or `already_refunded`), as on a direct delivery; the metric keeps M1's `late_<result>`
  label. `list_events` orders by `received_at DESC, id`. The sweep leaves `error_count` as
  it was, so whether a reopened clawback's next failure counts S4 depends on its history;
  tests do not assert S4 there (S1a covers it).

### 8.12 Decision record: Events-API sync or alarm and resend (D15)

Compared before the post-cut review (2026-11-18). Stripe is in test mode through Phase 1.

| | (a) Events-API sync (the design of rounds 1-16) | (b) Alarms + manual resend (chosen) |
|---|---|---|
| Spec | ~90 lines (the old §8.5) + ~70 elsewhere (5 traces, the sync item of the done-when list and parts of two more, ~15 entry-point lines, 2 alarms, the stub's events list, B7, m3-s, §11) | ~30 lines (§8.5's table and runbook, alarm S2a, T25, T26, an A22 case, m3-s, §8.10) |
| Tests | ~20 cases in the done-when list + 5 traces | 2 cases (`signature_mismatch`; a failed delivery, then a resend applies once) + the alarm rows of A8/A9 |
| Code (rough) | ~600-800 lines (sync module, window, backfill cursor, blocked and passed ids, manual/automatic runs, CLI, stub listing) + ~1,000 lines of tests; migration: tables `stripe_sync_state` (7 columns) and `stripe_sync_passed`, column `source`; 6 settings; 6 metrics; key permission Events read | ~20 lines (one exception subclass, one metric label) + a runbook section |
| Review record | 4 of the 26 majors (1.1, 3.2, 10.1, 11.1) and ~30 minors were in the sync | none yet |
| Left uncovered | Events older than Stripe's 30-day retention; a run blocked by a Stripe API outage waits for the next run | No automatic recovery: a missed delivery is applied only when the operator resends it (hours to days); failures before billing (DNS, TLS, Caddy, the endpoint disabled in Stripe) raise no Loom alarm (T26; since round 18 the H3 health check covers DNS, TLS, Caddy and billing, §8.5); events past Stripe's resend limit are lost |

Recommendation and decision: **(b)** for Phase 1. Its gaps cost a delay, not money, while
Stripe is in test mode, and the operator is the only payer; (a) was the largest single
block of the design and of its review findings. **Before live mode** an Events-API sync (or
an equivalent) is required (§8.13). Removed with it: the sync and its CLI, `floor`,
`synced_through`, the backfill, blocked and passed ids, the six `LOOM_STRIPE_SYNC_*`
settings, alarms S2a (recovered/gaps) and S3 (sync stuck) of the earlier draft (S3 is not reused; S2a now means a secret mismatch), the
`source` column, and the key's Events-read permission.

### 8.13 Live-mode prerequisites

Phase 1 stays in Stripe test mode. Everything below is required before live mode; §12
copies this list into `docs/phase1-runpod-plan.md` §13 ("Live-mode prerequisites").

| # | Prerequisite | Why live mode needs it | Why test mode does not |
|---|---|---|---|
| LM1 | An Events-API sync (or an equivalent) of deliveries Stripe gave up on (D15, §8.12) | A missed delivery is a real customer's payment, refund or dispute; while it waits for the operator, a customer is short of credit or keeps credit they were refunded, and the failures H3 cannot see (endpoint disabled in Stripe, certificate) and events past the resend limit lose money | The operator is the only payer; a missed event costs a delay, and the alarms and the runbook resend cover it |
| LM2 | A Loom-only live account, or foreign-event handling again (cut with D4) | Events of other products on a shared account would arrive as stuck clawbacks and checkouts for an admin to resolve by hand | The sandbox is Loom's alone (D4) |
| LM3 | Payload sanitising, or a store for the raw body, so every signed event can be stored (removed in round 18) | Live payloads carry customer-entered text (names, addresses, custom fields) that may hold characters `jsonb` refuses; such an event gets 503 on every attempt and is lost when Stripe gives up (§8.10) | Payloads come only from the operator's own test payments |
| LM4 | Telling transient failures from handler failures: a lock or statement timeout, deadlock or lost connection answered 503 without `last_error` and without S4 (the SQLSTATE classification and failure identity removed in round 18, or a simpler equivalent) | At live volume transient lock waits happen; each would raise a false S4 and S2c, and S4 would stop meaning "a handler bug" | A few deliveries a day; a false S4 is rare and the runbook covers it (§8.10) |
| LM5 | A re-run policy for a backlog: back-off for failing events and an order that neither starves old events nor delays new ones (the back-off and the step tiers removed in round 18) | A live backlog of stuck events grows; re-running all of them every minute costs a transaction each, and past 500 the oldest are never re-run (T21) | The backlog is the operator's own test events, resolved by hand |
| LM6 | Disputes and refunds: a won dispute reinstated automatically (D5) with the won-then-refunded case (T18) handled, and a refund that later fails (`charge.refund.updated`) handled | Real money moves both ways; the hand adjustments of §8.10 do not scale and are easy to miss | Test disputes and refunds are the operator's own, adjusted by hand |
| LM7 | Lift the live-mode refusals deliberately (`sk_live_`/`rk_live_`, `livemode: true` -> 400) in a reviewed change, with a restricted live key (Checkout Sessions write, plus Events read for LM1) and a live endpoint and secret | Billing refuses live keys and live events today (§8.1) | Not used |
| LM8 | Decide the public API key prefix and header names (`loom_sk_`, `X-Loom-*`, `LOOM_API_KEY` in the playground snippet): a contracts change with a transition for issued keys (§6.11) | Paying customers write these names into their code; changing them later breaks their integrations | M3 keeps the `loom` wire names; testers are few and known |

Considered and **not** required for live mode: the 6 s deadline with its cancelled task (an
answer Stripe stopped waiting for and a 503 are the same failed delivery to Stripe, and L1
makes a late COMMIT safe); the 3-consecutive-stops rule (the error store's own failure is
the stop signal); the `IMMEDIATE`/`DEFERRED` pair (`loom_post_transaction` checks before
it inserts); `DuplicatePurchase` (the L3 index with the 500 path suffices, and LM4 would
treat the race as transient anyway).

## 9. Web app

Configured for production: `AUTH_URL=https://<domain>`, `AUTH_SECRET`,
`AUTH_GOOGLE_ID`/`_SECRET`,
`EMAIL_SERVER=smtps://<ses smtp user>:<password>@email-smtp.us-east-1.amazonaws.com:465`,
`EMAIL_FROM` (`<name> <no-reply@<domain>>`), `LOOM_BRAND_NAME`, `DATABASE_URL` with
`sslmode=verify-full` and the RDS CA, `LOOM_REDIS_URL` (`rediss://`) with
`NODE_EXTRA_CA_CERTS`, `LOOM_GATEWAY_URL=http://gateway:8080`,
`LOOM_PUBLIC_API_URL=https://api.<domain>/v1`, `LOOM_BILLING_URL=http://billing:8090`,
`LOOM_INTERNAL_TOKEN`; the gateway's `LOOM_CORS_ORIGINS=https://<domain>`. The values that
carry the domain or the name are rendered by `loom-deploy` from the site parameter (§6.11).
The CSP already allows `https://checkout.stripe.com` and Google.

One code change, not only configuration: the product name comes from `LOOM_BRAND_NAME`, read
at request time, and no user-facing text names the product otherwise (§6.11's list; A35,
A36). It is the deploy worker's (§16). If the production build or the local production
stack (A15) finds another needed web change, it is made in the deploy branch with a web
test.

## 10. M3 acceptance

`loom-m3-acceptance` (`services/devstack`, written as a test, protected) runs **on the box**
in a one-shot container on the production compose network (started by `loom-deploy
acceptance` through SSM), reaching the web app, gateway and Stripe webhook **through the
public URLs** (real DNS, real TLS) and Postgres, Redis and the containers from inside, as
M1's driver does. A second mode, `--external`, runs from the operator's machine and checks
only what the internet sees.

Sign-in: the automated items sign in test users by inserting an Auth.js database session
for them (the web app uses database sessions) and sending its cookie; the admin is granted
with contracts §4's SQL; the acceptance's users and orgs are marked and left in place (as
M1's are). Real sign-in paths are checked by hand (B24) (§15 D6).

| Item | What | How |
|---|---|---|
| m3-0 | M1's flow: sign up -> pending -> another admin approves -> $5 -> key -> OpenAI SDK against `https://api.<domain>/v1`, both models, stream and not, chat and completions -> playground -> usage rows and balance drop match the engines' counters | automated |
| m3-b | A revoked key (and a suspended org) refused within 5 s, timed from the commit, through the public URL | automated |
| m3-d | Gateway p50 added overhead < 10 ms on the box (M1 item d's probe from the poolmgr container, same mTLS path); the Caddy + TLS hop measured from the box against the public URL and reported, not gated | automated |
| m3-g | Stripe: the script creates a checkout as a test user through the web app and prints Stripe's URL; **the operator pays** with test card 4242 4242 4242 4242; the script then waits for exactly one `purchase` keyed `stripe:<event id>` with the paid amount, the Redis credit, and the dashboard's balance; then (1) re-posts the stored event with a fresh valid signature -> 200 `duplicate`, ledger unchanged; (2) re-posts with an old `t` -> 400; (3) **the operator resends** the event from the dashboard -> `duplicate`; (4) **the operator refunds** part, then the rest -> two `refund` transactions summing to the purchase, balance down by it | automated checks around manual steps |
| m3-s | Missed delivery and resend (§8.5): the script stops `billing` on the box; **the operator** makes a second test payment; Caddy answers 502 and Stripe's delivery fails; H1's and H3's alarm emails arrive; the script starts `billing` again; **the operator resends** the event from the dashboard (unless Stripe's own retry delivered it first) -> exactly one `purchase` for it; a second resend -> `duplicate` | manual steps, automated checks |
| m3-h | Exposure (`--external`): only 80/443 answer; `api.<domain>` serves `/v1/*` only; `hooks.<domain>` only `POST /stripe/webhook` and `GET /stripe/health` (200 with `stripe_webhook`); `/internal/checkout-sessions`, `/health`, `/metrics` unreachable from outside; 80 redirects; HSTS; certificate valid for the three names; the sign-in page's title and heading show `brand_name` (§6.11) | automated (external) |
| m3-i | No prompt or completion text or API key in any container's logs or in the CloudWatch log group | automated |
| m3-p | P4: the deploy log shows `gateway` healthy before `poolmgr` was updated; `loom-deploy apply --only poolmgr` with a newer poolmgr than gateway is refused | automated |
| m3-x | Containers cannot reach the instance metadata service (IMDS hop limit) | automated |
| m3-a | Alarm routes: `aws cloudwatch set-alarm-state` on each alarm -> email arrives; each §5.3 metric present in CloudWatch | runbook, manual |

Results go to `docs/phase1-m3-acceptance.md` (like M1's): numbers, the release sha, what
was manual and who did it.

## 11. Contract changes (`docs/phase1-contracts.md`, same change as the code)

| Section | Change |
|---|---|
| §1 Layout | Rows for `infra/aws/platform`, `infra/aws/state`, `deploy/` |
| §2 Services table | The production compose (`deploy/compose.prod.yml`): Caddy 80/443, the three host names and their routes (incl. `GET /stripe/health` on `hooks.`, §8.5); metrics ports, Prometheus and Grafana on `127.0.0.1` only |
| §2 `LOOM_ENV` row | Unchanged (the services' start-up checks stay as they are); a new note under §2: the production deploy (`loom-deploy`) requires `sslmode=verify-full`, the domain and the product name (§6.11) |
| §2 variables | New: `LOOM_POOLMGR_MOCK_BACKEND`, `LOOM_POOLMGR_GATEWAY_CERT_DIR`, `LOOM_STRIPE_STALE_AFTER_S`, `LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S`, `LOOM_STRIPE_LOCK_TIMEOUT_MS`, `LOOM_STRIPE_STATEMENT_TIMEOUT_MS`, `LOOM_BRAND_NAME` (web and billing, optional: the product name; unset = no name shown), `LOOM_DOMAIN` (Caddy, production compose only); Postgres conninfo keepalives and `tcp_user_timeout` for billing; changed: `LOOM_POOLMGR_ALLOW_CIDRS` (also with the mock backend outside dev, private ranges only), `LOOM_POD_CA_CERT` (may hold several CAs; reloaded), `LOOM_GATEWAY_CLIENT_CERT`/`_KEY` (reloaded; or the manager's combined file), `STRIPE_SECRET_KEY` (restricted key, Checkout Sessions write only), `EMAIL_FROM` (no default name) |
| §3 additions table and Kind table | Migration 0012: unique purchase per `external_ref` (index `ledger_transactions_purchase_external_ref`; a violation reaches callers as `IdempotencyConflict`, `post_transaction` unchanged); `stripe_events` columns `result`, `last_error`, `last_error_at`, `error_count`, `stale_reported_at` (grants as for the other Stripe columns; no new table); the `purchase` row says "at most one per Checkout Session (unique index)"; the sentence "applied in the purchase's transaction" gains its exceptions (a clawback that fails in its savepoint is left to `retry_unapplied`; a lock wait past `lock_timeout` fails the delivery, which is stored and retried) and the reopen sweep of a clawback an admin resolved before its payment existed (§8.6); `stripe_event.resolve` (posts nothing) as an `admin_audit_log` action |
| §3 "Roles" | On RDS the owner is the master user `loom_owner` (member of `rds_superuser`, not a superuser); only `migrate` uses it |
| §6 "Settle spool" | The M3 note becomes: the spool is on the data volume (`/var/lib/loom/gateway-spool`) |
| §6 "Shutdown" | The "M3 (ECS behind an ALB)" paragraph is replaced: compose `stop_grace_period: 30s`; Caddy `lb_try_duration` during a restart |
| §7 "Checkout" | The webhook response contract (§8.3: a failing event stored with its error and 500; 503 when nothing could be stored), the subscribed types (§8.2), missed deliveries (alarms, the `hooks.` health route and the runbook resend; no sync until live mode, §8.5, §8.13), the Loom-only sandbox, staleness, the `stripe-event list/resolve` CLI and the reopen sweep (§8.6), `retry_unapplied` continuing past a failing event, `expires_at`, the product name from `LOOM_BRAND_NAME`; the billing metrics the alarms read |
| §8 step 1 | The per-IP limit is Caddy's (§15 D2), not ALB + WAF |
| §9 | Gateway client-certificate rotation and reload; CA bundle (§7) |
| §10 | The manager's mock backend outside dev; the pool-reason and CA-expiry gauges |
| §13 | Close "Stripe's webhook endpoint must subscribe ..." (now §7); keep "Disputes won" open or close it per D5 |

## 12. Other doc changes

`docs/runbook-platform.md` (new: Stage B from apply to teardown, deploy, P4, rollback,
secrets, Stripe endpoint, SES, Google, alarms test), `docs/runbook-pod-ca.md` (new, P3), `docs/runbook-stripe.md` (new, §8.5, §8.6),
`infra/aws/platform/README.md`, `services/billing/README.md` (response contract, missed
deliveries and resend, the health route, `resolve` and the reopen sweep, the Loom-only
sandbox), `apps/web/README.md`
(`LOOM_BRAND_NAME`, `EMAIL_FROM`),
`services/poolmgr/README.md` (mock backend flag, gauges, `rekey`, `new-ca`),
`docs/security.md` ("Phase 1 control plane": IMDS hop limit, secrets as files, exposure),
`docs/phase1-runpod-plan.md` §13 (M3 status; P1-P4 done; a new "Live-mode prerequisites"
list copied from §8.13, LM1-LM8), `PLAN.md` status line and
question 4-6 answers, `docs/phase1-m3-acceptance.md` (at the end).

## 13. Done when

Each item says how it is checked. Postgres and Redis tests must **run**, not skip (CI sets
both URLs). "tftest" = `terraform test` under `mock_provider` (no credentials).

**Stage A (free)**

Terraform

- A1 [A] `terraform fmt -check`, `init -backend=false`, `validate` pass for
  `infra/aws/platform` and `infra/aws/state`; a CI job runs them with the pinned version.
  *Check: CI job.*
- A2 [A] Network: no ingress except 80/443 to the instance; RDS ingress only from the
  instance's security group; RDS not public; no NAT gateway. *tftest.*
- A3 [A] Instance: IMDSv2 required, hop limit 1, encrypted volumes, data volume
  `prevent_destroy`, no key pair, recover action on the system status check. *tftest.*
- A4 [A] RDS: force SSL parameter, encryption, 7-day backups, deletion protection
  variable, final snapshot, managed master password (no password attribute anywhere in
  the configuration). *tftest + a grep test for `password =`.*
- A5 [A] Secrets: every secret of §5.2 exists, none has a `aws_secretsmanager_secret_version`
  in Terraform, each resource policy names only its reader role. *tftest.*
- A6 [A] IAM: instance role policy matches §5.2 exactly (actions and resources, the
  CloudWatch namespace condition); OIDC role trusts only this repo's `main` and
  `workflow_dispatch`; the serving reaper role has `s3:ListBucket` and `s3:GetObject` on
  `serving/leases/*` only (P1). *tftest on the `jsonencode` locals.*
- A7 [A] Serving reaper: serving rule on, lease bucket set, dry run on, schedule
  disabled while `serving_reaper_enabled = false`, enabled when true. *tftest.*
- A8 [A] Alarms: each §5.3 alarm exists with its metric, threshold, period and missing-data
  treatment; P1 alarms' `actions_enabled` follows `serving_reaper_enabled`; all alarm on
  the SNS topic. *tftest.*
- A9 [A] Metric names agree: every metric an alarm in `alarms.tf` reads is in the
  CloudWatch agent allowlist (`deploy/cloudwatch-agent.json`) and exported by the service
  that owns it (a test scrapes each service's `/metrics` in-process and finds the name
  and labels); the counter math (cumulative vs delta, §5.3) matches the agent config.
  *pytest.*
- A10 [A] DNS/SES: records of §5.2 when `route53_zone_id` is set, the same list as an
  output when not; CAA present. *tftest.*
- A11 [A] `tflint` and `trivy config` run in CI; every finding fixed or listed with a
  reason in `infra/aws/platform/README.md`. *CI job.*

Deploy

- A12 [A] `loom-deploy` uses only the standard library (import scan); `plan_steps` gives
  §6.6's order for every combination of changed images, and refuses a poolmgr newer than
  the gateway, `--only poolmgr` included (P4). *pytest.*
- A13 [A] Secret rendering: files `0400`, owned by the service's uid, on tmpfs; no secret
  in any argv, env file outside `/run/loom`, compose file or log line (captured). `secrets
  init` never overwrites a set secret; the SES SMTP derivation matches AWS's published
  test vector. *pytest with a fake `aws` and `docker` on PATH.*
- A14 [A] `deploy/compose.prod.yml`: no `LOOM_ENV`; only Caddy publishes public ports;
  images by digest; gateway `stop_grace_period` 30 s; `poolmgr` depends on `gateway`
  healthy; volumes on `/var/lib/loom`; no dev service. *pytest on the parsed file.*
- A15 [A] The production stack starts in production mode locally: `docker compose -p
  loom-m3prod -f deploy/compose.prod.yml` (a test override with local images, a local
  Postgres with TLS standing in for RDS, a throwaway CA) comes up healthy with every
  service's production checks on; `/internal`, `/metrics` and anything outside the routes
  of §6.7 answer 404 through Caddy (Caddy on an internal CA); M1 acceptance items 0, b, d
  pass against it. *Acceptance-marked pytest in
  `services/devstack/tests/test_devstack_m3_prodstack.py`, run by its own CI job
  `m3-prod-stack` (the existing `acceptance` job is unchanged).*

Stripe

- A16 [A] T1, T2, T3 (both orders and concurrent), T5, T6, T16, T18 (the documented
  limitation: the later refund takes nothing) against real Postgres, with
  barriers holding transactions open where the trace needs it. *pytest.*
- A17 [A] T4 sequential and the concurrent variant (R applies it at its next run). *pytest.*
- A18 [A] T7 (a)-(c). *pytest.*
- A19 [A] T8 and T13: Postgres unreachable, pool timeout, role gate pending -> 503 with
  nothing stored (the second transaction cannot connect either); then the retry applies
  once; `retry_unapplied` ends its run at its first statement and does not set its
  last-success gauge. A lock timeout on a lock other than the event's row (the session's
  advisory lock, a `stripe_payments` row; one wait, and several in one delivery) -> 500
  with the event stored with `last_error`, then applied by R once the lock is free; on the
  event's own row while another transaction holds it -> 503.
  Billing's conninfo carries the keepalives and `tcp_user_timeout` of §8.3. *pytest
  (faults injected).*
- A20 [A] T9 (a)-(c): a failure injected before COMMIT, between COMMIT and the response,
  and a lost COMMIT result (the second transaction's guard answers `duplicate` when the
  COMMIT landed and stores the event when it did not); one ledger transaction in each case
  after the redelivery. *pytest.*
- A21 [A] Migration 0012 applies on a fresh database and one at 0011; a second purchase
  with the same `external_ref` raises `unique_violation` on
  `ledger_transactions_purchase_external_ref`, which `post_transaction` raises as
  `IdempotencyConflict` (`ledger.py` unchanged); the T3 race past the look-up (the test
  holds an uncommitted purchase of the same session that took no advisory lock, then
  commits it) rolls the delivery back, stores it with `last_error`, answers 500, and the
  retry answers `already_credited`. *pytest (platform_common and billing).*
- A22 [A] Response codes and metric results of §8.3 for each row (`bad_signature`;
  `signature_mismatch` for a delivery signed with another secret and a current `t`, T25;
  `stale_timestamp`; 500 with the event stored and 503 with nothing stored, both counted in
  S2c); a failure at COMMIT (a deferred check raised by a scratch-database trigger) takes
  the same 500 path; the credit refresh delays the answer by at most its 2 s bound (Redis
  frozen, T14: 200 within 3 s). Missed delivery (§8.5): a delivery the stub fails
  (`fail_status` 503), then resent with a fresh signature, is applied once; a second
  resend answers `duplicate`. *pytest.*
- A23 [A] Poison event (T17, T19): the webhook stores it unapplied with `last_error` and
  answers 500; the second transaction never overwrites a processed row (`duplicate`, 200)
  and answers 503 when it fails itself; `retry_unapplied` continues past it and re-runs it
  every minute; S4 counted once (on the write that makes `error_count` 1), not on repeats,
  and again after a clean attempt cleared it; a poison clawback pending on a purchase does
  not stop the purchase's credit (savepoint) and counts S4 `pending_clawback` once; a lost
  connection inside that savepoint fails the whole delivery (stored by the second
  transaction, or 503); a payload with `\u0000` answers 503 and stores nothing (the
  limitation of §8.10). *pytest.*
- A24 [A] Staleness (§8.6): an event unapplied for 15 min increments the stale counter
  once (cases: a `checkout.session.completed` for an unknown org; a clawback with no
  payment; a poison clawback whose payment **has** a `stripe_payments` row); a second stuck
  event increments it again; the stale gauge counts unapplied events with
  `stale_reported_at`; a T4 event applied within the minute never counts, also behind a
  backlog of 600 unapplied rows (T21, newest first); a newly stored purchase behind that
  backlog is re-run at the next run (T22); an unapplied event older than 30 days is still
  re-run; any error inside a pending-clawback savepoint that leaves the transaction usable
  leaves the clawback for R and the purchase commits; the retry job's last-success gauge moves only on completed runs (S5).
  *pytest with a fake clock.*

- A31 [A] `stripe-event resolve` and `list` (through `resolve_event` and `list_events`):
  refuses a non-admin, an unknown and a processed event, a clawback whose PaymentIntent is
  not a non-empty string, and one whose PaymentIntent has a `stripe_payments` row
  (`ResolveRefused`, nothing changed); otherwise sets
  `processed_at`, `result` and one `admin_audit_log` row in one transaction and posts
  nothing (a credit event: no purchase; a clawback with no payment: no refund); `list`
  shows unapplied events with `last_error`. T24: a purchase credited after its clawback was
  resolved leaves it alone; the next R run's sweep reopens it (`stale_reported_at`
  cleared) and applies it, or leaves it unapplied with `last_error` when that fails (then
  counted stale again); the sweep skips a row another transaction holds and takes it on
  the next run. Races, with barriers: a resolve against an in-flight purchase of the same
  PaymentIntent, in both orders (the clawback is applied once, by the purchase or by the
  sweep), and against `retry_unapplied` holding the event row (the resolve waits, then
  refuses a processed event or resolves one still unapplied); a resolved credit event
  sharing the PaymentIntent is never reopened. *pytest.*
- A32 [A] T23: two (and three) pending clawbacks are all applied in the purchase's
  commit, each in a plain savepoint, with no `SET CONSTRAINTS` in billing; the protected M1
  test `test_refund_and_dispute_before_the_purchase_are_applied_with_it` stays green; a
  savepoint that raises is rolled back alone and the others commit. The pool timeout and
  the two statement-level timeouts are settings, and the timing tests (A19, A20, A22) pass
  the flake-check recipe (20 runs under CPU load within its 7 minutes). *pytest.*
- A33 [A] Remaining §8 points: `retry_unapplied` never waits on a delivery's **event row**
  (T10: `SKIP LOCKED`; it may still wait up to `lock_timeout` on a session or
  `stripe_payments` lock inside the handler) and skips a held row; it re-runs newest first
  up to `limit`; it ends its run when an attempt's error store fails (`StoreFailed`),
  without stale marking or the S5 gauge; the second transaction's guard answers
  `duplicate`. *pytest.*
- A34 [A] §8.8: `expires_at` = now + 1 h and Adaptive Pricing off in the session request;
  the product name is `<name> credits` from `LOOM_BRAND_NAME`, `API credits` when unset;
  the worker without Stripe keys and without a name starts (the protected worker test
  passes); outcome metrics are counted after COMMIT. *pytest.*

Pool manager, gateway, platform

- A25 [A] Mock backend outside dev (§6.8): `provider: local` with the flag starts with
  the CA and client certificate and without SSH key, bucket or bundle; refused without the
  flag, with `runpod`, or with a public `ALLOW_CIDRS`; gauge 1. *pytest.*
- A26 [A] Gauges: `ca_expiry_seconds` matches the CA's `notAfter`; `pool_reason` follows
  the pool hash's reason each tick; counters initialised at 0. *pytest.*
- A27 [A] Gateway client-certificate rotation (§7): issued every 24 h and at start under
  6 days; atomic file; the gateway picks up a new file and a new CA bundle for new
  connections within 60 s while an open stream finishes on the old one; a broken file
  keeps the old context; `rekey` re-issues every ready pod; `new-ca` writes a valid CA
  outside dev. *pytest with local pods.*
- A28 [A] `terraform plan` against the real account with read-only credentials and a
  local backend: no errors, the resource count and names as `README.md` lists, no
  replacement of anything outside the stack. *Run by the user (or with their yes);
  blocked on AWS access (§3).*
- A29 [A] `loom-deploy` refuses a database URL without `sslmode=verify-full` (D10) and
  an apply whose site parameter lacks the domain or the product name (§6.11); every service and `loom-migrate` connect with
  `verify-full` to a TLS Postgres (A15); the services' `production_problems` output is
  unchanged (the protected tests of §16 pass). *pytest.*
- A30 [A] Docs of §12 written; the runbooks reviewed against the code (every command and
  variable exists). *Review.*

Product name and domain

- A35 [A] Settings (§6.11): the Terraform variables `domain` and `brand_name` have no
  defaults and fill the SSM parameter `/loom/platform/site`, which the instance role alone
  may read; no domain literal in `infra/aws/platform/*.tf` or `deploy/**` (the Caddyfile
  uses `{$LOOM_DOMAIN}`; tests use placeholders such as `example.test`); `loom-deploy`
  renders the variables of §6.11's table from the parameter (fake `aws`); A15's stack runs
  the web image built **without** `LOOM_BRAND_NAME` and, with it set, serves the sign-in
  page with the name in its `<title>` and heading. *tftest + pytest.*
- A36 [A] No product name in user-facing text: a scan of the string literals and JSX text in
  `apps/web/src` (tests excluded), `stripe_api.py` and `loom_platform/errors.py` finds no
  `Loom` other than in `X-Loom-*` header names (identifiers such as `LoomEnv`, and
  comments, are not user-facing); with `LOOM_BRAND_NAME` set the root layout's title template, the
  sign-in heading and the app headers show it, and unset they show no name; `auth.config.ts`
  has no default sender name; the existing web tests stay green and `npm run lint`,
  `typecheck`, `test` and `build` pass. *pytest scan (`test_devstack_m3_brand.py`) + new
  vitest files + the web CI steps.*

Endpoint health check

- A37 [A] H3 (§8.5): `healthcheck.tf` has an `aws_route53_health_check` of type
  `HTTPS_STR_MATCH` for `hooks.<domain>` on port 443, path `/stripe/health`, search string
  `stripe_webhook`, SNI on, interval 30 s, failure threshold 3, with and without
  `route53_zone_id`; alarm H3 reads `AWS/Route53` `HealthCheckStatus` with dimension
  `HealthCheckId` = that check's id (Minimum < 1, 5 x 60 s, missing data breaching), is
  created through the `aws.us_east_1` provider and notifies `loom-platform-alerts` on ALARM
  and OK; the plan fails when the stack's region is not `us-east-1`. Billing answers `GET
  /stripe/health` with 200 and `stripe_webhook` while Postgres and Redis are unreachable;
  through A15's Caddy, `hooks.` serves that route and `POST /stripe/webhook` and answers
  404 for anything else. *tftest + pytest.*

**Stage B (money; only after the user's yes and cap, §14)**

- B1 [B] `infra/aws/state` applied; `backend.hcl` written; `terraform init` migrates
  nothing (fresh). *Runbook step.*
- B2 [B] `terraform apply` of `infra/aws/platform`; outputs recorded; budget and SNS
  subscription confirmed by the user. *Runbook step; `terraform plan` afterwards shows no
  changes.*
- B3 [B] DNS records resolve (apex, `api`, `hooks`); SES identity verified (DKIM). *Runbook
  check (`dig`, SES console).*
- B4 [B] Secrets: `loom-deploy secrets init`; the user puts Stripe, Google, SES SMTP values;
  every secret has a value (`describe-secret` shows a version), none readable by the
  user's principal. *Runbook check.*
- B5 [B] Images built by the workflow, pushed by digest; manifest in S3. *Workflow run.*
- B6 [B] First deploy: migrations as `loom_owner`, `loom_app` created, every service healthy
  in production mode; Let's Encrypt staging certificates, then production. *`loom-deploy
  status`.*
- B7 [B] Stripe sandbox: the restricted key with only Checkout Sessions write creates a
  session with `expires_at` and Adaptive Pricing off; Stripe's test-mode retry schedule and
  how long an event can still be resent (dashboard and `stripe events resend`) are recorded
  in the runbook (§8.5, §8.10). *Runbook check.*
- B10-B19 [B] `loom-m3-acceptance` items m3-0, m3-b, m3-d, m3-g, m3-s, m3-h, m3-i, m3-p,
  m3-x pass; m3-a done by hand. *The script's JSON report + the acceptance doc.*
- B20 [B] `docs/runbook-pod-ca.md` executed once on the box (mock pods): rollover with no
  failed request in a concurrent load run. *Runbook + load test numbers.*
- B21 [B] (optional, D8) RDS point-in-time restore to a new instance, `loom_platform`
  readable, instance deleted. *Runbook.*
- B22 [B] Teardown or keep (D8): if teardown, `terraform destroy` leaves only the final
  RDS snapshot, the state bucket and data-volume snapshots, listed. *Runbook + a
  `resourcegroupstaggingapi` listing by the `loom:stack` tag.*
- B23 [B] Webhook secret roll (T15) done once: no delivery fails. *Runbook.*
- B24 [B] By hand: a magic-link sign-in to a real inbox (SES; a verified address while in
  the sandbox; the mail's sender shows the product name and an address on the domain, its
  link the domain) and a Google sign-in, on the public URL. *Recorded in the acceptance
  doc.*
- B25 [B] H3 end to end: with the stack up, stop Caddy on the box: H3's ALARM email
  arrives (minutes recorded); start Caddy: the OK email arrives. *Runbook check.*

## 14. Costs and the Stage B cap

Monthly at us-east-1 list prices (check at apply): EC2 t4g.medium $24.53, 30 + 20 GB gp3
$4.00, public IPv4 $3.65, RDS db.t4g.micro $11.68 + 20 GB $2.30, Secrets Manager 12
secrets $4.80, CloudWatch ~15 custom metrics $4.50 + ~20 alarms $2.00 + logs ~$1, Route 53
zone $0.50 + the `hooks.` health check with HTTPS and string matching $2.50 (AWS-endpoint
prices; $4.75 if Route 53 bills it as a non-AWS endpoint) + its alarm H3 $0.10, ECR ~4 GB
$0.40, S3/DLM snapshots ~$1, SES ~$0. **About $63 a month**, ~$0.087 an hour. The domain registration is the user's (~$10-15 a year).

The plan (§15): a **$75 a month** AWS budget (D12) and a Stage B cap of **$25** for a
14-day window (apply, acceptance, fixes, teardown or the keep decision), about $15
expected. **Stage B still needs the user's separate yes** with that cap before anything is
applied. Keeping the stack running after the window is the D8 decision (~$63 a month until
M4).

## 15. Decisions (decided by the user, 2026-11-18)

The user accepted every recommendation below, with two changes: D15 (no automated sync in
Phase 1, §8.12) and D16 part 5 (the product name and the domain are not decided; each is a
setting, §6.11). The $75 monthly budget (D12) and the Stage B cap of $25 over 14 days (§14)
are the plan; Stage B still needs the user's separate yes.

| # | Decision | Decided |
|---|---|---|
| D1 | Where images are built | CI on GitHub's arm64 runner, pushed to ECR by digest (reproducible, scanned, the box stays small). Alternative: build on the box (no ECR/OIDC, but slow and memory-tight on 4 GiB) |
| D2 | Per-IP request limit (contracts §8 step 1 said ALB + WAF) | Caddy with the pinned rate-limit module, no extra cost. Alternatives: ALB + WAF (~$25 a month more), or none during invite-only M3 |
| D3 | Instance architecture | Keep `t4g.medium` (arm64), as plan §3. Alternative `t3.medium` x86 (+$6 a month) if an arm64 build problem appears |
| D4 | Stripe account | A Stripe sandbox used only by Loom's deployed box (no laptop or other stack on it): no foreign events, exact reconciliation. Foreign-event handling is not built (§8.1, §8.10) |
| D5 | Disputes won | Admin adjustment in M3 (test mode); automatic reinstatement later |
| D6 | Acceptance sign-in | DB-minted sessions for the automated items, plus one manual magic link and one manual Google sign-in (B24). Alternative: SES inbound to S3 on a subdomain so the script reads the magic link (more infra) |
| D7 | Host names | Apex for the web app, `api.` for the API, `hooks.` for Stripe |
| D8 | After the acceptance | Keep running if M4 starts within ~3 weeks, else tear down (B22) and re-apply for M4 (~1 h of work, Let's Encrypt certificates re-issued) |
| D9 | Mock backend outside dev | The explicit `LOOM_POOLMGR_MOCK_BACKEND` flag (§6.8), not `LOOM_ENV=dev` on the box |
| D10 | `sslmode=verify-full` required on the box | Yes, enforced by `loom-deploy` (not by the services' start-up checks, which protected tests pin) |
| D11 | Gateway client-certificate rotation in M3 | Yes (free now; needed before real pods) |
| D12 | AWS budget amount | $75 a month, emails at 50/80/100% |
| D13 | Grafana access | SSM port forwarding only; no public route |
| D14 | Separate data volume | Yes (Redis AOF, settle spool and Caddy certificates survive an instance replacement) |
| D15 | Stripe sync automated | **No**, for Phase 1 (changed from the recommendation): alarms and a runbook resend (§8.5; comparison §8.12). An Events-API sync (or an equivalent) is required before live mode (§8.10) |
| D16 | Open questions 4-6 of plan §15 | 4: approve the AWS control plane as in §14 at Stage B (with the Stage B yes); 5: **replaced**: the product name and the domain are not decided; each is one setting (§6.11); the user registers the domain before Stage B, and nothing in Stage A needs it; 6: SES in the same account, production access requested as soon as the domain is verified |

## 16. Plan (agents)

### Size

**Large.** Five areas (Terraform, deploy tooling, Stripe, pod PKI and the manager, the
acceptance), a new language in the repo (HCL), security-critical surfaces (IAM, secrets,
the public endpoint, the money path), a contracts change, and a paid stage at the end.
About 60-80 files. Per the multi-agent-setup skill's Large row: spec approved first,
tests written and reviewed before the lock, one worktree per area with disjoint file
ownership, a fresh reviewer per change, a verifier on a fixed commit, the user at each
checkpoint.

### Waves and file ownership

| Wave | Who (model) | Branch / worktree | Owns |
|---|---|---|---|
| 0 (sequential) | Orchestrator | `phase1-m3` (from `main`), main checkout | This spec; `docs/phase1-contracts.md` (§11 changes); the CI job for Terraform and the `terraform` pin in `.github/workflows/ci.yml` (protected: done while unlocked, before the lock, user approves the diff) |
| 1 tests | 4 test writers (Opus), one per area, + 1 holdout writer (Opus) | in each worktree below | Tests only (§13 A-items): `infra/aws/platform/tests/*.tftest.hcl`; `services/devstack/tests/test_devstack_m3_*.py` (deploy, compose, metric-name cross-check A9); `services/billing/tests/test_billing_m3s_*.py`, `services/platform_common/tests/test_platform_m3s_*.py` (migration 0012, the L3 index through `post_transaction`, the grants test) and the stub (`stripe_stub.py`); `services/poolmgr/tests/test_poolmgr_m3_*.py`, `services/gateway/tests/test_gateway_m3_cert_*.py`; `loom_devstack/m3_acceptance.py`, `services/devstack/tests/test_devstack_m3_brand.py` and new `apps/web/src/**/*.test.ts` files (A35, A36) (by the deploy-area writer). Holdout: the Stripe traces T1-T10 a second time, independently, for `.holdout` |
| 1 review | 4 fresh test reviewers (Opus) | | One per area; loop until approved |
| Lock | **User** (`/agent-setup lock`, typed in the terminal) | | After the worktrees exist on their branches |
| 2 build | 4 implementers (Sonnet) in parallel | `.claude/worktrees/m3-cp-infra` (`phase1-m3-infra`): `infra/aws/**` (bench untouched; incl. `domain`/`brand_name`, the site parameter and its IAM read, and `healthcheck.tf` with the `aws.us_east_1` alias, H3), `infra/aws/platform/README.md`. `m3-cp-deploy` (`phase1-m3-deploy`): `deploy/**` (incl. the Caddyfile's `GET /stripe/health` route on `hooks.`), `services/devstack/src/loom_devstack/deploy.py` and its entry point (incl. D10's `sslmode` and the domain/name checks, §6.11), `.github/workflows/images.yml`, `docs/runbook-platform.md`, `docs/security.md`, and the product-name text of §6.11: `apps/web/src/**` except tests, `apps/web/README.md`, `services/platform_common/src/loom_platform/errors.py` (it reads `apps/web/AGENTS.md` and the Next.js 16 docs first; `apps/web/package.json` is protected, so no new dependency). `m3-cp-stripe` (`phase1-m3-stripe`): `services/billing/src/**` except the stub (incl. `stripe_api.py`'s product name, `LOOM_BRAND_NAME` in `settings.py`, and the `GET /stripe/health` route in `api.py`), migration `0012_*.sql`, `services/billing/README.md`, `docs/runbook-stripe.md` (missed deliveries and resend, resolve cases, secret roll; the platform runbook links it). `m3-cp-pki` (`phase1-m3-pki`): `services/poolmgr/src/**`, `services/gateway/src/loom_gateway/{routing,settings,metrics}.py`, `docs/runbook-pod-ca.md`, `services/poolmgr/README.md` | Disjoint; contracts edits after wave 0 go through the orchestrator |
| 2 review | Fresh reviewer per change (Opus); Stripe design-level findings compared with the Fable review (below) | | |
| 2 verify | Verifier (Opus) on a fixed commit per branch; test-strength check (mutation) on `webhooks.py`, `resolve.py`, `api.py`'s webhook path, `deploy.plan_steps`, the manager's `production_problems`; hidden tests run on the Stripe branch | | |
| 3 integrate | Orchestrator merges into `phase1-m3` in this order, each after the user's yes: stripe, pki, deploy, infra (A9's cross-check passes only with all four); whole-branch review (Opus), verifier, CI (flake-check on the new test files first, per the retros), PR to `main` | | |
| Stage B | Only after the user's yes and cap: the user (or an agent with that yes) runs B1-B24 with the runbook | | |

Cross-area settings: the domain and the product name (§6.11). The infra worker writes the
site parameter, the deploy worker reads it in `loom-deploy` and renders the variables, the
Stripe worker reads `LOOM_BRAND_NAME` in billing and the deploy worker in the web app. Both
are optional for the services (the root `docker-compose.yml` needs no change); A15 starts
the production stack with them.

**Test isolation for the Stripe area.** `loom_platform.testing.pg_url` is one database per
pytest worker, shared by every test and never cleaned, and protected tests leave unapplied
Stripe rows behind; stale marking and the 500 limit are global. So tests of A23, A24, A31
and A33 that run the job with a moved `now` use a **function-scoped** scratch database cloned from a migrated template
(`CREATE DATABASE ... TEMPLATE`, the template migrated once per session, so each clone
costs no migrations; the pattern of
`services/platform_common/tests/test_platform_migration_0010.py`; a module-scoped one is
shared by pytest-repeat's copies) or reset the Stripe tables at test start, or assert only on their own rows, and stay correct under
flake-check's `--count 2`. R has no age window and re-runs every unapplied row newest
first, so in the shared database it also re-runs other tests' leftovers; a test that needs
its own rows inside R's 500 uses the scratch database. New M3 tests use random dispute, session and PaymentIntent ids
(never fixed ones such as the protected `dp_test_x`), so their second copy in one session
database does not meet the first copy's events. Fake-clock and backlog tests (a moved `now`,
the 600-row backlogs of A24, T21 and T22) leave no rows in the shared session database
(round 18 n5), so they run in the scratch database: 500 newer unapplied rows left behind
would push the row of the protected `test_retry_job_applies_a_clawback_that_raced_its_purchase`
out of R's 500, and rows stale-marked under a moved `now` would count in other tests'
stale gauge.

**Constraints from protected tests** (M3 changes none of them; they bind the tests and
the implementation):
- `services/billing/tests/test_billing_stripe_sig.py` (~235-248) and
  `services/platform_common/tests/test_platform_settings_db.py` (~29-46) pin the exact
  `production_problems()` output of billing and the platform settings: no new start-up
  refusal may be added there (hence D10 and the domain/name checks live in `loom-deploy`).
- `services/platform_common/tests/test_platform_m3_review2_worker_up.py` starts the billing
  worker with `LOOM_ENV=prod`, a plaintext database URL and no Stripe keys, and requires it
  to stay up: M3's worker calls no Stripe API (no sync, no look-ups), so nothing in it needs
  the keys or the product name.
- The web tests (`apps/web/src/**/*.test.ts`, `apps/web/tests/**`), `apps/web/package.json`
  and `apps/web/vitest.config.ts` are protected: the name change adds new test files and
  keeps the existing ones green. `playground.test.ts` pins the snippet's `LOOM_API_KEY`
  variable name, which stays (an internal `LOOM_` name, like the `loom_sk_` key prefix).
- `services/billing/tests/test_billing_webhook.py` and `test_billing_stub_e2e.py` deliver
  over HTTP, call `retry_unapplied(pool)` and read Redis right after a delivery: the M1 signatures and the awaited credit refresh (§8.3, at most 2 s) stay.
  `test_refund_and_dispute_before_the_purchase_are_applied_with_it` (~470) needs both
  pending clawbacks applied in the purchase's commit, which plain savepoints without `SET
  CONSTRAINTS` give (round-18 probe); `test_retry_job_applies_a_clawback_that_raced_its_purchase`
  (~495) needs R to apply a raced clawback and leave an unknown one unapplied, which the
  newest-first order keeps.
- `services/platform_common/tests/test_platform_ledger.py` (~36-38) posts a purchase with
  the fixed `external_ref="cs_test_1"`: fine once per session database under the 0012
  index, but it would fail if repeated in one session (`--count`); it is not in
  flake-check today. Noted only.

Worktrees are created by the orchestrator with `git worktree add
.claude/worktrees/m3-cp-<area> -b phase1-m3-<area> phase1-m3` **before** the lock (a
locked `git switch` is refused in a worktree, retros 2026-11-17); agents run without
isolation, cwd = their worktree, Linux commands through `run_in_wsl`. On agent-setup
0.6.1 the lock is local, not committed. Redis DBs: one per worker (11-14). The old `m3-*`
worktrees from the before-M3 fixes are not reused.

### What the user must set up before the lock

- `protect.paths` additions in `.claude/agent-setup.json` (agents cannot edit it):
  `services/billing/src/loom_billing/stripe_stub.py` (a test double the test writer
  extends), `services/devstack/src/loom_devstack/m3_acceptance.py` (the acceptance is a
  test), and, explicitly, `infra/aws/platform/tests/**` (already covered by
  `**/tests/**`; listed so it is not missed).
- `pytest.ini`: no change needed (all new Python tests are in existing test paths).
- Hidden tests: not configured yet; the user runs `/agent-setup holdout <command>` (for
  example the billing test command over `.holdout/`) so the holdout writer's Stripe trace
  tests stay out of the implementers' view.
- Terraform for agents: the pinned `hashicorp/terraform` image (pulled once; free).

### Models

Test writers and reviewers Opus; implementers Sonnet (no reason found for Opus: the
concurrency design is reviewed before tests, retros 2026-11-17); verifiers Opus; **every
Stripe design review on Fable** (Appendix A). Each M3 code review's design-level findings
in the Stripe area are compared with the Fable review; any it missed goes to the retro's
"Design flaws code review later found that this review missed" list.

### User checkpoints

1. Approve this spec and the decisions of §15 (done 2026-11-18, with the changes of §15).
2. Approve the wave-0 contracts and CI diff; add the `protect.paths` entries; configure the
   holdout command.
3. `/agent-setup lock` after the worktrees exist.
4. Each merge into `phase1-m3`, then the PR to `main` (`/agent-setup unlock` first).
5. Read-only AWS credentials for A28 (`terraform plan`).
6. Before Stage B: register the domain and choose the product name (the two Terraform
   variables, §6.11).
7. **Stage B yes with the cap** (§14), and later the keep-or-teardown decision (D8).

## Appendix A. Stripe design review rounds (Fable)

Each round is a fresh `agent-setup:reviewer` on model Fable, given only this spec and the
code and contract paths (not the author's reasoning).

Rounds 1-16 reviewed the design before the user's cut of 2026-11-18 (§15 D4, D15): their
item, trace, alarm and section numbers (A21-A36, T10-T31, S2a-S5, §8.5-§8.6) refer to that
earlier text, which had an Events-API sync, foreign-event look-ups, `park` and a `resolve`
that posted refunds. Round 17 reviewed the reduced design. Round 18 reviewed it again
after the user took round 17's list of removable mechanisms (the revision is summarised
under round 18).

**Round 1** (no blockers, 3 majors, 9 minors; ~107k tokens, 24 tool uses; the reviewer
confirmed three Postgres behaviours with a scratch-table probe: an `ON CONFLICT` insert
waits on an uncommitted duplicate and obeys `lock_timeout`; `lock_timeout` applies to
`pg_advisory_xact_lock`; a second partial unique index raises through `ON CONFLICT
(idempotency_key) DO NOTHING`).

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 1.1 | A per-event `handle_event` failure inside the sync was undefined: a poison event either blocks every newer missed event or is skipped and lost with no alarm; webhook 5xx answers counted nowhere | major | reasoning over the spec and `webhooks.py` (any deterministic exception) | §8.3 last row: roll back, store unapplied with `last_error` (raw body if `jsonb` refuses) in a second transaction, 500; §8.5 step 3-4: continue, stored events owned by `retry_unapplied`; `retry_unapplied` continues past a failing event; alarms S2c, S4; T18; A31 |
| 1.2 | Foreign-event look-up after 1 h equals the unapplied alarm's 60 min: an ALARM/OK email pair per foreign refund | major | reasoning (timings from §5.3, §8.6, `worker.py`) | Look-up after 10 min; staleness at 15 min (S1a); A23 pins the order |
| 1.3 | The unapplied gauge has no age limit, so one permanently unapplied event holds the alarm in ALARM forever and hides the next; no admin tool to resolve | major | gauge query confirmed in `webhooks.py`; alarm behaviour reasoning | S1a counter per newly stuck event, S1b gauge reminder; `loom-billing stripe-event list/resolve` (audited, posts nothing); A32, A33 |
| 1.4 | Sync counted events still being delivered (just created, or awaiting Stripe's retry) and `duplicate` race losers as "recovered" | minor | reasoning | 15-min grace (`created[lt]`); `duplicate` not counted; T19 |
| 1.5 | S2 named `loom_billing_webhooks_total`; the code's metric is `loom_billing_stripe_webhooks_total` | minor | confirmed (`metrics.py`) | Name fixed; A9 cross-checks names |
| 1.6 | Junk POSTs without a header count as `bad_signature` and could hold S2 in ALARM | minor | confirmed (`api.py`) | `stale_timestamp` split (HMAC valid, time out); S2b reads only it; recovered is S2a |
| 1.7 | "503 within ~6 s" not guaranteed: per-statement timeouts add up over five lock waits (15-20 s) | minor | `lock_timeout` on advisory locks confirmed by probe | One overall 6 s deadline (`asyncio.timeout`); A19 |
| 1.8 | Foreign look-up made while holding the event's row lock (20 s Stripe timeout) | minor | confirmed (`webhooks.py` `retry_unapplied`) | Look-up outside any transaction, then lock `SKIP LOCKED`, re-check, mark |
| 1.9 | 20 look-ups a run, oldest first: events that stay unapplied block later ones | minor | reasoning | `lookup_at`/`lookup_result`, never-looked-up first, back-off |
| 1.10 | Foreign clawback with a null PaymentIntent wedged unapplied | minor | confirmed (`webhooks.py` `_claw_back`) | `ignored_foreign` at once |
| 1.11 | Sync paging: Stripe lists newest first, default `limit` 10; at the cap the oldest drop every run; S3 fires after every restart | minor | reasoning; Stripe behaviour high confidence | `limit=100`; a capped run is not a success; sync runs at worker start |
| 1.12 | Dispute won (admin re-credit), then refunded: `clawed_back_micros` still full, the refund takes nothing | minor | confirmed (`webhooks.py`) | Documented (§8.10, T20); runbook step for the admin |

**Round 2** (no blockers, 2 majors, 9 minors; ~109k tokens, 17 tool uses; probe in the local
test Postgres: `jsonb` refuses `\u0000` (`UntranslatableCharacter`); a statement cancelled by
a 1-s `asyncio` timeout raises after ~1 s and leaves the connection idle; a second purchase
with the same `external_ref` raises `23505` through `ON CONFLICT (idempotency_key)`).

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 2.1 | A poison clawback stored before its purchase wedges the purchase: `_apply_pending_clawbacks` re-runs it inside the purchase's transaction with no savepoint, so every delivery of the purchase rolls back | major | code reading (`webhooks.py` 208-221, no try or savepoint); confirmed by the orchestrator against the same lines | Each pending clawback in a savepoint; a failing one keeps `last_error`, S4, the purchase commits (§8.3, T21, A31) |
| 2.2 | Events stored with only `id`/`type` (payload `jsonb` refused) can never be applied, and such a clawback, lacking `payment_intent`, would be dropped as foreign with no alarm | major | `jsonb` refusal confirmed by probe; rest reasoning over the spec and `webhooks.py` | Sanitise NUL and lone surrogates before every insert (original in `raw`), so every stored event is runnable; the null-PaymentIntent rule only for a `charge`/`dispute` object whose key is present with JSON null (§8.3, §8.6, A23, A31) |
| 2.3 | The 6 s deadline does not bound the answer when Postgres is unresponsive (psycopg's cancel path waits up to ~10 s more; no keepalives configured) | minor | psycopg source read; responsive case probed | Postgres work in a child task, 503 at 6 s while it cleans up; keepalives and `tcp_user_timeout`; A19 with a blackholed connection |
| 2.4 | S4 stays in ALARM for up to 30 days per poison event (counted each minute), hiding the next | minor | reasoning | S4 counts first failure and changed error text; re-runs back off; back-off resets on deploy |
| 2.5 | A foreign event goes stale after one transient look-up error (1-h back-off vs 15-min staleness) | minor | reasoning | 5 quick attempts; not stale while a look-up is pending |
| 2.6 | No state for the doubling back-off | minor | reasoning | `lookup_attempts` column |
| 2.7 | Resolve procedure (adjust first, then resolve) has a double-credit window | minor | reasoning | Resolve first (the guard), then adjust naming the event; runbook notes hand-resolved purchases have no `stripe_payments` row |
| 2.8 | Second transaction under-specified (own failure, timeouts, overwriting a processed row); 503 vs 500 classified by an enumerated list | minor | reasoning | §8.3 "Second transaction" (`ON CONFLICT ... WHERE processed_at IS NULL`, 503 if it fails); 503 by exception class and SQLSTATE |
| 2.9 | Sync window anchored on `now`: a stop longer than 72 h skips events; the first run picks up older events (other deployments on a shared account raise S1a) | minor | reasoning | `stripe_sync_state.synced_through`; first run starts at now - grace; `metadata.loom_deployment` and T22 |
| 2.10 | S2b (> 3 in 15 min) can never fire at test-mode volume; S1a's unbounded `type` label gives first-appearance series no increase | minor | reasoning | S2b >= 1; `type` mapped to the five types or `other`, initialised |
| 2.11 | 20 look-ups x 20 s timeout can delay the minute's re-run pass | minor | confirmed (`stripe_api.py` `TIMEOUT_S`) | Re-run and stale marking first, look-ups last, 5-s timeout, 30-s budget |
| 2.12 | (Unconfirmed, Stripe) a restricted key with Events read may not list `checkout.session.*`/`charge.*` events | note | not checkable offline | Stage B check B7; fallback permissions in §8.5 |

**Round 3** (no blockers, 2 majors, 13 minors; ~131k tokens, 20 tool uses; probe in a
scratch schema: an `ON CONFLICT` insert behind an uncommitted duplicate fails with `55P03`
under `lock_timeout`; the 0012 index raises `unique_violation` through `ON CONFLICT
(idempotency_key)` while a `refund` with the same `external_ref` is still allowed; the
second transaction's `DO UPDATE ... WHERE processed_at IS NULL` waited, then left a
processed row untouched). Both majors were in mechanisms added in round 2.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 3.1 | A changed or mistyped `LOOM_DEPLOYMENT_ID` makes a paid checkout `ignored_foreign`: final, no alarm, credit lost | major | reasoning (`_credit_checkout` has no such check, so the spec's rule decides) | The org decides: `ignored_foreign` only when the org is unknown here **and** the id names another deployment; a known org is credited, mismatch counted (§8.6, T23, A23) |
| 3.2 | The "never reaches back before this deployment" rule held only for the first run: the second run's formula lists 72 h of older events | major | arithmetic from the spec's formula | Persisted `floor`, `lower = max(floor, ...)`, `--since` overrides (§8.5, T24, A21) |
| 3.3 | Sync counted "recovered" for a row a webhook stored unapplied meanwhile | minor | reasoning | Count only when the sync's own INSERT created the row |
| 3.4 | `retry_unapplied` and pending-clawback savepoints treated 503-class errors as handler bugs (`last_error`, S4, back-off on healthy events) | minor | reasoning | R skips without `last_error`/S4; a savepoint re-raises 503 classes (delivery 503) |
| 3.5 | S4's "error text changed" is unstable (zero-sum trigger messages carry a fresh id); COMMIT-time errors not covered | minor | confirmed (`0003_ledger.sql`) | Failure identity = class + SQLSTATE (`last_error_kind`); non-503 COMMIT errors take the 500 path |
| 3.6 | Sanitising covered string values only; `jsonb` refuses NUL in keys too, so such a payload could never be stored | minor | reasoning (Postgres behaviour high confidence) | Keys sanitised too; a non-503 failure of the second transaction is counted and the sync continues |
| 3.7 | A deterministic 503-class failure on one event wedges the sync (oldest first, every run stops there) | minor | reasoning | After 3 runs stopped on the same id, store it unapplied and continue (`blocked_event_id`, `blocked_runs`) |
| 3.8 | Page cap is permanent on a busy account (newest first, oldest never reached); unclear whether a capped run processes what it listed | minor | reasoning | Capped runs process what they listed and walk back with `backfill_before`; runbook: raise the cap |
| 3.9 | `retry_unapplied`'s `LIMIT 500` oldest-first can starve a T4-race clawback past the staleness threshold | minor | confirmed (`webhooks.py` `retry_unapplied`) | Set-based, uncapped re-run of clawbacks whose PaymentIntent now has a purchase; set-based stale marking (T26, A32) |
| 3.10 | Late-commit after the 6 s deadline unspecified (child not cancelled; pool connection held) | minor | reasoning | Child cancelled at the deadline; a landing COMMIT is safe by L1 (T25, A20) |
| 3.11 | The L3 race answered as a handler failure (500, `last_error`, S4) | minor | probe | `23505` on the L3 index is retryable: 503, no `last_error` (A22) |
| 3.12 | Second-transaction SQL omitted `last_error_at`/`error_count`; R's update lacked the `processed_at IS NULL` guard | minor | reasoning | Both added |
| 3.13 | Body limit 64 KiB in the spec vs 256 KiB in `settings.py` | minor | confirmed | 256 KiB kept (Caddy and billing) |
| 3.14 | Clamping the window at 29 days was silent | minor | reasoning | Logged and counted (`sync_gaps_total`, S2a) |
| 3.15 | Two definitions of the stale gauge (S1b vs §8.6) | minor | reasoning | One definition: unapplied with `stale_reported_at` |

**Round 4** (no blockers, 1 major, 9 minors; ~158k tokens, 24 tool uses; probes in the local
test Postgres: the L3 index violation through `ON CONFLICT (idempotency_key)` arrives as
`UniqueViolation 23505` with `constraint_name` set, while the function's own key-conflict
raise has none; `25006` is `InternalError` and class `54` is `OperationalError` in psycopg;
`jsonb` refuses `NaN`/`Infinity` with `22P02`).

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 4.1 | The L3 `23505` reaches billing as `IdempotencyConflict` (`ledger.py` wraps every `UniqueViolation`), which §8.3 routes to the 500 path while §8.3/A22 demand 503; class + SQLSTATE cannot tell them apart; the fix lies in `ledger.py`, outside the stripe worker's files | major | probe + code (`ledger.py` 104-105); confirmed by the orchestrator | Named index `ledger_transactions_purchase_external_ref`; `post_transaction` raises `DuplicatePurchase` by `diag.constraint_name`; retryable; `ledger.py` added to §11 (contracts §3) and the stripe worker's files (§16); A22 |
| 4.2 | `retry_unapplied` step 2 oldest-first starves a newly stored purchase behind 500 older due rows (borderline major) | minor | today's ordering confirmed (`webhooks.py` 248-255) | `last_attempt_at`; never-attempted and credit types first (T27, A32) |
| 4.3 | Hand-resolved partial refund plus manual adjustment is invisible to L4: a later cumulative refund takes the whole purchase again | minor | reasoning over `_claw_back` | `resolve --clawed-back-micros` advances `stripe_payments` (T28, A33) |
| 4.4 | No alarm when `retry_unapplied` itself fails every minute (then nothing is marked stale) | minor | confirmed (`worker.py`: job errors counted, no alarm) | `loom_billing_stripe_retry_last_success_seconds`, alarm S5 (breaching on missing data) |
| 4.5 | Sync backfill by "oldest `created` seen" skips events sharing that second; upper bound not frozen | minor | reasoning; Stripe `created` is in seconds (high confidence) | Frozen `backfill_upper` + `starting_after` cursor `backfill_cursor` |
| 4.6 | Blocked-event rule wedges if the store itself fails; a deterministic retryable error in a pending-clawback savepoint blocks its purchase | minor | reasoning | Pass the event even if the store fails (counted, S4); retryable errors in a savepoint roll back to it and the purchase commits |
| 4.7 | `25006` takes the 500 path; class `54` would be retryable forever | minor | probe (psycopg class hierarchy) | `25006` retryable; retryable = only the listed SQLSTATEs + connection-level failures; other `OperationalError` classes non-retryable (A24) |
| 4.8 | "Every signed event can be stored" fails for `NaN`/`Infinity`/`1e999` | minor | probe (`22P02`) | Parse hooks refuse them: 400 `bad_event` (A24) |
| 4.9 | S4 wording still said "error text"; S2c (> 5 in 15 min) can never fire at test-mode volume; first deploy: events in the 15 min before the first run count as recovered | minor | reasoning | S4 wording; S2c >= 2 in 1 h; `floor` = the first run's time |
| 4.10 | PaymentIntent id from the payload put into a URL path | minor | reasoning | Validated against `^pi_[A-Za-z0-9]+$`; anything else never looked up (A33) |

**Round 5** (no blockers, 1 major, 10 minors; ~157k tokens, 21 tool uses; probe: a
`55P03` inside a nested transaction leaves the outer transaction usable with only the
savepoint's insert rolled back; a second key with the same `external_ref` raises `23505`
naming the partial index; `jsonb` refuses NUL in keys and values and a lone surrogate).

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 5.1 | A31 said a retryable error in a pending-clawback savepoint makes the delivery 503, while §8.3 and A32 say the purchase commits: no implementation could pass both (a leftover of the round-4 edit) | major | read in the three places; §8.3's behaviour confirmed implementable by probe | A31 reworded: only a connection-level failure or the deadline makes it 503 |
| 5.2 | `resolve --clawed-back-micros N` accepted any N (cap only at the purchase): an admin could take back more than was refunded | minor | reasoning | `resolve` now posts the `refund` itself with N bounded by the handler's own formula; larger N refused, maximum printed (T28, A33) |
| 5.3 | A clawback resolved before its late purchase arrives lets the purchase credit with no clawback; its staleness (15 min) came before the sync's latency (25 min) | minor | reasoning | 30-min staleness for `loom_credits` clawbacks; `resolve` refuses a no-payment clawback younger than 72 h + 25 min without `--force`; a purchase finding a resolved clawback alarms (S4 `resolved_clawback`) |
| 5.4 | Backfill cursor persistence point and lower bound unspecified; a cursor past retention errors forever | minor | reasoning | Cursor advances only after its chunk is processed; `backfill_lower` frozen too; `resource_missing` drops the backfill and counts a gap |
| 5.5 | The pending-clawback `SELECT ... FOR UPDATE` sits outside the savepoints, so a held row makes the purchase 503; a deferred check could fail only at the outer COMMIT | minor | confirmed (`webhooks.py` 211-219) | `FOR UPDATE SKIP LOCKED`; `SET CONSTRAINTS ALL IMMEDIATE` at the end of each savepoint |
| 5.6 | The L3 look-up filters on `org_id` too, unlike the index, so a same-session purchase under another org would be `DuplicatePurchase` (503) forever | minor | confirmed (`webhooks.py` 175-178); not reachable from Loom sessions | Look-up uses the index's predicate |
| 5.7 | The look-up regex rejects the stub's and tests' `pi_test_...` ids | minor | confirmed (`stripe_stub.py` 279, `test_billing_webhook.py`) | `^pi_[A-Za-z0-9_]+$` |
| 5.8 | Second-transaction SQL omitted `last_error_kind`; back-off due rule and the deploy reset unspecified | minor | reasoning | Column added; comparison before overwrite; due rule from `error_count`/`last_attempt_at`, reset by process start time |
| 5.9 | `floor` set by the first successful sync run, so an event just before it is never synced | minor | reasoning | `floor = now()` written by migration 0012 |
| 5.10 | Ownership gaps: platform_common tests, `LOOM_DEPLOYMENT_ID` in compose files, the Stripe runbook, an undefined `--dry-run` | minor | read in §16 | Test file pattern added to wave 1; `dev` default in dev, prod compose sets it; `docs/runbook-stripe.md` owned by the stripe worker; `--dry-run` defined |
| 5.11 | A hand-made refund booked as an adjustment drifts `stripe_clearing` from Stripe | minor | contracts §3 Kind table | Clawbacks resolved by hand post a `refund`; purchases credited by hand documented in §8.10 |

**Round 6** (no blockers, 1 major, 15 minors; ~180k tokens, 28 tool uses; probe in a scratch
database migrated with `loom_platform.migrate.upgrade` and dropped: after a released
savepoint that ran `SET CONSTRAINTS ALL IMMEDIATE`, the next `loom_post_transaction` failed
with `23514 ... has fewer than 2 entries`; with `IMMEDIATE` then `DEFERRED` in each
savepoint the next post succeeded; `DuplicatePurchase` is distinguishable by
`constraint_name` through the plpgsql function). The major was introduced by round 5's fix.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 6.1 | `SET CONSTRAINTS ALL IMMEDIATE` at the end of a released savepoint stays in force, so the second pending clawback's own post fails (`23514`): it gets a false S4, the purchase commits without it, and the protected M1 test `test_refund_and_dispute_before_the_purchase_are_applied_with_it` breaks | major | probe on the real schema; the test read by the orchestrator (`test_billing_webhook.py` 470) | Name the two deferrable constraints, `IMMEDIATE` then `DEFERRED` in each savepoint (T29, A34) |
| 6.2 | §8.9's stub omitted `created[lt]`, which the grace and frozen backfill depend on | minor | read against §8.5 | Stub lists with `created[lt]`, settable `created`, `resource_missing` for an unknown cursor |
| 6.3 | A 404 look-up was a final `ignored_foreign`; with a key from another account a Loom clawback is dropped for good | minor | reasoning; Stripe 404 for another account's object (high confidence) | 404 is an error: kept unapplied |
| 6.4 | Look-up answer `loom_credits` + unknown org + this or no deployment id was unspecified | minor | read | Kept unapplied, recorded, not looked up again, stale at 15 min |
| 6.5 | `resolve` (no-payment branch, `--force`) races an in-flight purchase: both commit, the credit stays, no alarm | minor | reasoning | Both take an advisory lock on the PaymentIntent id |
| 6.6 | `resolve` bounds and age undefined when the payload is unreadable | minor | reasoning | No readable PaymentIntent = "no payment"; no amount = bounded by the purchase's remainder; age by `received_at` |
| 6.7 | `floor` leaves events created before 0012 and never delivered | minor | reasoning | Runbook post-deploy `stripe-sync --since <deploy start - 1 h>` |
| 6.8 | S4 row lacked the `resolved_clawback` label | minor | read | Added, labels initialised |
| 6.9 | A23 "never" vs §8.6 "stale after 5 failed look-ups" | minor | read | A23 reworded |
| 6.10 | A21 vs §8.5 on which run stores a blocked event | minor | read | The 4th; a run ending before reaching it neither counts nor resets |
| 6.11 | `tcp_user_timeout` is in milliseconds; `open_pool` is outside the stripe worker's files | minor | libpq docs (high confidence); read `db.py` | `tcp_user_timeout=10000` appended by billing to its own conninfo |
| 6.12 | `retry_unapplied`: timeouts unstated; a connection failure costs a pool timeout per event; `last_attempt_at` on rollback unstated | minor | reasoning | Same timeouts; a connection-level failure stops the run (S5); `last_attempt_at` written before each attempt |
| 6.13 | Fixed timing constants make the timing tests slow and flaky under flake-check | minor | arithmetic, not measured | Deadline and both timeouts are settings, scaled down in tests (A34) |
| 6.14 | No named entry points for the sync and resolve; M1 signatures must stay callable | minor | read the protected tests | §8.11 entry points |
| 6.15 | T17 and T20 had no done-when item | minor | read | Added to A16; T13 and T16 marked |
| 6.16 | `asyncio.wait_for` would await the cancelled child's clean-up (~10 s) | minor | psycopg source | `asyncio.wait` + `cancel()` without awaiting (§8.3) |

**Round 7** (1 blocker, 2 majors, 9 minors; ~206k tokens, 48 tool uses; probes in a scratch
database confirmed the T29 constraint sequence, the `DuplicatePurchase` constraint name and
the psycopg cancel path). The blocker came from requirements added in earlier rounds
(`LOOM_DEPLOYMENT_ID`, round 2) and from D10 in the first draft.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 7.1 | New start-up requirements outside dev (`sslmode=verify-full`, `LOOM_DEPLOYMENT_ID`; the worker needing Stripe keys for the sync) contradict protected tests that pin `production_problems()` and start a prod-shaped worker without them; unset-id and no-key behaviour undefined | blocker | confirmed by the orchestrator (`test_billing_stripe_sig.py` 225-248, `test_platform_settings_db.py` 25-46, `test_platform_m3_review2_worker_up.py`) | Services' start-up checks unchanged; `loom-deploy` enforces both (D10, A29); deployment id optional, unset = no deployment-based classification; worker without keys starts and skips the sync; §16 "Constraints from protected tests" (A35) |
| 7.2 | `resolve` did not lock `stripe_payments`; a concurrent `charge.refunded` lets it take back 1500 for 1000 refunded | major | reasoning over `_claw_back` | `FOR UPDATE` on `stripe_payments` after the advisory lock; maximum computed under it; race tests in A33 |
| 7.3 | §8.11 entry points carried no settings (deployment id, foreign threshold, timeouts) | major | read | `settings=` keyword with M1-compatible defaults; one injected `now` for all timestamps |
| 7.4 | A background credit refresh makes two protected Redis reads racy | minor | reasoning (`test_billing_stub_e2e.py` 99, 117) | Refresh awaited up to 0.5 s, then background |
| 7.5 | §8.2 said a non-Loom session is `not_paid`; code and protected test say `ignored` | minor | confirmed (`test_billing_webhook.py` 216) | Table row fixed |
| 7.6 | Outcome metrics counted before COMMIT over-count on rolled-back deliveries | minor | confirmed (`webhooks.py` 138, 204) | Counted after COMMIT |
| 7.7 | Injected `now` vs SQL `now()` | minor | reasoning | One `now` per call passed to SQL; SQL `now()` only for `floor` |
| 7.8 | No done-when for §8.8 (`expires_at`, `loom_deployment` on the PaymentIntent) or the resolve races | minor | read | A35; A33 races |
| 7.9 | A manual `stripe-sync` silently skips when the worker holds the lock | minor | reasoning | CLI waits 60 s, else exits non-zero |
| 7.10 | The `resolved_clawback` look-up has no index | minor | read 0009 | Partial index in 0012 |
| 7.11 | Hand-resolved purchase, later refund: the admin must also take the credit back | minor | reasoning | Runbook step stated in §8.6 |
| 7.12 | Protected `test_platform_ledger.py` uses a fixed `external_ref` | note | read | Noted in §16 constraints |

**Round 8** (no blockers, 2 majors, 10 minors; ~204k tokens, 29 tool uses; probe on
temp tables confirmed the partial-index `constraint_name` through a plpgsql function and the
`SET CONSTRAINTS` savepoint sequence). No money-path trace was found; the majors are
interface gaps.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 8.1 | A clawback's staleness without a Stripe client was undefined ("waits on a look-up" that never runs), against A32 | major | reasoning; M1 worker calls `retry_unapplied(self.pool)` (confirmed) | Waits only with a client and a valid id; otherwise stale at 15 min; A32 names its cases |
| 8.2 | "Back-off resets on deploy" had no entry point | major | read §8.11 | `retry_unapplied(..., started_at=None)`; the worker passes its start time |
| 8.3 | `resolve` with an unreadable amount could over-claw a later refund | minor | arithmetic over `_claw_back` | `--cumulative-refunded C` for a `charge.refunded`; advances `refunded_micros` |
| 8.4 | A hand-resolved purchase does not guard its session against a second credit event | minor | reasoning | `resolved_purchase` check: stored unapplied, S4; index |
| 8.5 | Shared per-worker test database vs global jobs makes Stripe tests order-dependent | minor | confirmed (`testing.py`, protected leftovers) | §16 test-isolation rule |
| 8.6 | The 0.5 s refresh cap may flake protected tests under load | minor | reasoning | `LOOM_STRIPE_REFRESH_WAIT_S` |
| 8.7 | Conninfo "append" wording wrong for keyword conninfo | minor | confirmed (`testing.py` `make_conninfo`) | Merged with `make_conninfo` |
| 8.8 | `last_attempt_at` update could block on a held row | minor | reasoning | Only rows taken with `SKIP LOCKED`, under `lock_timeout` |
| 8.9 | Look-up classification not serialised with an in-flight purchase | minor | reasoning; safe today by PaymentIntent metadata | `pi:` advisory lock in the look-up's transaction |
| 8.10 | SQL `now()` in the second transaction and stale marking vs one injected `now` | minor | read | `%(now)s` parameters |
| 8.11 | Design points with no done-when item | minor | read | A36 |
| 8.12 | Rollback to M1 with poison rows starves M1's job | minor | confirmed (`webhooks.py` catches only `BadEvent`) | Rollback note in §6.6 |

**Round 9** (no blockers, 1 major, 10 minors; ~216k tokens, 35 tool uses; probe in a scratch
database at 0011 confirmed the savepoint sequence, the `DuplicatePurchase` constraint name,
an idempotent same-key replay, and a `refund` sharing an `external_ref`). The major is in
the admin `resolve` path added in rounds 1-5.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 9.1 | `resolve`'s maximum goes negative when a later cumulative refund overtook the poison one, so no N is accepted and the event can never be resolved | major | reasoning over the spec and `_claw_back` arithmetic | `max(0, ...)`; N = 0 always accepted, posts nothing (T28 second order, A33) |
| 9.2 | `resolve_event` lacked `cumulative_refunded`; flags and refusal signal unstated | minor | read | Signature, `ResolveRefused`, CLI exit 2 |
| 9.3 | No single entry point owns §8.3's behaviour; fault seams for A20/T9 and the holdout tests; several outcomes unstated | minor | read | `process_event` (raises `Retryable`); faults at the connection level; outage returns; `lookup_result` values; reset = one immediate attempt; recovered rule |
| 9.4 | Rollback note missed `session_resolved_by_admin` rows: M1 would credit them twice | minor | reasoning | Rollback resolves every unapplied credit event |
| 9.5 | Force-resolved clawback, then the purchase, then a further refund: 1500 for 1000 | minor | reasoning | The purchase raises `refunded_micros` to a resolved refund's cumulative amount |
| 9.6 | `resolve` of a credit event took no session lock and did not see an existing purchase | minor | reasoning | Session advisory lock; refused when a purchase exists, unless `--force` |
| 9.7 | A passed sync event is re-listed and re-blocked for 72 h, flapping S3 | minor | reasoning | `stripe_sync_passed` table; passed ids skipped |
| 9.8 | Re-listing 72 h every run caps on a busy shared account | minor | arithmetic | Runbook line; D4 |
| 9.9 | `SKIP LOCKED` can delay the early refund past M1 acceptance item j's check | minor | reasoning (`acceptance.py` 1593-1642) | Item j waits up to 70 s (deploy worker) |
| 9.10 | Migration 0012 lock order could deadlock with a running M1 delivery | minor | reasoning | Alter `stripe_events` first, index after |
| 9.11 | A36 "never waits on a delivery" too strong | minor | read | Worded as the event row only |
| 9.12 | (Unconfirmed) the refresh wait under load vs protected Redis reads | note | not run | Default 2 s; production 0.5 s; A24 sets it low |

**Round 10** (no blockers, 3 majors, 6 minors; ~228k tokens, 43 tool uses; probes:
`ON CONFLICT` and advisory waits honour `lock_timeout`; `DuplicatePurchase` constraint
name; a purchase plus four savepoints with one failing check; psycopg 3.3.6 against a TCP
proxy that stops forwarding: a cancelled task ended after 10.0 s). The orchestrator fixed
round 10 by simplifying (back to M1's blocking lock; a manual sync never moves the window;
the purchase applies a clawback resolved before it) rather than adding mechanisms.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 10.1 | A successful manual `--since` sync set `synced_through`, so the next automatic run skipped a span that was never synced (lost credits, no alarm) | major | reasoning on the spec text | Only automatic runs move `synced_through` or the backfill (T30, A36) |
| 10.2 | The staleness rule never marked a clawback stale when its payment exists and the run has a Stripe client (no S1a for a stuck clawback of a real payment) | major | reasoning on the spec text | "Waits on a look-up" also requires no `stripe_payments` row; A32 case |
| 10.3 | `SKIP LOCKED` on pending clawbacks broke contracts §3's "applied in the purchase's transaction" and needed a weakened acceptance item j | major | confirmed against `acceptance.py` 1629-1641, `webhooks.py`, contracts §3 | Back to M1's blocking `FOR UPDATE` under `lock_timeout`; `resolve` takes the `pi:` lock before the event row (no cycle); item j untouched |
| 10.4 | A clawback resolved before its purchase: accounting order and later disputes undefined | minor | reasoning | The purchase applies it then (`applied_after_resolve`); no hand adjustment for no-payment clawbacks |
| 10.5 | The bounded-time text claimed keepalives bound the clean-up; it is psycopg's ~10 s | minor | probe | Text corrected; A19 budget |
| 10.6 | A vanished client can leave server-side locks for hours | minor | reasoning (stock keepalive 7200 s) | `idle_in_transaction_session_timeout` on billing's connections; RDS server keepalives |
| 10.7 | §8.11 gaps (`raw` on the sync path, `retry`, `StoreFailed`, return types, back-off formula, units, backfill end, stub controls, holdout transport) | minor | read | All stated |
| 10.8 | A sync event whose second transaction also fails is dropped after 72 h | minor | reasoning | Goes into `stripe_sync_passed` at once (S4) |
| 10.9 | The rollback note forced hand credits for events M1 would apply | minor | reasoning | Narrowed; rollback across M3 is a last resort |

**Round 11** (no blockers, 2 majors, 12 minors; ~205k tokens, 34 tool uses; probes:
savepoint sequence; L3 constraint name through `loom_post_transaction`; set-based
`UPDATE` against a purchase's ordered `FOR UPDATE` deadlocked (`40P01`) under a seq-scan
plan). Both majors were in mechanisms added in rounds 4 and 10; the fix again simplifies
(park instead of resolve for a clawback still waiting for its payment).

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 11.1 | `synced_through` had no initial value: a sync that never succeeded skips `floor` .. `now - 72 h` when it finally runs | major | reasoning (Postgres `LEAST` ignores NULL) | The migration sets `synced_through = floor` (A21) |
| 11.2 | A clawback resolved before its purchase, applied by the purchase, is orphaned when that application fails (`processed_at` already set) | major | reasoning; `retry_unapplied` selects only `processed_at IS NULL` (confirmed) | `park` keeps a waiting clawback unprocessed; `resolve --no-payment-expected` for "never"; a purchase reopens such a resolve (T31, A33) |
| 11.3 | Set-based stale marking could deadlock with a purchase's ordered `FOR UPDATE` | minor | probe (`40P01` under a seq scan) | `... IN (SELECT ... ORDER BY received_at, id FOR UPDATE SKIP LOCKED)` |
| 11.4 | Purchase-resolve lock order unstated | minor | reasoning | Event row, then session lock; `resolve` under the same timeouts |
| 11.5 | `retry_unapplied`'s `SKIP LOCKED` no longer covered the attempt; `source` for retries | minor | read | `process_event(retry=True)` takes the row `SKIP LOCKED`; stored `source` |
| 11.6 | Passed sync ids lost when they age out | minor | reasoning | Automatic runs retry passed ids when nothing blocks |
| 11.7 | A capped manual `--since` silently covers only 5,000 events; blocked state | minor | reasoning | Exits non-zero; blocked state untouched |
| 11.8 | `session_resolved_by_admin` could block a genuine payment after a noise resolve | minor | reasoning | Only `--hand-credited` blocks; an existing purchase wins |
| 11.9 | Interface points to guess (404 result, `since` type, guard outcome, `last_error_kind` format, 4th run, exponent, tick tolerance) | minor | read | All stated in §8.11 |
| 11.10 | Sync advisory lock vs the new `idle_in_transaction_session_timeout` | minor | reasoning | Dedicated non-pooled autocommit connection, explicit unlock |
| 11.11 | T14 in production is a 502 from Caddy, not a 503 | minor | confirmed (`api.py`, `cli.py` open the pool before listening) | T14 reworded |
| 11.12 | Laptops in `stripe` mode without a deployment id | minor | reasoning | README line |
| 11.13 | Contracts row missed the exceptions to "applied in the purchase's transaction"; §16 wording on the protected tests | minor | read | Row and wording fixed |
| 11.14 | A broken purchase was first blamed on an innocent clawback | minor | probe | `IMMEDIATE`/`DEFERRED` once right after the purchase's post |

**Round 12** (no blockers, 2 majors, 12 minors; ~213k tokens, 33 tool uses; no probes, all
reading). Both majors were in the round-11 `park` mechanism and its interfaces.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 12.1 | A parked clawback stayed exempt from staleness after its payment existed, so a failed application (or a retryable one, or a reopened parked one) was silent: refund and credit both kept | major | reasoning on the spec text | Parking ends when the payment exists: the purchase or a reopen clears `parked_at`, sets `unparked_at`, clears `stale_reported_at`; staleness from `unparked_at` (A33) |
| 12.2 | §8.11 had no `no_payment_expected`, `hand_credited`, `park` or `list` entry points that A33/A36 must test | major | read | `resolve_event` keywords; `park_event`, `list_events` with their refusals |
| 12.3 | `resolved_purchase` check order unstated: done before the session lock it double-credits | minor | reasoning | After the session lock and the purchase look-up; race test |
| 12.4 | `--hand-credited` had no storage | minor | read | Column `hand_credited`; index predicate |
| 12.5 | §11 out of step (`parked_at`, `park`, `stripe-sync` CLI); `DELETE` grant for `stripe_sync_passed` | minor | read; contracts §3 grants | Rows updated; explicit grants |
| 12.6 | T27/A32 vs the due rule (a first stored failure was due only after 60 s) | minor | arithmetic | `delay = 0` for `error_count <= 1` |
| 12.7 | Manual sync lock wait fixed at 60 s; lock key unstated | minor | read | `LOOM_STRIPE_SYNC_LOCK_WAIT_S`; key stated |
| 12.8 | A capped manual run cannot be repaired by narrowing `--since` | minor | reasoning (newest first) | `--until`, printed by the capped run |
| 12.9 | Reopen wording, `error_count = 0`, `stale_reported_at` after reopen | minor | read | Purchase's transaction; due at once; stale reset |
| 12.10 | S1b definition vs parking | minor | read | Not parked |
| 12.11 | Non-exception S4 paths had no identity | minor | read | `last_error_kind` = path name |
| 12.12 | Deadline ownership for the sync and the retry | minor | read | `run_with_deadline`, raises `Retryable` |
| 12.13 | The sync's dedicated connection vs the role gate and keepalives | minor | contracts §3 | After the gate, billing's conninfo |
| 12.14 | Traps: `skipped` counted as processed; one `SignatureError` for both cases; R's own store failure; deployment-id case; `pi:` lock in the handler | minor | confirmed (`webhooks.py` `Outcome.processed`, `stripe_sig.py`) | All stated (§8.11) |

**Round 13** (no blockers, 1 major, 10 minors; ~213k tokens, 35 tool uses; probes: the
second transaction's guard on a processed row; `DuplicatePurchase` constraint name and its
`55P03` wait; the `SET CONSTRAINTS` sequence; a two-connection `park`/purchase race left a
parked clawback with a payment row). No money-loss trace was found this round.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 13.1 | The second-transaction SQL omitted `last_attempt_at` while the prose set it; T27/A32's ordering depended on which was followed | major | direct read; the T27 consequence reasoned | SQL sets `last_attempt_at`; step 2 takes first failures and never-attempted first; T27 states its backlog |
| 13.2 | `park` without the `pi:` lock: a concurrent purchase misses the unpark, leaving a parked clawback with a payment and no alarm | minor | probe (two connections) | `park` takes the `pi:` lock first; "parked" defined in SQL as `parked_at` and no payment row; staleness from `COALESCE(unparked_at, received_at)` |
| 13.3 | `sync_once` had no `until`; manual/automatic, lock miss and `skipped` unstated | minor | read | `manual=`, `until=`, `SyncLocked`, `reason="locked"`, field meanings |
| 13.4 | `Retryable` could not tell skip from stop | minor | reasoning | `Retryable.connection_level` |
| 13.5 | A parked clawback was silent forever if its payment never came | minor | reasoning | Stale after `LOOM_STRIPE_PARK_MAX_S` (7 days), in S1b |
| 13.6 | `resolve`'s dispute maximum ignored the once-per-dispute rule | minor | confirmed (`webhooks.py` 320-331) | Maximum 0 when a refund for the dispute id exists; `external_ref` stated |
| 13.7 | `--until` at the oldest listed second skips same-second events | minor | reasoning | Printed value + 1 s |
| 13.8 | Module-scoped scratch databases are shared by pytest-repeat's copies | minor | CLAUDE.md | Function-scoped or reset at test start |
| 13.9 | Events API array encoding unpinned | minor | confirmed (`form_encode`) | Indexed form; fakes accept both |
| 13.10 | Rollback note missed the reopen | minor | confirmed (M1 selects pending by `processed_at IS NULL`) | Third kind listed |
| 13.11 | S4 for `pending_clawback` counted before COMMIT double-counts on redelivery | minor | reasoning | Counted after COMMIT |
| 13.12 | (Unconfirmed) S5 at start-up; Adaptive Pricing; endpoint registration order | note | not checkable offline | R runs at start; Adaptive Pricing off per session (B7 check); endpoint registered before the first deploy |

**Round 14** (no blockers, 2 majors, 7 minors; probes: `prometheus_client` refuses a
counter `x_total` and a gauge `x` in one registry (`DuplicateTimeseries`); the
`SET CONSTRAINTS` sequence; `ON CONFLICT` wait under `lock_timeout`; `FOR UPDATE` does not
wait on an uncommitted insert). No money-loss trace was found.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 14.1 | S1a's counter `..._unapplied_stale_total` and S1b's gauge `..._unapplied_stale` collide in `prometheus_client`; billing would not start | major | probe | Gauge renamed `loom_billing_stripe_unapplied_stale_events` |
| 14.2 | `park` after the event was already stale gave no S1a at 7 days; A32 ambiguous; 7-day anchor unstated | major | reasoning | `park` clears `stale_reported_at`; 7 days from `parked_at`; A32 explicit |
| 14.3 | Hand-crediting a payment whose checkout event was never stored: a later sync credits it again | minor | reasoning | Runbook rule: store (sync) and resolve `--hand-credited` first |
| 14.4 | A passed sync id that leaves the window is dropped silently | minor | reasoning | Retried by `GET /v1/events/<id>` each automatic run; after 29 days counted as a gap |
| 14.5 | Interfaces: `SyncResult.passed`, guard-skipped second transaction status, `--claw-back-micros 0` accounting, one S1b definition, flag combinations, 500 code | minor | read | All stated |
| 14.6 | A deadline hit stopped the whole retry run | minor | reasoning | `Retryable.kind`; stop on `connection` or a second consecutive `deadline` |
| 14.7 | `last_attempt_at` pre-write restarted the back-off on a skipped attempt | minor | reasoning | Written inside the attempt and by the error path |
| 14.8 | (Unconfirmed) function-scoped databases vs the flake-check budget; Stripe's first retry gap vs S2a | note | not measured | Template-cloned databases; B7 checks the retry gap |

**Round 15** (no blockers, 1 major, 7 minors; probes: the `SET CONSTRAINTS` sequence, the
L3 constraint name through the function, `lock_timeout` on advisory and `ON CONFLICT` waits).
No money-loss trace was found; the major is a precedence gap between two rules.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 15.1 | A clawback parked before its look-up settled could read as "waiting on a look-up" forever, so the 7-day rule never fired (no S1a, no S1b) | major | reasoning | Parking ends the look-up wait; A32's 7-day case runs with a Stripe client and `lookup_attempts = 0` |
| 15.2 | The second transaction's insert took `received_at` from SQL `now()` | minor | read | `received_at = %(now)s` there and in `handle_event` |
| 15.3 | `Retryable.kind` undefined for classes `08`, `53`, `57P0x`, `25006` | minor | read | All `connection` |
| 15.4 | B7's grace rule inverted; the 30-min staleness hard-coded against the grace | minor | reasoning | Raise the grace when Stripe's first retry is later than it; 30 min derived from the settings |
| 15.5 | A passed sync event had no standing alarm | minor | reasoning | Gauge of `stripe_sync_passed` rows in S1b |
| 15.6 | Two deterministic slow events stopped every retry run before stale marking | minor | reasoning | Deadline skips rotate the order; stale marking runs unless the stop was `connection` |
| 15.7 | A dispute resolved with N = 0 left no once-per-dispute record | minor | confirmed (`webhooks.py` dispute check) | The check also counts a processed `charge.dispute.created` with the same dispute id |
| 15.8 | Interface points (30-day window, return count, `SyncResult.reason`, `list_events` keys, `park_event`/`resolve_event` returns, `manual` without `since`, `--no-payment-expected` with a payment, `skipped` naming) | minor | read | All stated |

**Round 16: APPROVE** (no blockers, no majors, 6 minors; ~233k tokens, 37 tool uses; probe:
the L3 constraint name, an idempotent re-post with the index in place, the `SET CONSTRAINTS`
sequence with a failing middle savepoint, and `re.match` accepting a trailing newline).
This round met the stop condition. Its minors were applied afterwards without another
review round (scoping and clarifications only; the first one keeps the protected webhook
tests repeat-safe under flake-check).

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 16.1 | The round-15 once-per-dispute clause was global and unindexed: the protected webhook tests' fixed `dp_test_x` would fail their second `--count 2` copy | minor | read (`test_billing_webhook.py` 396, 478; `webhooks.py` 324-328) | Scoped to the same PaymentIntent; partial index; new tests use random ids |
| 16.2 | `retry_unapplied` could stop at the same event every run on a per-event connection-class error (S5 only) | minor | reasoning | After 3 such stops with Postgres otherwise answering, the event is recorded and passed |
| 16.3 | A clawback left behind by its purchase kept its old back-off | minor | reasoning | `last_attempt_at = NULL`; a non-raising attempt resets `error_count` |
| 16.4 | `pi:` lock scope for credit-event resolve unstated (a possible cycle) | minor | reasoning | Only clawback `resolve` and `park` take `pi:` |
| 16.5 | The look-up pattern with `$` accepts a trailing newline | minor | probe | `re.fullmatch` |
| 16.6 | Points to guess (flag shapes, re-park, S5 on a stopped run, counts, the 15-min setting, module names, pending failure accounting, direct `handle_event` metrics, reopen filter, L4 wording) | minor | read | All stated |

**Round 17, after the cut: APPROVE** (no blockers, no majors, 9 minors, 4 notes; ~193k tokens, 26 tool
uses, 12.5 min; probe in a scratch database at 0011, dropped afterwards: the savepoint and
`SET CONSTRAINTS` sequence, the L3 constraint name and A21's race without a private seam,
the second transaction's guard, the reopen query re-checking its `WHERE` behind a
committing resolve, `jsonb` refusing NUL and lone surrogates). Minors 17.1-17.4 and
17.6-17.9 were applied without another round; 17.5 and the notes are open.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 17.1 | A resolve inside the T4 race window (clawback stored after the purchase's pending query, resolved before the purchase commits) drops the clawback with no alarm; only seconds wide and operator-made, so rated minor though it meets the letter of "credit kept, no alarm" | minor | reasoning; READ COMMITTED visibility probed | `resolve` refuses an event younger than `LOOM_STRIPE_STALE_AFTER_S` (A31) |
| 17.2 | A poison clawback of a credited payment has no exit after 30 days if step 1 had M1's window | minor | reasoning | Step 1 has no age window |
| 17.3 | S2c (>= 2 in 1 h) can miss one event's 503s; a 503 stores nothing | minor | reasoning | S2c >= 1 in 1 h |
| 17.4 | A36's scan would fail on the identifier `LoomEnv` (`env-check.ts`) | minor | code | Scan string literals and JSX text only |
| 17.5 | The 3-consecutive-connection-stops rule has no stated home for its state (a module global or the database) | minor | read | Open: the reviewer suggests removing the rule (S5 alarms anyway) |
| 17.6 | A non-raising attempt left `last_error`; the reopen's `error_count` unstated | minor | read | Cleared; reopen sets `error_count = 0` |
| 17.7 | Classes `53`, `57P0x`, `25006` inside a pending-clawback savepoint described two ways | minor | read | `Retryable.kind` decides |
| 17.8 | The write after a deadline skip can wait on the cancelled child's row lock | minor | reasoning | Best effort; `55P03` leaves the order |
| 17.9 | Reading the stored identity needs a `SELECT ... FOR UPDATE` before the upsert; concurrent first failures can both count | minor | reasoning | Stated; tests do not assert one under concurrency |
| 17.n1-n4 | `idle_in_transaction_session_timeout` on the pool shared with reconcile; cost of the fault-injection tests in flake-check; S2a/S2b raised by outsiders and by m3-g's old-`t` step; S5's metric math (`EPOCH(m) - m`) unchecked, a counter with missing = breaching would avoid it | note | not checked | Open |

Removable without losing a property the spec needs (the reviewer's answer, not applied; the
user decides): the 503/500 SQLSTATE classification (store every failure instead), the
`DuplicatePurchase` class (keep the L3 index), the 6 s deadline with its cancelled child,
deadline-skip rotation and T20's blackhole test, the 3-stops rule, payload sanitising and
`raw`, the `IMMEDIATE`/`DEFERRED` pair (`loom_post_transaction` already checks balance and
entry count before inserting), the back-off with `started_at` reset, failure identity for
S4, the step-1/step-2 tiers; replace the reopen by a `resolve` that only acknowledges a
clawback (or a step-1 sweep). Keep: the second transaction (reduced), a plain savepoint per
pending clawback, the `signature_mismatch` split, the stale counter, a constant refresh
bound. (Taken by the user on 2026-11-18 and applied before round 18; see below.)

**Round 18 revision (user decision 2026-11-18, before the review).** The user took round
17's list of removable mechanisms, asked for each removal to be checked against the traces
and the protected tests, and added the live-mode prerequisites (§8.13) and the `hooks.`
health check (H3, §8.5). Probes on a throwaway database (dropped): plain savepoints without
`SET CONSTRAINTS`; `loom_post_transaction`'s up-front refusals; the L3 race without
`DuplicatePurchase`; the reduced second transaction; the reopen sweep. §8 went from 514 to
506 lines (§8.3 175 -> 103, §8.11 90 -> 71, §8.6 44 -> 65 with the sweep, §8.5 29 -> 53
with H3, §8.10 23 -> 33, new §8.13 23); the round-18 minors below brought it to 520.

| Mechanism | Round 18 | Why (trace, test or probe) |
|---|---|---|
| 503/500 SQLSTATE classification | removed | Every failure takes the 500 path (stored with `last_error`); 503 only when that store fails too. T3, T8, T9, T17, T19 hold as rewritten. Live mode: LM4 |
| `DuplicatePurchase` | removed; L3 index kept | The race reaches billing as `IdempotencyConflict` (probe: `23505`, `constraint_name` = the index) and takes the 500 path; T3's retry finds `already_credited`. `ledger.py` unchanged |
| 6 s deadline, its cancelled child, the deadline-skip rotation | removed | Stripe treats a late answer like a 503 and L1 makes a late COMMIT safe (T9 b, c); T20 retired. Kept: the pool, lock and statement timeouts and the client keepalives. Dropped with the child: `idle_in_transaction_session_timeout` (the server keepalives end a vanished client) |
| 3-consecutive-connection-stops rule | removed (closes 17.5) | R stops only when an attempt's error store fails too; a per-event error is stored and passed |
| Payload sanitising and `raw` (and the non-finite-number parse hooks) | removed | No trace or protected test sends such a payload; an unstorable payload is 503 and lost (§8.10). Live mode: LM3 |
| `IMMEDIATE`/`DEFERRED` pair | removed | Probe: plain savepoints commit both clawbacks of `test_billing_webhook.py:470` and roll a failing middle one back alone; `loom_post_transaction` refuses one-entry (`22023`) and unbalanced (`23514`) posts before inserting, so the deferred checks cannot fail for its posts; round 6's failure came from the `IMMEDIATE` itself |
| Back-off and the deploy reset (`last_attempt_at`, `started_at`) | removed | R re-runs every unapplied event each minute, so a fixing release applies a poison event within a minute (T17). Live mode: LM5 |
| Failure identity (`last_error_kind`) | removed | S4 counts the write that makes `error_count` 1 (first failure since a clean attempt), so repeats are not re-counted (T17) |
| Step-1/step-2 tiers | removed | One newest-first pass up to 500 with no age window keeps T21 and T22, and re-runs a poison clawback of a credited payment however old (17.2) |
| Reopen inside the purchase | replaced by the reopen sweep (R's first step) | The purchase's query stays M1's; the sweep reopens a resolved clawback once its payment exists (probe); T24 rewritten. Chosen over an acknowledge-only `resolve`, which needs a staleness exemption that ends when the payment appears (the park rule of rounds 12-15). Also dropped: `resolve`'s refusal of an event younger than 15 min (17.1), whose race the sweep now covers |
| Second transaction | kept, reduced | Makes a failing event visible to R, S1a and an admin (T17, T19); also R's error store. No `last_error_kind`, `last_attempt_at` or `raw` |
| Plain savepoint per pending clawback | kept | T19 (a failing clawback must not block its purchase's credit); T23 and `test_billing_webhook.py:470` |
| Secret-mismatch alarm (S2a) | kept | T25 |
| Stale counter (S1a) and gauge (S1b) | kept | T11, T17, T24: the only alarm for an event stored but not applied |
| Credit-refresh bound | kept, as a constant (2 s) | Protected Redis reads right after a delivery (`test_billing_webhook.py:160`, `test_billing_stub_e2e.py` ~99 and ~117) and T14 (Redis frozen) |
| Endpoint health check | added (H3) | T26 narrowed to what H3 cannot see |

**Round 18: APPROVE** (no blockers, no majors, 6 minors, 5 notes, 2 unconfirmed; ~211k
tokens, 26 tool uses, 11 min; probes in two throwaway databases at 0011 with 0012's
columns and index added by hand, both dropped: plain savepoints, the L3 index and a
same-key replay, the second transaction's statement (also against a held row and an
uncommitted insert of the same id), the reopen sweep, stale marking with a held row).
Minors 18.1-18.6 and the region attribute were applied without another round; the notes
are open.

| # | Finding | Severity | Trace confirmed? | Fix in this spec |
|---|---|---|---|---|
| 18.1 | A lock timeout on the event's own row (held by R, a `resolve` or another delivery, or an uncommitted insert of the same id) also times out the second transaction, so the answer is 503, not the 500 §8.3 and A19 promised; A19's case could not pass | minor | probe (`55P03` on a held row and on an uncommitted insert) | §8.3, T2, A19: 500 for a lock wait elsewhere, 503 while the event's row is held; in R it ends the run |
| 18.2 | `resolve` resolved a clawback whose `payment_intent` is not a plain string, and the sweep could never reopen it although its payment is credited | minor | probe (an object-valued PaymentIntent never matched in three sweeps) | `resolve` refuses a clawback without a non-empty string PaymentIntent (fixed forward) (§8.6, §8.11, A31) |
| 18.3 | R on a stored row that fails the shape check was unspecified | minor | code (`webhooks.py` 269-270) | R skips it with no error store, as M1 does (§8.11) |
| 18.4 | Interface points to guess: the settings' field names, the `result` of a clawback applied by its purchase, `list_events` order, `error_count` after a reopen | minor | read | §8.11 |
| 18.5 | A24 said any error in a pending-clawback savepoint leaves the purchase committing; A23 and §8.3 say a lost connection fails the delivery | minor | read | A24: an error that leaves the transaction usable |
| 18.6 | §8 cited round-18 probes with no Appendix A entry | minor | read | This entry |
| 18.n1-n5 | H1 and H3 alarm from `terraform apply` until the first deploy (the runbook should expect it); S2a and S2b fire on one request anyone can send, and m3-g's old-`t` step fires S2b (17.n3); a hand adjustment for a resolved clawback is taken twice once the sweep reopens it (runbook: never adjust for one); dead pooled connections after an RDS restart can cost one 503 once Postgres is back; §16 should also forbid fake-clock and backlog tests from leaving rows in the shared database (500 newer rows would push the protected raced-clawback test's row out of R's 500) | note | reasoning | n1, n3, n5 applied in wave 0 (§5.3 runbook line, §8.6, §16 test isolation); n2, n4 open |
| 18.u | `data.aws_region` in provider 6.x reads `region` (`name` deprecated); A22's "200 within 3 s" with Redis frozen under flake-check load (17.n2) | unconfirmed | not checkable offline | Attribute named in §8.5; timing open |

**Totals.** 16 rounds before the cut, each a fresh Fable reviewer: 1 blocker (round 7), 26
majors, about 154 minors; about 3.0M reviewer tokens and 499 tool uses in all. Round 17,
after the cut: 0 blockers, 0 majors, 9 minors; round 18, after the user's removals: 0
blockers, 0 majors, 6 minors. Later majors were mostly introduced by the fixes of earlier rounds (rounds
2-15), in the mechanisms this review added on top of M1 (sync, foreign look-ups, `resolve`
and `park`), not in M1's core (L1-L4), which every round found sound.
