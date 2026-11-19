# Loom platform stack (AWS control plane)

Terraform for the M3 control plane: one EC2 box, RDS, Secrets Manager containers, IAM,
ECR, S3, DNS and SES, the serving reaper (disabled), CloudWatch alarms and an AWS budget.
Spec: `docs/phase1-m3-spec.md` §4, §5 (this stack is §5.1-§5.6). The remote-state bucket
is a separate bootstrap stack, `infra/aws/state`. `infra/aws/bench` is unrelated and keeps
its local state.

## What it creates

| File | Resources |
|---|---|
| `network.tf` | VPC `10.42.0.0/16`, one public subnet (the box), two private subnets (RDS), internet gateway, route tables, security groups (box: 80/443 in; RDS: 5432 from the box only). No NAT gateway |
| `ec2.tf`, `user_data.sh.tftpl`, `dlm.tf` | `t4g.medium` Amazon Linux 2023 arm64 box (IMDSv2, hop limit 1, no key pair, SSM only), 30 GB root and a separate 20 GB data volume at `/var/lib/loom` (`prevent_destroy`), Elastic IP, log group `/loom/platform`, daily snapshots of the data volume (7 kept) |
| `rds.tf` | PostgreSQL 16 `db.t4g.micro`, 20 GB gp3, encrypted, single AZ, 7-day backups, `rds.force_ssl = 1`, TCP keepalives, master user `loom_owner` with an RDS-managed secret |
| `secrets.tf` | Ten secret containers under `loom/platform/` (no values, ever) with resource policies naming their one reader role |
| `iam.tf` | Instance role, GitHub OIDC role for `images.yml`, serving reaper role, SES SMTP user `loom-ses-smtp` (spec §5.6 lists every grant) |
| `ecr.tf` | Repositories `loom/gateway`, `billing`, `web`, `poolmgr`, `devstack`, `caddy` (immutable tags, scan on push, last 20 images) |
| `s3.tf` | Lease bucket (M4) and releases bucket (`var.releases_bucket`): private, SSE, TLS-only policy, versioning off |
| `dns.tf`, `ses.tf` | A records (apex, `api`, `hooks`), CAA, SES Easy DKIM, custom MAIL FROM (MX and SPF), DMARC; SES identity and configuration set |
| `reaper_serving.tf` | `loom-serving-runpod-reaper` Lambda (dry run, schedule **disabled**) |
| `alarms.tf`, `healthcheck.tf` | SNS topic `loom-platform-alerts` and every §5.3 alarm; the Route 53 health check on `hooks.<domain>` and alarm H3 (created through the `aws.us_east_1` provider) |
| `budget.tf` | Monthly AWS budget, mail at 50%, 80%, 100% actual and 100% forecast |
| `site.tf` | SSM parameter `/loom/platform/site` (domain, product name and the keys `loom-deploy` reads) |

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `domain` | none | The site's domain. The apex serves the web app, `api.` the gateway, `hooks.` the Stripe webhook |
| `brand_name` | none | The product name users see |
| `releases_bucket` | none | Name of the releases bucket (S3 names are global) |
| `ops_email` | none | Alarm and budget mail; confirm the SNS subscription |
| `budget_usd` | none | Monthly budget in USD (the plan: 75) |
| `route53_zone_id` | `null` | When set, Terraform writes the DNS records; when `null`, they are the `dns_records` output |
| `serving_reaper_enabled` | `false` | Turns on the reaper's schedule and the P1 alarms' actions (M4) |
| `instance_type` | `t4g.medium` | Graviton only: the images are arm64 |
| `db_deletion_protection` | `true` | Set `false` and apply before `terraform destroy` |
| `create_github_oidc_provider` | `true` | Set `false` when the account already has the GitHub OIDC provider |
| `region` | `us-east-1` | Must be `us-east-1` (a precondition on the health check fails the plan otherwise) |

Copy `terraform.tfvars.example` to `terraform.tfvars` (gitignored).

## Checks that need no AWS credentials

Run from this directory with Terraform 1.16.7 (CI pins the same version):

```
terraform fmt -check -recursive -diff
terraform init -backend=false -input=false -lockfile=readonly
terraform validate
terraform test          # mock providers, tests/*.tftest.hcl
```

CI (`terraform` job) also runs tflint (AWS ruleset) and `trivy config`; the configs are in
`.github/terraform/`. Inline ignore comments are not allowed; accepted trivy findings are
listed in `.github/terraform/trivyignore.yaml` and in "Accepted findings" below.

`terraform test` builds the Lambda zip into `.build/` (gitignored).

## Plan and apply

Never run these against AWS without the user's go-ahead (spec §5.5, Stage B, cost §14).

1. Bootstrap the state bucket once: in `infra/aws/state`, `terraform init`, then
   `terraform apply -var state_bucket=<name>`. Keep its local `terraform.tfstate`.
2. Here: copy `backend.hcl.example` to `backend.hcl` (gitignored), set the bucket, then
   `terraform init -backend-config=backend.hcl`.
3. `terraform plan` with read-only credentials. Before the state bucket exists, copy
   `backend_override.tf.example` to `backend_override.tf` and `terraform init -reconfigure`
   (local state; plan only).
4. `terraform apply`. Until the first deploy is healthy, H1 and H3 and the alarms whose
   missing data is breaching (P3, G3, S5) are in ALARM: expected (spec §5.3, runbook).

## Outputs

`releases_bucket`, `lease_bucket`, `dns_records` (create these at your DNS host when
`route53_zone_id` is null), `elastic_ip`, `instance_id`, `rds_endpoint`,
`rds_master_secret_arn`, `ecr_repository_urls`, `github_images_role_arn`,
`alerts_topic_arn`, `site_parameter`, `ses_smtp_user`, `serving_reaper_function`.

## Accepted findings

tflint (AWS ruleset 0.50.0) is clean. `trivy config` (0.76.1, embedded checks) reports the
findings below on this stack. None is fixed in code, for the reason given; each needs an
entry in `.github/terraform/trivyignore.yaml` (the user's to accept) before CI's trivy
step passes. Spend and the spec's own choices drive most of them.

| ID | Severity | Resource | Reason |
|---|---|---|---|
| AWS-0095 | HIGH | `aws_sns_topic.alerts` | Encrypting the topic needs a customer-managed key (about $1 a month) whose policy lets CloudWatch alarms publish; the AWS-managed `alias/aws/sns` key blocks alarm delivery. The topic carries alarm names only |
| AWS-0132 | HIGH | `aws_s3_bucket_server_side_encryption_configuration.leases`, `.releases` (and `state` in `infra/aws/state`) | SSE-S3 (AES256) instead of a customer-managed KMS key (spec §5.2 "SSE"; key cost and policy for no gain: one account, no sensitive content) |
| AWS-0104 | CRITICAL | `aws_security_group.instance` | The box needs outbound internet (package mirrors, ECR, S3, Stripe, SES, Let's Encrypt); destination IPs are not fixed. Ingress is 80/443 only |
| AWS-0178 | MEDIUM | `aws_vpc.platform` | No VPC flow logs: extra log group, role and ingestion cost; the spec lists none |
| AWS-0090 | MEDIUM | `aws_s3_bucket.leases`, `.releases` | Versioning is off by the spec (§5.2 s3, §5.6), and a test pins it |
| AWS-0089 | LOW | `aws_s3_bucket.leases`, `.releases` (and `state`) | No access-log bucket; CloudTrail covers management events |
| AWS-0017 | LOW | `aws_cloudwatch_log_group.platform`, `.serving_reaper` | Log groups use the default encryption, not a customer-managed key |
| AWS-0027 | LOW | `aws_ebs_volume.data` | Encrypted with the AWS-managed key, not a customer-managed one |
| AWS-0033 | LOW | `aws_ecr_repository.loom` (six) | Same, for the repositories |
| AWS-0098 | LOW | `aws_secretsmanager_secret.platform` (ten) | Secrets use the AWS-managed key; a customer-managed key would add cost and a second policy to every reader |
| AWS-0143 | LOW | `aws_iam_user.ses_smtp` | The SES SMTP user holds its policy directly (spec §5.2, and a test reads `aws_iam_user_policy.ses_smtp`); it is the only user |
| AWS-0133 | LOW | `aws_db_instance.platform` | Performance Insights is off: the spec does not ask for it, and support on `db.t4g.micro` is unverified without AWS access |
| AWS-0066 | LOW | `aws_lambda_function.serving_reaper` | No X-Ray tracing: its role is an exact allowlist (spec §5.6, A6) with no tracing permission |
