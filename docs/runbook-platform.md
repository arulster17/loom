# Runbook: the Phase 1 control plane on AWS (M3)

Operating the platform (gateway, billing, web, pool manager, Redis, Prometheus, Grafana
behind Caddy) on one EC2 box with RDS: Stage B from `terraform apply` to teardown, the
deploy, the P4 rule, rollback, secrets, the Stripe endpoint, SES, Google, the alarm test.
The design is [phase1-m3-spec.md](phase1-m3-spec.md) (§4, §6, §10); the interfaces are
[phase1-contracts.md](phase1-contracts.md) §2. The pod CA is
[runbook-pod-ca.md](runbook-pod-ca.md), Stripe is [runbook-stripe.md](runbook-stripe.md).

**Spending money.** Everything from section 2 on starts or keeps paid AWS resources. Do it
only after the user's separate yes and cap for Stage B (spec §14). Nothing here runs RunPod
(M4): the backend on the box is the mock.

Variables used below (set them in your shell; none is a secret):

```sh
export AWS_REGION=us-east-1
export SHA=<the release's 40-character git sha>
export BUCKET=<terraform output releases_bucket>
export INSTANCE=<the instance id, terraform output>
```

## 0. What runs where

```
operator --(aws ssm send-command / start-session)--> box: loom-deploy (host Python 3.12)
box: Caddy 80/443 -> web, gateway, billing   | redis, prometheus, grafana, poolmgr
     /run/loom   tmpfs: rendered secrets and env files (lost at reboot, section 7)
     /var/lib/loom   data volume: release.json, apply-in-progress.json, deploy.log,
                     releases/<sha>/, redis AOF, settle spool, caddy certificates,
                     poolmgr state, prometheus TSDB
RDS PostgreSQL 16 (TLS verify-full), Secrets Manager, SSM parameter /loom/platform/site
```

Only 80 and 443 are open. There is no SSH: the operator uses SSM. Grafana and Prometheus
are on `127.0.0.1` and are reached with port forwarding (section 9).

## 1. One-time GitHub setup (before the first image build)

`.github/workflows/images.yml` builds the arm64 images and uploads the release. It runs
by manual dispatch on a commit of `main` and nothing else:

1. Repository settings, Environments: create `platform-images`, deployment branches
   **`main` only**. The AWS role trusts only the subject
   `repo:arulster17/loom:environment:platform-images`, so this rule is the branch check.
2. Repository settings, Variables: `LOOM_IMAGES_ROLE_ARN` (the OIDC role Terraform makes)
   and `LOOM_RELEASES_BUCKET` (the releases bucket). They are not secrets.

## 2. Terraform (B1, B2)

Remote state first (`infra/aws/state`, its own local state, which you keep), then the
platform stack; see `infra/aws/platform/README.md` for the variables (`domain`,
`brand_name`, `ops_email`, `releases_bucket`, `budget_usd`; no defaults for the first
four).

- B1: apply `infra/aws/state`; write `infra/aws/platform/backend.hcl` from
  `backend.hcl.example`; `terraform init -backend-config=backend.hcl`.
- B2: `terraform apply` in `infra/aws/platform`. Record the outputs. Confirm the SNS
  subscription mail and the budget. A second `terraform plan` shows no changes.

**Expected alarms.** From `terraform apply` until the first deploy is healthy (section 5),
H1 and H3 are in ALARM (nothing is scraped and nothing answers on `hooks.`), and so are
the other alarms whose missing data is breaching and whose metric only the running stack
publishes (P3, G3, S5). That is expected: acknowledge those mails and check that each
sends its OK mail after the first deploy. One still in ALARM then is a real fault.

## 3. DNS, SES, Stripe, Google (B3, B7, B24)

- **DNS** (B3): if `route53_zone_id` was set Terraform wrote the records; otherwise create
  the ones `terraform output` lists at your DNS host: A records for the apex, `api` and
  `hooks` to the Elastic IP, the CAA record, the SES DKIM CNAMEs, MAIL FROM and DMARC.
  Check: `dig +short <domain> api.<domain> hooks.<domain>` all give the Elastic IP.
- **SES**: DKIM verifies in minutes to 72 h. Request production access in the SES console
  (use case: transactional sign-in mail). Until it is granted, only verified addresses
  receive sign-in mail (B24 uses one).
- **Stripe**: a sandbox used only by this box. Create a **restricted** key `rk_test_...`
  with *Checkout Sessions: write* only, and the webhook endpoint
  `https://hooks.<domain>/stripe/webhook` with the five event types of spec §8.2 at the
  sandbox's default API version; note its signing secret `whsec_...`. Details and the
  resend procedure: [runbook-stripe.md](runbook-stripe.md). The endpoint may be registered
  before the box exists (Stripe retries until it answers).
- **Google** (optional; the web app runs without it): an OAuth client with redirect URI
  `https://<domain>/api/auth/callback/google`.
- The Stripe sandbox's public business name is set in the Stripe dashboard.

## 4. Secrets (B4)

All values live in Secrets Manager as `loom/platform/<name>`; none is ever on a command
line, in a compose file, in the release manifest or in a log. The instance role cannot read
`serving-runpod-key` (explicit deny); `loom-deploy` never names it.

Run on the box through SSM. `loom-deploy.py` is the module itself, copied into every
release; the first time, take it from the releases bucket (after section 5's image build).
Run it with `python3.12 -S` (the boot script installs 3.12; the host's `python3` is the
older system Python). Only the secrets' creation uses Run Command here (Run Command keeps
its parameters in history, so no secret value goes through it); everything that reads a secret value runs in an interactive session on the box
(`aws ssm start-session --region $AWS_REGION --target $INSTANCE`, then `sudo -i`), because
each secret's policy lets only its reader role (the instance role) read it: your own
principal cannot, by design.

```sh
aws ssm send-command --region $AWS_REGION --instance-ids $INSTANCE \
  --document-name AWS-RunShellScript --comment "loom-deploy secrets init" \
  --parameters 'commands=["aws s3 cp s3://'$BUCKET'/releases/'$SHA'/loom-deploy.py /var/lib/loom/loom-deploy.py","python3.12 -S /var/lib/loom/loom-deploy.py secrets init"]'
aws ssm get-command-invocation --region $AWS_REGION --instance-id $INSTANCE --command-id <id>
```

1. `secrets init` generates `internal-token`, `app-db-password`, `auth` (`auth_secret`),
   `redis` (password, CA, server certificate with SAN `redis`), `pod-ca` (2 years),
   `poolmgr-client`, `grafana`. It prints which it generated and which already had a value
   (never a value) and never overwrites one. **Run it before you put Google values into
   `auth`**: a secret that already has a value is left whole, so an `auth` holding only the
   Google keys would lack `auth_secret` and `apply` would stop naming it.
2. Put the Stripe values from **your laptop**, not Run Command and not the box (your
   principal may `PutSecretValue`; the secrets' policies deny only `GetSecretValue` to
   everyone but the reader role). Read both at the prompt (not echoed, not in shell
   history, not on a command line), build the JSON with `jq`, and pipe it on stdin. Anyone
   holding the webhook secret can forge credit events, so keep it out of history:

   ```sh
   read -rs SK   # paste the rk_test_ key, press Enter
   read -rs WH   # paste the whsec_ signing secret, press Enter
   jq -n --arg secret_key "$SK" --arg webhook_secret "$WH" \
     '{secret_key: $secret_key, webhook_secret: $webhook_secret}' \
     | aws secretsmanager put-secret-value --region $AWS_REGION \
         --secret-id loom/platform/stripe --secret-string file:///dev/stdin
   unset SK WH
   ```

   Google: merge `google_id` and `google_secret` into the `auth` JSON **on the box**, in an
   interactive session (the secret's policy denies reading it from your laptop; the
   instance role may read and put it). The values are typed at the prompt, not put on a
   command line:

   ```sh
   aws ssm start-session --region $AWS_REGION --target $INSTANCE
   # on the box:
   sudo -i
   python3.12 -S - <<'EOF'
   import getpass, json, subprocess
   def aws(*a, **k):
       return subprocess.run(["aws", "secretsmanager", *a], check=True, text=True, **k)
   cur = json.loads(aws("get-secret-value", "--secret-id", "loom/platform/auth",
                        "--query", "SecretString", "--output", "text",
                        stdout=subprocess.PIPE).stdout)
   cur["google_id"] = getpass.getpass("google_id: ")
   cur["google_secret"] = getpass.getpass("google_secret: ")
   aws("put-secret-value", "--secret-id", "loom/platform/auth",
       "--secret-string", "file:///dev/stdin", input=json.dumps(cur))
   EOF
   ```
3. **SES SMTP**: two steps, because only your own (operator) credentials may call IAM; the
   instance role has no `iam:` action.
   a. Get `loom-deploy.py` onto your laptop (any Python 3.9 or later; it uses only the
      standard library):

      ```sh
      aws s3 cp s3://$BUCKET/releases/$SHA/loom-deploy.py ./loom-deploy.py
      ```

   b. On your laptop, pipe the new access key for the IAM user `loom-ses-smtp` straight
      into the derivation, so the secret is never printed, pasted, put in a file or in
      the clipboard (it is read from stdin, not a command-line argument):

      ```sh
      aws iam create-access-key --user-name loom-ses-smtp \
        | AWS_REGION=$AWS_REGION python3 -S loom-deploy.py ses-smtp-password
      ```

      If the pipe fails after the key was created (no `python3`, `PutSecretValue` denied,
      wrong directory), the new key's secret is lost: delete that key
      (`aws iam list-access-keys --user-name loom-ses-smtp`, then
      `aws iam delete-access-key --user-name loom-ses-smtp --access-key-id <id>`) and run
      the pipe again (IAM allows two keys per user).

   Why on the laptop: `ses-smtp-password` needs only the `aws` CLI, Python 3.9+ and
   `secretsmanager:PutSecretValue` on `loom/platform/ses-smtp` with your credentials (the
   secret's policy denies only `GetSecretValue` to you); it never reads the box's paths.
   `loom-deploy` prints neither the access key secret nor the derived password. After a
   rotation, delete the IAM user's old access key.
4. Check (B4): each of the nine containers `describe-secret` shows a version. The user's
   principal cannot read them: only the instance role can.

## 5. Images and the first deploy (B5, B6)

**Images (B5).** Run the `images` workflow (Actions, *Run workflow*, branch `main`). It
builds the six Loom images for arm64, pushes them to ECR tagged with the git sha, and
uploads to `s3://$BUCKET/releases/`: `<sha>.json` (the manifest: image digests, migration
count, the compose file's and the bundle's sha256), `<sha>/compose.prod.yml`,
`<sha>/deploy.tar.gz` (the `deploy/` directory) and `<sha>/loom-deploy.py`. Compose uses
digests only. The Caddy image is built from `deploy/caddy/` (`go.mod`/`go.sum` pin Caddy
and the rate-limit module).

**Run loom-deploy: in an interactive session.** `apply`, `rollback` and `acceptance` take
minutes, and `acceptance` stops for operator input (m3-g, m3-s). Run Command shows no
output until the command ends, and `aws ssm wait command-executed` gives up after about
100 seconds, so none of the three can be driven that way. Open a session and define a
helper (the commands below that say `loom_deploy` mean this one, in the session):

```sh
aws ssm start-session --region $AWS_REGION --target $INSTANCE
# on the box (the session user is ssm-user; sudo gives the root the secrets and docker need):
export AWS_REGION=us-east-1 BUCKET=<releases bucket> SHA=<the release's sha>
sudo aws s3 cp s3://$BUCKET/releases/$SHA/loom-deploy.py /var/lib/loom/loom-deploy.py
loom_deploy() { sudo env AWS_REGION=$AWS_REGION ${LOOM_DEPLOY_ACME_STAGING:+LOOM_DEPLOY_ACME_STAGING=1} \
  python3.12 -S /var/lib/loom/loom-deploy.py "$@"; }
```

A session ends if the connection drops: run `tmux` first if it might (it is not
installed by default), or re-run `apply` (it is repeatable). For a short subcommand from
your laptop (`status`, `secrets init`) Run Command works; poll it, with an explicit
execution timeout:

```sh
loom_deploy_remote() {  # usage: loom_deploy_remote <loom-deploy arguments...>, short ones only
  id=$(aws ssm send-command --region $AWS_REGION --instance-ids $INSTANCE \
    --document-name AWS-RunShellScript --comment "loom-deploy $1" --timeout-seconds 60 \
    --parameters "executionTimeout=[\"600\"],commands=[\"aws s3 cp s3://$BUCKET/releases/$SHA/loom-deploy.py /var/lib/loom/loom-deploy.py\",\"python3.12 -S /var/lib/loom/loom-deploy.py $*\"]" \
    --query Command.CommandId --output text) || return 1
  case $id in ""|None) echo "send-command returned no command id" >&2; return 1;; esac
  n=0
  while :; do
    st=$(aws ssm get-command-invocation --region $AWS_REGION --command-id $id \
      --instance-id $INSTANCE --query Status --output text 2>/dev/null) || st=Pending
    case $st in Pending|InProgress|Delayed) sleep 5;; *) break;; esac
    n=$((n + 1))
    if [ $n -ge 130 ]; then  # about 11 minutes: past the 600 s execution timeout
      echo "command $id still $st after $n polls; check it with get-command-invocation" >&2
      return 1
    fi
  done
  aws ssm get-command-invocation --region $AWS_REGION --command-id $id --instance-id $INSTANCE \
    --query '[Status,StandardOutputContent,StandardErrorContent]' --output text
  [ "$st" = Success ]  # the helper's exit status is the remote command's outcome
}
```

**First issuance (staging certificates).** Let's Encrypt's production CA rate-limits
failures, so the first deploy gets staging certificates, then production. In the session:

```sh
LOOM_DEPLOY_ACME_STAGING=1 loom_deploy apply --release $SHA
```

With that variable `apply` renders `LOOM_ACME_CA` (the staging directory) into
`env/caddy.env`. Check with `curl -vk https://<domain>/` that the certificate is issued by
the staging CA (browsers do not trust it). Then apply the same release **without** the
variable (`loom_deploy apply --release $SHA` in a shell where it is unset): Caddy requests
production certificates (they are kept on the data volume, per CA).

**`apply` order** (stops at the first failure; every step is a line of
`/var/lib/loom/deploy.log`, JSON Lines):

1. fetch the manifest and files, check both sha256s and that the bundle's
   `compose.prod.yml` is the named one; extract to `/var/lib/loom/releases/<sha>/`;
2. read the secrets and the site parameter with the instance role; render
   `/run/loom/compose.env`, `env/<service>.env` (0400) and `secrets/<service>/` (0500
   directories, 0400 files owned by the service's uid) on the tmpfs; every database URL is
   checked for `sslmode=verify-full` and the RDS CA file (D10);
3. ECR login (once per registry the images name), `docker compose pull`;
4. `migrate` as the RDS owner `loom_owner` (creates `loom_platform` on the first deploy,
   applies the migrations, creates or updates `loom_app`); exit 0 required;
5. write `/var/lib/loom/apply-in-progress.json` (the marker of an unfinished apply), then
   `up -d --no-deps <service>` then wait healthy, in the order **redis** (only if its
   image changed, or the marker was already there), **billing**, **gateway**, **poolmgr**,
   web, caddy, prometheus, grafana; a service is healthy when `docker inspect` says
   `healthy` (or `running` with no healthcheck) within `LOOM_DEPLOY_HEALTH_TIMEOUT_S` (300);
6. record the release in `/var/lib/loom/release.json`, then remove the marker.

Exit codes: 0 done, 2 refused (the reason on stderr, nothing changed: not on a tmpfs, the
site parameter lacks a key, the P4 rule, `--only` while the marker is set), 1 any other
failure. Rendered files stay after a failed step; the next apply renders them again.

**After a failed apply.** An apply that stops during step 5 (or is cut off during it: a
dropped session, a reboot) leaves the marker, and `release.json` still names the release
from before it, though some services may already run the new one. `loom_deploy status`
shows the marker on its first line. Until a full apply succeeds, every `apply --only` and
`loom_deploy acceptance` are refused (exit 2, naming the marker). Re-apply the current
release, the `recorded:` one on `status`'s first line (also its `release` line):
`loom_deploy apply --release <current>` (all services, no `--only`), or apply the new
release again once its problem is fixed. If `status` says no release is recorded (the first
apply failed), apply that release again. Do not use `rollback` for this: it goes to the
release before the current one (`previous`), not back to the current one. Do not delete the
marker by hand. Run these with a `loom-deploy.py` that knows the marker: the failed
release's (or a later one's). Both helpers run the copy fetched for `$SHA`, and
`loom_deploy_remote` fetches it again on every call, so keep `SHA` at the failed release,
in the session and on your laptop, until the full apply succeeds. A copy from a release
before this rule neither refuses `--only` nor removes the marker.

**CloudWatch agent.** Install its config on the box once, after the first apply (both files
are in the extracted release bundle), then start it; do it before the B6 check, because
that check waits for the "OK" mails of alarms that only the agent's series can clear (H1,
P3, G3, S5). In the session:

```sh
rel=/var/lib/loom/releases/$SHA
sudo cp $rel/cloudwatch-prometheus.yml /opt/aws/amazon-cloudwatch-agent/etc/loom-prometheus.yml
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 \
  -c file:$rel/cloudwatch-agent.json -s
```

The agent scrapes `127.0.0.1:9100-9102` and sends only the series spec §5.3 names, plus
disk (one series per `path`, no host or device dimension) and memory, in namespace
`Loom/Platform` through the log group `/loom/platform/prometheus`.

**B6 check.** `loom_deploy status`: every service healthy, `loom_app` created. No
`LOOM_ENV` in any container, printing only the variable names (the values are secrets):

```sh
for c in $(sudo docker ps -q); do
  sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $c | cut -d= -f1
done | sort -u | grep -x LOOM_ENV || echo "LOOM_ENV: absent (ok)"
```

From the internet: `https://<domain>/signin` shows the product name in its title and
heading. Then, with the agent running, each alarm that began in ALARM (section 2) sends
its OK mail.

## 6. The P4 rule, rollback, changing the name or domain

**P4.** The gateway is updated before the pool manager in every release (the order above
and `poolmgr`'s `depends_on: gateway: service_healthy`). `loom-deploy` refuses any apply
that would leave the poolmgr on another release than the gateway, `apply --only poolmgr`
included (exit 2, naming the gateway). To move both: `apply --release <sha>` (all), or
`apply --release <sha> --only gateway --only poolmgr`.
While `/var/lib/loom/apply-in-progress.json` exists (an apply that did not finish), every
`--only` apply is refused, whatever it names: run a full apply first (section 5, "After a
failed apply").

**Migrations are forward-only and must be backward compatible** with the running release
(add, never drop or rename in the same release).

**Rollback.** `loom_deploy rollback` applies the previous release recorded in
`release.json` (`previous`) with the same order, as a full apply: the apply-in-progress
marker does not stop it, and it clears the marker when it succeeds. It is not the way back
from a failed apply: re-apply the current release instead (section 5, "After a failed
apply"). A rollback across a migration relies on the
backward-compatible rule. A rollback to a pre-M3 release is a last resort for billing: M1's
`retry_unapplied` catches only `BadEvent`, so a stored poison event fails its job every
minute and starves the rest, and M1 has no sweep that reopens a clawback an admin resolved
when its purchase arrives later. Before such a rollback, list both kinds: resolved
clawbacks whose purchase may still come (note them for a hand check after any late credit)
and each poison event (resolve it, or leave it for M1 only after checking M1's handler
applies it; otherwise stay on M3 and fix forward). Never make a hand adjustment for a
resolved clawback: the reopen sweep would take it twice.

**Changing the name or the domain.** Edit `terraform.tfvars` (`brand_name`, `domain`),
`terraform apply` (it rewrites the SSM parameter `/loom/platform/site`), then
`loom_deploy apply --release $SHA` (session helper, section 5) with the same release: it re-renders every service's
settings and updates every service but Redis. A new domain also needs DNS, the SES
identity, the Stripe endpoint URL and the Google redirect URI (section 3).

## 7. After a reboot or an EC2 auto-recovery

A reboot or an auto-recovery (alarm H2) empties `/run/loom`: the services come back
without their secret files and H1 and H3 fire. The step is
`loom_deploy apply --release <current sha>` (in a session, section 5) for the current release (the `release` in
`/var/lib/loom/release.json`). A same-release apply re-renders everything, recreates a
Redis that is not healthy and updates every other service. If the reboot cut off an apply, this same full apply also
clears its marker. M3 has no boot unit (LM10).

## 8. Alarm test (m3-a)

The CloudWatch agent was installed in section 5. For each alarm: `aws cloudwatch set-alarm-state --alarm-name <name> --state-value ALARM
--state-reason test` and check the mail; the alarm re-evaluates on its own. Every §5.3
metric, `up` included, must be present in CloudWatch (`aws cloudwatch list-metrics
--namespace Loom/Platform`). Whether the agent publishes `up` and the empty dimension sets
is not checked before this step.

**H3 end to end (B25).** With the stack up, stop Caddy (`sudo docker stop <the caddy container>` in a session): H3's
ALARM mail arrives (record the minutes); start it: the OK mail arrives.

## 9. Looking at it

- Grafana (not public): `aws ssm start-session --target $INSTANCE --document-name
  AWS-StartPortForwardingSession --parameters '{"portNumber":["3001"],"localPortNumber":["3001"]}'`,
  then `http://127.0.0.1:3001`; user `admin`. The password is `admin_password` in
  `loom/platform/grafana`, which your own principal cannot read (only the instance role
  can): read it in an interactive session on the box, not from your laptop:
  `sudo aws secretsmanager get-secret-value --region $AWS_REGION --secret-id
  loom/platform/grafana --query SecretString --output text` (a JSON object; take the
  `admin_password` value, and clear the terminal afterwards).
- Prometheus the same way on 9090.
- Logs: CloudWatch log group `/loom/platform` (30 days; the `awslogs` driver). No prompt,
  completion or key ever reaches them: the containers' logs are scanned by m3-i, and the
  CloudWatch log group is checked by B26 (by hand, with operator credentials).

## 10. M3 acceptance (B10-B19, B26)

On the box, in the one-shot `acceptance` container and m3-p's host half. Run it in an
interactive session (section 5's helper) so the output streams and the operator prompts
of m3-g and m3-s reach you; Run Command cannot do either:

```sh
loom_deploy acceptance --p4-release <any release but the current one>
```

`loom-deploy acceptance` runs `loom-m3-acceptance` in the compose network through the
public URLs; for m3-i the container scans the containers' logs only. Then, on the host
with the instance role, it runs m3-p's refusal (`apply --only poolmgr --release <the
other release>` exits 2 naming the gateway, and the poolmgr container is not recreated).
It writes `/var/lib/loom/acceptance/report.json` and `host.json` (`{"m3-p": {"ok",
"detail"}}`): `ok` is `true` or `false` when the host half ran, and `null` with detail
"not selected" when `-- --only` leaves m3-p out or `-- --external` is given; exit 0 only
when the container passed and m3-p passed or was not selected. A run with `--only` is not
B10-B19 evidence; one full run (no `--only`) is. It calls no `aws logs`
command: m3-i's CloudWatch half needs `logs:FilterLogEvents`, which no Loom role has, so
B26 runs it by hand. Manual items (m3-g, m3-s) wait for the operator (Stripe test card
4242 4242 4242 4242, resend, refund). m3-h runs from your machine:
`uv run loom-m3-acceptance --external --domain <domain> --brand-name <name>`. Results go to
`docs/phase1-m3-acceptance.md`.
`loom_deploy acceptance` is refused (exit 2, nothing run, the old `report.json` and
`host.json` left as they are) while `/var/lib/loom/apply-in-progress.json` exists (section
5, "After a failed apply"), and when a word after `--` asks for help (`-h`, `--help`,
`-hh`, ...): for `loom-m3-acceptance`'s options run `uv run loom-m3-acceptance --help` in a
checkout of the release. Do not pass, after `--`, the options `loom-deploy acceptance` sets
itself: `--domain`, `--release`, `--json`, `--deploy-log`, `--admin-email` and
`--webhook-secret-file`.

B26 runs m3-i's CloudWatch half (the only place it is checked) and repeats m3-p
independently, by hand with your own credentials and plain `aws` commands, not through
`loom-deploy acceptance` (spec §13).

## 11. Backups, restore drill, teardown (B21, B22)

RDS: automated backups (7 days) and point-in-time recovery. Data volume: daily EBS
snapshots, 7 kept. **Teardown:** follow `infra/aws/platform/README.md`, "Teardown", steps
0 to 8, in order; a plain `terraform destroy` fails on purpose (data volume, instance
and database protections, non-empty ECR repositories and buckets). Step 0 is on the box,
in a session: `sudo docker compose -p "${LOOM_DEPLOY_PROJECT:-loom}" down` and `sudo
systemctl stop amazon-cloudwatch-agent` (Alarms and the H3 health check then fire; expect
the mails). RDS takes a final snapshot; secrets are deleted with the 7-day recovery
window. List what remains by the `loom:stack` tag (`aws resourcegroupstaggingapi
get-resources`).

## 12. Facts about this deploy that surprise people

- Containers have no AWS credentials (IMDSv2 hop limit 1): only host processes
  (`loom-deploy`, the Docker `awslogs` driver, the CloudWatch and SSM agents) use the
  instance role.
- `LOOM_ENV` is unset everywhere, so every service runs its production checks. The
  services' own checks are not extended; `loom-deploy` adds the `verify-full` and the
  domain and name checks.
- The poolmgr runs the **mock** backend (`LOOM_POOLMGR_MOCK_BACKEND=true`,
  `deploy/serving.m3.yaml`): no RunPod call is made in M3.
- The gateway's client certificate is issued by the pool manager into
  `/var/lib/loom/gateway-cert` (shared read-only with the gateway) and rotated daily.
- Redis is TLS only with a password; the services' URLs are `rediss://` with
  `ssl_ca_certs` naming the CA file `loom-deploy` renders.
- The compose project is `LOOM_DEPLOY_PROJECT` (default `loom`), always passed with `-p`;
  there is no top-level `name:` in `deploy/compose.prod.yml`.
