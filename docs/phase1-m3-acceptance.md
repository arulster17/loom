# Phase 1 M3 acceptance

M3 is the AWS control plane: the web app, gateway, billing and pool manager on one box
behind Caddy, a Postgres on RDS, Terraform for all of it, deploy tooling, Stripe test-mode
webhooks and the pod PKI ([phase1-m3-spec.md](phase1-m3-spec.md); plan §13 in
[phase1-runpod-plan.md](phase1-runpod-plan.md)). It has two stages:

- **Stage A** is free and offline: code, tests, runbooks and Terraform checked without
  AWS credentials. It is built and recorded below.
- **Stage B** spends money: `terraform apply`, DNS, the deploy and the live acceptance
  (B1-B26), then teardown or keep. It needs the user's separate yes with its cap (spec §14:
  $25 over a 14-day window, inside a $75 a month AWS budget). It has **not run**. Its table
  at the end is empty.

This note records the Stage A results and holds the places for Stage B. The final
verification at the integrated branch head and CI on the PR come after this note. They are
not claimed here.

## How to reproduce (free)

The CI commands (root `CLAUDE.md`), from the repository root:

```bash
uv run ruff format --check . && uv run ruff check .
uv run mypy services/platform_common/src services/gateway/src services/billing/src
uv run mypy services/platform_common/src services/podproxy/src services/devstack/src
uv run mypy services/platform_common/src services/poolmgr/src
uv run pytest -q -n auto
```

Postgres and Redis tests skip silently when no server is reachable (`docker compose up -d
postgres redis`), so check the `s` count. Gateway tests need Linux (WSL, Docker or CI).

Terraform needs no credentials. CI job `terraform` runs this in `infra/aws/platform` and in
`infra/aws/state`, with Terraform 1.16.7 (the version and image digest are pinned in
`.github/workflows/ci.yml`):

```bash
terraform fmt -check -recursive -diff
terraform init -backend=false -input=false -lockfile=readonly
terraform validate
terraform test        # under mock_provider: no AWS call
```

The same job then runs `tflint` (config `.github/terraform/tflint.hcl`) and `trivy config`
with `.github/terraform/trivyignore.yaml`, both as pinned images. Every accepted trivy
finding has its reason in `infra/aws/platform/README.md`.

The production stack on one machine (A15; spec §13, §6.12) is CI job `m3-prod-stack`. It
needs Linux (the harness uses `/dev/shm`), Docker with compose 2.24 or later, and the
network (the third-party images are pulled):

```bash
uv sync --frozen
uv run pytest -q -m acceptance --timeout 3000 services/devstack/tests/test_devstack_m3_prodstack.py
```

The test builds the Loom images under local tags, runs `loom-deploy secrets init` and
`apply` against a fake `aws`, brings up `deploy/compose.prod.yml` as project `loom-m3prod`
(a local Postgres with TLS stands in for RDS, Caddy uses its internal CA, the domain is
`example.test`), checks the production-mode services and the Caddy routes, runs M1's items
0, b and d as m3-0, m3-b and m3-d in the `acceptance` service, and removes the project
itself. No AWS or Stripe call, no spend. Do not run anything else from the runbooks
without the user's go-ahead: `terraform apply`, `loom-deploy` against a real account and
the live acceptance all cost money.

## Stage A results (orchestrator log, 2026-11-19)

Four areas were built on their own branches, then merged into `phase1-m3` (head 66a8509
at the time of this note). Results per area:

| Area | Commit | Result |
|---|---|---|
| Stripe test mode (webhooks, resolve, migration 0012) | 4432a23 | billing and `platform_common`: 651 pass, 2 skip (the auth-method skips), 0 fail. `ruff` and the CI `mypy` line clean. Hidden tests 29/29 (A16-A20, A22-A24, A31-A33) |
| Pod PKI (CA expiry gauge, client-cert rotation, `rekey`, `new-ca`) | 757b111 | poolmgr and gateway: 447 pass, 0 skip, 0 fail. adf7240 after it changed only the runbook wording |
| Deploy (`loom-deploy`, `deploy/compose.prod.yml`, Caddy, `loom-m3-acceptance`) | 52ea0cd | devstack alone: 423 pass. 34 more failures need the other branches (cross-area), so they are not counted as results |
| Terraform (`infra/aws/platform`, `infra/aws/state`) | 2a52a13 | platform `terraform test` 59/59, state 0/0. `fmt`, `init`, `validate` ok. `tflint` exit 0. `trivy` exit 0 with `.github/terraform/trivyignore.yaml` (13 accepted findings, reasons in `infra/aws/platform/README.md`, LM11) |

On the combined tree (stripe 4432a23, pki 757b111, deploy 52ea0cd, infra 2a52a13):

| Check | Result |
|---|---|
| A9 set (metric names agree with the alarms and the agent allowlist) | 69 pass, 0 skip, 0 fail |
| devstack | 468 pass, 0 skip, 0 fail |
| A15 (the production stack locally, `test_devstack_m3_prodstack.py`) | 19 of 19 pass, in 217 s |
| Gate-style suite | 1891 pass, 2 known skips, 0 fail |

Flake recipe (10 sessions of `-n 4 --count 2` under CPU load) on the new and changed test
files: 0 flaky.

Not checked here: `terraform plan` against the real account (A28, needs read-only AWS
credentials from the user), and the runbooks' commands on a real box (Stage B).

## Known gaps accepted for M3

- **The pod CA rollover is not zero downtime** (spec §7; user, 2026-11-19). From the deploy
  that switches the CA, requests can fail until the pool manager has re-keyed every pod and
  the gateway has loaded its new certificate (minutes, longer with more pods). The runbook
  `docs/runbook-pod-ca.md` says so, and B20 records the outage. The fix is M4-1 (plan §13,
  "M4 prerequisites from M3").
- **m3-i's CloudWatch half** is not run by `loom-deploy acceptance` (no Loom role may read
  logs). It is B26, by hand with the operator's credentials.
- **Live mode** needs LM1-LM11 first (plan §13, "Live-mode prerequisites"; spec §8.13).
  Phase 1 stays in Stripe test mode.

## Stage B (not run yet)

Stage B needs the user's yes and cap (spec §14) before anything is applied. Every cell
below is filled in when the item runs: the result, the numbers, the release sha, and
whether it was automated or manual and who did it.

| Item | What (spec §13, §10) | Result | Numbers | Release sha | Automated or manual, and who |
|---|---|---|---|---|---|
| B1 | `infra/aws/state` applied, `backend.hcl` written | not run | not run | not run | not run |
| B2 | `terraform apply` of `infra/aws/platform`; budget and SNS subscription confirmed | not run | not run | not run | not run |
| B3 | DNS records resolve; SES identity verified | not run | not run | not run | not run |
| B4 | `loom-deploy secrets init`; every secret has a value | not run | not run | not run | not run |
| B5 | Images built by the workflow, pushed by digest | not run | not run | not run | not run |
| B6 | First deploy, every service healthy in production mode | not run | not run | not run | not run |
| B7 | Stripe sandbox: restricted key, `expires_at`, retry schedule recorded | not run | not run | not run | not run |
| B10-B19: m3-0 | M1's flow through the public URLs | not run | not run | not run | not run |
| B10-B19: m3-b | Revoked key refused within 5 s | not run | not run | not run | not run |
| B10-B19: m3-d | Gateway added overhead p50 under 10 ms; Caddy + TLS hop reported | not run | not run | not run | not run |
| B10-B19: m3-g | Stripe checkout, duplicate, old `t`, resend, refunds | not run | not run | not run | not run |
| B10-B19: m3-s | Missed delivery and resend | not run | not run | not run | not run |
| B10-B19: m3-h | Exposure (`--external`) | not run | not run | not run | not run |
| B10-B19: m3-i | No prompt, completion or key in container logs | not run | not run | not run | not run |
| B10-B19: m3-p | P4: gateway healthy before poolmgr; older gateway refused | not run | not run | not run | not run |
| B10-B19: m3-x | Containers cannot reach the instance metadata service | not run | not run | not run | not run |
| m3-a | Alarm routes: `set-alarm-state` on each alarm; each metric present | not run | not run | not run | not run |
| B20 | `docs/runbook-pod-ca.md` executed once under load; outage recorded | not run | not run | not run | not run |
| B21 | (optional) RDS point-in-time restore | not run | not run | not run | not run |
| B22 | Teardown or keep (D8) | not run | not run | not run | not run |
| B23 | Webhook secret roll (T15) | not run | not run | not run | not run |
| B24 | Magic-link and Google sign-in by hand | not run | not run | not run | not run |
| B25 | H3 end to end: stop Caddy, ALARM and OK emails | not run | not run | not run | not run |
| B26 | m3-i's CloudWatch half and m3-p's refusal re-run with plain `aws` | not run | not run | not run | not run |
