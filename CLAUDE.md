# CLAUDE.md

## Branches and docs
- Phase 1 M1 (the platform on the mock) and M2 (the pool manager, offline) are merged into `main`, with the before-M3 review fixes (`docs/m3-review-fixes-spec.md`). M3 (the AWS control plane; spec `docs/phase1-m3-spec.md`, results `docs/phase1-m3-acceptance.md`): Stage A (free) is merged into `main` (PR #15), and so are the post-M3 follow-ups: the deploy apply-in-progress marker (PR #16), the g2a cap test's setup retry (PR #17), the split flake-check (PR #18) and TensorRT-LLM's removal from the bench (PR #20). Still open: A28 (`terraform plan` with the user's read-only credentials). Stage B (`terraform apply`, DNS, deploy, live acceptance, teardown) has not run and needs the user's separate yes with its cap. Then M4 (prerequisite M4-1, plan §13). Plan: `docs/phase1-runpod-plan.md`; overall plan and user decisions: `docs/PLAN.md`.
- Interfaces between platform services are fixed in `docs/phase1-contracts.md`. Changing one means changing that doc, `services/platform_common` (`loom_platform`) and its tests in the same change.

## Python (uv workspace; members in root `pyproject.toml`)
- Always go through uv: `uv sync`, `uv run <tool>`. Add a dependency only to the member package that needs it.
- Tests: `uv run pytest -q` (`-n auto` for parallel, as CI does). Pytest settings (testpaths, markers, default `-m`, 120 s timeout) live in `pytest.ini`.
- `network` and `acceptance` are deselected by default; opt in with `-m network` / `-m acceptance` (a later `-m` replaces the default one). Both take minutes; `network` downloads wheels and datasets.
- Postgres/Redis tests skip silently when no server is reachable, so a green run can hide them (check the `s` count). Platform tests use `LOOM_TEST_DATABASE_URL` / `LOOM_TEST_REDIS_URL`, else the compose servers: `docker compose up -d postgres redis`. Bench `-m postgres` tests run only when `LOOM_TEST_DATABASE_URL` is set (e.g. `postgresql+psycopg://loom:loom@localhost:5432/loom`). Setting the variable turns an unreachable server into a failure.
- Test files are named `test_<svc>_*.py` with no `__init__.py`: pytest imports them by base name, so names must be unique across the repo. Reuse `loom_platform.testing` fixtures (`pg`, `pg_url`, `redis_client`).
- `uv sync --all-packages --extra lmeval` adds the lm-eval extra; a later plain `uv sync` removes it.

## Checks CI runs (`.github/workflows/ci.yml`)
```
uv run ruff format --check . && uv run ruff check .
uv run mypy bench/src
uv run mypy services/platform_common/src services/gateway/src services/billing/src
uv run mypy services/platform_common/src services/podproxy/src services/devstack/src
uv run mypy services/platform_common/src services/poolmgr/src
uv run pytest -q -n auto   # includes the pool manager's tests (services/poolmgr/tests)
```
Job `podproxy-bundle` (needs the network) rebuilds the pinned pod proxy bundle and checks it against `config/serving-artifacts.lock`:
```
uv run loom-podproxy-bundle --check --out "$RUNNER_TEMP/podproxy-bundle"
uv run pytest -q -m network services/podproxy/tests/test_podproxy_bundle.py services/podproxy/tests/test_podproxy_bundle_dirty.py
```
Job `acceptance` runs `docker compose up -d --build --wait`, then `uv run loom-m1-acceptance` and `uv run loom-m2-acceptance` (M2's takes ~13 min, mostly the idle-stop wait). The job sets `LOOM_POOLMGR_IDLE_STOP_S=300`; compose's default stays 600.

Job `terraform` (no AWS credentials; needs the network and docker) runs for each of `infra/aws/platform` and `infra/aws/state`, with Terraform 1.16.7 (pinned in the job's `env`; locally, run the `hashicorp/terraform:1.16.7` image, digest in the comment above the job):
```
terraform fmt -check -recursive -diff
terraform init -backend=false -input=false -lockfile=readonly
terraform validate
terraform test        # mock providers; infra/aws/platform must have tests/*.tftest.hcl
```
then fails on inline `trivy:ignore`/`tfsec:ignore`/`tflint-ignore` comments (accept a trivy finding only in `.github/terraform/trivyignore.yaml`, with its reason in `infra/aws/platform/README.md`), and runs `tflint` (config `.github/terraform/tflint.hcl`) and `trivy config` from digest-pinned docker images. Each stack's `.terraform.lock.hcl` must hold the hashes of every platform (`terraform providers lock -platform=linux_amd64 -platform=linux_arm64 -platform=windows_amd64 -platform=darwin_arm64`), or `init` passes in CI and `validate` fails.

Job `m3-prod-stack` (free; needs Linux, Docker with compose 2.24+ and the network) starts the production compose stack (`deploy/compose.prod.yml`, local images, TLS Postgres, throwaway CA) as project `loom-m3prod` and checks M1 acceptance items 0, b and d against it. The test starts and removes the stack itself:
```
uv run pytest -q -m acceptance --timeout 3000 services/devstack/tests/test_devstack_m3_prodstack.py
```

Workflow `images` (`.github/workflows/images.yml`, not part of `ci`, manual dispatch only) builds the arm64 images, pushes them to ECR and uploads the release to the releases bucket, using AWS credentials from OIDC (environment `platform-images`, `main` only). It is Stage B (item B5): never dispatch it without the user's yes.

Flaky tests (plan R15). Don't re-run CI to look for flakes:
- On PRs, the test files the PR adds or changes run 20 times under CPU load. That is 10 parallel jobs `flake-session (1..10)`, each one pytest session with `-n auto --count 2` (pytest-repeat) and a 7-minute limit. Separate sessions are needed because pytest-repeat's copies share module-scoped fixtures. Job `flake-check` then reads all their JUnit files. It fails if a test both passed and failed (`scripts/flaky_report.py`), if a session job failed, or if no session wrote a JUnit file. A session that hits its limit only warns: the report lists the tests that ran fewer than 20 times. Both are skipped when no test file changed.
- Workflow `flake-hunt` (`.github/workflows/flake-hunt.yml`, Mondays, or run it by hand) runs the full suite 3 times in shuffled order and opens or updates the issue labelled `flaky-tests`. It adds pytest-randomly with `uv run --with`. Never make pytest-randomly a dependency: once installed, it shuffles every run.
Web app, from `apps/web`: `npm ci`, `npm run lint`, `npm run typecheck`, `npm test`, `npm run test:integration` (real Postgres, migrated with `uv run loom-migrate`), `npm run build`.

## Native Windows
- Code is POSIX-first (`fcntl`, `os.killpg`, signals). A bare `uv run pytest` fails at collection: the gateway's conftest imports `fcntl`. Run gateway tests in Linux (WSL, Docker or CI); elsewhere pass the other test dirs explicitly, e.g. `uv run pytest -q bench/tests services/billing/tests`.
- mypy reports false `fcntl`/`signal` errors on Windows. Use `uv run mypy --platform linux ...` to match CI.

## Web app (`apps/web`)
- Next.js 16 differs from your training data. Read `apps/web/AGENTS.md` and the docs in `node_modules/next/dist/docs/` before writing web code. `next dev` re-adds that AGENTS.md block, so don't remove it.
- Local `npm run dev` needs `LOOM_ENV=dev` and `AUTH_SECRET` (see `apps/web/README.md`). The schema comes from `uv run loom-migrate --create-database`. The web app has no ORM.

## Spending money: ask first
- `bench run` / `bench reproduce` on RunPod or AWS specs, and `bench reap` without `--dry-run`, start or stop paid cloud resources. Never run them without the user's explicit go-ahead for that run and cap. `bench plan <spec>` only estimates spend, and everything under `bench/experiments/mock-*` is free.
- Before any real sweep, run the smoke spec on the same commit with a clean git tree (`docs/runbook-runpod.md`, "Smoke test"; AWS: `docs/runbook.md`). If `bench plan` refuses a plan (exit 3), lower the load points or TTL, never the caps.
- The whole local stack (`docker compose up -d --build`, then `uv run loom-m1-acceptance`) is free: it uses mock pods and a Stripe stub.
