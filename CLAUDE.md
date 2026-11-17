# CLAUDE.md

## Branches and docs
- Phase 1 M1 (the platform on the mock) is merged into `main`; M2 is next. Plan: `docs/phase1-runpod-plan.md`; overall plan and user decisions: `docs/PLAN.md`.
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

Flaky tests (plan R15). Don't re-run CI to look for flakes:
- On PRs, job `flake-check` runs the test files the PR adds or changes 20 times under CPU load, with a 7-minute limit. That is 10 pytest sessions, each with `-n auto --count 2` (pytest-repeat). Separate sessions are needed because pytest-repeat's copies share module-scoped fixtures. The job fails if a test both passed and failed (`scripts/flaky_report.py`), and is skipped when no test file changed.
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
