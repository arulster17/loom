# web

The Loom dashboard, playground and admin pages: Next.js 16 (App Router, strict
TypeScript), Tailwind 4, Auth.js v5 with `@auth/pg-adapter` (database sessions).

It reads and writes the platform database, whose schema is owned by the migrations in
`services/platform_common` (no ORM, no schema of its own). Contracts:
[docs/phase1-contracts.md](../../docs/phase1-contracts.md) (§2 environment, §3 schema,
§4 signups and approval, §5 API keys).

## Run it

```bash
docker compose up -d postgres redis          # from the repo root
uv run loom-migrate --create-database        # platform DB loom_platform
cd apps/web && npm ci
LOOM_ENV=dev AUTH_SECRET=dev-secret LOOM_ADMIN_EMAILS=you@example.com npm run dev   # http://localhost:3000
```

`LOOM_ENV=dev` is required for local runs: without it the server starts in production
mode, which refuses to start on a missing or docker-compose default secret (below).

Sign in with a magic link: locally the email goes to Mailpit (web UI on :8025). The
first sign-in creates your user and org, waiting for approval; an email listed in
`LOOM_ADMIN_EMAILS` becomes an admin and can approve itself at `/admin`.

## Checks

```bash
npm run lint && npm run typecheck && npm run build
npm test                   # unit tests (no services needed)
npm run test:integration   # real Postgres: creates loom_web_test_<random>, migrates it
                           # with `uv run loom-migrate`, drops it at the end
```

Integration tests use `LOOM_TEST_DATABASE_URL` (default
`postgresql://loom:loom@localhost:5432/loom`, the compose server); they skip when that is
unset and Postgres is down.

## Image

```bash
docker build -f apps/web/Dockerfile -t loom-web .    # from the repo root
```

Standalone Next.js server on port 3000, non-root, `config/models.yaml` copied to
`/app/config/models.yaml`. `apps/web/Dockerfile.dockerignore` keeps the context to the
web app and the registry.

## Environment

| Variable | Default | What |
|---|---|---|
| `DATABASE_URL` (or `LOOM_DATABASE_URL`) | `postgresql://loom:loom@localhost:5432/loom_platform` | Platform database |
| `LOOM_ENV` | `prod` when unset | `dev` or `prod`; compose sets `dev`. Outside `dev` the server exits at start (`src/instrumentation.ts`, `src/lib/env-check.ts`) when `AUTH_SECRET` or `AUTH_URL` is missing, the billing internal token is missing, or either is the compose dev default (`loom-dev-auth-secret-not-for-production`, `loom-dev-internal-token`) |
| `AUTH_SECRET` | none, **required** | Auth.js secret; also derives the playground key secret |
| `AUTH_URL` | from the request (dev only) | Public URL of the app, e.g. `http://localhost:3000` (links in emails, Stripe return URLs). Required outside `LOOM_ENV=dev`: Auth.js trusts the request's Host header without it |
| `EMAIL_SERVER` | `smtp://localhost:1025` | SMTP for magic links; compose: `smtp://mailpit:1025` |
| `EMAIL_FROM` | `Loom <no-reply@loom.local>` | Sender |
| `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` | none | Google sign-in appears only when both are set; redirect URI `<AUTH_URL>/api/auth/callback/google`. A Google sign-in is refused unless Google reports `email_verified: true` |
| `LOOM_GOOGLE_EMAIL_LINKING` | `true` | `true`/`false`: a Google sign-in may attach to an existing user with the same (verified) email, e.g. one who signed up by magic link (Auth.js `allowDangerousEmailAccountLinking`) |
| `LOOM_ADMIN_EMAILS` | empty | Comma-separated; these users get `users.role = 'admin'` at sign-in (audited) |
| `LOOM_SIGNUP_PROMO_MICROS` | `5000000` | Credit posted when an admin approves a signup |
| `LOOM_GATEWAY_URL` | `http://localhost:8080` | Gateway as the server reaches it (compose: `http://gateway:8080`) |
| `LOOM_PUBLIC_API_URL` | `http://localhost:8080/v1` | API base shown in code samples |
| `LOOM_BILLING_URL` | `http://localhost:8090` | Billing service (compose: `http://billing:8090`) |
| `LOOM_BILLING_INTERNAL_TOKEN` (or `LOOM_INTERNAL_TOKEN`, as compose sets it) | none, **required for Add credit** | Shared with billing; sent as `Authorization: Bearer` |
| `LOOM_REDIS_URL` | `redis://localhost:6379/0` | Read-only fallback for model status when the gateway is down (the Models page and playground normally read the gateway's public `GET /v1/models`) |
| `LOOM_MODELS_YAML` | `../../config/models.yaml` (image: `/app/config/models.yaml`) | Model registry |
| `LOOM_PLAYGROUND_KEY_SECRET` | derived from `AUTH_SECRET` | Optional separate secret for playground keys |
| `LOOM_PLAYGROUND_KEY_TTL_S` | `3600` | Playground key lifetime (min 600) |

## Design notes

**Authorization.** Every data function in `src/lib/data/` takes the caller's context
(`OrgContext` or `Actor`, from `src/lib/session.ts`, re-read from Postgres on every
request) and scopes every query by the caller's org id; another org's key is "not found",
exactly like a missing one. Writes re-check the org status (and, for admin actions, the
admin role) inside their transaction. Admin pages answer 404 to non-admins. Every admin
action writes `admin_audit_log` in the same transaction.

**Playground keys** (the contracts did not define this). The browser never holds an API
key. It posts to `/api/playground/chat` with its session cookie (same-origin checked);
the handler checks the session and that the org is active, then streams from the gateway
with the user's *playground key*: an ordinary `api_keys` row (`scopes = {playground}`,
name "Playground", one per org and user) whose plaintext is never stored. Its secret half
is `HMAC-SHA256(server secret, prefix)` mapped to base62, so the server rebuilds it per
request while the database keeps only prefix and SHA-256, as for any key. It is rotated
after `LOOM_PLAYGROUND_KEY_TTL_S` (the old one revoked) and revoked at sign-out. The
gateway needs nothing new: limits, credit, suspension and usage apply as for any key.
Usage shows it as "Playground"; the API keys page hides it.

**Checkout** (billing's contract, `services/billing/README.md`). The Billing page calls
`POST {LOOM_BILLING_URL}/internal/checkout-sessions` with the shared bearer token and
`{"org_id", "amount_micros", "success_url", "cancel_url", "customer_email"}`, then
redirects the browser to the `url` in the answer (locally the Stripe stub's pay page on
:12111). Amounts are whole cents, $5 to $500. Credit appears only when the signed
webhook posts it: the return page shows it as pending and refreshes itself for a minute.
Auto-recharge settings are stored; billing does not charge saved cards yet.

**Model disable.** The contracts have no runtime switch: a model is disabled with
`status: disabled` in `config/models.yaml`. The admin page says so.

**Privacy.** No prompt or completion is stored or logged; route handlers never log
request bodies.
