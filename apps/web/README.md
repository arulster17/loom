# web

Placeholder. The Next.js dashboard, playground and admin pages (App Router, TypeScript,
Tailwind, Auth.js with the Postgres adapter) are built in Phase 1 M1.

It reads and writes the platform database (`DATABASE_URL`, the same as
`LOOM_DATABASE_URL`), whose schema is owned by the migrations in
`services/platform_common`. Contracts: [docs/phase1-contracts.md](../../docs/phase1-contracts.md)
(§2 environment, §3 schema, §4 signups and approval, §5 API keys).
