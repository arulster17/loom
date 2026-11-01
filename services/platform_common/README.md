# platform_common (`loom_platform`)

Code shared by the gateway and the billing worker:

| Module | What |
|---|---|
| `migrations/*.sql`, `migrate.py` | Postgres schema; `loom-migrate [--create-database]` |
| `ledger.py` | Balanced, idempotent postings; balances; signup approval |
| `limiter.py`, `lua/*.lua` | Credit reservation, rate limits and settlement in Redis |
| `usage.py` | Usage events: stream format and insert |
| `pricing.py` | Request cost and reservation bound from the registry |
| `apikeys.py`, `auth.py` | Key format and hashing; auth record cache |
| `errors.py` | OpenAI-shaped API errors |
| `pool.py` | Model pool state in Redis |
| `settings.py`, `db.py` | `LOOM_*` settings; psycopg 3 connections |
| `testing.py` | pytest fixtures and factories for service tests |

Contracts: [docs/phase1-contracts.md](../../docs/phase1-contracts.md).

```bash
docker compose up -d postgres redis
uv run loom-migrate --create-database
uv run pytest -q services/platform_common/tests
```
