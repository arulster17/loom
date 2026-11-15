"""Test fixtures and factories for platform services (pytest only).

Use from a service's tests/conftest.py:

    from loom_platform.testing import pg_url, pg, redis_client  # noqa: F401

Postgres: `LOOM_TEST_DATABASE_URL` (SQLAlchemy-style URLs are fine), else the compose
database at localhost:5432. Each pytest session (each xdist worker) gets its own
throwaway database with every migration applied, dropped at the end. Redis:
`LOOM_TEST_REDIS_URL`, else redis://localhost:6379/15. When the variable is unset and
the server is not reachable, the tests skip; when it is set, unreachable is an error.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest
import redis.asyncio as aioredis
from psycopg import sql

from loom_platform.apikeys import NewApiKey, generate_api_key
from loom_platform.db import connect, libpq_url
from loom_platform.migrate import upgrade

PG_ENV = "LOOM_TEST_DATABASE_URL"
REDIS_ENV = "LOOM_TEST_REDIS_URL"
DEFAULT_PG = "postgresql://loom:loom@localhost:5432/loom"
DEFAULT_REDIS = "redis://localhost:6379/15"


def _admin_url() -> str:
    explicit = os.environ.get(PG_ENV)
    url = libpq_url(explicit or DEFAULT_PG)
    try:
        psycopg.connect(url, connect_timeout=2).close()
    except psycopg.OperationalError as e:
        if explicit:
            raise
        pytest.skip(f"Postgres not reachable at {DEFAULT_PG} and {PG_ENV} unset: {e}")
    return url


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    """URL of a fresh, fully migrated database for this session."""
    admin = _admin_url()
    name = f"loom_platform_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    info = psycopg.conninfo.conninfo_to_dict(admin)
    url = psycopg.conninfo.make_conninfo("", **{**info, "dbname": name})
    try:
        upgrade(url)
        yield url
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture
async def pg(pg_url: str) -> AsyncIterator[psycopg.AsyncConnection]:
    """A connection to the session database (not autocommit; commit explicitly)."""
    conn = await connect(pg_url)
    try:
        yield conn
    finally:
        await conn.rollback()
        await conn.close()


@pytest.fixture
async def redis_client() -> AsyncIterator[aioredis.Redis]:
    """Redis client. Tests use random org ids and stream names, so they never collide;
    keys they create under loom:org:<id> are left to expire with the test database."""
    explicit = os.environ.get(REDIS_ENV)
    # A blocking pool: redis-py's default pool raises MaxConnectionsError under bursts
    # instead of waiting for a free connection.
    pool = aioredis.BlockingConnectionPool.from_url(
        explicit or DEFAULT_REDIS, max_connections=32, timeout=10
    )
    client = aioredis.Redis(connection_pool=pool)
    try:
        await client.ping()
    except (aioredis.ConnectionError, OSError) as e:
        await client.aclose()
        if explicit:
            raise
        pytest.skip(f"Redis not reachable at {DEFAULT_REDIS} and {REDIS_ENV} unset: {e}")
    try:
        yield client
    finally:
        await client.aclose()


# ------------------------------------------------------------------ factories


async def create_user(
    conn: psycopg.AsyncConnection, *, role: str = "user", email: str | None = None
) -> uuid.UUID:
    email = email or f"{uuid.uuid4().hex[:10]}@example.test"
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO users (name, email, role) VALUES (%s, %s, %s) RETURNING id",
            (email.split("@")[0], email, role),
        )
        row = await cur.fetchone()
    assert row is not None
    return row[0]


async def create_org(
    conn: psycopg.AsyncConnection,
    *,
    status: str = "pending_approval",
    owner: uuid.UUID | None = None,
    name: str = "test org",
) -> uuid.UUID:
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO orgs (name, status) VALUES (%s, %s) RETURNING id", (name, status)
        )
        row = await cur.fetchone()
        assert row is not None
        if owner is not None:
            await cur.execute(
                "INSERT INTO org_members (org_id, user_id, role) VALUES (%s, %s, 'owner')",
                (row[0], owner),
            )
    return row[0]


async def create_api_key(
    conn: psycopg.AsyncConnection,
    org_id: uuid.UUID,
    *,
    created_by: uuid.UUID | None = None,
    name: str = "test key",
    spend_limit_micros: int | None = None,
) -> tuple[uuid.UUID, NewApiKey]:
    key = generate_api_key()
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO api_keys (org_id, created_by, name, prefix, key_hash, "
            "spend_limit_micros) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (org_id, created_by, name, key.prefix, key.key_hash, spend_limit_micros),
        )
        row = await cur.fetchone()
    assert row is not None
    return row[0], key
