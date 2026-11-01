"""Postgres connections: psycopg 3, async for services, sync for one-shot tools.

psycopg 3 rather than asyncpg: it is already the bench's driver (one driver in the
repo), it has both sync and async APIs (the migrate command and tests use sync), and
it takes the same libpq URLs as the web app's node-postgres. The gateway's hot path does
not touch Postgres (Redis caches keys and balances), so asyncpg's speed edge buys
nothing here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


def libpq_url(url: str) -> str:
    """Accept SQLAlchemy-style URLs ('postgresql+psycopg://...') as plain libpq URLs."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        raise ValueError(f"not a database URL: {url!r}")
    base = scheme.split("+", 1)[0]
    if base not in ("postgresql", "postgres"):
        raise ValueError(f"not a Postgres URL: {url!r}")
    return f"postgresql://{rest}"


async def connect(url: str, *, autocommit: bool = False) -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(libpq_url(url), autocommit=autocommit)


def connect_sync(url: str, *, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(libpq_url(url), autocommit=autocommit)


async def open_pool(url: str, *, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    """An opened pool; close it with `await pool.close()`."""
    pool = AsyncConnectionPool(
        libpq_url(url), min_size=min_size, max_size=max_size, open=False
    )
    await pool.open(wait=True)
    return pool


@asynccontextmanager
async def transaction(pool: AsyncConnectionPool) -> AsyncIterator[psycopg.AsyncConnection]:
    """A pooled connection inside one transaction: commit on success, roll back on error."""
    async with pool.connection() as conn, conn.transaction():
        yield conn


__all__ = [
    "connect",
    "connect_sync",
    "dict_row",
    "libpq_url",
    "open_pool",
    "transaction",
]
