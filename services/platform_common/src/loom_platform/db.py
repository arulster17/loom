"""Postgres connections: psycopg 3, async for services, sync for one-shot tools.

psycopg 3 rather than asyncpg: it is already the bench's driver (one driver in the
repo), it has both sync and async APIs (the migrate command and tests use sync), and
it takes the same libpq URLs as the web app's node-postgres. The gateway's hot path does
not touch Postgres (Redis caches keys and balances), so asyncpg's speed edge buys
nothing here.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import NoReturn

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

log = logging.getLogger("loom_platform.db")


def libpq_url(url: str) -> str:
    """Accept SQLAlchemy-style URLs ('postgresql+psycopg://...') as plain libpq URLs.
    Keyword/value conninfo strings ('host=... dbname=...') pass through unchanged."""
    scheme, sep, rest = url.partition("://")
    if not sep:
        if "=" in url:
            return url
        raise ValueError(f"not a database URL: {url!r}")
    base = scheme.split("+", 1)[0]
    if base not in ("postgresql", "postgres"):
        raise ValueError(f"not a Postgres URL: {url!r}")
    return f"postgresql://{rest}"


async def connect(url: str, *, autocommit: bool = False) -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(libpq_url(url), autocommit=autocommit)


def connect_sync(url: str, *, autocommit: bool = False) -> psycopg.Connection:
    return psycopg.connect(libpq_url(url), autocommit=autocommit)


class RoleGate:
    """The database role check a service could not run at start because Postgres was
    unreachable (`loom_platform.settings.enforce_environment`, outside dev only). Until
    it has passed, the process does no Postgres work: `open_pool` opens no pool and each
    listener waits in `wait()` before it connects. The check runs on the first
    successful connection and reads only the system catalogs; a problem is printed and
    the process exits 2 at once (`os._exit`: no task, finally block or shutdown hook
    runs any more, so nothing is done as the wrong role). With nothing deferred (dev, or
    the check already passed at start) `wait()` returns at once."""

    RETRY_MIN_S = 0.5
    RETRY_MAX_S = 5.0

    def __init__(self) -> None:
        self._deferred: tuple[str, str, str] | None = None  # (url, service, env)
        self._lock: asyncio.Lock | None = None

    @property
    def pending(self) -> bool:
        return self._deferred is not None

    def defer(self, database_url: str, service: str, env: str) -> None:
        self._deferred = (database_url, service, env)

    async def wait(self) -> None:
        if self._deferred is None:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._deferred is not None:  # not checked by an earlier waiter meanwhile
                await self._check(*self._deferred)
                self._deferred = None

    async def _check(self, database_url: str, service: str, env: str) -> None:
        from loom_platform.settings import ROLE_CHECK_SQL, error_summary, role_problems

        delay = self.RETRY_MIN_S
        while True:
            try:
                conn = await psycopg.AsyncConnection.connect(
                    libpq_url(database_url), autocommit=True, connect_timeout=5
                )
                async with conn:
                    cur = await conn.execute(ROLE_CHECK_SQL)
                    row = await cur.fetchone()
                break
            except psycopg.OperationalError as e:
                log.debug("role check: Postgres unreachable (%s)", type(e).__name__)
            except psycopg.Error as e:
                problem = f"cannot check LOOM_DATABASE_URL's database role: {error_summary(e)}"
                self._exit(service, env, [problem])
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.RETRY_MAX_S)
        problems = role_problems(row)
        if problems:
            self._exit(service, env, problems)
        print(f"{service}: Postgres reachable, database role checked", file=sys.stderr, flush=True)

    @staticmethod
    def _exit(service: str, env: str, problems: list[str]) -> NoReturn:
        from loom_platform.settings import print_refusal

        print_refusal(service, env, problems, started=True)
        sys.stdout.flush()
        os._exit(2)


# The process's one gate (a service has one LOOM_DATABASE_URL).
role_gate = RoleGate()
_opening: set[asyncio.Task[None]] = set()


async def _open_after_gate(pool: AsyncConnectionPool) -> None:
    await role_gate.wait()
    try:
        await pool.open(wait=False)
    except Exception:  # closed meanwhile (the service stopped first)
        log.debug("pool not opened", exc_info=True)


async def open_pool(
    url: str, *, min_size: int = 1, max_size: int = 10, background: bool = False
) -> AsyncConnectionPool:
    """An opened pool; close it with `await pool.close()`.

    While the role check is deferred (`role_gate`: Postgres was unreachable at start),
    the pool is opened only once the check has passed: `open_pool` waits for that, or,
    with `background=True`, returns the pool unopened at once and opens it as soon as the
    check has passed (until then `pool.connection()` raises `PoolClosed`, an
    `OperationalError`)."""
    pool = AsyncConnectionPool(libpq_url(url), min_size=min_size, max_size=max_size, open=False)
    if role_gate.pending and background:
        task = asyncio.create_task(_open_after_gate(pool))
        _opening.add(task)
        task.add_done_callback(_opening.discard)
        return pool
    await role_gate.wait()
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
    "role_gate",
    "transaction",
]
