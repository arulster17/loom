"""G3 (R2b): a Postgres failure while loading the balance is a 503
(docs/m3-review-fixes-spec.md, "Gateway")."""

from __future__ import annotations

from typing import Any

import httpx
import psycopg
import pytest
from gateway_m3_kit import MESSAGES, held
from psycopg_pool import PoolTimeout

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


async def _post(stack: Any, key: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as c:
        return await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": stack.small, "messages": MESSAGES, "max_tokens": 5},
        )


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("server closed the connection unexpectedly"),
        psycopg.errors.QueryCanceled("canceling statement due to statement timeout"),
        PoolTimeout("couldn't get a connection after 30.00 sec"),
    ],
    ids=["operational", "query_canceled", "pool_timeout"],
)
async def test_g3_postgres_error_after_noinit_is_503_balance_unavailable(stack, monkeypatch, error):
    """Any Postgres error in `load_from_postgres` after a `noinit` reserve answers 503
    `balance_unavailable` with `Retry-After: 1`, never a 500. Nothing is reserved."""
    t = await stack.tenant()  # a new org: its balance is not in Redis yet (noinit)
    calls = 0

    async def failing_load(*args: Any, **kw: Any) -> Any:
        nonlocal calls
        calls += 1
        raise error

    monkeypatch.setattr(stack.gateway.limiter, "load_from_postgres", failing_load)
    r = await _post(stack, t.key)
    assert calls >= 1, "the balance was not loaded from Postgres (test setup)"
    assert r.status_code == 503, r.text
    err = r.json()["error"]
    assert err["code"] == "balance_unavailable"
    assert err["type"] == "server_error"
    assert r.headers["retry-after"] == "1"
    assert await held(stack, t) == (0, 0, 0)
    assert await stack.events() == []
    assert stack.engine.requests == []

    # Once Postgres answers again the same key is served.
    monkeypatch.undo()
    r = await _post(stack, t.key)
    assert r.status_code == 200, r.text


class _FailingPool:
    """Stands in for a Postgres pool whose every `connection()` raises `error` (a
    PoolTimeout: the pool exhausted); anything else is passed to the real pool. `calls`
    counts the connection attempts."""

    def __init__(self, real: Any, error: Exception) -> None:
        self._real = real
        self._error = error
        self.calls = 0

    def connection(self, *args: Any, **kw: Any) -> Any:
        self.calls += 1
        raise self._error

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


async def test_g3_exhausted_pool_when_loading_the_balance_is_503(stack, monkeypatch):
    """A pool timeout taking the connection for the balance load (the gateway's Postgres
    pool exhausted) is the same 503 `balance_unavailable`, not a 500, and not auth's
    `service_unavailable`.

    Only the balance load gets the exhausted pool: authentication keeps the real one. The
    key is not cached, so auth reads it from Postgres on this very request; the test
    does not depend on an auth cache staying warm (a PoolTimeout during auth is a
    `service_unavailable`, contracts §6: see the test below)."""
    t = await stack.tenant()
    gw = stack.gateway
    assert gw.auth.pg_pool is gw.pg, "auth and the balance load share one pool (setup)"
    exhausted = _FailingPool(gw.pg, PoolTimeout("couldn't get a connection after 30.00 sec"))
    monkeypatch.setattr(gw, "pg", exhausted)

    r = await _post(stack, t.key)
    assert exhausted.calls == 1, "the balance load did not take a connection (test setup)"
    assert r.status_code == 503, r.text
    err = r.json()["error"]
    assert err["code"] == "balance_unavailable"
    assert err["type"] == "server_error"
    assert r.headers["retry-after"] == "1"
    assert await held(stack, t) == (0, 0, 0)
    assert await stack.events() == []
    assert stack.engine.requests == []

    # Once the pool has connections again the same key is served.
    monkeypatch.undo()
    r = await _post(stack, t.key)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("connection to server failed: Connection refused"),
        PoolTimeout("couldn't get a connection after 30.00 sec"),
    ],
    ids=["operational", "pool_timeout"],
)
async def test_postgres_error_during_auth_is_503_service_unavailable(stack, monkeypatch, error):
    """Contracts §6: a Postgres error during auth, for a key not cached in Redis, is 503
    `service_unavailable` with `Retry-After: 1` (not the balance load's
    `balance_unavailable`). Nothing is reserved, nothing reaches an engine."""
    t = await stack.tenant()  # a new key: not cached, auth must ask Postgres
    failing = _FailingPool(stack.gateway.auth.pg_pool, error)
    monkeypatch.setattr(stack.gateway.auth, "pg_pool", failing)

    r = await _post(stack, t.key)
    assert failing.calls >= 1, "auth did not look the key up in Postgres (test setup)"
    assert r.status_code == 503, r.text
    err = r.json()["error"]
    assert err["code"] == "service_unavailable"
    assert err["type"] == "server_error"
    assert r.headers["retry-after"] == "1"
    assert await held(stack, t) == (0, 0, 0)
    assert await stack.events() == []
    assert stack.engine.requests == []

    monkeypatch.undo()
    r = await _post(stack, t.key)
    assert r.status_code == 200, r.text
