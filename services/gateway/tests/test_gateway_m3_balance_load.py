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


async def test_g3_exhausted_pool_when_loading_the_balance_is_503(stack, monkeypatch):
    """A pool timeout taking the connection for the balance load (the gateway's Postgres
    pool exhausted) is the same 503, not a 500."""
    t = await stack.tenant()
    # Authenticate once (the record is then cached outside Postgres) without reserving:
    # an unknown model is refused after auth, before reserve.
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": "no-such-model", "messages": MESSAGES},
        )
    assert r.status_code == 404, r.text

    def exhausted(*args: Any, **kw: Any) -> Any:
        raise PoolTimeout("couldn't get a connection after 30.00 sec")

    monkeypatch.setattr(stack.gateway.pg, "connection", exhausted)
    r = await _post(stack, t.key)
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "balance_unavailable"
    assert r.headers["retry-after"] == "1"
    assert await held(stack, t) == (0, 0, 0)
    assert await stack.events() == []
