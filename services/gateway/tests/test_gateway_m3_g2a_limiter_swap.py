"""The G2a tests swap a limiter on another Redis endpoint into the gateway
(`gateway_m3_g2a_kit.limiter_on`). The "unsent" rule relies on the production pool's
configuration (its retries pinned to 0, its timeouts), so the swapped limiter must carry
every setting the gateway's own client and pool were built with, not a hand-copied subset
(docs/m3-review-fixes-spec.md, G2a). These tests check the kit against the running
gateway."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from gateway_m3_g2a_kit import LOCATION_KWARGS, configured_kwargs, free_port, limiter_on

from loom_platform.limiter import Limiter

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


@pytest.fixture
async def limiters() -> AsyncIterator[list[Limiter]]:
    """Limiters made by a test; their Redis clients are closed at teardown."""
    made: list[Limiter] = []
    yield made
    for lim in made:
        await lim.redis.aclose()


def _retry(r: Any) -> tuple[Any, ...]:
    """A redis-py `Retry` as comparable values: backoff class, retries, error types."""
    return (type(r._backoff), r.get_retries(), tuple(r._supported_errors))


def _comparable(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: _retry(v) if k == "retry" else v for k, v in kwargs.items()}


async def test_swapped_limiter_keeps_the_gateways_client_and_pool_settings(stack, limiters):
    """With no overrides, the swapped limiter's client, pool and connection settings are
    the gateway's own, including the pinned 0 retries (G2a), and a connection it opens
    retries 0 times."""
    gw = stack.gateway
    lim = limiter_on(stack, gw.settings.redis_url)
    limiters.append(lim)
    mine, theirs = lim.redis, gw.redis
    pool, prod = mine.connection_pool, theirs.connection_pool

    assert type(mine) is type(theirs)
    assert mine.single_connection_client == theirs.single_connection_client
    assert type(pool) is type(prod)
    assert pool is not prod
    assert pool.connection_class is prod.connection_class
    assert pool.max_connections == prod.max_connections
    assert pool.timeout == prod.timeout
    assert "retry" in configured_kwargs(prod), "the gateway's pool no longer pins `retry`"
    assert _comparable(configured_kwargs(pool)) == _comparable(configured_kwargs(prod))
    for k in LOCATION_KWARGS:
        assert pool.connection_kwargs.get(k) == prod.connection_kwargs.get(k), k
    assert lim.usage_stream == gw.limiter.usage_stream
    assert lim.settled_ttl_s == gw.limiter.settled_ttl_s

    assert prod.connection_kwargs["retry"].get_retries() == 0
    conn = await pool.get_connection()
    try:
        assert conn.retry.get_retries() == 0, "the swapped pool's connections retry"
        assert conn.socket_timeout == gw.settings.redis_socket_timeout_s
        assert conn.socket_connect_timeout == gw.settings.redis_connect_timeout_s
    finally:
        await pool.release(conn)
    assert await mine.ping()


async def test_swapped_limiter_changes_only_what_it_is_given(stack, limiters):
    """Overrides and the URL change only the location and the named settings; every other
    setting of the gateway's pool is kept."""
    gw = stack.gateway
    port = free_port()
    lim = limiter_on(
        stack,
        f"redis://127.0.0.1:{port}/3",
        max_connections=1,
        pool_timeout=0.2,
        socket_timeout=0.3,
        socket_connect_timeout=0.4,
    )
    limiters.append(lim)
    pool, prod = lim.redis.connection_pool, gw.redis.connection_pool

    assert (pool.max_connections, pool.timeout) == (1, 0.2)
    kw = pool.connection_kwargs
    assert (kw["host"], kw["port"], kw["db"]) == ("127.0.0.1", port, 3)
    assert (kw["socket_timeout"], kw["socket_connect_timeout"]) == (0.3, 0.4)
    named = {"socket_timeout", "socket_connect_timeout"}
    rest = {k: v for k, v in configured_kwargs(pool).items() if k not in named}
    rest_prod = {k: v for k, v in configured_kwargs(prod).items() if k not in named}
    assert _comparable(rest) == _comparable(rest_prod)
    assert pool.connection_class is prod.connection_class
