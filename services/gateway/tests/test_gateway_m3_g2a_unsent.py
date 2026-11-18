"""G2a: a reserve that failed before its command was written to Redis starts no background
release and counts `loom_gateway_reserve_release_skipped_total{reason="unsent"}`; one that
failed after the write is still released as in G2 (docs/m3-review-fixes-spec.md, G2a).
The client gets the same 503 in every case."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from gateway_m3_g2a_kit import (
    ScriptBlackhole,
    assert_unavailable,
    free_port,
    limiter_on,
    post,
    skipped,
    spy_release,
    unanswered_listener,
    use_limiter,
)
from gateway_m3_kit import wait_until
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from loom_platform.limiter import Limiter

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

# After the 503, how long a background release gets to show up before we conclude none
# was started. The release task is created before the response is sent, in this loop.
SETTLE_S = 0.5


@pytest.fixture
async def limiters() -> AsyncIterator[list[Limiter]]:
    """Limiters made by a test; their Redis clients are closed at teardown."""
    made: list[Limiter] = []
    yield made
    for lim in made:
        await lim.redis.aclose()


async def _assert_unsent_skipped(stack: Any, monkeypatch: pytest.MonkeyPatch, lim: Limiter) -> None:
    """Through the app with `lim` as the gateway's limiter: the reserve fails unsent, the
    client gets the 503, no release is started, and one `unsent` skip is counted."""
    t = await stack.tenant()
    use_limiter(monkeypatch, stack, lim)
    spy = spy_release(monkeypatch, lim)
    unsent0, cap0 = skipped(stack, "unsent"), skipped(stack, "cap")

    r = await post(stack, t.key)
    assert_unavailable(r)
    assert stack.engine.requests == []

    await asyncio.sleep(SETTLE_S)
    assert spy["calls"] == [], (
        "a background release was started for a reserve that never reached Redis"
    )
    assert skipped(stack, "unsent") - unsent0 == 1
    assert skipped(stack, "cap") == cap0


# ------------------------------------------------------------------ item 1: pool timeout


async def test_g2a_item1_pool_timeout_starts_no_release(stack, monkeypatch, limiters):
    """Done-when G2a item 1: waiting for a pool connection timed out (a real exhausted
    BlockingConnectionPool, "No connection available."): 503 service_unavailable with
    Retry-After: 1, no Limiter.release call, one skip counted with reason "unsent"."""
    lim = limiter_on(stack, stack.gateway.settings.redis_url, max_connections=1, pool_timeout=0.2)
    limiters.append(lim)
    pool = lim.redis.connection_pool
    taken = await pool.get_connection()  # the pool's only connection
    try:
        with pytest.raises(RedisConnectionError, match="No connection available"):
            await lim.redis.ping()  # test setup: the pool is exhausted
        await _assert_unsent_skipped(stack, monkeypatch, lim)
    finally:
        await pool.release(taken)


# ------------------------------------------------------------------ item 2: connect failed


async def test_g2a_item2_connection_refused_starts_no_release(stack, monkeypatch, limiters):
    """Done-when G2a item 2: connecting failed with connection refused (nobody listens on
    the port): the same 503, no release, reason "unsent"."""
    lim = limiter_on(stack, f"redis://127.0.0.1:{free_port()}/0", socket_connect_timeout=2)
    limiters.append(lim)
    with pytest.raises(RedisConnectionError):
        await lim.redis.ping()  # test setup: the connection is refused
    await _assert_unsent_skipped(stack, monkeypatch, lim)


async def test_g2a_item2_connect_timeout_starts_no_release(stack, monkeypatch, limiters):
    """Done-when G2a item 2: connecting failed with the connect timeout (the listener's
    accept queue is full, so the SYN goes unanswered): the same 503, no release, reason
    "unsent"."""
    with unanswered_listener() as port:
        lim = limiter_on(stack, f"redis://127.0.0.1:{port}/0", socket_connect_timeout=0.3)
        limiters.append(lim)
        with pytest.raises(RedisTimeoutError):
            await lim.redis.ping()  # test setup: the connect times out
        await _assert_unsent_skipped(stack, monkeypatch, lim)


# ------------------------------------------------------------------ item 3: possibly sent


@pytest.mark.parametrize("mode", ["stall", "reset"])
async def test_g2a_item3_failure_after_the_write_is_still_released(
    stack, monkeypatch, limiters, mode
):
    """Done-when G2a item 3 (guard; G2): the reserve command reached Redis and the reply
    was lost, by a socket timeout while reading it ("stall") or a connection reset
    ("reset"). The client gets the same 503, the background release of that request id
    starts as in G2, and no skip is counted."""
    async with ScriptBlackhole(mode) as fake:
        lim = limiter_on(stack, fake.url, socket_timeout=0.3, socket_connect_timeout=2)
        limiters.append(lim)
        t = await stack.tenant()
        use_limiter(monkeypatch, stack, lim)
        spy = spy_release(monkeypatch, lim)
        unsent0, cap0 = skipped(stack, "unsent"), skipped(stack, "cap")

        r = await post(stack, t.key)
        assert fake.scripts >= 1, "the reserve script call never reached the fake (test setup)"
        assert_unavailable(r)
        rid = uuid.UUID(r.headers["x-loom-request-id"])

        async def released() -> bool:
            return bool(spy["calls"])

        assert await wait_until(released, 5.0), (
            "no background release for a reserve whose reply was lost"
        )
        assert set(spy["calls"]) == {(t.org_id, rid)}
        assert skipped(stack, "unsent") == unsent0
        assert skipped(stack, "cap") == cap0
