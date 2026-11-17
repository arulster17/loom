"""G2a: at most 1,000 background releases run at once per gateway process. A possibly-sent
reserve that fails while 1,000 are running starts none and counts
`loom_gateway_reserve_release_skipped_total{reason="cap"}`; its 503 is unchanged. Once the
running releases finish, later failures start releases again
(docs/m3-review-fixes-spec.md, G2a)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from gateway_m3_g2a_kit import assert_unavailable, skipped, spy_release
from gateway_m3_kit import MESSAGES, wait_until
from redis.exceptions import TimeoutError as RedisTimeoutError

from loom_platform.limiter import Limiter

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

CAP = 1000
SETTLE_S = 0.5  # after a 503, time for a background release to show up if one was started


def _lose_every_reply(monkeypatch: pytest.MonkeyPatch, limiter: Limiter) -> list[Any]:
    """Every reserve fails as if its reply was lost after the command was written (a
    socket timeout while reading): possibly sent. Nothing reaches Redis, so the test
    stays cheap. Returns the request ids."""
    rids: list[Any] = []

    async def reserve(**kw: Any) -> Any:
        rids.append(kw["request_id"])
        raise RedisTimeoutError("Timeout reading from 127.0.0.1:6379")

    monkeypatch.setattr(limiter, "reserve", reserve)
    return rids


async def _post_many(stack: Any, key: str, n: int) -> list[httpx.Response]:
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    async with httpx.AsyncClient(timeout=60, limits=limits) as c:

        async def one() -> httpx.Response:
            return await c.post(
                f"{stack.url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": stack.small, "messages": MESSAGES, "max_tokens": 5},
            )

        return list(await asyncio.gather(*(one() for _ in range(n))))


async def test_g2a_cap_of_1000_running_releases(stack, monkeypatch):
    """Done-when G2a item 4: 1,000 possibly-sent reserve failures each start a background
    release (they block); the next one starts none, keeps its 503 and counts reason
    "cap"; after the blocked releases finish, a later failure starts a release again."""
    t = await stack.tenant()
    lim = stack.gateway.limiter
    rids = _lose_every_reply(monkeypatch, lim)
    gate = asyncio.Event()
    spy = spy_release(monkeypatch, lim, gate)
    unsent0, cap0 = skipped(stack, "unsent"), skipped(stack, "cap")

    try:
        # 1,000 failures: each starts a release, which blocks on the gate.
        responses = await _post_many(stack, t.key, CAP)
        for r in responses:
            assert_unavailable(r)

        async def all_started() -> bool:
            return len(spy["calls"]) == CAP

        assert await wait_until(all_started, 30.0), (
            f"{len(spy['calls'])} of {CAP} releases started (the cap is 1,000, not fewer)"
        )
        assert skipped(stack, "cap") == cap0
        assert {rid for _, rid in spy["calls"]} == set(rids)

        # One more while 1,000 are running: no release, its 503 unchanged, a cap skip.
        r = await _post_many(stack, t.key, 1)
        assert_unavailable(r[0])
        await asyncio.sleep(SETTLE_S)
        assert len(spy["calls"]) == CAP, "a release was started past the cap of 1,000"
        assert skipped(stack, "cap") - cap0 == 1
        assert skipped(stack, "unsent") == unsent0
    finally:
        gate.set()

    # The blocked releases finish: later failures start releases again.
    async def all_returned() -> bool:
        return len(spy["returned"]) == CAP

    assert await wait_until(all_returned, 30.0)
    await asyncio.sleep(SETTLE_S)  # the finished tasks leave the running set

    r = await _post_many(stack, t.key, 1)
    assert_unavailable(r[0])

    async def restarted() -> bool:
        return len(spy["calls"]) == CAP + 1

    assert await wait_until(restarted, 5.0), "no release after the running ones finished"
    assert spy["calls"][-1][1] == rids[-1]
    assert skipped(stack, "cap") - cap0 == 1
    assert skipped(stack, "unsent") == unsent0
