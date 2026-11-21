"""G2a: at most 1,000 background releases run at once per gateway process. A possibly-sent
reserve that fails while 1,000 are running starts none and counts
`loom_gateway_reserve_release_skipped_total{reason="cap"}`; its 503 is unchanged. Once the
running releases finish, later failures start releases again
(docs/m3-review-fixes-spec.md, G2a)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter
from typing import Any

import httpx
import psycopg
import pytest
from gateway_m3_g2a_kit import assert_unavailable, skipped, spy_release
from gateway_m3_kit import MESSAGES, wait_until
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from loom_platform.apikeys import hash_api_key
from loom_platform.auth import AuthRecord
from loom_platform.limiter import Limiter

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

CAP = 1000
SETTLE_S = 0.5  # after a 503, time for a background release to show up if one was started

# The one real lookup in `_pin_key_record` (test setup) retries a Redis or Postgres
# connection error or timeout: at most this many attempts, a short fixed pause between
# them, and this total time budget.
SETUP_LOOKUP_ATTEMPTS = 5
SETUP_LOOKUP_BUDGET_S = 30.0
SETUP_LOOKUP_PAUSE_S = 0.5
# Redis: a refused/dropped connection or no free pool connection (ConnectionError), a
# socket read timeout (TimeoutError). Postgres: psycopg's OperationalError, which also
# covers the connection pool's PoolTimeout.
_SETUP_LOOKUP_RETRIED = (RedisConnectionError, RedisTimeoutError, psycopg.OperationalError)


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


async def _setup_lookup(auth: Any, key_hash: bytes) -> AuthRecord | None:
    """The real `auth.lookup(key_hash)`, retried on a Redis or Postgres connection error
    or timeout (`_SETUP_LOOKUP_RETRIED`), at most `SETUP_LOOKUP_ATTEMPTS` times within
    `SETUP_LOOKUP_BUDGET_S` seconds in all. A result, `None` included (key not found), is
    returned at once; any other error propagates. Fails the test, naming the attempts,
    the time and the last error, if no attempt succeeds."""
    t0 = time.monotonic()
    deadline = t0 + SETUP_LOOKUP_BUDGET_S
    last: BaseException | None = None
    attempts = 0
    while attempts < SETUP_LOOKUP_ATTEMPTS:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        attempts += 1
        try:
            async with asyncio.timeout(remaining) as budget:
                return await auth.lookup(key_hash)
        except _SETUP_LOOKUP_RETRIED as e:
            last = e
        except TimeoutError:
            if not budget.expired():
                raise  # not the budget's own timeout: not retried
            last = TimeoutError("the time budget ran out during this attempt")
            break
        if attempts < SETUP_LOOKUP_ATTEMPTS:
            await asyncio.sleep(min(SETUP_LOOKUP_PAUSE_S, max(deadline - time.monotonic(), 0)))
    elapsed = time.monotonic() - t0
    kind = f"{type(last).__module__}.{type(last).__qualname__}"
    pytest.fail(
        f"test setup: the real lookup of the tenant's key failed {attempts} time(s) in "
        f"{elapsed:.1f} s (at most {SETUP_LOOKUP_ATTEMPTS} attempts within "
        f"{SETUP_LOOKUP_BUDGET_S:g} s); last error: {kind}: {last}"
    )


async def _pin_key_record(monkeypatch: pytest.MonkeyPatch, stack: Any, key: str) -> AuthRecord:
    """Look `key` up once through the gateway's real Authenticator (Redis, then
    Postgres), then answer every later lookup of it with that record without Redis or
    Postgres (any other key is unknown). `authenticate` still runs for every request:
    bearer parsing, the revoked and org-status checks. Without this, an auth-side Redis
    or Postgres timeout under load answers the same 503 before reserve runs, and that
    request never reaches the cap under test (auth's own 503s have their own tests).
    That one real lookup is test setup, so it retries a Redis or Postgres connection
    error or timeout a few times within a time budget (`_setup_lookup`); a key that is
    not found is not retried and fails at once."""
    auth = stack.gateway.auth
    key_hash = hash_api_key(key)
    record = await _setup_lookup(auth, key_hash)
    assert record is not None, "the tenant's key was not found (test setup)"

    async def lookup(h: bytes) -> AuthRecord | None:
        return record if h == key_hash else None

    monkeypatch.setattr(auth, "lookup", lookup)
    return record


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


def _rid(r: httpx.Response) -> uuid.UUID:
    """The request id the gateway answered with (`X-Loom-Request-Id`)."""
    return uuid.UUID(r.headers["x-loom-request-id"])


def _released_ids(spy: dict[str, list[Any]]) -> Counter[Any]:
    """How many times each request id was passed to `Limiter.release`."""
    return Counter(rid for _, rid in spy["calls"])


async def test_g2a_cap_of_1000_running_releases(stack, monkeypatch):
    """Done-when G2a item 4: 1,000 possibly-sent reserve failures each start a background
    release (they block); the next one starts none, keeps its 503 and counts reason
    "cap"; after the blocked releases finish, a later failure starts a release again.
    Checked by request id: the releases running are exactly those 1,000 requests', one
    each, never more than 1,000 at once; the request refused at the cap is never
    released. Auth answers from the tenant's record, looked up once for real first, so
    every request reaches reserve."""
    t = await stack.tenant()
    record = await _pin_key_record(monkeypatch, stack, t.key)
    assert record.org_id == t.org_id
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
        first = {_rid(r) for r in responses}
        assert len(first) == CAP, "the gateway reused a request id"
        assert first == set(rids), (
            "the 503s name other request ids than the reserves' "
            f"({len(first - set(rids))} never reached reserve)"
        )

        async def all_started() -> bool:
            return len(spy["calls"]) == CAP

        assert await wait_until(all_started, 30.0), (
            f"{len(spy['calls'])} of {CAP} releases started (the cap is 1,000, not fewer)"
        )
        assert skipped(stack, "cap") == cap0
        assert {rid for _, rid in spy["calls"]} == set(rids)
        released = _released_ids(spy)
        assert set(released) == first
        assert set(released.values()) == {1}, "a request id was released more than once"
        assert {org for org, _ in spy["calls"]} == {t.org_id}
        assert set(spy["running"]) == first and len(spy["running"]) == CAP
        assert spy["peak"][0] == CAP

        # One more while 1,000 are running: no release, its 503 unchanged, a cap skip.
        r = await _post_many(stack, t.key, 1)
        assert_unavailable(r[0])
        over_cap = _rid(r[0])
        assert over_cap not in first
        assert rids[-1] == over_cap, "the request refused at the cap never reached reserve"
        await asyncio.sleep(SETTLE_S)
        assert len(spy["calls"]) == CAP, "a release was started past the cap of 1,000"
        assert over_cap not in _released_ids(spy), "the request past the cap was released"
        assert spy["peak"][0] == CAP, "more than 1,000 releases ran at once"
        assert skipped(stack, "cap") - cap0 == 1
        assert skipped(stack, "unsent") == unsent0
    finally:
        gate.set()

    # The blocked releases finish: later failures start releases again.
    async def all_returned() -> bool:
        return len(spy["returned"]) == CAP

    assert await wait_until(all_returned, 30.0)
    assert set(spy["returned"]) == first
    assert spy["running"] == []
    await asyncio.sleep(SETTLE_S)  # the finished tasks leave the running set

    r = await _post_many(stack, t.key, 1)
    assert_unavailable(r[0])
    later = _rid(r[0])
    assert later not in first | {over_cap}

    async def restarted() -> bool:
        return len(spy["calls"]) == CAP + 1

    assert await wait_until(restarted, 5.0), "no release after the running ones finished"
    assert spy["calls"][-1][1] == rids[-1]
    assert spy["calls"][-1] == (t.org_id, later)
    released = _released_ids(spy)
    assert set(released) == first | {later}
    assert set(released.values()) == {1}
    assert over_cap not in released, "the request refused at the cap was released later"

    async def later_returned() -> bool:
        return later in spy["returned"]

    assert await wait_until(later_returned, 5.0)
    assert spy["peak"][0] == CAP
    assert skipped(stack, "cap") - cap0 == 1
    assert skipped(stack, "unsent") == unsent0
