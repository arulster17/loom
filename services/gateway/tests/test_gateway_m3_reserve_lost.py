"""G2 (R2a): a reserve whose reply is lost is released; a release refuses a later reserve
of the same request id; the 503 `model_unavailable` release still refunds tpm
(docs/m3-review-fixes-spec.md, "Gateway")."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import pytest
from gateway_m3_kit import MESSAGES, done_marker, held, wait_until
from gateway_testkit import replica, set_ready
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from loom_platform.limiter import NO_LIMITS, Limiter, org_prefix
from loom_platform.pool import PoolState, set_pool

# ------------------------------------------------------------------ helpers


async def _post(stack: Any, key: str, model: str, **extra: Any) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as c:
        return await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": MESSAGES, "max_tokens": 5, **extra},
        )


def _lose_admitted_reply(monkeypatch: pytest.MonkeyPatch, limiter: Limiter) -> list[Any]:
    """The reserve script runs in Redis, but its reply never reaches the gateway: the
    call raises a socket timeout after the reservation was written. Returns the
    request ids whose reply was lost."""
    real = limiter.reserve
    lost: list[Any] = []

    async def reserve(**kw: Any) -> Any:
        result = await real(**kw)
        if result.admitted and not lost:
            lost.append(kw["request_id"])
            raise RedisTimeoutError("Timeout reading from socket")
        return result

    monkeypatch.setattr(limiter, "reserve", reserve)
    return lost


def _flaky_release(monkeypatch: pytest.MonkeyPatch, limiter: Limiter, failures: int) -> list[Any]:
    """`Limiter.release` raises a connection reset for its first `failures` calls (Redis
    still unreachable), then works. Returns the request ids it was called with."""
    real = limiter.release
    calls: list[Any] = []

    async def release(org_id: Any, request_id: Any, *args: Any, **kw: Any) -> Any:
        calls.append(request_id)
        if len(calls) <= failures:
            raise RedisConnectionError("Connection reset by peer")
        return await real(org_id, request_id, *args, **kw)

    monkeypatch.setattr(limiter, "release", release)
    return calls


async def _org_snapshot(redis: Any, org_id: uuid.UUID) -> dict[bytes, bytes]:
    keys = sorted([k async for k in redis.scan_iter(f"{org_prefix(org_id)}:*")])
    return {k: await redis.dump(k) for k in keys}


# ------------------------------------------------------------------ (a) through the app


@pytest.mark.postgres
@pytest.mark.redis
async def test_g2a_reserve_whose_reply_was_lost_is_released(stack, monkeypatch):
    """Done when (a): a reserve that took effect in Redis but whose reply was lost leaves
    no reservation, rsum or in-flight count behind once Redis answers again. The client
    still gets 503 service_unavailable with Retry-After: 1."""
    t = await stack.tenant()
    lost = _lose_admitted_reply(monkeypatch, stack.gateway.limiter)

    r = await _post(stack, t.key, stack.small)
    assert lost, "the reserve never took effect (test setup)"
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "service_unavailable"
    assert r.headers["retry-after"] == "1"
    assert stack.engine.requests == []

    assert await wait_until(lambda: _released(stack, t), 5.0), (
        f"the lost reservation is still held: {await held(stack, t)}"
    )
    assert await stack.events() == []


async def _released(stack: Any, t: Any) -> bool:
    return await held(stack, t) == (0, 0, 0)


@pytest.mark.postgres
@pytest.mark.redis
async def test_g2a_release_of_a_lost_reserve_is_retried_until_redis_answers(stack, monkeypatch):
    """(a) with Redis still failing for a while: the background release is retried with
    backoff and succeeds once Redis answers, leaving nothing held."""
    t = await stack.tenant()
    lost = _lose_admitted_reply(monkeypatch, stack.gateway.limiter)
    calls = _flaky_release(monkeypatch, stack.gateway.limiter, failures=2)

    r = await _post(stack, t.key, stack.small)
    assert lost and r.status_code == 503
    assert r.json()["error"]["code"] == "service_unavailable"

    assert await wait_until(lambda: _released(stack, t), 15.0), (
        f"still held after retries: {await held(stack, t)}; release calls: {len(calls)}"
    )
    assert len(calls) >= 3
    assert set(calls) == {lost[0]}
    assert await stack.events() == []


# ------------------------------------------------------------------ (b) the done marker


@pytest.fixture
async def limiter(redis_client) -> Any:
    lim = Limiter(redis_client, usage_stream=f"loom:test:usage:{uuid.uuid4().hex}")
    orgs: list[uuid.UUID] = []

    async def new_org() -> tuple[uuid.UUID, uuid.UUID]:
        org, key = uuid.uuid4(), uuid.uuid4()
        orgs.append(org)
        await lim.init_org(
            org_id=org, key_id=key, credit_total=5_000_000, usage_total=0, key_usage_total=0
        )
        return org, key

    lim.new_org = new_org  # type: ignore[attr-defined]
    try:
        yield lim
    finally:
        for org in orgs:
            keys = [k async for k in redis_client.scan_iter(f"{org_prefix(org)}:*")]
            if keys:
                await redis_client.delete(*keys)
        await redis_client.delete(lim.usage_stream)


async def _reserve(lim: Limiter, org: uuid.UUID, key: uuid.UUID, rid: uuid.UUID) -> Any:
    return await lim.reserve(
        org_id=org,
        key_id=key,
        request_id=rid,
        max_cost_micros=1_000,
        tpm_estimate=100,
        limits=NO_LIMITS,
        ttl_ms=660_000,
    )


@pytest.mark.redis
async def test_g2b_release_before_a_delayed_reserve_makes_it_a_duplicate(limiter, redis_client):
    """Done when (b): a release that runs before the delayed reserve (queued behind a
    Redis stall) makes that reserve refuse with `duplicate_request` and write nothing."""
    org, key = await limiter.new_org()
    rid = uuid.uuid4()

    await limiter.release(org, rid)
    marker = done_marker(org, rid)
    assert await redis_client.exists(marker), "release did not mark the request id done"
    assert await redis_client.ttl(marker) > 0, "the done marker must expire"

    before = await _org_snapshot(redis_client, org)
    result = await _reserve(limiter, org, key, rid)
    assert (result.admitted, result.reason) == (False, "duplicate_request")
    assert await _org_snapshot(redis_client, org) == before
    assert await limiter.available(org) == 5_000_000


@pytest.mark.redis
async def test_g2b_released_reservation_cannot_be_reserved_again(limiter, redis_client):
    """(b), the ordinary order: reserve, release (the 503 path), then the same request id
    reserved again (a replayed script) is refused and holds nothing."""
    org, key = await limiter.new_org()
    rid = uuid.uuid4()

    first = await _reserve(limiter, org, key, rid)
    assert first.admitted
    await limiter.release(org, rid)
    assert await limiter.available(org) == 5_000_000

    again = await _reserve(limiter, org, key, rid)
    assert (again.admitted, again.reason) == (False, "duplicate_request")
    assert await limiter.available(org) == 5_000_000
    assert await redis_client.zcard(f"{org_prefix(org)}:rsv:exp") == 0


@pytest.mark.redis
async def test_g2b_release_marker_does_not_block_other_request_ids(limiter):
    """The marker is per request id: other requests of the org still reserve."""
    org, key = await limiter.new_org()
    await limiter.release(org, uuid.uuid4())
    other = await _reserve(limiter, org, key, uuid.uuid4())
    assert other.admitted and other.reason == "ok"


# ------------------------------------------------------------------ (c) 503 release guard


@pytest.mark.postgres
@pytest.mark.redis
async def test_g2c_model_unavailable_release_refunds_tpm_and_bills_nothing(stack):
    """Done when (c): the release after a 503 `model_unavailable` still refunds the tpm
    estimate and leaves no usage event. The key's tpm bucket holds exactly one request's
    estimate (every prompt bound is at least the 1024-token template overhead), so
    without the refund the second 503 would be a 429."""
    t = await stack.tenant(
        key_limits={"rpm_limit": 1_000, "tpm_limit": 1_000, "concurrency_limit": 100}
    )
    await set_pool(stack.redis, PoolState(stack.small, "unavailable", 300, reason="no_capacity"))
    for _ in range(3):
        r = await _post(stack, t.key, stack.small)
        assert r.status_code == 503, r.text
        assert r.json()["error"]["code"] == "model_unavailable"
        assert await held(stack, t) == (0, 0, 0)
    assert await stack.events() == []
    assert stack.engine.requests == []

    # The bucket is still full: a request to a ready pool is admitted and served.
    await set_ready(stack.redis, stack.small, [replica("e1", stack.ports["e1"])])
    r = await _post(stack, t.key, stack.small)
    assert r.status_code == 200, r.text
    [event] = await stack.wait_events(1)
    assert event.status == "ok"
