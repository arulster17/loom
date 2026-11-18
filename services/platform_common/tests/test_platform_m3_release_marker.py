"""release.lua's done marker and `Limiter.release`'s TTL check against a real Redis
(docs/m3-review-fixes-spec.md, G2 and G2a; contracts §6).

`release` marks the request id done (`:done:<rid>`, the marker reserve.lua refuses)
whether or not a reservation existed, with NX, for at least the reservation TTL; a later
reserve of that id answers `duplicate_request` and holds nothing. A marker TTL below 1 s
is refused with ValueError before anything is sent to Redis."""

from __future__ import annotations

import socket
import uuid

import pytest
from redis.asyncio import Redis

from loom_platform.limiter import NO_LIMITS, Limiter, org_key

pytestmark = pytest.mark.redis

RESERVATION_TTL_MS = 660_000  # LOOM_REQUEST_TIMEOUT_S + LOOM_RESERVATION_GRACE_S (600 + 60)


@pytest.fixture
async def limiter(redis_client):
    stream = f"loom:test:usage:{uuid.uuid4().hex}"
    yield Limiter(redis_client, usage_stream=stream)
    await redis_client.delete(stream)


@pytest.fixture
async def org(limiter):
    """An org and key loaded with $1 of credit and no usage."""
    org_id, key_id = uuid.uuid4(), uuid.uuid4()
    await limiter.init_org(
        org_id=org_id, key_id=key_id, credit_total=1_000_000, usage_total=0, key_usage_total=0
    )
    yield org_id, key_id
    keys = [k async for k in limiter.redis.scan_iter(match=f"loom:org:{org_id}:*")]
    if keys:
        await limiter.redis.delete(*keys)


async def _reserve(limiter, org_id, key_id, rid, cost=1000):
    return await limiter.reserve(
        org_id=org_id, key_id=key_id, request_id=rid, max_cost_micros=cost,
        tpm_estimate=100, limits=NO_LIMITS, ttl_ms=RESERVATION_TTL_MS,
    )  # fmt: skip


def _marker(org_id, rid) -> str:
    return org_key(org_id, f"done:{rid}")


async def _held(redis, org_id) -> tuple[int, int]:
    rsum = await redis.get(org_key(org_id, "rsum"))
    n = await redis.zcard(org_key(org_id, "rsv:exp"))
    return int(rsum or 0), int(n)


# ------------------------------------------------------------------ the marker (item 6)


async def test_release_marks_done_when_a_reservation_existed(limiter, org):
    """Done-when G2a item 6 (guard): release of a held reservation drops it and writes
    the done marker."""
    org_id, key_id = org
    rid = uuid.uuid4()
    assert (await _reserve(limiter, org_id, key_id, rid)).admitted
    assert await limiter.release(org_id, rid) is True
    assert await limiter.redis.exists(_marker(org_id, rid)) == 1
    assert await _held(limiter.redis, org_id) == (0, 0)


async def test_release_marks_done_when_no_reservation_existed(limiter, org):
    """Done-when G2a item 6 (guard): release of an id never reserved (the reserve has not
    run yet) still writes the done marker."""
    org_id, _ = org
    rid = uuid.uuid4()
    assert await limiter.release(org_id, rid) is False
    assert await limiter.redis.exists(_marker(org_id, rid)) == 1


async def test_release_marker_is_nx_keeping_value_and_ttl(limiter, org):
    """Done-when G2a item 6 (guard): the marker is written with NX, so a marker already
    there (from a settle, say) keeps its value and its TTL."""
    org_id, _ = org
    rid = uuid.uuid4()
    await limiter.redis.set(_marker(org_id, rid), "settled-earlier", ex=5000)
    await limiter.release(org_id, rid)
    await limiter.release(org_id, rid, 700)
    assert await limiter.redis.get(_marker(org_id, rid)) == b"settled-earlier"
    ttl = await limiter.redis.ttl(_marker(org_id, rid))
    assert 4900 < ttl <= 5000


async def test_release_marker_lasts_at_least_the_reservation_ttl(limiter, org):
    """Done-when G2a item 6 (guard): with the two-argument call, the marker lives at
    least as long as the reservation TTL, so a reserve queued behind a stall is refused
    for as long as it could still run."""
    org_id, key_id = org
    for reserved_first in (True, False):
        rid = uuid.uuid4()
        if reserved_first:
            assert (await _reserve(limiter, org_id, key_id, rid)).admitted
        await limiter.release(org_id, rid)
        ttl = await limiter.redis.ttl(_marker(org_id, rid))
        assert ttl >= RESERVATION_TTL_MS // 1000 - 5, (reserved_first, ttl)


async def test_release_with_an_explicit_marker_ttl(limiter, org):
    """Done-when G2a item 6 (guard): the optional TTL argument sets the marker's TTL;
    1 s is the smallest allowed."""
    org_id, _ = org
    rid, rid1 = uuid.uuid4(), uuid.uuid4()
    await limiter.release(org_id, rid, 900)
    assert 890 < await limiter.redis.ttl(_marker(org_id, rid)) <= 900
    await limiter.release(org_id, rid1, 1)
    assert 0 < await limiter.redis.pttl(_marker(org_id, rid1)) <= 1000


async def test_reserve_after_release_is_duplicate_and_holds_nothing(limiter, org):
    """Done-when G2a item 6 (guard; G2 (b)): a reserve that runs after the release of its
    id answers duplicate_request and reserves nothing."""
    org_id, key_id = org
    rid = uuid.uuid4()
    await limiter.release(org_id, rid)
    available = await limiter.available(org_id)
    r = await _reserve(limiter, org_id, key_id, rid)
    assert (r.admitted, r.reason) == (False, "duplicate_request")
    assert await _held(limiter.redis, org_id) == (0, 0)
    assert await limiter.available(org_id) == available


# ------------------------------------------------------------------ TTL below 1 s (item 5)


@pytest.mark.parametrize("ttl", [0, -1, 0.5])
async def test_release_refuses_a_marker_ttl_below_one_second(limiter, org, ttl):
    """Done-when G2a item 5: a marker TTL below 1 s raises ValueError (not a script error
    from `SET ... EX 0`) and changes nothing: the reservation is still held and no marker
    is written."""
    org_id, key_id = org
    rid = uuid.uuid4()
    assert (await _reserve(limiter, org_id, key_id, rid)).admitted
    before = await _held(limiter.redis, org_id)
    with pytest.raises(ValueError):
        await limiter.release(org_id, rid, ttl)
    assert await limiter.redis.exists(_marker(org_id, rid)) == 0
    assert await _held(limiter.redis, org_id) == before
    with pytest.raises(ValueError):
        await limiter.release(org_id, rid, done_ttl_s=ttl)
    assert await limiter.redis.exists(_marker(org_id, rid)) == 0


async def test_release_refuses_a_ttl_below_one_second_without_contacting_redis():
    """Done-when G2a item 5: the check comes before anything is sent: with a Redis
    nobody listens on, the call raises ValueError, not a connection error."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    redis = Redis(host="127.0.0.1", port=port, socket_connect_timeout=1)
    try:
        lim = Limiter(redis)
        with pytest.raises(ValueError):
            await lim.release(uuid.uuid4(), uuid.uuid4(), 0)
    finally:
        await redis.aclose()
