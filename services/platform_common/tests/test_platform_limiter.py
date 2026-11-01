"""reserve.lua / settle.lua / init_org.lua against a real Redis."""

import asyncio
import datetime as dt
import uuid

import pytest

from loom_platform.ledger import approve_org, org_usage_total
from loom_platform.limiter import NO_LIMITS, Limiter, Limits, api_key_key, org_key
from loom_platform.pricing import Price, cost_micros
from loom_platform.testing import create_api_key, create_org, create_user
from loom_platform.usage import UsageEvent, insert_usage_events

pytestmark = pytest.mark.redis

PRICE = Price(77_000, 77_000, 368_000)
TTL = 660_000


@pytest.fixture
def stream():
    return f"loom:test:usage:{uuid.uuid4().hex}"


@pytest.fixture
async def limiter(redis_client, stream):
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


def event(org_id, key_id, rid, prompt=1000, completion=500, status="ok"):
    return UsageEvent(
        id=rid, org_id=org_id, api_key_id=key_id, model_id="qwen3-8b",
        price_version=PRICE.version, cloud="local", prompt_tokens=prompt,
        completion_tokens=completion, latency_ms=50, ttft_ms=10, status=status,
        cost_micros=cost_micros(PRICE, prompt, completion),
        created_at=dt.datetime.now(dt.UTC),
    )  # fmt: skip


async def reserve(limiter, org_id, key_id, cost=1000, tpm=100, limits=NO_LIMITS, rid=None, ttl=TTL):
    return await limiter.reserve(
        org_id=org_id, key_id=key_id, request_id=rid or uuid.uuid4(), max_cost_micros=cost,
        tpm_estimate=tpm, limits=limits, ttl_ms=ttl,
    )  # fmt: skip


async def test_noinit_then_init(limiter):
    org_id, key_id = uuid.uuid4(), uuid.uuid4()
    r = await reserve(limiter, org_id, key_id)
    assert (r.admitted, r.reason) == (False, "noinit")
    assert await limiter.init_org(
        org_id=org_id, key_id=key_id, credit_total=500, usage_total=100, key_usage_total=0
    ) == (True, True)
    # Present values win over a later (older) snapshot.
    assert await limiter.init_org(
        org_id=org_id, key_id=key_id, credit_total=9, usage_total=9, key_usage_total=9
    ) == (False, False)
    assert await limiter.available(org_id) == 400
    r = await reserve(limiter, org_id, key_id, cost=300)
    assert (r.admitted, r.reason, r.available_micros) == (True, "ok", 100)
    # A second key of a loaded org still needs its own init.
    r = await reserve(limiter, org_id, uuid.uuid4(), cost=1)
    assert r.reason == "noinit"
    await limiter.redis.delete(*[k async for k in limiter.redis.scan_iter(f"loom:org:{org_id}:*")])


async def test_insufficient_balance_writes_nothing(limiter, org):
    org_id, key_id = org
    r = await reserve(limiter, org_id, key_id, cost=1_000_001)
    assert (r.admitted, r.reason, r.available_micros) == (False, "insufficient_balance", 1_000_000)
    assert await limiter.redis.get(org_key(org_id, "rsum")) is None
    assert await limiter.redis.zcard(org_key(org_id, "rsv:exp")) == 0
    r = await reserve(limiter, org_id, key_id, cost=1_000_000)
    assert r.admitted and r.available_micros == 0


async def test_settle_charges_releases_and_appends_event(limiter, org, stream):
    org_id, key_id = org
    rid = uuid.uuid4()
    assert (await reserve(limiter, org_id, key_id, cost=5_000, rid=rid)).admitted
    assert await limiter.available(org_id) == 1_000_000 - 5_000
    e = event(org_id, key_id, rid)
    res = await limiter.settle(e)
    assert res.settled and res.org_usage_total_micros == e.cost_micros == 261
    assert await limiter.available(org_id) == 1_000_000 - 261
    assert int(await limiter.redis.get(org_key(org_id, "rsum"))) == 0
    assert int(await limiter.redis.get(api_key_key(org_id, key_id, "inflight"))) == 0
    assert int(await limiter.redis.get(api_key_key(org_id, key_id, "usage"))) == 261
    entries = await limiter.redis.xrange(stream)
    assert len(entries) == 1 and entries[0][0].decode() == res.stream_id
    got = UsageEvent.from_stream_fields(entries[0][1])
    assert got == e.model_copy(update={"org_usage_total_micros": 261})

    # Settling twice is refused and changes nothing.
    again = await limiter.settle(e)
    assert not again.settled
    assert await limiter.available(org_id) == 1_000_000 - 261
    assert await limiter.redis.xlen(stream) == 1
    # And the request id cannot be reserved again.
    assert (await reserve(limiter, org_id, key_id, rid=rid)).reason == "duplicate_request"


async def test_settle_without_reservation_still_charges(limiter, org, stream):
    org_id, key_id = org
    res = await limiter.settle(event(org_id, key_id, uuid.uuid4(), prompt=0, completion=1000))
    assert res.settled and res.org_usage_total_micros == 368
    assert await limiter.available(org_id) == 1_000_000 - 368
    assert await limiter.redis.xlen(stream) == 1


async def test_duplicate_request_id(limiter, org):
    org_id, key_id = org
    rid = uuid.uuid4()
    assert (await reserve(limiter, org_id, key_id, rid=rid)).admitted
    assert (await reserve(limiter, org_id, key_id, rid=rid)).reason == "duplicate_request"


async def test_key_spend_limit(limiter, org):
    org_id, key_id = org
    lim = Limits(None, None, None, None, None, None, key_spend_limit_micros=1_000)
    a = await reserve(limiter, org_id, key_id, cost=600, limits=lim)
    b = await reserve(limiter, org_id, key_id, cost=600, limits=lim)
    assert a.admitted and (b.admitted, b.reason) == (False, "key_spend_limit")
    await limiter.settle(event(org_id, key_id, uuid.uuid4(), prompt=0, completion=1000))  # 368
    c = await reserve(limiter, org_id, key_id, cost=33, limits=lim)  # 600 + 368 + 33 > 1000
    d = await reserve(limiter, org_id, key_id, cost=32, limits=lim)
    assert c.reason == "key_spend_limit" and d.admitted


async def test_concurrency_limits(limiter, org):
    org_id, key_id = org
    lim = Limits(None, None, None, None, key_concurrency=2, org_concurrency=3)
    rids = [uuid.uuid4() for _ in range(2)]
    for rid in rids:
        assert (await reserve(limiter, org_id, key_id, rid=rid, limits=lim)).admitted
    r = await reserve(limiter, org_id, key_id, limits=lim)
    assert (r.reason, r.retry_after_ms) == ("key_concurrency", 1000)
    other_key = uuid.uuid4()
    await limiter.init_org(
        org_id=org_id, key_id=other_key, credit_total=0, usage_total=0, key_usage_total=0
    )
    assert (await reserve(limiter, org_id, other_key, limits=lim)).admitted
    assert (await reserve(limiter, org_id, other_key, limits=lim)).reason == "org_concurrency"
    await limiter.settle(event(org_id, key_id, rids[0]))
    assert (await reserve(limiter, org_id, key_id, limits=lim)).admitted


async def test_rpm_bucket(limiter, org):
    org_id, key_id = org
    lim = Limits(key_rpm=3, key_tpm=None, org_rpm=None, org_tpm=None,
                 key_concurrency=None, org_concurrency=None)  # fmt: skip
    for _ in range(3):
        assert (await reserve(limiter, org_id, key_id, limits=lim)).admitted
    r = await reserve(limiter, org_id, key_id, limits=lim)
    assert r.reason == "key_rpm"
    assert 0 < r.retry_after_ms <= 20_000  # one token refills in 60 s / 3
    # Org-level bucket applies across keys.
    lim2 = Limits(None, None, org_rpm=4, org_tpm=None, key_concurrency=None, org_concurrency=None)
    other_key = uuid.uuid4()
    await limiter.init_org(
        org_id=org_id, key_id=other_key, credit_total=0, usage_total=0, key_usage_total=0
    )
    for _ in range(4):
        assert (await reserve(limiter, org_id, other_key, limits=lim2)).admitted
    assert (await reserve(limiter, org_id, other_key, limits=lim2)).reason == "org_rpm"


async def test_tpm_estimate_then_settle_refunds(limiter, org):
    org_id, key_id = org
    lim = Limits(None, key_tpm=10_000, org_rpm=None, org_tpm=None,
                 key_concurrency=None, org_concurrency=None)  # fmt: skip
    rid = uuid.uuid4()
    assert (await reserve(limiter, org_id, key_id, tpm=8_000, rid=rid, limits=lim)).admitted
    r = await reserve(limiter, org_id, key_id, tpm=8_000, limits=lim)
    assert r.reason == "key_tpm" and r.retry_after_ms > 0
    # Actual usage was 1,500 tokens: settle refunds the 6,500 over-estimate.
    await limiter.settle(event(org_id, key_id, rid, prompt=1000, completion=500))
    assert (await reserve(limiter, org_id, key_id, tpm=8_000, limits=lim)).admitted


async def test_tpm_estimate_larger_than_bucket_runs_into_debt(limiter, org):
    org_id, key_id = org
    lim = Limits(None, key_tpm=1_000, org_rpm=None, org_tpm=None,
                 key_concurrency=None, org_concurrency=None)  # fmt: skip
    assert (await reserve(limiter, org_id, key_id, tpm=50_000, limits=lim)).admitted
    level = float(await limiter.redis.hget(api_key_key(org_id, key_id, "tb:tpm"), "t"))
    assert level == -1_000  # floored at minus one minute of tokens
    assert (await reserve(limiter, org_id, key_id, tpm=1, limits=lim)).reason == "key_tpm"


async def test_expired_reservations_are_released(limiter, org):
    org_id, key_id = org
    lim = Limits(None, None, None, None, key_concurrency=1, org_concurrency=None)
    assert (await reserve(limiter, org_id, key_id, cost=900_000, ttl=50, limits=lim)).admitted
    assert (await reserve(limiter, org_id, key_id, cost=900_000, limits=lim)).reason in (
        "insufficient_balance",
        "key_concurrency",
    )
    await asyncio.sleep(0.1)
    r = await reserve(limiter, org_id, key_id, cost=900_000, limits=lim)
    assert r.admitted, r
    assert int(await limiter.redis.get(api_key_key(org_id, key_id, "inflight"))) == 1


async def test_concurrent_reserves_never_overspend(limiter, org):
    """200 concurrent requests that could each cost 10,000 against $1: exactly 100 are
    admitted, and settling all of them at full cost leaves the balance at exactly 0."""
    org_id, key_id = org
    rids = [uuid.uuid4() for _ in range(200)]
    results = await asyncio.gather(
        *(reserve(limiter, org_id, key_id, cost=10_000, rid=rid) for rid in rids)
    )
    admitted = [rid for rid, r in zip(rids, results, strict=True) if r.admitted]
    assert len(admitted) == 100
    assert {r.reason for r in results if not r.admitted} == {"insufficient_balance"}
    # Settle each at a cost below its reservation: the difference returns to the balance.
    await asyncio.gather(
        *(limiter.settle(event(org_id, key_id, rid, prompt=0, completion=27)) for rid in admitted)
    )
    per = cost_micros(PRICE, 0, 27)
    assert await limiter.available(org_id) == 1_000_000 - 100 * per


async def test_set_credit_only_for_loaded_orgs(limiter, org):
    org_id, _ = org
    assert await limiter.set_credit(org_id, 2_000_000)
    assert await limiter.available(org_id) == 2_000_000
    stranger = uuid.uuid4()
    assert not await limiter.set_credit(stranger, 5)
    assert await limiter.redis.get(org_key(stranger, "credit")) is None


@pytest.mark.postgres
async def test_end_to_end_with_postgres(limiter, pg, stream):
    """Approve an org, load it into Redis, run requests, consume the stream into
    usage_events: Postgres usage equals Redis usage exactly."""
    admin = await create_user(pg, role="admin")
    org_id = await create_org(pg)
    key_id, _ = await create_api_key(pg, org_id)
    await approve_org(pg, org_id, admin, 5_000_000)
    await pg.commit()
    try:
        r = await reserve(limiter, org_id, key_id)
        assert r.reason == "noinit"
        assert await limiter.load_from_postgres(pg, org_id=org_id, key_id=key_id) == (True, True)
        assert await limiter.available(org_id) == 5_000_000
        for prompt, completion, status in (
            (1200, 300, "ok"),
            (800, 17, "aborted"),
            (50, 0, "error"),
        ):
            rid = uuid.uuid4()
            assert (await reserve(limiter, org_id, key_id, cost=10_000, rid=rid)).admitted
            await limiter.settle(event(org_id, key_id, rid, prompt, completion, status))
        events = [UsageEvent.from_stream_fields(f) for _, f in await limiter.redis.xrange(stream)]
        assert await insert_usage_events(pg, events) == 3
        await pg.commit()
        redis_usage = int(await limiter.redis.get(org_key(org_id, "usage")))
        assert await org_usage_total(pg, org_id) == redis_usage == events[-1].org_usage_total_micros
    finally:
        keys = [k async for k in limiter.redis.scan_iter(match=f"loom:org:{org_id}:*")]
        if keys:
            await limiter.redis.delete(*keys)
