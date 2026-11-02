"""Balance reconcile: credit from Postgres, drift detection, notifications, listener."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from billing_fixtures import GROUP, make_event, purchase

from loom_billing.notify import AutoRecharge, Notifier
from loom_billing.reconcile import CreditListener, Reconciler
from loom_platform.ledger import org_credit_total
from loom_platform.limiter import Limiter, org_key
from loom_platform.usage import insert_usage_events

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

THRESHOLD = 1_000_000


@pytest.fixture
def notifier(redis_client, stream):
    return Notifier(redis_client, stream=stream + ":notify", low_balance_micros=THRESHOLD)


@pytest.fixture
def reconciler(pool, redis_client, stream, notifier):
    return Reconciler(
        pool,
        redis_client,
        stream=stream,
        group=GROUP,
        notifier=notifier,
        recharge=AutoRecharge(notifier, enabled=True),
    )


async def load(pool, redis_client, stream, t) -> Limiter:
    limiter = Limiter(redis_client, usage_stream=stream)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    return limiter


async def credit_in_redis(redis_client, org) -> int:
    return int(await redis_client.get(org_key(org, "credit")))


async def test_credit_is_rewritten_from_postgres(
    pool, redis_client, stream, make_tenant, reconciler
):
    t = await make_tenant(credit_micros=5_000_000)
    await load(pool, redis_client, stream, t)
    async with pool.connection() as conn, conn.transaction():
        await purchase(conn, t.org_id, 2_500_000)
    await redis_client.set(org_key(t.org_id, "credit"), 1)  # stale/wrong
    usage_before = await redis_client.get(org_key(t.org_id, "usage"))

    run = await reconciler.run([t.org_id])
    assert run.credits_set == 1
    assert await credit_in_redis(redis_client, t.org_id) == 7_500_000
    assert await redis_client.get(org_key(t.org_id, "usage")) == usage_before  # untouched


async def test_scan_skips_orgs_unknown_to_postgres(
    pool, redis_client, stream, make_tenant, reconciler
):
    t = await make_tenant(credit_micros=3_000_000)
    await load(pool, redis_client, stream, t)
    await redis_client.set(org_key(t.org_id, "credit"), 0)
    stranger = uuid.uuid4()  # e.g. another test database sharing this Redis
    await redis_client.set(org_key(stranger, "credit"), 42)
    await redis_client.set(org_key(stranger, "usage"), 0)
    try:
        await reconciler.run()  # scans loom:org:*:credit
        assert await credit_in_redis(redis_client, t.org_id) == 3_000_000
        assert await credit_in_redis(redis_client, stranger) == 42
    finally:
        await redis_client.delete(org_key(stranger, "credit"), org_key(stranger, "usage"))


async def test_unloaded_org_is_not_created(pool, redis_client, stream, make_tenant, reconciler):
    t = await make_tenant(credit_micros=3_000_000)
    run = await reconciler.run([t.org_id])
    assert run.credits_set == 0
    assert await redis_client.get(org_key(t.org_id, "credit")) is None


async def test_drift_check(pool, redis_client, stream, make_tenant, make_consumer, reconciler):
    t = await make_tenant(credit_micros=10_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    for i in range(3):
        await limiter.settle(make_event(t, prompt=100 + i))

    # Events still in the stream: billing is behind, not wrong. No verdict.
    run = await reconciler.run([t.org_id])
    assert not run.drift_checked and run.drifts == []

    await consumer.drain()
    run = await reconciler.run([t.org_id])
    assert run.drift_checked and run.drifts == []

    # Redis counted usage whose event never reached Postgres.
    await redis_client.incrby(org_key(t.org_id, "usage"), 1234)
    run = await reconciler.run([t.org_id])
    assert [(d.direction, d.micros) for d in run.drifts] == [("missing_in_postgres", 1234)]
    await redis_client.decrby(org_key(t.org_id, "usage"), 1234)

    # Postgres has usage Redis never counted (e.g. Redis lost its last second).
    lost = make_event(t, prompt=9999)
    async with pool.connection() as conn, conn.transaction():
        await insert_usage_events(conn, [lost])
    run = await reconciler.run([t.org_id])
    assert [(d.direction, d.micros) for d in run.drifts] == [("redis_behind", lost.cost_micros)]


async def test_dead_lettered_event_shows_as_drift(
    pool, redis_client, stream, make_tenant, make_consumer, reconciler
):
    t = await make_tenant(credit_micros=10_000_000)
    limiter = await load(pool, redis_client, stream, t)
    ev = make_event(t, prompt=4242)
    await limiter.settle(ev)
    # Corrupt the entry so the consumer dead-letters it.
    entry_id = (await redis_client.xrange(stream))[-1][0]
    await redis_client.xdel(stream, entry_id)
    await redis_client.xadd(stream, {"v": "1", "id": str(ev.id)})
    await make_consumer().drain()
    run = await reconciler.run([t.org_id])
    assert [(d.direction, d.micros) for d in run.drifts] == [
        ("missing_in_postgres", ev.cost_micros)
    ]


async def notifications(redis_client, stream) -> list[dict[bytes, bytes]]:
    return [f for _, f in await redis_client.xrange(stream + ":notify")]


async def test_low_balance_notifies_once_per_crossing(
    pool, redis_client, stream, make_tenant, reconciler
):
    t = await make_tenant(credit_micros=5_000_000)
    await load(pool, redis_client, stream, t)
    await reconciler.run([t.org_id])
    assert await notifications(redis_client, stream) == []

    await redis_client.set(org_key(t.org_id, "usage"), 4_500_000)  # balance $0.50
    for _ in range(3):
        await reconciler.run([t.org_id])
    notes = await notifications(redis_client, stream)
    assert len(notes) == 1
    assert notes[0][b"type"] == b"low_balance"
    assert notes[0][b"org_id"] == str(t.org_id).encode()
    assert notes[0][b"balance_micros"] == b"500000"

    async with pool.connection() as conn, conn.transaction():
        await purchase(conn, t.org_id, 10_000_000)
    await reconciler.run([t.org_id])  # back above: re-armed
    await redis_client.set(org_key(t.org_id, "usage"), 14_900_000)
    await reconciler.run([t.org_id])
    assert len(await notifications(redis_client, stream)) == 2
    await redis_client.delete(org_key(t.org_id, "lowbal"))


async def test_auto_recharge_hook(pool, redis_client, stream, make_tenant, notifier):
    t = await make_tenant(credit_micros=5_000_000)
    await load(pool, redis_client, stream, t)
    async with pool.connection() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO auto_recharge (org_id, enabled, threshold_micros, amount_micros) "
            "VALUES (%s, true, 3000000, 20000000)",
            (t.org_id,),
        )
    await redis_client.set(org_key(t.org_id, "usage"), 2_500_000)  # balance $2.50 < $3

    off = Reconciler(
        pool, redis_client, stream=stream, group=GROUP,
        recharge=AutoRecharge(notifier, enabled=False),
    )  # fmt: skip
    await off.run([t.org_id])
    assert await notifications(redis_client, stream) == []

    on = Reconciler(
        pool, redis_client, stream=stream, group=GROUP,
        recharge=AutoRecharge(notifier, enabled=True),
    )  # fmt: skip
    await on.run([t.org_id])
    await on.run([t.org_id])
    notes = await notifications(redis_client, stream)
    assert [(n[b"type"], n[b"amount_micros"]) for n in notes] == [
        (b"auto_recharge_due", b"20000000")
    ]
    await redis_client.delete(org_key(t.org_id, "recharge"))


async def test_credit_listener_applies_notifications(
    pg_url, pool, redis_client, stream, make_tenant
):
    t = await make_tenant(credit_micros=1_000_000)
    limiter = await load(pool, redis_client, stream, t)
    listener = CreditListener(pg_url, pool, limiter)
    task = asyncio.create_task(listener.run())
    try:
        await asyncio.wait_for(listener.ready.wait(), 5)
        async with pool.connection() as conn, conn.transaction():
            await purchase(conn, t.org_id, 4_000_000)
        for _ in range(100):
            if await credit_in_redis(redis_client, t.org_id) == 5_000_000:
                break
            await asyncio.sleep(0.05)
        async with pool.connection() as conn:
            assert await credit_in_redis(redis_client, t.org_id) == await org_credit_total(
                conn, t.org_id
            )
        assert await listener.apply("not json") is False
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
