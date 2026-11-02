"""Redis data loss (FLUSHALL, a failover without AOF): the stream comes back without its
consumer group, org state is gone. Billing must notice and heal by itself
(docs/phase1-contracts.md §6 "Redis data loss").

The shared test Redis is never flushed: losing "everything" is simulated by deleting
every key of this test's org and its stream, which is all Redis held for it."""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from billing_fixtures import GROUP, make_event

from loom_billing import metrics
from loom_billing.batch import run_batch
from loom_billing.reconcile import Reconciler
from loom_platform.limiter import NO_LIMITS, Limiter, org_key
from loom_platform.usage import UsageEvent

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


async def _lose_everything(redis_client, stream: str, org: uuid.UUID) -> None:
    keys = [k async for k in redis_client.scan_iter(f"loom:org:{org}:*")]
    await redis_client.delete(stream, *keys)


async def _pg_usage(pool, org: uuid.UUID) -> int:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT loom_org_usage_total(%s)", (org,))
        row = await cur.fetchone()
    return int(row[0])


async def _reserve_and_settle(limiter: Limiter, pool, t, ev: UsageEvent) -> None:
    """The gateway's path: reserve (loading the org on noinit), then settle."""
    kw = dict(
        org_id=t.org_id,
        key_id=t.key_id,
        request_id=ev.id,
        max_cost_micros=ev.cost_micros,
        tpm_estimate=10,
        limits=NO_LIMITS,
        ttl_ms=60_000,
    )
    res = await limiter.reserve(**kw)
    if res.reason == "noinit":
        async with pool.connection() as conn:
            await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
        res = await limiter.reserve(**kw)
    assert res.admitted, res
    assert (await limiter.settle(ev)).settled


async def test_consumer_recreates_a_lost_group(redis_client, stream, make_tenant, make_consumer):
    t = await make_tenant(credit_micros=10_000_000)
    consumer = make_consumer()
    await consumer.setup()
    limiter = Limiter(redis_client, usage_stream=stream)
    await limiter.settle(make_event(t))
    assert (await consumer.drain()).inserted == 1

    await redis_client.delete(stream)  # FLUSHALL, as far as the stream is concerned
    after = make_event(t)
    await limiter.settle(after)  # XADD recreates the stream, without the group
    before = metrics.GROUP_RECREATED._value.get()
    r = await consumer.run_once(block_ms=0)
    assert r.inserted == 1
    assert metrics.GROUP_RECREATED._value.get() == before + 1
    assert (await consumer.drain()).read == 0


async def test_reconcile_reports_a_missing_group(pool, redis_client, stream, make_tenant):
    t = await make_tenant(credit_micros=10_000_000)
    limiter = Limiter(redis_client, usage_stream=stream)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    await limiter.settle(make_event(t))  # a stream with no group at all
    before = metrics.GROUP_MISSING._value.get()
    run = await Reconciler(pool, redis_client, stream=stream, group=GROUP).run([t.org_id])
    assert not run.drift_checked
    assert metrics.GROUP_MISSING._value.get() == before + 1


async def test_flushall_recovery_end_to_end(pool, redis_client, stream, make_tenant, make_consumer):
    t = await make_tenant(credit_micros=10_000_000)
    limiter = Limiter(redis_client, usage_stream=stream)
    consumer = make_consumer()
    await consumer.setup()
    reconciler = Reconciler(pool, redis_client, stream=stream, group=GROUP)

    billed = [make_event(t, prompt=1000 + i) for i in range(3)]
    for ev in billed:
        await _reserve_and_settle(limiter, pool, t, ev)
    await consumer.drain()
    await run_batch(pool, now=dt.datetime.now(dt.UTC) - dt.timedelta(minutes=2))

    await _lose_everything(redis_client, stream, t.org_id)

    # Settled while the org is not loaded (a request in flight at the flush): the event
    # is appended, no counter is created.
    orphan = make_event(t, prompt=777)
    assert (await limiter.settle(orphan)).org_usage_total_micros is None
    # The next request reloads the org from Postgres, before billing stored the orphan.
    later = make_event(t, prompt=555)
    await _reserve_and_settle(limiter, pool, t, later)

    r = await consumer.drain()  # recreates the group, stores both
    assert r.inserted == 2
    total = sum(e.cost_micros for e in [*billed, orphan, later])
    assert await _pg_usage(pool, t.org_id) == total

    # Redis is behind by the orphan; reconcile reports and heals it.
    run = await reconciler.run([t.org_id])
    assert [(d.direction, d.micros) for d in run.drifts] == [("redis_behind", orphan.cost_micros)]
    assert int(await redis_client.get(org_key(t.org_id, "usage"))) == total
    run = await reconciler.run([t.org_id])
    assert run.drift_checked and run.drifts == []

    # Credit and its version came back with the reload; balance is exact.
    assert int(await redis_client.get(org_key(t.org_id, "credit"))) == 10_000_000
    assert await limiter.available(t.org_id) == 10_000_000 - total
    await run_batch(pool)
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT count(*) FROM usage_events WHERE org_id = %s "
            "AND ledger_transaction_id IS NULL AND cost_micros > 0",
            (t.org_id,),
        )
        assert (await cur.fetchone())[0] == 0
        await cur.execute("SELECT loom_org_balance(%s)", (t.org_id,))
        assert (await cur.fetchone())[0] == 10_000_000 - total
