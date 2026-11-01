"""Usage stream consumer: at-least-once delivery, idempotent inserts, dead letters."""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest
from billing_fixtures import GROUP, make_event, xadd_event

from loom_billing.batch import run_batch
from loom_billing.consumer import UsageConsumer
from loom_billing.partitions import ensure_partitions
from loom_billing.streams import group_state
from loom_platform.ledger import org_balance, org_usage_total
from loom_platform.limiter import Limiter

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


async def count_events(pool, org) -> int:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT count(*) FROM usage_events WHERE org_id = %s", (org,))
        row = await cur.fetchone()
    return int(row[0])


async def test_settled_events_are_inserted_acked_and_trimmed(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant(credit_micros=10_000_000)
    limiter = Limiter(redis_client, usage_stream=stream)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    events = [make_event(t, prompt=100 * i, completion=7 * i) for i in range(1, 6)]
    for ev in events:
        assert (await limiter.settle(ev)).settled

    c = make_consumer()
    r = await c.drain()
    assert (r.inserted, r.duplicate, r.dead) == (5, 0, 0)
    assert await count_events(pool, t.org_id) == 5
    async with pool.connection() as conn:
        assert await org_usage_total(conn, t.org_id) == sum(e.cost_micros for e in events)
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 0
    assert await redis_client.xlen(stream) == 0  # trimmed below the group's position


async def test_redelivered_and_duplicate_entries_bill_once(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant()
    ev = make_event(t, prompt=5000, completion=2000)
    # The same event appended three times (a retried XADD, a replayed AOF).
    for _ in range(3):
        await xadd_event(redis_client, stream, ev)
    r = await make_consumer().drain()
    assert (r.inserted, r.duplicate) == (1, 2)
    # And once more after the first copies were consumed.
    await xadd_event(redis_client, stream, ev)
    r = await make_consumer().drain()
    assert (r.inserted, r.duplicate) == (0, 1)
    assert await count_events(pool, t.org_id) == 1

    run = await run_batch(pool)
    mine = [b for b in run.posted if b.org_id == t.org_id]
    assert len(mine) == 1 and mine[0].amount_micros == ev.cost_micros
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == -ev.cost_micros


class CrashBeforeAck(UsageConsumer):
    """Commits the insert, then dies before XACK."""

    async def process(self, entries):  # type: ignore[no-untyped-def]
        real = self.redis.xack

        async def boom(*a, **k):  # type: ignore[no-untyped-def]
            raise ConnectionError("killed before ack")

        self.redis.xack = boom  # type: ignore[method-assign]
        try:
            return await super().process(entries)
        finally:
            self.redis.xack = real  # type: ignore[method-assign]


async def test_crash_between_insert_and_ack_does_not_double_bill(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant()
    events = [make_event(t, prompt=300 + i) for i in range(4)]
    for ev in events:
        await xadd_event(redis_client, stream, ev)

    crashing = CrashBeforeAck(
        pool, redis_client, stream=stream, group=GROUP, consumer="c1",
        dead_stream=stream + ":dead", block_ms=10,
    )  # fmt: skip
    with pytest.raises(ConnectionError):
        await crashing.run_once(block_ms=0)
    # Rows committed, entries still pending.
    assert await count_events(pool, t.org_id) == 4
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 4

    # The same consumer restarts: recovers its own pending entries.
    r = await make_consumer("c1").drain()
    assert (r.inserted, r.duplicate) == (0, 4)
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 0
    assert await count_events(pool, t.org_id) == 4

    run = await run_batch(pool)
    mine = [b for b in run.posted if b.org_id == t.org_id]
    assert [b.amount_micros for b in mine] == [sum(e.cost_micros for e in events)]


async def test_peer_claims_a_dead_consumers_entries(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant()
    for i in range(3):
        await xadd_event(redis_client, stream, make_event(t, prompt=10 + i))
    crashing = CrashBeforeAck(
        pool, redis_client, stream=stream, group=GROUP, consumer="dead-host",
        dead_stream=stream + ":dead", block_ms=10,
    )  # fmt: skip
    with pytest.raises(ConnectionError):
        await crashing.run_once(block_ms=0)
    await asyncio.sleep(0.02)
    r = await make_consumer("other-host", claim_idle_ms=10).drain()
    assert (r.inserted, r.duplicate) == (0, 3)
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 0


async def test_malformed_entries_go_to_dead_letter(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant()
    good = make_event(t)
    await xadd_event(redis_client, stream, good)
    f = good.to_stream_fields()
    fields = dict(zip(f[::2], f[1::2], strict=True))
    bad_version = {**fields, "v": "2", "id": str(uuid.uuid4())}
    bad_cost = {**fields, "id": str(uuid.uuid4()), "cost_micros": "1"}
    garbage = {"hello": "world"}
    bad_ids = [await redis_client.xadd(stream, x) for x in (bad_version, bad_cost, garbage)]

    r = await make_consumer().drain()
    assert (r.inserted, r.dead) == (1, 3)
    dead = await redis_client.xrange(stream + ":dead")
    assert [d[1][b"dead_source_id"] for d in dead] == bad_ids
    assert all(d[1][b"dead_reason"].startswith(b"invalid:") for d in dead)
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 0  # acked, not stuck
    assert await count_events(pool, t.org_id) == 1


async def test_refused_by_postgres_goes_to_dead_letter(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant()
    from billing_fixtures import Tenant

    ghost = Tenant(uuid.uuid4(), uuid.uuid4())  # no such org/key: foreign key violation
    good = [make_event(t), make_event(t)]
    await xadd_event(redis_client, stream, good[0])
    await xadd_event(redis_client, stream, make_event(ghost))
    await xadd_event(redis_client, stream, good[1])
    r = await make_consumer().drain()
    assert (r.inserted, r.dead) == (2, 1)
    dead = await redis_client.xrange(stream + ":dead")
    assert dead[0][1][b"org_id"] == str(ghost.org_id).encode()
    assert dead[0][1][b"dead_reason"].startswith(b"refused:")


async def test_missing_partition_leaves_entry_pending_without_blocking(
    pool, redis_client, stream, make_tenant, make_consumer
):
    t = await make_tenant()
    far = dt.datetime(2097, 3, 4, 5, 6, 7, tzinfo=dt.UTC)
    stuck = make_event(t, created_at=far)
    await xadd_event(redis_client, stream, stuck)
    ok = make_event(t)
    await xadd_event(redis_client, stream, ok)

    c = make_consumer(claim_idle_ms=10)
    r = await c.drain()
    assert (r.inserted, r.left_pending, r.dead) == (1, 1, 0)
    # Later entries are not blocked by the stuck one.
    later = make_event(t)
    await xadd_event(redis_client, stream, later)
    await asyncio.sleep(0.02)
    r = await c.drain()
    assert r.inserted == 1
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 1
    assert await redis_client.xlen(stream) >= 1  # never trimmed while pending

    # Partition upkeep creates the month; the retry then lands.
    assert await ensure_partitions(pool, months=1, today=far.date()) == 1
    await asyncio.sleep(0.02)
    r = await c.drain()
    assert r.inserted == 1
    st = await group_state(redis_client, stream, GROUP)
    assert st is not None and st.pending == 0
    assert await count_events(pool, t.org_id) == 3


async def test_trimmed_entries_are_acked(pool, redis_client, stream, make_tenant, make_consumer):
    t = await make_tenant()
    c = make_consumer()
    await c.setup()
    await xadd_event(redis_client, stream, make_event(t))
    entries = await redis_client.xreadgroup(GROUP, "c1", {stream: ">"})
    entry_id = entries[0][1][0][0]
    await redis_client.xdel(stream, entry_id)
    r = await make_consumer("c1").drain()
    assert r.inserted == 0 and entry_id.decode() in r.acked
