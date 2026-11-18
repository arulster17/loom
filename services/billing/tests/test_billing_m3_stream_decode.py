"""B4 (plan R9, docs/m3-review-fixes-spec.md): a usage stream entry whose field name or
value is not valid UTF-8 is dead-lettered and acked, not a wedge. The good entries around
it are stored, nothing stays pending, and later rounds still consume new entries."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from billing_fixtures import GROUP, make_event, xadd_event

from loom_platform.usage import UsageEvent

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

BAD_BYTES = b"\xc3\x28\xff"  # not valid UTF-8


async def stored_ids(pool, ids: list[uuid.UUID]) -> set[uuid.UUID]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT id FROM usage_events WHERE id = ANY(%s)", (ids,))
        return {r[0] for r in await cur.fetchall()}


def _fields(ev: UsageEvent) -> dict[bytes | str, bytes | str]:
    f = ev.to_stream_fields()
    return dict(zip(f[::2], f[1::2], strict=True))


async def xadd_bad(redis_client, stream: str, ev: UsageEvent, where: str) -> str:
    """Append `ev` with one non-UTF-8 field name or value."""
    fields = _fields(ev)
    if where == "name":
        fields[b"note" + BAD_BYTES] = b"x"
    else:
        fields["model_id"] = b"qwen3-" + BAD_BYTES
    entry_id = await redis_client.xadd(stream, fields)
    return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)


async def pending_count(redis_client, stream: str) -> int:
    return int((await redis_client.xpending(stream, GROUP))["pending"])


async def assert_dead_lettered(redis_client, stream: str, bad: UsageEvent) -> None:
    dead = await redis_client.xrange(stream + ":dead")
    assert len(dead) == 1, dead
    _, fields = dead[0]
    assert fields.get(b"dead_reason"), fields
    # The entry is copied, not replaced by a stub: its valid fields are kept.
    assert fields.get(b"id") == str(bad.id).encode(), fields


@pytest.mark.parametrize("where", ["name", "value"])
async def test_non_utf8_entry_between_good_ones_is_dead_lettered(
    pool, redis_client, stream, make_tenant, make_consumer, where
):
    """done-when B4: good ones stored, the bad one dead-lettered, nothing pending, later
    rounds still consume."""
    t = await make_tenant()
    consumer = make_consumer()
    await consumer.setup()
    before = make_event(t, prompt=111)
    bad = make_event(t, prompt=222)
    after = make_event(t, prompt=333)
    await xadd_event(redis_client, stream, before)
    await xadd_bad(redis_client, stream, bad, where)
    await xadd_event(redis_client, stream, after)

    await consumer.drain()

    assert await stored_ids(pool, [before.id, bad.id, after.id]) == {before.id, after.id}
    await assert_dead_lettered(redis_client, stream, bad)
    assert await pending_count(redis_client, stream) == 0

    # Not wedged: a later round consumes a new entry.
    later = make_event(t, prompt=444)
    await xadd_event(redis_client, stream, later)
    await consumer.run_once(block_ms=0)
    assert await stored_ids(pool, [later.id]) == {later.id}
    assert await pending_count(redis_client, stream) == 0
    assert await redis_client.xlen(stream + ":dead") == 1


@pytest.mark.parametrize("path", ["own_pending_after_restart", "claimed_from_crashed_peer"])
async def test_non_utf8_entry_redelivered_from_pending_is_dead_lettered(
    pool, redis_client, stream, make_tenant, make_consumer, path
):
    """done-when B4, on the redelivery paths: the bad entry was delivered to a consumer
    that crashed before processing it, and is read again from the pending list (by the
    same consumer name at restart, or claimed by a peer)."""
    t = await make_tenant()
    await make_consumer("setup").setup()
    before = make_event(t, prompt=555)
    bad = make_event(t, prompt=666)
    after = make_event(t, prompt=777)
    await xadd_event(redis_client, stream, before)
    await xadd_bad(redis_client, stream, bad, "value")
    await xadd_event(redis_client, stream, after)
    # "crashed" read the three entries and died before processing them.
    await redis_client.xreadgroup(GROUP, "crashed", {stream: ">"}, count=10)
    assert await pending_count(redis_client, stream) == 3

    if path == "own_pending_after_restart":
        consumer = make_consumer("crashed")
    else:
        consumer = make_consumer("peer", claim_idle_ms=1)
        await asyncio.sleep(0.01)
    await consumer.drain()

    assert await stored_ids(pool, [before.id, bad.id, after.id]) == {before.id, after.id}
    await assert_dead_lettered(redis_client, stream, bad)
    assert await pending_count(redis_client, stream) == 0

    later = make_event(t, prompt=888)
    await xadd_event(redis_client, stream, later)
    await consumer.run_once(block_ms=0)
    assert await stored_ids(pool, [later.id]) == {later.id}
    assert await pending_count(redis_client, stream) == 0
