"""Migration 0008: usage totals = posted rollup + unposted events, equal to the old full
sums over usage_events at every step of randomized traffic and batches."""

from __future__ import annotations

import datetime as dt
import random

import pytest
from billing_fixtures import make_event

from loom_billing.batch import run_batch
from loom_platform.testing import create_api_key
from loom_platform.usage import insert_usage_events

pytestmark = [pytest.mark.postgres]

OLD_ORG = "SELECT coalesce(sum(cost_micros), 0) FROM usage_events WHERE org_id = %s"
OLD_KEY = "SELECT coalesce(sum(cost_micros), 0) FROM usage_events WHERE api_key_id = %s"


async def _scalar(pool, q: str, arg) -> int:  # type: ignore[no-untyped-def]
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(q, (arg,))
        row = await cur.fetchone()
    return int(row[0])


async def test_totals_equal_the_full_sums(pool, make_tenant):
    rng = random.Random(8)
    tenants = [await make_tenant(credit_micros=10**9) for _ in range(4)]
    keys = {t.org_id: [t.key_id] for t in tenants}
    async with pool.connection() as conn, conn.transaction():
        for t in tenants:
            k2, _ = await create_api_key(conn, t.org_id)
            keys[t.org_id].append(k2)

    async def check() -> None:
        for t in tenants:
            # Posted usage comes from the rollup, not from re-summing the events.
            rollup = await _scalar(
                pool,
                "SELECT coalesce((SELECT posted_micros FROM usage_posted_org "
                "WHERE org_id = %s), 0)",
                t.org_id,
            )
            posted = await _scalar(
                pool, OLD_ORG + " AND ledger_transaction_id IS NOT NULL", t.org_id
            )
            assert rollup == posted
            new = await _scalar(pool, "SELECT loom_org_usage_total(%s)", t.org_id)
            assert new == await _scalar(pool, OLD_ORG, t.org_id)
            for k in keys[t.org_id]:
                new = await _scalar(pool, "SELECT loom_key_usage_total(%s)", k)
                assert new == await _scalar(pool, OLD_KEY, k)

    now = dt.datetime.now(dt.UTC)
    for step in range(6):
        events = []
        for _ in range(rng.randint(5, 40)):
            t = rng.choice(tenants)
            k = rng.choice(keys[t.org_id])
            ev = make_event(
                t,
                prompt=rng.choice([0, rng.randint(1, 50_000)]),
                completion=rng.choice([0, rng.randint(1, 5_000)]),
                created_at=now - dt.timedelta(minutes=rng.randint(0, 600)),
            )
            events.append(ev.model_copy(update={"api_key_id": k}))
        async with pool.connection() as conn, conn.transaction():
            await insert_usage_events(conn, events)
        await check()
        if step % 2 == 0:
            await run_batch(pool, now=now + dt.timedelta(minutes=step))
            await check()

    # Edits the ledger never makes, still kept equal: unlink one posted event, delete one.
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "SELECT id FROM usage_events WHERE ledger_transaction_id IS NOT NULL "
            "AND cost_micros > 0 LIMIT 2"
        )
        ids = [r[0] for r in await cur.fetchall()]
        await cur.execute(
            "UPDATE usage_events SET ledger_transaction_id = NULL WHERE id = %s", (ids[0],)
        )
        await cur.execute("DELETE FROM usage_events WHERE id = %s", (ids[1],))
    await check()
