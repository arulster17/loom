"""B2 after review (docs/m3-review-fixes-spec.md, "Billing", B2): the drain wait
(`LOOM_RECONCILE_DRAIN_WAIT_S`) happens at most once per reconcile run, not once per
chunk of orgs, and not at all when the consumer group is missing (it cannot catch up
within the wait).

No public seam sets the chunk size, so the multi-chunk run reconciles more than two
chunks' worth of orgs (the reconciler works through 500 orgs at a time; 1,100 orgs here
make three chunks of any size up to 500). The runs are timed: a stalled run is compared
with the same run once the consumer has caught up, so the time the orgs themselves take
cancels out. Margins are a whole wait wide on each side."""

from __future__ import annotations

import time
import uuid

import pytest
from billing_fixtures import GROUP
from billing_m3_helpers import load, lose_in_redis, pg_usage, redis_usage, settle

from loom_billing.reconcile import Reconciler

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

BULK_ORGS = 1100
WAIT_S = 2.0


def reconciler(pool, redis_client, stream, wait_s: float) -> Reconciler:
    return Reconciler(pool, redis_client, stream=stream, group=GROUP, drain_wait_s=wait_s)


async def bulk_orgs(pool, n: int) -> list[uuid.UUID]:
    """`n` orgs that exist in Postgres (none loaded in Redis)."""
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO orgs (name, status) SELECT 'b2 review bulk', 'active' "
            "FROM generate_series(1, %s) RETURNING id",
            (n,),
        )
        return [r[0] for r in await cur.fetchall()]


async def test_item_b2_drain_wait_once_per_run_across_chunks(
    pool, redis_client, stream, make_tenant, make_consumer
):
    """done-when B2: a run over three chunks of orgs, with the consumer stalled behind
    an entry, waits once (about WAIT_S more than the same run with the consumer caught
    up), not once per chunk (about 3 x WAIT_S more)."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    await settle(limiter, t, prompt=1000)
    await consumer.drain()
    ids = [t.org_id, *await bulk_orgs(pool, BULK_ORGS)]
    await settle(limiter, t, prompt=1500)  # never consumed during the stalled run
    r = reconciler(pool, redis_client, stream, WAIT_S)

    start = time.monotonic()
    stalled = await r.run(ids)
    stalled_s = time.monotonic() - start
    await consumer.drain()
    start = time.monotonic()
    caught_up = await r.run(ids)
    caught_up_s = time.monotonic() - start

    assert stalled.orgs == caught_up.orgs == len(ids)
    assert not stalled.drift_checked and caught_up.drift_checked
    assert stalled_s >= WAIT_S, f"skipped after {stalled_s:.2f}s, before the {WAIT_S}s wait"
    extra = stalled_s - caught_up_s
    assert extra < 2 * WAIT_S, (
        f"the stalled run took {extra:.2f}s longer than the caught-up one "
        f"({stalled_s:.2f}s vs {caught_up_s:.2f}s): the {WAIT_S}s drain wait ran more "
        "than once in one run (once per chunk of orgs?)"
    )


async def test_item_b2_no_drain_wait_when_group_missing(pool, redis_client, stream, make_tenant):
    """done-when B2: the stream has entries but its consumer group is gone (never
    created here, as after Redis lost its data). The run does not wait for it, still
    heals redis_behind, and reports no missing_in_postgres."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    await settle(limiter, t, prompt=1500)  # in the stream; no group to consume it
    await lose_in_redis(pool, t)
    p = await pg_usage(pool, t.org_id)
    wait_s = 3.0

    start = time.monotonic()
    run = await reconciler(pool, redis_client, stream, wait_s).run([t.org_id])
    elapsed = time.monotonic() - start

    assert elapsed < wait_s / 2, (
        f"the run took {elapsed:.2f}s with the consumer group missing: it waited for a "
        f"group that cannot catch up (the {wait_s}s drain wait)"
    )
    directions = [d.direction for d in run.drifts if d.org_id == t.org_id]
    assert "missing_in_postgres" not in directions
    assert "redis_behind" in directions
    assert await redis_usage(redis_client, t.org_id) >= p
