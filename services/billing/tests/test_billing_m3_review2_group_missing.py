"""B2 (docs/m3-review-fixes-spec.md, "Billing", B2): with the consumer group missing,
the reconciler cannot know whether Redis's usage reached Postgres yet, so it skips the
`missing_in_postgres` direction (it needs the group to have consumed through the
snapshot's last id), and it does not wait for the group (it cannot catch up within the
wait). `redis_behind` is still checked and healed.

The run includes an org whose Redis usage exceeds Postgres's (P < U1: a settle counted
in Redis, sitting in the stream, never consumed because there is no group). A reconciler
that ran the `missing_in_postgres` check anyway would report that org; one that waited
for the group would take the whole drain wait."""

from __future__ import annotations

import time

import pytest
from billing_fixtures import GROUP
from billing_m3_helpers import load, lose_in_redis, pg_usage, redis_usage, settle

from loom_billing.reconcile import Reconciler
from loom_billing.streams import group_state

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

WAIT_S = 8.0


async def test_item2_group_missing_no_missing_in_postgres_for_redis_ahead_and_no_wait(
    pool, redis_client, stream, make_tenant
):
    """done-when B2: the consumer group is missing; org A has Redis usage above Postgres's
    (P < U1), org B lost usage in Redis (P > U2). The run finishes well within the drain
    wait, reports no missing_in_postgres (for A or anyone), and still reports and heals
    B's redis_behind."""
    a = await make_tenant(credit_micros=100_000_000)
    limiter_a = await load(pool, redis_client, stream, a)
    await settle(limiter_a, a, prompt=1500)  # counted in Redis, in the stream, unconsumed
    b = await make_tenant(credit_micros=100_000_000)
    await load(pool, redis_client, stream, b)
    await lose_in_redis(pool, b)

    assert await group_state(redis_client, stream, GROUP) is None, "setup: the group exists"
    assert await redis_client.xlen(stream) > 0, "setup: the stream is empty"
    p_a = await pg_usage(pool, a.org_id)
    assert p_a < await redis_usage(redis_client, a.org_id), "setup: A is not P < U1"
    p_b = await pg_usage(pool, b.org_id)

    r = Reconciler(pool, redis_client, stream=stream, group=GROUP, drain_wait_s=WAIT_S)
    start = time.monotonic()
    run = await r.run([a.org_id, b.org_id])
    elapsed = time.monotonic() - start

    assert elapsed < WAIT_S / 2, (
        f"the run took {elapsed:.2f}s with the consumer group missing: it waited for a "
        f"group that cannot catch up (drain wait {WAIT_S}s)"
    )
    missing = [d for d in run.drifts if d.direction == "missing_in_postgres"]
    assert missing == [], (
        "missing_in_postgres reported while the consumer group is missing (the settle "
        f"may simply not be consumed yet): {missing}"
    )
    behind = [d.org_id for d in run.drifts if d.direction == "redis_behind"]
    assert b.org_id in behind
    assert await redis_usage(redis_client, b.org_id) >= p_b
