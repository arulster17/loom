"""B1 after the third review (docs/m3-review-fixes-spec.md, "Billing", B1): the heal no
longer requires `usage < P`, so a busy org still heals at its first run.

The spec's case: U2 = 900, P = 1000 (Redis lost 100 that Postgres holds), and 150 settles
between the reconciler's snapshot and its heal. The heal must still add the missing
amount (P - U2): Redis `usage` ends at 1150, the true total. The old guard ("only while
`usage < P`") saw 1050 >= 1000 and healed nothing, leaving Redis 100 under for good.

Made deterministic with real Redis, a real reconciler and real settles: the reconciler
observes the `redis_behind` drift and is held at its heal (`HealGate`); while it is held
one settle lands; then it is released. The amounts are relative to the fixtures: the
missing amount is one lost event (`lose_in_redis`), the settle is chosen larger than it
(the spec's case), equal to it (the boundary: `usage == P` at the heal), or smaller (a
guard: the old guard healed that case too)."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from billing_fixtures import GROUP, make_event
from billing_m3_helpers import (
    HealGate,
    load,
    lose_in_redis,
    pg_usage,
    redis_usage,
    settle,
    wait_reached,
)

from loom_billing.reconcile import Reconciler

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

LOST_PROMPT = 500_000  # the event Redis lost (its cost is the missing amount)
SETTLE_PROMPTS = {
    "settled_more_than_missing": 750_000,  # the spec's case (150 > 100)
    "settled_equal_to_missing": LOST_PROMPT,  # boundary: usage == P at the heal
    "settled_less_than_missing": 100_000,  # guard: healed by the old code too
}


@pytest.mark.parametrize("case", list(SETTLE_PROMPTS))
async def test_item1_busy_org_heals_at_first_run(
    pool, redis_client, stream, make_tenant, make_consumer, case
):
    """done-when B1 (busy org): U2 below P by the missing amount; a settle lands between
    the reconciler's snapshot and its heal. The heal still adds the missing amount, so
    Redis `usage` ends at the true total (P plus the settle), never above it."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    for i in range(2):
        await settle(limiter, t, prompt=1000 + i)
    await consumer.drain()
    lost = await lose_in_redis(pool, t, prompt=LOST_PROMPT)
    p = await pg_usage(pool, t.org_id)
    u2 = await redis_usage(redis_client, t.org_id)
    missing = p - u2
    assert missing == lost.cost_micros > 0, "the setup made no redis_behind drift"

    planned = make_event(t, prompt=SETTLE_PROMPTS[case]).cost_micros
    if case == "settled_more_than_missing":
        assert planned > missing
    elif case == "settled_equal_to_missing":
        assert planned == missing
    else:
        assert planned < missing

    gate = HealGate(redis_client, t.org_id)
    task = asyncio.create_task(
        Reconciler(pool, gate, stream=stream, group=GROUP, drain_wait_s=0.2).run([t.org_id])
    )
    try:
        await wait_reached(gate, task)
        assert await redis_usage(redis_client, t.org_id) == u2, "healed before release"
        s = await settle(limiter, t, prompt=SETTLE_PROMPTS[case])
        assert s == planned
        assert await redis_usage(redis_client, t.org_id) == u2 + s
    finally:
        gate.release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(task), 15)
    await asyncio.wait_for(task, 15)  # re-raise the reconciler's error, if any

    await consumer.drain()  # the settle stored: Postgres holds the true total
    true_total = await pg_usage(pool, t.org_id)
    assert true_total == p + s
    usage = await redis_usage(redis_client, t.org_id)
    assert usage <= true_total, (
        f"over-healed: usage {usage} above the true total {true_total} by {usage - true_total}"
    )
    assert usage == true_total, (
        f"a busy org was not healed at its first run: usage {usage}, true total "
        f"{true_total}, still {true_total - usage} under (missing {missing}, settled {s} "
        "between the snapshot and the heal; spec B1: the `usage < P` guard is dropped, "
        "the heal token alone decides)"
    )
