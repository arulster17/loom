"""B1 after review (docs/m3-review-fixes-spec.md, "Billing", B1): reconcilers that
observed the same `redis_behind` drift but read *different* Postgres totals P heal it at
most once. Redis `usage` never ends above the true total, and settles made between the
snapshot and the heal stay counted.

The review's case, made deterministic: Redis lost an amount L; one settle X is still in
the stream. Reconciler A runs while X is unconsumed (its P lacks X) and is held at its
heal (`HealGate`). Then the consumer commits X, and reconciler B runs (its P includes X)
and is held at its heal too. A's missing amount is L - X, B's is L. With the old guard
("add while usage < P"), A healing first leaves usage below B's P, so B adds L again:
the true total plus L - X.

Amounts: X is much larger than the settles made during the test (s1 + s2 < X), so the
old guard's second heal is not masked by them, and L > X + s1."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from billing_fixtures import GROUP
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

LOST_PROMPT = 500_000  # L
PENDING_PROMPT = 200_000  # X: in the stream when A reads P, stored before B does


def reconciler(pool, client, stream, drain_wait_s: float = 0.2) -> Reconciler:
    return Reconciler(pool, client, stream=stream, group=GROUP, drain_wait_s=drain_wait_s)


@pytest.fixture
async def observed(pool, redis_client, stream, make_tenant, make_consumer):
    """Two gated reconcilers that observed one redis_behind drift with different P.

    Returns a dict: tenant, limiter, consumer, the gates and tasks of A (smaller P) and
    B (larger P), `true` (the true total when both are held: everything settled plus
    what Redis lost) and `observed_usage` (Redis usage when both are held)."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    for i in range(2):
        await settle(limiter, t, prompt=1000 + i)
    await consumer.drain()
    lost = await lose_in_redis(pool, t, prompt=LOST_PROMPT)
    x = await settle(limiter, t, prompt=PENDING_PROMPT)  # counted in Redis, not stored
    p_a = await pg_usage(pool, t.org_id)
    usage = await redis_usage(redis_client, t.org_id)
    assert p_a - usage == lost.cost_micros - x

    ga, gb = HealGate(redis_client, t.org_id), HealGate(redis_client, t.org_id)
    tasks: list[tuple[HealGate, asyncio.Task]] = []
    try:
        run_a = asyncio.create_task(reconciler(pool, ga, stream).run([t.org_id]))
        tasks.append((ga, run_a))
        await wait_reached(ga, run_a)  # A read P without X
        await consumer.drain()  # the consumer commits X between the two P reads
        p_b = await pg_usage(pool, t.org_id)
        assert p_b == p_a + x
        run_b = asyncio.create_task(reconciler(pool, gb, stream).run([t.org_id]))
        tasks.append((gb, run_b))
        await wait_reached(gb, run_b)  # B read P with X
        assert await redis_usage(redis_client, t.org_id) == usage, "healed before release"
        yield {
            "t": t,
            "limiter": limiter,
            "consumer": consumer,
            "a": (ga, run_a),
            "b": (gb, run_b),
            "true": p_b,
            "observed_usage": usage,
            "lost": lost.cost_micros,
            "x": x,
        }
    finally:
        for gate, _ in tasks:
            gate.release.set()
        for _, task in tasks:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, 15)


async def _converges(pool, redis_client, stream, o) -> None:
    """The next run reads afresh: with the stream consumed, usage ends at Postgres's."""
    await o["consumer"].drain()
    await reconciler(pool, redis_client, stream).run([o["t"].org_id])
    assert await redis_usage(redis_client, o["t"].org_id) == await pg_usage(pool, o["t"].org_id)


@pytest.mark.parametrize("first", ["smaller_p_first", "larger_p_first"])
async def test_item_b1_different_p_heal_one_after_the_other(
    pool, redis_client, stream, observed, first
):
    """done-when B1: the two reconcilers read different P (980/1000 in the review's
    example); their heals run one after the other, in either order, with a settle after
    both observed and one between the heals. Usage never ends above the true total, the
    settles stay counted, and the next run converges on Postgres."""
    o = observed
    t, limiter = o["t"], o["limiter"]
    s1 = await settle(limiter, t, prompt=2000)  # after both observed
    order = [o["a"], o["b"]] if first == "smaller_p_first" else [o["b"], o["a"]]
    (g1, r1), (g2, r2) = order
    g1.release.set()
    await asyncio.wait_for(r1, 15)
    s2 = await settle(limiter, t, prompt=3000)  # between the two heals
    g2.release.set()
    await asyncio.wait_for(r2, 15)

    usage = await redis_usage(redis_client, t.org_id)
    true_total = o["true"] + s1 + s2
    assert usage <= true_total, (
        f"over-healed: usage {usage} above the true total {true_total} by "
        f"{usage - true_total} (the missing amount was added twice)"
    )
    assert usage >= o["observed_usage"] + s1 + s2, "a settle made since the snapshot was lost"
    await _converges(pool, redis_client, stream, o)


async def test_item_b1_different_p_heal_concurrently(pool, redis_client, stream, observed):
    """done-when B1: different P, both heals sent at once (A's released first, in the
    same tick). Usage never ends above the true total; the settle stays counted."""
    o = observed
    t, limiter = o["t"], o["limiter"]
    s1 = await settle(limiter, t, prompt=2000)
    (ga, ra), (gb, rb) = o["a"], o["b"]
    ga.release.set()
    gb.release.set()
    await asyncio.wait_for(asyncio.gather(ra, rb), 15)

    usage = await redis_usage(redis_client, t.org_id)
    true_total = o["true"] + s1
    assert usage <= true_total, f"over-healed: usage {usage} above the true total {true_total}"
    assert usage >= o["observed_usage"] + s1, "the settle made since the snapshot was lost"
    await _converges(pool, redis_client, stream, o)


async def test_item_b1_different_p_settles_between_snapshot_and_heal_stay_counted(
    pool, redis_client, stream, observed
):
    """done-when B1 ("settles since stay counted"): several settles land between the
    snapshots and the heals (some stored by the consumer meanwhile). The larger-P
    reconciler heals first; the end usage is exactly the true total, every settle
    counted once and the missing amount added once."""
    o = observed
    t, limiter, consumer = o["t"], o["limiter"], o["consumer"]
    since = 0
    for i in range(3):
        since += await settle(limiter, t, prompt=1100 + i)
    await consumer.drain()  # stored too: Postgres moves past both reconcilers' P
    since += await settle(limiter, t, prompt=1200)
    (ga, ra), (gb, rb) = o["a"], o["b"]
    gb.release.set()
    await asyncio.wait_for(rb, 15)
    ga.release.set()
    await asyncio.wait_for(ra, 15)

    assert await redis_usage(redis_client, t.org_id) == o["true"] + since
