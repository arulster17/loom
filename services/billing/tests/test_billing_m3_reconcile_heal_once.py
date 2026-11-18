"""B1 (plan R5, docs/m3-review-fixes-spec.md): reconcilers that both observed the same
`redis_behind` drift heal it once in total, whether their heals run one after the other
(in either order) or concurrently. Redis `usage` ends at the Postgres total P that the
runs read plus the settles made since, never more.

Each reconciler gets a client (`HealGate`) that holds its heal, the server-side script
call for the org, until the test lets it go: both runs have read the org's usage and P
before either heals."""

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


@pytest.fixture
async def drifted(pool, redis_client, stream, make_tenant, make_consumer):
    """An org loaded in Redis, its stream consumed, then usage Redis lost:
    (tenant, limiter, P)."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    for i in range(2):
        await settle(limiter, t, prompt=1000 + i)
    await consumer.drain()
    lost = await lose_in_redis(pool, t)
    p = await pg_usage(pool, t.org_id)
    assert p - await redis_usage(redis_client, t.org_id) == lost.cost_micros
    return t, limiter, p


def reconciler(pool, client, stream) -> Reconciler:
    return Reconciler(pool, client, stream=stream, group=GROUP)


async def _finish(*pairs: tuple[HealGate, asyncio.Task]) -> None:
    for gate, _ in pairs:
        gate.release.set()
    for _, task in pairs:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 15)


@pytest.mark.parametrize("first", ["first_observer", "second_observer"])
async def test_item_b1_two_observers_heal_one_after_the_other(
    pool, redis_client, stream, drifted, first
):
    """done-when B1: both observed the drift; their heals run one after the other, in
    either order, with settles before and between them."""
    t, limiter, p = drifted
    g1, g2 = HealGate(redis_client, t.org_id), HealGate(redis_client, t.org_id)
    run1 = asyncio.create_task(reconciler(pool, g1, stream).run([t.org_id]))
    run2 = None
    try:
        await wait_reached(g1, run1)
        run2 = asyncio.create_task(reconciler(pool, g2, stream).run([t.org_id]))
        await wait_reached(g2, run2)

        s1 = await settle(limiter, t, prompt=2000)  # after both observed
        order = [(g1, run1), (g2, run2)]
        if first == "second_observer":
            order.reverse()
        (ga, ra), (gb, rb) = order
        ga.release.set()
        await asyncio.wait_for(ra, 15)
        s2 = await settle(limiter, t, prompt=3000)  # between the two heals
        gb.release.set()
        await asyncio.wait_for(rb, 15)

        assert await redis_usage(redis_client, t.org_id) == p + s1 + s2
    finally:
        await _finish((g1, run1), *([(g2, run2)] if run2 is not None else []))


async def test_item_b1_two_observers_heal_concurrently(pool, redis_client, stream, drifted):
    """done-when B1: both observed the drift; both heals are sent at once."""
    t, limiter, p = drifted
    g1, g2 = HealGate(redis_client, t.org_id), HealGate(redis_client, t.org_id)
    run1 = asyncio.create_task(reconciler(pool, g1, stream).run([t.org_id]))
    run2 = asyncio.create_task(reconciler(pool, g2, stream).run([t.org_id]))
    try:
        await wait_reached(g1, run1)
        await wait_reached(g2, run2)
        s1 = await settle(limiter, t, prompt=2000)
        g1.release.set()
        g2.release.set()
        await asyncio.wait_for(asyncio.gather(run1, run2), 15)

        assert await redis_usage(redis_client, t.org_id) == p + s1
    finally:
        await _finish((g1, run1), (g2, run2))


async def test_item_b1_overlapping_run_heals_first(pool, redis_client, stream, drifted):
    """done-when B1: one reconciler observed the drift; another worker's whole run
    (observe and heal) happens before the first one's heal."""
    t, limiter, p = drifted
    g1 = HealGate(redis_client, t.org_id)
    run1 = asyncio.create_task(reconciler(pool, g1, stream).run([t.org_id]))
    try:
        await wait_reached(g1, run1)
        await reconciler(pool, redis_client, stream).run([t.org_id])
        assert await redis_usage(redis_client, t.org_id) == p
        s1 = await settle(limiter, t, prompt=2000)
        g1.release.set()
        await asyncio.wait_for(run1, 15)

        assert await redis_usage(redis_client, t.org_id) == p + s1
    finally:
        await _finish((g1, run1))


async def test_item_b1_concurrent_runs_never_overheal(pool, redis_client, stream, drifted):
    """done-when B1, without gates: two reconcile runs started together, several times
    over, each time after Redis lost a new amount. Usage ends at P each time."""
    t, _, _ = drifted
    r1 = reconciler(pool, redis_client, stream)
    r2 = reconciler(pool, redis_client, stream)
    for i in range(5):
        if i:
            await lose_in_redis(pool, t, prompt=100_000 + i)
        p = await pg_usage(pool, t.org_id)
        await asyncio.gather(r1.run([t.org_id]), r2.run([t.org_id]))
        assert await redis_usage(redis_client, t.org_id) == p, f"round {i}"
