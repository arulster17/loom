"""B1 (docs/m3-review-fixes-spec.md, "Billing", B1): "An org loaded before this change
has no token; 'absent' is a value like any other." Reconcilers that observed the same
`redis_behind` drift of such an org heal it at most once in total, whether their heals
run concurrently or one after the other: Redis `usage` ends at the Postgres total P the
runs read plus the settles made since, never more, and a heal does happen.

The org is loaded as usual, then its heal-token key (contracts §6, `token_key`) is
deleted: the state an org loaded by the code before the token existed is in. Gated
reconcilers (`HealGate`) are held at their heal until both observed the drift."""

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
    token_key,
    wait_reached,
)

from loom_billing.reconcile import Reconciler

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


def reconciler(pool, client, stream) -> Reconciler:
    return Reconciler(pool, client, stream=stream, group=GROUP, drain_wait_s=0.2)


async def _drop_token(redis_client, org) -> None:
    """The org as the code before the heal token left it: loaded, no token."""
    await redis_client.delete(token_key(org))
    assert await redis_client.exists(token_key(org)) == 0


@pytest.fixture
async def drifted(pool, redis_client, stream, make_tenant, make_consumer):
    """A loaded org without a heal token, its stream consumed, then usage Redis lost:
    (tenant, limiter, P)."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    for i in range(2):
        await settle(limiter, t, prompt=1000 + i)
    await consumer.drain()
    lost = await lose_in_redis(pool, t)
    await _drop_token(redis_client, t.org_id)
    p = await pg_usage(pool, t.org_id)
    assert p - await redis_usage(redis_client, t.org_id) == lost.cost_micros
    return t, limiter, p


async def _finish(*pairs: tuple[HealGate, asyncio.Task]) -> None:
    for gate, _ in pairs:
        gate.release.set()
    for _, task in pairs:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 15)


async def test_item_b1_no_token_two_observers_heal_concurrently(
    pool, redis_client, stream, drifted
):
    """done-when B1 (no token): both reconcilers observed the drift of an org with no
    heal token (both read 'absent'); both heals are sent at once, with a settle after
    both observed. The missing amount is added once: usage ends at P plus the settle."""
    t, limiter, p = drifted
    g1, g2 = HealGate(redis_client, t.org_id), HealGate(redis_client, t.org_id)
    run1 = asyncio.create_task(reconciler(pool, g1, stream).run([t.org_id]))
    run2 = asyncio.create_task(reconciler(pool, g2, stream).run([t.org_id]))
    try:
        await wait_reached(g1, run1)
        await wait_reached(g2, run2)
        assert await redis_client.exists(token_key(t.org_id)) == 0, "a token appeared early"
        s1 = await settle(limiter, t, prompt=2000)
        g1.release.set()
        g2.release.set()
        await asyncio.wait_for(asyncio.gather(run1, run2), 15)

        usage = await redis_usage(redis_client, t.org_id)
        assert usage <= p + s1, (
            f"over-healed an org with no heal token: usage {usage} above the true total "
            f"{p + s1} by {usage - (p + s1)} (both heals applied: 'absent' must change "
            "with the first heal like any other token value)"
        )
        assert usage == p + s1, f"usage {usage}, expected {p + s1}: the drift was not healed"
        assert await redis_client.get(token_key(t.org_id)), "the heal set no fresh token"
    finally:
        await _finish((g1, run1), (g2, run2))


@pytest.mark.parametrize("first", ["first_observer", "second_observer"])
async def test_item_b1_no_token_two_observers_heal_one_after_the_other(
    pool, redis_client, stream, drifted, first
):
    """done-when B1 (no token): both observed the drift of an org with no heal token;
    their heals run one after the other, in either order, with settles before and
    between them. Usage ends at P plus both settles."""
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
        assert await redis_usage(redis_client, t.org_id) == p + s1, "the first heal failed"
        s2 = await settle(limiter, t, prompt=3000)  # between the two heals
        gb.release.set()
        await asyncio.wait_for(rb, 15)

        usage = await redis_usage(redis_client, t.org_id)
        assert usage == p + s1 + s2, (
            f"usage {usage}, expected {p + s1 + s2} (P plus both settles): the second "
            f"heal {'added the missing amount again' if usage > p + s1 + s2 else 'lost a settle'}"
        )
    finally:
        await _finish((g1, run1), *([(g2, run2)] if run2 is not None else []))


async def test_item_b1_no_token_concurrent_runs_never_overheal(pool, redis_client, stream, drifted):
    """done-when B1 (no token), without gates: two reconcile runs started together,
    several times over, each time after Redis lost a new amount and the org's token was
    removed again. Usage ends at P each time."""
    t, _, _ = drifted
    r1 = reconciler(pool, redis_client, stream)
    r2 = reconciler(pool, redis_client, stream)
    for i in range(5):
        if i:
            await lose_in_redis(pool, t, prompt=100_000 + i)
            await _drop_token(redis_client, t.org_id)
        p = await pg_usage(pool, t.org_id)
        assert await redis_usage(redis_client, t.org_id) < p, f"round {i}: no drift"
        await asyncio.gather(r1.run([t.org_id]), r2.run([t.org_id]))
        assert await redis_usage(redis_client, t.org_id) == p, f"round {i}"
