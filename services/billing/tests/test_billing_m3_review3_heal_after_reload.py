"""B1 after the second review (docs/m3-review-fixes-spec.md, "Billing", B1): the heal needs
both guards, the generation unchanged since the snapshot *and* Redis `usage` still below
P. The generation alone lets a heal run after a Redis reload resets it (absent = 0):
900 -> reload to 1000 -> heal to 1100.

The second review's case, made deterministic: Redis lost an amount (`redis_behind`: U2
below P, the org's heal generation absent). A reconciler takes its snapshot and is held
at its heal (`HealGate`). Redis then loses all of the org's keys (the heal generation
too) and the org is reloaded from Postgres at the true total, either by `init_org` with
the totals read from Postgres or by the gateway's answer to a `noinit` reserve
(`Limiter.load_from_postgres`). Released, the held heal must do nothing: Redis `usage`
stays at the reloaded value (plus any settle made since) and never rises above the
true total."""

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
from loom_platform.limiter import Limiter, org_prefix

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


async def _lose_org_keys(redis_client, org) -> None:
    """Redis loses everything it held for the org (usage, credit, heal generation, keys)."""
    keys = [k async for k in redis_client.scan_iter(match=f"{org_prefix(org)}:*")]
    if keys:
        await redis_client.delete(*keys)
    assert await redis_client.exists(f"{org_prefix(org)}:usage") == 0


async def _reload(how: str, pool, limiter: Limiter, t) -> None:
    if how == "noinit_load":
        async with pool.connection() as conn:
            org_new, _ = await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    else:
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT loom_org_credit_total(%s), loom_org_usage_total(%s), "
                "loom_key_usage_total(%s), loom_org_credit_version(%s)",
                (t.org_id, t.org_id, t.key_id, t.org_id),
            )
            row = await cur.fetchone()
        assert row is not None
        credit, usage, key_usage, version = row
        org_new, _ = await limiter.init_org(
            org_id=t.org_id,
            key_id=t.key_id,
            credit_total=credit,
            usage_total=usage,
            key_usage_total=key_usage,
            credit_version=version,
        )
    assert org_new, "the org was not reloaded (its keys were still there)"


@pytest.fixture
async def held(pool, redis_client, stream, make_tenant, make_consumer):
    """A reconciler held at its heal after observing redis_behind drift (generation
    absent). Yields tenant, limiter, consumer, the gate and the reconciler's task."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    for i in range(2):
        await settle(limiter, t, prompt=1000 + i)
    await consumer.drain()
    await lose_in_redis(pool, t)
    p = await pg_usage(pool, t.org_id)
    u2 = await redis_usage(redis_client, t.org_id)
    assert u2 < p, "the setup made no redis_behind drift"

    gate = HealGate(redis_client, t.org_id)
    task = asyncio.create_task(
        Reconciler(pool, gate, stream=stream, group=GROUP, drain_wait_s=0.2).run([t.org_id])
    )
    try:
        await wait_reached(gate, task)
        assert await redis_usage(redis_client, t.org_id) == u2, "healed before release"
        yield {"t": t, "limiter": limiter, "consumer": consumer, "gate": gate, "task": task, "p": p}
    finally:
        gate.release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 15)


@pytest.mark.parametrize("how", ["init_org", "noinit_load"])
async def test_item1_reload_between_snapshot_and_heal_is_not_healed_again(
    pool, redis_client, held, how
):
    """done-when B1 (reload): Redis loses the org's keys after the reconciler's
    snapshot and the org is reloaded from Postgres at the true total. The held heal
    does nothing: usage stays at the reloaded value, never above the true total."""
    t, gate, task = held["t"], held["gate"], held["task"]
    await _lose_org_keys(redis_client, t.org_id)
    await _reload(how, pool, held["limiter"], t)
    reloaded = await redis_usage(redis_client, t.org_id)
    true_total = await pg_usage(pool, t.org_id)
    assert reloaded == true_total == held["p"]

    gate.release.set()
    await asyncio.wait_for(task, 15)

    usage = await redis_usage(redis_client, t.org_id)
    assert usage <= true_total, (
        f"over-healed after a reload: usage {usage} above the true total {true_total} by "
        f"{usage - true_total} (the heal ran on a reloaded org; spec B1: it needs usage < P)"
    )
    assert usage == reloaded, f"usage {usage} moved off the reloaded value {reloaded}"


async def test_item1_reload_then_settle_before_heal_stays_counted(pool, redis_client, held):
    """done-when B1 (reload, "settles since stay counted"): after the reload a settle
    lands before the held heal is released. The heal adds nothing; the settle is
    counted once; usage ends at the true total (reloaded value plus the settle)."""
    t, gate, task, limiter = held["t"], held["gate"], held["task"], held["limiter"]
    await _lose_org_keys(redis_client, t.org_id)
    await _reload("noinit_load", pool, limiter, t)
    reloaded = await redis_usage(redis_client, t.org_id)
    s = await settle(limiter, t, prompt=3000)

    gate.release.set()
    await asyncio.wait_for(task, 15)

    await held["consumer"].drain()  # the settle stored: Postgres holds the true total
    true_total = await pg_usage(pool, t.org_id)
    usage = await redis_usage(redis_client, t.org_id)
    assert true_total == reloaded + s
    assert usage <= true_total, (
        f"over-healed after a reload: usage {usage} above the true total {true_total} by "
        f"{usage - true_total}"
    )
    assert usage == true_total, f"the settle since the reload was lost: {usage} != {true_total}"
