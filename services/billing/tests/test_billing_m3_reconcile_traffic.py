"""B2 (plan R6, docs/m3-review-fixes-spec.md): `redis_behind` heals under steady traffic.

The redis_behind check and heal run every reconcile, drained stream or not. The
missing_in_postgres check still needs the consumer group to have consumed through the
snapshot's last id: the reconciler polls `consumed_through` for up to
`LOOM_RECONCILE_DRAIN_WAIT_S` before skipping that direction for the minute.

The reconciler is the billing worker's (built from `BillingSettings`, so the wait comes
from the environment variable). "Steady traffic" is made deterministic by hooking the
drain check the spec names (`consumed_through`): the first check sees an entry not yet
consumed, and right after it the consumer catches up and a new entry is settled."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest
from billing_fixtures import GROUP, STUB
from billing_m3_helpers import load, lose_in_redis, pg_usage, redis_usage, settle

from loom_billing import reconcile, streams
from loom_billing.reconcile import Reconciler
from loom_billing.settings import BillingSettings
from loom_billing.worker import Worker
from loom_platform.limiter import org_key

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


@pytest.fixture
def make_reconciler(pool, redis_client, stream, pg_url, monkeypatch):
    def make(drain_wait_s: float) -> Reconciler:
        monkeypatch.setenv("LOOM_RECONCILE_DRAIN_WAIT_S", str(drain_wait_s))
        settings = BillingSettings(
            env="dev",
            database_url=pg_url,
            usage_stream=stream,
            usage_group=GROUP,
            notify_stream=stream + ":notify",
            **STUB,
        )
        return Worker(settings, pool, redis_client).reconciler

    return make


class AfterFirstDrainCheck:
    """Runs `action` once, right after the reconciler's first `consumed_through` call
    has answered (so that answer saw the stream as it was at the snapshot). Should the
    call not go through the patched name, `action` runs after `fallback_s` instead."""

    def __init__(self, monkeypatch, action: Callable[[], Awaitable[None]], fallback_s: float):
        self.action = action
        self.fallback_s = fallback_s
        self.calls = 0
        self._ran = False
        self._lock = asyncio.Lock()
        real = streams.consumed_through

        async def hooked(*args, **kw):
            answer = await real(*args, **kw)
            self.calls += 1
            await self._run()
            return answer

        monkeypatch.setattr(reconcile, "consumed_through", hooked, raising=False)
        monkeypatch.setattr(streams, "consumed_through", hooked)

    async def _run(self) -> None:
        async with self._lock:
            if not self._ran:
                self._ran = True
                await self.action()

    async def around(self, coro: Awaitable):
        async def fallback() -> None:
            await asyncio.sleep(self.fallback_s)
            await self._run()

        fb = asyncio.create_task(fallback())
        try:
            return await coro
        finally:
            fb.cancel()


@pytest.fixture
async def org(pool, redis_client, stream, make_tenant, make_consumer):
    """A loaded org whose stream is consumed: (tenant, limiter, consumer)."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    await settle(limiter, t, prompt=1000)
    await consumer.drain()
    return t, limiter, consumer


def directions(run, t) -> list[str]:
    return [d.direction for d in run.drifts if d.org_id == t.org_id]


async def test_item_b2_redis_behind_heals_at_first_run_under_steady_traffic(
    monkeypatch, pool, redis_client, org, make_reconciler
):
    """done-when B2: usage settling and being consumed continuously (an entry not yet
    consumed at the snapshot, a new one settled during the run); Redis lost an amount.
    The first reconcile run heals it."""
    t, limiter, consumer = org
    await lose_in_redis(pool, t)
    await settle(limiter, t, prompt=1500)  # in the stream at the snapshot
    p = await pg_usage(pool, t.org_id)
    added: list[int] = []

    async def traffic() -> None:
        await consumer.drain()  # the consumer catches up...
        added.append(await settle(limiter, t, prompt=1700))  # ...and usage keeps coming

    hook = AfterFirstDrainCheck(monkeypatch, traffic, fallback_s=0.2)
    run = await hook.around(make_reconciler(5.0).run([t.org_id]))

    assert "redis_behind" in directions(run, t)
    usage = await redis_usage(redis_client, t.org_id)
    true_total = await pg_usage(pool, t.org_id) + sum(added)  # stored + still in the stream
    assert usage >= p, "redis_behind not healed at the first run"
    assert usage <= true_total, "over-healed"


async def test_item_b2_redis_behind_heals_while_consumer_is_behind(
    pool, redis_client, org, make_reconciler
):
    """done-when B2: the consumer does not catch up within LOOM_RECONCILE_DRAIN_WAIT_S;
    the redis_behind heal still runs (P <= U2 holds regardless)."""
    t, limiter, _ = org
    await lose_in_redis(pool, t)
    pending = await settle(limiter, t, prompt=1500)  # never consumed during the run
    p = await pg_usage(pool, t.org_id)

    run = await make_reconciler(0.3).run([t.org_id])

    assert "redis_behind" in directions(run, t)
    usage = await redis_usage(redis_client, t.org_id)
    assert p <= usage <= p + pending


async def test_item_b2_missing_in_postgres_reported_when_consumer_catches_up(
    monkeypatch, pool, redis_client, org, make_reconciler
):
    """done-when B2: a real missing_in_postgres is still reported once the consumer
    catches up within the wait (it was behind at the snapshot)."""
    t, limiter, consumer = org
    await redis_client.incrby(org_key(t.org_id, "usage"), 4321)  # never reaches Postgres
    await settle(limiter, t, prompt=1500)  # in the stream at the snapshot

    hook = AfterFirstDrainCheck(monkeypatch, consumer.drain, fallback_s=0.5)
    start = time.monotonic()
    run = await hook.around(make_reconciler(10.0).run([t.org_id]))
    elapsed = time.monotonic() - start

    missing = [(d.direction, d.micros) for d in run.drifts if d.org_id == t.org_id]
    assert missing == [("missing_in_postgres", 4321)]
    assert elapsed < 5.0, f"kept waiting {elapsed:.1f}s after the consumer caught up"


async def test_item_b2_missing_in_postgres_skipped_when_consumer_stays_behind(
    pool, redis_client, org, make_reconciler
):
    """done-when B2 (guard): while the consumer has not consumed through the snapshot,
    an event counted in Redis but not yet stored is not reported as missing_in_postgres."""
    t, limiter, _ = org
    await settle(limiter, t, prompt=1500)  # counted in Redis, not yet in Postgres

    run = await make_reconciler(0.3).run([t.org_id])

    assert directions(run, t) == []
