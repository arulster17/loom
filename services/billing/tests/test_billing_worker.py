"""The worker as deployed: settings -> Worker wiring, the consumer loop, the jobs, the
scheduler. Mirrors one minute of the real process against real Postgres and Redis."""

from __future__ import annotations

import asyncio
import os

import pytest
from billing_fixtures import make_event

from loom_billing.settings import BillingSettings
from loom_billing.worker import Worker, consume_forever, every
from loom_platform.ledger import org_balance
from loom_platform.limiter import Limiter, org_key
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


async def test_worker_minute(pg_url, pool, redis_client, stream, make_tenant):
    settings = BillingSettings(
        database_url=pg_url,
        redis_url=os.environ.get(REDIS_ENV) or DEFAULT_REDIS,
        usage_stream=stream,
        usage_dead_stream=stream + ":dead",
        notify_stream=stream + ":notify",
        billing_consumer="billing-test",
        usage_block_ms=50,
    )
    w = Worker(settings, pool, redis_client)
    await w.partitions()
    await w.consumer.setup()

    t = await make_tenant(credit_micros=1_005_000)
    limiter = Limiter(redis_client, usage_stream=stream)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    events = [make_event(t, prompt=20_000, completion=1000) for _ in range(5)]
    for ev in events:
        await limiter.settle(ev)
    total = sum(e.cost_micros for e in events)

    task = asyncio.create_task(consume_forever(w.consumer))
    try:
        for _ in range(100):
            if await redis_client.xlen(stream) == 0:
                break
            await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await w.batch()
    await w.reconcile()
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == 1_005_000 - total
    assert int(await redis_client.get(org_key(t.org_id, "credit"))) == 1_005_000
    # $1.005 - cost is under the $1 default threshold: one low-balance notification.
    assert 1_005_000 - total < settings.low_balance_micros
    # (The scan also reaches orgs other tests left loaded; look at ours.)
    notes = await redis_client.xrange(stream + ":notify")
    mine = [n[1][b"type"] for n in notes if n[1][b"org_id"] == str(t.org_id).encode()]
    assert mine == [b"low_balance"]
    await redis_client.delete(org_key(t.org_id, "lowbal"))


async def test_every_runs_on_the_tick_and_survives_errors():
    calls: list[int] = []

    async def job() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first run fails")

    task = asyncio.create_task(every("test", 0.05, job))
    await asyncio.sleep(0.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls) >= 3  # 8 ticks of 50 ms; the failing first run did not stop it


async def test_redis_timeouts_cover_the_blocking_read():
    """A frozen Redis fails calls within the socket timeout, but an idle XREADGROUP
    (blocking `usage_block_ms`) must not count as frozen."""
    from loom_billing.worker import redis_from_settings

    s = BillingSettings(env="dev", usage_block_ms=2000, redis_socket_timeout_s=2.0)
    assert s.redis_read_timeout_s == 5.0
    r = redis_from_settings(s)
    kw = r.connection_pool.connection_kwargs
    assert kw["socket_timeout"] == 5.0
    assert kw["socket_connect_timeout"] == s.redis_connect_timeout_s == 1.0
    await r.aclose()
