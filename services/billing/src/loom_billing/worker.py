"""The billing worker: the consumer loop plus the periodic jobs, in one asyncio process.

| Task | When |
|---|---|
| usage consumer | continuously (XREADGROUP blocks up to LOOM_USAGE_BLOCK_MS) |
| per-minute ledger batch | each minute at :05 |
| reconcile (credit, drift, notifications) | each minute at :35 |
| partition upkeep | at start, then every LOOM_PARTITION_INTERVAL_S (a day) |
| loom_credit_changed listener | continuously |
| unapplied Stripe events re-run (webhooks.retry_unapplied) | each minute at :50 |

A failing job is logged and counted (`loom_billing_job_errors_total{job}`) and retried
at its next tick; the consumer backs off up to 30 s while Postgres or Redis is down.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from psycopg_pool import AsyncConnectionPool
from redis.asyncio import BlockingConnectionPool, Redis

from loom_billing import metrics
from loom_billing.batch import run_batch
from loom_billing.consumer import UsageConsumer
from loom_billing.notify import AutoRecharge, Notifier
from loom_billing.partitions import ensure_partitions
from loom_billing.reconcile import CreditListener, Reconciler
from loom_billing.settings import BillingSettings
from loom_billing.webhooks import retry_unapplied
from loom_platform.db import open_pool
from loom_platform.limiter import Limiter

log = logging.getLogger("loom_billing.worker")


def redis_from_settings(settings: BillingSettings) -> Redis:
    pool = BlockingConnectionPool.from_url(
        settings.redis_url,
        max_connections=16,
        timeout=10,
        socket_timeout=settings.redis_read_timeout_s,
        socket_connect_timeout=settings.redis_connect_timeout_s,
    )
    return Redis(connection_pool=pool)


async def every(
    name: str, interval_s: float, fn: Callable[[], Awaitable[object]], *, offset_s: float = 0.0
) -> None:
    """Run `fn` at `offset_s` past each multiple of `interval_s` (wall clock)."""
    while True:
        now = time.time()
        nxt = (now // interval_s + 1) * interval_s + offset_s
        if nxt - interval_s > now:
            nxt -= interval_s
        await asyncio.sleep(nxt - now)
        try:
            await fn()
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics.JOB_ERRORS.labels(name).inc()
            log.exception("job.failed", extra={"loom_fields": {"job": name}})


async def consume_forever(consumer: UsageConsumer) -> None:
    backoff = 0.5
    while True:
        try:
            await consumer.run_once()
            backoff = 0.5
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics.CONSUMER_ERRORS.inc()
            log.exception("consumer.round_failed")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


class Worker:
    def __init__(self, settings: BillingSettings, pool: AsyncConnectionPool, redis: Redis) -> None:
        self.settings = settings
        self.pool = pool
        self.redis = redis
        self.consumer = UsageConsumer(
            pool,
            redis,
            stream=settings.usage_stream,
            group=settings.usage_group,
            consumer=settings.consumer_name,
            dead_stream=settings.usage_dead_stream,
            count=settings.usage_read_count,
            block_ms=settings.usage_block_ms,
            claim_idle_ms=settings.usage_claim_idle_ms,
        )
        notifier = Notifier(
            redis, stream=settings.notify_stream, low_balance_micros=settings.low_balance_micros
        )
        self.reconciler = Reconciler(
            pool,
            redis,
            stream=settings.usage_stream,
            group=settings.usage_group,
            notifier=notifier,
            recharge=AutoRecharge(notifier, enabled=settings.auto_recharge_enabled),
        )
        self.listener = CreditListener(
            settings.database_url, pool, Limiter(redis, usage_stream=settings.usage_stream)
        )

    async def partitions(self) -> None:
        await ensure_partitions(self.pool, months=self.settings.partition_months_ahead)

    async def batch(self) -> None:
        await run_batch(self.pool)

    async def reconcile(self) -> None:
        await self.reconciler.run()

    async def stripe_retry(self) -> None:
        await retry_unapplied(self.pool)

    async def run(self) -> None:
        s = self.settings
        await self.partitions()
        await self.consumer.setup()
        log.info(
            "worker.started",
            extra={"loom_fields": {"consumer": s.consumer_name, "stream": s.usage_stream}},
        )
        async with asyncio.TaskGroup() as tg:
            tg.create_task(consume_forever(self.consumer))
            tg.create_task(self.listener.run())
            tg.create_task(every("batch", s.batch_interval_s, self.batch, offset_s=5))
            tg.create_task(every("reconcile", s.reconcile_interval_s, self.reconcile, offset_s=35))
            tg.create_task(every("partitions", s.partition_interval_s, self.partitions))
            retry = every("stripe_retry", s.reconcile_interval_s, self.stripe_retry, offset_s=50)
            tg.create_task(retry)


async def run_worker(settings: BillingSettings) -> None:
    pool = await open_pool(settings.database_url, max_size=8)
    redis = redis_from_settings(settings)
    try:
        await Worker(settings, pool, redis).run()
    finally:
        await pool.close()
        await redis.aclose()
