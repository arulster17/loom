"""Monthly `usage_events` partitions: the worker creates the current month and the next
`months - 1` at start and once a day (contracts §3). Migration 0004 already made a year
ahead, so this only keeps the window rolling."""

from __future__ import annotations

import datetime as dt
import logging

from psycopg_pool import AsyncConnectionPool

from loom_billing import metrics
from loom_billing.logs import fields

log = logging.getLogger("loom_billing.partitions")


async def ensure_partitions(
    pool: AsyncConnectionPool, *, months: int = 3, today: dt.date | None = None
) -> int:
    """Create missing monthly partitions from `today`'s month (UTC). Returns how many."""
    start = today or dt.datetime.now(dt.UTC).date()
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        await cur.execute("SELECT loom_ensure_usage_partitions(%s, %s)", (start, months))
        row = await cur.fetchone()
    created = int(row[0]) if row else 0
    if created:
        metrics.PARTITIONS_CREATED.inc(created)
        log.info("partitions.created", extra=fields(created=created, start=start.isoformat()))
    return created
