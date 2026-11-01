"""Per-org per-minute usage ledger batches (docs/phase1-contracts.md §7).

Every minute, for each org with unposted usage, one transaction:
1. lock the org's unposted events (`ledger_transaction_id IS NULL AND cost_micros > 0`,
   FOR UPDATE),
2. post one `usage` ledger transaction for their exact sum (org_credit -X, revenue +X),
   keyed `usage:<org>:<minute of the run>`,
3. point each covered event at it (`usage_events.ledger_transaction_id`).

A crash rolls all three back together. A second worker racing on the same org blocks
on the row locks and then finds no unposted rows (Postgres re-checks the WHERE after the
first commits). A second run in the same minute (a restart) finds the key already used
and leaves the org to the next minute, so a key never needs two different sums.

Batching never touches Redis: both sides are already inside the org's `usage` counter
(contracts §6).
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

from loom_billing import metrics
from loom_billing.logs import fields
from loom_platform.ledger import (
    REVENUE,
    Entry,
    IdempotencyConflict,
    org_credit_account,
    post_transaction,
    usage_batch_key,
)

log = logging.getLogger("loom_billing.batch")


@dataclass(frozen=True)
class OrgBatch:
    org_id: uuid.UUID
    transaction_id: uuid.UUID
    events: int
    amount_micros: int


@dataclass
class BatchRun:
    minute: dt.datetime
    posted: list[OrgBatch] = field(default_factory=list)
    skipped: list[tuple[uuid.UUID, str]] = field(default_factory=list)


async def orgs_with_unposted(pool: AsyncConnectionPool) -> list[uuid.UUID]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT DISTINCT org_id FROM usage_events "
            "WHERE ledger_transaction_id IS NULL AND cost_micros > 0"
        )
        return [r[0] for r in await cur.fetchall()]


async def batch_org(
    pool: AsyncConnectionPool, org_id: uuid.UUID, minute: dt.datetime
) -> OrgBatch | str:
    """Post one org's batch. Returns the batch, or why it was skipped."""
    key = usage_batch_key(org_id, minute)
    try:
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                "SELECT 1 FROM ledger_transactions WHERE idempotency_key = %s", (key,)
            )
            if await cur.fetchone() is not None:
                return "minute_already_posted"
            await cur.execute(
                "SELECT id, created_at, cost_micros FROM usage_events "
                "WHERE org_id = %s AND ledger_transaction_id IS NULL AND cost_micros > 0 "
                "FOR UPDATE",
                (org_id,),
            )
            rows = await cur.fetchall()
            if not rows:
                return "nothing_unposted"
            total = sum(int(r[2]) for r in rows)
            account = await org_credit_account(conn, org_id)
            posted = await post_transaction(
                conn,
                kind="usage",
                idempotency_key=key,
                org_id=org_id,
                entries=[Entry(account, -total), Entry(REVENUE, total)],
                memo=f"usage batch: {len(rows)} events",
            )
            if not posted.created:
                # Same key, same sum: another worker posted this exact batch and we only
                # saw its rows because ours is a stale snapshot. Roll back to be safe.
                raise IdempotencyConflict(key)
            ids = [r[0] for r in rows]
            created = [r[1] for r in rows]
            await cur.execute(
                "UPDATE usage_events u SET ledger_transaction_id = %s "
                "FROM unnest(%s::uuid[], %s::timestamptz[]) AS c(id, created_at) "
                "WHERE u.id = c.id AND u.created_at = c.created_at AND u.org_id = %s "
                "AND u.created_at BETWEEN %s AND %s AND u.ledger_transaction_id IS NULL",
                (posted.transaction_id, ids, created, org_id, min(created), max(created)),
            )
            if cur.rowcount != len(rows):
                raise RuntimeError(
                    f"usage batch {key}: updated {cur.rowcount} of {len(rows)} locked events"
                )
    except IdempotencyConflict:
        return "idempotency_conflict"
    return OrgBatch(org_id, posted.transaction_id, len(rows), total)


async def run_batch(pool: AsyncConnectionPool, *, now: dt.datetime | None = None) -> BatchRun:
    minute = (now or dt.datetime.now(dt.UTC)).astimezone(dt.UTC).replace(second=0, microsecond=0)
    run = BatchRun(minute)
    with metrics.BATCH_SECONDS.time():
        for org_id in await orgs_with_unposted(pool):
            res = await batch_org(pool, org_id, minute)
            if isinstance(res, str):
                run.skipped.append((org_id, res))
                metrics.BATCH_SKIPPED.labels(res).inc()
                log.info("batch.skipped", extra=fields(org_id=str(org_id), reason=res))
                continue
            run.posted.append(res)
            metrics.BATCH_TRANSACTIONS.inc()
            metrics.BATCH_EVENTS.inc(res.events)
            metrics.BATCH_MICROS.inc(res.amount_micros)
            log.info(
                "batch.posted",
                extra=fields(
                    org_id=str(org_id),
                    transaction_id=str(res.transaction_id),
                    events=res.events,
                    amount_micros=res.amount_micros,
                    minute=minute.isoformat(),
                ),
            )
    return run
