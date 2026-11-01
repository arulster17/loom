"""Balance reconcile (docs/phase1-contracts.md §6, §7).

Every minute, for each org loaded in Redis (`SCAN loom:org:*:credit`) that exists in
Postgres:

1. **Credit**: `Limiter.set_credit(org, credit total, version)`, both read in one
   snapshot. Only `credit` (and its version) is written, and never over a newer version,
   so a slow reconcile cannot undo a fresher refresh; `usage` (written by settle) is
   never touched here.
2. **Drift check** on `usage`, race-free:
   a. in one MULTI: read every org's Redis `usage` (U1) and the stream's last id (L);
   b. check the consumer group has delivered and acked everything up to L (else the
      check is skipped this minute: billing is behind, not wrong);
   c. read `loom_org_usage_total` (P) from Postgres, after (b);
   d. read Redis `usage` again (U2).
   Settle bumps `usage` and appends the event in one atomic call, and the consumer acks
   only after its commit, so a healthy system has U1 <= P <= U2. P < U1 means events
   counted in Redis never reached Postgres (lost or dead-lettered); P > U2 means Redis
   lost usage (e.g. AOF loss, or an org loaded while its events were still in the
   stream). Either is logged with the org id and counted in metrics; reconcile does
   not rewrite `usage`.
3. **Notifications**: balance = credit - U2 feeds the low-balance check and the
   auto-recharge hook (`loom_billing.notify`).

Orgs are not touched when absent from Postgres, so a Redis shared with another
database (tests) is left alone.

The `loom_credit_changed` listener (`CreditListener`) applies step 1 for one org as soon
as a purchase, promo, refund or adjustment commits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import psycopg
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from loom_billing import metrics
from loom_billing.logs import fields
from loom_billing.notify import AutoRecharge, Notifier
from loom_billing.streams import StreamId, consumed_through, parse_id, s
from loom_platform.db import libpq_url
from loom_platform.ledger import org_credit_state
from loom_platform.limiter import KEY_ROOT, Limiter, org_key

log = logging.getLogger("loom_billing.reconcile")

CHUNK = 500


@dataclass(frozen=True)
class Drift:
    org_id: uuid.UUID
    direction: str  # missing_in_postgres | redis_behind
    micros: int
    redis_usage: int
    postgres_usage: int


@dataclass
class ReconcileRun:
    orgs: int = 0
    credits_set: int = 0
    drift_checked: bool = False
    drifts: list[Drift] = field(default_factory=list)
    notifications: int = 0


async def loaded_orgs(redis: Redis) -> list[uuid.UUID]:
    """Org ids with a `credit` key in Redis."""
    out: set[uuid.UUID] = set()
    async for k in redis.scan_iter(match=f"{KEY_ROOT}:org:*:credit", count=1000):
        parts = s(k).split(":")
        if len(parts) != 4:
            continue
        try:
            out.add(uuid.UUID(parts[2]))
        except ValueError:
            continue
    return sorted(out)


class Reconciler:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        redis: Redis,
        *,
        stream: str,
        group: str,
        notifier: Notifier | None = None,
        recharge: AutoRecharge | None = None,
    ) -> None:
        self.pool = pool
        self.redis = redis
        self.limiter = Limiter(redis, usage_stream=stream)
        self.stream = stream
        self.group = group
        self.notifier = notifier
        self.recharge = recharge

    async def run(self, org_ids: Sequence[uuid.UUID] | None = None) -> ReconcileRun:
        """Reconcile `org_ids` (default: every org loaded in Redis)."""
        run = ReconcileRun()
        with metrics.RECONCILE_SECONDS.time():
            ids = list(org_ids) if org_ids is not None else await loaded_orgs(self.redis)
            missing = 0
            micros = {"missing_in_postgres": 0, "redis_behind": 0}
            for i in range(0, len(ids), CHUNK):
                await self._chunk(ids[i : i + CHUNK], run)
            for d in run.drifts:
                micros[d.direction] += d.micros
                missing += 1
            if run.drift_checked or run.drifts:
                metrics.DRIFT_ORGS.set(missing)
                for direction, v in micros.items():
                    metrics.DRIFT_MICROS.labels(direction).set(v)
        return run

    async def _chunk(self, ids: list[uuid.UUID], run: ReconcileRun) -> None:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT id FROM orgs WHERE id = ANY(%s)", (ids,))
                known = {r[0] for r in await cur.fetchall()}
            await conn.commit()
            unknown = [o for o in ids if o not in known]
            if unknown:
                log.debug("reconcile.unknown_orgs", extra=fields(count=len(unknown)))
            orgs = [o for o in ids if o in known]
            if not orgs:
                return
            run.orgs += len(orgs)

            # (a) U1 and the stream's last id, atomically.
            pipe = self.redis.pipeline(transaction=True)
            for o in orgs:
                pipe.get(org_key(o, "usage"))
            pipe.xrevrange(self.stream, count=1)
            res = await pipe.execute()
            u1 = [None if v is None else int(v) for v in res[:-1]]
            last: StreamId | None = parse_id(res[-1][0][0]) if res[-1] else None
            # (b) everything up to `last` consumed and committed?
            drained = await consumed_through(self.redis, self.stream, self.group, last)
            # (c) Postgres, one snapshot.
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT o.id, loom_org_credit_total(o.id), loom_org_usage_total(o.id), "
                    "loom_org_credit_version(o.id) FROM unnest(%s::uuid[]) AS o(id)",
                    (orgs,),
                )
                pg = {r[0]: (int(r[1]), int(r[2]), int(r[3])) for r in await cur.fetchall()}
            await conn.commit()
            # (d) U2.
            u2_raw = await self.redis.mget([org_key(o, "usage") for o in orgs])
            u2 = [None if v is None else int(v) for v in u2_raw]

            if drained:
                run.drift_checked = True
            else:
                metrics.DRIFT_CHECK_SKIPPED.inc()

            for o, before, after in zip(orgs, u1, u2, strict=True):
                credit, pg_usage, version = pg[o]
                if await self.limiter.set_credit(o, credit, version):
                    run.credits_set += 1
                    metrics.RECONCILE_ORGS.inc()
                if drained and before is not None and pg_usage < before:
                    self._drift(run, o, "missing_in_postgres", before - pg_usage, before, pg_usage)
                if drained and after is not None and pg_usage > after:
                    self._drift(run, o, "redis_behind", pg_usage - after, after, pg_usage)
                if after is None:
                    continue
                balance = credit - after
                if self.notifier is not None and await self.notifier.check_low_balance(o, balance):
                    run.notifications += 1
                if self.recharge is not None:
                    if await self.recharge.check(conn, o, balance):
                        run.notifications += 1
                    await conn.commit()

    def _drift(
        self, run: ReconcileRun, org: uuid.UUID, direction: str, micros: int, r: int, p: int
    ) -> None:
        run.drifts.append(Drift(org, direction, micros, r, p))
        metrics.DRIFT_DETECTED.labels(direction).inc()
        log.error(
            "reconcile.usage_drift",
            extra=fields(
                org_id=str(org),
                direction=direction,
                drift_micros=micros,
                redis_usage_micros=r,
                postgres_usage_micros=p,
            ),
        )


class CreditListener:
    """LISTEN loom_credit_changed; refresh that org's Redis credit on each notice.
    Reconnects with backoff; anything missed while down is fixed by the next minute's
    reconcile."""

    CHANNEL = "loom_credit_changed"

    def __init__(self, database_url: str, pool: AsyncConnectionPool, limiter: Limiter) -> None:
        self.database_url = database_url
        self.pool = pool
        self.limiter = limiter
        self.ready = asyncio.Event()

    async def apply(self, payload: str) -> bool:
        try:
            org_id = uuid.UUID(json.loads(payload)["org_id"])
        except (ValueError, KeyError, TypeError):
            log.warning("credit_listener.bad_payload")
            return False
        async with self.pool.connection() as conn:
            credit, version = await org_credit_state(conn, org_id)
        done = await self.limiter.set_credit(org_id, credit, version)
        metrics.CREDIT_NOTIFICATIONS.inc()
        log.info(
            "credit_listener.applied",
            extra=fields(org_id=str(org_id), credit_micros=credit, loaded=done),
        )
        return done

    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                conn = await psycopg.AsyncConnection.connect(
                    libpq_url(self.database_url), autocommit=True
                )
                async with conn:
                    await conn.execute(f"LISTEN {self.CHANNEL}")
                    self.ready.set()
                    backoff = 1.0
                    async for note in conn.notifies():
                        try:
                            await self.apply(note.payload)
                        except Exception:
                            log.exception("credit_listener.apply_failed")
            except asyncio.CancelledError:
                raise
            except Exception:
                self.ready.clear()
                log.exception("credit_listener.disconnected")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
