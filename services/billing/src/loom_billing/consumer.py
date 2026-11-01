"""Usage stream consumer (docs/phase1-contracts.md §7).

At-least-once: entries are read with XREADGROUP, inserted into `usage_events` and
committed, and only then XACKed. A crash after the commit and before the ack redelivers
the entries; `ON CONFLICT (id, created_at) DO NOTHING` makes the second insert a no-op,
so a request is stored (and later billed) once.

Order of work:
1. once at start, this consumer's own pending entries (left by a crash of a process
   with the same consumer name);
2. each round, entries left pending for `claim_idle_ms` by any consumer, this one
   included (XAUTOCLAIM): a crashed peer's entries, and retries of entries left pending
   on purpose, so one stuck entry never blocks the rest;
3. then new entries (`>`), blocking up to `block_ms` only if 1-2 found nothing.

Failures:
- an entry that does not parse (`UsageEvent.from_stream_fields`) or that Postgres
  refuses as data (unknown org/key, out-of-range value) is poison: it is copied to the
  dead-letter stream with the reason and then acked. Never dropped silently.
- a missing `usage_events` partition leaves the entry pending (contracts §3) and counts
  `usage_insert_errors_total{reason="no_partition"}`; partition upkeep fixes it.
- Postgres or Redis being down raises out of the round; the worker backs off and
  retries. Nothing is acked, so nothing is lost.

The gateway only XADDs (inside settle.lua); nothing here can block inference.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import psycopg
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError
from redis.asyncio import Redis

from loom_billing import metrics
from loom_billing.logs import fields
from loom_billing.streams import ensure_group, format_id, group_state, next_id, s
from loom_platform.usage import UsageEvent, insert_usage_events

log = logging.getLogger("loom_billing.consumer")

Entry = tuple[str, dict[str, str] | None]


class LeavePending(Exception):
    """This entry cannot be stored now but may be later: do not ack it."""


@dataclass
class RoundResult:
    read: int = 0
    inserted: int = 0
    duplicate: int = 0
    dead: int = 0
    left_pending: int = 0
    acked: list[str] = field(default_factory=list)


def _decode(fields_raw: Any) -> dict[str, str] | None:
    if not fields_raw:
        return None
    return {s(k): s(v) for k, v in fields_raw.items()}


def _no_partition(e: psycopg.Error) -> bool:
    return "no partition of relation" in str(e)


class UsageConsumer:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        redis: Redis,
        *,
        stream: str,
        group: str,
        consumer: str,
        dead_stream: str,
        count: int = 500,
        block_ms: int = 2000,
        claim_idle_ms: int = 60_000,
    ) -> None:
        self.pool = pool
        self.redis = redis
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.dead_stream = dead_stream
        self.count = count
        self.block_ms = block_ms
        self.claim_idle_ms = claim_idle_ms
        self._group_ready = False
        # Cursor through this consumer's own pending entries at start; None when done.
        self._recovery_cursor: str | None = "0"

    async def setup(self) -> None:
        await ensure_group(self.redis, self.stream, self.group)
        self._group_ready = True

    # ------------------------------------------------------------------ reading

    async def _own_pending(self, after: str) -> list[Entry]:
        res = await self.redis.xreadgroup(
            self.group, self.consumer, {self.stream: after}, count=self.count
        )
        return self._entries(res)

    async def _claim_stale(self) -> list[Entry]:
        res = await self.redis.xautoclaim(
            self.stream,
            self.group,
            self.consumer,
            min_idle_time=self.claim_idle_ms,
            start_id="0-0",
            count=self.count,
        )
        # [next start, [(id, fields)...], [deleted ids]] (Redis 7)
        claimed = [(s(i), _decode(f)) for i, f in res[1]]
        deleted = [(s(i), None) for i in (res[2] if len(res) > 2 else [])]
        return claimed + deleted

    async def _new(self, block_ms: int | None) -> list[Entry]:
        res = await self.redis.xreadgroup(
            self.group, self.consumer, {self.stream: ">"}, count=self.count, block=block_ms
        )
        return self._entries(res)

    @staticmethod
    def _entries(res: Any) -> list[Entry]:
        out: list[Entry] = []
        for _stream, items in res or []:
            for i, f in items:
                out.append((s(i), _decode(f)))
        return out

    async def run_once(self, *, block_ms: int | None = None) -> RoundResult:
        """One round. `block_ms` None uses the configured block; 0 means don't block."""
        if not self._group_ready:
            await self.setup()
        result = RoundResult()
        old: list[Entry] = []
        if self._recovery_cursor is not None:
            old = await self._own_pending(self._recovery_cursor)
            self._recovery_cursor = old[-1][0] if old else None
        if not old:
            old = await self._claim_stale()
        if old:
            _merge(result, await self.process(old))
        # New entries in the same round, so retried entries never starve them. Block
        # only when there was nothing else to do.
        wait = self.block_ms if block_ms is None else block_ms
        new = await self._new(None if old else (wait or None))
        if new:
            _merge(result, await self.process(new))
        await self.trim_and_measure()
        return result

    async def drain(self, *, max_rounds: int = 1000) -> RoundResult:
        """Consume until nothing is left (tests, one-shot tools)."""
        total = RoundResult()
        for _ in range(max_rounds):
            r = await self.run_once(block_ms=0)
            _merge(total, r)
            if r.read == r.left_pending:
                break
        return total

    # --------------------------------------------------------------- processing

    async def process(self, entries: Sequence[Entry]) -> RoundResult:
        result = RoundResult(read=len(entries))
        good: list[tuple[str, UsageEvent]] = []
        to_ack: list[str] = []
        for entry_id, data in entries:
            if data is None:  # trimmed or deleted before we read it: nothing to store
                metrics.USAGE_ENTRIES.labels("deleted").inc()
                to_ack.append(entry_id)
                continue
            try:
                raw = cast(Mapping[bytes | str, bytes | str], data)
                good.append((entry_id, UsageEvent.from_stream_fields(raw)))
            except (ValidationError, ValueError) as e:
                await self._dead_letter(entry_id, data, f"invalid: {_short(e)}")
                result.dead += 1
                to_ack.append(entry_id)

        if good:
            try:
                inserted = await self._insert([ev for _, ev in good])
                to_ack += [i for i, _ in good]
                result.inserted += inserted
                result.duplicate += len(good) - inserted
            except (psycopg.IntegrityError, psycopg.DataError):
                # Someone in the batch is refused: find who, one transaction each.
                for entry_id, ev in good:
                    try:
                        n = await self._insert([ev])
                    except LeavePending:
                        result.left_pending += 1
                        continue
                    except (psycopg.IntegrityError, psycopg.DataError) as e:
                        data = next(d for i, d in entries if i == entry_id)
                        assert data is not None
                        await self._dead_letter(entry_id, data, f"refused: {_short(e)}")
                        result.dead += 1
                        to_ack.append(entry_id)
                        continue
                    to_ack.append(entry_id)
                    result.inserted += n
                    result.duplicate += 1 - n
            except LeavePending:
                # Only raised for a one-event batch; with several, the loop above runs.
                result.left_pending += len(good)

        if to_ack:
            await self.redis.xack(self.stream, self.group, *to_ack)
        result.acked = to_ack
        metrics.USAGE_ENTRIES.labels("inserted").inc(result.inserted)
        metrics.USAGE_ENTRIES.labels("duplicate").inc(result.duplicate)
        metrics.USAGE_ENTRIES.labels("dead_letter").inc(result.dead)
        if result.inserted or result.dead or result.left_pending:
            log.info(
                "usage.consumed",
                extra=fields(
                    read=result.read,
                    inserted=result.inserted,
                    duplicate=result.duplicate,
                    dead=result.dead,
                    left_pending=result.left_pending,
                ),
            )
        return result

    async def _insert(self, events: Sequence[UsageEvent]) -> int:
        """Insert and commit. Missing partition -> LeavePending (single event) or the
        IntegrityError that sends the caller to the one-by-one path."""
        try:
            async with self.pool.connection() as conn, conn.transaction():
                return await insert_usage_events(conn, events)
        except psycopg.errors.CheckViolation as e:
            if _no_partition(e) and len(events) == 1:
                metrics.USAGE_INSERT_ERRORS.labels("no_partition").inc()
                log.error(
                    "usage.no_partition",
                    extra=fields(event_id=str(events[0].id), created_at=events[0].created_at),
                )
                raise LeavePending() from e
            raise

    async def _dead_letter(self, entry_id: str, data: dict[str, str], reason: str) -> None:
        payload = {
            **data,
            "dead_reason": reason,
            "dead_source_id": entry_id,
            "dead_consumer": self.consumer,
            "dead_at": dt.datetime.now(dt.UTC).isoformat(timespec="milliseconds"),
        }
        await self.redis.xadd(self.dead_stream, payload)  # type: ignore[arg-type]
        log.error(
            "usage.dead_letter",
            extra=fields(stream_id=entry_id, reason=reason, dead_stream=self.dead_stream),
        )

    # ------------------------------------------------------------------ upkeep

    async def trim_and_measure(self) -> None:
        """XTRIM MINID below the oldest entry the group still needs: the oldest pending
        one, else the first not yet delivered."""
        st = await group_state(self.redis, self.stream, self.group)
        if st is None:
            return
        metrics.USAGE_PENDING.set(st.pending)
        if st.lag is not None:
            metrics.USAGE_LAG.set(st.lag)
        keep_from = st.oldest_pending or next_id(st.last_delivered)
        if keep_from > (0, 1):
            await self.redis.xtrim(self.stream, minid=format_id(keep_from), approximate=False)


def _merge(into: RoundResult, r: RoundResult) -> None:
    for name in ("read", "inserted", "duplicate", "dead", "left_pending"):
        setattr(into, name, getattr(into, name) + getattr(r, name))
    into.acked += r.acked


def _short(e: Exception) -> str:
    text = str(e).replace("\n", " ")
    return text[:300]
