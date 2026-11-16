"""`serving_pods` queries (migration 0010; docs/phase1-m2-spec.md §5 "States and records").

A row is written **before** RunPod is called, so an ambiguous create can be found and a
restart adopts what it owns. Every timestamp written comes from the manager's injected
clock, never Postgres's `now()`, so timing rules are testable. No secret is ever stored.

Functions take a `psycopg.AsyncConnection` (rows as dicts) and do not commit unless they
say so; the manager runs each change in its own transaction.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

PodState = Literal["creating", "bootstrapping", "ready", "draining", "failed", "terminated"]
POD_STATES: tuple[PodState, ...] = (
    "creating",
    "bootstrapping",
    "ready",
    "draining",
    "failed",
    "terminated",
)
TerminatedBy = Literal["manager", "watchdog", "reaper", "admin", "runpod"]


@dataclass(frozen=True)
class PodRow:
    """One `serving_pods` row."""

    id: uuid.UUID
    model_id: str
    pod_name: str
    pod_id: str | None
    state: PodState
    created_at: dt.datetime
    gpu_type_id: str | None = None
    gpu_count: int | None = None
    data_center: str | None = None
    cost_per_hr_micros: int | None = None
    public_ip: str | None = None
    proxy_port: int | None = None
    ssh_port: int | None = None
    cert_not_after: dt.datetime | None = None
    lease_until: dt.datetime | None = None
    stages: dict[str, float] = field(default_factory=dict)
    failure: str | None = None
    terminated_by: TerminatedBy | None = None
    ready_at: dt.datetime | None = None
    drain_started_at: dt.datetime | None = None
    terminated_at: dt.datetime | None = None

    @property
    def live(self) -> bool:
        """Counts toward `max_replicas` (every state but terminated)."""
        return self.state != "terminated"

    @classmethod
    def from_db(cls, row: dict[str, Any]) -> PodRow:
        return cls(**row)


COLUMNS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(PodRow))
_UPDATABLE = frozenset(COLUMNS) - {"id", "model_id", "pod_name", "created_at"}
# The second key of the per-model transaction lock (two-key advisory locks live in their
# own key space, apart from the manager's single-writer lock).
_MODEL_LOCK = "loom-poolmgr-model"


def _row(row: dict[str, Any] | None) -> PodRow | None:
    return None if row is None else PodRow.from_db(row)


def _value(name: str, value: Any) -> Any:
    if name == "stages":
        return Jsonb(dict(value or {}))
    return value


async def insert_creating(
    conn: psycopg.AsyncConnection,
    *,
    model_id: str,
    pod_name: str,
    created_at: dt.datetime,
    max_replicas: int,
) -> PodRow | None:
    """Insert a `creating` row, unless the model already has `max_replicas` live rows:
    then None. The count and the insert run in one transaction under a per-model lock
    (`pg_advisory_xact_lock`), so concurrent creates never exceed the cap. Commits."""
    async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))", (_MODEL_LOCK, model_id)
        )
        await cur.execute(
            "SELECT count(*) AS n FROM serving_pods WHERE model_id = %s AND state <> 'terminated'",
            (model_id,),
        )
        counted = await cur.fetchone()
        if counted is not None and int(counted["n"]) >= max_replicas:
            return None
        await cur.execute(
            "INSERT INTO serving_pods (id, model_id, pod_name, state, created_at)"
            " VALUES (%s, %s, %s, 'creating', %s) RETURNING *",
            (uuid.uuid4(), model_id, pod_name, created_at),
        )
        inserted = _row(await cur.fetchone())
    if not conn.autocommit:
        await conn.commit()
    return inserted


async def update(conn: psycopg.AsyncConnection, row_id: uuid.UUID, **fields: Any) -> PodRow:
    """Set `fields` (column names of `serving_pods`) on one row; returns it."""
    bad = set(fields) - _UPDATABLE
    if bad or not fields:
        raise ValueError(f"cannot update serving_pods columns {sorted(bad) or '(none)'}")
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = %s").format(sql.Identifier(name)) for name in fields
    )
    query = sql.SQL("UPDATE serving_pods SET {} WHERE id = %s RETURNING *").format(assignments)
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, (*(_value(k, v) for k, v in fields.items()), row_id))
        row = _row(await cur.fetchone())
    if row is None:
        raise KeyError(f"no serving_pods row {row_id}")
    return row


async def _select(
    conn: psycopg.AsyncConnection, where: str, params: tuple[Any, ...]
) -> list[PodRow]:
    query = "SELECT * FROM serving_pods WHERE " + where + " ORDER BY created_at, pod_name"
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(query, params)  # type: ignore[arg-type]  # constant SQL
        return [PodRow.from_db(r) for r in await cur.fetchall()]


async def get(conn: psycopg.AsyncConnection, row_id: uuid.UUID) -> PodRow | None:
    rows = await _select(conn, "id = %s", (row_id,))
    return rows[0] if rows else None


async def by_pod_id(conn: psycopg.AsyncConnection, pod_id: str) -> PodRow | None:
    rows = await _select(conn, "pod_id = %s", (pod_id,))
    return rows[0] if rows else None


async def live_rows(conn: psycopg.AsyncConnection, model_id: str | None = None) -> list[PodRow]:
    """Rows not `terminated` (of one model, or all), oldest first."""
    if model_id is None:
        return await _select(conn, "state <> 'terminated'", ())
    return await _select(conn, "state <> 'terminated' AND model_id = %s", (model_id,))


async def rows_since(conn: psycopg.AsyncConnection, since: dt.datetime) -> list[PodRow]:
    """Rows created at or after `since` or still running then (month-to-date spend, the
    replacements of the last hour)."""
    return await _select(
        conn,
        "created_at >= %s OR terminated_at IS NULL OR terminated_at > %s",
        (since, since),
    )
