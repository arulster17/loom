"""`serving_pods` queries (migration 0010; docs/phase1-m2-spec.md §5 "States and records").

A row is written **before** RunPod is called, so an ambiguous create can be found and a
restart adopts what it owns. Every timestamp written comes from the manager's injected
clock, never Postgres's `now()`, so timing rules are testable. No secret is ever stored.

Functions take a `psycopg.AsyncConnection` (rows as dicts) and do not commit unless they
say so; the manager runs each change in its own transaction.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import psycopg

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
    raise NotImplementedError("phase B")


async def update(conn: psycopg.AsyncConnection, row_id: uuid.UUID, **fields: Any) -> PodRow:
    """Set `fields` (column names of `serving_pods`) on one row; returns it."""
    raise NotImplementedError("phase B")


async def get(conn: psycopg.AsyncConnection, row_id: uuid.UUID) -> PodRow | None:
    raise NotImplementedError("phase B")


async def by_pod_id(conn: psycopg.AsyncConnection, pod_id: str) -> PodRow | None:
    raise NotImplementedError("phase B")


async def live_rows(conn: psycopg.AsyncConnection, model_id: str | None = None) -> list[PodRow]:
    """Rows not `terminated` (of one model, or all), oldest first."""
    raise NotImplementedError("phase B")


async def rows_since(conn: psycopg.AsyncConnection, since: dt.datetime) -> list[PodRow]:
    """Rows created at or after `since` or still running then (month-to-date spend, the
    replacements of the last hour)."""
    raise NotImplementedError("phase B")
