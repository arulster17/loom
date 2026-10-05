"""Phase 0 results schema (docs/PLAN.md). Money is BIGINT micro-dollars.

Portable across Postgres and SQLite so unit tests need no server: JSON columns
are JSONB on Postgres, UUIDs native on Postgres, and timestamps always come
back as UTC-aware datetimes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Dialect,
    Double,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Text,
    TypeDecorator,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from loom_bench.records import LoadMode

JSONType = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator[datetime]):
    """timestamptz that rejects naive datetimes and always returns UTC."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(f"naive datetime is ambiguous: {value!r}")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:  # SQLite stores the UTC wall time without an offset
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def utcnow() -> datetime:
    return datetime.now(UTC)


class ExperimentStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


TERMINAL_EXPERIMENT_STATUSES = frozenset(
    {ExperimentStatus.COMPLETED, ExperimentStatus.ABORTED, ExperimentStatus.FAILED}
)


class ColdStartKind(StrEnum):
    COLD = "cold"
    WARM = "warm"


class GateDecision(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class TerminatedBy(StrEnum):
    RUNNER = "runner"
    REAPER = "reaper"
    SELF_TTL = "self-ttl"


def _one_of(column: str, values: type[StrEnum]) -> str:
    return f"{column} IN ({', '.join(repr(v.value) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {  # noqa: RUF012
        uuid.UUID: Uuid(),
        datetime: UTCDateTime(),
        dict[str, Any]: JSONType,
        float: Double(),
        int: Integer(),
        str: Text(),
    }


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4)


def _experiment_fk(*, nullable: bool = False) -> Mapped[Any]:
    return mapped_column(ForeignKey("bench_experiments.id"), nullable=nullable, index=True)


class BenchExperiment(Base):
    __tablename__ = "bench_experiments"
    __table_args__ = (
        CheckConstraint(_one_of("status", ExperimentStatus), name="status"),
        CheckConstraint("spent_micros >= 0", name="spent_nonnegative"),
    )

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str]
    spec: Mapped[dict[str, Any]]
    spec_hash: Mapped[str] = mapped_column(index=True)
    git_sha: Mapped[str | None]
    git_dirty: Mapped[bool | None]
    status: Mapped[str]
    budget_micros: Mapped[int | None] = mapped_column(BigInteger)
    spent_micros: Mapped[int] = mapped_column(BigInteger, default=0, server_default=text("0"))
    abort_reason: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None]


class BenchRun(Base):
    """One load point x one repetition."""

    __tablename__ = "bench_runs"
    __table_args__ = (
        CheckConstraint(_one_of("load_mode", LoadMode), name="load_mode"),
        Index(None, "experiment_id", "cell_key", "repetition"),
    )

    id: Mapped[uuid.UUID] = _pk()
    experiment_id: Mapped[uuid.UUID] = _experiment_fk()
    cell_key: Mapped[str | None]
    config_hash: Mapped[str] = mapped_column(index=True)
    workload: Mapped[str | None]
    load_mode: Mapped[str | None]
    load_value: Mapped[float | None]
    repetition: Mapped[int | None]
    status: Mapped[str]
    provenance: Mapped[dict[str, Any]]
    summary: Mapped[dict[str, Any] | None]
    requests_uri: Mapped[str | None]
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class BenchColdStart(Base):
    __tablename__ = "bench_cold_starts"
    __table_args__ = (CheckConstraint(_one_of("kind", ColdStartKind), name="kind"),)

    id: Mapped[uuid.UUID] = _pk()
    experiment_id: Mapped[uuid.UUID] = _experiment_fk()
    resource_id: Mapped[str | None]
    kind: Mapped[str]
    stages: Mapped[dict[str, Any]]  # stage name -> seconds
    total_s: Mapped[float]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class BenchEvalRun(Base):
    __tablename__ = "bench_eval_runs"
    __table_args__ = (Index(None, "config_hash", "task"),)

    id: Mapped[uuid.UUID] = _pk()
    experiment_id: Mapped[uuid.UUID] = _experiment_fk()
    config_hash: Mapped[str]
    task: Mapped[str]
    task_version: Mapped[str | None]
    n: Mapped[int]
    score: Mapped[float]
    ci_low: Mapped[float | None]
    ci_high: Mapped[float | None]
    provenance: Mapped[dict[str, Any]]
    samples_uri: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class BenchGateDecision(Base):
    __tablename__ = "bench_gate_decisions"
    __table_args__ = (
        CheckConstraint(_one_of("decision", GateDecision), name="decision"),
        Index(None, "candidate_config_hash"),
    )

    id: Mapped[uuid.UUID] = _pk()
    experiment_id: Mapped[uuid.UUID] = _experiment_fk()
    baseline_config_hash: Mapped[str]
    candidate_config_hash: Mapped[str]
    decision: Mapped[str]
    details: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class BenchResource(Base):
    """A cloud resource the Lab created; the reaper terminates live ones past ttl_at."""

    __tablename__ = "bench_resources"
    __table_args__ = (
        CheckConstraint(_one_of("terminated_by", TerminatedBy), name="terminated_by"),
        Index(None, "provider", "resource_id", unique=True),
        Index(
            "ix_bench_resources_live_ttl_at",
            "ttl_at",
            postgresql_where=text("terminated_at IS NULL"),
            sqlite_where=text("terminated_at IS NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = _pk()
    provider: Mapped[str]
    resource_type: Mapped[str]
    resource_id: Mapped[str]
    region: Mapped[str | None]
    experiment_id: Mapped[uuid.UUID | None] = _experiment_fk(nullable=True)
    tags: Mapped[dict[str, Any]]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    ttl_at: Mapped[datetime]
    terminated_at: Mapped[datetime | None]
    terminated_by: Mapped[str | None]


class BenchSpend(Base):
    __tablename__ = "bench_spend"
    __table_args__ = (CheckConstraint("amount_micros >= 0", name="amount_nonnegative"),)

    id: Mapped[uuid.UUID] = _pk()
    experiment_id: Mapped[uuid.UUID] = _experiment_fk()
    resource_id: Mapped[str | None]
    amount_micros: Mapped[int] = mapped_column(BigInteger)
    basis: Mapped[dict[str, Any]]  # price used, seconds, market
    recorded_at: Mapped[datetime] = mapped_column(default=utcnow)


class WaitlistSignup(Base):
    __tablename__ = "waitlist_signups"

    id: Mapped[uuid.UUID] = _pk()
    email: Mapped[str]
    source: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


Index("uq_waitlist_signups_email_lower", func.lower(WaitlistSignup.email), unique=True)
