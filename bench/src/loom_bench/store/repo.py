"""Thin write/read helpers over the results tables.

Every function takes the caller's Session and flushes but never commits, so a
caller groups writes with `session_scope()`. Pydantic models passed for JSON
columns are stored as `model_dump(mode="json")`.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from loom_bench.money import Micros
from loom_bench.provenance import GitInfo, config_hash, to_jsonable
from loom_bench.records import LoadMode
from loom_bench.store.models import (
    TERMINAL_EXPERIMENT_STATUSES,
    BenchColdStart,
    BenchEvalRun,
    BenchExperiment,
    BenchGateDecision,
    BenchResource,
    BenchRun,
    BenchSpend,
    ColdStartKind,
    ExperimentStatus,
    GateDecision,
    TerminatedBy,
    WaitlistSignup,
    utcnow,
)

JSONDoc = Mapping[str, Any] | BaseModel

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _doc(obj: JSONDoc) -> dict[str, Any]:
    out = to_jsonable(obj)
    if not isinstance(out, dict):
        raise TypeError(f"expected a JSON object, got {type(obj).__name__}")
    return out


def _enum[E: StrEnum](cls: type[E], value: E | str) -> str:
    return cls(value).value


def _micros(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"money must be integer micro-dollars, got {value!r}")
    return value


def create_experiment(
    session: Session,
    *,
    name: str,
    spec: JSONDoc,
    git: GitInfo,
    budget_micros: Micros | None,
    status: ExperimentStatus | str = ExperimentStatus.PLANNED,
    created_at: datetime | None = None,
) -> BenchExperiment:
    spec_doc = _doc(spec)
    exp = BenchExperiment(
        name=name,
        spec=spec_doc,
        spec_hash=config_hash(spec_doc),
        git_sha=git.sha,
        git_dirty=git.dirty,
        status=_enum(ExperimentStatus, status),
        budget_micros=None if budget_micros is None else _micros(budget_micros),
        spent_micros=0,
        created_at=created_at or utcnow(),
    )
    session.add(exp)
    session.flush()
    return exp


def _experiment(session: Session, experiment_id: uuid.UUID) -> BenchExperiment:
    exp = session.get(BenchExperiment, experiment_id)
    if exp is None:
        raise LookupError(f"no experiment {experiment_id}")
    return exp


def update_experiment_status(
    session: Session,
    experiment_id: uuid.UUID,
    status: ExperimentStatus | str,
    *,
    abort_reason: str | None = None,
    at: datetime | None = None,
) -> BenchExperiment:
    """Move an experiment to `status`; terminal statuses stamp finished_at and are final."""
    new = ExperimentStatus(status)
    exp = _experiment(session, experiment_id)
    if exp.status in TERMINAL_EXPERIMENT_STATUSES:
        raise ValueError(f"experiment {experiment_id} is already {exp.status}")
    exp.status = new.value
    if abort_reason is not None:
        exp.abort_reason = abort_reason
    if new in TERMINAL_EXPERIMENT_STATUSES:
        exp.finished_at = at or utcnow()
    session.flush()
    return exp


def add_spend(
    session: Session,
    experiment_id: uuid.UUID,
    amount_micros: Micros,
    *,
    basis: JSONDoc,
    resource_id: str | None = None,
    recorded_at: datetime | None = None,
) -> Micros:
    """Record spend and return the experiment's new running total.

    The total is bumped with a single UPDATE so concurrent writers never lose spend.
    """
    if _micros(amount_micros) < 0:
        raise ValueError("spend cannot be negative")
    total = session.execute(
        update(BenchExperiment)
        .where(BenchExperiment.id == experiment_id)
        .values(spent_micros=BenchExperiment.spent_micros + amount_micros)
        .returning(BenchExperiment.spent_micros)
    ).scalar_one_or_none()
    if total is None:
        raise LookupError(f"no experiment {experiment_id}")
    session.add(
        BenchSpend(
            experiment_id=experiment_id,
            resource_id=resource_id,
            amount_micros=amount_micros,
            basis=_doc(basis),
            recorded_at=recorded_at or utcnow(),
        )
    )
    session.flush()
    return total


def record_run(
    session: Session,
    *,
    experiment_id: uuid.UUID,
    config_hash: str,
    provenance: JSONDoc,
    status: str,
    summary: JSONDoc | None = None,
    cell_key: str | None = None,
    workload: str | None = None,
    load_mode: LoadMode | str | None = None,
    load_value: float | None = None,
    repetition: int | None = None,
    requests_uri: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    run_id: uuid.UUID | None = None,
) -> BenchRun:
    """Insert a run; `run_id` lets the caller fix the row id up front (the LoadJob's id), so
    files named after the run can be written before the row exists."""
    prov = _doc(provenance)
    if prov.get("config_hash", config_hash) != config_hash:
        raise ValueError("config_hash disagrees with the provenance record")
    run = BenchRun(
        id=run_id or uuid.uuid4(),
        experiment_id=experiment_id,
        cell_key=cell_key,
        config_hash=config_hash,
        workload=workload,
        load_mode=None if load_mode is None else _enum(LoadMode, load_mode),
        load_value=load_value,
        repetition=repetition,
        status=status,
        provenance=prov,
        summary=None if summary is None else _doc(summary),
        requests_uri=requests_uri,
        started_at=started_at,
        finished_at=finished_at,
    )
    session.add(run)
    session.flush()
    return run


def get_run(session: Session, run_id: uuid.UUID) -> BenchRun | None:
    return session.get(BenchRun, run_id)


# --- ids given by a person: full, or a unique prefix like git's short hashes ----------

MIN_ID_PREFIX = 4  # hex digits; shorter prefixes are refused rather than guessed
_HEX_REF = re.compile(r"^[0-9a-f-]+$")


def match_prefix(candidates: Sequence[str], ref: str, what: str) -> list[str]:
    """The candidates (uuids or hex digests) that start with `ref`, case and dashes
    ignored. `what` names the expected id in errors, e.g. "an experiment id"; text that
    is not hex, or shorter than `MIN_ID_PREFIX` digits, raises ValueError."""
    text = ref.strip().lower()
    if not _HEX_REF.match(text):
        raise ValueError(f"not {what}: {ref!r}")
    digits = text.replace("-", "")
    if len(digits) < MIN_ID_PREFIX:
        raise ValueError(
            f"{ref!r} is too short for {what}: give at least {MIN_ID_PREFIX} characters"
        )
    return [c for c in candidates if c.replace("-", "").lower().startswith(digits)]


def unique_match(matches: Sequence[str], ref: str, noun: str) -> str:
    """The one match of `ref`; LookupError when there is none or more than one."""
    if not matches:
        raise LookupError(f"no {noun} {ref}")
    if len(matches) > 1:
        shown = ", ".join(sorted(matches)[:5]) + (", ..." if len(matches) > 5 else "")
        raise LookupError(
            f"{ref!r} is ambiguous: it matches {len(matches)} ({shown}); give more characters"
        )
    return matches[0]


def _full_uuid(text: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(text.strip())
    except ValueError:
        return None


def _resolve_id(session: Session, column: Any, ref: str | uuid.UUID, noun: str) -> uuid.UUID:
    full = ref if isinstance(ref, uuid.UUID) else _full_uuid(ref)
    if full is not None:
        if session.scalar(select(column).where(column == full)) is None:
            raise LookupError(f"no {noun} {ref}")
        return full
    assert isinstance(ref, str)
    article = "an" if noun[0] in "aeiou" else "a"
    ids = [str(i) for i in session.scalars(select(column))]
    return uuid.UUID(unique_match(match_prefix(ids, ref, f"{article} {noun} id"), ref, noun))


def resolve_experiment_id(session: Session, ref: str | uuid.UUID) -> uuid.UUID:
    """An experiment id from its full form or a unique prefix (at least `MIN_ID_PREFIX`
    hex digits, e.g. "7a8237d0"). Raises ValueError for text that is not an id, and
    LookupError when no experiment or more than one matches."""
    return _resolve_id(session, BenchExperiment.id, ref, "experiment")


def resolve_run_id(session: Session, ref: str | uuid.UUID) -> uuid.UUID:
    """A run id from its full form or a unique prefix; errors as `resolve_experiment_id`."""
    return _resolve_id(session, BenchRun.id, ref, "run")


def list_runs(
    session: Session,
    *,
    experiment_id: uuid.UUID | None = None,
    config_hash: str | None = None,
) -> list[BenchRun]:
    stmt = select(BenchRun)
    if experiment_id is not None:
        stmt = stmt.where(BenchRun.experiment_id == experiment_id)
    if config_hash is not None:
        stmt = stmt.where(BenchRun.config_hash == config_hash)
    stmt = stmt.order_by(BenchRun.started_at.asc().nulls_last(), BenchRun.id)
    return list(session.scalars(stmt))


def list_cold_starts(session: Session, experiment_ids: Sequence[uuid.UUID]) -> list[BenchColdStart]:
    """Cold- and warm-start records of these experiments, oldest first."""
    if not experiment_ids:
        return []
    stmt = (
        select(BenchColdStart)
        .where(BenchColdStart.experiment_id.in_(list(experiment_ids)))
        .order_by(BenchColdStart.created_at, BenchColdStart.id)
    )
    return list(session.scalars(stmt))


def list_eval_runs(session: Session, experiment_ids: Sequence[uuid.UUID]) -> list[BenchEvalRun]:
    """Quality eval runs of these experiments, oldest first."""
    if not experiment_ids:
        return []
    stmt = (
        select(BenchEvalRun)
        .where(BenchEvalRun.experiment_id.in_(list(experiment_ids)))
        .order_by(BenchEvalRun.created_at, BenchEvalRun.id)
    )
    return list(session.scalars(stmt))


def list_gate_decisions(
    session: Session, experiment_ids: Sequence[uuid.UUID]
) -> list[BenchGateDecision]:
    """Quality gate decisions of these experiments, oldest first."""
    if not experiment_ids:
        return []
    stmt = (
        select(BenchGateDecision)
        .where(BenchGateDecision.experiment_id.in_(list(experiment_ids)))
        .order_by(BenchGateDecision.created_at, BenchGateDecision.id)
    )
    return list(session.scalars(stmt))


def record_cold_start(
    session: Session,
    *,
    experiment_id: uuid.UUID,
    kind: ColdStartKind | str,
    stages: Mapping[str, float],
    total_s: float,
    resource_id: str | None = None,
    config_hash: str | None = None,
    created_at: datetime | None = None,
) -> BenchColdStart:
    row = BenchColdStart(
        experiment_id=experiment_id,
        resource_id=resource_id,
        config_hash=config_hash,
        kind=_enum(ColdStartKind, kind),
        stages=_doc(stages),
        total_s=total_s,
        created_at=created_at or utcnow(),
    )
    session.add(row)
    session.flush()
    return row


def record_eval_run(
    session: Session,
    *,
    experiment_id: uuid.UUID,
    config_hash: str,
    task: str,
    task_version: str | None,
    n: int,
    score: float,
    ci_low: float | None,
    ci_high: float | None,
    provenance: JSONDoc,
    samples_uri: str | None = None,
    created_at: datetime | None = None,
) -> BenchEvalRun:
    row = BenchEvalRun(
        experiment_id=experiment_id,
        config_hash=config_hash,
        task=task,
        task_version=task_version,
        n=n,
        score=score,
        ci_low=ci_low,
        ci_high=ci_high,
        provenance=_doc(provenance),
        samples_uri=samples_uri,
        created_at=created_at or utcnow(),
    )
    session.add(row)
    session.flush()
    return row


def record_gate_decision(
    session: Session,
    *,
    experiment_id: uuid.UUID,
    baseline_config_hash: str,
    candidate_config_hash: str,
    decision: GateDecision | str,
    details: JSONDoc,
    created_at: datetime | None = None,
) -> BenchGateDecision:
    row = BenchGateDecision(
        experiment_id=experiment_id,
        baseline_config_hash=baseline_config_hash,
        candidate_config_hash=candidate_config_hash,
        decision=_enum(GateDecision, decision),
        details=_doc(details),
        created_at=created_at or utcnow(),
    )
    session.add(row)
    session.flush()
    return row


def record_resource(
    session: Session,
    *,
    provider: str,
    resource_type: str,
    resource_id: str,
    ttl_at: datetime,
    region: str | None = None,
    experiment_id: uuid.UUID | None = None,
    tags: Mapping[str, str] | None = None,
    created_at: datetime | None = None,
) -> BenchResource:
    row = BenchResource(
        provider=provider,
        resource_type=resource_type,
        resource_id=resource_id,
        region=region,
        experiment_id=experiment_id,
        tags=_doc(tags or {}),
        created_at=created_at or utcnow(),
        ttl_at=ttl_at,
    )
    session.add(row)
    session.flush()
    return row


def mark_terminated(
    session: Session,
    *,
    provider: str,
    resource_id: str,
    by: TerminatedBy | str,
    at: datetime | None = None,
) -> BenchResource:
    """Mark a resource terminated. The first termination wins; later calls keep it."""
    where = (BenchResource.provider == provider, BenchResource.resource_id == resource_id)
    session.execute(
        update(BenchResource)
        .where(*where, BenchResource.terminated_at.is_(None))
        .values(terminated_at=at or utcnow(), terminated_by=_enum(TerminatedBy, by))
        .execution_options(synchronize_session=False)
    )
    row = session.scalars(
        select(BenchResource).where(*where).execution_options(populate_existing=True)
    ).one_or_none()
    if row is None:
        raise LookupError(f"no {provider} resource {resource_id}")
    return row


def list_expired_resources(session: Session, now: datetime) -> list[BenchResource]:
    """Live resources whose TTL has passed, oldest deadline first."""
    stmt = (
        select(BenchResource)
        .where(BenchResource.terminated_at.is_(None), BenchResource.ttl_at <= now)
        .order_by(BenchResource.ttl_at, BenchResource.id)
    )
    return list(session.scalars(stmt))


def _find_signup(session: Session, email: str) -> WaitlistSignup | None:
    stmt = select(WaitlistSignup).where(func.lower(WaitlistSignup.email) == email.lower())
    return session.scalars(stmt).one_or_none()


def add_waitlist_signup(
    session: Session, email: str, *, source: str | None = None
) -> tuple[WaitlistSignup, bool]:
    """Add `email` unless it is already signed up (case-insensitive).

    Returns the signup row and whether it was created by this call.
    """
    email = email.strip()
    if not _EMAIL_RE.match(email):
        raise ValueError(f"not an email address: {email!r}")
    existing = _find_signup(session, email)
    if existing is not None:
        return existing, False
    signup = WaitlistSignup(email=email, source=source, created_at=utcnow())
    try:
        with session.begin_nested():
            session.add(signup)
    except IntegrityError:
        existing = _find_signup(session, email)
        if existing is None:
            raise
        return existing, False
    return signup, True
