"""Spend caps (`bench/budget.yaml`) and the runtime budget guard.

The guard accrues Σ host.hourly_micros × elapsed into `bench_spend` every
`interval_s`. Before each step the runner asks whether current spend plus the
step's estimate fits (`check_next`, graceful stop if not). When recorded spend
reaches the cap the guard trips: the in-flight provider call is cancelled and
the runner tears every host down (hard abort).

Worst-case overshoot past the cap is one accrual interval of every live host's
hourly price, plus the teardown time the guard still records afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from loom_bench.experiment import UsdMicros
from loom_bench.money import Micros, cost_for_seconds, format_usd
from loom_bench.providers.base import Host
from loom_bench.registry import REPO_ROOT, StrictModel, read_yaml
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment

DEFAULT_BUDGET_YAML = REPO_ROOT / "bench" / "budget.yaml"
BUDGET_YAML_ENV = "LOOM_BUDGET_YAML"

# Spend from these provider kinds is simulated money and never counts toward the overall cap.
SIMULATED_PROVIDERS = frozenset({"mock"})


class BudgetConfig(StrictModel):
    overall_cap: UsdMicros
    per_experiment_cap: UsdMicros

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.per_experiment_cap > self.overall_cap:
            raise ValueError("per_experiment_cap must not exceed overall_cap")
        return self


def load_budget(path: Path | str | None = None) -> BudgetConfig:
    if path is None:
        path = os.environ.get(BUDGET_YAML_ENV) or DEFAULT_BUDGET_YAML
    return BudgetConfig.model_validate(read_yaml(Path(path)))


def is_billable(provider_kind: str) -> bool:
    return provider_kind not in SIMULATED_PROVIDERS


def billable_spend(session: Session, *, exclude: uuid.UUID | None = None) -> Micros:
    """Real (non-simulated) spend recorded across all experiments."""
    rows = session.execute(
        select(BenchExperiment.id, BenchExperiment.spec, BenchExperiment.spent_micros)
    )
    total = 0
    for exp_id, spec, spent in rows:
        if exp_id == exclude:
            continue
        kind = (spec or {}).get("provider", {}).get("kind")
        if kind is None or is_billable(kind):
            total += spent
    return total


@dataclass(frozen=True)
class Caps:
    max_spend: Micros
    per_experiment: Micros
    overall: Micros
    overall_spent: Micros  # billable spend already recorded by other experiments

    @property
    def overall_remaining(self) -> Micros:
        return max(self.overall - self.overall_spent, 0)

    @property
    def effective(self) -> Micros:
        return min(self.max_spend, self.per_experiment, self.overall_remaining)

    def violations(self) -> list[str]:
        out = []
        if self.max_spend > self.per_experiment:
            out.append(
                f"budget.max_spend {format_usd(self.max_spend, 2)} is above the per-experiment "
                f"cap {format_usd(self.per_experiment, 2)}"
            )
        return out


def caps_for(max_spend: Micros, config: BudgetConfig, overall_spent: Micros) -> Caps:
    return Caps(
        max_spend=max_spend,
        per_experiment=config.per_experiment_cap,
        overall=config.overall_cap,
        overall_spent=overall_spent,
    )


class BudgetExceeded(Exception):
    """Base for budget outcomes; `reason` is stored on the experiment."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BudgetStop(BudgetExceeded):
    """The next step would not fit: stop before starting it."""


class BudgetAbort(BudgetExceeded):
    """Recorded spend reached the cap: cancel in-flight work and tear everything down."""


Clock = Callable[[], datetime]


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class _Accrual:
    host: Host
    recorded: Micros = 0  # spend already written for this host


class BudgetGuard:
    """Accrues host spend for one experiment and enforces its caps.

    Per host, accrued spend is `cost_for_seconds(hourly, now - launched_at)`
    computed from launch each time, so rounding never drifts; each accrual
    writes only the difference since the last one.
    """

    def __init__(
        self,
        *,
        db_url: str | None,
        experiment_id: uuid.UUID,
        cap_micros: Micros,
        overall_cap_micros: Micros,
        billable: bool,
        interval_s: float,
        clock: Clock = utcnow,
    ) -> None:
        self.db_url = db_url
        self.experiment_id = experiment_id
        self.cap_micros = cap_micros
        self.overall_cap_micros = overall_cap_micros
        self.billable = billable
        self.interval_s = interval_s
        self.clock = clock
        self.spent_micros: Micros = 0
        self.reason: str | None = None
        self.tripped = asyncio.Event()
        self._hosts: dict[str, _Accrual] = {}
        self._task: asyncio.Task[None] | None = None

    @property
    def live_hourly_micros(self) -> Micros:
        return sum(a.host.hourly_micros for a in self._hosts.values())

    def add_host(self, host: Host) -> None:
        self._hosts[host.host_id] = _Accrual(host)

    def remove_host(self, host: Host, *, at: datetime | None = None) -> None:
        """Final accrual up to `at` (teardown finished), then stop billing the host."""
        if host.host_id in self._hosts:
            self.accrue(at)
            del self._hosts[host.host_id]

    def accrue(self, now: datetime | None = None) -> Micros:
        """Write spend due since the last accrual; trip when a cap is reached."""
        now = now or self.clock()
        with session_scope(self.db_url) as session:
            for acc in self._hosts.values():
                seconds = max((now - acc.host.launched_at).total_seconds(), 0.0)
                due = cost_for_seconds(acc.host.hourly_micros, seconds) - acc.recorded
                if due <= 0:
                    continue
                self.spent_micros = repo.add_spend(
                    session,
                    self.experiment_id,
                    due,
                    resource_id=acc.host.host_id,
                    recorded_at=now,
                    basis={
                        "provider": acc.host.provider,
                        "hourly_micros": acc.host.hourly_micros,
                        "market": acc.host.request.market.value,
                        "instance_type": acc.host.request.instance_type,
                        "billed_seconds": seconds,
                        "simulated": not self.billable,
                    },
                )
                acc.recorded += due
            others = billable_spend(session, exclude=self.experiment_id) if self.billable else 0
        self._check_caps(others)
        return self.spent_micros

    def _check_caps(self, others_billable: Micros) -> None:
        if self.tripped.is_set():
            return
        if self.spent_micros >= self.cap_micros:
            self._trip(
                f"budget cap reached: spent {format_usd(self.spent_micros)} "
                f">= cap {format_usd(self.cap_micros)}"
            )
        elif self.billable and others_billable + self.spent_micros >= self.overall_cap_micros:
            self._trip(
                f"overall cap reached: {format_usd(others_billable + self.spent_micros)} "
                f">= {format_usd(self.overall_cap_micros)} across all experiments"
            )

    def _trip(self, reason: str) -> None:
        self.reason = reason
        self.tripped.set()

    def raise_if_tripped(self) -> None:
        if self.tripped.is_set():
            raise BudgetAbort(self.reason or "budget guard tripped")

    def check_next(self, seconds: float, *, extra_hourly_micros: Micros = 0, what: str) -> None:
        """Graceful stop when spend so far plus the next step's estimate would pass the cap."""
        self.accrue()
        self.raise_if_tripped()
        hourly = self.live_hourly_micros + extra_hourly_micros
        projected = self.spent_micros + cost_for_seconds(hourly, seconds)
        if projected > self.cap_micros:
            raise BudgetStop(
                f"stopped before {what}: projected {format_usd(projected)} "
                f"(spent {format_usd(self.spent_micros)} + ~{seconds:.0f}s at "
                f"{format_usd(hourly)}/h) > cap {format_usd(self.cap_micros)}"
            )

    async def guarded[T](self, aw: Awaitable[T]) -> T:
        """Await `aw`, cancelling it and raising BudgetAbort if the guard trips meanwhile."""
        self.raise_if_tripped()
        task = asyncio.ensure_future(aw)
        trip = asyncio.ensure_future(self.tripped.wait())
        try:
            await asyncio.wait({task, trip}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            trip.cancel()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            self.raise_if_tripped()
        return task.result()

    async def _loop(self) -> None:
        while not self.tripped.is_set():
            await asyncio.sleep(self.interval_s)
            try:
                self.accrue()
            except Exception as e:  # a blind guard must not let hosts keep running
                self._trip(f"budget accrual failed: {type(e).__name__}: {e}")

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
