import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from loom_bench.budget import BudgetAbort, BudgetGuard, BudgetStop, billable_spend
from loom_bench.provenance import GitInfo
from loom_bench.providers.base import Host, HostRequest
from loom_bench.records import Market
from loom_bench.store import repo
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchExperiment, BenchSpend

T0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _experiment(db, kind="aws_ec2", spent=0) -> object:
    with session_scope(db) as s:
        e = repo.create_experiment(
            s, name=kind, spec={"provider": {"kind": kind}}, git=GitInfo(), budget_micros=None
        )
        s.get(BenchExperiment, e.id).spent_micros = spent
        return e.id


def _host(hourly: int, host_id: str = "h1") -> Host:
    return Host(
        provider="aws_ec2",
        host_id=host_id,
        request=HostRequest(ttl_s=3600, market=Market.SPOT, instance_type="g6e.xlarge"),
        hourly_micros=hourly,
        launched_at=T0,
        ttl_at=T0 + timedelta(hours=1),
        info={"accrual_basis": {"source": "test"}},
    )


def _guard(db, exp_id, clock, cap=1_000_000, overall=150_000_000, billable=True, interval=1.0):
    return BudgetGuard(
        db_url=db,
        experiment_id=exp_id,
        cap_micros=cap,
        overall_cap_micros=overall,
        billable=billable,
        interval_s=interval,
        clock=clock,
    )


def test_accrual_is_exact_and_persisted(db):
    exp_id, clock = _experiment(db), Clock()
    guard = _guard(db, exp_id, clock)
    guard.add_host(_host(3_600_000))  # $3.60/h = 1000 micros/s
    clock.advance(10)
    assert guard.accrue() == 10_000
    clock.advance(0.3333)
    guard.accrue()
    clock.advance(9.6667)
    assert guard.accrue() == 20_000  # from launch each time: no rounding drift
    with session_scope(db) as s:
        rows = list(s.scalars(select(BenchSpend).where(BenchSpend.experiment_id == exp_id)))
        assert sum(r.amount_micros for r in rows) == 20_000
        assert rows[0].basis["accrual_basis"] == {"source": "test"}
        assert s.get(BenchExperiment, exp_id).spent_micros == 20_000


def test_removing_a_host_accrues_through_teardown_then_stops(db):
    exp_id, clock = _experiment(db), Clock()
    guard = _guard(db, exp_id, clock)
    host = _host(3_600_000)
    guard.add_host(host)
    guard.remove_host(host, at=T0 + timedelta(seconds=5))
    clock.advance(100)
    assert guard.accrue() == 5_000


def test_trips_when_spend_reaches_the_cap(db):
    exp_id, clock = _experiment(db), Clock()
    guard = _guard(db, exp_id, clock, cap=15_000)
    guard.add_host(_host(3_600_000))
    clock.advance(10)
    guard.accrue()
    assert not guard.tripped.is_set()
    clock.advance(5)
    guard.accrue()
    assert guard.tripped.is_set()
    with pytest.raises(BudgetAbort, match="budget cap reached"):
        guard.raise_if_tripped()


def test_projected_spend_stops_gracefully(db):
    exp_id, clock = _experiment(db), Clock()
    guard = _guard(db, exp_id, clock, cap=100_000)
    guard.add_host(_host(3_600_000))
    clock.advance(10)
    guard.check_next(80, what="small step")  # 10k + 80k <= 100k
    with pytest.raises(BudgetStop, match="projected"):
        guard.check_next(91, what="big step")
    with pytest.raises(BudgetStop):
        guard.check_next(1, extra_hourly_micros=360_000_000, what="a new host")


def test_overall_cap_counts_other_billable_experiments(db):
    _experiment(db, "aws_ec2", spent=149_995_000)
    _experiment(db, "mock", spent=10**12)
    exp_id, clock = _experiment(db), Clock()
    with session_scope(db) as s:
        assert billable_spend(s, exclude=exp_id) == 149_995_000
    guard = _guard(db, exp_id, clock, cap=10**9)
    guard.add_host(_host(3_600_000))
    clock.advance(5)
    guard.accrue()
    assert guard.tripped.is_set()
    assert "overall cap" in guard.reason


def test_simulated_spend_never_trips_the_overall_cap(db):
    _experiment(db, "aws_ec2", spent=149_999_999)
    exp_id, clock = _experiment(db, "mock"), Clock()
    guard = _guard(db, exp_id, clock, cap=10**9, billable=False)
    guard.add_host(_host(3_600_000))
    clock.advance(5)
    guard.accrue()
    assert not guard.tripped.is_set()


async def test_trip_cancels_the_guarded_call(db):
    exp_id = _experiment(db)
    guard = _guard(db, exp_id, lambda: datetime.now(UTC), cap=1_000, interval=0.05)
    host = _host(3_600_000_000).model_copy(update={"launched_at": datetime.now(UTC)})
    guard.add_host(host)  # $1/ms: trips on the first accrual
    cancelled = asyncio.Event()

    async def long_job() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    guard.start()
    try:
        with pytest.raises(BudgetAbort):
            await asyncio.wait_for(guard.guarded(long_job()), timeout=5)
    finally:
        await guard.stop()
    assert cancelled.is_set()
