"""The spend guard and its alerts (docs/phase1-m2-spec.md §13 item 14; §5 "Create": "Spend
guard, checked before the row is written: month_to_date + listed_price x lease_ttl_s <=
monthly_cap"; a refusal is an error log line and `creates_total{result="spend_cap"}`, not
a `serving_spend_alerts` row; fraction alerts once per (month, fraction); the guard never
terminates a running healthy pod; §7 `spend_cap`; §11 `reset`).

Spend history is written straight into `serving_pods` (rows of another model). Tests that
never need a pod's TLS to work pin the fake clock to fixed instants, so the month
arithmetic is exact. (The manager has no metrics surface yet, so the refusal's counter is
not checked here.)
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from decimal import Decimal

import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    delete_pod_rows,
    drive,
    env_fixture,  # noqa: F401
    insert_pod_row,
    serving_config,
    spend_alerts,
    tick,
    unique_model_id,
)

from loom_platform.db import connect
from loom_platform.serving import ServingConfig
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr import spend

UTC = dt.UTC
MID_NOVEMBER = dt.datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
HOUR = dt.timedelta(hours=1)
HISTORY = "history-model"


async def history(
    env: FakeServing,
    *,
    start: dt.datetime,
    end: dt.datetime | None,
    usd_per_hr: Decimal | float,
) -> None:
    """A pod of another model: `usd_per_hr` from `start` to `end` (None: still running)."""
    await insert_pod_row(
        env.database_url,
        model_id=HISTORY,
        pod_id=f"hist{uuid.uuid4().hex[:12]}",
        state="terminated" if end else "ready",
        cost_per_hr_micros=int(Decimal(str(usd_per_hr)) * 1_000_000),
        created_at=start,
        terminated_at=end,
        terminated_by="manager" if end else None,
    )


def guard_config(model: str, *, cap_usd: float, lease_ttl_s: int = 3600) -> ServingConfig:
    return serving_config(
        [model],
        spend={"monthly_cap_usd": cap_usd},
        lifecycle={"lease_ttl_s": lease_ttl_s, "lease_renew_every_s": 60},
    )


async def test_create_over_cap_is_refused_before_runpod(
    env: FakeServing, caplog: pytest.LogCaptureFixture
) -> None:
    """Item 14: a create that would cross the cap is refused before RunPod is called and
    before a row is written; the pool is `unavailable` / `spend_cap`; an error is logged;
    wakes do not retry."""
    caplog.set_level(logging.INFO)
    env.clock.now = MID_NOVEMBER.timestamp()
    model = unique_model_id()
    # Cap $1. Month to date $0.90; one more pod for a whole lease (1 h at $1/h): $1.90.
    await history(env, start=MID_NOVEMBER - 2 * HOUR, end=MID_NOVEMBER - HOUR, usd_per_hr=0.9)
    manager = env.manager(guard_config(model, cap_usd=1.0), listed_usd_per_hr=1.0)
    await manager.start()
    for _ in range(4):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    assert env.runpod.creates() == []
    assert env.runpod.stock_queries == 0
    assert await env.rows(model) == []
    pool = await env.pool(model)
    assert (pool.state, pool.reason, pool.retry_after_s) == ("unavailable", "spend_cap", 300)
    assert any(r.levelno >= logging.ERROR for r in caplog.records), "no error logged"


async def test_reset_clears_spend_cap_after_spend_drops(env: FakeServing) -> None:
    """§11 `reset` clears `spend_cap`: once spend leaves room again, wakes alone still do
    not create (the hold stays); after `reset` the next wake does."""
    env.clock.now = MID_NOVEMBER.timestamp()
    model = unique_model_id()
    await history(env, start=MID_NOVEMBER - 2 * HOUR, end=MID_NOVEMBER - HOUR, usd_per_hr=0.9)
    manager = env.manager(guard_config(model, cap_usd=1.0), listed_usd_per_hr=1.0)
    await manager.start()
    await env.wake(model)
    await tick(manager, env.clock, advance=5)
    assert (await env.pool(model)).reason == "spend_cap"
    await delete_pod_rows(env.database_url, HISTORY)  # room under the cap again
    for _ in range(3):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    assert env.runpod.creates() == [], "the spend_cap hold ended without a reset"
    assert (await env.pool(model)).reason == "spend_cap"
    await env.admin({"op": "reset", "model": model})
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: len(env.runpod.creates()) == 1, max_advance_s=1)


async def test_create_exactly_at_cap_is_allowed(env: FakeServing) -> None:
    """Item 14, `<=`: month to date $0.50 plus a whole lease at $0.50/h is exactly the $1
    cap, so the create goes ahead."""
    env.clock.now = MID_NOVEMBER.timestamp()
    model = unique_model_id()
    await history(env, start=MID_NOVEMBER - 2 * HOUR, end=MID_NOVEMBER - HOUR, usd_per_hr=0.5)
    manager = env.manager(guard_config(model, cap_usd=1.0), listed_usd_per_hr=0.5)
    await manager.start()
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: len(env.runpod.creates()) == 1, max_advance_s=1)


async def test_alerts_fire_once_per_month_and_fraction_across_restart(env: FakeServing) -> None:
    """Item 14: the 50 % and 80 % alerts (month to date $0.90 of $1) are recorded once for
    the month and repeat neither on later ticks nor after a restart; 100 % is not reached,
    and the refused create adds no row (§5)."""
    env.clock.now = MID_NOVEMBER.timestamp()
    model = unique_model_id()
    await history(env, start=MID_NOVEMBER - 2 * HOUR, end=MID_NOVEMBER - HOUR, usd_per_hr=0.9)
    config = guard_config(model, cap_usd=1.0)
    first = env.manager(config)
    await first.start()
    await env.wake(model)
    await tick(first, env.clock, advance=5)
    assert (await env.pool(model)).reason == "spend_cap"
    alerts = await spend_alerts(env.database_url)
    assert [(a["month"], float(a["fraction"])) for a in alerts] == [
        (dt.date(2026, 11, 1), 0.5),
        (dt.date(2026, 11, 1), 0.8),
    ]
    for _ in range(3):
        await tick(first, env.clock, advance=5)
    await first.stop()

    second = env.manager(config)
    await second.start()
    for _ in range(3):
        await tick(second, env.clock, advance=5)
    assert await spend_alerts(env.database_url) == alerts  # same rows, same sent_at


async def test_guard_never_terminates_a_ready_pod(env: FakeServing) -> None:
    """Item 14: spend crossing the cap while a pod serves leaves it ready and routed (a
    `ready` pod comes first in §7's order); the 100 % alert fires, once."""
    model = unique_model_id()
    manager = env.manager(guard_config(model, cap_usd=1.0, lease_ttl_s=600))
    await manager.start()
    pod = await bring_up(env, manager, model)
    now = dt.datetime.fromtimestamp(env.clock(), UTC)
    # $1.97 in the last minute: month to date is about twice the cap.
    await history(
        env,
        start=now - dt.timedelta(seconds=60),
        end=now - dt.timedelta(seconds=1),
        usd_per_hr=120,
    )
    for _ in range(12):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        pool = await env.pool(model)
        assert pool.state == "ready" and [r.id for r in pool.replicas] == [pod.pod_id]
    assert env.runpod.has(pod.pod_id)
    assert env.runpod.terminate_requests(pod.pod_id) == 0
    (row,) = await env.rows(model)
    assert row["state"] == "ready"
    fractions = [float(a["fraction"]) for a in await spend_alerts(env.database_url)]
    assert fractions == [0.5, 0.8, 1.0]


async def test_new_month_starts_from_zero(env: FakeServing) -> None:
    """Item 14: last month's spend does not count: $51 in November does not stop a
    December create under a $1 cap."""
    december = dt.datetime(2026, 12, 1, 0, 5, tzinfo=UTC)
    env.clock.now = december.timestamp()
    model = unique_model_id()
    await history(
        env,
        start=december - dt.timedelta(hours=10),
        end=december - dt.timedelta(minutes=10),
        usd_per_hr=Decimal("5.2"),
    )
    manager = env.manager(guard_config(model, cap_usd=1.0, lease_ttl_s=600))
    await manager.start()
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: len(env.runpod.creates()) == 1, max_advance_s=1)


async def test_month_to_date_across_a_month_boundary(env: FakeServing) -> None:
    """Item 14: month to date sums cost x billed hours over this UTC month's rows, running
    pods counted up to now, a row that spans the boundary split at midnight."""
    nov30 = dt.datetime(2026, 11, 30, tzinfo=UTC)
    dec1 = dt.datetime(2026, 12, 1, tzinfo=UTC)
    # A: November only, 1 h at $1/h.
    await history(env, start=nov30 + 22 * HOUR, end=nov30 + 23 * HOUR, usd_per_hr=1)
    # B: started 23:30 on Nov 30 at $2/h, still running.
    await history(env, start=nov30 + dt.timedelta(hours=23, minutes=30), end=None, usd_per_hr=2)
    # C: December only, 10 min at $6/h.
    await history(
        env,
        start=dec1 + dt.timedelta(minutes=10),
        end=dec1 + dt.timedelta(minutes=20),
        usd_per_hr=6,
    )
    conn = await connect(env.database_url)
    try:
        # Nov 30 23:45: A $1 + B's 15 min at $2/h $0.50; C has not started.
        nov_now = nov30 + dt.timedelta(hours=23, minutes=45)
        assert await spend.month_to_date_micros(conn, nov_now) == 1_500_000
        # Dec 1 00:30: B's December half hour $1 + C $1; A and B's November part excluded.
        dec_now = dec1 + dt.timedelta(minutes=30)
        assert await spend.month_to_date_micros(conn, dec_now) == 2_000_000
    finally:
        await conn.close()
