"""Postgres outages, lock waits and crashes around COMMIT (docs/phase1-m3-spec.md §8.3,
§8.7 T8, T9, T13; done-when A19, A20).

Faults are injected at the connection level only (§8.11): a TCP proxy in front of
Postgres (`FaultProxy`), an unopened pool (the role gate), and locks held by the test on
its own connections. Tests that run `retry_unapplied` use a scratch database of their
own (§16)."""

from __future__ import annotations

import asyncio
import re
import time

import pytest
from billing_m3s_kit import (
    HANDLER_FAILURES,
    RETRY_LAST_SUCCESS,
    WEBHOOK_ERRORS,
    billing_client,
    charge_event,
    checkout_event,
    deliver,
    ep,
    error_code,
    event_row,
    fault_proxy,
    gauge,
    holder,
    ledger_count,
    m3s_scratch_fixtures,
    m3s_settings,
    make_org,
    purchases,
    refunds,
    rid,
    sample,
    wait_task_or_blocked,
)
from psycopg_pool import AsyncConnectionPool

from loom_billing.api import create_app

pytestmark = [pytest.mark.postgres, pytest.mark.redis]
m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()

M = 10_000
# `<exception class>: <message>` (§8.3), NUL-free, at most 1,000 characters.
LAST_ERROR = re.compile(r"^[A-Za-z_][\w.]*: ", re.DOTALL)


def assert_stored_error(row: dict | None, *, count: int = 1) -> None:
    assert row is not None, "the second transaction stored nothing"
    assert row["processed_at"] is None and row["result"] == "unapplied"
    assert row["last_error"] and LAST_ERROR.match(row["last_error"]), row["last_error"]
    assert len(row["last_error"]) <= 1000 and "\x00" not in row["last_error"]
    assert row["last_error_at"] is not None
    assert row["error_count"] == count


async def unreachable_pool(url: str) -> AsyncConnectionPool:
    # A pool wait the code does not bound itself ends after the pool's own `timeout`.
    pool = AsyncConnectionPool(url, min_size=1, max_size=2, open=False, timeout=5)
    await pool.open(wait=False)
    return pool


# ------------------------------------------------------------------ A19: T8, T13


async def test_a19_t8_postgres_unreachable_503_nothing_stored_then_retry_applies(
    pool, pg_url, redis_client
):
    org = await make_org(pool)
    ev = checkout_event(org, 1300)
    s = m3s_settings()
    with fault_proxy(pg_url, opened=False) as proxy:  # refuses every connection
        down = await unreachable_pool(proxy.url(pg_url))
        try:
            async with billing_client(s, down, redis_client) as client:
                before = sample(WEBHOOK_ERRORS, status="503")
                t0 = time.monotonic()
                r = await deliver(client, ev)
                elapsed = time.monotonic() - t0
        finally:
            await down.close()
    assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
    assert sample(WEBHOOK_ERRORS, status="503") == before + 1
    # The pool wait (1 s) twice, and no more: no overall deadline beyond the timeouts.
    assert elapsed < 10, elapsed
    assert await event_row(pool, ev["id"]) is None
    async with billing_client(s, pool, redis_client) as client:  # Stripe's retry
        r = await deliver(client, ev)
        assert r.status_code == 200 and r.json()["result"] == "credited"
        r = await deliver(client, ev)
        assert r.status_code == 200 and r.json()["result"] == "duplicate"
    assert await purchases(pool, org) == [1300 * M]


async def test_a19_t13_role_gate_pending_unopened_pool_answers_503(pool, pg_url, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org, 600)
    unopened = AsyncConnectionPool(pg_url, open=False)  # as open_pool(background=True)
    async with billing_client(m3s_settings(), unopened, redis_client) as client:
        before = sample(WEBHOOK_ERRORS, status="503")
        r = await deliver(client, ev)
    assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
    assert sample(WEBHOOK_ERRORS, status="503") == before + 1
    assert await event_row(pool, ev["id"]) is None


async def test_a19_process_event_raises_store_failed_with_the_cause(pool, pg_url):
    process_event = ep("process_event")
    store_failed = ep("StoreFailed")
    bad_event = ep("BadEvent")
    org = await make_org(pool)
    unopened = AsyncConnectionPool(pg_url, open=False)
    with pytest.raises(store_failed) as caught:
        await process_event(unopened, checkout_event(org, 600), settings=m3s_settings())
    assert caught.value.__cause__ is not None
    # The shape check comes first, before any transaction (§8.11).
    live = checkout_event(org, 600)
    live["livemode"] = True
    for bad in (live, {"type": "checkout.session.completed"}, {"id": rid("evt_test")}, []):
        with pytest.raises(bad_event):
            await process_event(unopened, bad, settings=m3s_settings())


async def test_a19_retry_job_with_postgres_down_ends_its_run_without_success(pg_url):
    retry_unapplied = ep("retry_unapplied")
    with fault_proxy(pg_url, opened=False) as proxy:
        down = await unreachable_pool(proxy.url(pg_url))
        try:
            before = gauge(RETRY_LAST_SUCCESS)
            t0 = time.monotonic()
            n = await retry_unapplied(down, settings=m3s_settings())  # never raises
            elapsed = time.monotonic() - t0
        finally:
            await down.close()
    assert n == 0
    assert gauge(RETRY_LAST_SUCCESS) == before  # S5: no success recorded
    # The run ends at its first statement (the sweep): one pool wait, however long the
    # pool's own timeout is (the spec does not bound R's wait by the webhook's setting).
    assert elapsed <= 35, elapsed


# ------------------------------------------------------------------ A19: lock waits


async def test_a19_lock_timeout_on_a_stripe_payments_row_500_then_r_applies(
    scratch_pool, scratch_url, redis_client
):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    out = await process_event(
        scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s
    )
    assert out.result == "credited"
    refund = charge_event(pi, 400, 1000)
    async with holder(scratch_url) as h, billing_client(s, scratch_pool, redis_client) as c:
        await h.run("SELECT 1 FROM stripe_payments WHERE payment_intent = %s FOR UPDATE", pi)
        errs = sample(WEBHOOK_ERRORS, status="500")
        s4 = sample(HANDLER_FAILURES, path="webhook")
        r = await deliver(c, refund)
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert sample(WEBHOOK_ERRORS, status="500") == errs + 1
        assert sample(HANDLER_FAILURES, path="webhook") == s4 + 1
        assert_stored_error(await event_row(scratch_pool, refund["id"]))
        await h.rollback()
    assert await retry_unapplied(scratch_pool, settings=s) == 1
    assert await refunds(scratch_pool, org) == [-400 * M]
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["result"] == "refunded"
    assert row["last_error"] is None and row["error_count"] == 0


def _advisory_args(classid: int, objid: int, objsubid: int) -> tuple[str, tuple[int, ...]]:
    def s32(v: int) -> int:
        return v - (1 << 32) if v >= 1 << 31 else v

    if objsubid == 1:  # one bigint key
        key = (classid << 32) | objid
        return "SELECT pg_advisory_xact_lock(%s::bigint)", (
            key - (1 << 64) if key >= 1 << 63 else key,
        )
    return "SELECT pg_advisory_xact_lock(%s::int, %s::int)", (s32(classid), s32(objid))


async def test_a19_lock_timeout_on_the_sessions_advisory_lock_500_then_r(
    scratch_pool, scratch_url, redis_client
):
    """The session's advisory lock is found, not assumed: a purchase of the session held
    on a pending clawback's row shows it in pg_locks. Then the test holds that lock while
    a second event of the session is delivered."""
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings(stripe_lock_timeout_ms=8000)
    org = await make_org(scratch_pool)
    pi, sid = rid("pi_test"), rid("cs_test")
    early = charge_event(pi, 100, 1000)
    assert (await process_event(scratch_pool, early, settings=s)).result == "unapplied"
    async with holder(scratch_url) as h:
        await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", early["id"])
        first = asyncio.create_task(
            process_event(
                scratch_pool,
                checkout_event(org, 1000, session_id=sid, payment_intent=pi),
                settings=s,
            )
        )
        pid = await wait_task_or_blocked(scratch_url, h.pid, first)
        locks = await h.run(
            "SELECT classid::bigint, objid::bigint, objsubid FROM pg_locks "
            "WHERE locktype = 'advisory' AND granted AND pid = %s",
            pid,
        )
        await h.rollback()
        assert (await first).result == "credited"
    assert locks, "the purchase held no advisory lock while it waited"

    second = checkout_event(
        org, 1000, session_id=sid, payment_intent=pi, typ="checkout.session.async_payment_succeeded"
    )
    short = m3s_settings()
    async with holder(scratch_url) as h, billing_client(short, scratch_pool, redis_client) as c:
        for classid, objid, objsubid in locks:
            q, args = _advisory_args(int(classid), int(objid), int(objsubid))
            await h.run(q, *args)
        r = await deliver(c, second)
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert_stored_error(await event_row(scratch_pool, second["id"]))
        await h.rollback()
    await retry_unapplied(scratch_pool, settings=short)
    row = await event_row(scratch_pool, second["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "already_credited"
    assert await purchases(scratch_pool, org) == [1000 * M]


async def test_a19_several_lock_waits_in_one_delivery_500_then_r_applies(
    scratch_pool, scratch_url, redis_client
):
    """A purchase waits on two pending clawbacks' rows in turn: the first is released
    while it waits, the second is not (each wait has its own `lock_timeout`)."""
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    # Long enough that the first wait outlives the barrier and the release under load.
    s = m3s_settings(stripe_lock_timeout_ms=2000)
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    e2 = charge_event(pi, 200, 1000)
    e3 = charge_event(pi, 500, 1000)
    for e in (e2, e3):
        assert (await process_event(scratch_pool, e, settings=s)).result == "unapplied"
    purchase = checkout_event(org, 1000, payment_intent=pi)
    async with (
        holder(scratch_url) as ha,
        holder(scratch_url) as hb,
        billing_client(s, scratch_pool, redis_client) as c,
    ):
        await ha.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", e2["id"])
        await hb.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", e3["id"])
        task = asyncio.create_task(deliver(c, purchase))
        pid = await wait_task_or_blocked(scratch_url, ha.pid, task)
        await ha.rollback()
        await wait_task_or_blocked(scratch_url, hb.pid, task, waiter_pid=pid)
        r = await task
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert_stored_error(await event_row(scratch_pool, purchase["id"]))
        assert await purchases(scratch_pool, org) == []
        await hb.rollback()
    await retry_unapplied(scratch_pool, settings=s)
    assert await purchases(scratch_pool, org) == [1000 * M]
    assert sum(await refunds(scratch_pool, org)) == -500 * M  # cumulative: 500 in all
    for e in (purchase, e2, e3):
        row = await event_row(scratch_pool, e["id"])
        assert row is not None and row["processed_at"] is not None


async def test_a19_lock_timeout_on_the_events_own_row_503(pool, pg_url, redis_client):
    """The second transaction needs the same row, so it times out too: 503, and the
    stored row is left as it was."""
    org = await make_org(pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 300, 1000)
    s = m3s_settings(stripe_lock_timeout_ms=1000)
    async with billing_client(s, pool, redis_client) as c:
        assert (await deliver(c, refund)).json()["result"] == "unapplied"
        before = await event_row(pool, refund["id"])
        async with holder(pg_url) as h:
            await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", refund["id"])
            errs = sample(WEBHOOK_ERRORS, status="503")
            r = await deliver(c, refund)
            assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
            assert sample(WEBHOOK_ERRORS, status="503") == errs + 1
            await h.rollback()
        assert await event_row(pool, refund["id"]) == before
        assert (await deliver(c, checkout_event(org, 1000, payment_intent=pi))).status_code == 200
        r = await deliver(c, refund)
        assert r.status_code == 200 and r.json()["result"] == "duplicate"
    assert await refunds(pool, org) == [-300 * M]


# ------------------------------------------------------------------ A19: keepalives

KEEPALIVES = {
    "keepalives": "1",
    "keepalives_idle": "10",
    "keepalives_interval": "5",
    "keepalives_count": "2",
    "tcp_user_timeout": "10000",
}


@pytest.mark.parametrize("form", ["url", "keyword"])
async def test_a19_billing_conninfo_carries_keepalives(pg_url, form):
    import psycopg

    info = psycopg.conninfo.conninfo_to_dict(pg_url)
    if form == "url":
        url = (
            f"postgresql://{info['user']}:{info.get('password', '')}@{info.get('host')}:"
            f"{info.get('port', 5432)}/{info['dbname']}"
        )
    else:
        url = psycopg.conninfo.make_conninfo("", **info)
    app = create_app(m3s_settings(database_url=url))
    async with app.router.lifespan_context(app), app.state.pool.connection() as conn:
        params = conn.info.get_parameters()
    for k, v in KEEPALIVES.items():
        assert params.get(k) == v, (k, params.get(k))


# ------------------------------------------------------------------ A20: T9


async def test_a20_t9a_failure_before_commit_then_retry_applies_once(pool, pg_url, redis_client):
    """The process dies before COMMIT (the COMMIT never reaches Postgres) and nothing can
    connect: 503, nothing stored; Stripe's retry applies it."""
    org = await make_org(pool)
    ev = checkout_event(org, 1400)
    s = m3s_settings()
    with fault_proxy(pg_url) as proxy:
        p = AsyncConnectionPool(proxy.url(pg_url), min_size=1, max_size=4, open=False)
        await p.open(wait=True)
        try:
            async with billing_client(s, p, redis_client) as c:
                proxy.arm_commit(landed=False, then_refuse=True)
                r = await deliver(c, ev)
        finally:
            await p.close()
    assert proxy.fired.is_set()
    assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
    assert await event_row(pool, ev["id"]) is None
    assert await ledger_count(pool, ev["id"]) == 0
    async with billing_client(s, pool, redis_client) as c:
        r = await deliver(c, ev)
        assert r.status_code == 200 and r.json()["result"] == "credited"
    assert await ledger_count(pool, ev["id"]) == 1


async def test_a20_t9b_commit_then_answer_lost_retry_is_duplicate(pool, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org, 1500)
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        await deliver(c, ev)  # committed; Stripe never saw the answer
        r = await deliver(c, ev)
        assert r.status_code == 200 and r.json()["result"] == "duplicate"
    assert await ledger_count(pool, ev["id"]) == 1
    row = await event_row(pool, ev["id"])
    assert row is not None and row["result"] == "credited"


async def test_a20_t9c_commit_landed_result_lost_guard_answers_duplicate(
    pool, pg_url, redis_client
):
    org = await make_org(pool)
    ev = checkout_event(org, 1600)
    s = m3s_settings()
    with fault_proxy(pg_url) as proxy:
        p = AsyncConnectionPool(proxy.url(pg_url), min_size=2, max_size=4, open=False)
        await p.open(wait=True)
        try:
            async with billing_client(s, p, redis_client) as c:
                errs = {st: sample(WEBHOOK_ERRORS, status=st) for st in ("500", "503")}
                s4 = sample(HANDLER_FAILURES, path="webhook")
                proxy.arm_commit(landed=True)
                r = await deliver(c, ev)
        finally:
            await p.close()
    assert proxy.fired.is_set()
    assert r.status_code == 200 and r.json()["result"] == "duplicate", r.text
    # The guard's duplicate (§8.3): no S2c, no S4.
    assert {st: sample(WEBHOOK_ERRORS, status=st) for st in ("500", "503")} == errs
    assert sample(HANDLER_FAILURES, path="webhook") == s4
    assert await ledger_count(pool, ev["id"]) == 1
    row = await event_row(pool, ev["id"])
    assert row is not None and row["result"] == "credited" and row["last_error"] is None
    assert row["error_count"] == 0  # the guard touched nothing


async def test_a20_t9c_commit_not_landed_is_stored_then_retry_applies(pool, pg_url, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org, 1700)
    s = m3s_settings()
    with fault_proxy(pg_url) as proxy:
        p = AsyncConnectionPool(proxy.url(pg_url), min_size=2, max_size=4, open=False)
        await p.open(wait=True)
        try:
            async with billing_client(s, p, redis_client) as c:
                proxy.arm_commit(landed=False)
                r = await deliver(c, ev)
        finally:
            await p.close()
    assert proxy.fired.is_set()
    assert r.status_code == 500 and error_code(r) == "internal_error", r.text
    assert_stored_error(await event_row(pool, ev["id"]))
    assert await ledger_count(pool, ev["id"]) == 0
    async with billing_client(s, pool, redis_client) as c:
        r = await deliver(c, ev)
        assert r.status_code == 200 and r.json()["result"] == "credited"
    assert await ledger_count(pool, ev["id"]) == 1
    row = await event_row(pool, ev["id"])
    assert row is not None and row["last_error"] is None and row["error_count"] == 0


async def test_a20_t9c_commit_landed_and_no_connection_503_then_duplicate(
    pool, pg_url, redis_client
):
    org = await make_org(pool)
    ev = checkout_event(org, 1800)
    s = m3s_settings()
    with fault_proxy(pg_url) as proxy:
        p = AsyncConnectionPool(proxy.url(pg_url), min_size=1, max_size=4, open=False)
        await p.open(wait=True)
        try:
            async with billing_client(s, p, redis_client) as c:
                proxy.arm_commit(landed=True, then_refuse=True)
                r = await deliver(c, ev)
        finally:
            await p.close()
    assert proxy.fired.is_set()
    assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
    async with billing_client(s, pool, redis_client) as c:
        r = await deliver(c, ev)
        assert r.status_code == 200 and r.json()["result"] == "duplicate"
    assert await ledger_count(pool, ev["id"]) == 1
    assert await purchases(pool, org) == [1800 * M]
