"""`retry_unapplied` (R): skip-locked event rows, newest first up to `limit`, the end of a
run on a failed error store, and the T4 race (docs/phase1-m3-spec.md §8.3, §8.11;
done-when A33, A17's concurrent variant; traces T4, T10).

Each test has a scratch database of its own (§16): R re-runs every unapplied row."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import uuid

import pytest
from billing_m3s_kit import (
    HANDLER_FAILURES,
    RETRY_LAST_SUCCESS,
    STALE_TOTAL,
    charge_event,
    checkout_event,
    ep,
    event_row,
    gate,
    gauge,
    holder,
    install_faults,
    m3s_scratch_fixtures,
    m3s_settings,
    make_org,
    now,
    poison,
    refunds,
    rid,
    sample,
    store_raw,
    wait_task_or_blocked,
)

pytestmark = [pytest.mark.postgres]
m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()

M = 10_000
MINUTE = dt.timedelta(minutes=1)
MARKER = "m3s retry marker"


async def credited_payment(pool, s, t, cents: int = 1000):
    process_event = ep("process_event")
    org = await make_org(pool)
    pi = rid("pi_test")
    out = await process_event(
        pool, checkout_event(org, cents, payment_intent=pi), settings=s, now=t
    )
    assert out.result == "credited"
    return org, pi


# ------------------------------------------------------------------ T10


async def test_a33_t10_retry_skips_a_row_a_delivery_holds(scratch_pool, scratch_url):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings(stripe_lock_timeout_ms=5000)
    t0 = now()
    org, pi = await credited_payment(scratch_pool, s, t0)
    raced = charge_event(pi, 300, 1000)
    await store_raw(scratch_pool, raced, received_at=t0)
    async with holder(scratch_url) as h:  # a delivery of the same event, in flight
        await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", raced["id"])
        skipped = await process_event(scratch_pool, raced, retry=True, settings=s)
        assert skipped.result == "skipped" and not skipped.processed
        before = gauge(RETRY_LAST_SUCCESS)
        t = time.monotonic()
        assert await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE) == 0
        assert time.monotonic() - t < 4.5  # never waited on the held row
        assert gauge(RETRY_LAST_SUCCESS) != before  # the run finished
        row = await event_row(scratch_pool, raced["id"])
        assert row is not None and row["processed_at"] is None
        assert row["error_count"] == 0 and row["last_error"] is None
        await h.rollback()
    assert await retry_unapplied(scratch_pool, settings=s, now=t0 + 2 * MINUTE) == 1
    assert await refunds(scratch_pool, org) == [-300 * M]


async def test_a33_retry_may_wait_on_a_payment_lock_then_records_the_failure(
    scratch_pool, scratch_url
):
    """Inside the handler R can wait up to `lock_timeout` on a `stripe_payments` row; the
    attempt then fails, is stored with its error (S4 `retry` on its first failure), and
    the run goes on to finish."""
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings(stripe_lock_timeout_ms=500)
    t0 = now()
    org, pi = await credited_payment(scratch_pool, s, t0)
    raced = charge_event(pi, 300, 1000)
    await store_raw(scratch_pool, raced, received_at=t0)
    async with holder(scratch_url) as h:
        await h.run("SELECT 1 FROM stripe_payments WHERE payment_intent = %s FOR UPDATE", pi)
        retry_s4, before = sample(HANDLER_FAILURES, path="retry"), gauge(RETRY_LAST_SUCCESS)
        assert await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE) == 0
        assert sample(HANDLER_FAILURES, path="retry") == retry_s4 + 1
        assert gauge(RETRY_LAST_SUCCESS) != before
        row = await event_row(scratch_pool, raced["id"])
        assert row is not None and row["error_count"] == 1 and row["last_error"]
        await h.rollback()
    assert await retry_unapplied(scratch_pool, settings=s, now=t0 + 2 * MINUTE) == 1
    assert await refunds(scratch_pool, org) == [-300 * M]


# ------------------------------------------------------------------ order and limit


async def test_a33_reruns_newest_first_up_to_the_limit(scratch_pool):
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    t0 = now()
    ids = []
    for i in range(5):  # oldest first
        ev = checkout_event(uuid.uuid4(), 1000)  # unknown org: stays unapplied
        await store_raw(
            scratch_pool,
            ev,
            received_at=t0 - dt.timedelta(minutes=10 - i),
            result="unapplied",
            last_error=MARKER,
        )
        ids.append(ev["id"])
    assert await retry_unapplied(scratch_pool, limit=2, settings=s, now=t0) == 0
    rerun = []
    for evt in ids:
        row = await event_row(scratch_pool, evt)
        assert row is not None and row["processed_at"] is None
        rerun.append(row["last_error"] is None)  # a clean re-run clears last_error
    assert rerun == [False, False, False, True, True]


# ------------------------------------------------------------------ a stored bad event


async def test_a33_stored_row_failing_the_shape_check_is_skipped_without_an_error_store(
    scratch_pool,
):
    """§8.11: a stored row that fails the shape check (`BadEvent`; only a test can store
    one, here with `livemode: true`) is skipped by R with no error store; the run goes on
    and finishes, and the row goes stale for an admin to resolve."""
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    t0 = now()
    org, pi = await credited_payment(scratch_pool, s, t0)
    bad = checkout_event(uuid.uuid4(), 1000)
    bad["livemode"] = True
    await store_raw(scratch_pool, bad, received_at=t0 + dt.timedelta(seconds=2))  # newest
    raced = charge_event(pi, 300, 1000)
    await store_raw(scratch_pool, raced, received_at=t0)  # re-run after the bad one
    retry_s4 = sample(HANDLER_FAILURES, path="retry")
    before = gauge(RETRY_LAST_SUCCESS)

    assert await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE) == 1
    assert gauge(RETRY_LAST_SUCCESS) != before  # the run finished
    assert await refunds(scratch_pool, org) == [-300 * M]  # the run went past it
    row = await event_row(scratch_pool, bad["id"])
    assert row is not None and row["processed_at"] is None
    assert row["last_error"] is None and row["error_count"] == 0  # no error store
    assert row["last_error_at"] is None
    assert sample(HANDLER_FAILURES, path="retry") == retry_s4

    stale = sample(STALE_TOTAL, type="checkout.session.completed")
    t_stale = t0 + 16 * MINUTE
    await retry_unapplied(scratch_pool, settings=s, now=t_stale)
    row = await event_row(scratch_pool, bad["id"])
    assert row is not None and row["processed_at"] is None and row["error_count"] == 0
    assert row["stale_reported_at"] == t_stale
    assert sample(STALE_TOTAL, type="checkout.session.completed") == stale + 1


# ------------------------------------------------------------------ StoreFailed ends the run


async def test_a33_run_ends_when_an_attempts_error_store_fails(scratch_pool, scratch_url):
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    stale_candidate = checkout_event(uuid.uuid4(), 1000)
    await store_raw(
        scratch_pool, stale_candidate, received_at=t0 - dt.timedelta(hours=1),
        result="unapplied", last_error=MARKER,
    )  # fmt: skip
    older = checkout_event(uuid.uuid4(), 1000)
    await store_raw(
        scratch_pool, older, received_at=t0 - MINUTE, result="unapplied", last_error=MARKER
    )
    newest = charge_event(rid("pi_test"), 100, 1000)
    await store_raw(scratch_pool, newest, received_at=t0)
    await poison(scratch_url, newest["id"], hard=True)  # its error store fails too

    before = gauge(RETRY_LAST_SUCCESS)
    stale_before = sample(STALE_TOTAL, type="checkout.session.completed")
    n = await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE)  # never raises
    assert n == 0
    assert gauge(RETRY_LAST_SUCCESS) == before  # S5: not a completed run
    for ev in (older, stale_candidate):  # not re-run after the failed store
        row = await event_row(scratch_pool, ev["id"])
        assert row is not None and row["last_error"] == MARKER
    row = await event_row(scratch_pool, stale_candidate["id"])
    assert row is not None and row["stale_reported_at"] is None  # no stale marking
    assert sample(STALE_TOTAL, type="checkout.session.completed") == stale_before


# ------------------------------------------------------------------ A17: concurrent T4


async def test_a17_t4_concurrent_race_r_applies_the_clawback_at_its_next_run(
    scratch_pool, scratch_url
):
    """The refund reads `stripe_payments` before the purchase commits (no row: unapplied)
    and commits after the purchase's pending query ran. Both commit; R applies the
    refund at its next run, before it could count as stale."""
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings(stripe_lock_timeout_ms=10_000)
    t0 = now()
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    purchase = checkout_event(org, 1000, payment_intent=pi)
    refund = charge_event(pi, 700, 1000)
    await gate(scratch_url, "stripe_events", purchase["id"], key=92_001)
    async with holder(scratch_url) as h:
        await h.run("SELECT pg_advisory_xact_lock(92001)")
        buy = asyncio.create_task(process_event(scratch_pool, purchase, settings=s, now=t0))
        await wait_task_or_blocked(scratch_url, h.pid, buy)  # held at its COMMIT
        out = await process_event(scratch_pool, refund, settings=s, now=t0)
        assert out.result == "unapplied"  # the payment is not visible yet
        await h.rollback()
        assert (await buy).result == "credited"
    assert await refunds(scratch_pool, org) == []
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["processed_at"] is None

    stale = sample(STALE_TOTAL, type="charge.refunded")
    assert await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE) == 1
    assert await refunds(scratch_pool, org) == [-700 * M]
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["result"] == "refunded" and row["stale_reported_at"] is None
    await retry_unapplied(scratch_pool, settings=s, now=t0 + 20 * MINUTE)
    assert sample(STALE_TOTAL, type="charge.refunded") == stale  # never counted
