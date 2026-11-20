"""Staleness, the 500-event limit and newest-first re-runs, with a fake clock
(docs/phase1-m3-spec.md §8.3 R, §8.6; done-when A24; traces T11, T21, T22).

Every test runs `retry_unapplied` with a moved `now` or builds a backlog, so each has a
scratch database of its own (§16); counters are compared before and after."""

from __future__ import annotations

import datetime as dt
import inspect
import uuid

import pytest
from billing_m3s_kit import (
    RETRY_LAST_SUCCESS,
    STALE_EVENTS,
    STALE_TOTAL,
    charge_event,
    checkout_event,
    cure,
    ep,
    event_row,
    gauge,
    install_faults,
    m3s_scratch_fixtures,
    m3s_settings,
    make_org,
    now,
    poison,
    purchases,
    refunds,
    rid,
    sample,
    scalar,
    store_raw,
)

pytestmark = [pytest.mark.postgres]
m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()

M = 10_000
MINUTE = dt.timedelta(minutes=1)
STALE_AFTER = dt.timedelta(seconds=900)
MARKER = "m3s backlog marker"


async def _stuck(case: str, pool, url, s, t: dt.datetime) -> tuple[str, str]:
    """Store one event that stays unapplied; returns (event id, stale counter type)."""
    process_event = ep("process_event")
    if case == "unknown_org":
        ev = checkout_event(uuid.uuid4(), 1000)
        out = await process_event(pool, ev, settings=s, now=t)
        assert out.result == "unapplied" and not getattr(out, "stored_after_error", False)
        return ev["id"], "checkout.session.completed"
    if case == "clawback_no_payment":
        ev = charge_event(rid("pi_test"), 100, 1000)
        assert (await process_event(pool, ev, settings=s, now=t)).result == "unapplied"
        return ev["id"], "charge.refunded"
    # A poison clawback whose payment has a stripe_payments row.
    org = await make_org(pool)
    pi = rid("pi_test")
    paid = await process_event(
        pool, checkout_event(org, 1000, payment_intent=pi), settings=s, now=t
    )
    assert paid.result == "credited"
    ev = charge_event(pi, 100, 1000)
    await poison(url, ev["id"])
    out = await process_event(pool, ev, settings=s, now=t)
    assert out.stored_after_error
    return ev["id"], "charge.refunded"


@pytest.mark.parametrize("case", ["unknown_org", "clawback_no_payment", "poison_with_payment"])
async def test_a24_stuck_event_counts_once_then_a_second_counts_again(
    scratch_pool, scratch_url, case
):
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    evt, typ = await _stuck(case, scratch_pool, scratch_url, s, t0)
    base = sample(STALE_TOTAL, type=typ)

    await retry_unapplied(scratch_pool, settings=s, now=t0 + 14 * MINUTE)
    assert sample(STALE_TOTAL, type=typ) == base
    row = await event_row(scratch_pool, evt)
    assert row is not None and row["stale_reported_at"] is None

    t_stale = t0 + STALE_AFTER + dt.timedelta(seconds=30)
    await retry_unapplied(scratch_pool, settings=s, now=t_stale)
    assert sample(STALE_TOTAL, type=typ) == base + 1
    row = await event_row(scratch_pool, evt)
    assert row is not None and row["processed_at"] is None
    assert row["stale_reported_at"] == t_stale
    assert gauge(STALE_EVENTS) == 1

    await retry_unapplied(scratch_pool, settings=s, now=t_stale + 5 * MINUTE)
    assert sample(STALE_TOTAL, type=typ) == base + 1  # once per event
    row = await event_row(scratch_pool, evt)
    assert row is not None and row["stale_reported_at"] == t_stale

    t1 = t_stale + 10 * MINUTE  # a second stuck event, later
    second, typ2 = await _stuck("clawback_no_payment", scratch_pool, scratch_url, s, t1)
    base2 = sample(STALE_TOTAL, type=typ2)
    await retry_unapplied(scratch_pool, settings=s, now=t1 + STALE_AFTER + MINUTE)
    assert sample(STALE_TOTAL, type=typ2) == base2 + 1
    assert gauge(STALE_EVENTS) == 2
    row = await event_row(scratch_pool, second)
    assert row is not None and row["stale_reported_at"] is not None


async def test_a24_other_types_count_as_other_and_processed_ones_never(scratch_pool):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    t0 = now()
    ignored = {
        "id": rid("evt_test"),
        "type": "customer.created",
        "livemode": False,
        "data": {"object": {"id": rid("cus_test"), "object": "customer"}},
    }
    assert (await process_event(scratch_pool, ignored, settings=s, now=t0)).result == "ignored"
    base = {t: sample(STALE_TOTAL, type=t) for t in ("other", "charge.refunded")}
    await retry_unapplied(scratch_pool, settings=s, now=t0 + 2 * STALE_AFTER)
    assert sample(STALE_TOTAL, type="other") == base["other"]  # processed: never stale
    assert gauge(STALE_EVENTS) == 0
    row = await event_row(scratch_pool, ignored["id"])
    assert row is not None and row["stale_reported_at"] is None


# ------------------------------------------------------------------ backlogs (T21, T22)

BACKLOG_SQL = """
INSERT INTO stripe_events (id, type, payload, received_at, result, last_error)
SELECT 'evt_test_backlog_' || g || '_' || %(tag)s,
       'checkout.session.completed',
       jsonb_build_object(
           'id', 'evt_test_backlog_' || g || '_' || %(tag)s,
           'object', 'event',
           'type', 'checkout.session.completed',
           'livemode', false,
           'data', jsonb_build_object('object', jsonb_build_object(
               'id', 'cs_test_backlog_' || g || '_' || %(tag)s,
               'object', 'checkout.session',
               'amount_total', 1000,
               'currency', 'usd',
               'client_reference_id', gen_random_uuid()::text,
               'metadata', jsonb_build_object('purpose', 'loom_credits'),
               'payment_status', 'paid',
               'livemode', false))),
       %(start)s + g * interval '1 second',
       'unapplied',
       %(marker)s
FROM generate_series(1, %(n)s) g
"""


# A backlog larger than the run's `limit` (§6.13: tests pass a small `limit`; the default
# of 500 is checked on the signature, so the boundary stays tested within flake-check).
LIMIT = 5
BACKLOG = LIMIT + 1


async def backlog(pool, *, start: dt.datetime, n: int = BACKLOG) -> None:
    """`n` unapplied checkouts of unknown orgs (an operator's test), older than `start`
    plus n seconds, each marked so a re-run (which clears `last_error`) is visible."""
    async with pool.connection() as conn:
        await conn.execute(
            BACKLOG_SQL,  # type: ignore[arg-type]
            {"tag": uuid.uuid4().hex[:8], "start": start, "marker": MARKER, "n": n},
        )


async def markers(pool) -> int:
    return int(
        await scalar(pool, "SELECT count(*) FROM stripe_events WHERE last_error = %s", MARKER)
    )


async def test_a24_t21_raced_clawback_behind_a_backlog_is_applied_never_stale(scratch_pool):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    t0 = now()
    await backlog(scratch_pool, start=t0 - dt.timedelta(days=1))
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    paid = await process_event(
        scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s, now=t0
    )
    assert paid.result == "credited"
    raced = charge_event(pi, 600, 1000)
    await store_raw(scratch_pool, raced, received_at=t0)
    refunded_stale = sample(STALE_TOTAL, type="charge.refunded")

    assert await retry_unapplied(scratch_pool, limit=LIMIT, settings=s, now=t0 + MINUTE) == 1
    assert await refunds(scratch_pool, org) == [-600 * M]
    row = await event_row(scratch_pool, raced["id"])
    assert row is not None and row["result"] == "refunded" and row["stale_reported_at"] is None
    assert sample(STALE_TOTAL, type="charge.refunded") == refunded_stale
    # Newest first, `limit` per run: the clawback and the 4 newest; the 2 oldest wait.
    assert await markers(scratch_pool) == BACKLOG - (LIMIT - 1)
    assert gauge(STALE_EVENTS) == BACKLOG  # the whole backlog is stale (S1b)


async def test_a24_t22_purchase_stored_behind_a_backlog_is_rerun_next_run(
    scratch_pool, scratch_url
):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    await backlog(scratch_pool, start=t0 - dt.timedelta(hours=2))
    org = await make_org(scratch_pool)
    ev = checkout_event(org, 1200)
    await poison(scratch_url, ev["id"])
    out = await process_event(scratch_pool, ev, settings=s, now=t0)
    assert out.result == "unapplied" and out.stored_after_error  # the 500 path
    await cure(scratch_url, ev["id"])
    assert await retry_unapplied(scratch_pool, limit=LIMIT, settings=s, now=t0 + MINUTE) == 1
    row = await event_row(scratch_pool, ev["id"])
    assert row is not None and row["processed_at"] is not None and row["result"] == "credited"
    assert row["stale_reported_at"] is None
    assert await purchases(scratch_pool, org) == [1200 * M]
    # The purchase and the 4 newest of the backlog were re-run; the 2 oldest wait.
    assert await markers(scratch_pool) == BACKLOG - (LIMIT - 1)


def test_a24_retry_unapplied_default_limit_is_500():
    """§6.13, §8.11: `retry_unapplied(pool, *, limit=500, ...)`; the backlog tests above
    pass a small `limit`, this pins the default."""
    params = inspect.signature(ep("retry_unapplied")).parameters
    assert params["limit"].default == 500
    assert params["limit"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_a24_unapplied_event_older_than_30_days_is_still_rerun(scratch_pool):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    t0 = now()
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    assert (
        await process_event(
            scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s, now=t0
        )
    ).result == "credited"
    old = charge_event(pi, 200, 1000)
    await store_raw(scratch_pool, old, received_at=t0 - dt.timedelta(days=31))
    assert await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE) == 1
    assert await refunds(scratch_pool, org) == [-200 * M]


async def test_a24_s5_gauge_moves_on_a_completed_run(scratch_pool):
    """§6.13: the gauge is the Unix time in seconds of the run's `now` (the injected one),
    never the wall clock; hours ahead here, so the two cannot be confused."""
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings()
    t_run = now() + dt.timedelta(hours=3)
    await retry_unapplied(scratch_pool, settings=s, now=t_run)
    after = gauge(RETRY_LAST_SUCCESS)
    assert after is not None and abs(after - t_run.timestamp()) < 1, after
    t_next = t_run + MINUTE  # the next completed run moves it to that run's `now`
    await retry_unapplied(scratch_pool, settings=s, now=t_next)
    after = gauge(RETRY_LAST_SUCCESS)
    assert after is not None and abs(after - t_next.timestamp()) < 1, after
