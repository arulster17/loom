"""`stripe-event resolve` and `list` through `resolve_event` and `list_events`, the
reopen sweep, and their races (docs/phase1-m3-spec.md §8.6, §8.11; done-when A31; trace
T24).

Refusal tests assert only on their own rows in the session database; everything that
runs `retry_unapplied` or needs a trigger uses a scratch database (§16). Races hold a
transaction at its COMMIT with a scratch-database gate (a deferred trigger waiting on an
advisory lock the test holds) and wait on `pg_blocking_pids` (no sleeps)."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import time
import uuid

import pytest
from billing_m3s_kit import (
    STALE_TOTAL,
    charge_event,
    checkout_event,
    cure,
    dispute_event,
    ep,
    event_row,
    gate,
    holder,
    install_faults,
    ledger_count,
    m3s_scratch_fixtures,
    m3s_settings,
    make_admin,
    make_org,
    now,
    poison,
    purchases,
    refunds,
    rid,
    sample,
    scalar,
    store_raw,
    wait_task_or_blocked,
)

pytestmark = [pytest.mark.postgres]
m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()

M = 10_000
MINUTE = dt.timedelta(minutes=1)
RESOLVE = "loom_billing.resolve"


def resolve_api():
    return ep("resolve_event", RESOLVE), ep("ResolveRefused", RESOLVE)


def reason_token() -> str:
    """A reason no other test writes: in the shared session database a test finds its
    own audit rows by it (§6.13: `details` holds `reason`)."""
    return f"m3s reason {uuid.uuid4().hex}"


async def audit_rows(pool, reason: str | None = None) -> list[tuple]:
    """`admin_audit_log` rows as (actor_user_id, action, target_type, target_id,
    details): every row (a scratch database: all of them are the test's own), or those
    whose `details` carry `reason` (the session database)."""
    query = "SELECT actor_user_id, action, target_type, target_id, details FROM admin_audit_log"
    async with pool.connection() as conn, conn.cursor() as cur:
        if reason is None:
            await cur.execute(query + " ORDER BY id")
        else:
            await cur.execute(query + " WHERE details::text LIKE %s ORDER BY id", (f"%{reason}%",))
        return await cur.fetchall()


def assert_resolve_audit(row: tuple, *, admin_id: uuid.UUID, evt: str, reason: str) -> None:
    """§6.13: the admin's `users.id`, `stripe_event.resolve`, target `stripe_event` and
    the event id, `details` holding `reason` (other keys allowed)."""
    actor, action, target_type, target_id, details = row
    assert actor == admin_id and action == "stripe_event.resolve", row
    assert (target_type, target_id) == ("stripe_event", evt), row
    if isinstance(details, str):
        details = json.loads(details)
    assert isinstance(details, dict) and details.get("reason") == reason, details


# ------------------------------------------------------------------ refusals


async def _refusal_case(case: str, pool, s) -> tuple[str, str]:
    """(event id, admin email) for a resolve that must be refused."""
    process_event = ep("process_event")
    _, admin = await make_admin(pool)
    if case == "non_admin":
        _, user = await make_admin(pool, role="user")
        ev = checkout_event(uuid.uuid4(), 1000)
        await process_event(pool, ev, settings=s)
        return ev["id"], user
    if case == "unknown_admin":
        ev = checkout_event(uuid.uuid4(), 1000)
        await process_event(pool, ev, settings=s)
        return ev["id"], f"nobody-{uuid.uuid4().hex[:8]}@example.test"
    if case == "unknown_event":
        return rid("evt_test"), admin
    if case == "processed_event":
        org = await make_org(pool)
        ev = checkout_event(org, 1000)
        assert (await process_event(pool, ev, settings=s)).result == "credited"
        return ev["id"], admin
    if case.startswith("clawback_pi_"):
        pi = {"clawback_pi_missing": None, "clawback_pi_empty": "", "clawback_pi_number": 123}[case]
        ev = charge_event(pi, 100, 1000)  # type: ignore[arg-type]
        if pi is None:
            del ev["data"]["object"]["payment_intent"]
        assert (await process_event(pool, ev, settings=s)).result == "unapplied"
        return ev["id"], admin
    assert case == "clawback_with_payment"
    org = await make_org(pool)
    pi = rid("pi_test")
    await process_event(pool, checkout_event(org, 1000, payment_intent=pi), settings=s)
    ev = dispute_event(pi, 300)
    await store_raw(pool, ev)  # raced its purchase: unapplied, and its payment exists
    return ev["id"], admin


@pytest.mark.parametrize(
    "case",
    [
        "non_admin",
        "unknown_admin",
        "unknown_event",
        "processed_event",
        "clawback_pi_missing",
        "clawback_pi_empty",
        "clawback_pi_number",
        "clawback_with_payment",
    ],
)
async def test_a31_refusals_change_nothing(pool, case):
    resolve_event, refused = resolve_api()
    s = m3s_settings()
    evt, email = await _refusal_case(case, pool, s)
    before = await event_row(pool, evt)
    reason = reason_token()
    with pytest.raises(refused) as caught:
        await resolve_event(pool, evt, admin_email=email, reason=reason, settings=s)
    assert str(caught.value)  # the reason the CLI prints
    assert await event_row(pool, evt) == before
    assert await audit_rows(pool, reason) == []
    assert await ledger_count(pool, evt) == (1 if case == "processed_event" else 0)


# ------------------------------------------------------------------ resolving


async def test_a31_resolving_a_credit_event_posts_nothing_and_is_audited(pool):
    process_event = ep("process_event")
    resolve_event, refused = resolve_api()
    s = m3s_settings()
    admin_id, admin = await make_admin(pool)
    ev = checkout_event(uuid.uuid4(), 1000)  # an operator's own test: unknown org
    assert (await process_event(pool, ev, settings=s)).result == "unapplied"
    t = now()
    reason = reason_token()
    out = await resolve_event(pool, ev["id"], admin_email=admin, reason=reason, settings=s, now=t)
    assert out.result == "resolved_by_admin"
    row = await event_row(pool, ev["id"])
    assert row is not None and row["processed_at"] == t and row["result"] == "resolved_by_admin"
    audit = await audit_rows(pool, reason)
    assert len(audit) == 1
    assert_resolve_audit(audit[0], admin_id=admin_id, evt=ev["id"], reason=reason)
    assert await ledger_count(pool, ev["id"]) == 0
    again = reason_token()
    with pytest.raises(refused):  # once resolved, it is processed
        await resolve_event(pool, ev["id"], admin_email=admin, reason=again, settings=s)
    assert await audit_rows(pool, again) == []


async def test_a31_resolving_a_clawback_without_payment_takes_nothing(pool):
    process_event = ep("process_event")
    resolve_event, _ = resolve_api()
    s = m3s_settings()
    admin_id, admin = await make_admin(pool)
    ev = dispute_event(rid("pi_test"), 500)
    assert (await process_event(pool, ev, settings=s)).result == "unapplied"
    reason = reason_token()
    out = await resolve_event(pool, ev["id"], admin_email=admin, reason=reason, settings=s)
    assert out.result == "resolved_by_admin"
    row = await event_row(pool, ev["id"])
    assert row is not None and row["processed_at"] is not None
    assert await ledger_count(pool, ev["id"]) == 0
    audit = await audit_rows(pool, reason)
    assert len(audit) == 1
    assert_resolve_audit(audit[0], admin_id=admin_id, evt=ev["id"], reason=reason)


async def test_a31_list_events_shows_unapplied_with_their_errors(scratch_pool, scratch_url):
    process_event = ep("process_event")
    list_events = ep("list_events", RESOLVE)
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    org = await make_org(scratch_pool)
    done = checkout_event(org, 1000)
    await process_event(scratch_pool, done, settings=s, now=t0)
    stuck = charge_event(rid("pi_test"), 100, 1000)
    await process_event(scratch_pool, stuck, settings=s, now=t0 + MINUTE)
    bad = checkout_event(org, 500)
    await poison(scratch_url, bad["id"])
    await process_event(scratch_pool, bad, settings=s, now=t0 + 2 * MINUTE)

    keys = {"id", "type", "received_at", "result", "last_error", "error_count", "stale_reported_at"}
    unapplied = await list_events(scratch_pool, unapplied=True)
    assert [e["id"] for e in unapplied] == [bad["id"], stuck["id"]]  # received_at DESC
    for e in unapplied:
        assert keys <= set(e)
    by_id = {e["id"]: e for e in unapplied}
    assert "m3s poison" in by_id[bad["id"]]["last_error"] and by_id[bad["id"]]["error_count"] == 1
    assert by_id[stuck["id"]]["result"] == "unapplied" and by_id[stuck["id"]]["last_error"] is None
    every = await list_events(scratch_pool)
    assert [e["id"] for e in every] == [bad["id"], stuck["id"], done["id"]]
    assert every[2]["result"] == "credited"


# ------------------------------------------------------------------ T24: the reopen sweep


async def _poisoned_purchase_and_resolved_refund(pool, url, s, t0, *, stale_first: bool):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    resolve_event, _ = resolve_api()
    _, admin = await make_admin(pool)
    org = await make_org(pool)
    pi = rid("pi_test")
    purchase = checkout_event(org, 1000, payment_intent=pi)
    await poison(url, purchase["id"])
    assert (await process_event(pool, purchase, settings=s, now=t0)).stored_after_error
    refund = charge_event(pi, 500, 1000)
    assert (await process_event(pool, refund, settings=s, now=t0)).result == "unapplied"
    t = t0
    if stale_first:
        t = t0 + dt.timedelta(minutes=16)
        await retry_unapplied(pool, settings=s, now=t)
        row = await event_row(pool, refund["id"])
        assert row is not None and row["stale_reported_at"] is not None
    await resolve_event(pool, refund["id"], admin_email=admin, reason="no payment", settings=s)
    return org, purchase, refund, t


async def test_a31_t24_resolved_clawback_is_reopened_and_applied_once(scratch_pool, scratch_url):
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    org, purchase, refund, t = await _poisoned_purchase_and_resolved_refund(
        scratch_pool, scratch_url, s, t0, stale_first=False
    )
    await cure(scratch_url, purchase["id"])  # the release that fixes the purchase
    await retry_unapplied(scratch_pool, settings=s, now=t + MINUTE)
    assert await purchases(scratch_pool, org) == [1000 * M]
    assert await refunds(scratch_pool, org) == []  # its pending query skips the resolved one
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["result"] == "resolved_by_admin"

    await retry_unapplied(scratch_pool, settings=s, now=t + 2 * MINUTE)  # sweep, then re-run
    assert await refunds(scratch_pool, org) == [-500 * M]
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["processed_at"] is not None and row["result"] == "refunded"
    await retry_unapplied(scratch_pool, settings=s, now=t + 3 * MINUTE)
    assert await refunds(scratch_pool, org) == [-500 * M]  # once


async def test_a31_t24_reopened_clawback_that_fails_is_unapplied_and_stale_again(
    scratch_pool, scratch_url
):
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    org, purchase, refund, t = await _poisoned_purchase_and_resolved_refund(
        scratch_pool, scratch_url, s, t0, stale_first=True
    )
    await cure(scratch_url, purchase["id"])
    await retry_unapplied(scratch_pool, settings=s, now=t + MINUTE)  # credits the purchase
    assert await purchases(scratch_pool, org) == [1000 * M]
    await poison(scratch_url, refund["id"])  # its handling now fails
    t_reopen = t + 2 * MINUTE  # 18 minutes after the refund's received_at (t0)

    # One run (§8.3): the sweep reopens it (clearing stale_reported_at), the re-run fails
    # and stores the error, and stale marking marks it again at once: staleness counts
    # from received_at, long past.
    base = sample(STALE_TOTAL, type="charge.refunded")
    await retry_unapplied(scratch_pool, settings=s, now=t_reopen)
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["processed_at"] is None and row["result"] == "unapplied"
    assert row["last_error"] and "m3s poison" in row["last_error"]
    assert row["stale_reported_at"] == t_reopen  # cleared by the sweep, marked by this run
    assert sample(STALE_TOTAL, type="charge.refunded") == base + 1
    assert await refunds(scratch_pool, org) == []

    await retry_unapplied(scratch_pool, settings=s, now=t_reopen + MINUTE)
    assert sample(STALE_TOTAL, type="charge.refunded") == base + 1  # once
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["stale_reported_at"] == t_reopen
    assert row["processed_at"] is None and await refunds(scratch_pool, org) == []


async def test_a31_sweep_skips_a_held_row_and_takes_it_next_run(scratch_pool, scratch_url):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    resolve_event, _ = resolve_api()
    s = m3s_settings(stripe_lock_timeout_ms=5000)
    _, admin = await make_admin(scratch_pool)
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 300, 1000)
    await process_event(scratch_pool, refund, settings=s)
    await resolve_event(scratch_pool, refund["id"], admin_email=admin, reason="r", settings=s)
    assert (
        await process_event(scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s)
    ).result == "credited"
    async with holder(scratch_url) as h:
        await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", refund["id"])
        t0 = time.monotonic()
        await retry_unapplied(scratch_pool, settings=s)
        assert time.monotonic() - t0 < 4.5  # it did not wait on the held row
        row = await event_row(scratch_pool, refund["id"])
        assert row is not None and row["result"] == "resolved_by_admin"
        await h.rollback()
    await retry_unapplied(scratch_pool, settings=s)
    assert await refunds(scratch_pool, org) == [-300 * M]


async def test_a31_resolved_credit_event_sharing_the_payment_is_never_reopened(
    scratch_pool,
):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    resolve_event, _ = resolve_api()
    s = m3s_settings()
    _, admin = await make_admin(scratch_pool)
    pi = rid("pi_test")
    stray = checkout_event(uuid.uuid4(), 1000, payment_intent=pi)
    assert (await process_event(scratch_pool, stray, settings=s)).result == "unapplied"
    await resolve_event(scratch_pool, stray["id"], admin_email=admin, reason="r", settings=s)
    org = await make_org(scratch_pool)
    await process_event(scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s)
    await retry_unapplied(scratch_pool, settings=s)
    await retry_unapplied(scratch_pool, settings=s)
    row = await event_row(scratch_pool, stray["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "resolved_by_admin"
    assert await purchases(scratch_pool, org) == [1000 * M]


# ------------------------------------------------------------------ races


async def test_a31_race_purchase_takes_the_clawback_first_resolve_refuses(
    scratch_pool, scratch_url
):
    """The purchase locked and applied the clawback and waits at its COMMIT; the resolve
    waits on the clawback's row, then finds it processed."""
    process_event = ep("process_event")
    resolve_event, refused = resolve_api()
    await install_faults(scratch_url)
    s = m3s_settings(stripe_lock_timeout_ms=10_000)
    _, admin = await make_admin(scratch_pool)
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 400, 1000)
    await process_event(scratch_pool, refund, settings=s)
    purchase = checkout_event(org, 1000, payment_intent=pi)
    await gate(scratch_url, "stripe_events", purchase["id"], key=91_001)
    async with holder(scratch_url) as h:
        await h.run("SELECT pg_advisory_xact_lock(91001)")
        buy = asyncio.create_task(process_event(scratch_pool, purchase, settings=s))
        buyer = await wait_task_or_blocked(scratch_url, h.pid, buy)
        res = asyncio.create_task(
            resolve_event(scratch_pool, refund["id"], admin_email=admin, reason="r", settings=s)
        )
        await wait_task_or_blocked(scratch_url, buyer, res)
        await h.rollback()
        assert (await buy).result == "credited"
        with pytest.raises(refused):
            await res
    assert await refunds(scratch_pool, org) == [-400 * M]
    assert await audit_rows(scratch_pool) == []  # the scratch database's: none at all


async def test_a31_race_resolve_commits_first_purchase_skips_sweep_applies(
    scratch_pool, scratch_url
):
    """The resolve holds the clawback's row at its COMMIT; the purchase's pending query
    waits on it, then skips it (no longer unprocessed); the next R run's sweep applies
    it. Applied once."""
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    resolve_event, _ = resolve_api()
    await install_faults(scratch_url)
    s = m3s_settings(stripe_lock_timeout_ms=10_000)
    _, admin = await make_admin(scratch_pool)
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 400, 1000)
    await process_event(scratch_pool, refund, settings=s)
    await gate(scratch_url, "admin_audit_log", "*", key=91_002)  # the resolve's audit row
    async with holder(scratch_url) as h:
        await h.run("SELECT pg_advisory_xact_lock(91002)")
        res = asyncio.create_task(
            resolve_event(scratch_pool, refund["id"], admin_email=admin, reason="r", settings=s)
        )
        resolver = await wait_task_or_blocked(scratch_url, h.pid, res)
        buy = asyncio.create_task(
            process_event(scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s)
        )
        await wait_task_or_blocked(scratch_url, resolver, buy)
        await h.rollback()
        assert (await res).result == "resolved_by_admin"
        assert (await buy).result == "credited"
    assert await refunds(scratch_pool, org) == []
    await retry_unapplied(scratch_pool, settings=s)
    assert await refunds(scratch_pool, org) == [-400 * M]
    await retry_unapplied(scratch_pool, settings=s)
    assert await refunds(scratch_pool, org) == [-400 * M]


@pytest.mark.parametrize("r_outcome", ["applied", "still_unapplied"])
async def test_a31_race_resolve_waits_for_retry_holding_the_row(
    scratch_pool, scratch_url, r_outcome
):
    retry_unapplied = ep("retry_unapplied")
    resolve_event, refused = resolve_api()
    await install_faults(scratch_url)
    s = m3s_settings(stripe_lock_timeout_ms=10_000)
    _, admin = await make_admin(scratch_pool)
    if r_outcome == "applied":
        org = await make_org(scratch_pool)
        ev = checkout_event(org, 1000)  # stored, not yet applied (e.g. after a 500)
        await store_raw(scratch_pool, ev)
        await gate(scratch_url, "ledger_transactions", f"stripe:{ev['id']}", key=91_003)
    else:
        ev = checkout_event(uuid.uuid4(), 1000)  # unknown org: R leaves it unapplied
        await store_raw(scratch_pool, ev)
        await gate(scratch_url, "stripe_events", ev["id"], key=91_003)
    async with holder(scratch_url) as h:
        await h.run("SELECT pg_advisory_xact_lock(91003)")
        job = asyncio.create_task(retry_unapplied(scratch_pool, settings=s))
        r_pid = await wait_task_or_blocked(scratch_url, h.pid, job)
        res = asyncio.create_task(
            resolve_event(scratch_pool, ev["id"], admin_email=admin, reason="r", settings=s)
        )
        await wait_task_or_blocked(scratch_url, r_pid, res)
        await h.rollback()
        await job
        if r_outcome == "applied":
            with pytest.raises(refused):
                await res
        else:
            assert (await res).result == "resolved_by_admin"
    row = await event_row(scratch_pool, ev["id"])
    assert row is not None and row["processed_at"] is not None
    want = "credited" if r_outcome == "applied" else "resolved_by_admin"
    assert row["result"] == want
    n_audit = int(await scalar(scratch_pool, "SELECT count(*) FROM admin_audit_log"))
    assert n_audit == (0 if r_outcome == "applied" else 1)
