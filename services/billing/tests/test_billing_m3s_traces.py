"""Stripe traces against real Postgres through `process_event` (docs/phase1-m3-spec.md
§8.7, §8.11; done-when A16, A17 sequential, A18).

These run in the shared session database and assert only on their own rows (random
session, PaymentIntent and dispute ids, §16); none runs `retry_unapplied`."""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid

import pytest
from billing_fixtures import STUB_WEBHOOK_SECRET, ledger_is_balanced
from billing_m3s_kit import (
    WEBHOOKS,
    billing_client,
    charge_event,
    checkout_event,
    deliver,
    dispute_event,
    ep,
    event_row,
    holder,
    ledger_count,
    m3s_settings,
    make_org,
    purchases,
    refunds,
    rid,
    sample,
    wait_task_or_blocked,
)
from psycopg_pool import AsyncConnectionPool

from loom_platform.ledger import PROMO_EXPENSE, Entry, org_balance, org_credit_account
from loom_platform.ledger import post_transaction as post

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

M = 10_000  # micros per cent


# ------------------------------------------------------------------ A16: T1


async def test_a16_t1_duplicate_delivery_credits_once(pool):
    process_event = ep("process_event")
    s = m3s_settings()
    org = await make_org(pool)
    ev = checkout_event(org, 2500)
    first = await process_event(pool, ev, settings=s)
    assert first.result == "credited" and first.processed
    again = await process_event(pool, ev, settings=s)
    assert again.result == "duplicate"
    assert await purchases(pool, org) == [2500 * M]
    row = await event_row(pool, ev["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "credited"  # a duplicate does not overwrite the result
    assert row["last_error"] is None and row["error_count"] == 0


# ------------------------------------------------------------------ A16: T2


async def test_a16_t2_same_event_two_workers_second_waits_then_duplicate(pool, pg_url):
    """W1 inserts the row (uncommitted) and is held on a lock its purchase needs (the
    org row, held by the test); W2's insert waits on W1's uncommitted row; W1 commits:
    W2 reads the committed row -> duplicate."""
    process_event = ep("process_event")
    s = m3s_settings(stripe_lock_timeout_ms=8000)
    org = await make_org(pool)
    ev = checkout_event(org, 1500)
    async with holder(pg_url) as h:
        await h.run("SELECT 1 FROM orgs WHERE id = %s FOR UPDATE", org)
        w1 = asyncio.create_task(process_event(pool, ev, settings=s))
        w1_pid = await wait_task_or_blocked(pg_url, h.pid, w1)
        w2 = asyncio.create_task(process_event(pool, ev, settings=s))
        await wait_task_or_blocked(pg_url, w1_pid, w2)
        await h.rollback()  # W1 goes on and commits
        o1, o2 = await w1, await w2
    assert (o1.result, o2.result) == ("credited", "duplicate")
    assert await purchases(pool, org) == [1500 * M]
    assert await ledger_count(pool, ev["id"]) == 1


async def test_a16_t2_first_worker_fails_second_applies(pool, pg_url):
    """W1 fails (a lock timeout) while W2 waits on its uncommitted row: W2's insert then
    proceeds and W2 applies the event; W1's second transaction finds it processed
    (`duplicate`), times out on W2's row (503 path, StoreFailed), or stores it unapplied
    before W2 takes it (500 path; W2 then applies the stored row). Whatever the order:
    one purchase, credited once."""
    process_event = ep("process_event")
    store_failed = ep("StoreFailed")
    s1 = m3s_settings(stripe_lock_timeout_ms=1500)
    s2 = m3s_settings(stripe_lock_timeout_ms=10_000)
    org = await make_org(pool)
    ev = checkout_event(org, 1100)
    async with holder(pg_url) as h:
        await h.run("SELECT 1 FROM orgs WHERE id = %s FOR UPDATE", org)
        w1 = asyncio.create_task(process_event(pool, ev, settings=s1))
        w1_pid = await wait_task_or_blocked(pg_url, h.pid, w1)
        w2 = asyncio.create_task(process_event(pool, ev, settings=s2))
        w2_pid = await wait_task_or_blocked(pg_url, w1_pid, w2)
        # W1 times out on the org row and rolls back; W2 inserts and waits on the org.
        await wait_task_or_blocked(pg_url, h.pid, w2, waiter_pid=w2_pid)
        await h.rollback()
        o2 = await w2
        try:
            first = await w1
            o1 = "stored" if getattr(first, "stored_after_error", False) else first.result
        except store_failed:
            o1 = "store_failed"
    assert o2.result == "credited"
    assert o1 in ("duplicate", "store_failed", "stored"), o1
    if o1 == "stored":
        assert first.result == "unapplied"
    assert await purchases(pool, org) == [1100 * M]
    assert await ledger_count(pool, ev["id"]) == 1
    row = await event_row(pool, ev["id"])
    assert row is not None and row["processed_at"] is not None and row["result"] == "credited"


# ------------------------------------------------------------------ A16: T3


@pytest.mark.parametrize("unpaid_first", [True, False])
async def test_a16_t3_two_event_ids_one_session(pool, unpaid_first):
    process_event = ep("process_event")
    s = m3s_settings()
    org = await make_org(pool)
    sid = rid("cs_test")
    unpaid = checkout_event(org, 900, session_id=sid, paid=False)
    succeeded = checkout_event(
        org, 900, session_id=sid, typ="checkout.session.async_payment_succeeded"
    )
    order = [unpaid, succeeded] if unpaid_first else [succeeded, unpaid]
    results = {e["id"]: (await process_event(pool, e, settings=s)).result for e in order}
    assert results == {unpaid["id"]: "not_paid", succeeded["id"]: "credited"}
    assert await purchases(pool, org) == [900 * M]
    for e, want in ((unpaid, "not_paid"), (succeeded, "credited")):
        row = await event_row(pool, e["id"])
        assert row is not None and row["processed_at"] is not None and row["result"] == want


async def test_a16_t3_both_paid_concurrently_credit_once(pool):
    process_event = ep("process_event")
    s = m3s_settings()
    org = await make_org(pool)
    sid = rid("cs_test")
    a = checkout_event(org, 700, session_id=sid)
    b = checkout_event(org, 700, session_id=sid, typ="checkout.session.async_payment_succeeded")
    outs = await asyncio.gather(*(process_event(pool, e, settings=s) for e in (a, b)))
    assert sorted(o.result for o in outs) == ["already_credited", "credited"]
    assert await purchases(pool, org) == [700 * M]


# ------------------------------------------------------------------ A16: T5, T6


async def _credited_payment(pool, s, cents: int = 1000):
    process_event = ep("process_event")
    org = await make_org(pool)
    pi = rid("pi_test")
    out = await process_event(pool, checkout_event(org, cents, payment_intent=pi), settings=s)
    assert out.result == "credited"
    return org, pi


@pytest.mark.parametrize("order", ["new_first", "old_first", "concurrent"])
async def test_a16_t5_partial_refunds_in_any_order_take_1000(pool, order):
    process_event = ep("process_event")
    s = m3s_settings()
    org, pi = await _credited_payment(pool, s)
    e2 = charge_event(pi, 500, 1000)
    e3 = charge_event(pi, 1000, 1000)
    if order == "new_first":
        assert (await process_event(pool, e3, settings=s)).result == "refunded"
        assert (await process_event(pool, e2, settings=s)).result == "already_refunded"
        assert await refunds(pool, org) == [-1000 * M]
    elif order == "old_first":
        assert (await process_event(pool, e2, settings=s)).result == "refunded"
        assert (await process_event(pool, e3, settings=s)).result == "refunded"
        assert await refunds(pool, org) == [-500 * M, -500 * M]
    else:
        await asyncio.gather(*(process_event(pool, e, settings=s) for e in (e2, e3)))
    assert sum(await refunds(pool, org)) == -1000 * M
    for e in (e2, e3):
        row = await event_row(pool, e["id"])
        assert row is not None and row["processed_at"] is not None
        assert row["result"] in ("refunded", "already_refunded")


async def test_a16_t6_refund_and_dispute_never_exceed_the_purchase(pool):
    process_event = ep("process_event")
    s = m3s_settings()
    org, pi = await _credited_payment(pool, s)
    assert (await process_event(pool, charge_event(pi, 600, 1000), settings=s)).result == (
        "refunded"
    )
    out = await process_event(pool, dispute_event(pi, 1000), settings=s)
    assert out.result == "refunded" and out.amount_micros == 400 * M
    assert await refunds(pool, org) == [-600 * M, -400 * M]
    async with pool.connection() as conn:
        assert await org_balance(conn, org) == 0
        assert await ledger_is_balanced(conn)


# ------------------------------------------------------------------ A16: T16, T18


@pytest.mark.parametrize("status", ["suspended", "rejected", "pending_approval"])
async def test_a16_t16_non_active_org_is_credited_anyway(pool, status):
    process_event = ep("process_event")
    org = await make_org(pool, status=status)
    ev = checkout_event(org, 1200)
    assert (await process_event(pool, ev, settings=m3s_settings())).result == "credited"
    assert await purchases(pool, org) == [1200 * M]
    row = await event_row(pool, ev["id"])
    assert row is not None and row["result"] == "credited"


async def test_a16_t18_dispute_won_then_refunded_takes_nothing(pool):
    """The documented limitation (§8.10): an admin adjustment after a won dispute does
    not lower `clawed_back_micros`, so a later full refund takes nothing back."""
    process_event = ep("process_event")
    s = m3s_settings()
    org, pi = await _credited_payment(pool, s)
    assert (await process_event(pool, dispute_event(pi, 1000), settings=s)).result == "refunded"
    async with pool.connection() as conn, conn.transaction():
        acct = await org_credit_account(conn, org)
        await post(
            conn,
            kind="adjustment",
            idempotency_key=f"admin:adjust:{secrets.token_hex(8)}",
            org_id=org,
            entries=[Entry(acct, 1000 * M), Entry(PROMO_EXPENSE, -1000 * M)],
            memo="dispute won",
        )
    refund = charge_event(pi, 1000, 1000)
    assert (await process_event(pool, refund, settings=s)).result == "already_refunded"
    assert await refunds(pool, org) == [-1000 * M]
    async with pool.connection() as conn:
        assert await org_balance(conn, org) == 1000 * M
    row = await event_row(pool, refund["id"])
    assert row is not None and row["result"] == "already_refunded"


# ------------------------------------------------------------------ T11


@pytest.mark.parametrize("case", ["unknown_org", "wrong_currency", "no_amount"])
async def test_a16_t11_signed_but_not_applicable_is_stored_unapplied_200(pool, redis_client, case):
    """Not an error: 200 (Stripe retrying cannot fix it), stored unapplied with no
    `last_error` (S1a and an admin's resolve take it from there, A24, A31)."""
    ep("process_event")
    org = await make_org(pool)
    ev = {
        "unknown_org": lambda: checkout_event(uuid.uuid4(), 1000),
        "wrong_currency": lambda: checkout_event(org, 1000, currency="eur"),
        "no_amount": lambda: checkout_event(org, 1000, amount_total=None),
    }[case]()
    async with billing_client(m3s_settings(), pool, redis_client) as client:
        r = await deliver(client, ev)
    assert r.status_code == 200 and r.json()["result"] == "unapplied"
    row = await event_row(pool, ev["id"])
    assert row is not None and row["processed_at"] is None and row["result"] == "unapplied"
    assert row["last_error"] is None and row["error_count"] == 0
    assert await purchases(pool, org) == []


# ------------------------------------------------------------------ A17: T4 sequential


async def test_a17_t4_refund_before_purchase_applied_in_its_commit(pool):
    process_event = ep("process_event")
    s = m3s_settings()
    org = await make_org(pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 500, 2000)
    out = await process_event(pool, refund, settings=s)
    assert out.result == "unapplied" and not out.processed
    assert not getattr(out, "stored_after_error", False)  # no error: the 200 path
    row = await event_row(pool, refund["id"])
    assert row is not None and row["processed_at"] is None
    assert row["result"] == "unapplied" and row["last_error"] is None and row["error_count"] == 0

    late, credited = sample(WEBHOOKS, result="late_refunded"), sample(WEBHOOKS, result="credited")
    assert (
        await process_event(pool, checkout_event(org, 2000, payment_intent=pi), settings=s)
    ).result == "credited"
    assert await purchases(pool, org) == [2000 * M]
    assert await refunds(pool, org) == [-500 * M]
    row = await event_row(pool, refund["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "refunded"  # its own outcome (§8.11), not the purchase's
    # The metric keeps M1's `late_<result>` label for it (§8.11), counted once.
    assert sample(WEBHOOKS, result="late_refunded") == late + 1
    assert sample(WEBHOOKS, result="credited") == credited + 1


# ------------------------------------------------------------------ A18: T7


async def test_a18_t7a_replay_ten_minutes_later_is_refused_before_postgres(
    pool, pg_url, redis_client
):
    """Refused on the signature alone: the app's pool is never opened (any use of it
    would fail, and the answer would be 503), so a 400 proves Postgres was not reached."""
    org = await make_org(pool)
    ev = checkout_event(org, 800)
    unopened = AsyncConnectionPool(pg_url, open=False)
    async with billing_client(m3s_settings(), unopened, redis_client) as client:
        before = sample(WEBHOOKS, result="stale_timestamp")
        r = await deliver(client, ev, timestamp=int(time.time()) - 600)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_signature"
        assert sample(WEBHOOKS, result="stale_timestamp") == before + 1
    assert await event_row(pool, ev["id"]) is None
    assert await purchases(pool, org) == []


async def test_a18_t7b_replay_within_tolerance_is_a_duplicate(pool, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org, 800)
    async with billing_client(m3s_settings(), pool, redis_client) as client:
        t = int(time.time())
        r = await deliver(client, ev, timestamp=t)
        assert r.status_code == 200 and r.json()["result"] == "credited"
        r = await deliver(client, ev, timestamp=t)  # the captured request, byte for byte
        assert r.status_code == 200 and r.json()["result"] == "duplicate"
    assert await purchases(pool, org) == [800 * M]
    row = await event_row(pool, ev["id"])
    assert row is not None and row["result"] == "credited"


async def test_a18_t7c_replay_of_a_stored_unapplied_event_reruns_it(pool, redis_client):
    org = await make_org(pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 300, 1000)
    async with billing_client(m3s_settings(), pool, redis_client) as client:
        for _ in range(2):  # the second delivery re-runs the same handling
            r = await deliver(client, refund, secret=STUB_WEBHOOK_SECRET)
            assert r.status_code == 200 and r.json()["result"] == "unapplied"
            row = await event_row(pool, refund["id"])
            assert row is not None and row["processed_at"] is None
            assert row["result"] == "unapplied" and row["error_count"] == 0
        r = await deliver(client, checkout_event(org, 1000, payment_intent=pi))
        assert r.json()["result"] == "credited"
        r = await deliver(client, refund)
        assert r.status_code == 200 and r.json()["result"] == "duplicate"
    assert await refunds(pool, org) == [-300 * M]
