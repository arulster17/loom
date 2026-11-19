"""Poison events, the second transaction, pending clawbacks in savepoints
(docs/phase1-m3-spec.md §8.3, §8.7 T17, T19, T23; done-when A23, A32, and A24's
"error inside a pending-clawback savepoint").

A poison event is made by a trigger in the test's scratch database (`poison`): every
write of the event's row that leaves `last_error` NULL raises, which is every attempt
that would record an outcome (§8.3); the second transaction, which sets `last_error`,
still stores it. `cure` is the release that fixes the handler."""

from __future__ import annotations

import datetime as dt
import time

import pytest
from billing_fixtures import ledger_is_balanced
from billing_m3s_kit import (
    HANDLER_FAILURES,
    RETRY_LAST_SUCCESS,
    WEBHOOK_ERRORS,
    billing_client,
    charge_event,
    checkout_event,
    cure,
    deliver,
    dispute_event,
    ep,
    error_code,
    event_row,
    fault_proxy,
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
    store_raw,
)
from psycopg_pool import AsyncConnectionPool

pytestmark = [pytest.mark.postgres, pytest.mark.redis]
m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()

M = 10_000
MINUTE = dt.timedelta(minutes=1)


def s4(path: str) -> float:
    return sample(HANDLER_FAILURES, path=path)


# ------------------------------------------------------------------ A23: T17 on the webhook


async def test_a23_t17_poison_is_stored_500_and_counted_in_s4_once(
    scratch_pool, scratch_url, redis_client
):
    ep("process_event")
    await install_faults(scratch_url)
    org = await make_org(scratch_pool)
    ev = checkout_event(org, 1000)
    await poison(scratch_url, ev["id"])
    async with billing_client(m3s_settings(), scratch_pool, redis_client) as c:
        errs, webhook_s4 = sample(WEBHOOK_ERRORS, status="500"), s4("webhook")
        for attempt in (1, 2, 3):  # Stripe's retries fail the same way
            r = await deliver(c, ev)
            assert r.status_code == 500 and error_code(r) == "internal_error", r.text
            row = await event_row(scratch_pool, ev["id"])
            assert row is not None and row["processed_at"] is None
            assert row["result"] == "unapplied" and row["error_count"] == attempt
            assert row["last_error"] and "m3s poison" in row["last_error"]
        assert sample(WEBHOOK_ERRORS, status="500") == errs + 3
        assert s4("webhook") == webhook_s4 + 1  # once: on the write that made it 1
    assert await purchases(scratch_pool, org) == []


async def test_a23_t17_retry_continues_past_poison_and_reruns_it_every_run(
    scratch_pool, scratch_url
):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    # A credited payment and its raced refund (older), then a newer poison event.
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    assert (
        await process_event(
            scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s, now=t0
        )
    ).result == "credited"
    raced = charge_event(pi, 250, 1000)
    await store_raw(scratch_pool, raced, received_at=t0)
    bad = checkout_event(org, 500)
    await poison(scratch_url, bad["id"])
    out = await process_event(scratch_pool, bad, settings=s, now=t0 + dt.timedelta(seconds=10))
    assert out.result == "unapplied" and out.stored_after_error

    retry_s4 = s4("retry")
    t1 = t0 + MINUTE
    assert await retry_unapplied(scratch_pool, settings=s, now=t1) == 1
    assert await refunds(scratch_pool, org) == [-250 * M]  # past the poison
    row = await event_row(scratch_pool, bad["id"])
    assert row is not None and row["error_count"] == 2 and row["processed_at"] is None
    assert s4("retry") == retry_s4  # not its first failure
    last = gauge(RETRY_LAST_SUCCESS)
    assert last is not None  # the run finished

    await retry_unapplied(scratch_pool, settings=s, now=t1 + MINUTE)
    row = await event_row(scratch_pool, bad["id"])
    assert row is not None and row["error_count"] == 3  # re-run each minute, no back-off

    await cure(scratch_url, bad["id"])  # a release that fixes the handler
    assert await retry_unapplied(scratch_pool, settings=s, now=t1 + 2 * MINUTE) == 1
    row = await event_row(scratch_pool, bad["id"])
    assert row is not None and row["processed_at"] is not None and row["result"] == "credited"
    assert row["last_error"] is None and row["error_count"] == 0
    assert sorted(await purchases(scratch_pool, org)) == [500 * M, 1000 * M]


async def test_a23_s4_counted_again_after_a_clean_attempt(scratch_pool, scratch_url):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    ev = charge_event(rid("pi_test"), 100, 1000)  # no payment: unapplied when clean
    await poison(scratch_url, ev["id"])
    webhook_s4, retry_s4 = s4("webhook"), s4("retry")
    out = await process_event(scratch_pool, ev, settings=s, now=t0)
    assert out.stored_after_error and s4("webhook") == webhook_s4 + 1

    await cure(scratch_url, ev["id"])
    await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE)
    row = await event_row(scratch_pool, ev["id"])
    assert row is not None and row["result"] == "unapplied"
    assert row["last_error"] is None and row["error_count"] == 0  # a clean attempt clears

    await poison(scratch_url, ev["id"])
    await retry_unapplied(scratch_pool, settings=s, now=t0 + 2 * MINUTE)
    assert s4("retry") == retry_s4 + 1
    await retry_unapplied(scratch_pool, settings=s, now=t0 + 3 * MINUTE)
    assert s4("retry") == retry_s4 + 1
    row = await event_row(scratch_pool, ev["id"])
    assert row is not None and row["error_count"] == 2


# ------------------------------------------------------------------ A23: 503 paths


async def test_a23_error_store_failing_too_answers_503_and_stores_nothing(
    scratch_pool, scratch_url, redis_client
):
    process_event = ep("process_event")
    store_failed = ep("StoreFailed")
    await install_faults(scratch_url)
    org = await make_org(scratch_pool)
    ev = checkout_event(org, 1000)
    await poison(scratch_url, ev["id"], hard=True)
    async with billing_client(m3s_settings(), scratch_pool, redis_client) as c:
        errs = sample(WEBHOOK_ERRORS, status="503")
        r = await deliver(c, ev)
    assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
    assert sample(WEBHOOK_ERRORS, status="503") == errs + 1
    assert await event_row(scratch_pool, ev["id"]) is None
    with pytest.raises(store_failed) as caught:
        await process_event(scratch_pool, ev, settings=m3s_settings())
    assert caught.value.__cause__ is not None
    assert await event_row(scratch_pool, ev["id"]) is None


@pytest.mark.parametrize("text", ["a\u0000b", "a\ud800b"], ids=["nul", "lone_surrogate"])
async def test_a23_payload_jsonb_refuses_answers_503_and_stores_nothing(pool, redis_client, text):
    """The limitation of §8.10 (T17): `jsonb` refuses `\\u0000` and a lone surrogate, so
    neither transaction can store the event."""
    ep("process_event")
    org = await make_org(pool)
    ev = checkout_event(org, 1000, customer_details={"name": text})
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        errs = sample(WEBHOOK_ERRORS, status="503")
        r = await deliver(c, ev)
    assert r.status_code == 503 and error_code(r) == "service_unavailable", r.text
    assert sample(WEBHOOK_ERRORS, status="503") == errs + 1
    assert await event_row(pool, ev["id"]) is None
    assert await purchases(pool, org) == []


# ------------------------------------------------------------------ A23: T19


async def test_a23_t19_poison_clawback_does_not_block_its_purchase(
    scratch_pool, scratch_url, redis_client
):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    await install_faults(scratch_url)
    s = m3s_settings()
    t0 = now()
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 400, 1000)
    assert (await process_event(scratch_pool, refund, settings=s, now=t0)).result == "unapplied"
    await poison(scratch_url, refund["id"])
    pending_s4 = s4("pending_clawback")
    async with billing_client(s, scratch_pool, redis_client) as c:
        r = await deliver(c, checkout_event(org, 1000, payment_intent=pi))
    assert r.status_code == 200 and r.json()["result"] == "credited", r.text
    assert await purchases(scratch_pool, org) == [1000 * M]
    assert await refunds(scratch_pool, org) == []
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["processed_at"] is None and row["result"] == "unapplied"
    assert row["last_error"] and "m3s poison" in row["last_error"]
    assert row["error_count"] == 1 and row["last_error_at"] is not None
    assert s4("pending_clawback") == pending_s4 + 1
    async with scratch_pool.connection() as conn:
        assert await ledger_is_balanced(conn)

    retry_s4 = s4("retry")
    await retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE)  # still failing
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["error_count"] == 2 and s4("retry") == retry_s4
    await cure(scratch_url, refund["id"])
    await retry_unapplied(scratch_pool, settings=s, now=t0 + 2 * MINUTE)
    assert await refunds(scratch_pool, org) == [-400 * M]
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["result"] == "refunded" and row["error_count"] == 0


async def test_a23_lost_connection_inside_the_savepoint_fails_the_whole_delivery(
    pool, pg_url, redis_client
):
    process_event = ep("process_event")
    s = m3s_settings()
    org = await make_org(pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 300, 1000)
    assert (await process_event(pool, refund, settings=s)).result == "unapplied"
    purchase = checkout_event(org, 1000, payment_intent=pi)
    with fault_proxy(pg_url) as proxy:
        p = AsyncConnectionPool(proxy.url(pg_url), min_size=2, max_size=4, open=False)
        await p.open(wait=True)
        try:
            async with billing_client(s, p, redis_client) as c:
                proxy.arm_kill_after(r"^\s*SAVEPOINT\b")
                r = await deliver(c, purchase)
        finally:
            await p.close()
    assert proxy.fired.is_set(), "no savepoint was opened for the pending clawback"
    assert r.status_code in (500, 503), r.text
    assert await purchases(pool, org) == []
    assert await refunds(pool, org) == []
    row = await event_row(pool, purchase["id"])
    if r.status_code == 500:
        assert row is not None and row["processed_at"] is None and row["error_count"] == 1
    else:
        assert row is None
    async with billing_client(s, pool, redis_client) as c:  # Stripe's retry
        r = await deliver(c, purchase)
        assert r.status_code == 200 and r.json()["result"] == "credited"
    assert await purchases(pool, org) == [1000 * M]
    assert await refunds(pool, org) == [-300 * M]


# ------------------------------------------------------------------ A32: T23


@pytest.mark.parametrize("n", [2, 3])
async def test_a32_t23_pending_clawbacks_all_commit_with_the_purchase(pool, pg_url, n):
    process_event = ep("process_event")
    s = m3s_settings()
    org = await make_org(pool)
    pi = rid("pi_test")
    pending = [charge_event(pi, 300, 1000), dispute_event(pi, 200)]
    if n == 3:
        pending.append(dispute_event(pi, 100))
    for e in pending:
        assert (await process_event(pool, e, settings=s)).result == "unapplied"
    with fault_proxy(pg_url) as proxy:
        p = AsyncConnectionPool(proxy.url(pg_url), min_size=1, max_size=4, open=False)
        await p.open(wait=True)
        try:
            out = await process_event(p, checkout_event(org, 1000, payment_intent=pi), settings=s)
        finally:
            await p.close()
        statements = proxy.statements()
    assert out.result == "credited"
    want = [-300 * M, -200 * M] + ([-100 * M] if n == 3 else [])
    assert sorted(await refunds(pool, org)) == sorted(want)
    for e in pending:
        row = await event_row(pool, e["id"])
        assert row is not None and row["processed_at"] is not None
        assert row["result"] == "refunded"  # each its own outcome
    assert not [q for q in statements if "SET CONSTRAINTS" in q.upper()]
    assert sum(1 for q in statements if q.lstrip().upper().startswith("SAVEPOINT")) >= n
    async with pool.connection() as conn:
        assert await ledger_is_balanced(conn)


async def test_a32_a_raising_savepoint_is_rolled_back_alone(scratch_pool, scratch_url):
    process_event = ep("process_event")
    await install_faults(scratch_url)
    s = m3s_settings()
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    first, middle, last = (
        charge_event(pi, 300, 1000),
        dispute_event(pi, 200),
        dispute_event(pi, 100),
    )
    t0 = now()
    for i, e in enumerate((first, middle, last)):
        out = await process_event(scratch_pool, e, settings=s, now=t0 + i * dt.timedelta(seconds=1))
        assert out.result == "unapplied"
    await poison(scratch_url, middle["id"])
    out = await process_event(
        scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s, now=t0 + MINUTE
    )
    assert out.result == "credited"
    assert sorted(await refunds(scratch_pool, org)) == [-300 * M, -100 * M]
    for e, want in ((first, "refunded"), (last, "refunded")):
        row = await event_row(scratch_pool, e["id"])
        assert row is not None and row["processed_at"] is not None and row["result"] == want
    row = await event_row(scratch_pool, middle["id"])
    assert row is not None and row["processed_at"] is None and row["error_count"] == 1
    async with scratch_pool.connection() as conn:
        assert await ledger_is_balanced(conn)


# ------------------------------------------------------------------ A24: usable-transaction errors


@pytest.mark.parametrize(
    ("sqlstate", "sleep_s"),
    [("P0001", None), ("23514", None), ("55P03", None), ("P0001", 30.0)],
    ids=["raise", "check_violation", "lock_not_available", "statement_timeout"],
)
async def test_a24_error_in_a_pending_savepoint_leaves_it_for_r_and_the_purchase_commits(
    scratch_pool, scratch_url, sqlstate, sleep_s
):
    process_event = ep("process_event")
    await install_faults(scratch_url)
    s = m3s_settings(stripe_statement_timeout_ms=1000)
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    refund = charge_event(pi, 300, 1000)
    assert (await process_event(scratch_pool, refund, settings=s)).result == "unapplied"
    await poison(scratch_url, refund["id"], sqlstate=sqlstate, sleep_s=sleep_s)
    t0 = time.monotonic()
    out = await process_event(
        scratch_pool, checkout_event(org, 1000, payment_intent=pi), settings=s
    )
    assert time.monotonic() - t0 < 15  # a statement timeout, not the 30 s sleep
    assert out.result == "credited"
    assert await purchases(scratch_pool, org) == [1000 * M]
    row = await event_row(scratch_pool, refund["id"])
    assert row is not None and row["processed_at"] is None and row["error_count"] == 1
    assert row["last_error"]


async def test_a32_statement_timeout_applies_inside_the_delivery(
    scratch_pool, scratch_url, redis_client
):
    ep("process_event")
    await install_faults(scratch_url)
    s = m3s_settings(stripe_statement_timeout_ms=1000)
    org = await make_org(scratch_pool)
    ev = checkout_event(org, 1000)
    await poison(scratch_url, ev["id"], sleep_s=30.0)
    async with billing_client(s, scratch_pool, redis_client) as c:
        t0 = time.monotonic()
        r = await deliver(c, ev)
        elapsed = time.monotonic() - t0
    assert r.status_code == 500 and error_code(r) == "internal_error", r.text
    assert elapsed < 15, elapsed
    row = await event_row(scratch_pool, ev["id"])
    assert row is not None and row["error_count"] == 1 and row["last_error"]
