"""`retry_unapplied` (R) re-running a purchase whose pending clawback's event row a
redelivery of that clawback holds (docs/phase1-m3-spec.md §8.3 "Pending clawbacks in
savepoints" and "`retry_unapplied` (R)", trace T10; done-when A33).

A33: every wait inside the handler other than on the row R re-runs is bounded by, and
ends at, `lock_timeout`, including "a pending clawback's event row that a redelivery of
that clawback holds while R re-runs its purchase (§8.3's blocking `FOR UPDATE` on
pending clawbacks)". So R neither skips such a row (`SKIP LOCKED`) nor gives up on it at
once (`NOWAIT`): it waits on it, for `lock_timeout`.

Reading taken: §8.3 says that "a wait on the pending rows past `lock_timeout` does the
same" as a failed error update: the delivery fails as a whole. For R that is an attempt
that raises: the purchase's transaction is rolled back (no credit, the clawback row
untouched by it), the purchase is stored with its error by the second transaction (S4
`retry` on its first failure), and the run goes on with the next event and finishes
(S5 gauge set). R's own re-run of the held clawback takes the row `SKIP LOCKED` and skips
it quietly (T10). Once the holder lets go, the next run applies the purchase and, in its
transaction, the clawback: once each.

The test has a scratch database of its own (§16): R re-runs every unapplied row."""

from __future__ import annotations

import asyncio
import datetime as dt
import re
import time

import pytest
from billing_m3s_kit import (
    HANDLER_FAILURES,
    RETRY_LAST_SUCCESS,
    charge_event,
    checkout_event,
    ep,
    event_row,
    gauge,
    holder,
    ledger_count,
    m3s_scratch_fixtures,
    m3s_settings,
    make_org,
    now,
    purchases,
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
# `<exception class>: <message>` (§8.3).
LAST_ERROR = re.compile(r"^[A-Za-z_][\w.]*: ", re.DOTALL)
# Long enough for the barrier to see the wait under flake-check's CPU load; the statement
# timeout is far above it, so a wait that ignored `lock_timeout` would show in the time.
LOCK_TIMEOUT_MS = 3000
STATEMENT_TIMEOUT_MS = 20_000


async def test_a33_retry_waits_on_a_held_pending_clawback_then_records_the_purchase(
    scratch_pool, scratch_url
):
    process_event = ep("process_event")
    retry_unapplied = ep("retry_unapplied")
    s = m3s_settings(
        stripe_lock_timeout_ms=LOCK_TIMEOUT_MS, stripe_statement_timeout_ms=STATEMENT_TIMEOUT_MS
    )
    lock_timeout_s = LOCK_TIMEOUT_MS / 1000
    t0 = now()

    # The purchase was stored unapplied; a refund of its PaymentIntent arrived before it.
    org = await make_org(scratch_pool)
    pi = rid("pi_test")
    purchase = checkout_event(org, 1000, payment_intent=pi)
    clawback = charge_event(pi, 300, 1000)
    await store_raw(scratch_pool, clawback, received_at=t0 - dt.timedelta(seconds=2))
    await store_raw(scratch_pool, purchase, received_at=t0)  # newest: re-run first

    # An unrelated unapplied event, re-run after the purchase: shows the run went on.
    other_org = await make_org(scratch_pool)
    other_pi = rid("pi_test")
    bought = await process_event(
        scratch_pool,
        checkout_event(other_org, 1000, payment_intent=other_pi),
        settings=s,
        now=t0 - dt.timedelta(seconds=10),
    )
    assert bought.result == "credited"
    other = charge_event(other_pi, 400, 1000)
    await store_raw(scratch_pool, other, received_at=t0 - dt.timedelta(seconds=1))

    async with holder(scratch_url) as h:  # a redelivery of the clawback, in flight
        await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", clawback["id"])
        retry_s4, before = sample(HANDLER_FAILURES, path="retry"), gauge(RETRY_LAST_SUCCESS)
        started = time.monotonic()
        run = asyncio.create_task(retry_unapplied(scratch_pool, settings=s, now=t0 + MINUTE))
        # The purchase's attempt waits on the held row (a SKIP LOCKED or NOWAIT query
        # finishes without ever waiting, and this barrier fails).
        await wait_task_or_blocked(scratch_url, h.pid, run, timeout_s=lock_timeout_s + 15)
        n = await run
        elapsed = time.monotonic() - started

        # It waited for `lock_timeout`, and that wait ended the attempt there (not the
        # statement timeout, not a second wait).
        assert elapsed >= lock_timeout_s * 0.9, elapsed
        assert elapsed < lock_timeout_s + 8, elapsed

        # The purchase's attempt raised: rolled back and stored with its error, S4 once.
        assert await purchases(scratch_pool, org) == []
        assert await refunds(scratch_pool, org) == []
        row = await event_row(scratch_pool, purchase["id"])
        assert row is not None and row["processed_at"] is None
        assert row["result"] == "unapplied"
        assert row["last_error"] and LAST_ERROR.match(row["last_error"]), row["last_error"]
        assert row["last_error_at"] is not None and row["error_count"] == 1
        assert sample(HANDLER_FAILURES, path="retry") == retry_s4 + 1

        # The held clawback: untouched by the rolled-back purchase, skipped quietly by
        # its own re-run (T10).
        row = await event_row(scratch_pool, clawback["id"])
        assert row is not None and row["processed_at"] is None
        assert row["last_error"] is None and row["error_count"] == 0

        # The run went on with the next event and finished.
        assert n == 1
        assert await refunds(scratch_pool, other_org) == [-400 * M]
        assert gauge(RETRY_LAST_SUCCESS) != before

        await h.rollback()

    # The redelivery let go: the next run applies the purchase and its clawback, once each.
    await retry_unapplied(scratch_pool, settings=s, now=t0 + 2 * MINUTE)
    assert await purchases(scratch_pool, org) == [1000 * M]
    assert await refunds(scratch_pool, org) == [-300 * M]
    assert await ledger_count(scratch_pool, purchase["id"]) == 1
    assert await ledger_count(scratch_pool, clawback["id"]) == 1
    for ev in (purchase, clawback):
        row = await event_row(scratch_pool, ev["id"])
        assert row is not None and row["processed_at"] is not None, ev["type"]
    assert await refunds(scratch_pool, other_org) == [-400 * M]  # not applied again


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
