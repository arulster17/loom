"""Lock waits inside the handler that are not on the event's own row: a `stripe_payments`
row and the Checkout Session's advisory lock, met by a webhook delivery and by
`retry_unapplied` (R) (docs/phase1-m3-spec.md §8.3 "Second transaction" and "Timeouts";
done-when A19, A33).

A33: R "never waits on the event row it re-runs ... and every other wait inside the
handler ends at `lock_timeout`: a session or `stripe_payments` lock". A19: "A lock timeout
on a lock other than the event's row (the session's advisory lock, a `stripe_payments`
row ...) -> 500 with the event stored with `last_error`". The visible tests of those items
check the status and the stored error only; these check that the code really waited on
the held lock (a barrier sees the waiter blocked by the holder's backend) and that the
wait ended at `lock_timeout`: a query that refuses to wait (`NOWAIT`, a `pg_try_` advisory
lock that fails on false) finishes before the barrier and fails here.

Readings taken:
- "Ends at `lock_timeout`": the attempt takes at least about `lock_timeout` (0.9 x, for
  clock granularity) and well under the statement timeout, which is set far above it, so
  a wait bounded by the statement timeout instead would show in the time.
- R: the attempt that hit the timeout raises, so it is rolled back and stored with its
  error by the second transaction (§8.3: "`retry_unapplied` uses the same statement for
  its own failures"; S4 `retry` on its first failure); the run goes on with the next
  event and finishes (S5 gauge set), as for any other failed attempt.
- "The session's lock" is the advisory lock on the Checkout Session id (§8.1: "an
  advisory lock on the session id plus a look-up"). Its holder is not built from a guessed
  key: it is a purchase of the same session in flight, held (by the test) on a pending
  clawback's event row after it has taken the session's lock (§8.3's blocking
  `FOR UPDATE` on pending clawbacks). The waiter must be blocked by that purchase's
  backend on a lock of type `advisory`.

Each test has a scratch database of its own (§16): R re-runs every unapplied row."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import psycopg
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
    gauge,
    holder,
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
# `<exception class>: <message>` (§8.3).
LAST_ERROR = re.compile(r"^[A-Za-z_][\w.]*: ", re.DOTALL)
# Long enough for the barrier to see the wait under flake-check's CPU load; the statement
# timeout is far above it, so a wait that ignored `lock_timeout` would show in the time.
LOCK_TIMEOUT_MS = 2500
STATEMENT_TIMEOUT_MS = 20_000
LOCK_TIMEOUT_S = LOCK_TIMEOUT_MS / 1000
# The in-flight purchase that holds the session's lock waits far longer than the waiter.
HOLDER_LOCK_TIMEOUT_MS = 60_000
HOLDER_STATEMENT_TIMEOUT_MS = 90_000


def short_settings():
    return m3s_settings(
        stripe_lock_timeout_ms=LOCK_TIMEOUT_MS, stripe_statement_timeout_ms=STATEMENT_TIMEOUT_MS
    )


def assert_waited_for_lock_timeout(elapsed: float) -> None:
    assert elapsed >= LOCK_TIMEOUT_S * 0.9, elapsed
    assert elapsed < LOCK_TIMEOUT_S + 8, elapsed


def assert_stored_error(row: dict | None) -> None:
    assert row is not None, "the second transaction stored nothing"
    assert row["processed_at"] is None and row["result"] == "unapplied"
    assert row["last_error"] and LAST_ERROR.match(row["last_error"]), row["last_error"]
    assert row["last_error_at"] is not None and row["error_count"] == 1


async def credited_payment(pool, cents: int = 1000) -> tuple[uuid.UUID, str]:
    process_event = ep("process_event")
    org = await make_org(pool)
    pi = rid("pi_test")
    out = await process_event(
        pool, checkout_event(org, cents, payment_intent=pi), settings=short_settings()
    )
    assert out.result == "credited"
    return org, pi


@contextlib.asynccontextmanager
async def watcher(url: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    """The barrier's own connection, opened before the task it watches is created, so its
    first read follows the task's start as closely as possible."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        yield conn


async def wait_advisory_wait_blocked_by(
    conn: psycopg.AsyncConnection[Any],
    blocker: int,
    task: asyncio.Task[Any],
    *,
    timeout_s: float,
) -> int:
    """Poll (one query each time, on `conn`) until some backend is seen waiting on an
    `advisory` lock with `blocker` among its blocking pids; returns that backend's pid on
    the first sighting. The lock type and the blocker are read in the same query, so a
    wait on another lock of `blocker`, or on an advisory lock someone else holds, is not
    taken for it. Fails at once if `task` finishes first (with its own error, or its
    result), and after `timeout_s` (cancelling `task`)."""
    deadline = time.monotonic() + timeout_s
    seen: list[str] = []
    while True:
        if task.done():
            result = task.result()  # re-raises the task's error
            raise AssertionError(
                f"finished before a backend was seen waiting on an advisory lock blocked by "
                f"{blocker} (lock types waited on behind it: {seen}): {result!r}"
            )
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT pid, locktype FROM pg_locks "
                "WHERE NOT granted AND %s = ANY(pg_blocking_pids(pid))",
                (blocker,),
            )
            rows = await cur.fetchall()
        for pid, locktype in rows:
            if locktype == "advisory":
                return int(pid)
        seen += [str(r[1]) for r in rows if str(r[1]) not in seen]
        if time.monotonic() > deadline:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            raise AssertionError(
                f"no backend seen waiting on an advisory lock blocked by {blocker} within "
                f"{timeout_s} s (lock types waited on behind it: {seen})"
            )
        await asyncio.sleep(0.01)


@contextlib.asynccontextmanager
async def session_lock_held(
    pool, url: str, org: uuid.UUID, sid: str, pi: str
) -> AsyncIterator[int]:
    """A purchase of session `sid` in flight, holding the session's lock: it waits (with a
    long `lock_timeout`) on a pending clawback's event row the test holds. Yields that
    purchase's backend pid; on exit the clawback's row is released and the purchase
    commits (credited, once)."""
    process_event = ep("process_event")
    long = m3s_settings(
        stripe_lock_timeout_ms=HOLDER_LOCK_TIMEOUT_MS,
        stripe_statement_timeout_ms=HOLDER_STATEMENT_TIMEOUT_MS,
    )
    early = charge_event(pi, 100, 1000)
    assert (await process_event(pool, early, settings=long)).result == "unapplied"
    async with holder(url) as h:
        await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", early["id"])
        first = asyncio.create_task(
            process_event(
                pool, checkout_event(org, 1000, session_id=sid, payment_intent=pi), settings=long
            )
        )
        try:
            yield await wait_task_or_blocked(url, h.pid, first, timeout_s=30)
        finally:
            await h.rollback()
            out = await asyncio.wait_for(first, 60)
    assert out.result == "credited"


# ------------------------------------------------------------------ a `stripe_payments` row


async def test_a33_retry_blocks_on_a_held_stripe_payments_row_for_lock_timeout(
    scratch_pool, scratch_url
):
    """R re-runs a refund whose payment's row another transaction keeps `FOR UPDATE`: it
    waits on that row for `lock_timeout`, the attempt is stored with its error, and the
    run goes on with the next event."""
    retry_unapplied = ep("retry_unapplied")
    s = short_settings()
    org, pi = await credited_payment(scratch_pool)
    other_org, other_pi = await credited_payment(scratch_pool)
    t0 = now()
    raced = charge_event(pi, 300, 1000)
    await store_raw(scratch_pool, raced, received_at=t0)  # newest: re-run first
    other = charge_event(other_pi, 400, 1000)
    await store_raw(scratch_pool, other, received_at=t0 - dt.timedelta(seconds=1))

    async with holder(scratch_url) as h:
        await h.run("SELECT 1 FROM stripe_payments WHERE payment_intent = %s FOR UPDATE", pi)
        retry_s4, before = sample(HANDLER_FAILURES, path="retry"), gauge(RETRY_LAST_SUCCESS)
        started = time.monotonic()
        run = asyncio.create_task(retry_unapplied(scratch_pool, settings=s))
        await wait_task_or_blocked(scratch_url, h.pid, run, timeout_s=LOCK_TIMEOUT_S + 15)
        n = await run
        elapsed = time.monotonic() - started
        assert_waited_for_lock_timeout(elapsed)

        assert_stored_error(await event_row(scratch_pool, raced["id"]))
        assert await refunds(scratch_pool, org) == []
        assert sample(HANDLER_FAILURES, path="retry") == retry_s4 + 1
        # The run went on with the next event and finished.
        assert n == 1
        assert await refunds(scratch_pool, other_org) == [-400 * M]
        assert gauge(RETRY_LAST_SUCCESS) != before
        await h.rollback()

    assert await retry_unapplied(scratch_pool, settings=s) == 1
    assert await refunds(scratch_pool, org) == [-300 * M]


@pytest.mark.redis
async def test_a19_delivery_blocks_on_a_held_stripe_payments_row_then_500_stored(
    scratch_pool, scratch_url, redis_client
):
    """A delivered refund whose payment's row another transaction keeps `FOR UPDATE`
    waits on it for `lock_timeout`, then answers 500 with the event stored with its
    error; R applies it once the row is free."""
    retry_unapplied = ep("retry_unapplied")
    s = short_settings()
    org, pi = await credited_payment(scratch_pool)
    refund = charge_event(pi, 400, 1000)
    async with holder(scratch_url) as h, billing_client(s, scratch_pool, redis_client) as c:
        await h.run("SELECT 1 FROM stripe_payments WHERE payment_intent = %s FOR UPDATE", pi)
        errs = sample(WEBHOOK_ERRORS, status="500")
        started = time.monotonic()
        task = asyncio.create_task(deliver(c, refund))
        await wait_task_or_blocked(scratch_url, h.pid, task, timeout_s=LOCK_TIMEOUT_S + 15)
        r = await task
        elapsed = time.monotonic() - started
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert_waited_for_lock_timeout(elapsed)
        assert sample(WEBHOOK_ERRORS, status="500") == errs + 1
        assert_stored_error(await event_row(scratch_pool, refund["id"]))
        assert await refunds(scratch_pool, org) == []
        await h.rollback()

    assert await retry_unapplied(scratch_pool, settings=s) == 1
    assert await refunds(scratch_pool, org) == [-400 * M]


# ------------------------------------------------------------------ the session's lock


@pytest.mark.redis
async def test_a19_delivery_blocks_on_the_sessions_advisory_lock_then_500_stored(
    scratch_pool, scratch_url, redis_client
):
    """A second credit event of a Checkout Session, delivered while a purchase of that
    session holds the session's advisory lock: it waits on that lock for `lock_timeout`,
    then answers 500 with the event stored with its error; R later answers
    `already_credited` (one purchase)."""
    retry_unapplied = ep("retry_unapplied")
    s = short_settings()
    org = await make_org(scratch_pool)
    pi, sid = rid("pi_test"), rid("cs_test")
    second = checkout_event(
        org, 1000, session_id=sid, payment_intent=pi, typ="checkout.session.async_payment_succeeded"
    )
    async with (
        session_lock_held(scratch_pool, scratch_url, org, sid, pi) as purchase_pid,
        billing_client(s, scratch_pool, redis_client) as c,
        watcher(scratch_url) as w,
    ):
        errs = sample(WEBHOOK_ERRORS, status="500")
        started = time.monotonic()
        task = asyncio.create_task(deliver(c, second))
        await wait_advisory_wait_blocked_by(w, purchase_pid, task, timeout_s=LOCK_TIMEOUT_S + 15)
        r = await task
        elapsed = time.monotonic() - started
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert_waited_for_lock_timeout(elapsed)
        assert sample(WEBHOOK_ERRORS, status="500") == errs + 1
        assert_stored_error(await event_row(scratch_pool, second["id"]))

    assert await purchases(scratch_pool, org) == [1000 * M]
    await retry_unapplied(scratch_pool, settings=s)
    row = await event_row(scratch_pool, second["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "already_credited"
    assert await purchases(scratch_pool, org) == [1000 * M]


async def test_a33_retry_blocks_on_the_sessions_advisory_lock_for_lock_timeout(
    scratch_pool, scratch_url
):
    """R re-runs a credit event of a Checkout Session while a purchase of that session
    holds the session's advisory lock: it waits on that lock for `lock_timeout`, the
    attempt is stored with its error, and the run goes on with the next event."""
    retry_unapplied = ep("retry_unapplied")
    s = short_settings()
    org = await make_org(scratch_pool)
    pi, sid = rid("pi_test"), rid("cs_test")
    other_org, other_pi = await credited_payment(scratch_pool)
    second = checkout_event(
        org, 1000, session_id=sid, payment_intent=pi, typ="checkout.session.async_payment_succeeded"
    )
    async with (
        session_lock_held(scratch_pool, scratch_url, org, sid, pi) as purchase_pid,
        watcher(scratch_url) as w,
    ):
        t0 = now()
        await store_raw(scratch_pool, second, received_at=t0)  # newest: re-run first
        other = charge_event(other_pi, 400, 1000)
        await store_raw(scratch_pool, other, received_at=t0 - dt.timedelta(seconds=1))
        retry_s4, before = sample(HANDLER_FAILURES, path="retry"), gauge(RETRY_LAST_SUCCESS)
        started = time.monotonic()
        run = asyncio.create_task(retry_unapplied(scratch_pool, settings=s))
        await wait_advisory_wait_blocked_by(w, purchase_pid, run, timeout_s=LOCK_TIMEOUT_S + 15)
        n = await run
        elapsed = time.monotonic() - started
        assert_waited_for_lock_timeout(elapsed)

        assert_stored_error(await event_row(scratch_pool, second["id"]))
        assert sample(HANDLER_FAILURES, path="retry") == retry_s4 + 1
        # The run went on with the next event and finished.
        assert n == 1
        assert await refunds(scratch_pool, other_org) == [-400 * M]
        assert gauge(RETRY_LAST_SUCCESS) != before

    assert await purchases(scratch_pool, org) == [1000 * M]
    await retry_unapplied(scratch_pool, settings=s)
    row = await event_row(scratch_pool, second["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "already_credited"
    assert await purchases(scratch_pool, org) == [1000 * M]
