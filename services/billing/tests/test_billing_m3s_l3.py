"""L3: one purchase per Checkout Session, enforced by migration 0012's unique index when a
race slips past the look-up (docs/phase1-m3-spec.md §8.4, §8.7 T3; done-when A21's
billing half). Runs in the session database on its own rows (random session ids)."""

from __future__ import annotations

import asyncio
import secrets

import pytest
from billing_m3s_kit import (
    HANDLER_FAILURES,
    WEBHOOK_ERRORS,
    billing_client,
    checkout_event,
    deliver,
    ep,
    error_code,
    event_row,
    holder,
    m3s_settings,
    make_org,
    purchases,
    rid,
    sample,
    wait_task_or_blocked,
)

from loom_platform.ledger import STRIPE_CLEARING

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

M = 10_000


async def test_a21_t3_race_past_the_lookup_is_stored_500_then_already_credited(
    pool, pg_url, redis_client
):
    """The test holds an uncommitted purchase of the same session that took no advisory
    lock; the delivery's look-up cannot see it, its insert waits on the index; the test
    commits: unique_violation -> rolled back, stored with `last_error`, 500; Stripe's
    retry finds the purchase: `already_credited`."""
    ep("process_event")
    s = m3s_settings(stripe_lock_timeout_ms=10_000)
    org = await make_org(pool)
    sid = rid("cs_test")
    ev = checkout_event(org, 1500, session_id=sid)
    async with holder(pg_url) as h, billing_client(s, pool, redis_client) as c:
        acct = (await h.run("SELECT loom_org_credit_account(%s)", org))[0][0]
        await h.run(
            "SELECT loom_post_transaction('purchase', %s, %s, %s, 'racing purchase', "
            "%s::uuid[], %s::bigint[])",
            f"stripe:evt_test_{secrets.token_hex(8)}",
            org,
            sid,
            [acct, STRIPE_CLEARING],
            [1500 * M, -1500 * M],
        )
        errs, s4 = sample(WEBHOOK_ERRORS, status="500"), sample(HANDLER_FAILURES, path="webhook")
        task = asyncio.create_task(deliver(c, ev))
        await wait_task_or_blocked(pg_url, h.pid, task)
        await h.commit()
        r = await task
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert sample(WEBHOOK_ERRORS, status="500") == errs + 1
        assert sample(HANDLER_FAILURES, path="webhook") == s4 + 1
        row = await event_row(pool, ev["id"])
        assert row is not None and row["processed_at"] is None and row["error_count"] == 1
        assert row["last_error"] and "IdempotencyConflict" in row["last_error"]

        r = await deliver(c, ev)  # Stripe's retry
        assert r.status_code == 200 and r.json()["result"] == "already_credited"
    assert await purchases(pool, org) == [1500 * M]
    row = await event_row(pool, ev["id"])
    assert row is not None and row["processed_at"] is not None
    assert row["result"] == "already_credited" and row["last_error"] is None


async def test_a21_lookup_finds_another_orgs_purchase_of_the_session(pool):
    """M3 drops the look-up's org filter (it has the index's predicate): a session
    already credited, even to another org, is `already_credited`, not a 500."""
    process_event = ep("process_event")
    s = m3s_settings()
    a, b = await make_org(pool), await make_org(pool)
    sid = rid("cs_test")
    assert (
        await process_event(pool, checkout_event(a, 900, session_id=sid), settings=s)
    ).result == "credited"
    stray = checkout_event(b, 900, session_id=sid, typ="checkout.session.async_payment_succeeded")
    out = await process_event(pool, stray, settings=s)
    assert out.result == "already_credited"
    assert not getattr(out, "stored_after_error", False)
    assert await purchases(pool, b) == []


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
