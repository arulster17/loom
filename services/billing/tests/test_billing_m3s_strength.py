"""Strength tests for the Stripe area (docs/phase1-m3-spec.md §8.3, §8.6, §8.11; done-when
A22, A31; trace T14): a credit refresh that raises is logged and the answer stays 200 within
the bound; malformed `Stripe-Signature` headers are 400 `bad_signature`, never a 500, and a
malformed `v1` next to a valid one still verifies; `list_events` keys are exactly §8.11's;
`stripe-event resolve` on a row another connection holds exits with the runbook's busy code
and changes nothing. Random ids; every row a test stores is deleted at its end (§16)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import redis.asyncio as aioredis
from billing_fixtures import STUB_WEBHOOK_SECRET
from billing_m3s_kit import (
    WEBHOOKS,
    billing_client,
    body_of,
    checkout_event,
    closed_port,
    deliver,
    ep,
    error_code,
    event_row,
    holder,
    m3s_settings,
    make_admin,
    make_org,
    purchases,
    rid,
    sample,
    scalar,
    store_raw,
)
from psycopg_pool import AsyncConnectionPool
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from loom_billing import stripe_sig

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

M = 10_000
SIG_RESULTS = ("bad_signature", "signature_mismatch", "stale_timestamp")
RUNBOOK = Path(__file__).resolve().parents[3] / "docs" / "runbook-stripe.md"
LIST_KEYS = {
    "id",
    "type",
    "received_at",
    "result",
    "last_error",
    "error_count",
    "stale_reported_at",
}


def sig_counts() -> dict[str, float]:
    return {r: sample(WEBHOOKS, result=r) for r in SIG_RESULTS}


def diff(before: dict[str, float]) -> dict[str, float]:
    return {r: sample(WEBHOOKS, result=r) - v for r, v in before.items()}


async def drop_event(pool: AsyncConnectionPool, evt_id: str) -> None:
    async with pool.connection() as conn:
        await conn.execute("DELETE FROM stripe_events WHERE id = %s", (evt_id,))


# ------------------------------------------------------------------ 1. A22 / T14


async def test_a22_t14_credit_refresh_that_raises_is_logged_and_answer_stays_200(pool, caplog):
    """A refresh whose Redis refuses the connection raises (not a hang): the error is in a
    log record billing emits, and the applied purchase still answers 200 within the bound."""
    org = await make_org(pool)
    ev = checkout_event(org, 1700)
    port = closed_port()
    refusing = aioredis.Redis(
        host="127.0.0.1", port=port, socket_connect_timeout=1, retry=Retry(NoBackoff(), 0)
    )

    def refresh_errors() -> list[logging.LogRecord]:
        return [
            r
            for r in caplog.records
            if r.name.startswith("loom_billing")
            and r.levelno >= logging.WARNING
            and r.exc_info is not None
            and r.exc_info[1] is not None
            and str(port) in logging.Formatter().format(r)
        ]

    caplog.set_level(logging.INFO)
    try:
        async with billing_client(m3s_settings(), pool, refusing) as c:
            t0 = time.monotonic()
            r = await deliver(c, ev)
            elapsed = time.monotonic() - t0
            assert r.status_code == 200 and r.json()["result"] == "credited", r.text
            assert elapsed < 3.0, elapsed
            for _ in range(150):  # up to ~3 s: the refresh may finish in the background
                if refresh_errors():
                    break
                await asyncio.sleep(0.02)
        assert refresh_errors(), [(r.name, r.getMessage()) for r in caplog.records]
        assert await purchases(pool, org) == [1700 * M]
    finally:
        await refusing.aclose()
        await drop_event(pool, ev["id"])


# ------------------------------------------------------------------ 2. A22 malformed headers


async def _post(client, raw: bytes, header: bytes):
    return await client.post(
        "/stripe/webhook",
        content=raw,
        headers={"Content-Type": "application/json", "Stripe-Signature": header},
    )


async def test_a22_malformed_signature_headers_are_bad_signature_not_500(pool, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org, 1100)
    raw = body_of(ev)
    now = int(time.time())
    huge = "9" * 400  # hundreds of digits: no unix time, and no float either
    valid_huge = stripe_sig.compute(raw, STUB_WEBHOOK_SECRET, int(huge))
    other = stripe_sig.compute(raw, "whsec_another_endpoint", now)
    non_ascii = "é" * 64
    headers = [
        f"t={huge},v1={other}".encode(),
        f"t={huge},v1={valid_huge}".encode(),
        f"t=-{huge},v1={other}".encode(),
        f"t={now},v1={non_ascii}".encode(),
    ]
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        before = sig_counts()
        for header in headers:
            r = await _post(c, raw, header)
            assert r.status_code == 400 and error_code(r) == "bad_signature", header
    assert diff(before) == {"bad_signature": 4, "signature_mismatch": 0, "stale_timestamp": 0}
    assert await event_row(pool, ev["id"]) is None


@pytest.mark.parametrize("valid_first", [False, True])
async def test_a22_non_ascii_v1_beside_a_valid_v1_still_verifies(pool, redis_client, valid_first):
    """A rolled secret: one `v1` is junk (non-ASCII), the other is valid: the delivery is
    genuine and applied."""
    org = await make_org(pool)
    ev = checkout_event(org, 1300)
    raw = body_of(ev)
    now = int(time.time())
    good = stripe_sig.compute(raw, STUB_WEBHOOK_SECRET, now)
    bad = "é" * 64
    v1s = (good, bad) if valid_first else (bad, good)
    header = f"t={now},v1={v1s[0]},v1={v1s[1]}".encode()
    try:
        async with billing_client(m3s_settings(), pool, redis_client) as c:
            before = sig_counts()
            r = await _post(c, raw, header)
        assert r.status_code == 200 and r.json()["result"] == "credited", r.text
        assert diff(before) == {"bad_signature": 0, "signature_mismatch": 0, "stale_timestamp": 0}
        assert await purchases(pool, org) == [1300 * M]
    finally:
        await drop_event(pool, ev["id"])


# ------------------------------------------------------------------ 3. A31 list_events keys


async def test_a31_list_events_keys_are_exactly_the_specified_ones(pool):
    list_events = ep("list_events", "loom_billing.resolve")
    ev = checkout_event(await make_org(pool), 900)
    err = f"RuntimeError: {rid('boom')}"
    await store_raw(pool, ev, result="unapplied", last_error=err, error_count=2)
    try:
        for unapplied in (True, False):
            rows = await list_events(pool, unapplied=unapplied)
            assert rows, unapplied
            assert all(isinstance(row, dict) for row in rows)
            wrong = [sorted(row) for row in rows if set(row) != LIST_KEYS]
            assert wrong == [], (unapplied, wrong[:3])
            assert all(len(row) == len(LIST_KEYS) for row in rows)
            mine = [row for row in rows if row["id"] == ev["id"]]
            assert len(mine) == 1
            assert mine[0]["type"] == ev["type"]
            assert mine[0]["result"] == "unapplied"
            assert mine[0]["last_error"] == err
            assert mine[0]["error_count"] == 2
    finally:
        await drop_event(pool, ev["id"])


# ------------------------------------------------------------------ 4. A31 resolve busy


def runbook_busy_exit() -> int:
    text = RUNBOOK.read_text(encoding="utf-8")
    found = re.search(r"busy, retry:.*?exits\s+\*\*(\d+)\*\*", text, re.DOTALL)
    assert found, "docs/runbook-stripe.md names no busy exit code for resolve"
    return int(found.group(1))


async def test_a31_resolve_on_a_row_held_for_update_exits_busy_and_changes_nothing(pool, pg_url):
    busy = runbook_busy_exit()
    ev = checkout_event(await make_org(pool), 800)  # credit event, stuck unapplied
    await store_raw(pool, ev, result="unapplied", last_error="RuntimeError: x", error_count=1)
    _, email = await make_admin(pool)
    reason = rid("why")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LOOM_", "STRIPE_"))}
    env.update(
        LOOM_ENV="dev",
        LOOM_DATABASE_URL=pg_url,
        LOOM_STRIPE_MODE="stub",
        LOOM_STRIPE_STUB_WEBHOOK_SECRET=STUB_WEBHOOK_SECRET,
        LOOM_STRIPE_LOCK_TIMEOUT_MS="300",
        LOOM_STRIPE_STATEMENT_TIMEOUT_MS="10000",
        LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S="10",
    )
    argv = [
        sys.executable,
        "-m",
        "loom_billing",
        "stripe-event",
        "resolve",
        ev["id"],
        "--admin-email",
        email,
        "--reason",
        reason,
    ]
    try:
        before = await event_row(pool, ev["id"])
        assert before is not None and before["processed_at"] is None
        async with holder(pg_url) as h:
            await h.run("SELECT 1 FROM stripe_events WHERE id = %s FOR UPDATE", ev["id"])
            done = await asyncio.to_thread(
                subprocess.run,
                argv,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
                check=False,
            )
            await h.rollback()
        assert done.returncode == busy, (done.returncode, done.stderr[-3000:])
        lines = done.stderr.strip().splitlines()
        assert len(lines) == 1, done.stderr[-3000:]
        assert lines[0].startswith("busy, retry"), lines
        assert await event_row(pool, ev["id"]) == before
        audits = await scalar(
            pool,
            "SELECT count(*) FROM admin_audit_log WHERE target_id = %s OR details->>'reason' = %s",
            ev["id"],
            reason,
        )
        assert audits == 0
    finally:
        await drop_event(pool, ev["id"])
