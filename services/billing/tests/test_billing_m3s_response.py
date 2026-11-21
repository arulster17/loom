"""The webhook's response contract and metric results, the credit-refresh bound, the
health route, missed deliveries through the stub, and the checkout request
(docs/phase1-m3-spec.md §8.3, §8.5, §8.8, §8.9, §8.11; done-when A22, A32's settings,
A34, A37's billing half; traces T12, T14, T25)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.parse
from collections.abc import Mapping

import httpx
import pytest
import redis.asyncio as aioredis
from billing_fixtures import STUB, STUB_WEBHOOK_SECRET
from billing_m3s_kit import (
    CREDITED_MICROS,
    HANDLER_FAILURES,
    STALE_TOTAL,
    TOKEN,
    WEBHOOK_ERRORS,
    WEBHOOKS,
    Blackhole,
    billing_client,
    body_of,
    checkout_event,
    closed_port,
    deliver,
    ep,
    error_code,
    event_row,
    gate,
    ledger_count,
    m3s_settings,
    m3s_shared_faults_fixtures,
    make_org,
    poison,
    purchases,
    rid,
    sample,
    ungate,
)
from psycopg_pool import AsyncConnectionPool

from loom_billing import stripe_sig
from loom_billing.api import create_app
from loom_billing.settings import BillingSettings
from loom_billing.stripe_stub import create_stub_app
from loom_platform.ledger import org_balance

pytestmark = [pytest.mark.postgres, pytest.mark.redis]
faults_url, faults_pool = m3s_shared_faults_fixtures()  # tests of their own rows only

M = 10_000
SIG_RESULTS = ("bad_signature", "signature_mismatch", "stale_timestamp")


def sig_counts() -> dict[str, float]:
    return {r: sample(WEBHOOKS, result=r) for r in SIG_RESULTS}


def diff(before: dict[str, float]) -> dict[str, float]:
    return {r: sample(WEBHOOKS, result=r) - v for r, v in before.items()}


# ------------------------------------------------------------------ signatures (unit)


def test_a22_verify_raises_the_new_subclasses():
    stale_cls = ep("StaleTimestamp", "loom_billing.stripe_sig")
    mismatch_cls = ep("SignatureMismatch", "loom_billing.stripe_sig")
    assert issubclass(stale_cls, stripe_sig.SignatureError)
    assert issubclass(mismatch_cls, stripe_sig.SignatureError)
    body, secret, now = b'{"id": "evt_x"}', STUB_WEBHOOK_SECRET, 1_800_000_000
    good = stripe_sig.compute(body, secret, now)
    other = stripe_sig.compute(body, "whsec_other_secret", now)

    assert stripe_sig.verify(body, f"t={now},v1={good}", secret, now=now) == now
    # Rolling secret: any v1 matching is enough.
    assert stripe_sig.verify(body, f"t={now},v1={other},v1={good}", secret, now=now) == now
    # Valid HMAC, t outside the tolerance, either side.
    for t in (now - 301, now + 301, now - 600):
        sig = stripe_sig.compute(body, secret, t)
        with pytest.raises(stale_cls):
            stripe_sig.verify(body, f"t={t},v1={sig}", secret, now=now)
    # Well-formed, current t, no v1 matching (another secret: T25).
    for header in (f"t={now},v1={other}", f"t={now},v1={other},v1=deadbeef"):
        with pytest.raises(mismatch_cls):
            stripe_sig.verify(body, header, secret, now=now)
    # Junk: plain SignatureError, neither subclass.
    for header in (None, "", "garbage", f"t={now}", "v1=abc", "t=x,v1=abc"):
        with pytest.raises(stripe_sig.SignatureError) as caught:
            stripe_sig.verify(body, header, secret, now=now)
        assert not isinstance(caught.value, stale_cls | mismatch_cls), header
    # No v1 matches and t is stale: junk, not a mismatch.
    with pytest.raises(stripe_sig.SignatureError) as caught:
        stripe_sig.verify(body, f"t={now - 3600},v1={other}", secret, now=now)
    assert not isinstance(caught.value, stale_cls | mismatch_cls)


# ------------------------------------------------------------------ signatures (route)


async def test_a22_junk_signatures_count_as_bad_signature(pool, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org)
    raw = body_of(ev)
    other = stripe_sig.sign(raw, "whsec_another_endpoint", int(time.time()) - 3600)
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        before = sig_counts()
        for header in ("", "garbage", "t=1", "v1=abc", other):
            r = await deliver(c, ev, header=header)
            assert r.status_code == 400 and error_code(r) == "bad_signature", header
    assert diff(before) == {"bad_signature": 5, "signature_mismatch": 0, "stale_timestamp": 0}
    assert await event_row(pool, ev["id"]) is None


async def test_a22_t25_wrong_secret_with_current_t_is_a_signature_mismatch(pool, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org)
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        before = sig_counts()
        r = await deliver(c, ev, secret="whsec_not_the_endpoints_secret")
        assert r.status_code == 400 and error_code(r) == "bad_signature"
    assert diff(before) == {"bad_signature": 0, "signature_mismatch": 1, "stale_timestamp": 0}
    assert await event_row(pool, ev["id"]) is None


async def test_a22_t12_valid_hmac_outside_tolerance_is_a_stale_timestamp(pool, redis_client):
    org = await make_org(pool)
    ev = checkout_event(org)
    async with billing_client(m3s_settings(), pool, redis_client) as c:
        before = sig_counts()
        for skew in (-400, 400):
            r = await deliver(c, ev, timestamp=int(time.time()) + skew)
            assert r.status_code == 400 and error_code(r) == "bad_signature"
    assert diff(before) == {"bad_signature": 0, "signature_mismatch": 0, "stale_timestamp": 2}
    assert await event_row(pool, ev["id"]) is None


async def test_a22_bad_events_are_refused_before_any_transaction(pg_url, redis_client):
    """400 `bad_event` is decided before Postgres is touched: with a pool that has no
    connection to give (the role gate), these still answer 400, not 503."""
    unopened = AsyncConnectionPool(pg_url, open=False)
    live = checkout_event("00000000-0000-0000-0000-000000000000")
    live["livemode"] = True
    bodies = [
        body_of(live),
        b"not json",
        json.dumps({"type": "checkout.session.completed"}).encode(),
        json.dumps({"id": rid("evt_test")}).encode(),
        b"[1, 2]",
    ]
    async with billing_client(m3s_settings(), unopened, redis_client) as c:
        for raw in bodies:
            r = await deliver(c, body=raw)
            assert r.status_code == 400 and error_code(r) == "bad_event", raw


# ------------------------------------------------------------------ 500 / COMMIT failures


async def test_a22_handler_failure_answers_500_with_the_event_stored(
    faults_pool, faults_url, redis_client
):
    ep("process_event")
    org = await make_org(faults_pool)
    ev = checkout_event(org, 900)
    await poison(faults_url, ev["id"])
    async with billing_client(m3s_settings(), faults_pool, redis_client) as c:
        errs = sample(WEBHOOK_ERRORS, status="500")
        s4 = sample(HANDLER_FAILURES, path="webhook")
        r = await deliver(c, ev)
    assert r.status_code == 500 and error_code(r) == "internal_error", r.text
    assert sample(WEBHOOK_ERRORS, status="500") == errs + 1
    assert sample(HANDLER_FAILURES, path="webhook") == s4 + 1
    row = await event_row(faults_pool, ev["id"])
    assert row is not None and row["processed_at"] is None and row["result"] == "unapplied"
    assert row["last_error"] and "m3s poison" in row["last_error"]
    assert row["error_count"] == 1
    assert await purchases(faults_pool, org) == []


async def test_a22_failure_at_commit_takes_the_500_path_and_counts_nothing(
    faults_pool, faults_url, redis_client
):
    """A deferred check raising at COMMIT (a scratch-database trigger): rolled back,
    stored with `last_error`, 500; the outcome metrics are counted only after a COMMIT
    that succeeded (A34)."""
    ep("process_event")
    org = await make_org(faults_pool)
    ev = checkout_event(org, 2100)
    key = f"stripe:{ev['id']}"
    await gate(faults_url, "ledger_transactions", key, fail=True, sqlstate="23514")
    async with billing_client(m3s_settings(), faults_pool, redis_client) as c:
        credited, micros = sample(WEBHOOKS, result="credited"), sample(CREDITED_MICROS)
        r = await deliver(c, ev)
        assert r.status_code == 500 and error_code(r) == "internal_error", r.text
        assert sample(WEBHOOKS, result="credited") == credited
        assert sample(CREDITED_MICROS) == micros
        row = await event_row(faults_pool, ev["id"])
        assert row is not None and row["processed_at"] is None and row["error_count"] == 1
        assert row["last_error"] and "m3s deferred check failed" in row["last_error"]
        assert await ledger_count(faults_pool, ev["id"]) == 0

        await ungate(faults_url, "ledger_transactions", key)
        r = await deliver(c, ev)  # Stripe's retry
        assert r.status_code == 200 and r.json()["result"] == "credited"
        assert sample(WEBHOOKS, result="credited") == credited + 1
        assert sample(CREDITED_MICROS) == micros + 2100 * M
    assert await ledger_count(faults_pool, ev["id"]) == 1


# ------------------------------------------------------------------ T14: frozen Redis


def test_a22_refresh_bound_is_the_constant():
    from loom_billing import api

    assert api.REFRESH_WAIT_S == 2.0  # type: ignore[attr-defined]


async def test_a22_t14_frozen_redis_delays_the_answer_by_at_most_the_bound(pool):
    org = await make_org(pool)
    ev = checkout_event(org, 1900)
    hole = Blackhole()
    frozen = aioredis.Redis(host="127.0.0.1", port=hole.port)  # no timeouts: it hangs
    try:
        async with billing_client(m3s_settings(), pool, frozen) as c:
            t0 = time.monotonic()
            r = await deliver(c, ev)
            elapsed = time.monotonic() - t0
        assert r.status_code == 200 and r.json()["result"] == "credited"
        assert elapsed < 3.0, elapsed
        assert await purchases(pool, org) == [1900 * M]
    finally:
        hole.close()  # the refresh left in the background fails and ends
        await asyncio.sleep(0)
        await frozen.aclose()


# ------------------------------------------------------------------ A37: health route


async def test_a37_stripe_health_answers_with_postgres_and_redis_unreachable():
    port = closed_port()
    dead_pg = AsyncConnectionPool(f"postgresql://loom:loom@127.0.0.1:{port}/loom", open=False)
    dead_redis = aioredis.Redis(
        host="127.0.0.1", port=port, socket_timeout=1, socket_connect_timeout=1
    )
    try:
        async with billing_client(m3s_settings(), dead_pg, dead_redis) as c:
            r = await c.get("/stripe/health")
            assert r.status_code == 200
            assert r.json() == {"listener": "stripe_webhook", "status": "ok"}
            assert b"stripe_webhook" in r.content[:5120]
            # The internal health check is unchanged: it does look at Postgres and Redis.
            r = await c.get("/health")
            assert r.status_code == 503 and r.json()["status"] == "degraded"
    finally:
        await dead_redis.aclose()


# ------------------------------------------------------------------ §8.5: missed delivery


async def test_a22_missed_delivery_failed_by_the_stub_then_resent_applies_once(pool, redis_client):
    org = await make_org(pool)
    settings = m3s_settings()
    box: dict[str, object] = {}
    answers: list[tuple[int, dict]] = []
    signed_at: list[int] = []

    async def into_billing(url: str, body: bytes, headers: Mapping[str, str]) -> int:
        t = dict(p.split("=", 1) for p in headers["Stripe-Signature"].split(","))["t"]
        signed_at.append(int(t))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=box["billing"]),  # type: ignore[arg-type]
            base_url="http://billing",
        ) as c:
            r = await c.post("/stripe/webhook", content=body, headers=dict(headers))
        answers.append((r.status_code, r.json()))
        return r.status_code

    stub = create_stub_app(
        public_url="http://stub",
        webhook_url="http://billing/stripe/webhook",
        webhook_secret=STUB_WEBHOOK_SECRET,
        deliver=into_billing,
    )
    from billing_fixtures import asgi_transport

    box["billing"] = create_app(
        settings, pool=pool, redis=redis_client, stripe_transport=asgi_transport(stub)
    )
    auth = {"Authorization": f"Bearer {TOKEN}"}
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=box["billing"]), base_url="http://b"
        ) as web,  # type: ignore[arg-type]
        httpx.AsyncClient(transport=httpx.ASGITransport(app=stub), base_url="http://s") as user,
    ):
        r = await web.post(
            "/internal/checkout-sessions",
            json={"org_id": str(org), "amount_micros": 7_000_000,
                  "success_url": "http://web/ok", "cancel_url": "http://web/no"},
            headers=auth,
        )  # fmt: skip
        assert r.status_code == 200, r.text
        sid = r.json()["id"]
        assert (await user.post("/stub/control", json={"fail_status": 503})).status_code == 200
        r = await user.post(f"/checkout/{sid}/complete", params={"format": "json"})
        assert r.status_code == 502  # the stub reports the failed delivery
        assert answers == []  # billing saw nothing
        events = (await user.get("/stub/events")).json()
        assert [e["type"] for e in events] == ["checkout.session.completed"]
        assert events[0]["last_status"] == 503
        evt_id = events[0]["id"]
        assert await event_row(pool, evt_id) is None

        # The cause fixed, the operator resends it: a fresh signature, applied once.
        await user.post("/stub/control", json={"fail_status": None})
        r = await user.post(f"/stub/events/{evt_id}/resend")
        assert r.status_code == 200 and r.json()["webhook_status"] == 200
        assert answers[-1][0] == 200 and answers[-1][1]["result"] == "credited"
        assert abs(signed_at[-1] - time.time()) < 60
        r = await user.post(f"/stub/events/{evt_id}/resend")
        assert r.json()["webhook_status"] == 200
        assert answers[-1][0] == 200 and answers[-1][1]["result"] == "duplicate"

        # Dropped deliveries never reach billing either.
        await user.post("/stub/control", json={"drop_deliveries": True})
        r = await user.post(f"/stub/events/{evt_id}/resend")
        assert r.json()["webhook_status"] == 0 and len(answers) == 2
    assert await ledger_count(pool, evt_id) == 1
    async with pool.connection() as conn:
        assert await org_balance(conn, org) == 7_000_000
    row = await event_row(pool, evt_id)
    assert row is not None and row["result"] == "credited"


# ------------------------------------------------------------------ A34: checkout request


async def _checkout_form(pool, redis_client, settings: BillingSettings) -> dict[str, str]:
    seen: list[bytes] = []

    async def capture(method, url, headers, body):
        seen.append(body or b"")
        return 200, json.dumps(
            {"id": rid("cs_test"), "url": "https://checkout.example.test/x", "livemode": False}
        ).encode()

    org = await make_org(pool)
    async with billing_client(settings, pool, redis_client, stripe_transport=capture) as c:
        r = await c.post(
            "/internal/checkout-sessions",
            json={"org_id": str(org), "amount_micros": 10_000_000,
                  "success_url": "http://web/ok", "cancel_url": "http://web/no"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )  # fmt: skip
        assert r.status_code == 200, r.text
    assert len(seen) == 1
    return dict(urllib.parse.parse_qsl(seen[0].decode(), keep_blank_values=True))


NAME_FIELD = "line_items[0][price_data][product_data][name]"


async def test_a34_session_expires_in_an_hour_without_adaptive_pricing(pool, redis_client):
    t0 = time.time()
    form = await _checkout_form(pool, redis_client, m3s_settings())
    assert "expires_at" in form, sorted(form)
    assert t0 + 3600 - 60 <= int(form["expires_at"]) <= time.time() + 3600 + 60
    assert form.get("adaptive_pricing[enabled]") == "false"
    assert form[NAME_FIELD] == "API credits"  # LOOM_BRAND_NAME unset


async def test_a34_product_name_comes_from_the_brand_name(pool, redis_client, monkeypatch):
    form = await _checkout_form(pool, redis_client, m3s_settings(brand_name="Acme"))
    assert form[NAME_FIELD] == "Acme credits"
    monkeypatch.setenv("LOOM_BRAND_NAME", "Zephyr")
    s = BillingSettings(**{**STUB, "billing_internal_token": TOKEN})
    assert s.brand_name == "Zephyr"  # type: ignore[attr-defined]
    form = await _checkout_form(pool, redis_client, s)
    assert form[NAME_FIELD] == "Zephyr credits"


async def test_a34_stub_session_carries_the_requested_expiry(pool, redis_client):
    stub = create_stub_app(
        public_url="http://stub", webhook_url="http://x", webhook_secret=STUB_WEBHOOK_SECRET
    )
    from billing_fixtures import asgi_transport

    org = await make_org(pool)
    t0 = int(time.time())
    async with billing_client(
        m3s_settings(), pool, redis_client, stripe_transport=asgi_transport(stub)
    ) as c:
        r = await c.post(
            "/internal/checkout-sessions",
            json={"org_id": str(org), "amount_micros": 5_000_000,
                  "success_url": "http://web/ok", "cancel_url": "http://web/no"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )  # fmt: skip
        assert r.status_code == 200, r.text
    session = stub.state.sessions[r.json()["id"]]
    assert t0 + 3600 - 60 <= session["expires_at"] <= int(time.time()) + 3600 + 60


# ------------------------------------------------------------------ A32: settings


def test_a32_stripe_settings_defaults_and_environment(monkeypatch):
    for var in (
        "LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S",
        "LOOM_STRIPE_LOCK_TIMEOUT_MS",
        "LOOM_STRIPE_STATEMENT_TIMEOUT_MS",
        "LOOM_STRIPE_STALE_AFTER_S",
        "LOOM_BRAND_NAME",
    ):
        monkeypatch.delenv(var, raising=False)
    s = BillingSettings(**STUB)
    assert s.stripe_webhook_db_timeout_s == 5  # type: ignore[attr-defined]
    assert s.stripe_lock_timeout_ms == 3000  # type: ignore[attr-defined]
    assert s.stripe_statement_timeout_ms == 5000  # type: ignore[attr-defined]
    assert s.stripe_stale_after_s == 900  # type: ignore[attr-defined]
    assert s.brand_name is None  # type: ignore[attr-defined]
    monkeypatch.setenv("LOOM_STRIPE_WEBHOOK_DB_TIMEOUT_S", "2")
    monkeypatch.setenv("LOOM_STRIPE_LOCK_TIMEOUT_MS", "700")
    monkeypatch.setenv("LOOM_STRIPE_STATEMENT_TIMEOUT_MS", "900")
    monkeypatch.setenv("LOOM_STRIPE_STALE_AFTER_S", "60")
    s = BillingSettings(**STUB)
    assert s.stripe_webhook_db_timeout_s == 2  # type: ignore[attr-defined]
    assert s.stripe_lock_timeout_ms == 700  # type: ignore[attr-defined]
    assert s.stripe_statement_timeout_ms == 900  # type: ignore[attr-defined]
    assert s.stripe_stale_after_s == 60  # type: ignore[attr-defined]


# ------------------------------------------------------------------ S1a / S4 label init

SUBSCRIBED_TYPES = (
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "checkout.session.async_payment_failed",
    "charge.refunded",
    "charge.dispute.created",
)
INITIALISED = (
    (STALE_TOTAL, "type", (*SUBSCRIBED_TYPES, "other")),  # §8.6
    (HANDLER_FAILURES, "path", ("webhook", "retry", "pending_clawback")),  # §8.11
)
# A fresh interpreter (its own registry, nothing counted yet): import billing's modules and
# create the API app, without touching Postgres or Redis; print every sample of interest
# (null when the series does not exist).
LABELS_SCRIPT = r"""
import json, sys
import redis.asyncio as aioredis
from prometheus_client import REGISTRY
from psycopg_pool import AsyncConnectionPool
from loom_billing.settings import BillingSettings
import loom_billing.webhooks
import loom_billing.worker
from loom_billing.api import create_app

settings = BillingSettings(**json.loads(sys.argv[1]))
create_app(
    settings,
    pool=AsyncConnectionPool("postgresql://loom@127.0.0.1:1/loom", open=False),
    redis=aioredis.Redis(host="127.0.0.1", port=1),
)
out = {}
for name, key, values in json.loads(sys.argv[2]):
    for value in values:
        out[f"{name}{{{key}={value}}}"] = REGISTRY.get_sample_value(name, {key: value})
print(json.dumps(out))
"""


def test_a24_stale_and_handler_failure_labels_exist_at_zero_from_the_start():
    """S1a's `type` and S4's `path` values are all initialised at 0 (§8.6, §8.11), so an
    alarm on an increase sees the first one: each series exists (not merely reads as 0)
    in a fresh process once billing is imported and its app created."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_")}
    settings = {**STUB, "billing_internal_token": TOKEN}
    done = subprocess.run(
        [sys.executable, "-c", LABELS_SCRIPT, json.dumps(settings), json.dumps(INITIALISED)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    samples = json.loads(done.stdout.strip().splitlines()[-1])
    absent = [k for k, v in samples.items() if v is None]
    assert absent == [], f"series missing (absent, not 0): {absent}"
    assert {k: v for k, v in samples.items() if v != 0.0} == {}


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
