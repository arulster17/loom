"""Stripe webhook endpoint: signatures, idempotent purchase credit, and the internal
checkout endpoint's auth and validation."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid

import httpx
import pytest
from billing_fixtures import STUB, STUB_WEBHOOK_SECRET, ledger_is_balanced

from loom_billing.api import create_app
from loom_billing.settings import BillingSettings
from loom_billing.stripe_sig import sign
from loom_platform.ledger import (
    REVENUE,
    STRIPE_CLEARING,
    Entry,
    org_balance,
    org_credit_account,
    post_transaction,
)
from loom_platform.limiter import Limiter, org_key

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TOKEN = "internal-test-token"


def settings(**kw) -> BillingSettings:
    kw.setdefault("billing_internal_token", TOKEN)
    if "stripe_secret_key" not in kw:
        kw = {**STUB, **kw}
    return BillingSettings(**kw)


@pytest.fixture
async def client(pool, redis_client):
    app = create_app(settings(), pool=pool, redis=redis_client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://billing"
    ) as c:
        yield c


def checkout_event(
    org: uuid.UUID | str,
    cents: int = 2000,
    *,
    evt_id: str | None = None,
    session_id: str | None = None,
    typ: str = "checkout.session.completed",
    paid: bool = True,
    **session_overrides,
) -> dict:
    session = {
        "id": session_id or f"cs_test_{secrets.token_hex(8)}",
        "object": "checkout.session",
        "amount_total": cents,
        "currency": "usd",
        "client_reference_id": str(org),
        "metadata": {"org_id": str(org), "credit_micros": str(cents * 10_000),
                     "purpose": "loom_credits"},
        "mode": "payment",
        "payment_status": "paid" if paid else "unpaid",
        "status": "complete",
        "livemode": False,
    }  # fmt: skip
    session.update(session_overrides)
    return {
        "id": evt_id or f"evt_test_{secrets.token_hex(8)}",
        "object": "event",
        "type": typ,
        "livemode": False,
        "created": int(time.time()),
        "data": {"object": session},
    }


async def deliver(
    client: httpx.AsyncClient,
    event: dict,
    *,
    secret: str = STUB_WEBHOOK_SECRET,
    timestamp: int | None = None,
    header: str | None = None,
) -> httpx.Response:
    body = json.dumps(event).encode()
    sig = header if header is not None else sign(body, secret, timestamp)
    return await client.post(
        "/stripe/webhook",
        content=body,
        headers={"Content-Type": "application/json", "Stripe-Signature": sig},
    )


async def stripe_row(pool, evt_id):
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT type, processed_at IS NOT NULL FROM stripe_events WHERE id = %s", (evt_id,)
        )
        return await cur.fetchone()


async def purchases(pool, org) -> list[tuple[str, int]]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT t.idempotency_key, e.amount_micros FROM ledger_transactions t "
            "JOIN ledger_entries e ON e.transaction_id = t.id "
            "WHERE t.org_id = %s AND t.kind = 'purchase' AND e.account_id <> %s",
            (org, STRIPE_CLEARING),
        )
        return await cur.fetchall()


async def test_bad_signatures_are_rejected_and_not_stored(client, pool, make_tenant):
    t = await make_tenant()
    ev = checkout_event(t.org_id)
    for kw in (
        {"header": ""},
        {"secret": "whsec_wrong"},
        {"timestamp": int(time.time()) - 3600},
        {"header": "t=1,v1=deadbeef"},
    ):
        r = await deliver(client, ev, **kw)
        assert r.status_code == 400, kw
        assert r.json()["error"]["code"] == "bad_signature"
    assert await stripe_row(pool, ev["id"]) is None
    assert await purchases(pool, t.org_id) == []
    # Body tampered after signing.
    body = json.dumps(ev).encode()
    header = sign(body, STUB_WEBHOOK_SECRET)
    forged = body.replace(b'"amount_total": 2000', b'"amount_total": 9000')
    r = await client.post("/stripe/webhook", content=forged, headers={"Stripe-Signature": header})
    assert r.status_code == 400


async def test_completed_checkout_credits_once_across_replays(
    client, pool, redis_client, make_tenant
):
    t = await make_tenant()
    limiter = Limiter(redis_client)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    ev = checkout_event(t.org_id, cents=2500)

    r = await deliver(client, ev)
    assert r.status_code == 200 and r.json()["result"] == "credited"
    for _ in range(3):  # Stripe retries / an attacker replays a captured request
        r = await deliver(client, ev)
        assert r.status_code == 200 and r.json()["result"] == "duplicate"

    assert await purchases(pool, t.org_id) == [(f"stripe:{ev['id']}", 25_000_000)]
    assert await stripe_row(pool, ev["id"]) == ("checkout.session.completed", True)
    # Redis credit updated at once for a loaded org.
    assert int(await redis_client.get(org_key(t.org_id, "credit"))) == 25_000_000
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == 25_000_000
        assert await ledger_is_balanced(conn)


async def test_concurrent_deliveries_credit_once(client, pool, make_tenant):
    t = await make_tenant()
    ev = checkout_event(t.org_id, cents=1000)
    rs = await asyncio.gather(*(deliver(client, ev) for _ in range(8)))
    assert all(r.status_code == 200 for r in rs)
    assert sorted(r.json()["result"] for r in rs) == ["credited"] + ["duplicate"] * 7
    assert await purchases(pool, t.org_id) == [(f"stripe:{ev['id']}", 10_000_000)]


async def test_a_session_is_credited_once_across_event_ids(client, pool, make_tenant):
    t = await make_tenant()
    sid = f"cs_test_{secrets.token_hex(8)}"
    unpaid = checkout_event(t.org_id, session_id=sid, paid=False)
    r = await deliver(client, unpaid)
    assert r.json()["result"] == "not_paid"
    assert await purchases(pool, t.org_id) == []

    paid = checkout_event(t.org_id, session_id=sid, typ="checkout.session.async_payment_succeeded")
    assert (await deliver(client, paid)).json()["result"] == "credited"
    other = checkout_event(t.org_id, session_id=sid)  # a stray second event, same session
    assert (await deliver(client, other)).json()["result"] == "already_credited"
    assert await purchases(pool, t.org_id) == [(f"stripe:{paid['id']}", 20_000_000)]


async def test_credit_goes_to_the_sessions_org_only(client, pool, make_tenant):
    a = await make_tenant()
    b = await make_tenant()
    await deliver(client, checkout_event(a.org_id, cents=700))
    assert await purchases(pool, a.org_id) == [(await _key_of(pool, a.org_id), 7_000_000)]
    assert await purchases(pool, b.org_id) == []
    # client_reference_id and metadata disagreeing is refused, not guessed.
    ev = checkout_event(a.org_id, client_reference_id=str(b.org_id))
    assert (await deliver(client, ev)).json()["result"] == "unapplied"
    assert await purchases(pool, b.org_id) == []


async def _key_of(pool, org):
    return (await purchases(pool, org))[0][0]


async def test_unappliable_and_ignored_events(client, pool, make_tenant):
    t = await make_tenant()
    ghost = checkout_event(uuid.uuid4())
    r = await deliver(client, ghost)
    assert r.status_code == 200 and r.json()["result"] == "unapplied"
    assert await stripe_row(pool, ghost["id"]) == ("checkout.session.completed", False)

    eur = checkout_event(t.org_id, currency="eur")
    assert (await deliver(client, eur)).json()["result"] == "unapplied"

    foreign = checkout_event(t.org_id, metadata={"purpose": "something_else"})
    assert (await deliver(client, foreign)).json()["result"] == "ignored"

    other = {"id": f"evt_test_{secrets.token_hex(6)}", "type": "customer.created",
             "livemode": False, "data": {"object": {}}}  # fmt: skip
    assert (await deliver(client, other)).json()["result"] == "ignored"
    assert await stripe_row(pool, other["id"]) == ("customer.created", True)
    assert await purchases(pool, t.org_id) == []


async def test_live_mode_and_garbage_refused(client, pool, make_tenant):
    t = await make_tenant()
    live = checkout_event(t.org_id)
    live["livemode"] = True
    r = await deliver(client, live)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_event"
    assert await stripe_row(pool, live["id"]) is None
    body = b"not json"
    r = await client.post(
        "/stripe/webhook",
        content=body,
        headers={"Stripe-Signature": sign(body, STUB_WEBHOOK_SECRET)},
    )
    assert r.status_code == 400


async def test_real_mode_verifies_with_the_stripe_secret(pool, redis_client, make_tenant):
    s = settings(stripe_secret_key="sk_test_51x", stripe_webhook_secret="whsec_real_test")
    app = create_app(s, pool=pool, redis=redis_client)
    t = await make_tenant()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b") as c:
        ev = checkout_event(t.org_id)
        assert (await deliver(c, ev)).status_code == 400  # the stub secret no longer works
        r = await deliver(c, ev, secret="whsec_real_test")
        assert r.json()["result"] == "credited"


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "stripe_mode": "stub"}


# ------------------------------------------------------------ checkout endpoint


def body(org, amount=10_000_000, **kw):
    return {"org_id": str(org), "amount_micros": amount, "success_url": "http://web/ok",
            "cancel_url": "http://web/cancel", **kw}  # fmt: skip


async def test_checkout_requires_the_internal_token(client, pool, redis_client, make_tenant):
    t = await make_tenant()
    r = await client.post("/internal/checkout-sessions", json=body(t.org_id))
    assert r.status_code == 401
    r = await client.post(
        "/internal/checkout-sessions",
        json=body(t.org_id),
        headers={"Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401
    unset = create_app(BillingSettings(**STUB), pool=pool, redis=redis_client)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=unset), base_url="http://b"
    ) as c:
        r = await c.post(
            "/internal/checkout-sessions",
            json=body(t.org_id),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 503


@pytest.mark.parametrize(
    ("amount", "status", "code"),
    [
        (10_000_001, 400, "invalid_amount"),  # not whole cents
        (1_000_000, 400, "invalid_amount"),  # below $5
        (10_000_000_000, 400, "invalid_amount"),  # above $500
    ],
)
async def test_checkout_validates_amount(client, make_tenant, amount, status, code):
    t = await make_tenant()
    r = await client.post(
        "/internal/checkout-sessions",
        json=body(t.org_id, amount),
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert r.status_code == status and r.json()["error"]["code"] == code


async def test_checkout_needs_an_active_org(client, make_tenant):
    auth = {"Authorization": f"Bearer {TOKEN}"}
    pending = await make_tenant(status="pending_approval")
    r = await client.post("/internal/checkout-sessions", json=body(pending.org_id), headers=auth)
    assert r.status_code == 409
    r = await client.post("/internal/checkout-sessions", json=body(uuid.uuid4()), headers=auth)
    assert r.status_code == 404
    r = await client.post("/internal/checkout-sessions", json={"org_id": "x"}, headers=auth)
    assert r.status_code == 400


# ------------------------------------------------------------ refunds and disputes


def _event(typ: str, obj: dict) -> dict:
    return {
        "id": f"evt_test_{secrets.token_hex(8)}",
        "object": "event",
        "type": typ,
        "livemode": False,
        "data": {"object": obj},
    }


def charge_event(pi: str, amount_refunded: int, cents: int = 2000) -> dict:
    charge = {
        "id": "ch_test_1",
        "object": "charge",
        "amount": cents,
        "currency": "usd",
        "amount_refunded": amount_refunded,
        "payment_intent": pi,
        "livemode": False,
    }
    return _event("charge.refunded", charge)


def dispute_event(pi: str, cents: int, dispute_id: str | None = None) -> dict:
    dispute = {
        "id": dispute_id or f"dp_test_{secrets.token_hex(6)}",
        "object": "dispute",
        "amount": cents,
        "currency": "usd",
        "payment_intent": pi,
        "charge": "ch_test_1",
    }
    return _event("charge.dispute.created", dispute)


async def refunds(pool, org) -> list[int]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT e.amount_micros FROM ledger_transactions t "
            "JOIN ledger_entries e ON e.transaction_id = t.id "
            "WHERE t.org_id = %s AND t.kind = 'refund' AND e.account_id <> %s "
            "ORDER BY t.created_at",
            (org, STRIPE_CLEARING),
        )
        return [r[0] for r in await cur.fetchall()]


async def test_refunds_claw_back_credit_once_and_cumulatively(
    client, pool, redis_client, make_tenant
):
    t = await make_tenant()
    limiter = Limiter(redis_client)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    pi = f"pi_test_{secrets.token_hex(8)}"
    assert (await deliver(client, checkout_event(t.org_id, 2000, payment_intent=pi))).json()[
        "result"
    ] == "credited"

    first = charge_event(pi, amount_refunded=500)  # $5 of $20 refunded
    assert (await deliver(client, first)).json()["result"] == "refunded"
    assert (await deliver(client, first)).json()["result"] == "duplicate"
    # Another event repeating the same cumulative total takes nothing more.
    assert (await deliver(client, charge_event(pi, 500))).json()["result"] == "already_refunded"
    # A second refund: Stripe's total is now $12; $7 more is taken back.
    assert (await deliver(client, charge_event(pi, 1200))).json()["result"] == "refunded"
    # An older event arriving late (total $5) changes nothing.
    assert (await deliver(client, charge_event(pi, 500))).json()["result"] == "already_refunded"
    assert await refunds(pool, t.org_id) == [-5_000_000, -7_000_000]
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == 8_000_000
        assert await ledger_is_balanced(conn)
    assert int(await redis_client.get(org_key(t.org_id, "credit"))) == 8_000_000

    # A dispute on the rest takes the remaining $8 at most, once per dispute id.
    dp = dispute_event(pi, 2000, "dp_test_x")
    assert (await deliver(client, dp)).json()["result"] == "refunded"
    again = dispute_event(pi, 2000, "dp_test_x")
    assert (await deliver(client, again)).json()["result"] == "already_refunded"
    assert await refunds(pool, t.org_id) == [-5_000_000, -7_000_000, -8_000_000]
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == 0


async def post_usage(conn, org, micros: int) -> None:
    acct = await org_credit_account(conn, org)
    await post_transaction(
        conn,
        kind="usage",
        idempotency_key=f"usage:{org}:{secrets.token_hex(4)}",
        org_id=org,
        entries=[Entry(acct, -micros), Entry(REVENUE, micros)],
    )
    await conn.commit()


async def test_dispute_can_take_the_balance_negative(client, pool, make_tenant):
    t = await make_tenant()
    pi = f"pi_test_{secrets.token_hex(8)}"
    await deliver(client, checkout_event(t.org_id, 1000, payment_intent=pi))
    async with pool.connection() as conn:
        # The org spent $6 of its $10 before disputing the payment.
        await post_usage(conn, t.org_id, 6_000_000)
    assert (await deliver(client, dispute_event(pi, 1000))).json()["result"] == "refunded"
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == -6_000_000
        assert await ledger_is_balanced(conn)


async def test_clawback_for_an_unknown_payment_is_kept_for_an_admin(client, pool, make_tenant):
    t = await make_tenant()
    ev = charge_event("pi_test_unknown", 500)
    assert (await deliver(client, ev)).json()["result"] == "unapplied"
    assert await stripe_row(pool, ev["id"]) == ("charge.refunded", False)
    bad = dispute_event("pi_test_unknown", 500)
    bad["data"]["object"]["object"] = "charge"
    assert (await deliver(client, bad)).json()["result"] == "unapplied"
    assert await refunds(pool, t.org_id) == []
