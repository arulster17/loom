"""The add-credits flow end to end against the local Stripe stub: the web app asks
billing for a Checkout session, the user pays on the stub's page, the stub sends a
signed webhook, billing credits the org once."""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.request
from collections.abc import AsyncIterator, Mapping

import httpx
import pytest
import uvicorn
from billing_fixtures import STUB, STUB_WEBHOOK_SECRET, asgi_transport, ledger_is_balanced

from loom_billing.api import create_app
from loom_billing.settings import BillingSettings
from loom_billing.stripe_stub import create_stub_app
from loom_platform.ledger import org_balance
from loom_platform.limiter import Limiter, org_key

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TOKEN = "internal-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def checkout_body(org, amount=12_340_000):
    return {
        "org_id": str(org),
        "amount_micros": amount,
        "success_url": "http://web/billing/success?session={CHECKOUT_SESSION_ID}",
        "cancel_url": "http://web/billing",
        "customer_email": "owner@example.test",
    }


async def test_checkout_through_stub_credits_the_org(pool, redis_client, make_tenant):
    t = await make_tenant()
    limiter = Limiter(redis_client)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)

    settings = BillingSettings(billing_internal_token=TOKEN, **STUB)
    billing_box: dict[str, object] = {}

    async def deliver(url: str, body: bytes, headers: Mapping[str, str]) -> int:
        assert url == settings.stripe_stub_webhook_url
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=billing_box["app"]),  # type: ignore[arg-type]
            base_url="http://billing",
        ) as c:
            r = await c.post("/stripe/webhook", content=body, headers=dict(headers))
            return r.status_code

    stub = create_stub_app(
        public_url=settings.stripe_stub_public_url,
        webhook_url=settings.stripe_stub_webhook_url,
        webhook_secret=STUB_WEBHOOK_SECRET,
        deliver=deliver,
    )
    billing = create_app(
        settings, pool=pool, redis=redis_client, stripe_transport=asgi_transport(stub)
    )
    billing_box["app"] = billing

    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=billing), base_url="http://b") as web,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=stub), base_url="http://s") as user,
    ):
        r = await web.post(
            "/internal/checkout-sessions", json=checkout_body(t.org_id), headers=AUTH
        )
        assert r.status_code == 200, r.text
        session = r.json()
        assert session["id"].startswith("cs_test_")
        assert session["url"] == f"http://localhost:12111/checkout/{session['id']}"
        assert session["amount_micros"] == 12_340_000

        stored = stub.state.sessions[session["id"]]
        assert stored["amount_total"] == 1234 and stored["currency"] == "usd"
        assert stored["client_reference_id"] == str(t.org_id)
        assert stored["metadata"]["purpose"] == "loom_credits"

        page = await user.get(f"/checkout/{session['id']}")
        assert page.status_code == 200 and "$12.34" in page.text

        # Nothing credited before payment.
        async with pool.connection() as conn:
            assert await org_balance(conn, t.org_id) == 0

        r = await user.post(f"/checkout/{session['id']}/complete")
        assert r.status_code == 303
        assert r.headers["location"] == f"http://web/billing/success?session={session['id']}"
        async with pool.connection() as conn:
            assert await org_balance(conn, t.org_id) == 12_340_000
        assert int(await redis_client.get(org_key(t.org_id, "credit"))) == 12_340_000

        # Paying again re-sends the same event: credited once.
        r = await user.post(f"/checkout/{session['id']}/complete", params={"format": "json"})
        assert r.status_code == 200 and r.json()["webhook_status"] == 200
        async with pool.connection() as conn:
            assert await org_balance(conn, t.org_id) == 12_340_000
            assert await ledger_is_balanced(conn)


async def test_stub_refuses_bad_requests():
    stub = create_stub_app(
        public_url="http://stub", webhook_url="http://x", webhook_secret=STUB_WEBHOOK_SECRET
    )
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=stub), base_url="http://s") as c:
        form = {"mode": "payment", "success_url": "a", "cancel_url": "b",
                "line_items[0][price_data][currency]": "usd",
                "line_items[0][price_data][unit_amount]": "500",
                "line_items[0][quantity]": "1"}  # fmt: skip
        r = await c.post("/v1/checkout/sessions", data=form)
        assert r.status_code == 401
        r = await c.post(
            "/v1/checkout/sessions", data=form, headers={"Authorization": "Bearer sk_live_x"}
        )
        assert r.status_code == 401
        ok = {"Authorization": "Bearer sk_test_x"}
        r = await c.post("/v1/checkout/sessions", data={**form, "mode": "subscription"}, headers=ok)
        assert r.status_code == 400
        r = await c.post("/v1/checkout/sessions", data=form, headers=ok)
        assert r.status_code == 200
        sid = r.json()["id"]
        assert (await c.get(f"/v1/checkout/sessions/{sid}", headers=ok)).json()["id"] == sid
        # Cancel, then paying is refused.
        r = await c.post(f"/checkout/{sid}/cancel")
        assert r.status_code == 303 and r.headers["location"] == "b"
        assert (await c.post(f"/checkout/{sid}/complete")).status_code == 400
        assert (await c.get("/checkout/cs_test_nope")).status_code == 404


@contextlib.asynccontextmanager
async def serve(app: object) -> AsyncIterator[str]:
    """Run an ASGI app on a real ephemeral port in this event loop."""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_config=None, lifespan="off")  # type: ignore[arg-type]
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


async def test_checkout_over_real_http(pool, redis_client, make_tenant):
    """Same flow over sockets with the production urllib transport, as in compose."""
    t = await make_tenant()
    holder: dict[str, str] = {}

    async def lazy_deliver(url: str, body: bytes, headers: Mapping[str, str]) -> int:
        from loom_billing.stripe_stub import urllib_deliver

        return await urllib_deliver(holder["billing"] + "/stripe/webhook", body, headers)

    stub = create_stub_app(
        public_url="http://localhost:12111",
        webhook_url="(set below)",
        webhook_secret=STUB_WEBHOOK_SECRET,
        deliver=lazy_deliver,
    )
    async with serve(stub) as stub_url:
        settings = BillingSettings(billing_internal_token=TOKEN, stripe_stub_url=stub_url, **STUB)
        billing = create_app(settings, pool=pool, redis=redis_client)
        async with serve(billing) as billing_url:
            holder["billing"] = billing_url

            def call(method: str, url: str, data: bytes | None = None) -> tuple[int, bytes]:
                req = urllib.request.Request(
                    url, data=data, method=method,
                    headers={**AUTH, "Content-Type": "application/json"},
                )  # fmt: skip
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status, resp.read()

            status, raw = await asyncio.to_thread(
                call,
                "POST",
                billing_url + "/internal/checkout-sessions",
                json.dumps(checkout_body(t.org_id, 5_000_000)).encode(),
            )
            assert status == 200
            sid = json.loads(raw)["id"]
            status, raw = await asyncio.to_thread(
                call, "POST", f"{stub_url}/checkout/{sid}/complete?format=json", b""
            )
            assert status == 200 and json.loads(raw)["webhook_status"] == 200
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == 5_000_000
