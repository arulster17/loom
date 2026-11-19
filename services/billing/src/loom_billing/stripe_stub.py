"""Local Stripe stub: just enough of Checkout for the add-credits flow with no Stripe
account (port 12111, `python -m loom_billing stripe-stub`).

- `POST /v1/checkout/sessions`: Stripe's wire format (form-encoded, bracket notation,
  `Authorization: Bearer sk_test_...`), returns a `checkout.session` object whose `url`
  is the stub's own payment page. Live keys are refused.
- `GET /v1/checkout/sessions/{id}`: retrieve.
- `GET /checkout/{id}`: a payment page with "Pay" and "Cancel".
- `POST /checkout/{id}/complete`: marks the session paid and delivers a
  `checkout.session.completed` event to `LOOM_STRIPE_STUB_WEBHOOK_URL`, signed exactly as
  Stripe signs (`Stripe-Signature: t=..,v1=..`) with `LOOM_STRIPE_STUB_WEBHOOK_SECRET`,
  then redirects (303) to the session's `success_url`. Completing again re-sends the
  same event id (a replay, which billing must ignore). `?format=json` answers JSON
  instead of redirecting (scripts and tests).
- `POST /checkout/{id}/cancel`: redirects to `cancel_url`.
- `POST /v1/refunds`: Stripe's refund API (form `payment_intent`, optional `amount` in
  cents, default the rest); delivers `charge.refunded` with the charge's cumulative
  `amount_refunded`.
- `POST /stub/disputes`: not a Stripe API (Stripe tests disputes with special cards):
  JSON `{"payment_intent", "amount"?}` delivers `charge.dispute.created`.
- `POST /stub/control` (M3, docs/phase1-m3-spec.md §8.9): JSON
  `{"drop_deliveries": bool, "fail_status": int | null}`. While `drop_deliveries` is
  true the stub sends nothing and reports status 0 (unreachable); while `fail_status` is
  set it sends nothing and reports that status (a delivery that failed on the way, as
  Stripe would record it). Either field may be omitted (unchanged); answers the state.
- `GET /stub/events`: every event the stub made, newest last (`id`, `type`,
  `last_status`). `POST /stub/events/{id}/resend`: re-posts the stored event with a fresh
  signature (a new `t`), as Stripe's dashboard resend does; answers
  `{"event_id", "webhook_status"}` (subject to `/stub/control`).

Sessions carry `expires_at`: the request's (refused unless 30 minutes to 24 hours ahead,
as Stripe refuses it), else 24 hours after creation (Stripe's default).

State is in memory; restarting the stub forgets sessions. Not a security boundary: it
exists only for local development and tests.
"""

from __future__ import annotations

import html
import json
import logging
import secrets
import time
import urllib.parse
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from loom_billing.logs import fields
from loom_billing.stripe_api import urllib_transport
from loom_billing.stripe_sig import sign

log = logging.getLogger("loom_billing.stripe_stub")

# (url, body, headers) -> status
Deliver = Callable[[str, bytes, Mapping[str, str]], Awaitable[int]]


async def urllib_deliver(url: str, body: bytes, headers: Mapping[str, str]) -> int:
    status, _ = await urllib_transport("POST", url, headers, body)
    return status


def _err(status: int, message: str, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"error": {"type": type_, "message": message}}, status_code=status)


def _nest(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Undo Stripe's bracket encoding: a[b][0][c]=1 -> {'a': {'b': {'0': {'c': '1'}}}}."""
    root: dict[str, Any] = {}
    for key, value in pairs:
        parts = key.replace("]", "").split("[")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return root


def create_stub_app(
    *,
    public_url: str,
    webhook_url: str,
    webhook_secret: str,
    deliver: Deliver = urllib_deliver,
) -> FastAPI:
    app = FastAPI(title="Loom Stripe stub", docs_url=None, redoc_url=None)
    sessions: dict[str, dict[str, Any]] = {}
    events: dict[str, str] = {}  # session id -> event id (stable across re-sends)
    charges: dict[str, dict[str, Any]] = {}  # payment intent -> charge
    sent: dict[str, dict[str, Any]] = {}  # event id -> {"event", "last_status"}
    control: dict[str, Any] = {"drop_deliveries": False, "fail_status": None}
    app.state.sessions = sessions
    app.state.charges = charges
    app.state.events = sent
    app.state.control = control
    base = public_url.rstrip("/")

    def _auth(authorization: str | None) -> JSONResponse | None:
        scheme, _, key = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not key.startswith(("sk_test_", "rk_test_")):
            return _err(401, "Invalid API Key provided (the stub takes sk_test_ keys only).")
        return None

    @app.post("/v1/checkout/sessions")
    async def create_session(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> JSONResponse:
        if (bad := _auth(authorization)) is not None:
            return bad
        form = _nest(urllib.parse.parse_qsl((await request.body()).decode(), True))
        if form.get("mode") != "payment":
            return _err(400, "stub supports mode=payment only")
        for need in ("success_url", "cancel_url"):
            if not form.get(need):
                return _err(400, f"Missing required param: {need}.")
        items = form.get("line_items") or {}
        total = 0
        currency = None
        try:
            for item in items.values():
                pd = item["price_data"]
                total += int(pd["unit_amount"]) * int(item.get("quantity", "1"))
                currency = pd["currency"]
        except (KeyError, ValueError, TypeError, AttributeError):
            return _err(400, "line_items need price_data[unit_amount], price_data[currency]")
        if total < 50:
            return _err(400, "amount_total must be at least 50 cents")
        created = int(time.time())
        expires_at = created + 86_400
        if form.get("expires_at") is not None:
            try:
                expires_at = int(form["expires_at"])
            except (TypeError, ValueError):
                return _err(400, "expires_at must be a Unix timestamp")
            # Stripe: between 30 minutes and 24 hours after creation (a minute of slack).
            if not created + 1800 - 60 <= expires_at <= created + 86_400 + 60:
                return _err(400, "expires_at must be 30 minutes to 24 hours from now")
        sid = "cs_test_" + secrets.token_hex(12)
        session = {
            "id": sid,
            "object": "checkout.session",
            "amount_subtotal": total,
            "amount_total": total,
            "currency": currency,
            "client_reference_id": form.get("client_reference_id"),
            "customer_email": form.get("customer_email"),
            "metadata": form.get("metadata") or {},
            "mode": "payment",
            "payment_intent": None,
            "payment_status": "unpaid",
            "status": "open",
            "success_url": form["success_url"],
            "cancel_url": form["cancel_url"],
            "url": f"{base}/checkout/{sid}",
            "livemode": False,
            "created": created,
            "expires_at": expires_at,
        }
        sessions[sid] = session
        log.info("stub.session_created", extra=fields(session_id=sid, amount_cents=total))
        return JSONResponse(session)

    @app.get("/v1/checkout/sessions/{sid}")
    async def get_session(
        sid: str, authorization: Annotated[str | None, Header()] = None
    ) -> JSONResponse:
        if (bad := _auth(authorization)) is not None:
            return bad
        if sid not in sessions:
            return _err(404, f"No such checkout.session: '{sid}'")
        return JSONResponse(sessions[sid])

    @app.get("/checkout/{sid}", response_class=HTMLResponse)
    async def page(sid: str) -> Response:
        s = sessions.get(sid)
        if s is None:
            return HTMLResponse("<h1>Unknown checkout session</h1>", status_code=404)
        amount = f"${s['amount_total'] / 100:,.2f}"
        e = html.escape
        done = s["payment_status"] == "paid"
        return HTMLResponse(
            f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Loom test checkout</title>
<style>body{{font-family:system-ui,sans-serif;max-width:28rem;margin:4rem auto;padding:0 1rem}}
button{{font-size:1rem;padding:.6rem 1.2rem;margin-right:.5rem}}
.note{{color:#666;font-size:.85rem}}</style></head>
<body>
<h1>Add {e(amount)} of Loom credit</h1>
<p class="note">Local Stripe stub: no card is charged. Session <code>{e(sid)}</code>.</p>
<p>Status: <strong>{e(s["payment_status"])}</strong></p>
<form method="post" action="/checkout/{e(sid)}/complete" style="display:inline">
  <button type="submit">{"Re-send payment event" if done else "Pay " + e(amount)}</button>
</form>
<form method="post" action="/checkout/{e(sid)}/cancel" style="display:inline">
  <button type="submit">Cancel</button>
</form>
</body></html>"""
        )

    async def send_event(typ: str, obj: dict[str, Any], evt_id: str | None = None) -> int:
        """Deliver a signed event, exactly as Stripe would; returns the HTTP status (0 when
        the webhook endpoint was unreachable)."""
        evt_id = evt_id or "evt_test_" + secrets.token_hex(12)
        event = {
            "id": evt_id,
            "object": "event",
            "api_version": "2024-06-20",
            "created": int(time.time()),
            "livemode": False,
            "pending_webhooks": 1,
            "type": typ,
            "data": {"object": obj},
        }
        if evt_id in sent:  # a re-send of the same event: the stored object, as Stripe does
            event = sent[evt_id]["event"]
        else:  # a snapshot: later changes to the charge or session do not alter it
            event = json.loads(json.dumps(event))
            sent[evt_id] = {"event": event, "last_status": None}
        return await post_event(event)

    async def post_event(event: dict[str, Any]) -> int:
        """Sign `event` now (a fresh `t`) and post it, unless `/stub/control` drops or
        fails deliveries."""
        evt_id, typ = event["id"], event["type"]
        if control["drop_deliveries"]:
            status = 0
            log.info("stub.webhook_dropped", extra=fields(type=typ, event_id=evt_id))
        elif control["fail_status"] is not None:
            status = int(control["fail_status"])
            log.info("stub.webhook_failed_by_control", extra=fields(event_id=evt_id, status=status))
        else:
            body = json.dumps(event).encode()
            headers = {
                "Content-Type": "application/json",
                "Stripe-Signature": sign(body, webhook_secret),
            }
            try:
                status = await deliver(webhook_url, body, headers)
            except (TimeoutError, OSError) as exc:
                log.error("stub.webhook_failed", extra=fields(event_id=evt_id, reason=str(exc)))
                status = 0
            log.info("stub.webhook_sent", extra=fields(type=typ, event_id=evt_id, status=status))
        sent.setdefault(evt_id, {"event": event, "last_status": None})["last_status"] = status
        return status

    @app.post("/stub/control")
    async def set_control(request: Request) -> JSONResponse:
        try:
            body = json.loads(await request.body() or b"{}")
            if not isinstance(body, dict):
                raise ValueError("not an object")
            drop = body.get("drop_deliveries", control["drop_deliveries"])
            fail = body.get("fail_status", control["fail_status"])
            if not isinstance(drop, bool):
                raise ValueError("drop_deliveries must be a boolean")
            if fail is not None and (
                not isinstance(fail, int) or isinstance(fail, bool) or not 100 <= fail <= 599
            ):
                raise ValueError("fail_status must be an HTTP status or null")
        except ValueError as e:
            return _err(400, f'body: {{"drop_deliveries": bool, "fail_status": int | null}} ({e})')
        control.update(drop_deliveries=drop, fail_status=fail)
        return JSONResponse(dict(control))

    @app.get("/stub/events")
    async def list_events() -> JSONResponse:
        return JSONResponse(
            [
                {"id": k, "type": v["event"]["type"], "last_status": v["last_status"]}
                for k, v in sent.items()
            ]
        )

    @app.post("/stub/events/{evt_id}/resend")
    async def resend(evt_id: str) -> JSONResponse:
        stored = sent.get(evt_id)
        if stored is None:
            return _err(404, f"No such event: '{evt_id}'")
        status = await post_event(stored["event"])
        return JSONResponse({"event_id": evt_id, "webhook_status": status})

    def delivery_failed(status: int) -> JSONResponse:
        return _err(502, f"webhook delivery to {webhook_url} failed ({status}); retry", "api_error")

    @app.post("/v1/refunds")
    async def refund(
        request: Request, authorization: Annotated[str | None, Header()] = None
    ) -> JSONResponse:
        if (bad := _auth(authorization)) is not None:
            return bad
        form = dict(urllib.parse.parse_qsl((await request.body()).decode(), True))
        charge = charges.get(form.get("payment_intent", ""))
        if charge is None:
            return _err(404, "No such payment_intent (pay a checkout session first).")
        left = charge["amount"] - charge["amount_refunded"]
        try:
            amount = int(form.get("amount") or left)
        except ValueError:
            return _err(400, "amount must be an integer number of cents")
        if left <= 0:
            return _err(400, f"Charge {charge['id']} has already been refunded.")
        if not 0 < amount <= left:
            return _err(400, f"amount must be between 1 and {left}")
        charge["amount_refunded"] += amount
        charge["refunded"] = charge["amount_refunded"] == charge["amount"]
        refund_obj = {
            "id": "re_test_" + secrets.token_hex(12),
            "object": "refund",
            "amount": amount,
            "charge": charge["id"],
            "payment_intent": charge["payment_intent"],
            "currency": charge["currency"],
            "status": "succeeded",
        }
        charge["refunds"]["data"].insert(0, refund_obj)
        status = await send_event("charge.refunded", charge)
        if status // 100 != 2:
            return delivery_failed(status)
        return JSONResponse(refund_obj)

    @app.post("/stub/disputes")
    async def dispute(request: Request) -> JSONResponse:
        try:
            form = json.loads(await request.body() or b"{}")
            charge = charges.get(form.get("payment_intent", ""))
            amount = int(form.get("amount") or (charge or {}).get("amount", 0))
        except (ValueError, TypeError, AttributeError):
            return _err(400, 'body: {"payment_intent": "pi_...", "amount": cents?}')
        if charge is None:
            return _err(404, "No such payment_intent (pay a checkout session first).")
        if not 0 < amount <= charge["amount"]:
            return _err(400, f"amount must be between 1 and {charge['amount']}")
        dispute_obj = {
            "id": "dp_test_" + secrets.token_hex(12),
            "object": "dispute",
            "amount": amount,
            "charge": charge["id"],
            "payment_intent": charge["payment_intent"],
            "currency": charge["currency"],
            "reason": "fraudulent",
            "status": "needs_response",
            "livemode": False,
        }
        status = await send_event("charge.dispute.created", dispute_obj)
        if status // 100 != 2:
            return delivery_failed(status)
        return JSONResponse(dispute_obj)

    @app.post("/checkout/{sid}/complete")
    async def complete(sid: str, format: str | None = None) -> Response:
        s = sessions.get(sid)
        if s is None:
            return _err(404, f"No such checkout.session: '{sid}'")
        if s["status"] == "expired":
            return _err(400, "session was cancelled")
        if s["payment_status"] != "paid":
            pi = "pi_test_" + secrets.token_hex(12)
            s.update(status="complete", payment_status="paid", payment_intent=pi)
            charges[pi] = {
                "id": "ch_test_" + secrets.token_hex(12),
                "object": "charge",
                "amount": s["amount_total"],
                "amount_refunded": 0,
                "currency": s["currency"],
                "payment_intent": pi,
                "refunded": False,
                "refunds": {"object": "list", "data": []},
                "livemode": False,
            }
        evt_id = events.setdefault(sid, "evt_test_" + secrets.token_hex(12))
        status = await send_event("checkout.session.completed", s, evt_id)
        if status // 100 != 2:
            return delivery_failed(status)
        if format == "json":
            return JSONResponse({"session": s, "event_id": evt_id, "webhook_status": status})
        return RedirectResponse(
            s["success_url"].replace("{CHECKOUT_SESSION_ID}", sid), status_code=303
        )

    @app.post("/checkout/{sid}/cancel")
    async def cancel(sid: str) -> Response:
        s = sessions.get(sid)
        if s is None:
            return _err(404, f"No such checkout.session: '{sid}'")
        if s["payment_status"] != "paid":
            s["status"] = "expired"
        return RedirectResponse(s["cancel_url"], status_code=303)

    return app
