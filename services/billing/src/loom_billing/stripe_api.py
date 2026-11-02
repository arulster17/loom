"""The two Stripe API calls billing makes, over plain HTTP (no Stripe SDK dependency).

Checkout session create is `POST /v1/checkout/sessions`, form-encoded with Stripe's
bracket notation, `Authorization: Bearer <secret key>`. The same code talks to real
Stripe (test keys) or to the local stub, which implements the same wire format.

The HTTP transport is injectable: the default uses urllib in a thread (keeps the image
free of an HTTP client dependency); tests pass an in-process transport.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from loom_billing.settings import MICROS_PER_CENT

# (method, url, headers, body) -> (status, body)
Transport = Callable[[str, str, Mapping[str, str], bytes | None], Awaitable[tuple[int, bytes]]]

TIMEOUT_S = 20.0
PURPOSE = "loom_credits"


async def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, bytes]:
    def call() -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    return await asyncio.to_thread(call)


class StripeError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Stripe API {status}: {message}")
        self.status = status


@dataclass(frozen=True)
class CheckoutSession:
    id: str
    url: str
    amount_micros: int


def form_encode(params: Mapping[str, Any]) -> bytes:
    """Stripe's nested form encoding: {'a': {'b': 1}, 'c': [{'d': 2}]} ->
    a[b]=1&c[0][d]=2."""
    pairs: list[tuple[str, str]] = []

    def walk(prefix: str, v: Any) -> None:
        if isinstance(v, Mapping):
            for k, sub in v.items():
                walk(f"{prefix}[{k}]" if prefix else str(k), sub)
        elif isinstance(v, list | tuple):
            for i, sub in enumerate(v):
                walk(f"{prefix}[{i}]", sub)
        elif v is None:
            return
        elif isinstance(v, bool):
            pairs.append((prefix, "true" if v else "false"))
        else:
            pairs.append((prefix, str(v)))

    walk("", params)
    return urllib.parse.urlencode(pairs).encode()


class StripeClient:
    def __init__(
        self, api_base: str, secret_key: str, *, transport: Transport = urllib_transport
    ) -> None:
        if "_live_" in secret_key:
            raise ValueError("refusing a live Stripe key")
        self.api_base = api_base.rstrip("/")
        self.secret_key = secret_key
        self.transport = transport

    async def create_checkout_session(
        self,
        *,
        org_id: uuid.UUID,
        amount_micros: int,
        success_url: str,
        cancel_url: str,
        customer_email: str | None = None,
        idempotency_key: str | None = None,
    ) -> CheckoutSession:
        if amount_micros <= 0 or amount_micros % MICROS_PER_CENT:
            raise ValueError("amount must be a positive whole number of cents")
        cents = amount_micros // MICROS_PER_CENT
        meta = {"org_id": str(org_id), "credit_micros": str(amount_micros), "purpose": PURPOSE}
        params: dict[str, Any] = {
            "mode": "payment",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "client_reference_id": str(org_id),
            "customer_email": customer_email,
            "metadata": meta,
            "payment_intent_data": {"metadata": meta},
            "line_items": [
                {
                    "quantity": 1,
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": cents,
                        "product_data": {"name": "Loom credits"},
                    },
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        status, body = await self.transport(
            "POST", f"{self.api_base}/v1/checkout/sessions", headers, form_encode(params)
        )
        data = _json(body)
        if status // 100 != 2:
            msg = data.get("error", {}).get("message", "") if isinstance(data, dict) else ""
            raise StripeError(status, msg or "request failed")
        if data.get("livemode"):
            raise StripeError(status, "live-mode session returned; refusing")
        return CheckoutSession(str(data["id"]), str(data["url"]), amount_micros)


def _json(body: bytes) -> dict[str, Any]:
    try:
        v = json.loads(body or b"{}")
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}
