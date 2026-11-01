"""Billing HTTP API (port 8090).

- `POST /stripe/webhook`: Stripe (or the stub) delivers events. Signature verified on
  the raw body (`Stripe-Signature`, tolerance `LOOM_STRIPE_WEBHOOK_TOLERANCE_S`), then
  `webhooks.handle_event` in one transaction. 400 on a bad signature or body; 200 once
  stored (including replays); 500 if Postgres fails, so Stripe retries.
- `POST /internal/checkout-sessions`: the web app asks for a Checkout session to add
  credit. Not public: `Authorization: Bearer $LOOM_BILLING_INTERNAL_TOKEN`, a shared
  secret between web and billing (the web app has already authenticated the user and
  checked they belong to the org). Body `{"org_id", "amount_micros", "success_url",
  "cancel_url", "customer_email"?}`; answers `{"id", "url", "amount_micros"}`. The org
  must be `active`; the amount a whole number of cents within
  [`LOOM_CHECKOUT_MIN_MICROS`, `LOOM_CHECKOUT_MAX_MICROS`].
- `GET /health`: Postgres and Redis reachable, and the Stripe mode.

Errors use the OpenAI shape from `loom_platform.errors`.
"""

from __future__ import annotations

import hmac
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field, ValidationError
from redis.asyncio import Redis

from loom_billing import metrics
from loom_billing.logs import fields
from loom_billing.settings import MICROS_PER_CENT, BillingSettings
from loom_billing.stripe_api import StripeClient, StripeError, Transport, urllib_transport
from loom_billing.stripe_sig import SignatureError, verify
from loom_billing.webhooks import BadEvent, handle_event
from loom_platform.db import open_pool
from loom_platform.errors import ApiError
from loom_platform.ledger import org_credit_state
from loom_platform.limiter import Limiter

log = logging.getLogger("loom_billing.api")


class CheckoutRequest(BaseModel):
    org_id: uuid.UUID
    amount_micros: int
    success_url: Annotated[str, Field(min_length=1, max_length=2048)]
    cancel_url: Annotated[str, Field(min_length=1, max_length=2048)]
    customer_email: Annotated[str | None, Field(max_length=320)] = None


def _error(status: int, code: str, message: str, type_: str = "invalid_request_error") -> ApiError:
    return ApiError(status, type_, code, message)


def create_app(
    settings: BillingSettings,
    *,
    pool: AsyncConnectionPool | None = None,
    redis: Redis | None = None,
    stripe_transport: Transport = urllib_transport,
) -> FastAPI:
    """The app. Pass both `pool` and `redis` to share them (tests, the combined
    process; usable without running the lifespan); otherwise the lifespan opens and
    closes its own."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if shared:
            yield
            return
        app.state.pool = await open_pool(settings.database_url, max_size=5)
        app.state.redis = Redis.from_url(settings.redis_url)
        app.state.limiter = Limiter(app.state.redis, usage_stream=settings.usage_stream)
        try:
            yield
        finally:
            await app.state.pool.close()
            await app.state.redis.aclose()

    app = FastAPI(title="Loom billing", lifespan=lifespan, docs_url=None, redoc_url=None)
    shared = pool is not None and redis is not None
    if pool is not None and redis is not None:
        app.state.pool = pool
        app.state.redis = redis
        app.state.limiter = Limiter(redis, usage_stream=settings.usage_stream)
    base, key = settings.stripe_api()
    stripe = StripeClient(base, key, transport=stripe_transport)
    secret = settings.webhook_secret()

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError) -> JSONResponse:
        return JSONResponse(e.body(), status_code=e.status, headers=e.headers)

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        ok = True
        try:
            async with request.app.state.pool.connection() as conn:
                await conn.execute("SELECT 1")
        except Exception:
            ok = False
        try:
            await request.app.state.redis.ping()
        except Exception:
            ok = False
        return JSONResponse(
            {"status": "ok" if ok else "degraded", "stripe_mode": settings.stripe_mode},
            status_code=200 if ok else 503,
        )

    @app.post("/stripe/webhook")
    async def stripe_webhook(
        request: Request,
        stripe_signature: Annotated[str | None, Header(alias="Stripe-Signature")] = None,
    ) -> JSONResponse:
        payload = await request.body()
        try:
            verify(
                payload,
                stripe_signature,
                secret,
                tolerance_s=settings.stripe_webhook_tolerance_s,
            )
        except SignatureError as e:
            metrics.WEBHOOKS.labels("bad_signature").inc()
            log.warning("stripe.bad_signature", extra=fields(reason=str(e)))
            raise _error(400, "bad_signature", "Invalid Stripe signature.") from None
        try:
            event = json.loads(payload)
            async with request.app.state.pool.connection() as conn, conn.transaction():
                outcome = await handle_event(conn, event)
        except (ValueError, BadEvent) as e:
            metrics.WEBHOOKS.labels("bad_event").inc()
            log.warning("stripe.bad_event", extra=fields(reason=str(e)))
            raise _error(400, "bad_event", "Unreadable or refused event.") from None
        if outcome.result == "credited" and outcome.org_id is not None:
            await _refresh_credit(request, outcome.org_id)
        return JSONResponse({"received": True, "result": outcome.result})

    @app.post("/internal/checkout-sessions")
    async def checkout(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
    ) -> JSONResponse:
        _check_internal_token(settings, authorization)
        try:
            body = CheckoutRequest.model_validate_json(await request.body())
        except ValidationError as e:
            raise _error(400, "invalid_request", _first_error(e)) from None
        amount = body.amount_micros
        if amount % MICROS_PER_CENT:
            raise _error(400, "invalid_amount", "amount_micros must be a whole number of cents.")
        if not settings.checkout_min_micros <= amount <= settings.checkout_max_micros:
            raise _error(
                400,
                "invalid_amount",
                f"amount_micros must be between {settings.checkout_min_micros} and "
                f"{settings.checkout_max_micros}.",
            )
        async with request.app.state.pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT status FROM orgs WHERE id = %s", (body.org_id,))
            row = await cur.fetchone()
        if row is None:
            raise _error(404, "org_not_found", "No such organization.")
        if row[0] != "active":
            raise _error(409, "org_not_active", f"Organization is {row[0]}.", "permission_error")
        try:
            session = await stripe.create_checkout_session(
                org_id=body.org_id,
                amount_micros=amount,
                success_url=body.success_url,
                cancel_url=body.cancel_url,
                customer_email=body.customer_email,
            )
        except (StripeError, OSError) as e:
            metrics.CHECKOUT_SESSIONS.labels("error").inc()
            log.error("checkout.stripe_error", extra=fields(reason=str(e)))
            raise _error(502, "stripe_error", "Payment provider error.", "server_error") from None
        metrics.CHECKOUT_SESSIONS.labels("created").inc()
        log.info(
            "checkout.created",
            extra=fields(org_id=str(body.org_id), session_id=session.id, amount_micros=amount),
        )
        return JSONResponse(
            {"id": session.id, "url": session.url, "amount_micros": session.amount_micros}
        )

    return app


async def _refresh_credit(request: Request, org_id: uuid.UUID) -> None:
    """Push the new credit to Redis now (the loom_credit_changed listener would too)."""
    try:
        async with request.app.state.pool.connection() as conn:
            credit, version = await org_credit_state(conn, org_id)
        await request.app.state.limiter.set_credit(org_id, credit, version)
    except Exception:
        log.exception("stripe.credit_refresh_failed")


def _check_internal_token(settings: BillingSettings, authorization: str | None) -> None:
    token = settings.billing_internal_token
    if token is None:
        raise _error(
            503, "not_configured", "LOOM_BILLING_INTERNAL_TOKEN is not set.", "server_error"
        )
    scheme, _, given = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        given.encode(), token.get_secret_value().encode()
    ):
        raise _error(401, "unauthorized", "Missing or wrong internal token.")


def _first_error(e: ValidationError) -> str:
    err = e.errors()[0]
    loc = ".".join(str(x) for x in err.get("loc", ()))
    return f"{loc}: {err.get('msg')}" if loc else str(err.get("msg"))
