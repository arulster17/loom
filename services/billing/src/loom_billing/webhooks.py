"""Stripe webhook events -> ledger, idempotently (contracts §3: purchase keyed
`stripe:<event id>`).

One transaction per delivery:
1. `INSERT INTO stripe_events ... ON CONFLICT DO NOTHING`, then lock the row.
2. Already `processed_at`: a replay; nothing happens.
3. Otherwise handle the type and, if it was applied (or is a type we ignore), set
   `processed_at`. An event that cannot be applied (unknown org, wrong currency) keeps
   `processed_at` NULL: it is stored for an admin to look at, logged at error, and a
   replay tries again. Stripe still gets 200, since retrying would not fix it.

Purchases: `checkout.session.completed` with `payment_status = paid`, or
`checkout.session.async_payment_succeeded`, for a session this service created
(`metadata.purpose = loom_credits`). The org is `client_reference_id`; the amount is
what Stripe says was paid (`amount_total`, USD cents). A session is credited at most
once even across different event ids (completed, then async-succeeded): an advisory
lock on the session id and a check for an earlier purchase with that `external_ref`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb

from loom_billing import metrics
from loom_billing.logs import fields
from loom_billing.settings import MICROS_PER_CENT
from loom_billing.stripe_api import PURPOSE
from loom_platform.ledger import STRIPE_CLEARING, Entry, org_credit_account, post_transaction

log = logging.getLogger("loom_billing.webhooks")

CREDIT_TYPES = {"checkout.session.completed", "checkout.session.async_payment_succeeded"}


@dataclass(frozen=True)
class Outcome:
    result: str  # credited | duplicate | ignored | not_paid | already_credited | unapplied
    org_id: uuid.UUID | None = None
    amount_micros: int = 0
    transaction_id: uuid.UUID | None = None
    reason: str | None = None

    @property
    def processed(self) -> bool:
        return self.result != "unapplied"


class BadEvent(ValueError):
    pass


def check_event_shape(event: Any) -> tuple[str, str]:
    if not isinstance(event, dict):
        raise BadEvent("event is not an object")
    evt_id, typ = event.get("id"), event.get("type")
    if not isinstance(evt_id, str) or not evt_id or not isinstance(typ, str):
        raise BadEvent("event without id/type")
    if event.get("livemode") is True:
        raise BadEvent("live-mode event refused (test mode only)")
    return evt_id, typ


async def handle_event(conn: AsyncConnection, event: dict[str, Any]) -> Outcome:
    """Process one verified event in the caller's transaction."""
    evt_id, typ = check_event_shape(event)
    async with conn.cursor() as cur:
        await cur.execute(
            "INSERT INTO stripe_events (id, type, payload) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (evt_id, typ, Jsonb(event)),
        )
        await cur.execute(
            "SELECT processed_at FROM stripe_events WHERE id = %s FOR UPDATE", (evt_id,)
        )
        row = await cur.fetchone()
    assert row is not None
    if row[0] is not None:
        outcome = Outcome("duplicate")
    elif typ in CREDIT_TYPES:
        outcome = await _credit_checkout(conn, evt_id, typ, event)
    else:
        outcome = Outcome("ignored")
    if outcome.processed and outcome.result != "duplicate":
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE stripe_events SET processed_at = now() WHERE id = %s", (evt_id,)
            )
    level = logging.ERROR if outcome.result == "unapplied" else logging.INFO
    log.log(
        level,
        "stripe.event",
        extra=fields(
            event_id=evt_id,
            type=typ,
            result=outcome.result,
            reason=outcome.reason,
            org_id=str(outcome.org_id) if outcome.org_id else None,
            amount_micros=outcome.amount_micros,
        ),
    )
    metrics.WEBHOOKS.labels(outcome.result).inc()
    return outcome


async def _credit_checkout(
    conn: AsyncConnection, evt_id: str, typ: str, event: dict[str, Any]
) -> Outcome:
    obj = (event.get("data") or {}).get("object")
    if not isinstance(obj, dict) or obj.get("object") != "checkout.session":
        return Outcome("unapplied", reason="data.object is not a checkout.session")
    meta = obj.get("metadata") or {}
    if meta.get("purpose") != PURPOSE:
        return Outcome("ignored", reason="not a Loom credits session")
    session_id = obj.get("id")
    if not isinstance(session_id, str) or not session_id:
        return Outcome("unapplied", reason="session without id")
    if typ == "checkout.session.completed" and obj.get("payment_status") != "paid":
        # Delayed payment method: async_payment_succeeded follows if it clears.
        return Outcome("not_paid")
    try:
        org_id = uuid.UUID(str(obj.get("client_reference_id") or meta.get("org_id")))
    except ValueError:
        return Outcome("unapplied", reason="no org id on the session")
    if meta.get("org_id") and meta.get("org_id") != str(org_id):
        return Outcome("unapplied", org_id=org_id, reason="client_reference_id != metadata")
    if obj.get("currency") != "usd":
        return Outcome("unapplied", org_id=org_id, reason=f"currency {obj.get('currency')!r}")
    cents = obj.get("amount_total")
    if not isinstance(cents, int) or isinstance(cents, bool) or cents <= 0:
        return Outcome("unapplied", org_id=org_id, reason="no positive amount_total")
    amount = cents * MICROS_PER_CENT

    async with conn.cursor() as cur:
        await cur.execute("SELECT 1 FROM orgs WHERE id = %s", (org_id,))
        if await cur.fetchone() is None:
            return Outcome("unapplied", org_id=org_id, reason="unknown org")
        await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (session_id,))
        await cur.execute(
            "SELECT id FROM ledger_transactions WHERE kind = 'purchase' AND org_id = %s "
            "AND external_ref = %s",
            (org_id, session_id),
        )
        earlier = await cur.fetchone()
    if earlier is not None:
        return Outcome("already_credited", org_id=org_id, transaction_id=earlier[0])

    account = await org_credit_account(conn, org_id)
    posted = await post_transaction(
        conn,
        kind="purchase",
        idempotency_key=f"stripe:{evt_id}",
        org_id=org_id,
        entries=[Entry(account, amount), Entry(STRIPE_CLEARING, -amount)],
        external_ref=session_id,
        memo="Stripe checkout",
    )
    metrics.CREDITED_MICROS.inc(amount)
    return Outcome("credited", org_id, amount, posted.transaction_id)
