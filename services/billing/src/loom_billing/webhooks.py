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
The session's PaymentIntent is recorded in `stripe_payments` with the purchase.

Clawbacks (`refund` transactions: org credit -X, stripe_clearing +X, keyed
`stripe:<event id>`), found through the PaymentIntent in `stripe_payments` (row-locked):

- `charge.refunded`: the charge's cumulative `amount_refunded` minus what was already
  applied for that payment, so each refund is taken back once whatever the order or
  repetition of events (an older event with a smaller total takes nothing).
- `charge.dispute.created`: the disputed `amount`, once per dispute id.

Together they never take back more than the purchase. The balance may go negative,
which is correct: the money was returned.

Out of order: Stripe does not order deliveries, so a refund or dispute can arrive
before the purchase it reverses is recorded. It is stored unapplied (Stripe gets 200);
crediting the purchase then applies every pending clawback for the same PaymentIntent
in the same transaction, and `retry_unapplied` (billing worker, every minute) re-runs
unapplied events, which covers the race where both commit at once. Without both the
customer would be refunded and keep the credit. `loom_billing_stripe_unapplied_events`
counts what is still waiting (alert when it stays above 0).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from loom_billing import metrics
from loom_billing.logs import fields
from loom_billing.settings import MICROS_PER_CENT
from loom_billing.stripe_api import PURPOSE
from loom_platform.ledger import STRIPE_CLEARING, Entry, org_credit_account, post_transaction

log = logging.getLogger("loom_billing.webhooks")

CREDIT_TYPES = {"checkout.session.completed", "checkout.session.async_payment_succeeded"}
CLAWBACK_TYPES = {"charge.refunded", "charge.dispute.created"}


@dataclass(frozen=True)
class Outcome:
    # credited | refunded | duplicate | ignored | not_paid | already_credited |
    # already_refunded | unapplied
    result: str
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


async def handle_event(
    conn: AsyncConnection, event: dict[str, Any], *, retry: bool = False
) -> Outcome:
    """Process one verified event in the caller's transaction. `retry`: a re-run of a
    stored unapplied event (`retry_unapplied`), which stays quiet while still unapplied."""
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
    elif typ in CLAWBACK_TYPES:
        outcome = await _claw_back(conn, evt_id, typ, event)
    else:
        outcome = Outcome("ignored")
    if outcome.processed and outcome.result != "duplicate":
        async with conn.cursor() as cur:
            await cur.execute(
                "UPDATE stripe_events SET processed_at = now() WHERE id = %s", (evt_id,)
            )
    if retry and outcome.result == "unapplied":
        return outcome  # logged and counted when it first arrived
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
    payment_intent = obj.get("payment_intent")
    if isinstance(payment_intent, str) and payment_intent:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO stripe_payments (payment_intent, session_id, org_id, "
                "purchase_transaction_id, amount_micros) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (payment_intent, session_id, org_id, posted.transaction_id, amount),
            )
        await _apply_pending_clawbacks(conn, payment_intent)
    metrics.CREDITED_MICROS.inc(amount)
    return Outcome("credited", org_id, amount, posted.transaction_id)


async def _apply_pending_clawbacks(conn: AsyncConnection, payment_intent: str) -> None:
    """Refunds and disputes of this PaymentIntent that arrived before its purchase was
    recorded: apply them now, in the purchase's transaction."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, type, payload FROM stripe_events "
            "WHERE processed_at IS NULL AND type = ANY(%s) "
            "AND payload -> 'data' -> 'object' ->> 'payment_intent' = %s "
            "ORDER BY received_at, id FOR UPDATE",
            (sorted(CLAWBACK_TYPES), payment_intent),
        )
        pending = await cur.fetchall()
    for evt_id, typ, payload in pending:
        outcome = await _claw_back(conn, evt_id, typ, payload)
        if outcome.processed:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE stripe_events SET processed_at = now() WHERE id = %s", (evt_id,)
                )
        metrics.WEBHOOKS.labels(f"late_{outcome.result}").inc()
        log.info(
            "stripe.event_applied_late",
            extra=fields(
                event_id=evt_id,
                type=typ,
                result=outcome.result,
                org_id=str(outcome.org_id) if outcome.org_id else None,
                amount_micros=outcome.amount_micros,
            ),
        )


# Unapplied events younger than this are re-run every minute; older ones are left to
# an admin (still counted in the gauge).
RETRY_WINDOW = "30 days"


async def retry_unapplied(pool: AsyncConnectionPool, *, limit: int = 500) -> int:
    """Re-run stored unapplied events (oldest first), each in its own transaction, and
    refresh the unapplied gauge. Returns how many were applied now."""
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id FROM stripe_events WHERE processed_at IS NULL "
            f"AND received_at > now() - interval '{RETRY_WINDOW}' "
            "ORDER BY received_at, id LIMIT %s",
            (limit,),
        )
        ids = [r[0] for r in await cur.fetchall()]
    applied = 0
    for evt_id in ids:
        async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute(
                "SELECT payload FROM stripe_events WHERE id = %s AND processed_at IS NULL "
                "FOR UPDATE SKIP LOCKED",
                (evt_id,),
            )
            row = await cur.fetchone()
            if row is None:
                continue  # processed meanwhile, or a delivery holds it right now
            try:
                outcome = await handle_event(conn, row[0], retry=True)
            except BadEvent:
                continue
            if outcome.processed:
                applied += 1
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT count(*) FROM stripe_events WHERE processed_at IS NULL")
        row = await cur.fetchone()
    metrics.STRIPE_UNAPPLIED.set(int(row[0]) if row else 0)
    return applied


def _cents(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None


async def _claw_back(
    conn: AsyncConnection, evt_id: str, typ: str, event: dict[str, Any]
) -> Outcome:
    obj = (event.get("data") or {}).get("object")
    want = "charge" if typ == "charge.refunded" else "dispute"
    if not isinstance(obj, dict) or obj.get("object") != want:
        return Outcome("unapplied", reason=f"data.object is not a {want}")
    payment_intent = obj.get("payment_intent")
    if not isinstance(payment_intent, str) or not payment_intent:
        return Outcome("unapplied", reason="no payment_intent")
    if obj.get("currency") != "usd":
        return Outcome("unapplied", reason=f"currency {obj.get('currency')!r}")
    cents = _cents(obj.get("amount_refunded" if want == "charge" else "amount"))
    if cents is None:
        return Outcome("unapplied", reason="no amount")
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT org_id, amount_micros, refunded_micros, clawed_back_micros "
            "FROM stripe_payments WHERE payment_intent = %s FOR UPDATE",
            (payment_intent,),
        )
        row = await cur.fetchone()
        if row is None:
            return Outcome("unapplied", reason="payment not credited by Loom (yet)")
        org_id, purchased, refunded, clawed = row
        if want == "charge":
            # Cumulative: take only what is new since the last refund applied.
            new = cents * MICROS_PER_CENT - refunded
            if new <= 0:
                return Outcome("already_refunded", org_id)
            await cur.execute(
                "UPDATE stripe_payments SET refunded_micros = %s WHERE payment_intent = %s",
                (cents * MICROS_PER_CENT, payment_intent),
            )
            ref, memo = payment_intent, "Stripe refund"
        else:
            dispute_id = obj.get("id")
            if not isinstance(dispute_id, str) or not dispute_id:
                return Outcome("unapplied", org_id, reason="dispute without id")
            ref, memo = dispute_id, "Stripe dispute"
            await cur.execute(
                "SELECT 1 FROM ledger_transactions WHERE kind = 'refund' AND org_id = %s "
                "AND external_ref = %s",
                (org_id, ref),
            )
            if await cur.fetchone() is not None:
                return Outcome("already_refunded", org_id)
            new = cents * MICROS_PER_CENT
        amount = min(new, purchased - clawed)
        if amount <= 0:
            return Outcome("already_refunded", org_id, reason="whole purchase already taken back")
        await cur.execute(
            "UPDATE stripe_payments SET clawed_back_micros = clawed_back_micros + %s "
            "WHERE payment_intent = %s",
            (amount, payment_intent),
        )
    account = await org_credit_account(conn, org_id)
    posted = await post_transaction(
        conn,
        kind="refund",
        idempotency_key=f"stripe:{evt_id}",
        org_id=org_id,
        entries=[Entry(account, -amount), Entry(STRIPE_CLEARING, amount)],
        external_ref=ref,
        memo=memo,
    )
    metrics.CLAWED_BACK_MICROS.inc(amount)
    return Outcome("refunded", org_id, amount, posted.transaction_id)
