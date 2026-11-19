"""`loom-billing stripe-event list|resolve` (docs/phase1-m3-spec.md §8.6).

`resolve` closes a stuck event by hand. In one transaction, under the webhook's lock and
statement timeouts, it checks the admin (`users.role = 'admin'`), locks the event row
`FOR UPDATE`, refuses an unknown or already processed event and, for a clawback, one whose
PaymentIntent is not a non-empty string or already has a `stripe_payments` row (only a
release that fixes the handler applies such a clawback), and otherwise sets `processed_at`
and `result = 'resolved_by_admin'` and writes `admin_audit_log` (`stripe_event.resolve`, the
reason in `details`). It **posts nothing, ever**. A clawback resolved while its payment had
no row is reopened by `retry_unapplied`'s sweep once that payment is credited.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from loom_billing.logs import fields
from loom_billing.settings import BillingSettings
from loom_billing.webhooks import CLAWBACK_TYPES, Outcome, timed_transaction

log = logging.getLogger("loom_billing.resolve")

LIST_COLUMNS = (
    "id",
    "type",
    "received_at",
    "processed_at",
    "result",
    "last_error",
    "error_count",
    "stale_reported_at",
)


class ResolveRefused(Exception):
    """The resolve was refused and nothing changed; `str(e)` is the reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def resolve_event(
    pool: AsyncConnectionPool,
    event_id: str,
    *,
    admin_email: str,
    reason: str,
    settings: BillingSettings | None = None,
    now: dt.datetime | None = None,
) -> Outcome:
    """Resolve one stuck event; raises `ResolveRefused` (nothing changed) when refused."""
    s = settings if settings is not None else BillingSettings.model_construct()
    at = now if now is not None else dt.datetime.now(dt.UTC)
    async with timed_transaction(pool, s) as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT id FROM users WHERE email = %s AND role = 'admin'", (admin_email,)
        )
        admin = await cur.fetchone()
        if admin is None:
            raise ResolveRefused(f"{admin_email!r} is not an admin")
        await cur.execute(
            "SELECT type, payload, processed_at FROM stripe_events WHERE id = %s FOR UPDATE",
            (event_id,),
        )
        row = await cur.fetchone()
        if row is None:
            raise ResolveRefused(f"no stored event {event_id!r}")
        typ, payload, processed_at = row
        if processed_at is not None:
            raise ResolveRefused(f"event {event_id!r} is already processed")
        if typ in CLAWBACK_TYPES:
            payment_intent = _payment_intent(payload)
            if payment_intent is None:
                raise ResolveRefused(
                    f"event {event_id!r} has no PaymentIntent the handler reads "
                    "(a payload shape fixed by a release, like a poison event)"
                )
            await cur.execute(
                "SELECT 1 FROM stripe_payments WHERE payment_intent = %s", (payment_intent,)
            )
            if await cur.fetchone() is not None:
                raise ResolveRefused(
                    f"payment {payment_intent!r} is credited: only a release that fixes "
                    "the handler applies this clawback"
                )
        await cur.execute(
            "UPDATE stripe_events SET processed_at = %s, result = 'resolved_by_admin' "
            "WHERE id = %s",
            (at, event_id),
        )
        await cur.execute(
            "INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id, "
            "details) VALUES (%s, 'stripe_event.resolve', 'stripe_event', %s, %s)",
            (admin[0], event_id, Jsonb({"reason": reason, "event_type": typ})),
        )
    log.warning(
        "stripe.event_resolved",
        extra=fields(event_id=event_id, type=typ, admin=admin_email, reason=reason),
    )
    return Outcome("resolved_by_admin")


def _payment_intent(payload: Any) -> str | None:
    obj = ((payload or {}).get("data") or {}).get("object") if isinstance(payload, dict) else None
    pi = obj.get("payment_intent") if isinstance(obj, dict) else None
    return pi if isinstance(pi, str) and pi else None


async def list_events(
    pool: AsyncConnectionPool, *, unapplied: bool = False
) -> list[dict[str, Any]]:
    """Stored events, newest first (`received_at DESC, id`); only unprocessed ones with
    `unapplied`."""
    where = "WHERE processed_at IS NULL " if unapplied else ""
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            f"SELECT {', '.join(LIST_COLUMNS)} FROM stripe_events {where}"
            "ORDER BY received_at DESC, id"
        )
        rows = await cur.fetchall()
    return [dict(zip(LIST_COLUMNS, r, strict=True)) for r in rows]
