"""Stripe webhook events -> ledger, idempotently (contracts §3: purchase keyed
`stripe:<event id>`; docs/phase1-m3-spec.md §8).

Entry points:

- `process_event(pool, event, ...)`: **owns the Postgres behaviour of a delivery** (§8.3): one
  transaction with `SET LOCAL` lock and statement timeouts, `handle_event`, COMMIT; anything
  raised by that part (a handler bug, a lock timeout, a lost connection, the COMMIT itself)
  rolls back and runs a **second transaction** that stores the event unapplied with
  `last_error` (or finds it processed meanwhile: `duplicate`); if that fails too it raises
  `StoreFailed`. Outcome metrics and S4 are counted only after a COMMIT. No exception class
  or SQLSTATE is inspected.
- `handle_event(conn, event, ...)`: the in-transaction core, never calls Stripe.
- `retry_unapplied(pool, ...)`: R, each minute: reopen sweep, newest-first re-run, stale
  marking.

One transaction per delivery:
1. `INSERT INTO stripe_events ... ON CONFLICT DO NOTHING`, then lock the row.
2. Already `processed_at`: a replay; nothing happens (`duplicate`).
3. Otherwise handle the type and write `result` (clearing `last_error` and `error_count`);
   `processed_at` is set unless the event could not be applied (unknown org, wrong
   currency): it is then stored unapplied for an admin to look at, logged at error, and
   Stripe still gets 200, since retrying would not fix it.

Purchases: `checkout.session.completed` with `payment_status = paid`, or
`checkout.session.async_payment_succeeded`, for a session this service created
(`metadata.purpose = loom_credits`). The org is `client_reference_id`; the amount is
what Stripe says was paid (`amount_total`, USD cents). A session is credited at most
once even across different event ids (completed, then async-succeeded): an advisory
lock on the session id, a look-up of an earlier purchase with that `external_ref`, and
migration 0012's unique index behind both. The session's PaymentIntent is recorded in
`stripe_payments` with the purchase.

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
in the same transaction, each in its own savepoint (one that raises keeps its error and
never blocks the credit), and `retry_unapplied` (billing worker, every minute) re-runs
unapplied events, which covers the race where both commit at once. Without both the
customer would be refunded and keep the credit. `loom_billing_stripe_unapplied_events`
counts what is still waiting.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from loom_billing import metrics
from loom_billing.logs import fields
from loom_billing.settings import MICROS_PER_CENT, BillingSettings
from loom_billing.stripe_api import PURPOSE
from loom_platform.ledger import STRIPE_CLEARING, Entry, org_credit_account, post_transaction

log = logging.getLogger("loom_billing.webhooks")

CREDIT_TYPES = {"checkout.session.completed", "checkout.session.async_payment_succeeded"}
CLAWBACK_TYPES = {"charge.refunded", "charge.dispute.created"}

# `settings=None` means the defaults, never the environment at call time (spec §8.11).
_DEFAULT_SETTINGS = BillingSettings.model_construct()
LAST_ERROR_MAX = 1000
REOPENED = "reopened: payment credited after resolve"


@dataclass(frozen=True)
class Outcome:
    # credited | refunded | duplicate | ignored | not_paid | already_credited |
    # already_refunded | unapplied | skipped (retry: a delivery holds the row) |
    # resolved_by_admin (resolve)
    result: str
    org_id: uuid.UUID | None = None
    amount_micros: int = 0
    transaction_id: uuid.UUID | None = None
    reason: str | None = None
    # The event was stored unapplied by the second transaction after an error.
    stored_after_error: bool = False
    # What a purchase applied late (its pending clawbacks) and how many of them failed for
    # the first time: counted by `process_event` after COMMIT, not by `handle_event`.
    late: tuple[Outcome, ...] = field(default=(), compare=False, repr=False)
    pending_first_failures: int = field(default=0, compare=False, repr=False)

    @property
    def processed(self) -> bool:
        return self.result not in ("unapplied", "skipped")


class BadEvent(ValueError):
    pass


class StoreFailed(Exception):
    """The second transaction failed too (Postgres is not answering, or a payload `jsonb`
    refuses): nothing was stored. `__cause__` is the original exception."""


def check_event_shape(event: Any) -> tuple[str, str]:
    if not isinstance(event, dict):
        raise BadEvent("event is not an object")
    evt_id, typ = event.get("id"), event.get("type")
    if not isinstance(evt_id, str) or not evt_id or not isinstance(typ, str):
        raise BadEvent("event without id/type")
    if event.get("livemode") is True:
        raise BadEvent("live-mode event refused (test mode only)")
    return evt_id, typ


def _settings(settings: BillingSettings | None) -> BillingSettings:
    return settings if settings is not None else _DEFAULT_SETTINGS


def _now(now: dt.datetime | None) -> dt.datetime:
    return now if now is not None else dt.datetime.now(dt.UTC)


def last_error_text(exc: BaseException) -> str:
    """`<exception class>: <message>`, NUL characters replaced (a `text` column refuses
    them), cut to 1,000 characters."""
    return f"{type(exc).__name__}: {exc}".replace("\x00", "")[:LAST_ERROR_MAX]


async def set_timeouts(conn: AsyncConnection, settings: BillingSettings) -> None:
    """`SET LOCAL lock_timeout` / `statement_timeout` for the current transaction (milliseconds)."""
    await conn.execute(
        "SELECT set_config('lock_timeout', %s, true), set_config('statement_timeout', %s, true)",
        (str(settings.stripe_lock_timeout_ms), str(settings.stripe_statement_timeout_ms)),
    )


@asynccontextmanager
async def timed_transaction(
    pool: AsyncConnectionPool, settings: BillingSettings
) -> AsyncIterator[AsyncConnection]:
    """A pooled connection (the pool wait bounded by `stripe_webhook_db_timeout_s`) in one
    transaction under the lock and statement timeouts: commit on success, roll back on error."""
    async with (
        pool.connection(timeout=settings.stripe_webhook_db_timeout_s) as conn,
        conn.transaction(),
    ):
        await set_timeouts(conn, settings)
        yield conn


# ---------------------------------------------------------------------- the delivery


async def process_event(
    pool: AsyncConnectionPool,
    event: dict[str, Any],
    *,
    retry: bool = False,
    settings: BillingSettings | None = None,
    now: dt.datetime | None = None,
) -> Outcome:
    """One event through §8.3's Postgres behaviour. Raises `BadEvent` (before any
    transaction) or `StoreFailed`; returns `Outcome("unapplied", stored_after_error=True)`
    when the second transaction stored the event after an error, `Outcome("duplicate")` when
    its guard found the event processed meanwhile, `Outcome("skipped")` when `retry` and a
    delivery holds the row."""
    evt_id, typ = check_event_shape(event)
    s = _settings(settings)
    at = _now(now)
    try:
        async with timed_transaction(pool, s) as conn:
            outcome = await handle_event(conn, event, retry=retry, settings=s, now=at)
    except Exception as e:
        return await _store_error(pool, event, evt_id, typ, e, retry=retry, settings=s, now=at)
    _count(outcome, retry=retry)
    return outcome


def _count(outcome: Outcome, *, retry: bool) -> None:
    """Outcome metrics, after COMMIT (a rolled-back attempt is never counted)."""
    if outcome.result == "skipped" or (retry and outcome.result in ("duplicate", "unapplied")):
        return
    metrics.WEBHOOKS.labels(outcome.result).inc()
    if outcome.result == "credited":
        metrics.CREDITED_MICROS.inc(outcome.amount_micros)
    elif outcome.result == "refunded":
        metrics.CLAWED_BACK_MICROS.inc(outcome.amount_micros)
    for late in outcome.late:
        metrics.WEBHOOKS.labels(f"late_{late.result}").inc()
        if late.result == "refunded":
            metrics.CLAWED_BACK_MICROS.inc(late.amount_micros)
    if outcome.pending_first_failures:
        metrics.HANDLER_FAILURES.labels("pending_clawback").inc(outcome.pending_first_failures)


async def _store_error(
    pool: AsyncConnectionPool,
    event: dict[str, Any],
    evt_id: str,
    typ: str,
    error: Exception,
    *,
    retry: bool,
    settings: BillingSettings,
    now: dt.datetime,
) -> Outcome:
    """The second transaction (§8.3): a new transaction from the pool, under the same
    timeouts, that stores the event unapplied with its error, never touching a row another
    delivery processed meanwhile."""
    text = last_error_text(error)
    log.error(
        "stripe.event_failed",
        extra=fields(event_id=evt_id, type=typ, retry=retry, error=text),
    )
    try:
        async with timed_transaction(pool, settings) as conn, conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO stripe_events (id, type, payload, received_at, result, "
                "last_error, last_error_at, error_count) "
                "VALUES (%(id)s, %(type)s, %(payload)s, %(now)s, 'unapplied', "
                "%(error)s, %(now)s, 1) "
                "ON CONFLICT (id) DO UPDATE SET result = 'unapplied', "
                "last_error = EXCLUDED.last_error, last_error_at = %(now)s, "
                "error_count = stripe_events.error_count + 1 "
                "WHERE stripe_events.processed_at IS NULL RETURNING error_count",
                {"id": evt_id, "type": typ, "payload": Jsonb(event), "now": now, "error": text},
            )
            row = await cur.fetchone()
    except Exception as e:
        log.error("stripe.event_store_failed", extra=fields(event_id=evt_id, error=str(e)[:300]))
        raise StoreFailed(f"could not store event {evt_id}: {type(e).__name__}") from error
    if row is None:  # processed meanwhile (the COMMIT landed, or another delivery applied it)
        return Outcome("duplicate")
    if row[0] == 1:
        metrics.HANDLER_FAILURES.labels("retry" if retry else "webhook").inc()
    return Outcome("unapplied", reason=text, stored_after_error=True)


async def handle_event(
    conn: AsyncConnection,
    event: dict[str, Any],
    *,
    retry: bool = False,
    settings: BillingSettings | None = None,
    now: dt.datetime | None = None,
) -> Outcome:
    """Process one verified event in the caller's transaction; never calls Stripe, counts
    no outcome metric (`process_event` does, after COMMIT). `retry`: a re-run of a stored
    unapplied event (`retry_unapplied`): the row is taken `FOR UPDATE SKIP LOCKED` and
    `Outcome("skipped")` is returned when a delivery holds it; it stays quiet while still
    unapplied. Writes `result` and clears `last_error` and `error_count` for every outcome
    but `duplicate` and `skipped`."""
    del settings  # the timeouts are set by the caller's transaction (`set_timeouts`)
    evt_id, typ = check_event_shape(event)
    at = _now(now)
    async with conn.cursor() as cur:
        if retry:
            await cur.execute(
                "SELECT processed_at FROM stripe_events WHERE id = %s FOR UPDATE SKIP LOCKED",
                (evt_id,),
            )
        else:
            await cur.execute(
                "INSERT INTO stripe_events (id, type, payload, received_at) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                (evt_id, typ, Jsonb(event), at),
            )
            await cur.execute(
                "SELECT processed_at FROM stripe_events WHERE id = %s FOR UPDATE", (evt_id,)
            )
        row = await cur.fetchone()
    if row is None:
        return Outcome("skipped")  # retry only: a delivery holds the row (or it is gone)
    if row[0] is not None:
        outcome = Outcome("duplicate")
    elif typ in CREDIT_TYPES:
        outcome = await _credit_checkout(conn, evt_id, typ, event, at)
    elif typ in CLAWBACK_TYPES:
        outcome = await _claw_back(conn, evt_id, typ, event)
    else:
        outcome = Outcome("ignored")
    if outcome.result != "duplicate":
        await _write_result(conn, evt_id, outcome, at)
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
    return outcome


async def _write_result(
    conn: AsyncConnection, evt_id: str, outcome: Outcome, at: dt.datetime
) -> None:
    """The attempt's outcome on the event row: `result`, `processed_at` when it was
    applied, and a clean `last_error` / `error_count`."""
    async with conn.cursor() as cur:
        await cur.execute(
            "UPDATE stripe_events SET result = %s, processed_at = %s, last_error = NULL, "
            "error_count = 0 WHERE id = %s",
            (outcome.result, at if outcome.processed else None, evt_id),
        )


async def _credit_checkout(
    conn: AsyncConnection, evt_id: str, typ: str, event: dict[str, Any], at: dt.datetime
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
        # The unique index's predicate (migration 0012, L3): any org's purchase of the session.
        await cur.execute(
            "SELECT id FROM ledger_transactions WHERE kind = 'purchase' AND external_ref = %s",
            (session_id,),
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
    late: list[Outcome] = []
    first_failures = 0
    payment_intent = obj.get("payment_intent")
    if isinstance(payment_intent, str) and payment_intent:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO stripe_payments (payment_intent, session_id, org_id, "
                "purchase_transaction_id, amount_micros) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT DO NOTHING",
                (payment_intent, session_id, org_id, posted.transaction_id, amount),
            )
        late, first_failures = await _apply_pending_clawbacks(conn, payment_intent, at)
    return Outcome(
        "credited",
        org_id,
        amount,
        posted.transaction_id,
        late=tuple(late),
        pending_first_failures=first_failures,
    )


async def _apply_pending_clawbacks(
    conn: AsyncConnection, payment_intent: str, at: dt.datetime
) -> tuple[list[Outcome], int]:
    """Refunds and disputes of this PaymentIntent that arrived before its purchase was
    recorded: apply them now, in the purchase's transaction, each in its own plain
    savepoint. One that raises is rolled back alone and keeps its error (`result =
    'unapplied'`, `last_error`, `error_count + 1`) for `retry_unapplied`: it never blocks
    the credit of its payment. If recording the error fails too (the transaction itself is
    gone), the exception propagates and the whole delivery takes the 500/503 path.
    Returns the applied outcomes (for the `late_<result>` metric) and the number whose
    error count became 1 (S4 `pending_clawback`)."""
    async with conn.cursor() as cur:
        await cur.execute(
            "SELECT id, type, payload FROM stripe_events "
            "WHERE processed_at IS NULL AND type = ANY(%s) "
            "AND payload -> 'data' -> 'object' ->> 'payment_intent' = %s "
            "ORDER BY received_at, id FOR UPDATE",
            (sorted(CLAWBACK_TYPES), payment_intent),
        )
        pending = await cur.fetchall()
    applied: list[Outcome] = []
    first_failures = 0
    for evt_id, typ, payload in pending:
        try:
            async with conn.transaction():  # a plain savepoint, no SET CONSTRAINTS
                outcome = await _claw_back(conn, evt_id, typ, payload)
                await _write_result(conn, evt_id, outcome, at)
        except Exception as e:
            text = last_error_text(e)
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE stripe_events SET result = 'unapplied', last_error = %s, "
                    "last_error_at = %s, error_count = error_count + 1 WHERE id = %s "
                    "RETURNING error_count",
                    (text, at, evt_id),
                )
                row = await cur.fetchone()
            log.error(
                "stripe.pending_clawback_failed",
                extra=fields(event_id=evt_id, type=typ, error=text),
            )
            if row is not None and row[0] == 1:
                first_failures += 1
            continue
        applied.append(outcome)
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
    return applied, first_failures


# ---------------------------------------------------------------------- R


async def retry_unapplied(
    pool: AsyncConnectionPool,
    *,
    limit: int = 500,
    settings: BillingSettings | None = None,
    now: dt.datetime | None = None,
) -> int:
    """`retry_unapplied` (R, §8.3): 1. the reopen sweep; 2. re-run unapplied events newest
    first, up to `limit`, no age window, each in its own transaction through
    `process_event(retry=True)`; 3. stale marking. Returns how many events this run marked
    processed. Never raises: a failure outside a per-event attempt, or an attempt whose
    error store fails too (`StoreFailed`), ends the run there and leaves the last-success
    gauge (alarm S5) unset."""
    s = _settings(settings)
    at = _now(now)
    applied = 0
    try:
        await _reopen_sweep(pool, s)
        async with timed_transaction(pool, s) as conn, conn.cursor() as cur:
            await cur.execute(
                "SELECT payload FROM stripe_events WHERE processed_at IS NULL "
                "ORDER BY received_at DESC, id LIMIT %s",
                (limit,),
            )
            events = [r[0] for r in await cur.fetchall()]
        for event in events:
            try:
                outcome = await process_event(pool, event, retry=True, settings=s, now=at)
            except BadEvent:
                continue  # only a test can store one; it goes stale and an admin resolves it
            if outcome.processed and outcome.result != "duplicate":
                applied += 1
        await _mark_stale(pool, s, at)
    except Exception:
        log.exception("stripe.retry_run_ended")
        return applied
    metrics.STRIPE_RETRY_LAST_SUCCESS.set(at.timestamp())
    return applied


async def _reopen_sweep(pool: AsyncConnectionPool, s: BillingSettings) -> None:
    """Reopen every clawback resolved while its payment had no `stripe_payments` row whose
    payment now exists (§8.6). Skips rows another transaction holds; the next run takes them."""
    async with timed_transaction(pool, s) as conn, conn.cursor() as cur:
        await cur.execute(
            "UPDATE stripe_events SET processed_at = NULL, result = 'unapplied', "
            "stale_reported_at = NULL, last_error = %s WHERE id IN ("
            "SELECT e.id FROM stripe_events e JOIN stripe_payments p "
            "ON p.payment_intent = e.payload -> 'data' -> 'object' ->> 'payment_intent' "
            "WHERE e.result = 'resolved_by_admin' AND e.type = ANY(%s) "
            "ORDER BY e.received_at, e.id FOR UPDATE OF e SKIP LOCKED) RETURNING id",
            (REOPENED, sorted(CLAWBACK_TYPES)),
        )
        reopened = [r[0] for r in await cur.fetchall()]
    for evt_id in reopened:
        log.warning("stripe.event_reopened", extra=fields(event_id=evt_id))


async def _mark_stale(pool: AsyncConnectionPool, s: BillingSettings, at: dt.datetime) -> None:
    """Mark events unapplied for `stripe_stale_after_s` since `received_at` stale, once
    (set-based, in the purchase's lock order, never waiting), count each by type, and
    refresh the two gauges."""
    cutoff = at - dt.timedelta(seconds=s.stripe_stale_after_s)
    async with timed_transaction(pool, s) as conn, conn.cursor() as cur:
        await cur.execute(
            "UPDATE stripe_events SET stale_reported_at = %(now)s WHERE id IN ("
            "SELECT id FROM stripe_events WHERE processed_at IS NULL "
            "AND stale_reported_at IS NULL AND received_at <= %(cutoff)s "
            "ORDER BY received_at, id FOR UPDATE SKIP LOCKED) RETURNING type",
            {"now": at, "cutoff": cutoff},
        )
        types = [r[0] for r in await cur.fetchall()]
        await cur.execute(
            "SELECT count(*), count(stale_reported_at) FROM stripe_events "
            "WHERE processed_at IS NULL"
        )
        counts = await cur.fetchone()
    for typ in types:
        metrics.STRIPE_STALE.labels(typ if typ in metrics.SUBSCRIBED_TYPES else "other").inc()
    if counts is not None:
        metrics.STRIPE_UNAPPLIED.set(int(counts[0]))
        metrics.STRIPE_STALE_EVENTS.set(int(counts[1]))


# ---------------------------------------------------------------------- clawbacks


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
    return Outcome("refunded", org_id, amount, posted.transaction_id)


__all__ = [
    "CLAWBACK_TYPES",
    "CREDIT_TYPES",
    "BadEvent",
    "Outcome",
    "StoreFailed",
    "check_event_shape",
    "handle_event",
    "last_error_text",
    "process_event",
    "retry_unapplied",
    "set_timeouts",
]
