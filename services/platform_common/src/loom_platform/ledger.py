"""Ledger primitives over the SQL functions in migrations 0003 and 0005.

The posting logic lives in Postgres (`loom_post_transaction`, `loom_approve_org`) so the
web app (TypeScript) and the Python services post through the same code. These wrappers
run in the caller's transaction; the zero-sum check fires at its commit.

Idempotency key formats (one transaction per key, ever):
    usage:<org_id>:<YYYY-MM-DDTHH:MMZ>   per-org per-minute usage batch (billing worker)
    promo:signup:<org_id>                signup credit (loom_approve_org)
    stripe:<stripe event id>             purchase / refund from a Stripe webhook
    admin:<admin_audit_log id>           admin adjustment, keyed by its audit row
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, LiteralString

import psycopg
from psycopg import errors

REVENUE = uuid.UUID("00000000-0000-0000-0000-000000000001")
STRIPE_CLEARING = uuid.UUID("00000000-0000-0000-0000-000000000002")
PROMO_EXPENSE = uuid.UUID("00000000-0000-0000-0000-000000000003")
SYSTEM_ACCOUNTS = {
    "revenue": REVENUE,
    "stripe_clearing": STRIPE_CLEARING,
    "promo_expense": PROMO_EXPENSE,
}

TransactionKind = Literal["purchase", "usage", "refund", "promo", "adjustment"]


class LedgerError(Exception):
    pass


class UnbalancedTransaction(LedgerError):
    pass


class IdempotencyConflict(LedgerError):
    """The idempotency key was already posted with different content."""


class ForeignAccount(LedgerError):
    """An entry names an unknown account or another org's account."""


@dataclass(frozen=True)
class Entry:
    account_id: uuid.UUID
    amount_micros: int


@dataclass(frozen=True)
class Posted:
    transaction_id: uuid.UUID
    created: bool  # False: the key was posted before; nothing new was written


def usage_batch_key(org_id: uuid.UUID, minute: dt.datetime) -> str:
    """Idempotency key of an org's usage batch for the minute containing `minute`."""
    m = minute.astimezone(dt.UTC).replace(second=0, microsecond=0)
    return f"usage:{org_id}:{m.strftime('%Y-%m-%dT%H:%MZ')}"


async def post_transaction(
    conn: psycopg.AsyncConnection,
    *,
    kind: TransactionKind,
    idempotency_key: str,
    org_id: uuid.UUID | None,
    entries: Sequence[Entry],
    external_ref: str | None = None,
    memo: str | None = None,
) -> Posted:
    """Post a balanced transaction once. Raises UnbalancedTransaction,
    IdempotencyConflict or ForeignAccount; the caller's transaction is then unusable
    until rolled back."""
    if sum(e.amount_micros for e in entries) != 0:
        raise UnbalancedTransaction(f"entries sum to {sum(e.amount_micros for e in entries)}")
    if len(entries) < 2 or any(e.amount_micros == 0 for e in entries):
        raise UnbalancedTransaction("need at least 2 non-zero entries")
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT transaction_id, created FROM loom_post_transaction("
                "%s, %s, %s, %s, %s, %s::uuid[], %s::bigint[])",
                (
                    kind,
                    idempotency_key,
                    org_id,
                    external_ref,
                    memo,
                    [e.account_id for e in entries],
                    [e.amount_micros for e in entries],
                ),
            )
            row = await cur.fetchone()
    except errors.UniqueViolation as e:
        raise IdempotencyConflict(str(e)) from e
    except errors.CheckViolation as e:
        raise UnbalancedTransaction(str(e)) from e
    except errors.ForeignKeyViolation as e:
        raise ForeignAccount(str(e)) from e
    assert row is not None
    return Posted(row[0], row[1])


async def org_credit_account(conn: psycopg.AsyncConnection, org_id: uuid.UUID) -> uuid.UUID:
    return await _scalar(conn, "SELECT loom_org_credit_account(%s)", org_id)


async def org_balance(conn: psycopg.AsyncConnection, org_id: uuid.UUID) -> int:
    """Posted ledger balance of the org's credit account."""
    return await _scalar(conn, "SELECT loom_org_balance(%s)", org_id)


async def org_credit_total(conn: psycopg.AsyncConnection, org_id: uuid.UUID) -> int:
    return await _scalar(conn, "SELECT loom_org_credit_total(%s)", org_id)


async def org_usage_total(conn: psycopg.AsyncConnection, org_id: uuid.UUID) -> int:
    return await _scalar(conn, "SELECT loom_org_usage_total(%s)", org_id)


async def approve_org(
    conn: psycopg.AsyncConnection,
    org_id: uuid.UUID,
    admin_user_id: uuid.UUID,
    promo_micros: int,
) -> uuid.UUID | None:
    """Activate a pending org, post its signup credit, audit it. Returns the promo
    transaction id (None when promo_micros is 0)."""
    return await _scalar(
        conn, "SELECT loom_approve_org(%s, %s, %s)", org_id, admin_user_id, promo_micros
    )


async def _scalar(conn: psycopg.AsyncConnection, query: LiteralString, *params: object) -> Any:
    async with conn.cursor() as cur:
        await cur.execute(query, params)
        row = await cur.fetchone()
    assert row is not None
    return row[0]
