"""Ledger posting primitives and the signup approval flow, against Postgres."""

import datetime as dt
import uuid

import pytest
from psycopg import errors

from loom_platform.ledger import (
    PROMO_EXPENSE,
    REVENUE,
    STRIPE_CLEARING,
    Entry,
    ForeignAccount,
    IdempotencyConflict,
    UnbalancedTransaction,
    approve_org,
    org_balance,
    org_credit_account,
    org_credit_total,
    org_usage_total,
    post_transaction,
    usage_batch_key,
)
from loom_platform.pricing import Price, cost_micros
from loom_platform.testing import create_api_key, create_org, create_user
from loom_platform.usage import UsageEvent, insert_usage_events

pytestmark = pytest.mark.postgres


async def test_post_is_idempotent(pg):
    org = await create_org(pg, status="active")
    acct = await org_credit_account(pg, org)
    entries = [Entry(acct, 2_000_000), Entry(STRIPE_CLEARING, -2_000_000)]
    first = await post_transaction(
        pg, kind="purchase", idempotency_key=f"stripe:{uuid.uuid4()}", org_id=org,
        entries=entries, external_ref="cs_test_1",
    )  # fmt: skip
    again = await post_transaction(
        pg, kind="purchase", idempotency_key=f"stripe:{uuid.uuid4()}", org_id=org,
        entries=entries,
    )  # fmt: skip
    assert first.created and again.created  # different keys: two purchases
    key = f"stripe:{uuid.uuid4()}"
    a = await post_transaction(
        pg, kind="purchase", idempotency_key=key, org_id=org, entries=entries
    )
    b = await post_transaction(
        pg, kind="purchase", idempotency_key=key, org_id=org, entries=list(reversed(entries))
    )
    assert a.created and not b.created and a.transaction_id == b.transaction_id
    await pg.commit()
    assert await org_balance(pg, org) == 6_000_000


async def test_same_key_other_content_conflicts(pg):
    org = await create_org(pg, status="active")
    acct = await org_credit_account(pg, org)
    key = f"stripe:{uuid.uuid4()}"
    await post_transaction(
        pg, kind="purchase", idempotency_key=key, org_id=org,
        entries=[Entry(acct, 100), Entry(STRIPE_CLEARING, -100)],
    )  # fmt: skip
    with pytest.raises(IdempotencyConflict):
        await post_transaction(
            pg, kind="purchase", idempotency_key=key, org_id=org,
            entries=[Entry(acct, 200), Entry(STRIPE_CLEARING, -200)],
        )  # fmt: skip


async def test_unbalanced_refused_before_and_in_sql(pg):
    org = await create_org(pg, status="active")
    acct = await org_credit_account(pg, org)
    with pytest.raises(UnbalancedTransaction):
        await post_transaction(
            pg, kind="adjustment", idempotency_key="x", org_id=org,
            entries=[Entry(acct, 5), Entry(REVENUE, -4)],
        )  # fmt: skip
    # The SQL function refuses it on its own too (for the web app, which calls it directly).
    with pytest.raises(errors.CheckViolation):
        await pg.execute(
            "SELECT loom_post_transaction('adjustment', 'y', %s, NULL, NULL, %s::uuid[], "
            "%s::bigint[])",
            (org, [acct, REVENUE], [5, -4]),
        )


async def test_cannot_post_to_another_orgs_account(pg):
    mine = await create_org(pg, status="active")
    theirs = await create_org(pg, status="active")
    their_acct = await org_credit_account(pg, theirs)
    with pytest.raises(ForeignAccount):
        await post_transaction(
            pg, kind="adjustment", idempotency_key=f"t:{uuid.uuid4()}", org_id=mine,
            entries=[Entry(their_acct, 10), Entry(PROMO_EXPENSE, -10)],
        )  # fmt: skip


async def test_signup_credit_only_on_approval(pg):
    admin = await create_user(pg, role="admin")
    owner = await create_user(pg)
    org = await create_org(pg, owner=owner)
    await pg.commit()
    assert await org_balance(pg, org) == 0

    # A non-admin cannot approve.
    with pytest.raises(errors.InsufficientPrivilege):
        await approve_org(pg, org, owner, 5_000_000)
    await pg.rollback()

    txn = await approve_org(pg, org, admin, 5_000_000)
    await pg.commit()
    assert txn is not None
    status = (await (await pg.execute("SELECT status FROM orgs WHERE id=%s", (org,))).fetchone())[0]
    assert status == "active"
    assert await org_balance(pg, org) == 5_000_000
    assert await org_credit_total(pg, org) == 5_000_000
    audit = await (
        await pg.execute(
            "SELECT actor_user_id, action, details->>'promo_micros' FROM admin_audit_log "
            "WHERE target_id = %s",
            (str(org),),
        )
    ).fetchall()
    assert audit == [(admin, "org.approve", "5000000")]

    # Approving again is refused; the credit cannot be posted twice.
    with pytest.raises(errors.ObjectNotInPrerequisiteState):
        await approve_org(pg, org, admin, 5_000_000)
    await pg.rollback()
    await pg.execute("UPDATE orgs SET status = 'pending_approval' WHERE id = %s", (org,))
    with pytest.raises(errors.UniqueViolation):  # same promo key, other content
        await approve_org(pg, org, admin, 7_000_000)
    await pg.rollback()
    await pg.execute("UPDATE orgs SET status = 'pending_approval' WHERE id = %s", (org,))
    again = await approve_org(pg, org, admin, 5_000_000)  # same content: no-op posting
    await pg.commit()
    assert again == txn
    assert await org_balance(pg, org) == 5_000_000


async def test_approve_without_credit(pg):
    admin = await create_user(pg, role="admin")
    org = await create_org(pg)
    assert await approve_org(pg, org, admin, 0) is None
    assert await org_balance(pg, org) == 0


def _event(org, key, *, prompt, completion, minute):
    price = Price(77_000, 77_000, 368_000)
    return UsageEvent(
        id=uuid.uuid4(), org_id=org, api_key_id=key, model_id="qwen3-8b",
        price_version=price.version, cloud="local", prompt_tokens=prompt,
        completion_tokens=completion, latency_ms=100, ttft_ms=20, status="ok",
        cost_micros=cost_micros(price, prompt, completion), created_at=minute,
    )  # fmt: skip


async def test_balance_identity_with_usage_batch(pg):
    """available = credit_total - usage_total, before and after the minute batch, and
    the posted balance catches up once the batch posts."""
    admin = await create_user(pg, role="admin")
    org = await create_org(pg)
    key, _ = await create_api_key(pg, org)
    await approve_org(pg, org, admin, 5_000_000)
    now = dt.datetime.now(dt.UTC)
    events = [_event(org, key, prompt=10_000, completion=2_000, minute=now) for _ in range(3)]
    assert await insert_usage_events(pg, events) == 3
    assert await insert_usage_events(pg, events) == 0  # redelivery
    used = sum(e.cost_micros for e in events)
    assert await org_usage_total(pg, org) == used
    credit = await org_credit_total(pg, org)
    assert credit - used == 5_000_000 - used
    assert await org_balance(pg, org) == 5_000_000  # not batched yet

    # The billing worker's per-minute batch, as the contract describes it.
    acct = await org_credit_account(pg, org)
    posted = await post_transaction(
        pg, kind="usage", idempotency_key=usage_batch_key(org, now), org_id=org,
        entries=[Entry(acct, -used), Entry(REVENUE, used)],
    )  # fmt: skip
    await pg.execute(
        "UPDATE usage_events SET ledger_transaction_id = %s "
        "WHERE org_id = %s AND ledger_transaction_id IS NULL AND cost_micros > 0",
        (posted.transaction_id, org),
    )
    await pg.commit()
    assert await org_balance(pg, org) == 5_000_000 - used
    assert await org_credit_total(pg, org) == credit  # usage never moves credit
    assert await org_usage_total(pg, org) == used


def test_usage_batch_key():
    org = uuid.UUID(int=1)
    t = dt.datetime(2026, 11, 1, 13, 7, 59, 999, tzinfo=dt.UTC)
    assert usage_batch_key(org, t) == f"usage:{org}:2026-11-01T13:07Z"
