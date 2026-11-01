"""Per-org per-minute usage batches: exact sums, idempotency, tenant isolation."""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest
from billing_fixtures import ledger_is_balanced, make_event

from loom_billing.batch import batch_org, run_batch
from loom_billing.partitions import ensure_partitions
from loom_platform.ledger import REVENUE, org_balance, org_credit_account, usage_batch_key
from loom_platform.usage import insert_usage_events

pytestmark = pytest.mark.postgres

MINUTE = dt.datetime(2026, 11, 1, 12, 34, 56, tzinfo=dt.UTC)


async def store(pool, events) -> None:
    async with pool.connection() as conn, conn.transaction():
        await insert_usage_events(conn, events)


async def test_batch_sum_equals_usage_events_exactly(pool, make_tenant):
    t = await make_tenant(credit_micros=50_000_000)
    # Odd token counts so each cost is a rounded-up fraction; plus a zero-cost error.
    events = [make_event(t, prompt=37 * i + 1, completion=13 * i, cached=i) for i in range(1, 40)]
    events.append(make_event(t, prompt=0, completion=0, status="error"))
    await store(pool, events)

    run = await run_batch(pool, now=MINUTE)
    mine = [b for b in run.posted if b.org_id == t.org_id]
    assert len(mine) == 1
    batch = mine[0]
    expected = sum(e.cost_micros for e in events)
    assert batch.amount_micros == expected
    assert batch.events == 39  # the zero-cost event needs no posting

    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT kind, idempotency_key, org_id FROM ledger_transactions WHERE id = %s",
            (batch.transaction_id,),
        )
        assert await cur.fetchone() == ("usage", usage_batch_key(t.org_id, MINUTE), t.org_id)
        await cur.execute(
            "SELECT account_id, amount_micros FROM ledger_entries WHERE transaction_id = %s "
            "ORDER BY amount_micros",
            (batch.transaction_id,),
        )
        acct = await org_credit_account(conn, t.org_id)
        assert await cur.fetchall() == [(acct, -expected), (REVENUE, expected)]
        # Every covered event references the transaction; the sum over them matches.
        await cur.execute(
            "SELECT coalesce(sum(cost_micros), 0), count(*) FROM usage_events "
            "WHERE ledger_transaction_id = %s",
            (batch.transaction_id,),
        )
        assert await cur.fetchone() == (expected, 39)
        await cur.execute(
            "SELECT count(*) FROM usage_events WHERE org_id = %s AND ledger_transaction_id IS NULL",
            (t.org_id,),
        )
        assert await cur.fetchone() == (1,)
        assert await org_balance(conn, t.org_id) == 50_000_000 - expected
        assert await ledger_is_balanced(conn)


async def test_same_minute_twice_waits_for_next_minute(pool, make_tenant):
    t = await make_tenant()
    first = [make_event(t, prompt=1000) for _ in range(3)]
    await store(pool, first)
    run1 = await run_batch(pool, now=MINUTE)
    assert [b.amount_micros for b in run1.posted if b.org_id == t.org_id] == [
        sum(e.cost_micros for e in first)
    ]
    # Nothing new: no transaction, no error.
    again = await run_batch(pool, now=MINUTE)
    assert not [b for b in again.posted if b.org_id == t.org_id]

    later = [make_event(t, prompt=2000) for _ in range(2)]
    await store(pool, later)
    # A restart re-running the same minute must not post a second sum under the key.
    assert await batch_org(pool, t.org_id, MINUTE) == "minute_already_posted"
    run2 = await run_batch(pool, now=MINUTE + dt.timedelta(minutes=1))
    assert [b.amount_micros for b in run2.posted if b.org_id == t.org_id] == [
        sum(e.cost_micros for e in later)
    ]
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == -sum(e.cost_micros for e in first + later)
        assert await ledger_is_balanced(conn)


async def test_concurrent_batches_post_each_event_once(pool, make_tenant):
    t = await make_tenant()
    events = [make_event(t, prompt=100 + i) for i in range(50)]
    await store(pool, events)
    results = await asyncio.gather(*(batch_org(pool, t.org_id, MINUTE) for _ in range(6)))
    posted = [r for r in results if not isinstance(r, str)]
    assert len(posted) == 1
    assert posted[0].amount_micros == sum(e.cost_micros for e in events)
    assert all(
        r in ("nothing_unposted", "minute_already_posted", "idempotency_conflict")
        for r in results
        if isinstance(r, str)
    )
    async with pool.connection() as conn:
        assert await org_balance(conn, t.org_id) == -sum(e.cost_micros for e in events)


async def test_events_never_post_to_another_orgs_account(pool, make_tenant):
    a = await make_tenant(credit_micros=1_000_000)
    b = await make_tenant(credit_micros=2_000_000)
    ea = [make_event(a, prompt=1000 + i) for i in range(7)]
    eb = [make_event(b, prompt=5000 + i, completion=900) for i in range(4)]
    # Interleaved, one insert transaction.
    await store(pool, [x for pair in zip(ea, eb, strict=False) for x in pair] + ea[4:])

    await run_batch(pool, now=MINUTE)
    async with pool.connection() as conn, conn.cursor() as cur:
        # Each event's transaction belongs to the event's own org...
        await cur.execute(
            "SELECT count(*) FROM usage_events u JOIN ledger_transactions t "
            "ON t.id = u.ledger_transaction_id WHERE u.org_id IN (%s, %s) "
            "AND t.org_id <> u.org_id",
            (a.org_id, b.org_id),
        )
        assert await cur.fetchone() == (0,)
        # ...and its entries touch only that org's credit account and revenue.
        await cur.execute(
            "SELECT t.org_id, e.account_id, la.org_id, la.kind FROM ledger_transactions t "
            "JOIN ledger_entries e ON e.transaction_id = t.id "
            "JOIN ledger_accounts la ON la.id = e.account_id "
            "WHERE t.kind = 'usage' AND t.org_id IN (%s, %s)",
            (a.org_id, b.org_id),
        )
        for tx_org, _acct, acct_org, kind in await cur.fetchall():
            assert (kind == "revenue" and acct_org is None) or (
                kind == "org_credit" and acct_org == tx_org
            )
        assert await org_balance(conn, a.org_id) == 1_000_000 - sum(e.cost_micros for e in ea)
        assert await org_balance(conn, b.org_id) == 2_000_000 - sum(e.cost_micros for e in eb)
        assert await ledger_is_balanced(conn)


async def test_partitions_created_ahead_and_idempotent(pool):
    start = dt.date(2091, 1, 15)
    assert await ensure_partitions(pool, months=3, today=start) == 3
    assert await ensure_partitions(pool, months=3, today=start) == 0
    assert await ensure_partitions(pool, months=4, today=start) == 1
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT count(*) FROM pg_class WHERE relkind = 'r' "
            "AND relname LIKE 'usage_events_2091_0%%'"
        )
        assert await cur.fetchone() == (4,)
