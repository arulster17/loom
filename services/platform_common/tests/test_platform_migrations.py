"""Schema tests against a real Postgres: migrations, ledger invariants, append-only."""

import asyncio
import datetime as dt
import json
import uuid

import psycopg
import pytest
from psycopg import errors, sql

from loom_platform.db import connect
from loom_platform.ledger import PROMO_EXPENSE, REVENUE
from loom_platform.migrate import MigrationError, load_migrations, upgrade
from loom_platform.testing import create_api_key, create_org, create_user

pytestmark = pytest.mark.postgres


def test_upgrade_is_idempotent(pg_url):
    assert upgrade(pg_url) == []
    with psycopg.connect(pg_url) as conn:
        rows = conn.execute("SELECT version FROM schema_migrations ORDER BY 1").fetchall()
    assert [r[0] for r in rows] == [m.version for m in load_migrations()]


def test_edited_migration_is_refused(pg_url):
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute("UPDATE schema_migrations SET checksum = 'x' WHERE version = 1")
    try:
        with pytest.raises(MigrationError, match="changed after it was applied"):
            upgrade(pg_url)
    finally:
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute(
                "UPDATE schema_migrations SET checksum = %s WHERE version = 1",
                (load_migrations()[0].checksum,),
            )


def test_every_planned_table_exists(pg_url):
    with psycopg.connect(pg_url) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }
        partitioned = conn.execute(
            "SELECT relkind FROM pg_class WHERE relname = 'usage_events'"
        ).fetchone()
    for t in (
        "users", "accounts", "sessions", "verification_token", "orgs", "org_members",
        "api_keys", "ledger_accounts", "ledger_transactions", "ledger_entries",
        "stripe_events", "auto_recharge", "admin_audit_log",
    ):  # fmt: skip
        assert t in tables, t
    assert partitioned == ("p",)
    this_month = dt.datetime.now(dt.UTC).strftime("usage_events_%Y_%m")
    assert this_month in tables


def test_authjs_adapter_columns(pg_url):
    """The exact columns @auth/pg-adapter reads and writes."""
    want = {
        "users": {"id", "name", "email", "emailVerified", "image"},
        "accounts": {
            "id", "userId", "type", "provider", "providerAccountId", "refresh_token",
            "access_token", "expires_at", "id_token", "scope", "session_state", "token_type",
        },
        "sessions": {"id", "userId", "expires", "sessionToken"},
        "verification_token": {"identifier", "expires", "token"},
    }  # fmt: skip
    with psycopg.connect(pg_url) as conn:
        for table, cols in want.items():
            have = {
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                    (table,),
                ).fetchall()
            }
            assert cols <= have, (table, cols - have)
        # The adapter's own insert statements, with ids from the column defaults.
        user_id = conn.execute(
            'INSERT INTO users (name, email, "emailVerified", image) VALUES (%s, %s, %s, %s) '
            'RETURNING id, name, email, "emailVerified", image',
            ("A", f"{uuid.uuid4().hex}@x.test", None, None),
        ).fetchone()[0]
        conn.execute(
            'INSERT INTO accounts ("userId", provider, type, "providerAccountId", '
            "access_token, expires_at, refresh_token, id_token, scope, session_state, "
            "token_type) VALUES (%s, 'google', 'oidc', %s, NULL, 1, NULL, NULL, NULL, "
            "NULL, NULL)",
            (str(user_id), uuid.uuid4().hex),
        )
        conn.execute(
            'INSERT INTO sessions ("userId", expires, "sessionToken") VALUES (%s, now(), %s)',
            (str(user_id), uuid.uuid4().hex),
        )
        role = conn.execute("SELECT role FROM users WHERE id = %s", (user_id,)).fetchone()[0]
        assert role == "user"
        conn.rollback()


async def test_new_org_is_pending_and_has_credit_account(pg):
    org = await create_org(pg)
    row = await (
        await pg.execute(
            "SELECT o.status, a.kind FROM orgs o JOIN ledger_accounts a ON a.org_id = o.id "
            "WHERE o.id = %s",
            (org,),
        )
    ).fetchone()
    assert row == ("pending_approval", "org_credit")


async def _balanced_txn(conn, org, amount=1_000, key=None):
    acct = (await (await conn.execute("SELECT loom_org_credit_account(%s)", (org,))).fetchone())[0]
    txn = (
        await (
            await conn.execute(
                "INSERT INTO ledger_transactions (kind, idempotency_key, org_id) "
                "VALUES ('adjustment', %s, %s) RETURNING id",
                (key or uuid.uuid4().hex, org),
            )
        ).fetchone()
    )[0]
    return acct, txn


async def test_unbalanced_transaction_rejected_at_commit(pg):
    org = await create_org(pg)
    await pg.commit()
    acct, txn = await _balanced_txn(pg, org)
    await pg.execute(
        "INSERT INTO ledger_entries (transaction_id, account_id, amount_micros) "
        "VALUES (%s, %s, 1000), (%s, %s, -999)",
        (txn, acct, txn, PROMO_EXPENSE),
    )
    with pytest.raises(errors.CheckViolation, match="unbalanced"):
        await pg.commit()


async def test_balanced_transaction_commits(pg):
    org = await create_org(pg)
    acct, txn = await _balanced_txn(pg, org)
    await pg.execute(
        "INSERT INTO ledger_entries (transaction_id, account_id, amount_micros) "
        "VALUES (%s, %s, 1000), (%s, %s, -1000)",
        (txn, acct, txn, PROMO_EXPENSE),
    )
    await pg.commit()
    bal = (await (await pg.execute("SELECT loom_org_balance(%s)", (org,))).fetchone())[0]
    assert bal == 1000


async def test_transaction_without_entries_rejected(pg):
    org = await create_org(pg)
    await pg.commit()
    await _balanced_txn(pg, org)
    with pytest.raises(errors.CheckViolation, match="fewer than 2 entries"):
        await pg.commit()


async def test_single_entry_or_zero_amount_rejected(pg):
    org = await create_org(pg)
    await pg.commit()
    acct, txn = await _balanced_txn(pg, org)
    with pytest.raises(errors.CheckViolation):
        await pg.execute(
            "INSERT INTO ledger_entries (transaction_id, account_id, amount_micros) "
            "VALUES (%s, %s, 0)",
            (txn, acct),
        )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE ledger_entries SET amount_micros = amount_micros + 1",
        "DELETE FROM ledger_entries",
        "TRUNCATE ledger_entries",
        "UPDATE ledger_transactions SET memo = 'x'",
        "DELETE FROM ledger_transactions",
        "TRUNCATE ledger_transactions CASCADE",
        "UPDATE admin_audit_log SET action = 'x'",
        "DELETE FROM admin_audit_log",
        "TRUNCATE admin_audit_log",
    ],
)
async def test_append_only_tables(pg, statement):
    org = await create_org(pg)
    admin = await create_user(pg, role="admin")
    await pg.execute(
        "SELECT loom_post_transaction('promo', %s, %s, NULL, NULL, "
        "ARRAY[loom_org_credit_account(%s), %s]::uuid[], ARRAY[5, -5]::bigint[])",
        (uuid.uuid4().hex, org, org, PROMO_EXPENSE),
    )
    await pg.execute(
        "INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id) "
        "VALUES (%s, 'test', 'org', %s)",
        (admin, str(org)),
    )
    await pg.commit()
    with pytest.raises(errors.InsufficientPrivilege, match="append-only"):
        await pg.execute(statement)


async def test_idempotency_key_is_unique(pg):
    org = await create_org(pg)
    await pg.commit()
    key = f"test:{uuid.uuid4()}"
    await pg.execute(
        "INSERT INTO ledger_transactions (kind, idempotency_key, org_id) "
        "VALUES ('adjustment', %s, %s)",
        (key, org),
    )
    with pytest.raises(errors.UniqueViolation):
        await pg.execute(
            "INSERT INTO ledger_transactions (kind, idempotency_key, org_id) "
            "VALUES ('purchase', %s, %s)",
            (key, org),
        )


async def test_usage_events_dedupe_and_partition(pg):
    org = await create_org(pg, status="active")
    key_id, _ = await create_api_key(pg, org)
    rid = uuid.uuid4()
    now = dt.datetime.now(dt.UTC)
    q = (
        "INSERT INTO usage_events (id, org_id, api_key_id, model_id, price_version, cloud, "
        "prompt_tokens, completion_tokens, latency_ms, status, cost_micros, created_at) "
        "VALUES (%s, %s, %s, 'qwen3-8b', '77000:77000:368000', 'local', 10, 5, 12, 'ok', 3, %s) "
        "ON CONFLICT (id, created_at) DO NOTHING"
    )
    cur = await pg.execute(q, (rid, org, key_id, now))
    assert cur.rowcount == 1
    cur = await pg.execute(q, (rid, org, key_id, now))
    assert cur.rowcount == 0
    total = (await (await pg.execute("SELECT loom_org_usage_total(%s)", (org,))).fetchone())[0]
    assert total == 3
    # No partition far in the future: the insert fails loudly instead of misfiling.
    with pytest.raises(errors.CheckViolation):
        await pg.execute(q, (uuid.uuid4(), org, key_id, now + dt.timedelta(days=3650)))


async def test_ensure_partitions_is_idempotent(pg):
    made = (
        await (await pg.execute("SELECT loom_ensure_usage_partitions('2031-01-15', 2)")).fetchone()
    )[0]
    again = (
        await (await pg.execute("SELECT loom_ensure_usage_partitions('2031-01-01', 2)")).fetchone()
    )[0]
    assert (made, again) == (2, 0)


async def test_revocation_and_suspension_notify(pg_url):
    listener = await connect(pg_url, autocommit=True)
    writer = await connect(pg_url)
    try:
        await listener.execute("LISTEN loom_auth_changed")
        org = await create_org(writer, status="active")
        key_id, key = await create_api_key(writer, org)
        await writer.commit()
        await writer.execute("UPDATE api_keys SET revoked_at = now() WHERE id = %s", (key_id,))
        await writer.execute("UPDATE orgs SET status = 'suspended' WHERE id = %s", (org,))
        await writer.commit()
        got = []
        gen = listener.notifies(timeout=5)
        async for n in gen:
            got.append(json.loads(n.payload))
            if len(got) == 2:
                break
        assert got[0] == {
            "kind": "api_key",
            "org_id": str(org),
            "key_id": str(key_id),
            "key_hash": key.key_hash.hex(),
        }
        assert got[1] == {"kind": "org", "org_id": str(org), "status": "suspended"}
    finally:
        await listener.close()
        await writer.close()


async def test_concurrent_posts_of_one_key_write_once(pg_url):
    """Two connections race to post the same idempotency key: one transaction results."""
    setup = await connect(pg_url)
    org = await create_org(setup, status="active")
    await setup.commit()
    await setup.close()
    key = f"race:{uuid.uuid4()}"

    async def post():
        conn = await connect(pg_url)
        try:
            row = await (
                await conn.execute(
                    "SELECT created FROM loom_post_transaction('adjustment', %s, %s, NULL, "
                    "NULL, ARRAY[loom_org_credit_account(%s), %s]::uuid[], "
                    "ARRAY[7, -7]::bigint[])",
                    (key, org, org, REVENUE),
                )
            ).fetchone()
            await conn.commit()
            return row[0]
        finally:
            await conn.close()

    created = await asyncio.gather(*(post() for _ in range(8)))
    assert sorted(created) == [False] * 7 + [True]
    check = await connect(pg_url)
    try:
        n = (
            await (
                await check.execute(
                    sql.SQL("SELECT count(*) FROM ledger_transactions WHERE idempotency_key = %s"),
                    (key,),
                )
            ).fetchone()
        )[0]
        assert n == 1
        assert (await (await check.execute("SELECT loom_org_balance(%s)", (org,))).fetchone())[
            0
        ] == 7
    finally:
        await check.close()
