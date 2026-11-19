"""Migration 0012 (docs/phase1-m3-spec.md §8.4, §11, done-when A21): the `stripe_events`
bookkeeping columns, the unique purchase per Checkout Session
(`ledger_transactions_purchase_external_ref`), reached by callers as
`IdempotencyConflict` through `post_transaction` (`ledger.py` unchanged), the reopen
sweep's partial index, and the runtime role's grants on the new columns.

Fresh databases are made next to the session's and dropped afterwards (the pattern of
test_platform_migration_0010.py)."""

from __future__ import annotations

import contextlib
import json
import secrets
import time
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from platform_m3_roles import (
    APP_ROLE,
    admin_info,
    admin_url,
    app_password,
    create_database,
    drop_database,
    run_migrate,
    url_for,
)
from psycopg import errors, sql

from loom_platform import migrate
from loom_platform.db import connect
from loom_platform.ledger import (
    STRIPE_CLEARING,
    Entry,
    IdempotencyConflict,
    org_credit_account,
    post_transaction,
)
from loom_platform.migrate import load_migrations, upgrade
from loom_platform.testing import create_org

pytestmark = pytest.mark.postgres

INDEX = "ledger_transactions_purchase_external_ref"
# column -> accepted data types; each new column is nullable or defaulted (§8.4), so the
# running M1 code's inserts keep working.
NEW_COLUMNS: dict[str, tuple[str, ...]] = {
    "result": ("text",),
    "last_error": ("text",),
    "last_error_at": ("timestamp with time zone",),
    "stale_reported_at": ("timestamp with time zone",),
}
COUNT_TYPES = ("integer", "bigint", "smallint")


@contextlib.contextmanager
def scratch_database(pg_url: str, template: str | None = None) -> Iterator[str]:
    """A database next to the session's (empty, or a clone of `template`), dropped
    afterwards."""
    name = f"loom_m3s_0012_{uuid.uuid4().hex[:12]}"
    info = psycopg.conninfo.conninfo_to_dict(pg_url)
    admin = psycopg.conninfo.make_conninfo("", **{**info, "dbname": "postgres"})
    create = sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name))
    if template is not None:
        create = sql.SQL("{} TEMPLATE {}").format(create, sql.Identifier(template))
    with psycopg.connect(admin, autocommit=True) as conn:
        for attempt in range(50):
            try:
                conn.execute(create)
                break
            except errors.ObjectInUse:  # another clone being made from it
                if attempt == 49:
                    raise
                time.sleep(0.1)
    try:
        yield psycopg.conninfo.make_conninfo("", **{**info, "dbname": name})
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


def columns(url: str, table: str) -> dict[str, tuple[str, str, str | None]]:
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s",
            (table,),
        ).fetchall()
    return {r[0]: (r[1], r[2], r[3]) for r in rows}


def index_info(url: str, name: str) -> tuple[str, bool, str | None, str] | None:
    """(table, unique, predicate, definition) of an index, or None."""
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT t.relname, x.indisunique, pg_get_expr(x.indpred, x.indrelid), "
            "pg_get_indexdef(x.indexrelid) FROM pg_index x "
            "JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid "
            "WHERE i.relname = %s",
            (name,),
        ).fetchone()
    return None if row is None else (row[0], row[1], row[2], row[3])


def check_schema(url: str) -> None:
    have = columns(url, "stripe_events")
    for col, types in NEW_COLUMNS.items():
        assert col in have, f"stripe_events.{col} missing"
        dtype, nullable, default = have[col]
        assert dtype in types, (col, have[col])
        assert nullable == "YES" or default is not None, (col, "neither nullable nor defaulted")
    assert "error_count" in have, "stripe_events.error_count missing"
    dtype, _, default = have["error_count"]
    assert dtype in COUNT_TYPES, dtype
    assert default is not None and default.split("::")[0].strip("'() ") == "0", default

    info = index_info(url, INDEX)
    assert info is not None, f"index {INDEX} missing"
    table, unique, predicate, definition = info
    assert table == "ledger_transactions" and unique
    assert "external_ref" in definition
    assert predicate is not None
    assert "purchase" in predicate and "external_ref IS NOT NULL" in predicate, predicate

    # The reopen sweep's partial index (§8.4): on stripe_events, only resolved rows.
    with psycopg.connect(url) as conn:
        preds = [
            r[0]
            for r in conn.execute(
                "SELECT pg_get_expr(x.indpred, x.indrelid) || ' ' || "
                "pg_get_indexdef(x.indexrelid) FROM pg_index x "
                "WHERE x.indrelid = 'stripe_events'::regclass AND x.indpred IS NOT NULL"
            ).fetchall()
        ]
    assert any("resolved_by_admin" in p and "payment_intent" in p for p in preds), preds


def test_a21_0012_is_registered() -> None:
    versions = [m.version for m in load_migrations()]
    assert 12 in versions


def test_a21_applies_on_a_fresh_database(app_db) -> None:
    """The module's fresh database (`loom-migrate` on an empty one, the grants tests'
    too: one full migration serves both)."""
    owner, _ = app_db
    with psycopg.connect(owner) as conn:
        versions = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    assert 12 in versions
    check_schema(owner)
    assert upgrade(owner) == []


@pytest.fixture(scope="module")
def at_0011(pg_url: str) -> Iterator[str]:
    """Name of a database migrated up to 0011, once per module; each test works on a
    clone of it (`scratch_database(pg_url, at_0011)`), never on it."""
    every = load_migrations()
    assert [m.version for m in every][:11] == list(range(1, 12))
    with scratch_database(pg_url) as url:
        with pytest.MonkeyPatch.context() as m:
            m.setattr(migrate, "load_migrations", lambda: every[:11])
            assert [mig.version for mig in upgrade(url)] == list(range(1, 12))
        yield str(psycopg.conninfo.conninfo_to_dict(url)["dbname"])


def _purchase(conn: psycopg.Connection, org: uuid.UUID, ref: str | None, key: str) -> None:
    acct = conn.execute("SELECT loom_org_credit_account(%s)", (org,)).fetchone()
    assert acct is not None
    conn.execute(
        "SELECT loom_post_transaction('purchase', %s, %s, %s, 'test', %s::uuid[], %s::bigint[])",
        (key, org, ref, [acct[0], STRIPE_CLEARING], [1_000_000, -1_000_000]),
    )


# M1's columns of stripe_events (0011), in a fixed order.
M1_ROWS = "SELECT id, type, payload, received_at, processed_at FROM stripe_events ORDER BY id"


def test_a21_applies_on_a_database_at_0011_and_keeps_m1_rows(pg_url: str, at_0011: str) -> None:
    with scratch_database(pg_url, at_0011) as url:
        every = load_migrations()
        assert "result" not in columns(url, "stripe_events")
        unapplied = f"evt_test_{secrets.token_hex(8)}"
        processed = f"evt_test_{secrets.token_hex(8)}"
        with psycopg.connect(url) as conn:
            org = conn.execute(
                "INSERT INTO orgs (name, status) VALUES ('before 0012', 'active') RETURNING id"
            ).fetchone()[0]  # type: ignore[index]
            conn.execute(
                "INSERT INTO stripe_events (id, type, payload) VALUES (%s, %s, %s)",
                (unapplied, "charge.refunded", json.dumps({"id": unapplied})),
            )
            conn.execute(
                "INSERT INTO stripe_events (id, type, payload, processed_at) "
                "VALUES (%s, %s, %s, now())",
                (processed, "checkout.session.completed", json.dumps({"id": processed})),
            )
            _purchase(conn, org, "cs_test_before_0012", f"stripe:{processed}")
            conn.commit()
            m1_rows = conn.execute(M1_ROWS).fetchall()
        done = [m.version for m in upgrade(url)]
        assert done[:1] == [12]
        assert done == [m.version for m in every if m.version >= 12]
        check_schema(url)
        with psycopg.connect(url) as conn:
            # M1's rows are otherwise unchanged: the unapplied one still unprocessed, the
            # processed one still processed; the new counter starts at its default, 0.
            after = conn.execute(M1_ROWS).fetchall()
            assert after == m1_rows
            counts = conn.execute("SELECT id, error_count FROM stripe_events").fetchall()
            assert {r[1] for r in counts} == {0}, counts
            assert {r[0] for r in after if r[4] is None} == {unapplied}
            # The running M1 code's insert keeps working (new columns nullable/defaulted).
            conn.execute(
                "INSERT INTO stripe_events (id, type, payload) VALUES (%s, %s, %s)",
                (unapplied + "_m1", "charge.refunded", json.dumps({"id": unapplied})),
            )
            conn.commit()


def test_a21_refuses_a_database_holding_two_purchases_of_one_session(
    pg_url: str, at_0011: str
) -> None:
    with scratch_database(pg_url, at_0011) as url:
        with psycopg.connect(url) as conn:
            org = conn.execute(
                "INSERT INTO orgs (name, status) VALUES ('dup', 'active') RETURNING id"
            ).fetchone()[0]  # type: ignore[index]
            _purchase(conn, org, "cs_test_twice", "stripe:evt_a")
            _purchase(conn, org, "cs_test_twice", "stripe:evt_b")
            conn.commit()
        with pytest.raises(Exception):  # noqa: B017 - any failure stops the deploy
            upgrade(url)
        with psycopg.connect(url) as conn:
            versions = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        assert 12 not in versions
        assert index_info(url, INDEX) is None


async def _org(conn: psycopg.AsyncConnection) -> uuid.UUID:
    return await create_org(conn, status="active")


async def _post(conn: psycopg.AsyncConnection, org: uuid.UUID, kind: str, ref: str | None) -> None:
    acct = await org_credit_account(conn, org)
    sign = 1 if kind == "purchase" else -1
    await post_transaction(
        conn,
        kind=kind,  # type: ignore[arg-type]
        idempotency_key=f"stripe:evt_test_{secrets.token_hex(8)}",
        org_id=org,
        entries=[Entry(acct, sign * 1_000_000), Entry(STRIPE_CLEARING, -sign * 1_000_000)],
        external_ref=ref,
    )


async def test_a21_second_purchase_of_a_session_is_an_idempotency_conflict(pg_url: str) -> None:
    ref = f"cs_test_{secrets.token_hex(8)}"
    conn = await connect(pg_url)
    try:
        org = await _org(conn)
        await _post(conn, org, "purchase", ref)
        await conn.commit()
        with pytest.raises(IdempotencyConflict) as caught:
            await _post(conn, org, "purchase", ref)
        await conn.rollback()
        cause = caught.value.__cause__
        assert isinstance(cause, errors.UniqueViolation)
        assert cause.sqlstate == "23505"
        assert cause.diag.constraint_name == INDEX
        # Another org cannot take the same session either (the index has no org).
        other = await _org(conn)
        await conn.commit()
        with pytest.raises(IdempotencyConflict):
            await _post(conn, other, "purchase", ref)
        await conn.rollback()
    finally:
        await conn.rollback()
        await conn.close()


async def test_a21_index_is_partial(pg_url: str) -> None:
    """Only purchases with an external_ref are unique: refunds may share a ref (one per
    refund event of a PaymentIntent) and purchases without a ref are not limited."""
    ref = f"pi_test_{secrets.token_hex(8)}"
    conn = await connect(pg_url)
    try:
        org = await _org(conn)
        await _post(conn, org, "purchase", None)
        await _post(conn, org, "purchase", None)
        await _post(conn, org, "purchase", f"cs_test_{secrets.token_hex(8)}")
        await _post(conn, org, "refund", ref)
        await _post(conn, org, "refund", ref)
        await conn.commit()
        assert index_info(pg_url, INDEX) is not None
    finally:
        await conn.rollback()
        await conn.close()


# ------------------------------------------------------------ grants


@pytest.fixture(scope="module")
def app_db() -> Iterator[tuple[str, str]]:
    """(owner URL, loom_app URL) of a fresh database set up by loom-migrate."""
    info = admin_info()
    name = create_database(info)
    try:
        owner = admin_url(info, name)
        done = run_migrate(owner, app_password())
        assert done.returncode == 0, done.stdout + done.stderr
        yield owner, url_for(info, dbname=name, user=APP_ROLE, password=app_password())
    finally:
        drop_database(info, name)


def test_a21_runtime_role_reads_and_writes_the_new_columns(app_db) -> None:
    """Billing (as loom_app) stores errors, results and stale marks, and an admin's
    resolve writes `admin_audit_log` (contracts §3: grants as for the other Stripe
    columns)."""
    owner, app = app_db
    assert "result" in columns(owner, "stripe_events"), "migration 0012 not applied"
    evt = f"evt_test_{secrets.token_hex(8)}"
    with psycopg.connect(app, connect_timeout=5) as conn:
        conn.execute(
            "INSERT INTO stripe_events (id, type, payload, received_at, result, last_error, "
            "last_error_at, error_count) VALUES (%s, 'charge.refunded', '{}', now(), "
            "'unapplied', 'X: y', now(), 1)",
            (evt,),
        )
        conn.execute(
            "UPDATE stripe_events SET result = 'resolved_by_admin', processed_at = now(), "
            "last_error = NULL, error_count = 0, stale_reported_at = now(), "
            "last_error_at = NULL WHERE id = %s",
            (evt,),
        )
        row = conn.execute(
            "SELECT result, last_error, error_count, stale_reported_at IS NOT NULL "
            "FROM stripe_events WHERE id = %s",
            (evt,),
        ).fetchone()
        assert row == ("resolved_by_admin", None, 0, True)
        conn.execute(
            "INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id, "
            "details) VALUES (%s, 'stripe_event.resolve', 'stripe_event', %s, %s)",
            (uuid.uuid4(), evt, json.dumps({"reason": "test"})),
        )
        conn.commit()
