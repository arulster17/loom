"""B3 (plan R8, docs/m3-review-fixes-spec.md): services run as `loom_app`, a role that
cannot switch off the ledger's guards.

Each database here is fresh and set up the way the spec says: `loom-migrate` run as the
owner with `LOOM_APP_DATABASE_PASSWORD`. The group role is not named by the spec: it is
found as the role(s) `loom_app` is a member of."""

from __future__ import annotations

import datetime as dt
import secrets
import subprocess
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from platform_m3_roles import (
    APP_ROLE,
    APPEND_ONLY,
    admin_info,
    admin_url,
    app_password,
    compose_app_password,
    create_database,
    drop_database,
    grant_snapshot,
    group_roles,
    migrate_command,
    migrate_env,
    run_migrate,
    url_for,
)
from psycopg import errors

from loom_platform.db import connect
from loom_platform.ledger import STRIPE_CLEARING, Entry, org_credit_account, post_transaction
from loom_platform.pricing import Price, cost_micros
from loom_platform.settings import PlatformSettings
from loom_platform.testing import create_api_key, create_org
from loom_platform.usage import UsageEvent, insert_usage_events

pytestmark = [pytest.mark.postgres]

UPDATE_COLUMN = {
    "ledger_entries": "amount_micros",
    "ledger_transactions": "memo",
    "admin_audit_log": "action",
}


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


@pytest.fixture
def owner_conn(app_db) -> Iterator[psycopg.Connection]:
    with psycopg.connect(app_db[0], autocommit=True) as conn:
        yield conn


@pytest.fixture
def app_conn(app_db) -> Iterator[psycopg.Connection]:
    """A connection as loom_app (fails while loom-migrate does not create the login)."""
    with psycopg.connect(app_db[1], autocommit=True, connect_timeout=5) as conn:
        yield conn


def triggers(owner_conn: psycopg.Connection, table: str) -> list[tuple[str, str, str]]:
    """(trigger, function schema, function name) of the table's own triggers."""
    rows = owner_conn.execute(
        "SELECT t.tgname, n.nspname, p.proname FROM pg_trigger t "
        "JOIN pg_proc p ON p.oid = t.tgfoid JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE t.tgrelid = %s::regclass AND NOT t.tgisinternal ORDER BY 1",
        (table,),
    ).fetchall()
    return [(str(a), str(b), str(c)) for a, b, c in rows]


# ------------------------------------------------------------ the role itself


def test_item_b3_app_role_is_a_plain_login(owner_conn, app_conn):
    """loom_app is not a superuser, owns nothing, and cannot create in the schema."""
    row = owner_conn.execute(
        "SELECT rolcanlogin, rolsuper, rolcreaterole, rolcreatedb, rolbypassrls "
        "FROM pg_roles WHERE rolname = %s",
        (APP_ROLE,),
    ).fetchone()
    assert row is not None, "loom-migrate did not create the loom_app login"
    assert row == (True, False, False, False, False)
    owned = owner_conn.execute(
        "SELECT count(*) FROM pg_class WHERE relowner = %s::regrole", (APP_ROLE,)
    ).fetchone()
    assert owned == (0,)
    funcs = owner_conn.execute(
        "SELECT count(*) FROM pg_proc WHERE proowner = %s::regrole", (APP_ROLE,)
    ).fetchone()
    assert funcs == (0,)
    with pytest.raises(errors.InsufficientPrivilege):
        app_conn.execute("CREATE TABLE loom_m3_probe (x int)")


def test_item_b3_group_role_grants_every_table(owner_conn, app_conn):
    """Every table has an explicit grant to the group role loom_app is a member of, so a
    later migration cannot forget one. Sequences are usable (USAGE)."""
    groups = group_roles(owner_conn)
    assert groups, "loom_app is a member of no group role"
    for g in groups:
        login = owner_conn.execute(
            "SELECT rolcanlogin FROM pg_roles WHERE rolname = %s", (g,)
        ).fetchone()
        assert login == (False,), f"group role {g} can log in"
    missing = owner_conn.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND NOT c.relispartition "
        "AND c.relname <> 'schema_migrations' "
        "AND NOT EXISTS (SELECT 1 FROM aclexplode(c.relacl) a "
        "  JOIN pg_roles r ON r.oid = a.grantee WHERE r.rolname = ANY(%s)) ORDER BY 1",
        (groups,),
    ).fetchall()
    assert [r[0] for r in missing] == []
    no_usage = owner_conn.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind = 'S' "
        "AND CASE WHEN c.relkind = 'S' THEN NOT has_sequence_privilege(%s, c.oid, 'USAGE') "
        "ELSE false END ORDER BY 1",
        (APP_ROLE,),
    ).fetchall()
    assert [r[0] for r in no_usage] == []


@pytest.mark.parametrize("table", APPEND_ONLY)
def test_item_b3_append_only_tables_grant_only_select_insert(owner_conn, app_conn, table):
    privileges = {
        p: owner_conn.execute(
            "SELECT has_table_privilege(%s, %s, %s)", (APP_ROLE, table, p)
        ).fetchone()[0]  # type: ignore[index]
        for p in ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
    }
    assert {p for p, ok in privileges.items() if ok} == {"SELECT", "INSERT"}


# ------------------------------------------------------- what loom_app cannot do


@pytest.mark.parametrize("table", APPEND_ONLY)
@pytest.mark.parametrize("statement", ["UPDATE", "DELETE", "TRUNCATE"])
def test_item_b3_app_role_cannot_rewrite_append_only_tables(app_conn, table, statement):
    """Refused by privileges (not only by the triggers the owner could disable)."""
    query = {
        "UPDATE": f"UPDATE {table} SET {UPDATE_COLUMN[table]} = {UPDATE_COLUMN[table]} WHERE false",
        "DELETE": f"DELETE FROM {table} WHERE false",
        "TRUNCATE": f"TRUNCATE {table}",
    }[statement]
    with pytest.raises(errors.InsufficientPrivilege):
        app_conn.execute(query.encode())


@pytest.mark.parametrize("table", APPEND_ONLY)
def test_item_b3_app_role_cannot_switch_off_triggers(owner_conn, app_conn, table):
    """ALTER TABLE ... DISABLE TRIGGER, DROP TRIGGER and replacing a trigger function all
    fail as loom_app."""
    found = triggers(owner_conn, table)
    assert found, f"{table} has no triggers"
    attempts = [
        f"ALTER TABLE {table} DISABLE TRIGGER ALL",
        f"ALTER TABLE {table} DISABLE TRIGGER USER",
    ]
    for name, schema, fn in found:
        attempts += [
            f'ALTER TABLE {table} DISABLE TRIGGER "{name}"',
            f'DROP TRIGGER "{name}" ON {table}',
            f'CREATE OR REPLACE FUNCTION "{schema}"."{fn}"() RETURNS trigger '
            "LANGUAGE plpgsql AS $$ BEGIN RETURN NULL; END $$",
            f'ALTER FUNCTION "{schema}"."{fn}"() OWNER TO {APP_ROLE}',
        ]
    for q in attempts:
        with pytest.raises(errors.InsufficientPrivilege):
            app_conn.execute(q.encode())
    assert triggers(owner_conn, table) == found
    enabled = owner_conn.execute(
        "SELECT count(*) FROM pg_trigger WHERE tgrelid = %s::regclass "
        "AND NOT tgisinternal AND tgenabled = 'D'",
        (table,),
    ).fetchone()
    assert enabled == (0,)


def test_item_b3_app_role_cannot_skip_triggers_by_replication_role(app_conn):
    with pytest.raises(errors.InsufficientPrivilege):
        app_conn.execute("SET session_replication_role = replica")


# --------------------------------------------------- what loom_app still does


async def test_item_b3_app_role_does_normal_service_work(app_db):
    """Post a ledger transaction through loom_post_transaction, insert usage events and
    an audit row, and read the totals, all as loom_app."""
    conn = await connect(app_db[1])
    try:
        async with conn.transaction():
            org = await create_org(conn, status="active")
            key, _ = await create_api_key(conn, org)
            acct = await org_credit_account(conn, org)
            posted = await post_transaction(
                conn,
                kind="purchase",
                idempotency_key=f"stripe:evt_m3_{uuid.uuid4().hex}",
                org_id=org,
                entries=[Entry(acct, 5_000_000), Entry(STRIPE_CLEARING, -5_000_000)],
            )
            assert posted.created
            price = Price(77_000, 38_000, 308_000)
            ev = UsageEvent(
                id=uuid.uuid4(),
                org_id=org,
                api_key_id=key,
                model_id="qwen3-8b",
                price_version=price.version,
                cloud="local",
                prompt_tokens=1000,
                completion_tokens=500,
                cached_prompt_tokens=0,
                latency_ms=120,
                ttft_ms=15,
                status="ok",
                cost_micros=cost_micros(price, 1000, 500, 0),
                created_at=dt.datetime.now(dt.UTC),
            )
            assert await insert_usage_events(conn, [ev]) == 1
            await conn.execute(
                "INSERT INTO admin_audit_log (actor_user_id, action, target_type, target_id) "
                "VALUES (%s, 'm3_probe', 'org', %s)",
                (uuid.uuid4(), str(org)),
            )
            cur = await conn.execute(
                "SELECT loom_org_credit_total(%s), loom_org_usage_total(%s)", (org, org)
            )
            assert await cur.fetchone() == (5_000_000, ev.cost_micros)
    finally:
        await conn.close()


def _has_partition(owner_conn: psycopg.Connection, month: dt.date) -> bool:
    owner_conn.execute("SET TIME ZONE 'UTC'")  # partition bounds print in the session zone
    row = owner_conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE i.inhparent = 'usage_events'::regclass "
        "AND pg_get_expr(c.relpartbound, c.oid) LIKE %s)",
        (f"%FROM ('{month.isoformat()}%",),
    ).fetchone()
    return bool(row and row[0])


def test_item_b3_app_role_runs_partition_upkeep(owner_conn, app_conn):
    """The billing worker (as loom_app) keeps the monthly usage_events partitions rolling
    (loom_ensure_usage_partitions), including creating a month that is missing."""
    month = dt.date(dt.date.today().year + 5, 1, 1)
    while _has_partition(owner_conn, month):
        month = dt.date(month.year + 1, 1, 1)
    row = app_conn.execute("SELECT loom_ensure_usage_partitions(%s, 1)", (month,)).fetchone()
    assert row == (1,)
    assert _has_partition(owner_conn, month)


# ------------------------------------------------- setup run on many databases


def test_item_b3_role_setup_concurrent_on_many_databases():
    """loom-migrate's role setup run at the same time against several fresh databases
    on one server: every run succeeds (no "role already exists" or duplicate-key error)
    and the grants come out the same."""
    info = admin_info()
    names = [create_database(info) for _ in range(6)]
    try:
        procs = [
            subprocess.Popen(
                migrate_command(admin_url(info, n)),
                env=migrate_env(app_password()),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for n in names
        ]
        outs = [p.communicate(timeout=180) for p in procs]
        for p, (out, err) in zip(procs, outs, strict=True):
            assert p.returncode == 0, out + err
            assert "already exists" not in err and "duplicate key" not in err, err
        snapshots = []
        for n in names:
            with psycopg.connect(admin_url(info, n), autocommit=True) as conn:
                groups = group_roles(conn)
                assert groups, "loom_app is a member of no group role"
                snapshots.append(grant_snapshot(conn, [*groups, APP_ROLE]))
            app = url_for(info, dbname=n, user=APP_ROLE, password=app_password())
            with psycopg.connect(app, connect_timeout=5) as conn:
                assert conn.execute("SELECT current_user").fetchone() == (APP_ROLE,)
        assert snapshots[0], "no grants to the group role"
        for s in snapshots[1:]:
            assert s == snapshots[0]
    finally:
        for n in names:
            drop_database(info, n)


# ----------------------------------------------------------- production check


def test_item_b3_dev_app_password_is_a_production_problem():
    """docker-compose gives loom_app a dev password; a service using it outside dev is
    refused (one more production problem than with a strong password)."""
    dev = compose_app_password()
    assert dev, "docker-compose.yml sets no loom_app password (LOOM_APP_DATABASE_PASSWORD)"
    redis = "rediss://:" + secrets.token_hex(16) + "@redis.internal:6380/0"

    def problems(password: str) -> list[str]:
        url = f"postgresql://{APP_ROLE}:{password}@db.internal:5432/loom_platform"
        return PlatformSettings(env="prod", database_url=url, redis_url=redis).production_problems()

    strong = problems(secrets.token_hex(16))
    weak = problems(dev)
    assert len(weak) > len(strong), weak
