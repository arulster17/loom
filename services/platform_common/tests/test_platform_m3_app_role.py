"""B3 (plan R8, docs/m3-review-fixes-spec.md): services run as `loom_app`, a role that
cannot switch off the ledger's guards.

Each database here is fresh and set up the way the spec says: `loom-migrate` run as the
owner with `LOOM_APP_DATABASE_PASSWORD` and the default role names (`loom_app`, group
`loom_runtime`). The race test calls the spec's seams (`migrate.upgrade(url,
group_role=...)`, `migrate.ensure_app_login(...)`) with fresh role names instead."""

from __future__ import annotations

import datetime as dt
import inspect
import secrets
import threading
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psycopg
import pytest
from platform_m3_roles import (
    APP_ROLE,
    APPEND_ONLY,
    GROUP_ROLE,
    admin_info,
    admin_url,
    app_password,
    compose_app_password,
    create_database,
    drop_database,
    grant_snapshot,
    group_roles,
    run_migrate,
    url_for,
)
from psycopg import errors, sql

from loom_platform import migrate
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
    assert GROUP_ROLE in groups, f"loom_app is not a member of {GROUP_ROLE}: {groups}"
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


def test_item_b3_partition_upkeep_runs_with_owner_rights(owner_conn):
    """loom_ensure_usage_partitions runs with the owner's rights (SECURITY DEFINER) and a
    fixed search_path, so loom_app itself still cannot create objects."""
    rows = owner_conn.execute(
        "SELECT p.oid::regprocedure::text, p.prosecdef, p.proconfig FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace "
        "WHERE n.nspname = 'public' AND p.proname = 'loom_ensure_usage_partitions'"
    ).fetchall()
    assert rows, "no loom_ensure_usage_partitions function"
    for sig, secdef, config in rows:
        assert secdef is True, f"{sig} is not SECURITY DEFINER"
        assert any(c.startswith("search_path=") for c in config or []), (
            f"{sig} has no fixed search_path: {config}"
        )


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


RACE_ROUNDS = 3
RACE_DATABASES = 6


def _race_round(info: dict[str, str]) -> None:
    """One round: fresh login and group names, 6 fresh databases, 6 threads released at
    once into upgrade + ensure_app_login. Drops what it made."""
    tag = uuid.uuid4().hex[:10]
    login, group = f"loom_t_race_{tag}_app", f"loom_t_race_{tag}_grp"
    password = secrets.token_hex(16)
    names: list[str] = []
    barrier = threading.Barrier(RACE_DATABASES)

    def setup(db: str) -> list[migrate.Migration]:
        barrier.wait(timeout=60)
        applied = migrate.upgrade(db, group_role=group)
        migrate.ensure_app_login(db, login=login, group_role=group, password=password)
        return applied

    try:
        for _ in range(RACE_DATABASES):
            names.append(create_database(info))
        urls = [admin_url(info, n) for n in names]
        with ThreadPoolExecutor(max_workers=RACE_DATABASES) as ex:
            futures = [ex.submit(setup, u) for u in urls]
            failures = [(n, f.exception(timeout=90)) for n, f in zip(names, futures, strict=True)]
        failed = [(n, repr(e)) for n, e in failures if e is not None]
        assert failed == [], f"role setup failed when run concurrently: {failed}"
        everything = [m.version for m in migrate.load_migrations()]
        for n, f in zip(names, futures, strict=True):
            assert [m.version for m in f.result()] == everything, f"{n}: not fully migrated"

        snapshots = []
        for n, url in zip(names, urls, strict=True):
            with psycopg.connect(url, autocommit=True) as conn:
                assert group_roles(conn, login) == [group], f"{n}: {login} not in {group}"
                can_login = conn.execute(
                    "SELECT rolname, rolcanlogin FROM pg_roles WHERE rolname = ANY(%s) ORDER BY 1",
                    ([login, group],),
                ).fetchall()
                assert sorted(can_login) == sorted([(login, True), (group, False)])
                snapshots.append(grant_snapshot(conn, [group, login]))
            as_login = url_for(info, dbname=n, user=login, password=password)
            with psycopg.connect(as_login, connect_timeout=5) as conn:
                assert conn.execute("SELECT current_user").fetchone() == (login,)
        assert snapshots[0], "no grants to the group role"
        assert {grantee for _, grantee, _ in snapshots[0]} == {group}, (
            "grants go to the group role only"
        )
        for n, snap in zip(names[1:], snapshots[1:], strict=True):
            assert snap == snapshots[0], f"{n}: grants differ from {names[0]}"
    finally:
        _drop_all(info, names, [login, group])


def _drop_all(info: dict[str, str], names: list[str], roles: list[str]) -> None:
    """Drop the round's databases, then its roles (the login before the group). Every
    step is tried; any failure is raised at the end so nothing is left silently."""
    problems: list[str] = []

    def attempt(what: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as e:
            problems.append(f"{what}: {e!r}")

    for n in names:
        attempt(f"drop database {n}", lambda n=n: drop_database(info, n))

    def drop_role(role: str) -> None:
        maintenance = admin_url(info, info.get("dbname", "postgres"))
        with psycopg.connect(maintenance, autocommit=True) as c:
            c.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))

    for role in roles:
        attempt(f"drop role {role}", lambda role=role: drop_role(role))
    assert problems == [], f"left behind on the server: {problems}"


def test_item_b3_role_setup_concurrent_on_many_databases():
    """Roles belong to the whole server: the role setup run at the same time against
    several fresh databases succeeds every time and the grants come out the same.

    Each of several rounds makes fresh random login and group names (the spec's
    configurable role names) and 6 fresh databases, then releases 6 threads at once,
    through a barrier, into `migrate.upgrade(db, group_role=g)` followed by
    `migrate.ensure_app_login(db, login=l, group_role=g, password=p)`. Every call must
    return normally (no exception at all: "role already exists" and duplicate-key errors
    surface as exceptions here, so no log text is read), every database ends fully
    migrated, and the grant snapshots are equal and non-empty.

    The spec says role creation must treat both `duplicate_object` and
    `unique_violation` as "already exists". The test cannot force the interleaving that
    produces either error; fresh names each round (so every round really creates the
    roles) plus the barrier, repeated over several rounds, are how it gives the race a
    real chance to happen."""
    params = inspect.signature(migrate.upgrade).parameters
    assert "group_role" in params, "spec B3 seam missing: migrate.upgrade(url, *, group_role)"
    assert hasattr(migrate, "ensure_app_login"), (
        "spec B3 seam missing: migrate.ensure_app_login(url, *, login, group_role, password)"
    )
    info = admin_info()
    for _ in range(RACE_ROUNDS):
        _race_round(info)


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
