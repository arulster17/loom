"""Apply the platform's SQL migrations: `loom-migrate` / `python -m loom_platform.migrate`.

Migrations are numbered plain SQL files in `loom_platform/migrations/` (NNNN_name.sql),
applied in order, each in its own transaction, and recorded in `schema_migrations` with
a SHA-256 of the file. An applied file that has since changed is an error: write a new
migration instead. A session advisory lock keeps two runners from racing.

Plain SQL keeps the schema readable from the web app's side (Auth.js reads the same
tables) without a Python ORM in between.

Roles (docs/phase1-contracts.md §3): the URL's role is the **owner**; it owns every
object. Services connect as a **runtime** login (`LOOM_APP_DATABASE_USER`, default
`loom_app`) that holds its privileges through a NOLOGIN group role
(`LOOM_APP_DATABASE_GROUP`, default `loom_runtime`). `upgrade` creates the group role
when missing, before applying any migration, and hands its name to the migrations in
the session setting `loom.runtime_role`; `ensure_app_login` creates or updates the
login. `loom-migrate` runs both, the second only when `LOOM_APP_DATABASE_PASSWORD` is
set. Roles belong to the whole server, not one database, and many databases may be set
up at once (parallel test workers): creating a role or a membership that a concurrent
run just created is not an error.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources

import psycopg
from psycopg import errors, sql

from loom_platform.db import connect_sync, libpq_url
from loom_platform.settings import PlatformSettings

_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
# Arbitrary constant: pg_advisory_lock key for the migration runner.
_LOCK_KEY = 0x4C4F4F4D_4D494752  # "LOOMMIGR"

DEFAULT_APP_LOGIN = "loom_app"
DEFAULT_APP_GROUP = "loom_runtime"
# The session setting the migrations read the group role's name from (0011 onwards).
RUNTIME_ROLE_SETTING = "loom.runtime_role"
# A concurrent run created the role or membership first: "already exists". Which one
# Postgres raises depends on whether the other transaction had committed.
_ALREADY_EXISTS = (errors.DuplicateObject, errors.UniqueViolation)


def app_login_role() -> str:
    return os.environ.get("LOOM_APP_DATABASE_USER") or DEFAULT_APP_LOGIN


def app_group_role() -> str:
    return os.environ.get("LOOM_APP_DATABASE_GROUP") or DEFAULT_APP_GROUP


def _concurrently[T](fn: Callable[[], T], attempts: int = 20) -> T:
    """Run a role statement, retrying when a concurrent run updated the same catalog row
    ("tuple concurrently updated": two ALTER ROLEs of one role at once)."""
    for i in range(attempts):
        try:
            return fn()
        except errors.InternalError_ as e:
            if "concurrently updated" not in str(e) or i == attempts - 1:
                raise
            time.sleep(random.uniform(0.01, 0.05) * (i + 1))
    raise AssertionError("unreachable")


def _role(conn: psycopg.Connection, name: str) -> tuple[bool, bool, bool, bool, bool] | None:
    """(can login, superuser, createrole, createdb, bypassrls), or None if missing."""
    row = conn.execute(
        "SELECT rolcanlogin, rolsuper, rolcreaterole, rolcreatedb, rolbypassrls "
        "FROM pg_roles WHERE rolname = %s",
        (name,),
    ).fetchone()
    return None if row is None else (row[0], row[1], row[2], row[3], row[4])


def _create_role(conn: psycopg.Connection, name: str, options: sql.Composable) -> bool:
    """CREATE ROLE unless it exists; False when it already did (or a concurrent run made
    it first). `conn` is in autocommit mode, so a refused CREATE leaves it usable."""
    if _role(conn, name) is not None:
        return False
    try:
        conn.execute(sql.SQL("CREATE ROLE {} ").format(sql.Identifier(name)) + options)
    except _ALREADY_EXISTS:
        return False
    return True


def ensure_group_role(conn: psycopg.Connection, group_role: str) -> None:
    """The runtime group role exists and cannot log in. `conn`: autocommit."""
    if _create_role(conn, group_role, sql.SQL("NOLOGIN")):
        return
    state = _role(conn, group_role)
    if state is not None and state[0]:
        _concurrently(
            lambda: conn.execute(
                sql.SQL("ALTER ROLE {} NOLOGIN").format(sql.Identifier(group_role))
            )
        )


def ensure_app_login(url: str, *, login: str, group_role: str, password: str) -> None:
    """Create or update the runtime login: its password, LOGIN, and membership of
    `group_role` (created too if missing). Safe to run at the same time against many
    databases of one server."""
    if login == group_role:
        raise MigrationError("the runtime login and its group role need different names")
    with connect_sync(url, autocommit=True) as conn:
        ensure_group_role(conn, group_role)
        ident = sql.Identifier(login)
        # Sent as a SCRAM verifier made client-side (as psql's \password does), so the
        # plain password never appears in a statement the server could log.
        verifier = conn.pgconn.encrypt_password(password.encode(), login.encode(), b"scram-sha-256")
        secret = sql.Literal(verifier.decode())
        if not _create_role(conn, login, sql.SQL("LOGIN INHERIT PASSWORD {}").format(secret)):
            _concurrently(
                lambda: conn.execute(
                    sql.SQL("ALTER ROLE {} LOGIN INHERIT PASSWORD {}").format(ident, secret)
                )
            )
        state = _role(conn, login)
        assert state is not None
        if any(state[1:]):
            # Never the case for a role made here. Only a superuser may take SUPERUSER
            # or BYPASSRLS away, so a non-superuser owner gets an error naming it.
            _concurrently(
                lambda: conn.execute(
                    sql.SQL("ALTER ROLE {} NOSUPERUSER NOCREATEROLE NOCREATEDB NOBYPASSRLS").format(
                        ident
                    )
                )
            )
        member = conn.execute(
            "SELECT 1 FROM pg_auth_members m JOIN pg_roles g ON g.oid = m.roleid "
            "JOIN pg_roles u ON u.oid = m.member WHERE g.rolname = %s AND u.rolname = %s",
            (group_role, login),
        ).fetchone()
        if member is None:
            with contextlib.suppress(*_ALREADY_EXISTS):
                conn.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(group_role), ident))


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()


class MigrationError(RuntimeError):
    pass


def load_migrations() -> list[Migration]:
    out: list[Migration] = []
    for entry in resources.files("loom_platform.migrations").iterdir():
        m = _FILE_RE.match(entry.name)
        if m is None:
            continue
        out.append(Migration(int(m.group(1)), m.group(2), entry.read_text(encoding="utf-8")))
    out.sort(key=lambda mig: mig.version)
    versions = [mig.version for mig in out]
    if versions != list(range(1, len(out) + 1)):
        raise MigrationError(f"migration numbers must run 1..n without gaps: {versions}")
    return out


def _ensure_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    int PRIMARY KEY,
            name       text NOT NULL,
            checksum   text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def applied(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    rows = conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    return {v: (n, c) for v, n, c in rows}


def upgrade(url: str, *, group_role: str | None = None) -> list[Migration]:
    """Apply pending migrations; returns the ones applied now. First creates the runtime
    group role (`group_role`; None: `LOOM_APP_DATABASE_GROUP`, else `loom_runtime`) if
    missing; the migrations grant to it."""
    group = group_role or app_group_role()
    migrations = load_migrations()
    done: list[Migration] = []
    with connect_sync(url, autocommit=True) as conn:
        ensure_group_role(conn, group)
        conn.execute("SELECT set_config(%s, %s, false)", (RUNTIME_ROLE_SETTING, group))
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            _ensure_table(conn)
            have = applied(conn)
            for mig in migrations:
                if mig.version in have:
                    name, checksum = have[mig.version]
                    if checksum != mig.checksum:
                        raise MigrationError(
                            f"migration {mig.version:04d}_{mig.name} changed after it was "
                            f"applied (recorded as {name}); add a new migration instead"
                        )
                    continue
                with conn.transaction():
                    conn.execute(mig.sql.encode())  # bytes: no %-placeholder parsing
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) "
                        "VALUES (%s, %s, %s)",
                        (mig.version, mig.name, mig.checksum),
                    )
                done.append(mig)
            unknown = sorted(set(have) - {m.version for m in migrations})
            if unknown:
                raise MigrationError(f"database has migrations this code does not: {unknown}")
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    return done


def create_database(url: str) -> bool:
    """Create the URL's database if it does not exist (connecting to 'postgres')."""
    info = psycopg.conninfo.conninfo_to_dict(libpq_url(url))
    dbname = info.get("dbname")
    if not dbname:
        raise MigrationError("database URL names no database")
    admin = {**info, "dbname": "postgres"}
    with psycopg.connect(psycopg.conninfo.make_conninfo("", **admin), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if exists:
            return False
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(str(dbname))))
        return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="loom-migrate", description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", help="default: $LOOM_DATABASE_URL")
    parser.add_argument(
        "--create-database", action="store_true", help="create the database if missing"
    )
    args = parser.parse_args(argv)
    url = args.database_url or PlatformSettings().database_url
    if args.create_database and create_database(url):
        print("created database")
    group = app_group_role()
    done = upgrade(url, group_role=group)
    for mig in done:
        print(f"applied {mig.version:04d}_{mig.name}")
    password = os.environ.get("LOOM_APP_DATABASE_PASSWORD")
    if password:
        login = app_login_role()
        ensure_app_login(url, login=login, group_role=group, password=password)
        print(f"runtime login {login} ready (member of {group})")
    if not done:
        print("up to date")


if __name__ == "__main__":
    main()
