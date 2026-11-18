"""B3 (docs/m3-review-fixes-spec.md, "Billing", B3): `loom-migrate` refuses a configured
login name that is the connecting role itself, `current_user` *or* `session_user`,
rather than altering it.

The two differ inside loom-migrate's own connection when the connection assumes another
role at startup, which libpq and Postgres both offer without any code in loom-migrate:
the URL's `options=-c role=<r>` (the same as `PGOPTIONS`), or a per-role default set by
`ALTER ROLE <login> SET role = <r>`. Then `session_user` is the login in the URL and
`current_user` is the assumed role.

Every role here is fresh and dropped afterwards (the database first, then each role in
its own try/finally). The login in the URL is a non-superuser member of the assumed
role, which is a superuser (NOLOGIN), so loom-migrate can migrate as it, and so an
`ALTER ROLE` on the login would succeed: refusing is the only way the login stays as it
was. The login's state is read through `pg_authid` as the admin."""

from __future__ import annotations

import secrets
import subprocess
import urllib.parse
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from platform_m3_review2_kit import drop_all
from platform_m3_roles import (
    admin_info,
    admin_url,
    create_database,
    migrate_command,
    migrate_env,
    url_for,
)
from psycopg import sql

pytestmark = [pytest.mark.postgres]


def _server(info: dict[str, str]) -> psycopg.Connection:
    return psycopg.connect(admin_url(info, info.get("dbname", "postgres")), autocommit=True)


def role_state(info: dict[str, str], name: str) -> tuple | None:
    """What loom-migrate could alter about a role: attributes, password verifier, the
    memberships it holds or grants, and its per-role settings."""
    with _server(info) as conn:
        row = conn.execute(
            "SELECT rolsuper, rolinherit, rolcreaterole, rolcreatedb, rolcanlogin, "
            "rolreplication, rolbypassrls, rolconnlimit, rolpassword, rolvaliduntil "
            "FROM pg_authid WHERE rolname = %s",
            (name,),
        ).fetchone()
        if row is None:
            return None
        members = conn.execute(
            "SELECT g.rolname, u.rolname, m.admin_option FROM pg_auth_members m "
            "JOIN pg_roles g ON g.oid = m.roleid JOIN pg_roles u ON u.oid = m.member "
            "WHERE g.rolname = %s OR u.rolname = %s ORDER BY 1, 2",
            (name, name),
        ).fetchall()
        settings = conn.execute(
            "SELECT s.setconfig FROM pg_db_role_setting s JOIN pg_roles r "
            "ON r.oid = s.setrole WHERE r.rolname = %s ORDER BY s.setdatabase",
            (name,),
        ).fetchall()
    return (tuple(row), tuple(map(tuple, members)), tuple(map(tuple, settings)))


@pytest.fixture
def server() -> Iterator[dict]:
    """A fresh database; a fresh login (non-superuser, with a password) that is a member
    of a fresh superuser NOLOGIN role it can assume; a fresh group name."""
    info = admin_info()
    tag = uuid.uuid4().hex[:10]
    names = {
        "assumed": f"loom_t_r2a_{tag}",
        "login": f"loom_t_r2u_{tag}",
        "group": f"loom_t_r2g_{tag}",
    }
    password = secrets.token_hex(16)
    db: str | None = None
    try:
        db = create_database(info)
        with _server(info) as c:
            c.execute(
                sql.SQL("CREATE ROLE {} NOLOGIN SUPERUSER").format(sql.Identifier(names["assumed"]))
            )
            c.execute(
                sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER PASSWORD {}").format(
                    sql.Identifier(names["login"]), sql.Literal(password)
                )
            )
            c.execute(
                sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(names["assumed"]), sql.Identifier(names["login"])
                )
            )
        yield {"info": info, "db": db, "names": names, "password": password}
    finally:
        drop_all(info, db, [names["assumed"], names["login"], names["group"]])


def run_migrate(url: str, *, login: str, group: str) -> subprocess.CompletedProcess[str]:
    env = migrate_env(secrets.token_hex(16))
    env["LOOM_APP_DATABASE_USER"] = login
    env["LOOM_APP_DATABASE_GROUP"] = group
    env.pop("PGOPTIONS", None)
    return subprocess.run(migrate_command(url), env=env, capture_output=True, text=True, timeout=90)


def assert_users(url: str, session: str, current: str) -> None:
    """The setup is what the test claims: over `url`, session_user and current_user
    differ."""
    with psycopg.connect(url) as conn:
        row = conn.execute("SELECT session_user::text, current_user::text").fetchone()
    assert row == (session, current), f"setup: (session_user, current_user) is {row}"


def assert_refused(done: subprocess.CompletedProcess[str], role: str, before, after) -> None:
    out = done.stdout + done.stderr
    assert after == before, (
        f"loom-migrate altered {role} (session_user) instead of refusing:\n{out}"
    )
    assert done.returncode != 0, f"loom-migrate accepted {role} (session_user):\n{out}"
    assert role in out, f"the refusal does not name {role}:\n{out}"


def test_item3_migrate_refuses_login_that_is_session_user_via_url_role_option(server):
    """done-when B3: connected as L with `options=-c role=R` in the URL (session_user L,
    current_user R), LOOM_APP_DATABASE_USER=L is refused and L is unchanged."""
    info, db, names = server["info"], server["db"], server["names"]
    login, assumed = names["login"], names["assumed"]
    options = urllib.parse.quote(f"-c role={assumed}", safe="")
    url = url_for(info, dbname=db, user=login, password=server["password"]) + f"?options={options}"
    assert_users(url, login, assumed)
    before = role_state(info, login)

    done = run_migrate(url, login=login, group=names["group"])

    assert_refused(done, login, before, role_state(info, login))


def test_item3_migrate_refuses_login_that_is_session_user_via_role_default(server):
    """done-when B3: L has `ALTER ROLE L SET role = R` (every session of L runs as R:
    session_user L, current_user R); LOOM_APP_DATABASE_USER=L is refused and L is
    unchanged."""
    info, db, names = server["info"], server["db"], server["names"]
    login, assumed = names["login"], names["assumed"]
    with _server(info) as c:
        c.execute(
            sql.SQL("ALTER ROLE {} SET role = {}").format(
                sql.Identifier(login), sql.Literal(assumed)
            )
        )
    url = url_for(info, dbname=db, user=login, password=server["password"])
    assert_users(url, login, assumed)
    before = role_state(info, login)

    done = run_migrate(url, login=login, group=names["group"])

    assert_refused(done, login, before, role_state(info, login))
