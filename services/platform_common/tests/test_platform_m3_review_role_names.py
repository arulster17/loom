"""B3 after review (docs/m3-review-fixes-spec.md, "Billing", B3): `loom-migrate` refuses a
configured login or group name (`LOOM_APP_DATABASE_USER` / `LOOM_APP_DATABASE_GROUP`)
that is the connecting role itself, and a group name that exists as a `LOGIN` role,
rather than altering it. Each refusal changes nothing about the roles involved: their
attributes, password and memberships are as before, and no new login appears.

Every role here is fresh (unique names) and dropped afterwards. The connecting role of
the first two cases is a fresh superuser, so the old code's ALTER ROLE on it succeeds
(a non-superuser could not change its own attributes, which would hide the bug); the
third case connects as the test server's admin role, whose name is never configured.
Roles are compared through `pg_authid` (attributes and password verifier, read as the
admin), so the check does not depend on the server's pg_hba authentication method."""

from __future__ import annotations

import secrets
import subprocess
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


def _server_url(info: dict[str, str]) -> str:
    return admin_url(info, info.get("dbname", "postgres"))


def role_state(conn: psycopg.Connection, name: str) -> tuple | None:
    """Everything `loom-migrate` could alter about a role: its attributes, password
    verifier, and the memberships it holds or grants (None: no such role)."""
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
    return (tuple(row), tuple(map(tuple, members)))


@pytest.fixture
def server() -> Iterator[dict]:
    """A fresh database, and a registry of fresh role names dropped afterwards (logins
    first, then groups, then the connecting roles)."""
    info = admin_info()
    db = create_database(info)
    tag = uuid.uuid4().hex[:10]
    made: dict[str, str] = {}

    def name(kind: str) -> str:
        made.setdefault(kind, f"loom_t_rn{kind[0]}_{tag}")
        return made[kind]

    try:
        yield {"info": info, "db": db, "name": name}
    finally:
        # the database, then each role, every drop run even if an earlier one fails
        drop_all(info, db, [made[k] for k in ("login", "group", "connecting") if k in made])


def create_login(info: dict[str, str], name: str, *, superuser: bool) -> str:
    password = secrets.token_hex(16)
    with psycopg.connect(_server_url(info), autocommit=True) as c:
        c.execute(
            sql.SQL("CREATE ROLE {} LOGIN {} PASSWORD {}").format(
                sql.Identifier(name),
                sql.SQL("SUPERUSER" if superuser else "NOSUPERUSER"),
                sql.Literal(password),
            )
        )
    return password


def states(info: dict[str, str], *names: str) -> dict[str, tuple | None]:
    with psycopg.connect(_server_url(info), autocommit=True) as c:
        return {n: role_state(c, n) for n in names}


def run_migrate(url: str, *, login: str, group: str) -> subprocess.CompletedProcess[str]:
    env = migrate_env(secrets.token_hex(16))
    env["LOOM_APP_DATABASE_USER"] = login
    env["LOOM_APP_DATABASE_GROUP"] = group
    try:  # under pytest's 120 s timeout, so a hang fails here with its output
        return subprocess.run(
            migrate_command(url), env=env, capture_output=True, text=True, timeout=90
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"loom-migrate did not exit within 90 s: {e.stderr!r}") from e


def assert_refused(
    done: subprocess.CompletedProcess[str], role: str, before: dict, after: dict
) -> None:
    """Refused (non-zero exit), nothing changed, and the refusal names the role."""
    out = done.stdout + done.stderr
    assert done.returncode != 0, f"loom-migrate accepted it:\n{out}"
    assert after == before, f"loom-migrate altered {role} (or made a login):\n{out}"
    assert role in out, f"the refusal does not name {role}:\n{out}"


def test_item_b3_migrate_refuses_login_that_is_the_connecting_role(server):
    """done-when B3: LOOM_APP_DATABASE_USER names the role loom-migrate connects as.
    It refuses, and the connecting role is unchanged (password, attributes, no new
    membership of the group)."""
    info, db = server["info"], server["db"]
    me, group = server["name"]("connecting"), server["name"]("group")
    password = create_login(info, me, superuser=True)
    before = states(info, me)

    done = run_migrate(url_for(info, dbname=db, user=me, password=password), login=me, group=group)

    assert_refused(done, me, before, states(info, me))


def test_item_b3_migrate_refuses_group_that_is_the_connecting_role(server):
    """done-when B3: LOOM_APP_DATABASE_GROUP names the role loom-migrate connects as.
    It refuses, the connecting role keeps LOGIN and everything else, and no login is
    created or made a member of it."""
    info, db = server["info"], server["db"]
    me, login = server["name"]("connecting"), server["name"]("login")
    password = create_login(info, me, superuser=True)
    before = states(info, me, login)
    assert before[login] is None

    done = run_migrate(url_for(info, dbname=db, user=me, password=password), login=login, group=me)

    assert_refused(done, me, before, states(info, me, login))


def test_item_b3_migrate_refuses_group_that_is_a_login_role(server):
    """done-when B3: LOOM_APP_DATABASE_GROUP names an existing LOGIN role (not the
    connecting one). It refuses rather than turning it NOLOGIN, and no login is created
    or made a member of it."""
    info, db = server["info"], server["db"]
    group, login = server["name"]("group"), server["name"]("login")
    create_login(info, group, superuser=False)
    before = states(info, group, login)
    assert before[login] is None

    done = run_migrate(admin_url(info, db), login=login, group=group)

    assert_refused(done, group, before, states(info, group, login))
