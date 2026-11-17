"""Helpers for the before-M3 role tests (test_platform_m3_*.py, spec B3): fresh databases
migrated by `loom-migrate` with `LOOM_APP_DATABASE_PASSWORD`, URLs for each role.

Roles belong to the whole Postgres server and other test workers use the default roles
(`loom_app`, group `loom_runtime`) at the same time, so those are never dropped and the
`loom_app` password is always the same one: `LOOM_APP_DATABASE_PASSWORD` when the
environment sets it, else the dev password docker-compose gives it (so a dev stack on the
same server keeps working), else a fixed test password. The race test uses fresh role
names of its own (spec B3: role names are configurable) and drops them."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import urllib.parse
import uuid
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from loom_platform.db import libpq_url
from loom_platform.testing import DEFAULT_PG, PG_ENV

APP_ROLE = "loom_app"
GROUP_ROLE = "loom_runtime"  # spec B3: LOOM_APP_DATABASE_GROUP's default
APPEND_ONLY = ("ledger_entries", "ledger_transactions", "admin_audit_log")
REPO_ROOT = Path(__file__).resolve().parents[3]
FALLBACK_APP_PASSWORD = "loom-app-test-password"

# A YAML anchor (`&name`) may sit between the key and its value; an alias (`*name`) is
# never the password itself (its anchored definition is matched instead).
_ANCHOR = r"(?:&[^\s]+\s+)?"
_VALUE = r"([^'\"\s$&*#][^'\"\s$#]*)"
_COMPOSE_PATTERNS = (
    r"LOOM_APP_DATABASE_PASSWORD['\"]?\s*[:=]\s*"
    + _ANCHOR
    + r"['\"]?\$\{[A-Z0-9_]+:-([^}'\"\s]+)\}",
    r"LOOM_APP_DATABASE_PASSWORD['\"]?\s*[:=]\s*" + _ANCHOR + r"['\"]?" + _VALUE,
    r"postgres(?:ql)?(?:\+psycopg)?://loom_app:\$\{[A-Z0-9_]+:-([^}]+)\}@",
    r"postgres(?:ql)?(?:\+psycopg)?://loom_app:([^@\s$'\"]+)@",
)
_TRAILING_COMMENT = re.compile(r"\s#.*$")


def _compose_lines(text: str) -> list[str]:
    """docker-compose.yml's lines without comments (whole-line or trailing)."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        out.append(_TRAILING_COMMENT.sub("", line))
    return out


def compose_app_password() -> str | None:
    """The loom_app dev password docker-compose.yml sets, if any. Comment lines and YAML
    anchors/aliases are skipped."""
    path = REPO_ROOT / "docker-compose.yml"
    if not path.exists():
        return None
    lines = _compose_lines(path.read_text(encoding="utf-8"))
    for pattern in _COMPOSE_PATTERNS:
        for line in lines:
            m = re.search(pattern, line)
            if m:
                return urllib.parse.unquote(m.group(1))
    return None


def app_password() -> str:
    """The loom_app password the tests give loom-migrate: the environment's
    LOOM_APP_DATABASE_PASSWORD if set, else the compose dev password, else a fixed one."""
    return (
        os.environ.get("LOOM_APP_DATABASE_PASSWORD")
        or compose_app_password()
        or FALLBACK_APP_PASSWORD
    )


def admin_info() -> dict[str, str]:
    explicit = os.environ.get(PG_ENV)
    url = libpq_url(explicit or DEFAULT_PG)
    try:
        psycopg.connect(url, connect_timeout=2).close()
    except psycopg.OperationalError as e:
        if explicit:
            raise
        pytest.skip(f"Postgres not reachable at {DEFAULT_PG} and {PG_ENV} unset: {e}")
    return {k: str(v) for k, v in psycopg.conninfo.conninfo_to_dict(url).items()}


def url_for(info: dict[str, str], *, dbname: str, user: str, password: str) -> str:
    """A postgresql:// URL (the form the services' settings read)."""
    host = info.get("host", "localhost")
    port = info.get("port", "5432")
    q = urllib.parse.quote
    return f"postgresql://{q(user, safe='')}:{q(password, safe='')}@{host}:{port}/{dbname}"


def admin_url(info: dict[str, str], dbname: str) -> str:
    return url_for(info, dbname=dbname, user=info["user"], password=info.get("password", ""))


def create_database(info: dict[str, str]) -> str:
    name = f"loom_m3_roles_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url(info, info.get("dbname", "postgres")), autocommit=True) as c:
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    return name


def drop_database(info: dict[str, str], name: str) -> None:
    with psycopg.connect(admin_url(info, info.get("dbname", "postgres")), autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def migrate_command(url: str) -> list[str]:
    return [sys.executable, "-m", "loom_platform.migrate", "--database-url", url]


def migrate_env(password: str) -> dict[str, str]:
    """loom-migrate's environment: no LOOM_* from the caller (so the role names are the
    defaults), dev mode, and the loom_app password."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_")}
    env["LOOM_ENV"] = "dev"
    env["LOOM_APP_DATABASE_PASSWORD"] = password
    return env


def run_migrate(url: str, password: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        migrate_command(url),
        env=migrate_env(password),
        capture_output=True,
        text=True,
        timeout=180,
    )


def group_roles(conn: psycopg.Connection, login: str = APP_ROLE) -> list[str]:
    """The roles `login` is a member of (the spec's NOLOGIN group), built-ins aside."""
    rows = conn.execute(
        "SELECT g.rolname FROM pg_auth_members m "
        "JOIN pg_roles g ON g.oid = m.roleid JOIN pg_roles u ON u.oid = m.member "
        "WHERE u.rolname = %s AND g.rolname NOT LIKE 'pg\\_%%' ORDER BY 1",
        (login,),
    ).fetchall()
    return [r[0] for r in rows]


def grant_snapshot(conn: psycopg.Connection, roles: list[str]) -> list[tuple[str, str, str]]:
    """(object, grantee, privilege) for every explicit grant to `roles` in schema public:
    tables, views, sequences (not partitions) and functions."""
    rels = conn.execute(
        "SELECT c.relkind::text || ':' || c.relname, r.rolname, a.privilege_type "
        "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "CROSS JOIN LATERAL aclexplode(c.relacl) a JOIN pg_roles r ON r.oid = a.grantee "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'S', 'v', 'm') "
        "AND NOT c.relispartition AND r.rolname = ANY(%s)",
        (roles,),
    ).fetchall()
    funcs = conn.execute(
        "SELECT 'f:' || p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')', "
        "r.rolname, a.privilege_type "
        "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(p.proacl) a JOIN pg_roles r ON r.oid = a.grantee "
        "WHERE n.nspname = 'public' AND r.rolname = ANY(%s)",
        (roles,),
    ).fetchall()
    return sorted((str(a), str(b), str(c)) for a, b, c in rels + funcs)
