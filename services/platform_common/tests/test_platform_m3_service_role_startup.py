"""B3 (plan R8, docs/m3-review-fixes-spec.md): outside LOOM_ENV=dev, each Python service
checks at startup, over its own connection, that its role is not a superuser and owns no
table in the schema, and otherwise exits 2 naming the problem.

The services are started as subprocesses with LOOM_ENV=prod. Other production problems
(Redis without AUTH or TLS, missing certificates, the dev password) are listed in the same
exit-2 message, so the assertions look for the role's problem among them, by wording
specific to it: "superuser" on the same line as the role's name (or "role"), "owns" /
"owner" on the same line as "table". A bare "own" (say, "set your own password") is not
a role problem."""

from __future__ import annotations

import os
import re
import secrets
import subprocess
import sys
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from platform_m3_review2_kit import drop_all
from platform_m3_roles import (
    APP_ROLE,
    admin_info,
    admin_url,
    app_password,
    create_database,
    run_migrate,
    url_for,
)
from psycopg import sql

pytestmark = [pytest.mark.postgres]

SERVICES = {
    "gateway": [sys.executable, "-m", "loom_gateway"],
    "billing": [sys.executable, "-m", "loom_billing", "worker"],
    "poolmgr": [sys.executable, "-m", "loom_poolmgr.cli", "run"],
}
_OWN = r"\bown(?:s|er|ed|ership)\b"
_TABLE = r"\btables?\b"
OWNER = re.compile(rf"{_OWN}[^\n]{{0,80}}{_TABLE}|{_TABLE}[^\n]{{0,80}}{_OWN}", re.IGNORECASE)


def superuser_problem(stderr: str, role: str) -> bool:
    """A line naming the role (or "role") and "superuser"."""
    who = rf"(?:\b{re.escape(role)}\b|\brole\b)"
    pattern = rf"{who}[^\n]{{0,80}}\bsuperuser\b|\bsuperuser\b[^\n]{{0,80}}{who}"
    return re.search(pattern, stderr, re.IGNORECASE) is not None


@pytest.fixture(scope="module")
def db() -> Iterator[dict[str, str]]:
    """A fresh database set up by loom-migrate, plus a non-superuser login that owns one
    of its tables (like the migration owner would). URLs per role."""
    info = admin_info()
    name = create_database(info)
    owner_role = f"loom_t_owner_{uuid.uuid4().hex[:10]}"
    owner_pw = secrets.token_hex(16)
    try:
        done = run_migrate(admin_url(info, name), app_password())
        assert done.returncode == 0, done.stdout + done.stderr
        with psycopg.connect(admin_url(info, name), autocommit=True) as conn:
            conn.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(owner_role), sql.Literal(owner_pw)
                )
            )
            conn.execute(sql.SQL("ALTER TABLE orgs OWNER TO {}").format(sql.Identifier(owner_role)))
        yield {
            "superuser_name": info["user"],
            "superuser": admin_url(info, name),
            "owner": url_for(info, dbname=name, user=owner_role, password=owner_pw),
            "app": url_for(info, dbname=name, user=APP_ROLE, password=app_password()),
        }
    finally:
        # the database first (it holds the table the role owns), then the role, even if
        # the database drop fails
        drop_all(info, name, [owner_role])


def start_in_prod(service: str, database_url: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_")}
    env.update(
        LOOM_ENV="prod",
        LOOM_DATABASE_URL=database_url,
        LOOM_REDIS_URL="redis://127.0.0.1:1/0",  # unreachable: Redis is not under test
    )
    try:
        return subprocess.run(
            SERVICES[service], env=env, capture_output=True, text=True, timeout=90
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"{service} did not exit (it started?): {e.stderr!r}") from e


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_item_b3_service_as_superuser_exits_2(db, service):
    done = start_in_prod(service, db["superuser"])
    assert done.returncode == 2, done.stderr
    assert superuser_problem(done.stderr, db["superuser_name"]), done.stderr


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_item_b3_service_as_table_owner_exits_2(db, service):
    done = start_in_prod(service, db["owner"])
    assert done.returncode == 2, done.stderr
    assert OWNER.search(done.stderr), done.stderr


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_item_b3_service_as_app_role_has_no_role_problem(db, service):
    """Guard: connected as loom_app, the role check finds nothing (the service may still
    refuse for the other production problems)."""
    done = start_in_prod(service, db["app"])
    assert not superuser_problem(done.stderr, APP_ROLE), done.stderr
    assert not OWNER.search(done.stderr), done.stderr
