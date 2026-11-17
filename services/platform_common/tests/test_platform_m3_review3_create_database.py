"""B3 after the second review (docs/m3-review-fixes-spec.md, "Billing", B3):
"`loom-migrate --create-database` checks the role names before creating anything."

`loom-migrate` refuses a configured login or group name that is the connecting role
itself, and a group name that exists as a LOGIN role. With `--create-database`, a
refused name must leave no new database behind.

Every role is fresh and dropped afterwards; every database name is fresh and dropped
afterwards whether or not it was created. The cases naming the connecting role connect
as a fresh superuser (never the server's admin role), so nothing the server already had
can be altered. `LOOM_APP_DATABASE_PASSWORD` is not set: no runtime login is created."""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from platform_m3_review2_kit import drop_roles
from platform_m3_review3_kit import Login, fresh_login
from platform_m3_roles import admin_info, admin_url, drop_database, migrate_command, url_for

pytestmark = [pytest.mark.postgres]


@pytest.fixture(scope="module")
def info() -> dict[str, str]:
    return admin_info()


@pytest.fixture
def new_db(info) -> Iterator[str]:
    name = f"loom_t_r3_cdb_{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        drop_database(info, name)


def _exists(info: dict[str, str], name: str) -> bool:
    with psycopg.connect(admin_url(info, info.get("dbname", "postgres")), autocommit=True) as c:
        row = c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
    return row is not None


def _migrate(url: str, **names: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_")}
    env["LOOM_ENV"] = "dev"
    env.update(names)
    return subprocess.run(
        [*migrate_command(url), "--create-database"],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def _refused(proc: subprocess.CompletedProcess[str], info, db: str, setting: str) -> None:
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"loom-migrate accepted the refused name:\n{out}"
    assert setting in out, f"the refusal does not name {setting}:\n{out}"
    assert not _exists(info, db), (
        f"loom-migrate --create-database created database {db} before refusing {setting} "
        f"(spec: it checks the role names before creating anything):\n{out}"
    )


@pytest.fixture
def superuser(info) -> Iterator[Login]:
    with fresh_login(info, superuser=True) as lg:
        yield lg


def test_item6_login_is_connecting_role_creates_no_database(info, superuser, new_db):
    """done-when B3: LOOM_APP_DATABASE_USER names the connecting role: refused, and no
    database is created."""
    url = url_for(info, dbname=new_db, user=superuser.name, password=superuser.password)
    proc = _migrate(
        url,
        LOOM_APP_DATABASE_USER=superuser.name,
        LOOM_APP_DATABASE_GROUP=f"loom_t_r3_g_{uuid.uuid4().hex[:12]}",
    )
    _refused(proc, info, new_db, "LOOM_APP_DATABASE_USER")


def test_item6_group_is_connecting_role_creates_no_database(info, superuser, new_db):
    """done-when B3: LOOM_APP_DATABASE_GROUP names the connecting role: refused, and no
    database is created."""
    url = url_for(info, dbname=new_db, user=superuser.name, password=superuser.password)
    proc = _migrate(
        url,
        LOOM_APP_DATABASE_USER=f"loom_t_r3_l_{uuid.uuid4().hex[:12]}",
        LOOM_APP_DATABASE_GROUP=superuser.name,
    )
    _refused(proc, info, new_db, "LOOM_APP_DATABASE_GROUP")


def test_item6_group_is_a_login_role_creates_no_database(info, new_db):
    """done-when B3: LOOM_APP_DATABASE_GROUP names an existing LOGIN role: refused, and
    no database is created."""
    with fresh_login(info) as other:
        proc = _migrate(
            admin_url(info, new_db),
            LOOM_APP_DATABASE_USER=f"loom_t_r3_l_{uuid.uuid4().hex[:12]}",
            LOOM_APP_DATABASE_GROUP=other.name,
        )
        _refused(proc, info, new_db, "LOOM_APP_DATABASE_GROUP")


def test_item6_guard_accepted_names_create_the_database(info, new_db):
    """done-when B3 (guard): with fresh names that nothing refuses, `--create-database`
    creates and migrates the database (so the cases above pass for the refusal, not
    because the flag does nothing)."""
    group = f"loom_t_r3_g_{uuid.uuid4().hex[:12]}"
    try:
        proc = _migrate(
            admin_url(info, new_db),
            LOOM_APP_DATABASE_USER=f"loom_t_r3_l_{uuid.uuid4().hex[:12]}",
            LOOM_APP_DATABASE_GROUP=group,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert _exists(info, new_db), f"--create-database did not create {new_db}:\n{out}"
    finally:
        try:
            drop_database(info, new_db)  # before the group: the grants in it name the group
        finally:
            drop_roles(info, [group])
