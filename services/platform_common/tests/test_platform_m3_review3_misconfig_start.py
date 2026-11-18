"""B3 after the second review (docs/m3-review-fixes-spec.md, "Billing", B3, "Only an outage
counts as unreachable"): outside dev, a Postgres connection error that is a
misconfiguration (`password authentication failed`, a role or database that `does not
exist`) exits 2 at start, naming the error, as before. It is not an outage: the service
must not warn that Postgres is unreachable and keep running.

The billing worker and the gateway get production-shaped configurations (no static
production problem; Redis reachable through `TlsRedis`) and a `LOOM_DATABASE_URL` that
points straight at the test server with a fresh login (dropped afterwards):

- a wrong password for that login;
- that login's correct password and a database that does not exist;
- a role that does not exist. On a server whose auth method hides unknown roles
  (scram-sha-256 / md5 answer `password authentication failed`), this case reads as a
  wrong password and is skipped (the wrong-password case covers it)."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from platform_m3_review3_kit import (
    COMMANDS,
    PASSWORD_FAILED,
    SERVICES,
    Login,
    RedisFront,
    database_missing,
    fresh_login,
    pg_unreachable_lines,
    redis_front,
    role_missing,
    service_env_for,
    unknown_role_reads_as,
    wrong_password,
    wrong_password_rejected,
)
from platform_m3_review_proxy import Service
from platform_m3_roles import admin_info, url_for

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

EXIT_S = 45  # a refusal at start takes seconds; generous for a loaded CI runner


@pytest.fixture(scope="module")
def info() -> dict[str, str]:
    return admin_info()


@pytest.fixture(scope="module")
def login(info) -> Iterator[Login]:
    with fresh_login(info) as lg:
        yield lg


@pytest.fixture
def rf(tmp_path: Path) -> Iterator[RedisFront]:
    with redis_front(tmp_path) as front:
        yield front


def _refuses_at_start(kind: str, env: dict[str, str], named: re.Pattern[str]) -> None:
    svc = Service(COMMANDS[kind], env)
    try:
        code = svc.wait_exit(EXIT_S)
        out = svc.output
        assert code is not None, (
            f"the {kind} kept running {EXIT_S}s with a misconfigured LOOM_DATABASE_URL "
            f"(spec: a misconfiguration exits 2 at start, it is not an outage):\n{out}"
        )
        assert code == 2, f"the {kind} exited {code}, not 2:\n{out}"
        assert named.search(out), (
            f"the exit-2 message does not name the error ({named.pattern}):\n{out}"
        )
        unreachable = pg_unreachable_lines(out)
        assert unreachable == [], (
            f"the {kind} reported the misconfiguration as Postgres unreachable:\n{unreachable}"
        )
    finally:
        svc.stop()


@pytest.mark.parametrize("kind", SERVICES)
def test_item2_wrong_password_at_start_exits_2(info, login, rf, tmp_path, kind):
    """done-when B3 (misconfiguration at start): a wrong password for an existing login
    exits 2 naming `password authentication failed`; no 'Postgres unreachable' warning."""
    wrong_password_rejected(info, login)
    url = url_for(
        info, dbname=info.get("dbname", "postgres"), user=login.name, password=wrong_password(login)
    )
    _refuses_at_start(kind, service_env_for(kind, rf, url, tmp_path), PASSWORD_FAILED)


@pytest.mark.parametrize("kind", SERVICES)
def test_item2_missing_database_at_start_exits_2(info, login, rf, tmp_path, kind):
    """done-when B3 (misconfiguration at start): the right credentials and a database
    that does not exist exit 2 naming `database "..." does not exist`."""
    dbname = f"loom_t_r3_nodb_{uuid.uuid4().hex[:12]}"  # never created
    url = url_for(info, dbname=dbname, user=login.name, password=login.password)
    _refuses_at_start(kind, service_env_for(kind, rf, url, tmp_path), database_missing(dbname))


@pytest.mark.parametrize("kind", SERVICES)
def test_item2_unknown_role_at_start_exits_2(info, rf, tmp_path, kind):
    """done-when B3 (misconfiguration at start): a role that does not exist exits 2
    naming `role "..." does not exist`, where the server's auth method says so (skipped
    where it reads as a wrong password)."""
    name = f"loom_t_r3_norole_{uuid.uuid4().hex[:12]}"  # never created
    if unknown_role_reads_as(info, name) == "wrong_password":
        pytest.skip(
            "this server's auth method answers an unknown role with 'password "
            "authentication failed': it reads as a wrong password (covered above)"
        )
    url = url_for(info, dbname=info.get("dbname", "postgres"), user=name, password="x")
    _refuses_at_start(kind, service_env_for(kind, rf, url, tmp_path), role_missing(name))
