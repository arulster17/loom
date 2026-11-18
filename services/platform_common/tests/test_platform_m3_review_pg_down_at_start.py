"""B3 after review (docs/m3-review-fixes-spec.md, "Billing", B3, "Postgres unreachable at
start"): outside dev, a service whose Postgres is unreachable at start starts with a
warning instead of refusing (exit 2). The role check runs on the first successful
connection and gates the pool: no work query runs on any pool connection until the check
has passed; if it fails, the service logs the problem and exits 2 without having done
any work as the wrong role.

The service is billing's worker (`python -m loom_billing worker`): it is the one whose
production problems can all be cleared without certificates (stripe mode, no HTTP API),
so a refusal here can only come from the database. Its Redis URL has a password and TLS
but points at a closed port (Redis is not under test; an unreachable Redis is only a
warning at start). Its metrics server is off.

Postgres is reached through `PgProxy` (platform_m3_review_proxy): refusing connections
at first, then forwarding to the real server. The proxy records every statement every
connection sends, before forwarding it. "Work" is what `work_classifier` calls work: a
statement that refers to one of the schema's tables, views, sequences or functions (a
mere mention in a literal, comment or alias is not), or that writes, locks, listens or
notifies; the role check only reads the catalogs.

Roles (fresh, dropped afterwards): a runtime login made by the spec's seams
(`upgrade(url, group_role=...)` + `ensure_app_login`, with a fresh group, so loom_app and
loom_runtime are not touched), a non-superuser login that owns a table, and a superuser
login. Each has a random password (no dev-password problem)."""

from __future__ import annotations

import re
import secrets
import sys
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from platform_m3_review2_kit import drop_all
from platform_m3_review_proxy import (
    PgProxy,
    Service,
    proxied_url,
    schema_object_names,
    service_env,
    work_classifier,
)
from platform_m3_roles import admin_info, admin_url, create_database
from psycopg import sql

from loom_platform.migrate import ensure_app_login, upgrade

pytestmark = [pytest.mark.postgres]

WORKER = [sys.executable, "-m", "loom_billing", "worker"]
# Waits, at most 90 s in all per test (pytest's timeout is 120 s, so a failing wait ends
# in its assertion, not in a timeout): the unreachable-at-start warning, then what the
# worker does once Postgres is reachable (gate retry of a few seconds, then the check).
WARN_S = 30.0
REACHABLE_S = 60.0
# A warning about Postgres / the database: one line naming both.
_WARN = r"\bwarn(?:ing)?\b"
_PG = r"(?:postgres|database)"
PG_WARNING = re.compile(rf"{_WARN}[^\n]*{_PG}|{_PG}[^\n]*{_WARN}", re.IGNORECASE)
_OWN = r"\bown(?:s|er|ed|ership)\b"
_TABLE = r"\btables?\b"
OWNER = re.compile(rf"{_OWN}[^\n]{{0,80}}{_TABLE}|{_TABLE}[^\n]{{0,80}}{_OWN}", re.IGNORECASE)


def superuser_problem(output: str, role: str) -> bool:
    who = rf"(?:\b{re.escape(role)}\b|\brole\b)"
    pattern = rf"{who}[^\n]{{0,80}}\bsuperuser\b|\bsuperuser\b[^\n]{{0,80}}{who}"
    return re.search(pattern, output, re.IGNORECASE) is not None


def role_problem(output: str, role: str) -> bool:
    return superuser_problem(output, role) or OWNER.search(output) is not None


@pytest.fixture(scope="module")
def roles() -> Iterator[dict]:
    info = admin_info()
    db = create_database(info)
    tag = uuid.uuid4().hex[:10]
    names = {
        "group": f"loom_t_rvg_{tag}",
        "runtime": f"loom_t_rvl_{tag}",
        "owner": f"loom_t_rvo_{tag}",
        "superuser": f"loom_t_rvs_{tag}",
    }
    pw = {k: secrets.token_hex(16) for k in ("runtime", "owner", "superuser")}
    admin = admin_url(info, db)
    try:
        upgrade(admin, group_role=names["group"])
        ensure_app_login(
            admin, login=names["runtime"], group_role=names["group"], password=pw["runtime"]
        )
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(names["owner"]), sql.Literal(pw["owner"])
                )
            )
            conn.execute(
                sql.SQL("ALTER TABLE orgs OWNER TO {}").format(sql.Identifier(names["owner"]))
            )
            conn.execute(
                sql.SQL("CREATE ROLE {} LOGIN SUPERUSER PASSWORD {}").format(
                    sql.Identifier(names["superuser"]), sql.Literal(pw["superuser"])
                )
            )
            objects = schema_object_names(conn)
        yield {
            "info": info,
            "db": db,
            "names": names,
            "passwords": pw,
            "is_work": work_classifier(objects),
        }
    finally:
        # the database first, then each role: every drop runs even if an earlier one fails
        drop_all(info, db, [names[k] for k in ("runtime", "owner", "superuser", "group")])


@pytest.fixture
def proxy(roles) -> Iterator[PgProxy]:
    info = roles["info"]
    p = PgProxy(info.get("host", "localhost"), int(info.get("port", "5432")))
    try:
        yield p
    finally:
        p.close()


@pytest.fixture
def start_worker(roles, proxy) -> Iterator:
    started: list[Service] = []

    def start(role: str) -> Service:
        url = proxied_url(
            proxy,
            dbname=roles["db"],
            user=roles["names"][role],
            password=roles["passwords"][role],
        )
        env = service_env(
            LOOM_ENV="prod",
            LOOM_DATABASE_URL=url,
            LOOM_REDIS_URL=f"rediss://:{secrets.token_hex(12)}@127.0.0.1:1/0",
            LOOM_BILLING_METRICS_PORT="0",
            LOOM_LOG_LEVEL="INFO",
        )
        svc = Service(WORKER, env)
        started.append(svc)
        return svc

    try:
        yield start
    finally:
        for svc in started:
            svc.stop()


def _started_while_down(svc: Service) -> None:
    """The worker warned about Postgres and is still running (it did not refuse)."""
    warned = svc.wait_for(PG_WARNING, WARN_S)
    assert svc.returncode != 2, (
        "exited 2 while Postgres was unreachable at start (spec: start with a warning "
        f"instead of refusing):\n{svc.output}"
    )
    assert warned, f"no warning about Postgres being unreachable:\n{svc.output}"
    assert svc.returncode is None, (
        f"exited ({svc.returncode}) before Postgres came up:\n{svc.output}"
    )


def test_item_b3_unreachable_at_start_warns_and_does_not_exit_2(start_worker, proxy):
    """done-when B3: Postgres unreachable at start, outside dev: the service logs a
    warning and does not exit 2 for that reason (it is still running a few seconds on)."""
    svc = start_worker("runtime")
    _started_while_down(svc)
    code = svc.wait_exit(3)
    assert code != 2, f"exited 2 with Postgres still unreachable:\n{svc.output}"
    assert proxy.sessions() == [], "connected although Postgres was unreachable"


@pytest.mark.parametrize("role", ["owner", "superuser"])
def test_item_b3_wrong_role_exits_2_without_work_once_reachable(roles, start_worker, proxy, role):
    """done-when B3: started while Postgres is unreachable as the wrong role (owns a
    table / superuser); once Postgres is reachable the service exits 2 naming the role
    problem, and no work query ran on any connection before that."""
    svc = start_worker(role)
    _started_while_down(svc)
    proxy.open()
    code = svc.wait_exit(REACHABLE_S)

    assert code == 2, f"exit {code} as the wrong role:\n{svc.output}"
    assert role_problem(svc.output, roles["names"][role]), svc.output
    sent = proxy.statements()
    assert sent, "exited 2 without sending any statement: the role was never checked"
    work = [q for _, q in sent if roles["is_work"](q)]
    assert work == [], f"work ran as the wrong role before the check failed: {work}"


def test_item_b3_runtime_role_starts_once_reachable(roles, start_worker, proxy):
    """done-when B3 (guard for the role part): started while Postgres is unreachable as
    the runtime role; once Postgres is reachable the role check passes and the worker
    goes on to its work, without a role problem and without exiting 2."""
    svc = start_worker("runtime")
    _started_while_down(svc)
    proxy.open()
    login = roles["names"]["runtime"]
    is_work = roles["is_work"]

    def did_work() -> bool:
        return any(u == login and is_work(q) for u, q in proxy.statements())

    deadline_s = REACHABLE_S
    waited = 0.0
    while not did_work() and svc.returncode is None and waited < deadline_s:
        svc.wait_exit(0.25)
        waited += 0.25
    assert svc.returncode != 2, f"exited 2 as the runtime role:\n{svc.output}"
    assert not role_problem(svc.output, login), svc.output
    assert did_work(), (
        f"no work query as {login} once Postgres was reachable (exit {svc.returncode}):\n"
        f"{svc.output}\nstatements: {proxy.statements()}"
    )
