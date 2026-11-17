"""B3 after review, stricter guard (docs/m3-review-fixes-spec.md, "Billing", B3): started
outside dev as `loom_app` with Postgres reachable, each Python service really checks its
role. Its startup output has no "cannot check" role message (and no failed login), and
the check went over a connection that logged in as loom_app and sent a statement. So a
login failure, or a check that never ran, cannot pass as "no role problem" (the looser
guard in test_platform_m3_service_role_startup.py only looks for the role problems).

The services exit 2 here anyway: other production problems (the dev password, Redis
without TLS, certificates) are listed in the same message, and the role check runs even
so. Postgres is reached through `PgProxy`, open from the start, which records the user
and statements of every connection."""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator

import pytest
from platform_m3_review_proxy import PgProxy, Service, proxied_url, service_env
from platform_m3_roles import (
    APP_ROLE,
    admin_info,
    admin_url,
    app_password,
    create_database,
    drop_database,
    run_migrate,
)

pytestmark = [pytest.mark.postgres]

SERVICES = {
    "gateway": [sys.executable, "-m", "loom_gateway"],
    "billing": [sys.executable, "-m", "loom_billing", "worker"],
    "poolmgr": [sys.executable, "-m", "loom_poolmgr.cli", "run"],
}
CANNOT_CHECK = re.compile(r"cannot check", re.IGNORECASE)
LOGIN_FAILED = re.compile(
    r"authentication failed|password authentication|role \S+ does not exist", re.IGNORECASE
)
_OWN = r"\bown(?:s|er|ed|ership)\b"
_TABLE = r"\btables?\b"
OWNER = re.compile(rf"{_OWN}[^\n]{{0,80}}{_TABLE}|{_TABLE}[^\n]{{0,80}}{_OWN}", re.IGNORECASE)
SUPERUSER = re.compile(
    rf"(?:\b{APP_ROLE}\b|\brole\b)[^\n]{{0,80}}\bsuperuser\b|\bsuperuser\b[^\n]{{0,80}}"
    rf"(?:\b{APP_ROLE}\b|\brole\b)",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def db() -> Iterator[dict]:
    """A fresh database set up by loom-migrate (loom_app gets its usual password)."""
    info = admin_info()
    name = create_database(info)
    try:
        done = run_migrate(admin_url(info, name), app_password())
        assert done.returncode == 0, done.stdout + done.stderr
        yield {"info": info, "db": name}
    finally:
        drop_database(info, name)


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_item_b3_app_role_start_has_no_cannot_check_message(db, service):
    """done-when B3 (guard): as loom_app, Postgres reachable: no "cannot check" or
    failed-login message, no role problem, and the role was checked over a connection
    logged in as loom_app."""
    info = db["info"]
    proxy = PgProxy(info.get("host", "localhost"), int(info.get("port", "5432")))
    proxy.open()
    svc: Service | None = None
    try:
        url = proxied_url(proxy, dbname=db["db"], user=APP_ROLE, password=app_password())
        env = service_env(
            LOOM_ENV="prod",
            LOOM_DATABASE_URL=url,
            LOOM_REDIS_URL="redis://127.0.0.1:1/0",  # unreachable: Redis is not under test
            LOOM_BILLING_METRICS_PORT="0",
        )
        svc = Service(SERVICES[service], env)
        code = svc.wait_exit(90)
        out = svc.output
        assert code is not None, f"{service} did not exit (it started?):\n{out}"
        assert not CANNOT_CHECK.search(out), out
        assert not LOGIN_FAILED.search(out), out
        assert not SUPERUSER.search(out), out
        assert not OWNER.search(out), out
        checked = [s for s in proxy.sessions() if s.user == APP_ROLE and s.statements]
        assert checked, (
            f"{service} never sent a statement as {APP_ROLE}: its role was not checked "
            f"(sessions: {proxy.sessions()})\n{out}"
        )
    finally:
        if svc is not None:
            svc.stop()
        proxy.close()
