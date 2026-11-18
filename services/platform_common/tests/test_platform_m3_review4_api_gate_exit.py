"""B3 after the third review (docs/m3-review-fixes-spec.md, "Billing", B3, "Postgres
unreachable at start"): the billing HTTP API mode (`python -m loom_billing api`) exits 2
when its deferred role check fails, the same way the worker does, "without having done
any work as the wrong role".

The api starts outside dev with Postgres unreachable (a `QuietPgProxy` not yet open),
so it warns and the role check is deferred to the gate. Then the proxy opens onto a
fresh migrated database on the test server, where the configured login is a superuser
(a fresh role; both dropped afterwards). The deferred check fails: the api exits 2
naming the problem, and no work statement reached Postgres."""

from __future__ import annotations

import secrets
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from platform_m3_review2_kit import (
    PG_WARNING,
    QuietPgProxy,
    drop_all,
    free_port,
    superuser_problem,
)
from platform_m3_review3_kit import Login, RedisFront, fresh_login, redis_front, worker_env
from platform_m3_review_proxy import Service, proxied_url, schema_object_names, work_classifier
from platform_m3_roles import admin_info, admin_url, create_database

from loom_platform.migrate import upgrade

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

API = [sys.executable, "-m", "loom_billing", "api"]
WARN_S = 30  # the unreachable-at-start warning
EXIT_S = 45  # after the proxy opens: the gate retries every few seconds at most


@pytest.fixture(scope="module")
def info() -> dict[str, str]:
    return admin_info()


@pytest.fixture(scope="module")
def migrated(info) -> Iterator[tuple[str, list[str]]]:
    """A fresh database with the schema (migrated through the spec's seam, with a fresh
    group role), and its schema objects: so a work query the api sent would name one of
    them and be seen as work (an unmigrated database has none to name)."""
    group = f"loom_t_r4g_{uuid.uuid4().hex[:12]}"
    db: str | None = None
    try:
        db = create_database(info)
        upgrade(admin_url(info, db), group_role=group)
        with psycopg.connect(admin_url(info, db), autocommit=True) as conn:
            names = schema_object_names(conn)
        assert names, "the migrated database has no schema objects"
        yield db, names
    finally:
        drop_all(info, db, [group])


@pytest.fixture(scope="module")
def superuser(info) -> Iterator[Login]:
    with fresh_login(info, superuser=True) as lg:
        yield lg


@pytest.fixture
def rf(tmp_path: Path) -> Iterator[RedisFront]:
    with redis_front(tmp_path) as front:
        yield front


@pytest.fixture
def proxy(info) -> Iterator[QuietPgProxy]:
    p = QuietPgProxy(info.get("host", "localhost"), int(info.get("port", "5432")))
    try:
        yield p
    finally:
        p.close()


def _api_env(rf: RedisFront, url: str) -> dict[str, str]:
    env = worker_env(rf, url)
    env.update(
        LOOM_BILLING_HOST="127.0.0.1",
        LOOM_BILLING_PORT=str(free_port()),
        LOOM_BILLING_INTERNAL_TOKEN=secrets.token_hex(24),
        LOOM_STRIPE_MODE="stripe",
        STRIPE_SECRET_KEY="sk_test_" + secrets.token_hex(12),
        STRIPE_WEBHOOK_SECRET="whsec_" + secrets.token_hex(12),
    )
    return env


def test_item4_api_exits_2_when_the_deferred_role_check_fails(info, migrated, superuser, rf, proxy):
    """done-when B3 (billing api mode): started outside dev with Postgres unreachable
    (warns, keeps running), then Postgres becomes reachable and the login turns out to
    be a superuser. The api exits 2 naming the problem, with no work query sent."""
    dbname, names = migrated
    is_work = work_classifier(names)
    url = proxied_url(proxy, dbname=dbname, user=superuser.name, password=superuser.password)
    svc = Service(API, _api_env(rf, url))
    try:
        warned = svc.wait_for(PG_WARNING, WARN_S)
        assert svc.returncode is None, (
            f"the api exited ({svc.returncode}) while Postgres was unreachable at start "
            f"(spec: start with a warning):\n{svc.output}"
        )
        assert warned, f"no warning about Postgres being unreachable at start:\n{svc.output}"

        proxy.open()
        code = svc.wait_exit(EXIT_S)
        out = svc.output
        assert any(s.user == superuser.name for s in proxy.sessions_since_open()), (
            f"the api never tried Postgres once it was reachable:\n{out}"
        )
        assert code is not None, (
            f"the api kept running {EXIT_S}s after its deferred role check found a "
            "superuser (spec B3: the service exits 2):\n" + out
        )
        assert code == 2, f"the api exited {code}, not 2 (spec B3):\n{out}"
        assert superuser_problem(out, superuser.name), (
            f"the exit-2 message does not name the superuser problem:\n{out}"
        )
        assert proxy.wait_quiet(15), "a proxied session never ended after the api exited"
        work = [(u, q) for u, q in proxy.statements() if is_work(q)]
        assert work == [], f"the api ran work as the wrong role before exiting: {work}"
    finally:
        svc.stop()
