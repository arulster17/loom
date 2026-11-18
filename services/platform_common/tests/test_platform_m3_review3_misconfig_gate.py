"""B3 after the second review (docs/m3-review-fixes-spec.md, "Billing", B3, "Only an outage
counts as unreachable"): a misconfiguration exits 2 "at start and in the gate".

Outside dev, the billing worker and the gateway start with Postgres unreachable (a
`QuietPgProxy` that refuses connections), so the role check is deferred to the gate.
Then the proxy opens onto the test server, which rejects the configured credentials (a
fresh login, given a wrong password). The gate's connection now fails with `password
authentication failed`: a misconfiguration, not an outage, so the service exits 2
naming it, instead of retrying forever."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from platform_m3_review2_kit import PG_WARNING, QuietPgProxy
from platform_m3_review3_kit import (
    COMMANDS,
    PASSWORD_FAILED,
    SERVICES,
    Login,
    RedisFront,
    fresh_login,
    redis_front,
    service_env_for,
    wrong_password,
    wrong_password_rejected,
)
from platform_m3_review_proxy import Service, proxied_url
from platform_m3_roles import admin_info

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

WARN_S = 30  # the unreachable-at-start warning
EXIT_S = 45  # after the proxy opens: the gate retries every few seconds at most


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


@pytest.fixture
def proxy(info) -> Iterator[QuietPgProxy]:
    p = QuietPgProxy(info.get("host", "localhost"), int(info.get("port", "5432")))
    try:
        yield p
    finally:
        p.close()


@pytest.mark.parametrize("kind", SERVICES)
def test_item3_wrong_password_in_gate_exits_2(info, login, rf, proxy, tmp_path, kind):
    """done-when B3 (misconfiguration in the deferred gate): started with Postgres
    unreachable (warns, keeps running), then Postgres becomes reachable but rejects the
    password. The service exits 2 naming `password authentication failed`, rather than
    retrying it as an outage."""
    wrong_password_rejected(info, login)
    url = proxied_url(
        proxy,
        dbname=info.get("dbname", "postgres"),
        user=login.name,
        password=wrong_password(login),
    )
    svc = Service(COMMANDS[kind], service_env_for(kind, rf, url, tmp_path))
    try:
        warned = svc.wait_for(PG_WARNING, WARN_S)
        assert svc.returncode is None, (
            f"the {kind} exited ({svc.returncode}) while Postgres was unreachable at start "
            f"(spec: start with a warning):\n{svc.output}"
        )
        assert warned, f"no warning about Postgres being unreachable at start:\n{svc.output}"
        assert proxy.sessions() == [], "connected although the proxy refused connections"

        proxy.open()
        code = svc.wait_exit(EXIT_S)
        out = svc.output
        assert any(s.user == login.name for s in proxy.sessions_since_open()), (
            f"the {kind} never tried Postgres once it was reachable:\n{out}"
        )
        assert code is not None, (
            f"the {kind} kept running {EXIT_S}s after Postgres rejected its password "
            "(spec: a misconfiguration exits 2 in the gate, it is not retried):\n" + out
        )
        assert code == 2, f"the {kind} exited {code}, not 2:\n{out}"
        assert PASSWORD_FAILED.search(out), (
            f"the exit-2 message does not name `password authentication failed`:\n{out}"
        )
    finally:
        svc.stop()
