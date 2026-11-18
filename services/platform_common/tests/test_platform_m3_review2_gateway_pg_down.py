"""B3, "Postgres unreachable at start", for the gateway (docs/m3-review-fixes-spec.md,
"Billing", B3; the rule covers every Python service: gateway, billing worker and pool
manager).

Outside dev, the gateway (`python -m loom_gateway`) is started with Postgres unreachable
and every static production problem cleared: throwaway certificates for mTLS to pods,
non-default CORS origins, a spool path, a random database password and a TLS Redis URL
with a password. Redis is reachable (`TlsRedis` in front of the test Redis, accepting
any AUTH password), so a 503 cannot come from Redis being down: the request's key is
unknown to Redis and needs Postgres.

Postgres is reached through `QuietPgProxy` (a `PgProxy`: refuses connections until
`open()`, then forwards and records every statement before forwarding it). Statements
are read only after every proxied session has ended. "Work" is what
`platform_m3_review_proxy.work_classifier` calls work: a statement that refers to one of
the schema's tables, views, sequences or functions (a mere mention in a literal, comment
or alias is not), or one that writes, locks, listens or notifies.

Roles (fresh, dropped afterwards, each drop on its own): a runtime login made through
the spec's seams (`upgrade(url, group_role=...)`, `ensure_app_login(...)`) with a fresh
group, a non-superuser login that owns a table, and a superuser login.

The pool manager is not covered here: outside dev its only allowed provider is RunPod
(local is dev only, fake is refused), so starting it needs a RunPod API key, a lease
bucket and a proxy bundle, and it would talk to the paid RunPod API."""

from __future__ import annotations

import secrets
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import psycopg
import pytest
from platform_m3_review2_kit import (
    PG_WARNING,
    REPO_ROOT,
    QuietPgProxy,
    TlsRedis,
    assert_start_warning_wording,
    drop_all,
    free_port,
    make_certs,
    owner_problem,
    redis_upstream,
    role_problem,
    superuser_problem,
)
from platform_m3_review_proxy import (
    Service,
    proxied_url,
    schema_object_names,
    service_env,
    work_classifier,
)
from platform_m3_roles import admin_info, admin_url, create_database
from psycopg import sql

from loom_platform.apikeys import generate_api_key
from loom_platform.migrate import ensure_app_login, upgrade

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

GATEWAY = [sys.executable, "-m", "loom_gateway"]
CHAT = {"model": "qwen3-8b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4}


@pytest.fixture(scope="module")
def roles() -> Iterator[dict]:
    info = admin_info()
    tag = uuid.uuid4().hex[:10]
    names = {
        "superuser": f"loom_t_r2s_{tag}",
        "owner": f"loom_t_r2o_{tag}",
        "runtime": f"loom_t_r2l_{tag}",
        "group": f"loom_t_r2g_{tag}",
    }
    pw = {k: secrets.token_hex(16) for k in ("runtime", "owner", "superuser")}
    db: str | None = None
    try:
        db = create_database(info)
        admin = admin_url(info, db)
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
        # superuser first; every drop runs even if an earlier one fails
        drop_all(info, db, [names["superuser"], names["owner"], names["runtime"], names["group"]])


@pytest.fixture(scope="module")
def certs(tmp_path_factory: pytest.TempPathFactory):
    return make_certs(tmp_path_factory.mktemp("review2-gw-certs"))


@pytest.fixture
def tls_redis(certs) -> Iterator[tuple[TlsRedis, int]]:
    host, port, db = redis_upstream()
    front = TlsRedis((host, port), certs)
    try:
        yield front, db
    finally:
        front.close()


@pytest.fixture
def proxy(roles) -> Iterator[QuietPgProxy]:
    info = roles["info"]
    p = QuietPgProxy(info.get("host", "localhost"), int(info.get("port", "5432")))
    try:
        yield p
    finally:
        p.close()


class Gateway:
    def __init__(self, svc: Service, port: int) -> None:
        self.svc = svc
        self.base = f"http://127.0.0.1:{port}"

    def wait_healthy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.svc.returncode is None:
            try:
                if httpx.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        return False

    def chat(self, timeout_s: float = 35) -> httpx.Response:
        key = generate_api_key().plaintext  # fresh: never cached anywhere
        return httpx.post(
            f"{self.base}/v1/chat/completions",
            json=CHAT,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout_s,
        )


@pytest.fixture
def start_gateway(roles, proxy, tls_redis, certs, tmp_path: Path) -> Iterator:
    started: list[Service] = []
    front, redis_db = tls_redis

    def start(role: str) -> Gateway:
        port = free_port()
        env = service_env(
            LOOM_ENV="prod",
            LOOM_DATABASE_URL=proxied_url(
                proxy,
                dbname=roles["db"],
                user=roles["names"][role],
                password=roles["passwords"][role],
            ),
            LOOM_REDIS_URL=front.url(redis_db, secrets.token_hex(12), certs.ca),
            LOOM_POD_CA_CERT=str(certs.ca),
            LOOM_GATEWAY_CLIENT_CERT=str(certs.client_cert),
            LOOM_GATEWAY_CLIENT_KEY=str(certs.client_key),
            LOOM_CORS_ORIGINS="https://app.review2.example",
            LOOM_SETTLE_SPOOL_PATH=str(tmp_path / f"spool-{role}.jsonl"),
            LOOM_GATEWAY_HOST="127.0.0.1",
            LOOM_GATEWAY_PORT=str(port),
            LOOM_GATEWAY_METRICS_PORT=str(free_port()),
            LOOM_USAGE_STREAM=f"loom:test:usage:{uuid.uuid4().hex[:12]}",
            LOOM_SHUTDOWN_GRACE_S="0",
            LOOM_LOG_LEVEL="INFO",
            LOOM_MODELS_YAML=str(REPO_ROOT / "config" / "models.yaml"),
        )
        svc = Service(GATEWAY, env)
        started.append(svc)
        return Gateway(svc, port)

    try:
        yield start
    finally:
        for svc in started:
            svc.stop()


def _started_while_down(gw: Gateway, warn_s: float = 25) -> str:
    """The gateway warned about Postgres and is still running; returns the warning line
    (the first line naming Postgres it wrote)."""
    svc = gw.svc
    warned = svc.wait_line(PG_WARNING, warn_s)
    assert svc.returncode != 2, (
        "exited 2 while Postgres was unreachable at start (spec: start with a warning "
        f"instead of refusing):\n{svc.output}"
    )
    assert warned is not None, f"no warning about Postgres being unreachable:\n{svc.output}"
    assert svc.returncode is None, (
        f"exited ({svc.returncode}) before Postgres came up:\n{svc.output}"
    )
    return warned[1]


def test_item1a_gateway_pg_down_at_start_warns_stays_up_and_answers_503(start_gateway, proxy):
    """done-when B3 (unreachable at start, gateway): outside dev with Postgres
    unreachable, the gateway does not exit 2, logs one warning line naming Postgres
    (without the role problems' wording), stays up, and answers /v1/chat/completions
    with a 503, not a crash."""
    gw = start_gateway("runtime")
    assert_start_warning_wording(_started_while_down(gw))
    assert gw.wait_healthy(15), f"the gateway never answered /health:\n{gw.svc.output}"

    try:
        resp = gw.chat()
    except httpx.HTTPError as e:
        pytest.fail(f"/v1/chat/completions failed ({e!r}) instead of a 503:\n{gw.svc.output}")
    assert resp.status_code == 503, (
        f"/v1/chat/completions with Postgres down answered {resp.status_code} "
        f"{resp.text[:300]!r}, not 503:\n{gw.svc.output}"
    )
    assert gw.svc.returncode is None, f"the gateway exited after the request:\n{gw.svc.output}"
    assert proxy.sessions() == [], "connected although Postgres was unreachable"


@pytest.mark.parametrize("role", ["owner", "superuser"])
def test_item1b_gateway_wrong_role_exits_2_without_work_once_reachable(
    roles, start_gateway, proxy, role
):
    """done-when B3 (unreachable at start, gateway): started while Postgres is
    unreachable as the wrong role (owns a table / superuser). Once Postgres is reachable
    the gateway exits 2 naming the role problem, and no work statement reached Postgres
    on any connection before that (statements read after every session ended)."""
    gw = start_gateway(role)
    _started_while_down(gw, warn_s=25)
    proxy.open()
    code = gw.svc.wait_exit(55)
    out = gw.svc.output
    assert code == 2, f"exit {code} as the {role} role once Postgres was reachable:\n{out}"
    assert proxy.wait_quiet(10), "a proxied session never ended after the gateway exited"

    login = roles["names"][role]
    if role == "superuser":
        assert superuser_problem(out, login), out
    else:
        assert owner_problem(out), out
    sent = [q for s in proxy.sessions_since_open() for q in s.statements]
    assert sent, "exited 2 without sending any statement: the role was never checked"
    work = [q for q in sent if roles["is_work"](q)]
    assert work == [], f"work reached Postgres as the wrong role before the check: {work}"


def test_item1c_gateway_runtime_role_starts_once_reachable(roles, start_gateway, proxy):
    """done-when B3 (guard): started while Postgres is unreachable as a fresh runtime
    login (made through `upgrade(url, group_role=...)` + `ensure_app_login`); once
    Postgres is reachable it serves normally: an unknown key gets 401 (looked up in
    Postgres), no role problem, no exit."""
    gw = start_gateway("runtime")
    _started_while_down(gw, warn_s=20)
    proxy.open()
    login = roles["names"]["runtime"]

    status = None
    deadline = time.monotonic() + 55
    while time.monotonic() < deadline and gw.svc.returncode is None:
        try:
            status = gw.chat(timeout_s=10).status_code
        except httpx.HTTPError:
            status = None
        if status == 401:
            break
        time.sleep(0.5)

    out = gw.svc.output
    assert gw.svc.returncode is None, f"exited ({gw.svc.returncode}) as the runtime role:\n{out}"
    assert not role_problem(out, login), out
    assert status == 401, (
        f"an unknown key still got {status} once Postgres was reachable (expected 401 "
        f"after a Postgres lookup):\n{out}"
    )
    assert any(s.user == login for s in proxy.sessions_since_open())
