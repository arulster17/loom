"""Helpers for the third round of B3 review tests (test_platform_m3_review3_*.py, spec
docs/m3-review-fixes-spec.md, "Billing", B3, "Only an outage counts as unreachable").

Builds on platform_m3_review_proxy and platform_m3_review2_kit (imported, not changed):

- `fresh_login`: a throwaway LOGIN role with a random password, dropped afterwards.
- `wrong_password_rejected` / `unknown_role_reads_as`: what the test server answers to a
  wrong password and to an unknown role (it depends on its pg_hba auth method: with
  `trust` a wrong password is accepted, with `scram-sha-256`/`md5` an unknown role reads
  as a wrong password).
- `redis_front`: `TlsRedis` in front of the test Redis with fresh stream names, so a
  service gets a production-shaped `rediss://:<password>@` URL (no static production
  problem) and still reaches Redis; the streams are deleted afterwards.
- `worker_env` / `gateway_env`: production-shaped (`LOOM_ENV=prod`) environments with
  every static production problem cleared, as in the review2 tests.
- `pg_unreachable_lines`: lines that call Postgres unreachable (a misconfiguration must
  not be reported, or handled, as an outage)."""

from __future__ import annotations

import contextlib
import re
import secrets
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
import redis
from platform_m3_review2_kit import (
    REPO_ROOT,
    Certs,
    TlsRedis,
    drop_roles,
    free_port,
    make_certs,
    redis_upstream,
)
from platform_m3_review_proxy import service_env
from platform_m3_roles import admin_url, url_for
from psycopg import sql

GATEWAY = [sys.executable, "-m", "loom_gateway"]
WORKER = [sys.executable, "-m", "loom_billing", "worker"]
SERVICES = ("gateway", "worker")
COMMANDS = {"gateway": GATEWAY, "worker": WORKER}

PASSWORD_FAILED = re.compile(r"password authentication failed", re.IGNORECASE)
_UNREACHABLE = re.compile(r"\bunreachable\b", re.IGNORECASE)
_PG = re.compile(r"postgres|database", re.IGNORECASE)


def pg_unreachable_lines(output: str) -> list[str]:
    """Lines saying Postgres (or the database) is unreachable."""
    return [line for line in output.splitlines() if _UNREACHABLE.search(line) and _PG.search(line)]


def role_missing(name: str) -> re.Pattern[str]:
    return re.compile(rf'role "?{re.escape(name)}"? does not exist', re.IGNORECASE)


def database_missing(name: str) -> re.Pattern[str]:
    return re.compile(rf'database "?{re.escape(name)}"? does not exist', re.IGNORECASE)


# ------------------------------------------------------------------ roles


@dataclass(frozen=True)
class Login:
    name: str
    password: str


def _server_url(info: dict[str, str]) -> str:
    return admin_url(info, info.get("dbname", "postgres"))


@contextlib.contextmanager
def fresh_login(info: dict[str, str], *, superuser: bool = False) -> Iterator[Login]:
    """A fresh LOGIN role (NOSUPERUSER unless asked), dropped afterwards."""
    name = f"loom_t_r3_{uuid.uuid4().hex[:12]}"
    password = secrets.token_hex(16)
    attrs = sql.SQL("SUPERUSER" if superuser else "NOSUPERUSER")
    try:
        with psycopg.connect(_server_url(info), autocommit=True) as c:
            c.execute(
                sql.SQL("CREATE ROLE {} LOGIN {} PASSWORD {}").format(
                    sql.Identifier(name), attrs, sql.Literal(password)
                )
            )
        yield Login(name, password)
    finally:
        drop_roles(info, [name])


def wrong_password(login: Login) -> str:
    return login.password + "-wrong"


def wrong_password_rejected(info: dict[str, str], login: Login) -> None:
    """Skip unless the test server rejects a wrong password with the usual message (a
    `trust` server accepts any password: nothing to test)."""
    url = url_for(
        info, dbname=info.get("dbname", "postgres"), user=login.name, password=wrong_password(login)
    )
    try:
        psycopg.connect(url, connect_timeout=5).close()
    except psycopg.OperationalError as e:
        if PASSWORD_FAILED.search(str(e)):
            return
        pytest.fail(f"unexpected answer to a wrong password: {e}")
    pytest.skip("the test server accepts a wrong password (trust auth): nothing to refuse")


def unknown_role_reads_as(info: dict[str, str], name: str) -> str:
    """'role_missing' when the server says the role does not exist, 'wrong_password' when
    it answers as for a wrong password (scram/md5 auth hides unknown roles)."""
    url = url_for(info, dbname=info.get("dbname", "postgres"), user=name, password="x")
    try:
        psycopg.connect(url, connect_timeout=5).close()
    except psycopg.OperationalError as e:
        if role_missing(name).search(str(e)):
            return "role_missing"
        if PASSWORD_FAILED.search(str(e)):
            return "wrong_password"
        pytest.fail(f"unexpected answer for an unknown role: {e}")
    pytest.fail(f"connected as the unknown role {name}")


# ------------------------------------------------------------------ Redis


@dataclass(frozen=True)
class RedisFront:
    front: TlsRedis
    url: str
    certs: Certs
    db: int
    streams: dict[str, str]


@contextlib.contextmanager
def redis_front(directory: Path) -> Iterator[RedisFront]:
    host, port, db = redis_upstream()
    directory.mkdir(parents=True, exist_ok=True)
    certs = make_certs(directory)
    front = TlsRedis((host, port), certs)
    tag = uuid.uuid4().hex[:12]
    streams = {
        "LOOM_USAGE_STREAM": f"loom:test:usage:{tag}",
        "LOOM_USAGE_DEAD_STREAM": f"loom:test:usage:{tag}:dead",
        "LOOM_NOTIFY_STREAM": f"loom:test:usage:{tag}:notify",
    }
    try:
        yield RedisFront(front, front.url(db, secrets.token_hex(12), certs.ca), certs, db, streams)
    finally:
        try:
            front.close()
        finally:
            plain = redis.Redis(host=host, port=port, db=db)
            try:
                plain.delete(*streams.values())
            finally:
                plain.close()


# ------------------------------------------------------------------ environments


def worker_env(rf: RedisFront, database_url: str) -> dict[str, str]:
    return service_env(
        LOOM_ENV="prod",
        LOOM_DATABASE_URL=database_url,
        LOOM_REDIS_URL=rf.url,
        LOOM_BILLING_METRICS_PORT="0",
        LOOM_LOG_LEVEL="INFO",
        **rf.streams,
    )


def gateway_env(
    rf: RedisFront,
    database_url: str,
    *,
    port: int,
    spool: Path,
    pod_ca: Path | None = None,
    client_cert: Path | None = None,
    client_key: Path | None = None,
    models_yaml: Path | None = None,
    usage_stream: str | None = None,
    grace_s: str = "0",
) -> dict[str, str]:
    return service_env(
        LOOM_ENV="prod",
        LOOM_DATABASE_URL=database_url,
        LOOM_REDIS_URL=rf.url,
        LOOM_POD_CA_CERT=str(pod_ca or rf.certs.ca),
        LOOM_GATEWAY_CLIENT_CERT=str(client_cert or rf.certs.client_cert),
        LOOM_GATEWAY_CLIENT_KEY=str(client_key or rf.certs.client_key),
        LOOM_CORS_ORIGINS="https://app.review3.example",
        LOOM_SETTLE_SPOOL_PATH=str(spool),
        LOOM_GATEWAY_HOST="127.0.0.1",
        LOOM_GATEWAY_PORT=str(port),
        LOOM_GATEWAY_METRICS_PORT=str(free_port()),
        LOOM_USAGE_STREAM=usage_stream or rf.streams["LOOM_USAGE_STREAM"],
        LOOM_SHUTDOWN_GRACE_S=grace_s,
        LOOM_LOG_LEVEL="INFO",
        LOOM_MODELS_YAML=str(models_yaml or REPO_ROOT / "config" / "models.yaml"),
    )


def service_env_for(kind: str, rf: RedisFront, database_url: str, tmp_path: Path) -> dict[str, str]:
    if kind == "worker":
        return worker_env(rf, database_url)
    return gateway_env(
        rf, database_url, port=free_port(), spool=tmp_path / f"spool-{uuid.uuid4().hex[:8]}.jsonl"
    )
