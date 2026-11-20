"""Helpers for the M3 pool manager tests (docs/phase1-m3-spec.md §6.8, §7, §13 A25-A27).

`loom-poolmgr run` is driven in process through its real entry point (`cli.main`), with
`cli.PoolManager` replaced by `Driven`, a subclass that changes only what keeps a test
apart from the others sharing the Postgres and Redis servers:

- its one model gets an id of its own (a copy of the serving file's first model), and no
  other model is published (`servable=()`), so the pool hashes of the real model ids in
  the shared Redis are never written;
- it connects to the test's database and Redis (outside dev the settings must carry a
  non-compose password and `rediss://`, which no test server has: the URLs the settings
  check are placeholders, and the manager is handed the real ones);
- its admin queue is a key of its own;
- its metrics endpoint binds a port the OS picks (`serve_metrics()`; the port is read
  back from the server, so no other process can take it in between);
- optionally its clock is a `FakeClock` (`clock`/`sleep`, the manager's injected time).
  Everything the manager times by its injected clock follows it, including the gateway
  certificate's 24 h schedule and its 7-day validity (issued "now" on that clock).

Per-model metric series (`creates_total{model,...}`, `pool_reason{model,...}`) are read
for the kit's model id (`PoolmgrRun.model`), the one model the manager is handed: their
initialisation must follow the manager's model set (the models it was built with), not a
list read elsewhere. Scrapes that expect them wait for the first tick (`wait_tick(0)`).

Everything else is the real `run` path: the settings and their production checks, the
serving file, the CA files, `LOOM_POOLMGR_GATEWAY_CERT_DIR`, the metrics endpoint
(`LOOM_POOLMGR_METRICS_PORT`, scraped over HTTP), the real `PoolManager.start()` and
`run()`. `Driven.run` runs the manager's own loop while the test's `scenario` (a
coroutine in the manager's event loop) looks at files and metrics and moves the clock;
when it returns the manager is stopped as on SIGTERM.

Outside dev the database role check (`loom_platform.settings.database_role_problems`) is
replaced by one that finds nothing (it would connect to the placeholder URL), and the
process's role gate by a fresh one, so nothing is deferred into later tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as dt
import os
import re
import socket
import ssl
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import redis as sync_redis
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtendedKeyUsageOID
from poolmgr_fakes import FakeClock, clear_serving_tables, release_manager_locks, unique_model_id
from prometheus_client.parser import text_string_to_metric_families

from loom_bench.registry import REPO_ROOT
from loom_platform import db as platform_db
from loom_platform import settings as platform_settings
from loom_platform.db import RoleGate
from loom_platform.pool import pool_key, wake_key
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV
from loom_poolmgr import cli
from loom_poolmgr import metrics as poolmgr_metrics
from loom_poolmgr.manager import PoolManager
from loom_poolmgr.pki import POOLMGR_CN, CertificateAuthority, IssuedCert, write_issued

COMMITTED_SERVING = REPO_ROOT / "config" / "serving.yaml"
DAY_S = 86400
# The reasons a pool hash can carry (docs/phase1-contracts.md §10).
REASONS = ("crash_loop", "spend_cap", "price_too_high", "not_configured", "no_capacity")
# `loom_poolmgr_creates_total{result}` (M2 spec §11), read by alarm P2a.
CREATE_RESULTS = ("ok", "no_capacity", "price_too_high", "spend_cap", "error")
# Outside dev the platform refuses the compose passwords and a Redis without AUTH/TLS.
PROD_DATABASE_URL = "postgresql+psycopg://loom_app:m3-test-not-a-dev-password@127.0.0.1:9/loom"
PROD_REDIS_URL = "rediss://:m3-test-secret@127.0.0.1:9/0"
EGRESS = "203.0.113.10/32"  # the box's Elastic IP in deploy/serving.m3.yaml (a stand-in)
# How long a gateway-certificate check waits for the manager's write (real time).
FILE_WAIT_S = 10.0


# ------------------------------------------------------------------ files


def serving_file(
    directory: Path, *, provider: str = "local", egress: list[str] | None = None
) -> Path:
    """The committed serving file with `provider` and `gateway_egress_cidrs` (the M3 box
    file `deploy/serving.m3.yaml` is the committed models and timings, `provider: local`,
    the Elastic IP as egress)."""
    doc = yaml.safe_load(COMMITTED_SERVING.read_text())
    doc["provider"] = provider
    doc["network"]["gateway_egress_cidrs"] = [EGRESS] if egress is None else egress
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"serving-{provider}-{uuid.uuid4().hex[:6]}.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


@dataclasses.dataclass(frozen=True)
class CaFiles:
    ca: CertificateAuthority
    ca_cert: Path
    ca_key: Path
    client_cert: Path
    client_key: Path


def write_ca(
    directory: Path, *, days: int = 365, now: float | None = None, cn: str = "Loom m3 test CA"
) -> CaFiles:
    """A pod CA (common name `cn`) valid `days` from `now` plus the manager's client
    certificate, as the files `LOOM_POOLMGR_CA_CERT/_KEY` and
    `LOOM_POOLMGR_CLIENT_CERT/_KEY` name."""
    directory.mkdir(parents=True, exist_ok=True)
    ca = CertificateAuthority.generate(cn, days=days, now=now)
    (directory / "ca.crt").write_bytes(ca.cert_pem())
    (directory / "ca.key").write_bytes(IssuedCert(ca.cert, ca.key).key_pem())
    write_issued(directory, POOLMGR_CN, ca.issue_client(now=now))
    return CaFiles(
        ca,
        directory / "ca.crt",
        directory / "ca.key",
        directory / f"{POOLMGR_CN}.crt",
        directory / f"{POOLMGR_CN}.key",
    )


def combined_pem(issued: IssuedCert, *, key_first: bool = True) -> bytes:
    """Key and certificate in one PEM (the manager's gateway file format)."""
    parts = [issued.key_pem(), issued.cert_pem()]
    return b"".join(parts if key_first else parts[::-1])


def gateway_pems(directory: Path) -> list[Path]:
    """Files in `directory` holding both a private key and a certificate."""
    if not directory.is_dir():
        return []
    out = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        try:
            data = p.read_bytes()
        except OSError:
            continue
        if b"PRIVATE KEY-----" in data and b"BEGIN CERTIFICATE-----" in data:
            out.append(p)
    return out


def pem_identity(directory: Path) -> tuple[int, int] | None:
    """(inode, certificate serial) of the one gateway PEM in `directory`, None when there
    is not exactly one or it does not parse (yet)."""
    found = gateway_pems(directory)
    if len(found) != 1:
        return None
    try:
        return found[0].stat().st_ino, pem_cert(found[0]).serial_number
    except (OSError, ValueError):
        return None


async def wait_for_new_pem(
    directory: Path, old: tuple[int, int], *, timeout_s: float = FILE_WAIT_S
) -> Path:
    """Poll (real time, up to `timeout_s`) until the gateway PEM in `directory` is no
    longer `old` (its inode or its certificate's serial changed); the file."""
    deadline = time.monotonic() + timeout_s
    while True:
        now = pem_identity(directory)
        if now is not None and now != old:
            return gateway_pems(directory)[0]
        if time.monotonic() > deadline:
            raise AssertionError(
                f"the gateway certificate in {directory} was not replaced within {timeout_s} s"
            )
        await asyncio.sleep(0.05)


async def assert_pem_unchanged(directory: Path, old: tuple[int, int], *, for_s: float) -> None:
    """For `for_s` (real time) the gateway PEM in `directory` stays `old`."""
    deadline = time.monotonic() + for_s
    while time.monotonic() < deadline:
        now = pem_identity(directory)
        assert now == old, f"the gateway certificate changed: {old} -> {now}"
        await asyncio.sleep(0.05)


def pem_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def pem_key(path: Path) -> Any:
    """The private key in a PEM file that may also hold certificates."""
    m = re.search(
        rb"-----BEGIN ([A-Z ]*PRIVATE KEY)-----.*?-----END \1-----", path.read_bytes(), re.S
    )
    assert m is not None, f"{path}: no private key"
    return serialization.load_pem_private_key(m.group(0), password=None)


def check_gateway_cert(path: Path, ca: x509.Certificate) -> x509.Certificate:
    """The manager's gateway file: one PEM with a matching key and certificate that a TLS
    client loads as is (`load_cert_chain(path)`), the certificate a 7-day `clientAuth`
    certificate signed by `ca`."""
    cert = pem_cert(path)
    key = pem_key(path)
    pub = serialization.PublicFormat.SubjectPublicKeyInfo
    der = serialization.Encoding.DER
    assert cert.public_key().public_bytes(der, pub) == key.public_key().public_bytes(der, pub), (
        "the key in the gateway file is not the certificate's"
    )
    ssl.create_default_context().load_cert_chain(str(path))
    cert.verify_directly_issued_by(ca)
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.CLIENT_AUTH in list(eku), list(eku)
    life = (cert.not_valid_after_utc - cert.not_valid_before_utc).total_seconds()
    assert 7 * DAY_S - 600 <= life <= 7 * DAY_S + 3600, f"validity {life / DAY_S:.3f} days"
    return cert


# ------------------------------------------------------------------ metrics


def samples(text: str) -> dict[tuple[str, frozenset[tuple[str, str]]], float]:
    out: dict[tuple[str, frozenset[tuple[str, str]]], float] = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            out[(s.name, frozenset(s.labels.items()))] = float(s.value)
    return out


def sample(text: str, name: str, **labels: str) -> float | None:
    """One sample of the exposition `text` (None when absent)."""
    return samples(text).get((name, frozenset(labels.items())))


# ------------------------------------------------------------------ the `run` driver


Scenario = Callable[[PoolManager, "PoolmgrRun"], Awaitable[None]]


def free_port() -> int:
    """A TCP port on 127.0.0.1 the OS reports free (released again at once)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class PoolmgrRun:
    """`loom-poolmgr run` (and the other commands) in process; see the module docstring.
    Built by the `poolmgr_run` fixture; `live()` before a `run` that must start."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.model = unique_model_id()
        self.admin_queue = f"loom:poolmgr:admin:test:{uuid.uuid4().hex}"
        self.clock: FakeClock | None = None
        self.scenario: Scenario | None = None
        self.constructed: list[dict[str, Any]] = []
        self.manager: PoolManager | None = None
        self.pg_url: str | None = None
        self.redis_url: str | None = None
        self.metrics_port = 0
        self._http_servers: list[Any] = []
        self._redis_clients: list[Any] = []
        for var in list(os.environ):
            if var.startswith("LOOM_POOLMGR_") or var in (
                "LOOM_SERVING_YAML",
                "LOOM_ENV",
                "RUNPOD_API_KEY",
            ):
                monkeypatch.delenv(var, raising=False)
        self.state_dir = tmp_path / "state"
        self.cert_dir = tmp_path / "gateway-cert"
        self.cert_dir.mkdir()
        self.ca = write_ca(tmp_path / "ca")
        self.set_ca(self.ca)
        monkeypatch.setenv("LOOM_POOLMGR_STATE_DIR", str(self.state_dir))
        monkeypatch.setenv("LOOM_POOLMGR_GATEWAY_CERT_DIR", str(self.cert_dir))
        monkeypatch.setenv("LOOM_POOLMGR_METRICS_PORT", "0")
        monkeypatch.setenv("LOOM_POOLMGR_LOCAL_COLD_START_S", "0")
        monkeypatch.setenv("LOOM_POOLMGR_SERVING_YAML", str(serving_file(tmp_path)))
        self.dev()
        # A fresh role gate: nothing a refused or deferred check leaves behind reaches
        # later tests in this process.
        gate = RoleGate()
        monkeypatch.setattr(platform_db, "role_gate", gate)
        monkeypatch.setattr(cli, "role_gate", gate)
        real_start = poolmgr_metrics.start_http_server

        def start_http_server(port: int, *a: Any, **kw: Any) -> Any:
            # Whatever port the settings name, bind one the OS picks and record it.
            served = real_start(0, *a, **kw)
            self._http_servers.append(served)
            server = served[0] if isinstance(served, tuple) else served
            self.metrics_port = int(server.server_port)
            return served

        monkeypatch.setattr(poolmgr_metrics, "start_http_server", start_http_server)
        monkeypatch.setattr(cli, "PoolManager", self._driven_class())

    # ---------------------------------------------------------------- settings

    def dev(self) -> None:
        self.mp.setenv("LOOM_ENV", "dev")
        self.mp.setenv("LOOM_DATABASE_URL", PROD_DATABASE_URL)
        self.mp.setenv("LOOM_REDIS_URL", "redis://127.0.0.1:9/0")

    def prod(self) -> None:
        """`LOOM_ENV=prod` with platform settings that pass the platform's own checks
        (the URLs are placeholders: `Driven` is handed the test servers)."""
        self.mp.setenv("LOOM_ENV", "prod")
        self.mp.setenv("LOOM_DATABASE_URL", PROD_DATABASE_URL)
        self.mp.setenv("LOOM_REDIS_URL", PROD_REDIS_URL)
        self.mp.setattr(platform_settings, "database_role_problems", lambda *a, **k: [])

    def set_ca(self, ca: CaFiles) -> None:
        self.mp.setenv("LOOM_POOLMGR_CA_CERT", str(ca.ca_cert))
        self.mp.setenv("LOOM_POOLMGR_CA_KEY", str(ca.ca_key))
        self.mp.setenv("LOOM_POOLMGR_CLIENT_CERT", str(ca.client_cert))
        self.mp.setenv("LOOM_POOLMGR_CLIENT_KEY", str(ca.client_key))

    def serve_metrics(self) -> None:
        """Turn the metrics endpoint on (0 is off). The setting names a port the OS
        reported free just now, and `scrape()` starts out using it; the wrapped
        `start_http_server` still binds port 0 and `scrape()` then uses the port that
        server got (no other process can take it in between). Should the manager start
        its endpoint some way the wrapper does not see, it binds the setting's port and
        scraping still mostly works."""
        port = free_port()
        self.metrics_port = port
        self.mp.setenv("LOOM_POOLMGR_METRICS_PORT", str(port))

    def live(self, pg_url: str) -> None:
        """Let `run` really start: the test's database (emptied of serving rows, as the
        manager adopts every live row) and Redis."""
        explicit = os.environ.get(REDIS_ENV)
        url = explicit or DEFAULT_REDIS
        client = sync_redis.Redis.from_url(url, socket_timeout=5)
        try:
            client.ping()
        except (sync_redis.ConnectionError, OSError) as e:
            if explicit:
                raise
            pytest.skip(f"Redis not reachable at {url}: {e}")
        finally:
            client.close()
        self.pg_url, self.redis_url = pg_url, url
        asyncio.run(clear_serving_tables(pg_url))

    # ---------------------------------------------------------------- running

    def main(self, *argv: str) -> int | str | None:
        """`loom-poolmgr <argv>`; its exit code (None or 0: success)."""
        try:
            cli.main(list(argv))
        except SystemExit as e:
            return e.code
        return 0

    async def scrape(self) -> str:
        assert self.metrics_port, "no metrics endpoint started (serve_metrics() first)"
        async with httpx.AsyncClient(trust_env=False, timeout=10) as c:
            r = await c.get(f"http://127.0.0.1:{self.metrics_port}/metrics")
        r.raise_for_status()
        return r.text

    async def ticks(self) -> float:
        text = await self.scrape()
        return sample(text, "loom_poolmgr_reconcile_seconds_count") or 0.0

    async def wait_tick(self, after: float, *, timeout_s: float = 15.0) -> None:
        """Wait (real time) until the manager's loop has finished a tick past `after`
        (`loom_poolmgr_reconcile_seconds_count`)."""
        deadline = time.monotonic() + timeout_s
        while await self.ticks() <= after:
            if time.monotonic() > deadline:
                raise AssertionError(f"no reconcile tick within {timeout_s} s")
            await asyncio.sleep(0.05)

    async def advance_to(self, at: float) -> None:
        """Move the fake clock to `at` and wait for the tick that wakes. What the tick
        starts (a write, a spawned task) may still be running: wait for its effect with
        `wait_for_new_pem` / `assert_pem_unchanged`."""
        assert self.clock is not None
        before = await self.ticks()
        self.clock.set(at)
        await self.wait_tick(before)

    def cleanup(self) -> None:
        for served in self._http_servers:
            server = served[0] if isinstance(served, tuple) else served
            with contextlib.suppress(Exception):
                server.shutdown()
                server.server_close()
        if self.pg_url is not None:
            asyncio.run(release_manager_locks(self.pg_url))
            asyncio.run(clear_serving_tables(self.pg_url))
        if self.redis_url is not None:
            client = sync_redis.Redis.from_url(self.redis_url, socket_timeout=5)
            try:
                client.delete(*model_redis_keys(self.model), self.admin_queue)
            finally:
                client.close()

    def _driven_class(self) -> type[PoolManager]:
        h = self

        class Driven(PoolManager):
            def __init__(self, config: Any, **kw: Any) -> None:
                h.constructed.append({"config": config, **kw})
                if h.pg_url is None or h.redis_url is None:
                    raise AssertionError("the manager was built, but the test did not call live()")
                base = next(iter(config.models))
                served = config.models[base]
                placement = config.placement.model_copy(
                    update={
                        "container_disk_gb": {h.model: config.placement.container_disk_gb[base]}
                    }
                )
                config = config.model_copy(
                    update={"models": {h.model: served}, "placement": placement}
                )
                dep = kw["deployments"][base]
                kw["deployments"] = {h.model: dataclasses.replace(dep, model_id=h.model)}
                kw["servable"] = ()
                kw["database_url"] = h.pg_url
                from redis.asyncio import Redis

                redis = Redis.from_url(h.redis_url)
                h._redis_clients.append(redis)
                kw["redis"] = redis
                kw["admin_queue"] = h.admin_queue
                if h.clock is not None:
                    kw["clock"] = h.clock
                    kw["sleep"] = h.clock.sleep
                super().__init__(config, **kw)
                h.manager = self

            async def run(self, stop: asyncio.Event) -> None:
                loop = asyncio.ensure_future(super().run(stop))
                try:
                    if h.scenario is not None:
                        await h.scenario(self, h)
                finally:
                    stop.set()
                    await asyncio.wait_for(loop, 30)

            async def stop(self) -> None:
                try:
                    await super().stop()
                finally:
                    for r in h._redis_clients:
                        with contextlib.suppress(Exception):
                            await r.aclose()
                    h._redis_clients.clear()

        return Driven


@pytest.fixture(name="poolmgr_run")
def poolmgr_run_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[PoolmgrRun]:
    h = PoolmgrRun(monkeypatch, tmp_path)
    try:
        yield h
    finally:
        h.cleanup()


def model_redis_keys(model: str) -> tuple[str, ...]:
    """The Redis keys a manager writes for `model`."""
    return (
        pool_key(model),
        wake_key(model),
        f"{pool_key(model)}:last_request",
        f"loom:poolmgr:reset:{model}",
    )


def utc(epoch: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(epoch, dt.UTC)


@pytest.fixture(name="model_keys")
async def model_keys_fixture(redis_client: Any) -> AsyncIterator[list[str]]:
    """Model ids whose pool hash, wake key and reset time (`loom:poolmgr:reset:<model>`,
    contracts §11) the test leaves in Redis: deleted after."""
    models: list[str] = []
    yield models
    keys = [k for m in models for k in model_redis_keys(m)]
    if keys:
        await redis_client.delete(*keys)
