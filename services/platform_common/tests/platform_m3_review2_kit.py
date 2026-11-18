"""Helpers for the second round of B3 review tests (test_platform_m3_review2_*.py, spec
docs/m3-review-fixes-spec.md, "Billing", B3, "Postgres unreachable at start").

Builds on platform_m3_review_proxy (imported, not changed):

- `QuietPgProxy`: a `PgProxy` that also knows when every proxied session has ended, so
  the statements are read only after each session has seen the client's EOF (its last
  bytes recorded).
- `TlsRedis`: a TLS front for the test Redis that accepts any `AUTH` password, so a
  service can be given a production-shaped `rediss://:<password>@` URL (no static
  production problem) and still reach a real Redis.
- `make_certs`: a throwaway CA, a server certificate for 127.0.0.1 and a client
  certificate.
- `drop_all`: drops a database, then each role, each drop in its own try/finally.
- Log-line checks for the unreachable-at-start warning and the role problems."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import ipaddress
import os
import re
import socket
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
import redis
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from platform_m3_review_proxy import PgProxy, Session
from platform_m3_roles import admin_url, drop_database
from psycopg import sql

from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV

REPO_ROOT = Path(__file__).resolve().parents[3]

# ------------------------------------------------------------------ log lines

_WARN = r"\bwarn(?:ing)?\b"
_PG = r"(?:postgres|database)"
PG_WARNING = re.compile(rf"{_WARN}[^\n]*{_PG}|{_PG}[^\n]*{_WARN}", re.IGNORECASE)
_OWN = r"\bown(?:s|er|ed|ership)\b"
_TABLE = r"\btables?\b"
OWNER = re.compile(rf"{_OWN}[^\n]{{0,80}}{_TABLE}|{_TABLE}[^\n]{{0,80}}{_OWN}", re.IGNORECASE)
_ROLE_WORDING = re.compile(r"\bsuperuser\b|\bowns\b", re.IGNORECASE)


def superuser_problem(output: str, role: str) -> bool:
    who = rf"(?:\b{re.escape(role)}\b|\brole\b)"
    pattern = rf"{who}[^\n]{{0,80}}\bsuperuser\b|\bsuperuser\b[^\n]{{0,80}}{who}"
    return re.search(pattern, output, re.IGNORECASE) is not None


def owner_problem(output: str) -> bool:
    return OWNER.search(output) is not None


def role_problem(output: str, role: str) -> bool:
    return superuser_problem(output, role) or owner_problem(output)


def warning_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if PG_WARNING.search(line)]


def assert_start_warning_wording(line: str) -> None:
    """Spec: the unreachable-at-start warning is one log line that names Postgres (or
    "database") and does not repeat the role problems' wording ("superuser", "owns").
    `line` is that warning: the first line naming Postgres the service wrote (a later
    line, such as a gate warning, cannot stand in for it)."""
    assert PG_WARNING.search(line), f"not a warning line naming Postgres / the database: {line}"
    assert not _ROLE_WORDING.search(line), (
        "the unreachable-at-start warning repeats the role problems' wording (superuser / "
        f"owns), so it reads like a refusal:\n{line}"
    )


def assert_warning_wording(output: str) -> None:
    """`assert_start_warning_wording` on the first warning line naming Postgres in
    `output` (the unreachable-at-start warning)."""
    lines = warning_lines(output)
    assert lines, f"no warning line naming Postgres / the database:\n{output}"
    assert_start_warning_wording(lines[0])


# ------------------------------------------------------------------ Postgres proxy


class QuietPgProxy(PgProxy):
    """A PgProxy that counts its sessions in progress. `wait_quiet` returns once every
    accepted connection has ended (each session saw the client's EOF, so its last
    statement is recorded) and none has started for a while (a connection still in the
    listen backlog when the client died is accepted and drained too)."""

    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        super().__init__(upstream_host, upstream_port)
        self._cv = threading.Condition()
        self._active = 0
        self._started = 0
        self.opened_at_session: int | None = None

    def open(self) -> None:
        self.opened_at_session = len(self.sessions())
        super().open()

    def _serve(self, client: socket.socket) -> None:
        with self._cv:
            self._active += 1
            self._started += 1
        try:
            super()._serve(client)
        finally:
            with self._cv:
                self._active -= 1
                self._cv.notify_all()

    def wait_quiet(self, timeout_s: float = 15, settle_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while True:
                while self._active:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        return False
                    self._cv.wait(left)
                seen = self._started
                self._cv.wait(settle_s)
                if not self._active and self._started == seen:
                    return True
                if time.monotonic() >= deadline:
                    return False

    def sessions_since_open(self) -> list[Session]:
        """Sessions accepted after `open()` (all of them: nothing connects before)."""
        start = self.opened_at_session or 0
        return self.sessions()[start:]


# ------------------------------------------------------------------ certificates


@dataclass(frozen=True)
class Certs:
    ca: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path


def _key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _write_key(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _cert(
    subject: str,
    key: ec.EllipticCurvePrivateKey,
    issuer: x509.Name | None,
    issuer_key: ec.EllipticCurvePrivateKey,
    *,
    ca: bool = False,
    server: bool = False,
) -> x509.Certificate:
    now = dt.datetime.now(dt.UTC)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
    b = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer or name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if ca:
        b = b.add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    elif server:
        b = b.add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        ).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    else:
        b = b.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
    return b.sign(issuer_key, hashes.SHA256())


def make_certs(directory: Path) -> Certs:
    pem = serialization.Encoding.PEM
    ca_key, srv_key, cli_key = _key(), _key(), _key()
    ca = _cert("loom-review2-test-ca", ca_key, None, ca_key, ca=True)
    srv = _cert("localhost", srv_key, ca.subject, ca_key, server=True)
    cli = _cert("loom-gateway", cli_key, ca.subject, ca_key)
    out = Certs(
        ca=directory / "ca.pem",
        server_cert=directory / "server.pem",
        server_key=directory / "server.key",
        client_cert=directory / "client.pem",
        client_key=directory / "client.key",
    )
    out.ca.write_bytes(ca.public_bytes(pem))
    out.server_cert.write_bytes(srv.public_bytes(pem))
    _write_key(out.server_key, srv_key)
    out.client_cert.write_bytes(cli.public_bytes(pem))
    _write_key(out.client_key, cli_key)
    return out


# ------------------------------------------------------------------ Redis over TLS


def redis_upstream() -> tuple[str, int, int]:
    """(host, port, db) of the test Redis (LOOM_TEST_REDIS_URL, else the default);
    skips when the default is unreachable and the variable is unset."""
    explicit = os.environ.get(REDIS_ENV)
    url = explicit or DEFAULT_REDIS
    client = redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)
    try:
        client.ping()
    except (redis.exceptions.ConnectionError, OSError) as e:
        if explicit:
            raise
        pytest.skip(f"Redis not reachable at {DEFAULT_REDIS} and {REDIS_ENV} unset: {e}")
    finally:
        client.close()
    kw = client.connection_pool.connection_kwargs
    return str(kw.get("host", "localhost")), int(kw.get("port", 6379)), int(kw.get("db", 0))


async def _read_command(reader: asyncio.StreamReader) -> tuple[bytes, list[bytes]] | None:
    """One RESP array command: its raw bytes and its arguments (None at EOF)."""
    head = await reader.readline()
    if not head:
        return None
    raw = head
    if not head.startswith(b"*"):  # inline command
        return raw, head.split()
    args = []
    for _ in range(int(head[1:].strip())):
        size_line = await reader.readline()
        raw += size_line
        data = await reader.readexactly(int(size_line[1:].strip()) + 2)
        raw += data
        args.append(data[:-2])
    return raw, args


def _encode(args: list[bytes]) -> bytes:
    out = b"*%d\r\n" % len(args)
    for a in args:
        out += b"$%d\r\n%s\r\n" % (len(a), a)
    return out


async def _pump(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
    try:
        while data := await src.read(65536):
            dst.write(data)
            await dst.drain()
    except (OSError, ssl.SSLError):
        pass
    finally:
        dst.close()


class TlsRedis:
    """TLS on 127.0.0.1 in front of the test Redis. The client's first command, when it
    is `AUTH` (or `HELLO ... AUTH user pass`), is answered here (any password is
    accepted); everything else is forwarded unchanged."""

    def __init__(self, upstream: tuple[str, int], certs: Certs) -> None:
        self.upstream = upstream
        self.connections = 0
        ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ctx.load_cert_chain(certs.server_cert, certs.server_key)
        self._tasks: set[asyncio.Task[None]] = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._server = asyncio.run_coroutine_threadsafe(
            asyncio.start_server(self._handle, "127.0.0.1", 0, ssl=ctx), self._loop
        ).result(10)
        self.port: int = self._server.sockets[0].getsockname()[1]

    def url(self, db: int, password: str, ca: Path) -> str:
        q = urllib.parse.quote
        auth = q(password, safe="")
        return f"rediss://:{auth}@127.0.0.1:{self.port}/{db}?ssl_ca_certs={q(str(ca))}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)  # type: ignore[arg-type]
        self.connections += 1
        up_w: asyncio.StreamWriter | None = None
        try:
            up_r, up_w = await asyncio.open_connection(*self.upstream)
            first = await _read_command(reader)
            if first is None:
                return
            raw, args = first
            upper = [a.upper() for a in args]
            if upper and upper[0] == b"AUTH":
                writer.write(b"+OK\r\n")
                await writer.drain()
            elif upper and upper[0] == b"HELLO" and b"AUTH" in upper:
                i = upper.index(b"AUTH")
                up_w.write(_encode(args[:i] + args[i + 3 :]))
            else:
                up_w.write(raw)
            await asyncio.gather(_pump(reader, up_w), _pump(up_r, writer))
        except (OSError, ssl.SSLError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            for w in (writer, up_w):
                if w is not None:
                    w.close()
            self._tasks.discard(task)  # type: ignore[arg-type]

    def close(self) -> None:
        async def shut() -> None:
            self._server.close()
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(shut(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)


# ------------------------------------------------------------------ cleanup


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def drop_role(info: dict[str, str], name: str) -> None:
    with psycopg.connect(admin_url(info, info.get("dbname", "postgres")), autocommit=True) as c:
        c.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(name)))


def drop_roles(info: dict[str, str], names: list[str]) -> None:
    """Each role dropped in its own try/finally: one failing drop never skips the rest."""
    if not names:
        return
    try:
        drop_role(info, names[0])
    finally:
        drop_roles(info, names[1:])


def drop_all(info: dict[str, str], db: str | None, roles: list[str]) -> None:
    """The database first (objects in it may belong to the roles), then every role,
    whether or not the database drop succeeded."""
    try:
        if db is not None:
            drop_database(info, db)
    finally:
        drop_roles(info, roles)
