"""Helpers for the B3 review tests (test_platform_m3_review_*.py): a Postgres TCP proxy
that records every statement a client sends, and a service subprocess whose output is
read as it runs.

`PgProxy` holds a port that refuses connections (bound, not listening: Postgres is
"unreachable") until `open()`, then forwards each connection to the real server. It
answers SSL/GSSAPI encryption requests with "no" itself, so the client's protocol
messages stay readable, and records the SQL of every simple query ('Q') and every
Parse ('P') per connection, with the connection's user, *before* forwarding it. That is
the seam for "no work query ran": everything a service asks Postgres to run passes
through the proxy and is recorded."""

from __future__ import annotations

import contextlib
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

import psycopg

_SSL_REQUEST = 80877103
_GSSENC_REQUEST = 80877104
_CANCEL_REQUEST = 80877102


@dataclass
class Session:
    user: str
    database: str
    statements: list[str] = field(default_factory=list)


def _shutdown(s: socket.socket) -> None:
    with contextlib.suppress(OSError):
        s.shutdown(socket.SHUT_RDWR)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class PgProxy:
    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self.upstream = (upstream_host, upstream_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))  # bound, not listening: connections refused
        self.port: int = self._sock.getsockname()[1]
        self._lock = threading.Lock()
        self._sessions: list[Session] = []
        self._socks: list[socket.socket] = []
        self._closed = False

    # ------------------------------------------------------------------ control

    def open(self) -> None:
        """Postgres becomes reachable through the proxy."""
        self._sock.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            socks = [self._sock, *self._socks]
        for s in socks:
            _shutdown(s)
            s.close()

    def sessions(self) -> list[Session]:
        with self._lock:
            return [Session(s.user, s.database, list(s.statements)) for s in self._sessions]

    def statements(self) -> list[tuple[str, str]]:
        """(user, sql) of every statement sent, all connections."""
        return [(s.user, q) for s in self.sessions() for q in s.statements]

    # ------------------------------------------------------------------ forwarding

    def _track(self, s: socket.socket) -> None:
        with self._lock:
            self._socks.append(s)

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            self._track(client)
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            while True:  # startup phase: untyped messages
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!i", head)
                body = _recv_exact(client, length - 4)
                if body is None:
                    return
                (code,) = struct.unpack("!i", body[:4])
                if code in (_SSL_REQUEST, _GSSENC_REQUEST):
                    client.sendall(b"N")
                    continue
                upstream = socket.create_connection(self.upstream, timeout=10)
                upstream.settimeout(None)
                self._track(upstream)
                upstream.sendall(head + body)
                if code == _CANCEL_REQUEST:
                    return
                parts = body[4:].split(b"\0")
                params = dict(zip(parts[0::2], parts[1::2], strict=False))
                session = Session(
                    params.get(b"user", b"").decode(errors="replace"),
                    params.get(b"database", b"").decode(errors="replace"),
                )
                with self._lock:
                    self._sessions.append(session)
                break
            threading.Thread(target=self._pump, args=(upstream, client), daemon=True).start()
            while True:  # typed messages
                kind = _recv_exact(client, 1)
                if kind is None:
                    return
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!i", head)
                body = _recv_exact(client, length - 4) if length > 4 else b""
                if body is None:
                    return
                sql: bytes | None = None
                if kind == b"Q":
                    sql = body.split(b"\0", 1)[0]
                elif kind == b"P":
                    sql = body.split(b"\0", 2)[1]
                if sql is not None:
                    with self._lock:
                        session.statements.append(sql.decode(errors="replace"))
                upstream.sendall(kind + head + body)
        except OSError:
            return
        finally:
            for s in (client, upstream):
                if s is not None:
                    _shutdown(s)
                    s.close()

    @staticmethod
    def _pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while True:
                data = src.recv(65536)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                _shutdown(s)


def proxied_url(proxy: PgProxy, *, dbname: str, user: str, password: str) -> str:
    q = urllib.parse.quote
    return f"postgresql://{q(user, safe='')}:{q(password, safe='')}@127.0.0.1:{proxy.port}/{dbname}"


_WORK_VERBS = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|COPY|CREATE|ALTER|DROP|GRANT|REVOKE|"
    r"LISTEN|NOTIFY|CALL|DO|LOCK|VACUUM|ANALYZE|REFRESH|CLUSTER|REINDEX|COMMENT)\b",
    re.IGNORECASE,
)


def schema_object_names(conn: psycopg.Connection) -> list[str]:
    """Names of the schema's own tables, views and functions (not an extension's):
    a statement naming one of them is service work, not a catalog check."""
    rows = conn.execute(
        "SELECT c.relname::text FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm') "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_class'::regclass "
        "  AND d.objid = c.oid AND d.deptype = 'e') "
        "UNION SELECT p.proname::text FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid = p.pronamespace WHERE n.nspname = 'public' "
        "AND NOT EXISTS (SELECT 1 FROM pg_depend d WHERE d.classid = 'pg_proc'::regclass "
        "  AND d.objid = p.oid AND d.deptype = 'e')"
    ).fetchall()
    return sorted({r[0] for r in rows})


def work_classifier(names: list[str]) -> Callable[[str], bool]:
    """Whether a statement is service work: it names one of the schema's tables, views
    or functions, or it writes, listens or notifies. Catalog reads (pg_roles, pg_class,
    current_user ...), SET/SHOW and transaction control are not work."""
    named = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)

    def is_work(sql: str) -> bool:
        return bool(_WORK_VERBS.search(sql) or named.search(sql))

    return is_work


class Service:
    """A service subprocess; stdout and stderr are read together, line by line."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        self._lines: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            with self._lock:
                self._lines.append(line)

    @property
    def output(self) -> str:
        with self._lock:
            return "".join(self._lines)

    @property
    def returncode(self) -> int | None:
        return self.proc.poll()

    def wait_for(self, pattern: re.Pattern[str], timeout_s: float) -> bool:
        """Whether a line matching `pattern` appears within `timeout_s` (False early if
        the process exits without one)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if pattern.search(self.output):
                return True
            if self.proc.poll() is not None:
                self._reader.join(5)
                return bool(pattern.search(self.output))
            time.sleep(0.05)
        return False

    def wait_exit(self, timeout_s: float) -> int | None:
        try:
            code = self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            return None
        self._reader.join(5)
        return code

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(10)
        self._reader.join(5)


def service_env(**overrides: str) -> dict[str, str]:
    """The caller's environment without any LOOM_* variable, plus `overrides`."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_")}
    env.update(overrides)
    return env
