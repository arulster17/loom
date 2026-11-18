"""Helpers for the authentication-failure tests (test_platform_m3_auth_*.py; spec
docs/m3-review-fixes-spec.md, "Billing", B3, "Only an outage counts as unreachable",
extended by the user decision of 2026-11-17: a connection error whose message says
`authentication failed` (any method) or `pg_hba.conf rejects` exits 2 too, at start and
in the deferred gate; every other connection error is still retried).

Builds on platform_m3_review4_kit (imported, not changed):

- `AuthPg`: a TCP server that answers a client's startup the way Postgres does when it
  rejects the login: it declines SSL/GSS encryption, reads the startup message,
  optionally asks for a cleartext password first and reads it (so the failure comes
  during the authentication exchange), then sends a `FATAL` ErrorResponse with the given
  SQLSTATE and message, or closes the connection without a word. It can start closed
  (bound, not listening: Postgres unreachable) and `open()` later, for the deferred
  gate. It counts the startups it answered.
- The cases: Postgres's own wording for each authentication method's failure and for
  a `pg_hba.conf` reject, and outages that happen during the authentication exchange.
- `prod_env`: a production-shaped (`LOOM_ENV=prod`) environment per service: the
  gateway and the billing worker with every static production problem cleared (as in
  the review3 tests), the pool manager as far as a test can take it (its RunPod-only
  problems stay, so it always exits 2; what is checked is whether its exit-2 message
  lists the connection error)."""

from __future__ import annotations

import contextlib
import re
import socket
import struct
import sys
import threading
from pathlib import Path

from platform_m3_review3_kit import RedisFront, service_env_for
from platform_m3_review4_kit import error_response
from platform_m3_review_proxy import service_env

POOLMGR = [sys.executable, "-m", "loom_poolmgr.cli", "run"]

_SSL_REQUEST = 80877103
_GSSENC_REQUEST = 80877104
_CANCEL_REQUEST = 80877102
_AUTH_CLEARTEXT = b"R" + struct.pack("!ii", 8, 3)

USER = "loom_t_auth_svc"
DBNAME = "loom_auth_review"


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class AuthPg:
    """Answers every startup with `error` (None: closes without a word), after asking
    for a password first when `ask_password`. Refuses connections until `open()` when
    started with `listening=False`."""

    def __init__(
        self, error: bytes | None, *, ask_password: bool = False, listening: bool = True
    ) -> None:
        self.error = error
        self.ask_password = ask_password
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))  # bound: connections refused until listen()
        self.port: int = self._sock.getsockname()[1]
        self._lock = threading.Lock()
        self._startups = 0
        self._passwords = 0
        if listening:
            self.open()

    @property
    def startups(self) -> int:
        with self._lock:
            return self._startups

    @property
    def passwords(self) -> int:
        """Password messages received (the client went through the auth request)."""
        with self._lock:
            return self._passwords

    def url(self) -> str:
        return f"postgresql://{USER}:auth-review-password@127.0.0.1:{self.port}/{DBNAME}"

    def open(self) -> None:
        self._sock.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(client,), daemon=True).start()

    def _serve(self, client: socket.socket) -> None:
        client.settimeout(10)
        try:
            while True:
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
                if code == _CANCEL_REQUEST:
                    return
                break
            with self._lock:
                self._startups += 1
            if self.ask_password:
                client.sendall(_AUTH_CLEARTEXT)
                kind = _recv_exact(client, 1)
                size = _recv_exact(client, 4)
                if kind != b"p" or size is None:
                    return
                (n,) = struct.unpack("!i", size)
                if _recv_exact(client, n - 4) is None:
                    return
                with self._lock:
                    self._passwords += 1
            if self.error is not None:
                client.sendall(self.error)
        except OSError:
            return
        finally:
            with contextlib.suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()


@contextlib.contextmanager
def auth_servers(cases: dict[str, tuple[bytes | None, bool]], *, listening: bool = True):
    """One `AuthPg` per case (name -> (error, ask_password)), all closed afterwards."""
    servers: dict[str, AuthPg] = {}
    try:
        for name, (err, ask) in cases.items():
            servers[name] = AuthPg(err, ask_password=ask, listening=listening)
        yield servers
    finally:
        for s in servers.values():
            s.close()


# ------------------------------------------------------------------ cases

# Postgres's wording (src/backend/libpq/auth.c) for a failed login per method, and for a
# connection pg_hba.conf rejects. Each is a misconfiguration: retrying cannot fix it. The
# password-like methods ask for a password first, as the server does.
_FOR = f'for user "{USER}"'
MISCONFIG: dict[str, tuple[str, bool, re.Pattern[str]]] = {
    "peer": (
        f"Peer authentication failed {_FOR}",
        False,
        re.compile(r"peer authentication failed", re.IGNORECASE),
    ),
    "ident": (
        f"Ident authentication failed {_FOR}",
        False,
        re.compile(r"ident authentication failed", re.IGNORECASE),
    ),
    "cert": (
        f"certificate authentication failed {_FOR}",
        False,
        re.compile(r"certificate authentication failed", re.IGNORECASE),
    ),
    "ldap": (
        f"LDAP authentication failed {_FOR}",
        True,
        re.compile(r"ldap authentication failed", re.IGNORECASE),
    ),
    "host_rejected": (  # the method-less form, "authentication failed" alone
        f"authentication failed {_FOR}: host rejected",
        False,
        re.compile(r"authentication failed for user \S+: host rejected", re.IGNORECASE),
    ),
    "pg_hba_rejects": (
        f'pg_hba.conf rejects connection for host "127.0.0.1", {_FOR[4:]}, '
        f'database "{DBNAME}", no encryption',
        False,
        re.compile(r"pg_hba\.conf rejects connection", re.IGNORECASE),
    ),
}
# Already on the list before the decision: a guard that the checks below tell exit 2 for
# a misconfiguration from a service that just kept running.
LISTED: dict[str, tuple[str, bool, re.Pattern[str]]] = {
    "password": (
        f"password authentication failed {_FOR}",
        True,
        re.compile(r"password authentication failed", re.IGNORECASE),
    ),
}


def misconfig_error(case: str) -> tuple[bytes, bool, re.Pattern[str]]:
    """(ErrorResponse, ask_password, the pattern naming it) for a MISCONFIG/LISTED case."""
    message, ask, named = {**MISCONFIG, **LISTED}[case]
    sqlstate = "28P01" if case == "password" else "28000"  # as auth.c's auth_failed()
    return error_response(sqlstate, message), ask, named


# Outages during the authentication exchange: the server goes away, is shut down, or
# times the client out while authenticating. None says `authentication failed` or
# `pg_hba.conf rejects` (one says "authentication"): they stay retried.
AUTH_OUTAGES: dict[str, tuple[bytes | None, bool]] = {
    "closed_after_password": (None, True),
    "admin_shutdown_after_password": (
        error_response("57P01", "terminating connection due to administrator command"),
        True,
    ),
    "authentication_timeout": (
        error_response("57014", "canceling authentication due to timeout"),
        False,
    ),
    "shutting_down_after_password": (
        error_response("57P03", "the database system is shutting down"),
        True,
    ),
}


# ------------------------------------------------------------------ environments


def prod_env(kind: str, rf: RedisFront, database_url: str, tmp_path: Path) -> dict[str, str]:
    """`LOOM_ENV=prod` for `kind` (gateway, worker, poolmgr) with this database URL."""
    if kind == "poolmgr":
        return service_env(
            LOOM_ENV="prod",
            LOOM_DATABASE_URL=database_url,
            LOOM_REDIS_URL=rf.url,
            LOOM_LOG_LEVEL="INFO",
        )
    return service_env_for(kind, rf, database_url, tmp_path)
