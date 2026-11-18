"""Helpers for the fourth round of B3 review tests (test_platform_m3_review4_*.py, spec
docs/m3-review-fixes-spec.md, "Billing", B3, "Every other connection error is
retried").

Builds on platform_m3_review_proxy, platform_m3_review2_kit and platform_m3_review3_kit
(imported, not changed):

- `ForgedPg`: a TCP server that speaks just enough of the Postgres protocol to answer a
  client's startup like a real server in a given state: it declines SSL/GSS encryption,
  reads the startup message, then sends a `FATAL` ErrorResponse with the given SQLSTATE
  and message (what Postgres sends while starting up, in recovery, out of slots ...) or
  closes the connection without a word. It counts the startup messages it answered.
- `run_workers`: starts one billing worker per case outside dev (each with its own
  `TlsRedis` front, so Redis is reachable and production-shaped), all at once, and
  reports for each whether it warned about Postgres and was still running after a watch
  window, with its output."""

from __future__ import annotations

import contextlib
import socket
import struct
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from platform_m3_review2_kit import PG_WARNING
from platform_m3_review3_kit import WORKER, redis_front, worker_env
from platform_m3_review_proxy import Service

_SSL_REQUEST = 80877103
_GSSENC_REQUEST = 80877104
_CANCEL_REQUEST = 80877102


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def error_response(sqlstate: str, message: str, severity: str = "FATAL") -> bytes:
    fields = b""
    for code, value in (("S", severity), ("V", severity), ("C", sqlstate), ("M", message)):
        fields += code.encode() + value.encode() + b"\0"
    fields += b"\0"
    return b"E" + struct.pack("!i", 4 + len(fields)) + fields


class ForgedPg:
    """Answers every startup with `error` (an ErrorResponse), or closes the connection
    when `error` is None."""

    def __init__(self, error: bytes | None) -> None:
        self.error = error
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self.port: int = self._sock.getsockname()[1]
        self._lock = threading.Lock()
        self._startups = 0
        self._closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def startups(self) -> int:
        with self._lock:
            return self._startups

    def url(self, *, dbname: str = "loom_review4", user: str = "loom_t_r4_worker") -> str:
        return f"postgresql://{user}:r4-password@127.0.0.1:{self.port}/{dbname}"

    def close(self) -> None:
        with self._lock:
            self._closed = True
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
                with self._lock:
                    self._startups += 1
                if self.error is not None:
                    client.sendall(self.error)
                return
        except OSError:
            return
        finally:
            with contextlib.suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()


@dataclass
class Outcome:
    warned: bool
    returncode: int | None
    output: str

    @property
    def kept_running(self) -> bool:
        return self.warned and self.returncode is None


def run_workers(
    urls: dict[str, str], directory: Path, *, warn_s: float, watch_s: float
) -> dict[str, Outcome]:
    """Start a billing worker outside dev for each case's LOOM_DATABASE_URL, all at once;
    wait (up to `warn_s`) for each to warn about Postgres or exit, then `watch_s` more,
    and report each one. Every worker is stopped before returning."""
    with contextlib.ExitStack() as stack:
        services: dict[str, Service] = {}
        for name, url in urls.items():
            rf = stack.enter_context(redis_front(directory / name))
            svc = Service(WORKER, worker_env(rf, url))
            stack.callback(svc.stop)
            services[name] = svc
        deadline = time.monotonic() + warn_s
        warned: dict[str, bool] = {}
        for name, svc in services.items():
            warned[name] = svc.wait_for(PG_WARNING, max(0.0, deadline - time.monotonic()))
        time.sleep(watch_s)
        return {
            name: Outcome(warned[name], svc.returncode, svc.output)
            for name, svc in services.items()
        }


@contextlib.contextmanager
def forged(errors: dict[str, bytes | None]) -> Iterator[dict[str, ForgedPg]]:
    servers: dict[str, ForgedPg] = {}
    try:
        for name, err in errors.items():
            servers[name] = ForgedPg(err)
        yield servers
    finally:
        for s in servers.values():
            s.close()


def report(outcomes: dict[str, Outcome], names: list[str]) -> str:
    parts = []
    for name in names:
        o = outcomes[name]
        parts.append(f"--- {name}: warned={o.warned} returncode={o.returncode}\n{o.output.strip()}")
    return "\n".join(parts)
