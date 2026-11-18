"""Helpers for the G2a tests (`test_gateway_m3_g2a_*.py`, docs/m3-review-fixes-spec.md):
Redis endpoints that fail a reserve before or after its command is written, a limiter on
such an endpoint swapped into the gateway, a `Limiter.release` spy, and the skip counter.

The failures are real network failures seen by redis-py, not hand-raised exceptions:
- unsent: a pool whose only connection is held (`No connection available.`), a port
  nobody listens on (connection refused), a listener whose accept queue is full (the
  connect times out);
- possibly sent: a fake Redis server that answers the connection handshake, reads the
  script call, then never replies (socket timeout) or resets the connection."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import socket
import struct
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from gateway_m3_kit import MESSAGES, sample

from loom_platform.limiter import Limiter

SKIPPED = "loom_gateway_reserve_release_skipped_total"


def skipped(stack: Any, reason: str) -> float:
    """`loom_gateway_reserve_release_skipped_total{reason}` in the gateway's registry."""
    return sample(stack, SKIPPED, {"reason": reason})


async def post(stack: Any, key: str) -> httpx.Response:
    async with httpx.AsyncClient(timeout=60) as c:
        return await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": stack.small, "messages": MESSAGES, "max_tokens": 5},
        )


def assert_unavailable(r: httpx.Response) -> None:
    """The 503 a reserve that failed on Redis answers, in every case (G2, G2a)."""
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "service_unavailable"
    assert r.headers["retry-after"] == "1"


# Connection kwargs of a redis-py pool that are not the gateway's configuration: where it
# connects (taken from the new URL instead), and values redis-py derives per pool (each
# new pool makes its own).
LOCATION_KWARGS = frozenset({"host", "port", "db", "username", "password", "path"})
PER_POOL_KWARGS = frozenset(
    {
        "orig_host_address",
        "orig_socket_timeout",
        "orig_socket_connect_timeout",
        "maint_notifications_pool_handler",
        "maint_notifications_config",
        "himport_registry",
    }
)


def configured_kwargs(pool: Any) -> dict[str, Any]:
    """The connection kwargs `pool` was configured with (retry, timeouts, ...), without
    its location and the values redis-py derives per pool."""
    return {
        k: v
        for k, v in pool.connection_kwargs.items()
        if k not in LOCATION_KWARGS | PER_POOL_KWARGS
    }


def limiter_on(
    stack: Any,
    url: str,
    *,
    max_connections: int | None = None,
    pool_timeout: float | None = None,
    socket_timeout: float | None = None,
    socket_connect_timeout: float | None = None,
) -> Limiter:
    """A Limiter like the gateway's on the Redis at `url`: the same client, pool and
    connection classes, and every connection kwarg the gateway's pool was built with
    (its pinned `retry` included), except the location (from `url`) and the arguments
    given here. Close its client at teardown (`await lim.redis.aclose()`)."""
    gw = stack.gateway
    client = gw.redis
    pool = client.connection_pool
    kwargs = copy.deepcopy(configured_kwargs(pool))
    if socket_timeout is not None:
        kwargs["socket_timeout"] = socket_timeout
    if socket_connect_timeout is not None:
        kwargs["socket_connect_timeout"] = socket_connect_timeout
    new_pool = type(pool).from_url(
        url,
        connection_class=pool.connection_class,
        max_connections=max_connections or pool.max_connections,
        timeout=pool.timeout if pool_timeout is None else pool_timeout,
        **kwargs,
    )
    return Limiter(
        type(client)(
            connection_pool=new_pool, single_connection_client=client.single_connection_client
        ),
        usage_stream=gw.limiter.usage_stream,
        settled_ttl_s=gw.limiter.settled_ttl_s,
    )


def use_limiter(monkeypatch: pytest.MonkeyPatch, stack: Any, limiter: Limiter) -> None:
    """Make `limiter` the gateway's limiter for this test."""
    monkeypatch.setattr(stack.gateway, "limiter", limiter)


def spy_release(
    monkeypatch: pytest.MonkeyPatch, limiter: Limiter, gate: asyncio.Event | None = None
) -> dict[str, list[Any]]:
    """Replace `limiter.release` with a spy that records `(org_id, request_id)` in
    `calls`, waits for `gate` when given, records the request id in `returned`, and
    succeeds without touching Redis. `running` holds the request ids of the calls under
    way, and `peak[0]` the most that were under way at once."""
    seen: dict[str, list[Any]] = {"calls": [], "returned": [], "running": [], "peak": [0]}

    async def release(org_id: Any, request_id: Any, *args: Any, **kw: Any) -> bool:
        seen["calls"].append((org_id, request_id))
        seen["running"].append(request_id)
        seen["peak"][0] = max(seen["peak"][0], len(seen["running"]))
        try:
            if gate is not None:
                await gate.wait()
        finally:
            seen["running"].remove(request_id)
        seen["returned"].append(request_id)
        return True

    monkeypatch.setattr(limiter, "release", release)
    return seen


def free_port() -> int:
    """A local port nobody listens on (connecting to it is refused)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


@contextlib.contextmanager
def unanswered_listener() -> Iterator[int]:
    """A local port whose accept queue is full: the kernel drops further SYNs, so a
    connect to it times out. Yields the port."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(0)
    port: int = srv.getsockname()[1]
    fillers: list[socket.socket] = []
    try:
        for _ in range(8):
            f = socket.socket()
            f.setblocking(False)
            with contextlib.suppress(BlockingIOError):
                f.connect(("127.0.0.1", port))
            fillers.append(f)
        yield port
    finally:
        for f in fillers:
            f.close()
        srv.close()


async def _read_command(reader: asyncio.StreamReader) -> list[bytes] | None:
    line = await reader.readline()
    if not line:
        return None
    if not line.startswith(b"*"):
        return line.split()
    parts: list[bytes] = []
    for _ in range(int(line[1:])):
        header = await reader.readline()
        data = await reader.readexactly(int(header[1:]) + 2)
        parts.append(data[:-2])
    return parts


class ScriptBlackhole:
    """A fake Redis server on 127.0.0.1. It answers `+OK` to every command of the
    connection handshake (CLIENT SETINFO, SELECT, ...), and on a script call (EVALSHA,
    EVAL), once the command has been read, either never replies (`mode="stall"`: the
    client's socket timeout fires while reading the reply) or resets the connection
    (`mode="reset"`). `scripts` counts the script calls read."""

    def __init__(self, mode: str) -> None:
        assert mode in ("stall", "reset")
        self.mode = mode
        self.scripts = 0
        self.port = 0
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def __aenter__(self) -> ScriptBlackhole:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.close()
        for w in list(self._writers):
            w.transport.abort()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"redis://127.0.0.1:{self.port}/0"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.add(writer)
        try:
            while (cmd := await _read_command(reader)) is not None:
                if cmd and cmd[0].upper() in (b"EVALSHA", b"EVAL"):
                    self.scripts += 1
                    if self.mode == "reset":
                        sock = writer.get_extra_info("socket")
                        sock.setsockopt(
                            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                        )
                        writer.transport.abort()
                        return
                    continue  # stall: never answer
                writer.write(b"+OK\r\n")
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            self._writers.discard(writer)
            writer.transport.abort()
