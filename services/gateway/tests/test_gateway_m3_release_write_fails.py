"""G2 / G2a: the background release of a lost reserve, when the release's own write to
Redis fails. The release is retried with backoff (first retry within 0.2 s, doubling, at
most 2 s apart) until it lands or `LOOM_RESERVATION_GRACE_S` has passed, then given up
(docs/m3-review-fixes-spec.md, G2 and G2a).

Unlike the tests that replace `Limiter.release` with one that raises, every failure here
is a real one seen by redis-py, on the production pool configuration
(`gateway_m3_g2a_kit.limiter_on`). A TCP proxy in front of the real Redis forwards the
reserve script, which takes effect, then drops its reply and resets the connection. From
then on the release's script call either reaches the proxy and is reset unforwarded
("reset": the write never lands in Redis), or finds nothing listening ("refused"), until
the test heals the proxy."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import socket
import struct
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
from gateway_m3_g2a_kit import assert_unavailable, limiter_on, post, skipped, use_limiter
from gateway_m3_kit import done_marker, held, wait_until
from redis.exceptions import RedisError

from loom_platform.limiter import Limiter, script_source

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

SCRIPT_CALLS = (b"EVALSHA", b"EVAL")
RESERVE_SHA = hashlib.sha1(script_source("reserve").encode()).hexdigest().encode()
RELEASE_SHA = hashlib.sha1(script_source("release").encode()).hexdigest().encode()

FAILED_ATTEMPTS = 3  # failed release attempts before the test heals the proxy
LOAD_SLACK_S = 1.0  # timer and scheduling slack under CPU load (flake-check)
FIRST_RETRY_S = 0.2  # spec: first retry within 0.2 s
MAX_SPACING_S = 2.0  # spec: at most 2 s apart
TTL_SLACK_MS = 5_000

GRACE_S = 4  # reservation_grace_s for the give-up test
GIVE_UP_MARGIN_S = 5.0  # max retry spacing (2 s) plus slack for CPU load
QUIET_S = 3.0  # longer than the max retry spacing: a live loop would call again


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


def _encode(parts: list[bytes]) -> bytes:
    out = [b"*%d\r\n" % len(parts)]
    for p in parts:
        out += [b"$%d\r\n" % len(p), p, b"\r\n"]
    return b"".join(out)


def _reset(writer: asyncio.StreamWriter) -> None:
    """Close the connection with a TCP reset."""
    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


def _sha(cmd: list[bytes]) -> bytes:
    if cmd[0].upper() == b"EVALSHA":
        return cmd[1].lower()
    return hashlib.sha1(cmd[1]).hexdigest().encode()


class FaultyRedis:
    """A TCP proxy on 127.0.0.1 in front of the real Redis. It relays everything, except:
    - the first reserve script call is forwarded (it runs in Redis), but its reply is
      dropped and the client's connection reset: the reserve's reply was lost. Its
      request id is recorded in `lost`. (A NOSCRIPT reply is relayed; redis-py then
      loads the script and calls again.)
    - from then until `heal()`, with `then="reset"` every script call is read and the
      connection reset without forwarding it (its time and request id go to `blocked`);
      with `then="refuse"` the proxy stops listening and drops its connections, so
      connecting is refused.
    `landed` holds the request ids of release script calls forwarded to Redis."""

    def __init__(self, upstream: tuple[str, int], then: str) -> None:
        assert then in ("reset", "refuse")
        self.upstream = upstream
        self.then = then
        self.port = 0
        self.lost: list[bytes] = []
        self.blocked: list[tuple[float, bytes]] = []
        self.landed: list[bytes] = []
        self.failing = False
        self._server: asyncio.Server | None = None
        self._holder: socket.socket | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> FaultyRedis:
        # The port stays ours while the proxy lives: this socket is bound to it and never
        # listens. While no listener is open, Linux answers a connect with a reset
        # (ECONNREFUSED, as when Redis is down), and no other process (xdist worker,
        # outgoing connect) can be given the port before `heal()` listens on it again.
        self._holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._holder.bind(("127.0.0.1", 0))
            self.port = self._holder.getsockname()[1]
            await self._listen()
        except BaseException:
            self._holder.close()
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            self._stop_listening()
            for w in list(self._writers):
                w.transport.abort()
            for t in list(self._tasks):
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            if self._holder is not None:
                self._holder.close()

    async def _listen(self) -> None:
        # Both sockets set SO_REUSEADDR, so Linux lets this one bind the port the
        # (non-listening) holder is bound to; asyncio makes it listen.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", self.port))
            self._server = await asyncio.start_server(self._handle, sock=sock)
        except BaseException:
            sock.close()
            raise

    def _stop_listening(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None

    async def heal(self) -> None:
        """Script calls are relayed again (listening again on the same port)."""
        self.failing = False
        if self.then == "refuse":
            await self._listen()

    def _after_loss(self) -> None:
        self.failing = True
        if self.then == "refuse":
            self._stop_listening()
            for w in list(self._writers):
                _reset(w)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._tasks.add(task)
        self._writers.add(writer)
        uw: asyncio.StreamWriter | None = None
        pump: asyncio.Task[None] | None = None
        try:
            if self.failing and self.then == "refuse":
                _reset(writer)  # accepted just before the listener closed
                return
            ur, uw = await asyncio.open_connection(*self.upstream)
            dropping: list[bytes] = []  # the reserve whose reply is to be dropped

            async def relay_replies() -> None:
                while data := await ur.read(65536):
                    if dropping:
                        if data.startswith(b"-NOSCRIPT"):
                            dropping.clear()
                        else:
                            self.lost.append(dropping[0])
                            dropping.clear()
                            self._after_loss()
                            _reset(writer)
                            return
                    writer.write(data)
                    await writer.drain()
                _reset(writer)

            pump = asyncio.create_task(relay_replies())
            while (cmd := await _read_command(reader)) is not None:
                script = len(cmd) >= 5 and cmd[0].upper() in SCRIPT_CALLS
                if script and self.failing:
                    self.blocked.append((time.perf_counter(), cmd[4]))
                    _reset(writer)
                    return
                if script and not self.lost and _sha(cmd) == RESERVE_SHA:
                    dropping[:] = [cmd[4]]
                if script and _sha(cmd) == RELEASE_SHA:
                    self.landed.append(cmd[4])
                uw.write(_encode(cmd))
                await uw.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError, OSError):
            pass
        finally:
            if pump is not None:
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pump
            if uw is not None:
                uw.transport.abort()
            self._writers.discard(writer)
            writer.transport.abort()
            self._tasks.discard(task)


def _proxied_url(redis_url: str, port: int) -> str:
    """`redis_url` (credentials and db kept) pointed at 127.0.0.1:`port`."""
    u = urlsplit(redis_url)
    auth = u.netloc.rpartition("@")[0]
    return urlunsplit(u._replace(netloc=f"{auth + '@' if auth else ''}127.0.0.1:{port}"))


def _upstream(redis_url: str) -> tuple[str, int]:
    u = urlsplit(redis_url)
    return u.hostname or "localhost", u.port or 6379


@dataclass
class Attempt:
    start: float
    end: float
    org_id: Any
    request_id: Any
    error: BaseException | None


def record_release(monkeypatch: pytest.MonkeyPatch, limiter: Limiter) -> list[Attempt]:
    """Wrap the real `limiter.release` (it still runs against Redis) to record each call:
    when it started and ended, its arguments, and the error it raised."""
    real = limiter.release
    attempts: list[Attempt] = []

    async def release(org_id: Any, request_id: Any, *args: Any, **kw: Any) -> Any:
        t0 = time.perf_counter()
        try:
            out = await real(org_id, request_id, *args, **kw)
        except BaseException as e:
            attempts.append(Attempt(t0, time.perf_counter(), org_id, request_id, e))
            raise
        attempts.append(Attempt(t0, time.perf_counter(), org_id, request_id, None))
        return out

    monkeypatch.setattr(limiter, "release", release)
    return attempts


@pytest.fixture
async def limiters() -> AsyncIterator[list[Limiter]]:
    """Limiters made by a test; their Redis clients are closed at teardown."""
    made: list[Limiter] = []
    yield made
    for lim in made:
        await lim.redis.aclose()


async def _prepare(stack: Any) -> Any:
    """A tenant whose balance is already loaded in Redis, so the first reserve script call
    is admitted rather than `noinit`."""
    t = await stack.tenant()
    gw = stack.gateway
    async with gw.pg.connection() as conn:
        await gw.limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    return t


async def _swap_in(
    stack: Any, monkeypatch: pytest.MonkeyPatch, limiters: list[Limiter], proxy: FaultyRedis
) -> list[Attempt]:
    url = _proxied_url(stack.gateway.settings.redis_url, proxy.port)
    lim = limiter_on(stack, url)
    limiters.append(lim)
    use_limiter(monkeypatch, stack, lim)
    return record_release(monkeypatch, lim)


@pytest.mark.parametrize("then", ["reset", "refused"])
async def test_g2_release_whose_write_fails_is_retried_until_it_lands(
    stack, monkeypatch, limiters, then
):
    """G2 / G2a: the reserve took effect but its reply was lost; the release's writes then
    fail for real (connection reset with the command unforwarded, or connection
    refused). The 503 is answered at once; the release is retried with backoff for that
    request id only, each failed attempt a real Redis error; once Redis answers, the
    release lands once: nothing held, the done marker set for at least the reservation
    TTL, no usage event, no skip counted."""
    t = await _prepare(stack)
    unsent0, cap0 = skipped(stack, "unsent"), skipped(stack, "cap")
    async with FaultyRedis(
        _upstream(stack.gateway.settings.redis_url), "reset" if then == "reset" else "refuse"
    ) as proxy:
        attempts = await _swap_in(stack, monkeypatch, limiters, proxy)

        r = await post(stack, t.key)
        assert_unavailable(r)
        rid = uuid.UUID(r.headers["x-loom-request-id"])
        assert proxy.lost == [str(rid).encode()], "the reserve did not run in Redis (setup)"
        assert stack.engine.requests == []
        rsum, n, inflight = await held(stack, t)
        assert rsum > 0 and n == 1 and inflight == 1, "the lost reserve holds nothing (setup)"

        async def failed_enough() -> bool:
            return len(attempts) >= FAILED_ATTEMPTS

        assert await wait_until(failed_enough, 15.0), (
            f"the release was not retried: {len(attempts)} attempts"
        )
        await proxy.heal()

        async def released() -> bool:
            return await held(stack, t) == (0, 0, 0)

        assert await wait_until(released, MAX_SPACING_S + 10.0), (
            f"still held after Redis answered again: {await held(stack, t)}; "
            f"{len(attempts)} attempts"
        )

        async def returned() -> bool:
            return bool(attempts) and attempts[-1].error is None

        assert await wait_until(returned, 5.0)
        await asyncio.sleep(0.5)  # a release that kept going after landing would show up

    assert {(a.org_id, a.request_id) for a in attempts} == {(t.org_id, rid)}
    failed, ok = attempts[:-1], attempts[-1]
    assert ok.error is None
    assert len(failed) >= FAILED_ATTEMPTS, f"only {len(failed)} failed attempts"
    for a in failed:
        assert isinstance(a.error, RedisError | OSError), repr(a.error)
    assert proxy.landed == [str(rid).encode()], (
        f"release script calls that reached Redis: {proxy.landed}"
    )
    if then == "reset":
        # The failed writes reached the wire and were cut before Redis saw them.
        assert [b for _, b in proxy.blocked] == [str(rid).encode()] * len(failed)

    gaps = [b.start - a.end for a, b in itertools.pairwise(attempts)]
    assert gaps[0] <= FIRST_RETRY_S + LOAD_SLACK_S, f"first retry after {gaps[0]:.2f} s"
    assert max(gaps) <= MAX_SPACING_S + LOAD_SLACK_S, f"retries spaced {gaps}"

    pttl = await stack.redis.pttl(done_marker(t.org_id, rid))
    assert pttl != -2, "no :done: marker after the release landed"
    assert pttl != -1, "the :done: marker must expire"
    assert pttl >= stack.gateway.settings.reservation_ttl_ms - TTL_SLACK_MS, pttl
    assert await stack.events() == []
    assert skipped(stack, "unsent") == unsent0
    assert skipped(stack, "cap") == cap0


async def test_g2_release_whose_write_never_lands_is_given_up_after_the_grace(
    make_stack, monkeypatch, limiters
):
    """G2 / G2a: Redis keeps failing the release's writes (each reaches the proxy and is
    reset unforwarded). The release is retried until LOOM_RESERVATION_GRACE_S has passed,
    then given up: no attempt after the grace plus a margin, nothing written to Redis,
    the reservation left to its TTL, no usage event."""
    stack = await make_stack(reservation_grace_s=GRACE_S)
    t = await _prepare(stack)
    async with FaultyRedis(_upstream(stack.gateway.settings.redis_url), "reset") as proxy:
        attempts = await _swap_in(stack, monkeypatch, limiters, proxy)

        t_send = time.perf_counter()
        r = await post(stack, t.key)
        t_resp = time.perf_counter()
        assert_unavailable(r)
        assert t_resp - t_send < GRACE_S - 1, "the 503 waited for the background release"
        rid = uuid.UUID(r.headers["x-loom-request-id"])
        assert proxy.lost == [str(rid).encode()], "the reserve did not run in Redis (setup)"

        await asyncio.sleep(max(0.0, t_resp + GRACE_S + GIVE_UP_MARGIN_S - time.perf_counter()))
        n = len(attempts)
        assert n >= 2, f"the release was not retried ({n} attempts)"
        assert {(a.org_id, a.request_id) for a in attempts} == {(t.org_id, rid)}
        for a in attempts:
            assert isinstance(a.error, RedisError | OSError), repr(a.error)
        assert attempts[-1].start >= t_send + GRACE_S - 2.5, (
            f"gave up {attempts[-1].start - t_send:.2f} s after the request, "
            f"well before the {GRACE_S} s grace"
        )

        await asyncio.sleep(QUIET_S)
        assert len(attempts) == n, f"still retrying after the grace: {len(attempts) - n} more"
        assert len(proxy.blocked) == n
        assert proxy.landed == []

    rsum, held_n, inflight = await held(stack, t)
    assert rsum > 0 and held_n == 1 and inflight == 1, "a release that never landed released"
    assert await stack.redis.exists(done_marker(t.org_id, rid)) == 0
    assert await stack.events() == []
