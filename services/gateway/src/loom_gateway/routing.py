"""Model pools and replica choice (docs/phase1-contracts.md §9-10,
docs/phase1-runpod-plan.md §7).

Only `ready` replicas are candidates. Among them the gateway picks the one with the
fewest of its own open requests (exact with one gateway), trying a replica that failed
in the last `replica_penalty_s` only when nothing else is left, and ties going to the
first listed. No replica: 503 `model_unavailable` at once; the gateway never queues.
"""

from __future__ import annotations

import asyncio
import base64
import calendar
import logging
import os
import re
import ssl
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import httpx
from redis.asyncio import Redis

from loom_gateway.metrics import GatewayMetrics
from loom_gateway.settings import DEFAULT_COLD_START_S, GatewaySettings
from loom_platform import errors
from loom_platform.pool import PoolState, Replica, get_pool, request_wake
from loom_platform.serving import load_serving, serving_path

log = logging.getLogger("loom_gateway.routing")

_PEM_CERT = re.compile(rb"-----BEGIN CERTIFICATE-----(.+?)-----END CERTIFICATE-----", re.S)


def _der_item(buf: bytes, pos: int) -> tuple[int, int, int]:
    """(tag, start of content, end of content) of the DER element at `pos`."""
    tag, first = buf[pos], buf[pos + 1]
    if first & 0x80:
        n = first & 0x7F
        length = int.from_bytes(buf[pos + 2 : pos + 2 + n], "big")
        start = pos + 2 + n
    else:
        length, start = first, pos + 2
    if start + length > len(buf):
        raise ValueError("truncated DER")
    return tag, start, start + length


def certificate_not_after(pem: bytes) -> float:
    """`notAfter` (epoch seconds) of the first certificate in `pem`. A minimal DER walk,
    so the gateway needs no X.509 library for its expiry gauge."""
    m = _PEM_CERT.search(pem)
    if m is None:
        raise ValueError("no certificate in the file")
    der = base64.b64decode(b"".join(m.group(1).split()))
    _, start, _ = _der_item(der, 0)  # Certificate
    _, pos, end = _der_item(der, start)  # tbsCertificate
    fields: list[tuple[int, int, int]] = []
    while pos < end:
        tag, s, e = _der_item(der, pos)
        fields.append((tag, s, e))
        pos = e
    if fields and fields[0][0] == 0xA0:  # the optional [0] version
        fields = fields[1:]
    tag, s, e = fields[3]  # serial, signature, issuer, validity
    if tag != 0x30:
        raise ValueError("no validity in the certificate")
    nb_end = _der_item(der, s)[2]  # after notBefore
    tag, s2, e2 = _der_item(der, nb_end)  # notAfter
    text = der[s2:e2].decode("ascii")
    if tag == 0x17:  # UTCTime YYMMDDHHMMSSZ
        year = int(text[:2])
        text = f"{1900 + year if year >= 50 else 2000 + year}{text[2:]}"
    elif tag != 0x18:  # GeneralizedTime YYYYMMDDHHMMSSZ
        raise ValueError("bad notAfter")
    return float(
        calendar.timegm(
            (int(text[:4]), int(text[4:6]), int(text[6:8]), int(text[8:10]), int(text[10:12]), 0)
        )
        + int(text[12:14])
    )


class TlsSource:
    """The gateway's upstream TLS context, rebuilt when the client certificate, its key
    or `LOOM_POD_CA_CERT` change on disk (spec §7). `refresh()` looks at the files at
    most every `LOOM_GATEWAY_CERT_CHECK_S`; it is called before each upstream request
    and at each metrics scrape, so a changed file is in use for the next new connection
    and the gauge stays current without traffic. A file that fails to load keeps the
    old context and logs an error (once per change of the file)."""

    def __init__(self, s: GatewaySettings) -> None:
        self.cert = s.gateway_client_cert
        self.key = s.gateway_client_key
        self.ca = s.pod_ca_cert
        self.check_s = s.gateway_cert_check_s
        self._lock = threading.Lock()
        self._sig = self._signature()
        # (version, context, client certificate's notAfter): replaced as a whole.
        self.state: tuple[int, ssl.SSLContext, float | None] = (0, *self._build())
        self._next_check = time.monotonic() + self.check_s

    def _paths(self) -> tuple[str | None, ...]:
        return (self.cert, self.key, self.ca)

    def _signature(self) -> tuple[tuple[int, int, int] | None, ...]:
        out: list[tuple[int, int, int] | None] = []
        for path in self._paths():
            try:
                st = os.stat(path) if path else None
            except OSError:
                st = None
            out.append(None if st is None else (st.st_ino, st.st_mtime_ns, st.st_size))
        return tuple(out)

    def _build(self) -> tuple[ssl.SSLContext, float | None]:
        ctx = (
            ssl.create_default_context(cafile=self.ca) if self.ca else ssl.create_default_context()
        )
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        not_after = None
        if self.cert:
            ctx.load_cert_chain(self.cert, self.key)
            with open(self.cert, "rb") as f:
                not_after = certificate_not_after(f.read())
        return ctx, not_after

    def refresh(self) -> None:
        now = time.monotonic()
        if now < self._next_check or not self._lock.acquire(blocking=False):
            return
        try:
            if now < self._next_check:
                return
            self._next_check = now + self.check_s
            sig = self._signature()
            if sig == self._sig:
                return
            self._sig = sig
            try:
                ctx, not_after = self._build()
            except Exception as e:
                log.error(
                    "client certificate or pod CA bundle failed to load; keeping the old TLS "
                    "context",
                    extra={"fields": {"error": f"{type(e).__name__}: {e}"}},
                )
                return
            self.state = (self.state[0] + 1, ctx, not_after)
            log.info("reloaded the client certificate and pod CA bundle")
        finally:
            self._lock.release()

    def expiry_seconds(self) -> float | None:
        """Seconds until the client certificate in use expires (None: none configured)."""
        self.refresh()
        not_after = self.state[2]
        return None if not_after is None else not_after - time.time()


_sources: dict[tuple[object, ...], TlsSource] = {}
_sources_lock = threading.Lock()


def tls_source(s: GatewaySettings) -> TlsSource:
    """The `TlsSource` of these settings' files, shared by the upstream client and the
    expiry gauge (one load, one set of reload errors). Its contexts hold no event loop."""
    key = (s.gateway_client_cert, s.gateway_client_key, s.pod_ca_cert, s.gateway_cert_check_s)
    with _sources_lock:
        source = _sources.get(key)
        if source is None:
            source = _sources[key] = TlsSource(s)
        return source


class _CountedStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, done: Callable[[], None]) -> None:
        self._inner = inner
        self._done: Callable[[], None] | None = done

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            yield chunk

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            done, self._done = self._done, None
            if done is not None:
                done()


class _Generation:
    def __init__(self, version: int, transport: httpx.AsyncHTTPTransport) -> None:
        self.version = version
        self.transport = transport
        self.active = 0
        self.retired = False


class ReloadingTransport(httpx.AsyncBaseTransport):
    """An `AsyncHTTPTransport` per TLS context. When the source has a new context, new
    requests use a new transport (so a new connection pool and new handshakes); the old
    one lives until its open responses have finished and is then closed."""

    def __init__(self, source: TlsSource, limits: httpx.Limits) -> None:
        self._source = source
        self._limits = limits
        self._gen = self._generation()
        self._retired: list[_Generation] = []
        self._tasks: set[asyncio.Task[None]] = set()

    def _generation(self) -> _Generation:
        version, ctx, _ = self._source.state
        return _Generation(
            version,
            httpx.AsyncHTTPTransport(verify=ctx, http2=True, limits=self._limits, trust_env=False),
        )

    def _close_later(self, gen: _Generation) -> None:
        if gen in self._retired:
            self._retired.remove(gen)
        task = asyncio.ensure_future(gen.transport.aclose())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _release(self, gen: _Generation) -> None:
        gen.active -= 1
        if gen.retired and gen.active <= 0:
            self._close_later(gen)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._source.refresh()
        if self._source.state[0] != self._gen.version:
            old, self._gen = self._gen, self._generation()
            old.retired = True
            if old.active <= 0:
                self._close_later(old)
            else:
                self._retired.append(old)
        gen = self._gen
        gen.active += 1
        try:
            response = await gen.transport.handle_async_request(request)
        except BaseException:
            self._release(gen)
            raise
        assert isinstance(response.stream, httpx.AsyncByteStream)
        response.stream = _CountedStream(response.stream, lambda: self._release(gen))
        return response

    async def aclose(self) -> None:
        for gen in [self._gen, *self._retired]:
            await gen.transport.aclose()
        self._retired.clear()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)


def upstream_client(s: GatewaySettings) -> httpx.AsyncClient:
    """One pooled client for every pod: TLS 1.3, client certificate, pod CA, HTTP/2 where
    the proxy offers it. The server name is checked per request (`sni_hostname`). The
    TLS context follows the files on disk (`TlsSource`): new connections use the newest,
    open streams keep theirs."""
    return httpx.AsyncClient(
        transport=ReloadingTransport(
            tls_source(s), httpx.Limits(max_connections=2048, max_keepalive_connections=512)
        ),
        timeout=httpx.Timeout(
            connect=s.upstream_connect_timeout_s,
            read=float(s.request_timeout_s),
            write=30.0,
            pool=s.upstream_connect_timeout_s,
        ),
        trust_env=False,
        follow_redirects=False,
    )


def serving_cold_starts() -> dict[str, int]:
    """`cold_start_estimate_s` per model from config/serving.yaml (`LOOM_SERVING_YAML`
    honoured). No file there: nothing, so every model gets DEFAULT_COLD_START_S. A file
    that exists but does not validate raises (the gateway fails fast at start; spec §11)."""
    path = serving_path()
    try:
        cfg = load_serving(path, check_registry=False)
    except FileNotFoundError:
        log.warning(
            "no serving config; cold-start estimates default",
            extra={"fields": {"path": str(path), "cold_start_s": DEFAULT_COLD_START_S}},
        )
        return {}
    return {model_id: m.cold_start_estimate_s for model_id, m in cfg.models.items()}


@dataclass
class _ReplicaStats:
    inflight: int = 0
    penalty_until: float = 0.0


class Router:
    def __init__(
        self,
        redis: Redis,
        settings: GatewaySettings,
        metrics: GatewayMetrics,
        cold_start_s: dict[str, int] | None = None,
    ) -> None:
        self.redis = redis
        self.settings = settings
        self.metrics = metrics
        self.cold_start_s = {**serving_cold_starts(), **(cold_start_s or {})}
        if settings.gateway_client_cert:
            # Read at each scrape (loaded lazily: the files need not exist here).
            metrics.set_client_cert_expiry(lambda: tls_source(settings).expiry_seconds())
        self._pools: dict[str, tuple[float, PoolState]] = {}
        self._stats: dict[str, _ReplicaStats] = {}

    def cold_start_estimate(self, model_id: str) -> int:
        return self.cold_start_s.get(model_id, DEFAULT_COLD_START_S)

    async def pool(self, model_id: str) -> PoolState:
        now = time.monotonic()
        hit = self._pools.get(model_id)
        if hit is not None and hit[0] > now:
            return hit[1]
        pool = await get_pool(
            self.redis, model_id, cold_retry_after_s=self.cold_start_estimate(model_id)
        )
        if self.settings.pool_cache_ttl_s > 0:
            self._pools[model_id] = (now + self.settings.pool_cache_ttl_s, pool)
        return pool

    async def check_routable(self, pool: PoolState) -> None:
        """Raise 503 model_unavailable (with Retry-After) unless the pool can serve.
        Every 503 is counted by pool state and reason."""
        if pool.routable:
            return
        state = "cold" if pool.state == "ready" else pool.state
        self.metrics.model_unavailable.labels(pool.model_id, state, pool.reason or "none").inc()
        if pool.state == "warming":
            raise errors.model_unavailable(
                pool.model_id,
                pool.retry_after_s or self.cold_start_estimate(pool.model_id),
                True,
            )
        # cold, missing, 'ready' with no replica, or unavailable: ask for a start. For
        # `unavailable` the manager decides whether a wake can clear the reason (it retries
        # no_capacity while the wake lives; the other reasons need an admin).
        if await request_wake(self.redis, pool.model_id):
            self.metrics.wakes.labels(pool.model_id).inc()
        if pool.state == "unavailable":
            raise errors.model_unavailable(pool.model_id, pool.retry_after_s or 300, False)
        raise errors.model_unavailable(
            pool.model_id, pool.retry_after_s or self.cold_start_estimate(pool.model_id), True
        )

    # ------------------------------------------------------------ replica choice

    def _st(self, replica: Replica) -> _ReplicaStats:
        st = self._stats.get(replica.id)
        if st is None:
            st = self._stats[replica.id] = _ReplicaStats()
        return st

    def pick(self, pool: PoolState, exclude: set[str]) -> Replica | None:
        now = time.monotonic()
        best: tuple[bool, int, int] | None = None
        choice: Replica | None = None
        for i, r in enumerate(pool.replicas):
            if r.id in exclude:
                continue
            if not self._transport_ok(pool.model_id, r):
                continue
            st = self._st(r)
            key = (st.penalty_until > now, st.inflight, i)
            if best is None or key < best:
                best, choice = key, r
        return choice

    def _transport_ok(self, model_id: str, replica: Replica) -> bool:
        """Outside LOOM_ENV=dev a replica is only ever reached over https: a plain http://
        URL would send the pod token and the prompt in clear text."""
        if self.settings.env == "dev" or replica.url.lower().startswith("https://"):
            return True
        self.metrics.upstream_failures.labels(model_id, replica.id).inc()
        log.error(
            "replica refused: not https outside LOOM_ENV=dev",
            extra={"fields": {"replica": replica.id}},
        )
        return False

    def acquire(self, replica: Replica) -> None:
        self._st(replica).inflight += 1

    def release(self, replica: Replica, *, failed: bool = False) -> None:
        st = self._st(replica)
        st.inflight = max(0, st.inflight - 1)
        if failed:
            st.penalty_until = time.monotonic() + self.settings.replica_penalty_s

    def inflight(self, replica_id: str) -> int:
        st = self._stats.get(replica_id)
        return st.inflight if st else 0
