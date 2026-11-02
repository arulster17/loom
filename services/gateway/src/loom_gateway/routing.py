"""Model pools and replica choice (docs/phase1-contracts.md §9-10,
docs/phase1-runpod-plan.md §7).

Only `ready` replicas are candidates. Among them the gateway picks the one with the
fewest of its own open requests (exact with one gateway), trying a replica that failed
in the last `replica_penalty_s` only when nothing else is left, and ties going to the
first listed. No replica: 503 `model_unavailable` at once; the gateway never queues.
"""

from __future__ import annotations

import logging
import ssl
import time
from dataclasses import dataclass

import httpx
from redis.asyncio import Redis

from loom_gateway.metrics import GatewayMetrics
from loom_gateway.settings import COLD_START_ESTIMATE_S, DEFAULT_COLD_START_S, GatewaySettings
from loom_platform import errors
from loom_platform.pool import PoolState, Replica, get_pool, request_wake

log = logging.getLogger("loom_gateway.routing")


def upstream_client(s: GatewaySettings) -> httpx.AsyncClient:
    """One pooled client for every pod: TLS 1.3, client certificate, pod CA, HTTP/2 where
    the proxy offers it. The server name is checked per request (`sni_hostname`)."""
    if s.pod_ca_cert:
        ctx = ssl.create_default_context(cafile=s.pod_ca_cert)
    else:
        ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    if s.gateway_client_cert:
        ctx.load_cert_chain(s.gateway_client_cert, s.gateway_client_key)
    return httpx.AsyncClient(
        verify=ctx,
        http2=True,
        timeout=httpx.Timeout(
            connect=s.upstream_connect_timeout_s,
            read=float(s.request_timeout_s),
            write=30.0,
            pool=s.upstream_connect_timeout_s,
        ),
        limits=httpx.Limits(max_connections=2048, max_keepalive_connections=512),
        trust_env=False,
        follow_redirects=False,
    )


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
        self.cold_start_s = {**COLD_START_ESTIMATE_S, **(cold_start_s or {})}
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
        """Raise 503 model_unavailable (with Retry-After) unless the pool can serve."""
        if pool.routable:
            return
        if pool.state == "unavailable":
            raise errors.model_unavailable(pool.model_id, pool.retry_after_s or 300, False)
        if pool.state == "warming":
            raise errors.model_unavailable(
                pool.model_id,
                pool.retry_after_s or self.cold_start_estimate(pool.model_id),
                True,
            )
        # cold, missing, or 'ready' with no replica: ask for a start.
        if await request_wake(self.redis, pool.model_id):
            self.metrics.wakes.labels(pool.model_id).inc()
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
