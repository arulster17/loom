"""Model pool state shared by the pool manager (writer) and the gateway (reader).

`loom:pool:<model_id>` is a hash: state, retry_after_s, replicas (JSON), updated_at_ms.
In M1 a pool simulator (or a seed command) writes it; from M2 the pool manager does.
A model with no hash is treated as cold. See docs/phase1-contracts.md, "Model pools".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, cast

from redis.asyncio import Redis

PoolStateName = Literal["ready", "warming", "cold", "unavailable"]
WAKE_CHANNEL = "loom:pool:wake"
WAKE_TTL_S = 600


def pool_key(model_id: str) -> str:
    return f"loom:pool:{model_id}"


def wake_key(model_id: str) -> str:
    return f"loom:pool:{model_id}:wake"


@dataclass(frozen=True)
class Replica:
    id: str  # pod id (RunPod) or a local name
    url: str  # https://<host>:<port> of the pod's mTLS proxy
    server_name: str  # expected TLS server name in the proxy's certificate
    token: str  # per-pod request token, sent as X-Loom-Pod-Token


@dataclass(frozen=True)
class PoolState:
    model_id: str
    state: PoolStateName
    # For 503s: how long until the model is likely ready (cold start estimate) or
    # worth retrying (no capacity).
    retry_after_s: int
    replicas: tuple[Replica, ...] = field(default_factory=tuple)
    updated_at_ms: int = 0

    @property
    def routable(self) -> bool:
        return self.state == "ready" and bool(self.replicas)


def _str(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


async def get_pool(redis: Redis, model_id: str, *, cold_retry_after_s: int = 300) -> PoolState:
    raw = await redis.hgetall(pool_key(model_id))
    if not raw:
        return PoolState(model_id, "cold", cold_retry_after_s)
    h = {_str(k): _str(v) for k, v in raw.items()}
    replicas = tuple(Replica(**r) for r in json.loads(h.get("replicas") or "[]"))
    state = h["state"]
    if state not in ("ready", "warming", "cold", "unavailable"):
        raise ValueError(f"bad pool state {state!r} for {model_id}")
    return PoolState(
        model_id=model_id,
        state=cast(PoolStateName, state),
        retry_after_s=int(h.get("retry_after_s") or cold_retry_after_s),
        replicas=replicas,
        updated_at_ms=int(h.get("updated_at_ms") or 0),
    )


async def set_pool(redis: Redis, pool: PoolState) -> None:
    await redis.hset(
        pool_key(pool.model_id),
        mapping={
            "state": pool.state,
            "retry_after_s": pool.retry_after_s,
            "replicas": json.dumps([r.__dict__ for r in pool.replicas]),
            "updated_at_ms": pool.updated_at_ms or int(time.time() * 1000),
        },
    )


async def request_wake(redis: Redis, model_id: str) -> bool:
    """Ask the pool manager to start a cold model. True if this call made the request
    (later calls within WAKE_TTL_S are no-ops)."""
    made = bool(
        await redis.set(wake_key(model_id), int(time.time() * 1000), nx=True, ex=WAKE_TTL_S)
    )
    if made:
        await redis.publish(WAKE_CHANNEL, model_id)
    return made
