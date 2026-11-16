"""`PoolState.reason` (docs/phase1-m2-spec.md §7, §12, §13 item 2): the pool hash gains a
`reason` field, written and read; hashes written by M1 code (no `reason`) still load."""

from __future__ import annotations

import json
import time
import uuid

import pytest

from loom_platform.pool import PoolState, Replica, get_pool, pool_key, set_pool

pytestmark = pytest.mark.redis

REASONS = ["crash_loop", "spend_cap", "price_too_high", "not_configured", "no_capacity"]
REPLICA = Replica("pod-1", "https://203.0.113.1:40001", "pod-1", "tok")


def _model() -> str:
    return f"test-model-{uuid.uuid4().hex[:8]}"


def test_reason_defaults_to_none() -> None:
    pool = PoolState("m", "ready", 0, (REPLICA,))
    assert pool.reason is None
    assert PoolState("m", "unavailable", 300, reason="no_capacity").reason == "no_capacity"


@pytest.mark.parametrize("reason", REASONS)
async def test_reason_round_trips(redis_client, reason: str) -> None:
    model = _model()
    try:
        await set_pool(redis_client, PoolState(model, "unavailable", 300, reason=reason))
        got = await get_pool(redis_client, model)
        assert (got.state, got.retry_after_s, got.reason) == ("unavailable", 300, reason)
        assert got.replicas == ()
        # The field is `reason` in the hash itself (contracts §10).
        raw = await redis_client.hget(pool_key(model), "reason")
        assert raw in (reason, reason.encode())
    finally:
        await redis_client.delete(pool_key(model))


async def test_no_reason_round_trips_as_none(redis_client) -> None:
    model = _model()
    try:
        await set_pool(redis_client, PoolState(model, "ready", 0, (REPLICA,)))
        got = await get_pool(redis_client, model)
        assert got.routable and got.replicas == (REPLICA,) and got.reason is None
    finally:
        await redis_client.delete(pool_key(model))


async def test_rewriting_without_a_reason_clears_the_old_one(redis_client) -> None:
    """The manager rewrites the hash as a pool recovers: a stale `spend_cap` must not
    outlive the `unavailable` state it explained."""
    model = _model()
    try:
        await set_pool(redis_client, PoolState(model, "unavailable", 300, reason="spend_cap"))
        await set_pool(redis_client, PoolState(model, "ready", 0, (REPLICA,)))
        got = await get_pool(redis_client, model)
        assert (got.state, got.reason) == ("ready", None)
        await set_pool(redis_client, PoolState(model, "unavailable", 300, reason="crash_loop"))
        await set_pool(redis_client, PoolState(model, "unavailable", 120, reason="no_capacity"))
        got = await get_pool(redis_client, model)
        assert (got.retry_after_s, got.reason) == (120, "no_capacity")
    finally:
        await redis_client.delete(pool_key(model))


@pytest.mark.parametrize("state", ["ready", "warming", "cold", "unavailable"])
async def test_m1_hash_without_reason_still_loads(redis_client, state: str) -> None:
    """Exactly the fields M1's `set_pool` (and poolsim) write."""
    model = _model()
    replicas = [REPLICA.__dict__] if state == "ready" else []
    try:
        await redis_client.hset(
            pool_key(model),
            mapping={
                "state": state,
                "retry_after_s": 0 if state == "ready" else 240,
                "replicas": json.dumps(replicas),
                "updated_at_ms": int(time.time() * 1000),
            },
        )
        got = await get_pool(redis_client, model)
        assert got.state == state and got.reason is None
        assert got.routable == (state == "ready")
        assert got.replicas == ((REPLICA,) if state == "ready" else ())
    finally:
        await redis_client.delete(pool_key(model))


async def test_missing_hash_is_cold_without_reason(redis_client) -> None:
    got = await get_pool(redis_client, _model(), cold_retry_after_s=450)
    assert (got.state, got.retry_after_s, got.reason) == ("cold", 450, None)
