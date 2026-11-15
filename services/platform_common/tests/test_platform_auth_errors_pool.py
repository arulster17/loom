import asyncio
import uuid

import pytest

from loom_platform import errors
from loom_platform.auth import (
    MISS,
    AuthRecord,
    auth_cache_key,
    drop_cached,
    get_cached,
    load_auth_record,
    put_cached,
)
from loom_platform.limiter import ReserveResult
from loom_platform.pool import PoolState, Replica, get_pool, request_wake, set_pool, wake_key
from loom_platform.settings import PlatformSettings
from loom_platform.testing import create_api_key, create_org

RECORD = AuthRecord(
    key_id=uuid.uuid4(), org_id=uuid.uuid4(), org_status="active", revoked=False,
    scopes=("chat",), spend_limit_micros=None, key_rpm=5, key_tpm=None,
    key_concurrency=None, org_rpm=None, org_tpm=1000, org_concurrency=None,
)  # fmt: skip


def test_auth_record_json_and_limits():
    assert AuthRecord.from_json(RECORD.to_json()) == RECORD
    assert AuthRecord.from_json('{"found":false}') is None
    s = PlatformSettings()
    lim = RECORD.limits(s)
    assert (lim.key_rpm, lim.org_tpm) == (5, 1000)
    assert lim.key_tpm == s.default_key_tpm and lim.org_rpm == s.default_org_rpm
    assert RECORD.usable
    assert not AuthRecord(**{**RECORD.__dict__, "org_status": "suspended"}).usable


@pytest.mark.postgres
async def test_load_auth_record(pg):
    org = await create_org(pg, status="pending_approval")
    key_id, key = await create_api_key(pg, org, spend_limit_micros=7)
    rec = await load_auth_record(pg, key.key_hash)
    assert rec is not None
    assert (rec.key_id, rec.org_id, rec.org_status, rec.revoked) == (
        key_id, org, "pending_approval", False,
    )  # fmt: skip
    assert rec.spend_limit_micros == 7 and not rec.usable
    await pg.execute("UPDATE api_keys SET revoked_at = now() WHERE id = %s", (key_id,))
    rec = await load_auth_record(pg, key.key_hash)
    assert rec is not None and rec.revoked
    assert await load_auth_record(pg, b"\0" * 32) is None


@pytest.mark.redis
async def test_auth_cache_ttl_and_negative_entries(redis_client):
    h = uuid.uuid4().bytes * 2
    assert await get_cached(redis_client, h) is MISS
    await put_cached(redis_client, h, RECORD, ttl_s=1)
    assert await get_cached(redis_client, h) == RECORD
    assert 0 < await redis_client.ttl(auth_cache_key(h)) <= 1
    await put_cached(redis_client, h, None, ttl_s=1)
    assert await get_cached(redis_client, h) is None
    await drop_cached(redis_client, h)
    assert await get_cached(redis_client, h) is MISS
    await put_cached(redis_client, h, RECORD, ttl_s=1)
    await asyncio.sleep(1.2)
    assert await get_cached(redis_client, h) is MISS


def test_error_shapes():
    e = errors.invalid_api_key()
    assert e.status == 401
    assert e.body() == {
        "error": {
            "message": "Invalid or revoked API key.",
            "type": "invalid_request_error",
            "param": None,
            "code": "invalid_api_key",
        }
    }
    assert errors.org_not_active("pending_approval").body()["error"]["code"] == (
        "org_pending_approval"
    )
    assert errors.org_not_active("suspended").status == 403
    assert errors.model_not_found("x").status == 404
    u = errors.model_unavailable("qwen3-8b", 240, starting=True)
    assert (u.status, u.code, u.headers) == (503, "model_unavailable", {"Retry-After": "240"})
    assert "starting up" in u.message
    assert errors.model_unavailable("m", 300, starting=False).code == "model_unavailable"
    assert errors.upstream_lost().sse_chunk().startswith(b'data: {"error": ')


@pytest.mark.parametrize(
    ("reason", "status", "code"),
    [
        ("insufficient_balance", 402, "insufficient_balance"),
        ("key_spend_limit", 402, "key_spend_limit"),
        ("key_rpm", 429, "rate_limit_exceeded"),
        ("org_tpm", 429, "rate_limit_exceeded"),
        ("key_concurrency", 429, "concurrency_limit_exceeded"),
        ("org_concurrency", 429, "concurrency_limit_exceeded"),
        ("duplicate_request", 400, "duplicate_request"),
    ],
)
def test_reserve_reasons_map_to_errors(reason, status, code):
    e = errors.from_reserve(ReserveResult(False, reason, 1500, 10), needed_micros=99)
    assert (e.status, e.code) == (status, code)
    if status == 429:
        assert e.headers == {"Retry-After": "2"}


def test_noinit_is_not_an_error():
    with pytest.raises(ValueError):
        errors.from_reserve(ReserveResult(False, "noinit", 0, 0), needed_micros=1)


@pytest.mark.redis
async def test_pool_state_round_trip_and_wake(redis_client):
    model = f"test-model-{uuid.uuid4().hex[:8]}"
    cold = await get_pool(redis_client, model, cold_retry_after_s=240)
    assert (cold.state, cold.retry_after_s, cold.routable) == ("cold", 240, False)
    ready = PoolState(
        model, "ready", 0, (Replica("mock-pod-1", "https://mockpod:8443", "mock-pod-1", "t"),)
    )
    await set_pool(redis_client, ready)
    got = await get_pool(redis_client, model)
    assert got.routable and got.replicas == ready.replicas and got.updated_at_ms > 0
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("loom:pool:wake")
    assert await request_wake(redis_client, model)
    assert not await request_wake(redis_client, model)
    msg = None
    for _ in range(20):
        msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if msg:
            break
    assert msg is not None and msg["data"] == model.encode()
    await pubsub.aclose()
    await redis_client.delete(f"loom:pool:{model}", wake_key(model))
