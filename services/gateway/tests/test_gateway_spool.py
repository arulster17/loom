"""Redis outages around settle (docs/phase1-contracts.md §6 "Settle spool"): a settle
Redis refuses is spooled to disk and replayed until Redis accepts it, exactly once; a
frozen Redis makes the gateway answer 503 within its socket timeout."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import time
import urllib.parse
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from gateway_testkit import Behavior, TcpProxy

from loom_gateway.metrics import GatewayMetrics
from loom_gateway.spool import SettleSpool
from loom_platform.limiter import SETTLED_TTL_S, Limiter, api_key_key, org_key
from loom_platform.pricing import Price, cost_micros
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV
from loom_platform.usage import UsageEvent

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

QWEN = Price(77_000, 77_000, 368_000)


@pytest.fixture
async def redis_proxy() -> AsyncIterator[tuple[TcpProxy, str]]:
    """A proxy in front of the test Redis and the redis URL that goes through it."""
    url = urllib.parse.urlsplit(os.environ.get(REDIS_ENV) or DEFAULT_REDIS)
    proxy = TcpProxy(url.hostname or "localhost", url.port or 6379)
    await proxy.start()
    via = url._replace(netloc=f"127.0.0.1:{proxy.port}").geturl()
    try:
        yield proxy, via
    finally:
        await proxy.stop()


async def _wait(pred, timeout_s: float) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if await pred():
            return True
        await asyncio.sleep(0.05)
    return False


async def test_settle_during_redis_outage_lands_once_after_it_returns(
    make_stack, redis_proxy, tmp_path: Path
) -> None:
    proxy, via = redis_proxy
    stack = await make_stack(redis_url=via, settle_spool_path=str(tmp_path / "spool.jsonl"))
    stack.engine.behavior = Behavior(tokens=12, token_delay_s=0.1)
    t = await stack.tenant()
    async with (
        httpx.AsyncClient(timeout=30) as c,
        c.stream(
            "POST",
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={
                "model": stack.small,
                "messages": [{"role": "user", "content": "Count."}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as r,
    ):
        assert r.status_code == 200
        lines = []
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                lines.append(line[6:])
                if len(lines) == 3:
                    await proxy.stop()  # Redis gone mid-stream
    assert lines[-1] == "[DONE]"
    usage = json.loads(lines[-2])["usage"]
    assert usage["completion_tokens"] == 12

    spool = stack.gateway.spool
    assert spool.depth == 1
    assert await stack.events() == []  # nothing reached Redis
    assert await stack.reserved(t) != (0, 0)  # still held while Redis is away

    await proxy.start()
    assert await _wait(lambda: _has_events(stack, 1), 15), "replay never landed"
    # Another replay round must not add a second copy.
    await asyncio.sleep(1.5)
    events = await stack.events()
    assert len(events) == 1
    ev = events[0]
    assert (ev.prompt_tokens, ev.completion_tokens, ev.status) == (11, 12, "ok")
    assert ev.cost_micros == cost_micros(QWEN, 11, 12, 3) == usage["cost_micros"]
    assert await stack.usage(t) == ev.cost_micros
    assert await stack.reserved(t) == (0, 0)
    inflight = await stack.redis.get(api_key_key(t.org_id, t.key_id, "inflight"))
    assert int(inflight or 0) == 0
    assert spool.depth == 0
    assert (tmp_path / "spool.jsonl").read_text() == ""
    assert stack.gateway.metrics.settle_failures._value.get() == 0


async def _has_events(stack, n: int) -> bool:  # type: ignore[no-untyped-def]
    return len(await stack.events()) >= n


async def test_frozen_redis_answers_503_fast(make_stack, redis_proxy) -> None:
    proxy, via = redis_proxy
    stack = await make_stack(redis_url=via, redis_socket_timeout_s=1.0)
    t = await stack.tenant()
    proxy.freeze()
    async with httpx.AsyncClient(timeout=30) as c:
        start = time.monotonic()
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": [{"role": "user", "content": "hi"}]},
        )
        took = time.monotonic() - start
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "service_unavailable"
        assert r.headers["Retry-After"] == "1"
        assert took < 5
        start = time.monotonic()
        ready = await c.get(f"{stack.url}/ready")
        assert ready.status_code == 503
        assert time.monotonic() - start < 5
    await proxy.stop()
    await proxy.start()


def _event(org: uuid.UUID, key: uuid.UUID, *, age_s: float = 0) -> UsageEvent:
    return UsageEvent(
        id=uuid.uuid4(),
        org_id=org,
        api_key_id=key,
        model_id="qwen3-8b",
        price_version=QWEN.version,
        cloud="local",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=5,
        status="ok",
        cost_micros=cost_micros(QWEN, 10, 5),
        created_at=dt.datetime.now(dt.UTC) - dt.timedelta(seconds=age_s),
    )


async def _loaded_org(redis, org: uuid.UUID, key: uuid.UUID) -> Limiter:  # type: ignore[no-untyped-def]
    stream = f"loom:test:usage:{uuid.uuid4().hex}"
    limiter = Limiter(redis, usage_stream=stream)
    await limiter.init_org(
        org_id=org, key_id=key, credit_total=10**6, usage_total=0, key_usage_total=0
    )
    return limiter


async def test_replay_of_a_settle_that_did_land_is_not_charged_twice(
    redis_client, tmp_path: Path
) -> None:
    """The reply was lost (the client saw an error) but Redis ran the settle: the replay
    meets the done marker and changes nothing."""
    org, key = uuid.uuid4(), uuid.uuid4()
    limiter = await _loaded_org(redis_client, org, key)
    ev = _event(org, key)
    assert (await limiter.settle(ev)).settled
    spool = SettleSpool(
        tmp_path / "s.jsonl", limiter, GatewayMetrics(), settled_ttl_s=SETTLED_TTL_S
    )
    spool.open()
    await spool.append(ev)
    assert await spool.replay_once()
    assert spool.depth == 0
    assert await redis_client.xlen(limiter.usage_stream) == 1
    assert int(await redis_client.get(org_key(org, "usage"))) == ev.cost_micros
    await spool.stop()
    await redis_client.delete(limiter.usage_stream)


async def test_spool_survives_a_restart_and_a_torn_line(redis_client, tmp_path: Path) -> None:
    org, key = uuid.uuid4(), uuid.uuid4()
    limiter = await _loaded_org(redis_client, org, key)
    path = tmp_path / "s.jsonl"
    events = [_event(org, key), _event(org, key)]
    path.write_text("".join(e.model_dump_json() + "\n" for e in events) + '{"id": "torn')
    spool = SettleSpool(path, limiter, GatewayMetrics(), settled_ttl_s=SETTLED_TTL_S)
    spool.open()
    assert spool.depth == 2
    with pytest.raises(RuntimeError, match="in use"):
        SettleSpool(path, limiter, GatewayMetrics(), settled_ttl_s=SETTLED_TTL_S).open()
    assert await spool.replay_once()
    assert await redis_client.xlen(limiter.usage_stream) == 2
    assert int(await redis_client.get(org_key(org, "usage"))) == 2 * events[0].cost_micros
    assert path.read_text() == ""
    await spool.stop()
    await redis_client.delete(limiter.usage_stream)


async def test_old_entry_already_billed_is_dropped(redis_client, tmp_path: Path) -> None:
    """Past the done marker's lifetime the replay asks Postgres first."""
    org, key = uuid.uuid4(), uuid.uuid4()
    limiter = await _loaded_org(redis_client, org, key)
    billed = _event(org, key, age_s=SETTLED_TTL_S + 10)
    fresh = _event(org, key)
    seen: list[uuid.UUID] = []

    async def in_postgres(e: UsageEvent) -> bool:
        seen.append(e.id)
        return e.id == billed.id

    spool = SettleSpool(
        tmp_path / "s.jsonl",
        limiter,
        GatewayMetrics(),
        settled_ttl_s=SETTLED_TTL_S,
        settled_check=in_postgres,
    )
    spool.open()
    await spool.append(billed)
    await spool.append(fresh)
    assert await spool.replay_once()
    assert seen == [billed.id]  # the fresh one is covered by the done marker
    entries = await redis_client.xrange(limiter.usage_stream)
    assert [UsageEvent.from_stream_fields(f).id for _, f in entries] == [fresh.id]
    await spool.stop()
    await redis_client.delete(limiter.usage_stream)
