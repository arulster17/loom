"""Billing-critical paths: aborted streams, upstream loss, retries before the first byte
only, revocation within 5 s, tenant isolation, and no balance overshoot under load."""

from __future__ import annotations

import asyncio
import json
import time
import uuid

import httpx
import openai
import psycopg
import pytest
from gateway_testkit import HIGH_LIMITS, replica, set_ready

from loom_platform.limiter import org_key
from loom_platform.pool import Replica
from loom_platform.pricing import Price, cost_micros

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

MESSAGES = [{"role": "user", "content": "Count."}]
QWEN = Price(77_000, 77_000, 368_000)


async def _stream_text(stack, key: str, **body) -> tuple[int, list]:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": stack.small, "messages": MESSAGES, "stream": True, **body},
        )
    objs = []
    for event in r.text.split("\n\n"):
        if event.startswith("data: "):
            p = event[6:]
            objs.append(p if p == "[DONE]" else json.loads(p))
    return r.status_code, objs


# ------------------------------------------------------------------ aborts


async def test_client_abort_bills_exactly_the_streamed_tokens(stack):
    t = await stack.tenant()
    stack.engine.behavior.tokens = 50
    stack.engine.behavior.token_delay_s = 0.25
    received = 0
    async with (
        httpx.AsyncClient(timeout=30) as c,
        c.stream(
            "POST",
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": MESSAGES, "stream": True},
        ) as r,
    ):
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data: {"):
                delta = json.loads(line[6:])["choices"][0]["delta"]
                if delta.get("content"):
                    received += 1
            if received == 4:
                break  # leaving the block closes the connection mid-stream
    [event] = await stack.wait_events(1)
    assert event.status == "aborted"
    assert (event.prompt_tokens, event.completion_tokens) == (11, 4)
    assert event.cost_micros == cost_micros(QWEN, 11, 4, 0)
    assert event.ttft_ms is not None
    assert await stack.reserved(t) == (0, 0)
    assert await stack.usage(t) == event.cost_micros
    # The gateway closed the upstream stream, so the engine stopped generating.
    await asyncio.sleep(0.6)
    assert stack.engine.aborted == 1
    assert stack.engine.generated[0] in (4, 5)


async def test_client_abort_without_engine_usage_is_flagged(stack):
    """Contract gap check: an engine without per-chunk usage leaves an aborted stream
    with no engine count; it is billed nothing and counted in usage_missing."""
    t = await stack.tenant()
    stack.engine.behavior.tokens = 50
    stack.engine.behavior.token_delay_s = 0.25
    stack.engine.behavior.continuous_usage = False
    async with (
        httpx.AsyncClient(timeout=30) as c,
        c.stream(
            "POST",
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": MESSAGES, "stream": True},
        ) as r,
    ):
        n = 0
        async for line in r.aiter_lines():
            n += line.startswith("data: ")
            if n == 3:
                break
    [event] = await stack.wait_events(1)
    assert event.status == "aborted" and event.completion_tokens == 0
    metrics = stack.gateway.metrics.render().decode()
    assert f'loom_gateway_usage_missing_total{{model="{stack.small}"}} 1.0' in metrics


async def test_upstream_lost_mid_stream_is_not_retried(make_stack):
    stack = await make_stack(engines=("e1", "e2"))
    t = await stack.tenant()
    e1, e2 = stack.engines["e1"], stack.engines["e2"]
    e1.behavior.tokens = 10
    e1.behavior.drop_after = 3
    status, objs = await _stream_text(stack, t.key)
    assert status == 200
    content = [o for o in objs if isinstance(o, dict) and "choices" in o]
    assert "".join(o["choices"][0]["delta"].get("content") or "" for o in content) == (
        "tok1 tok2 tok3 "
    )
    assert objs[-2]["error"]["code"] == "upstream_lost"
    assert objs[-2]["error"]["type"] == "server_error"
    assert objs[-1] == "[DONE]"
    [event] = await stack.wait_events(1)
    assert event.status == "aborted" and event.completion_tokens == 3
    assert e2.requests == []  # tokens had reached the client: never retried
    assert await stack.reserved(t) == (0, 0)

    # The SDK surfaces the error chunk as an exception after the tokens.
    e2.behavior.tokens, e2.behavior.drop_after = 10, 3
    stream = await stack.openai(t).chat.completions.create(
        model=stack.small, messages=MESSAGES, stream=True
    )
    with pytest.raises(openai.APIError):
        async for _ in stream:
            pass


async def test_request_timeout_cuts_the_stream(make_stack):
    stack = await make_stack(request_timeout_s=1)
    t = await stack.tenant()
    stack.engine.behavior.tokens = 20
    stack.engine.behavior.token_delay_s = 0.3
    t0 = time.monotonic()
    _, objs = await _stream_text(stack, t.key)
    assert time.monotonic() - t0 < 3
    assert objs[-2]["error"]["code"] == "upstream_lost" and objs[-1] == "[DONE]"
    [event] = await stack.wait_events(1)
    assert event.status == "aborted"
    assert 1 <= event.completion_tokens <= 4


@pytest.mark.parametrize("stream", [False, True])
async def test_request_timeout_before_the_first_byte(make_stack, stream):
    """A replica that never answers is cut at LOOM_REQUEST_TIMEOUT_S, retries included,
    so the settle lands well inside the reservation's TTL."""
    stack = await make_stack(engines=("e1", "e2"), request_timeout_s=1)
    t = await stack.tenant()
    for e in stack.engines.values():
        e.behavior.first_token_delay_s = 3
    t0 = time.monotonic()
    with pytest.raises(openai.APIStatusError) as err:
        await stack.openai(t).chat.completions.create(
            model=stack.small, messages=MESSAGES, stream=stream
        )
    assert time.monotonic() - t0 < 2.5
    assert err.value.status_code == 502 and err.value.code == "upstream_lost"
    [event] = await stack.wait_events(1)
    assert event.cost_micros == 0
    assert await stack.reserved(t) == (0, 0)
    assert stack.gateway.router.inflight("e1") == 0
    # A non-streaming call cut mid-flight may have generated tokens the engine never
    # reported: it is flagged (aborted, usage_missing), not silently an error.
    metrics = stack.gateway.metrics.render().decode()
    missing = f'loom_gateway_usage_missing_total{{model="{stack.small}"}} 1.0' in metrics
    assert (event.status, missing) == (("error", False) if stream else ("aborted", True))


# ------------------------------------------------------------------ retries


@pytest.mark.parametrize("stream", [False, True])
async def test_retry_on_another_replica_before_first_byte(make_stack, stream):
    stack = await make_stack(engines=("e1", "e2"))
    t = await stack.tenant()
    e1, e2 = stack.engines["e1"], stack.engines["e2"]
    e1.behavior.fail_status = 503
    resp = await stack.openai(t).chat.completions.create(
        model=stack.small, messages=MESSAGES, stream=stream
    )
    if stream:
        text = "".join([c.choices[0].delta.content or "" async for c in resp if c.choices])
    else:
        text = resp.choices[0].message.content
    assert text == "tok1 tok2 tok3 tok4 tok5 "
    assert (len(e1.requests), len(e2.requests)) == (1, 1)
    [event] = await stack.wait_events(1)  # billed once
    assert event.status == "ok" and event.completion_tokens == 5
    # The failed replica is tried last for a while.
    await stack.openai(t).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert (len(e1.requests), len(e2.requests)) == (1, 2)


async def test_retry_when_stream_dies_before_its_first_event(make_stack):
    stack = await make_stack(engines=("e1", "e2"))
    t = await stack.tenant()
    stack.engines["e1"].behavior.drop_after = -1  # 200, then the connection drops
    status, objs = await _stream_text(stack, t.key)
    assert status == 200 and objs[-1] == "[DONE]"
    assert not any(isinstance(o, dict) and "error" in o for o in objs)
    assert len(stack.engines["e2"].requests) == 1
    [event] = await stack.wait_events(1)
    assert event.status == "ok"


async def test_retry_on_connection_refused(make_stack, redis_client):
    stack = await make_stack()
    t = await stack.tenant()
    dead = Replica("dead", "https://127.0.0.1:9", "mock-pod-1", "pod-token-test")
    await set_ready(redis_client, stack.small, [dead, replica("e1", stack.ports["e1"])])
    r = await stack.openai(t).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert r.choices[0].message.content
    metrics = stack.gateway.metrics.render().decode()
    assert f'loom_gateway_upstream_retries_total{{model="{stack.small}"}} 1.0' in metrics


async def test_every_replica_failing_is_502_and_unbilled(make_stack):
    stack = await make_stack(engines=("e1", "e2", "e3"))
    t = await stack.tenant()
    for e in stack.engines.values():
        e.behavior.fail_status = 500
    for stream in (False, True):
        with pytest.raises(openai.APIStatusError) as err:
            await stack.openai(t).chat.completions.create(
                model=stack.small, messages=MESSAGES, stream=stream
            )
        assert err.value.status_code == 502 and err.value.code == "upstream_lost"
    # One retry per request: two of the three replicas, never all three.
    assert sum(len(e.requests) for e in stack.engines.values()) == 4
    events = await stack.events()
    assert [(e.status, e.cost_micros) for e in events] == [("error", 0), ("error", 0)]
    assert await stack.reserved(t) == (0, 0)


# ------------------------------------------------------------------ mTLS


@pytest.mark.parametrize("case", ["no_client_cert", "rogue_client_cert", "wrong_server_name"])
async def test_mtls_is_enforced(make_stack, certs, redis_client, case):
    overrides: dict = {}
    if case == "no_client_cert":
        overrides = {"gateway_client_cert": None, "gateway_client_key": None}
    elif case == "rogue_client_cert":
        overrides = {
            "gateway_client_cert": str(certs.rogue_client_cert),
            "gateway_client_key": str(certs.rogue_client_key),
        }
    stack = await make_stack(**overrides)
    if case == "wrong_server_name":
        r = replica("e1", stack.ports["e1"])
        await set_ready(redis_client, stack.small, [Replica(r.id, r.url, "other-pod", r.token)])
    t = await stack.tenant()
    with pytest.raises(openai.APIStatusError) as err:
        await stack.openai(t).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert err.value.status_code == 502
    assert stack.engine.requests == []  # nothing got past the handshake


async def test_plain_http_replica_for_local_dev(make_stack):
    stack = await make_stack(tls=False)
    t = await stack.tenant()
    r = await stack.openai(t).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert r.usage.completion_tokens == 5


async def test_plain_http_replica_is_never_used_outside_dev(make_stack):
    """Outside LOOM_ENV=dev the pod token (and the prompt) never go over plain http."""
    stack = await make_stack(tls=False, env="prod")
    t = await stack.tenant()
    with pytest.raises(openai.APIStatusError) as e:
        await stack.openai(t).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert e.value.status_code == 502 and e.value.code == "upstream_lost"
    assert stack.engine.requests == []
    assert await stack.reserved(t) == (0, 0)


# ------------------------------------------------------------------ revocation


async def _poll_until(stack, key: str, status: int, limit_s: float) -> float:
    """Seconds until a request with `key` gets `status`."""
    t0 = time.monotonic()
    async with httpx.AsyncClient() as c:
        while time.monotonic() - t0 < limit_s:
            r = await c.post(
                f"{stack.url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": stack.small, "messages": MESSAGES, "max_tokens": 1},
            )
            if r.status_code == status:
                return time.monotonic() - t0
            assert r.status_code == 200, r.text
            await asyncio.sleep(0.05)
    raise AssertionError(f"still not {status} after {limit_s} s")


async def _revoke(pg_url: str, sql: str, arg: uuid.UUID) -> None:
    async with await psycopg.AsyncConnection.connect(pg_url) as conn:
        await conn.execute(sql, (arg,))  # type: ignore[arg-type]
        await conn.commit()


@pytest.mark.parametrize("listen", [True, False])
async def test_revoked_key_fails_within_5s(make_stack, listen):
    stack = await make_stack(auth_listen=listen)
    t = await stack.tenant(key_limits=HIGH_LIMITS, org_limits=HIGH_LIMITS)
    other = await stack.tenant(key_limits=HIGH_LIMITS, org_limits=HIGH_LIMITS)
    client = stack.openai(t)
    await client.chat.completions.create(model=stack.small, messages=MESSAGES, max_tokens=1)
    await client.chat.completions.create(model=stack.small, messages=MESSAGES, max_tokens=1)
    await _revoke(stack.pg_url, "UPDATE api_keys SET revoked_at = now() WHERE id = %s", t.key_id)
    elapsed = await _poll_until(stack, t.key, 401, 6.0)
    assert elapsed < 5.0
    if listen:
        assert elapsed < 1.0  # the NOTIFY dropped the cached record at once
    # Other keys are untouched.
    await stack.openai(other).chat.completions.create(
        model=stack.small, messages=MESSAGES, max_tokens=1
    )


async def test_suspended_org_fails_within_5s(stack):
    t = await stack.tenant(key_limits=HIGH_LIMITS, org_limits=HIGH_LIMITS)
    await stack.openai(t).chat.completions.create(
        model=stack.small, messages=MESSAGES, max_tokens=1
    )
    await _revoke(stack.pg_url, "UPDATE orgs SET status = 'suspended' WHERE id = %s", t.org_id)
    elapsed = await _poll_until(stack, t.key, 403, 6.0)
    assert elapsed < 1.0


# ------------------------------------------------------------------ tenants


async def test_cross_tenant_isolation(stack):
    a = await stack.tenant()
    b = await stack.tenant()
    poor = await stack.tenant(credit_micros=10)
    spoof = {
        "OpenAI-Organization": str(b.org_id),
        "X-Loom-Org-Id": str(b.org_id),
        "X-Loom-Request-Id": str(uuid.uuid4()),
    }
    client = stack.track(
        openai.AsyncOpenAI(
            base_url=f"{stack.url}/v1", api_key=a.key, max_retries=0, default_headers=spoof
        )
    )
    raw = await client.chat.completions.with_raw_response.create(
        model=stack.small, messages=MESSAGES, user=str(b.org_id)
    )
    assert raw.headers["x-loom-request-id"] != spoof["X-Loom-Request-Id"]
    stream = await client.chat.completions.create(model=stack.small, messages=MESSAGES, stream=True)
    async for _ in stream:
        pass
    events = await stack.wait_events(2)
    assert {(e.org_id, e.api_key_id) for e in events} == {(a.org_id, a.key_id)}
    assert await stack.usage(a) == sum(e.cost_micros for e in events)
    # B was never charged, never even loaded.
    assert await stack.redis.get(org_key(b.org_id, "usage")) is None

    # A key with no money cannot draw on another org's balance.
    poor_client = stack.track(
        openai.AsyncOpenAI(
            base_url=f"{stack.url}/v1",
            api_key=poor.key,
            max_retries=0,
            default_headers={"OpenAI-Organization": str(b.org_id)},
        )
    )
    with pytest.raises(openai.APIStatusError) as err:
        await poor_client.chat.completions.create(model=stack.small, messages=MESSAGES)
    assert err.value.status_code == 402
    assert await stack.redis.get(org_key(b.org_id, "usage")) is None
    assert len(await stack.events()) == 2


async def test_balance_cannot_overshoot_under_concurrency(stack):
    """Contracts §6 worst-case overshoot: zero when the engine honours max_tokens."""
    credit = 10_000
    t = await stack.tenant(credit_micros=credit, key_limits=HIGH_LIMITS, org_limits=HIGH_LIMITS)
    stack.engine.behavior.tokens = "max"  # every request generates its full max_tokens
    stack.engine.behavior.first_token_delay_s = 0.2

    async def one(stream: bool) -> int:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{stack.url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {t.key}"},
                json={
                    "model": stack.small,
                    "messages": MESSAGES,
                    "max_tokens": 2000,
                    "stream": stream,
                },
            )
            await r.aread()
            return r.status_code

    statuses: list[int] = []
    for _ in range(4):  # waves until the money runs out
        statuses += await asyncio.gather(*(one(i % 2 == 0) for i in range(40)))
    ok, refused = statuses.count(200), statuses.count(402)
    assert ok + refused == len(statuses)
    assert ok >= 10 and refused > 0
    events = await stack.wait_events(ok)
    assert len(events) == ok
    assert all(e.completion_tokens == 2000 for e in events)
    usage = await stack.usage(t)
    assert usage == sum(e.cost_micros for e in events)
    assert usage <= credit  # zero overshoot
    assert await stack.reserved(t) == (0, 0)
    assert credit - usage < 1000  # and it was nearly all spent
