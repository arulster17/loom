"""The public API end to end: the OpenAI SDK unchanged against the gateway, a fake engine
behind mTLS, real Postgres and Redis."""

from __future__ import annotations

import json
import uuid

import httpx
import openai
import pytest

from loom_platform.pool import PoolState, Replica, set_pool, wake_key
from loom_platform.pricing import Price, cost_micros, max_cost_micros, prompt_tokens_upper_bound

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

MESSAGES = [{"role": "user", "content": "Say hello."}]
QWEN = Price(77_000, 77_000, 368_000)


def sse_objects(text: str) -> list[dict | str]:
    out: list[dict | str] = []
    for event in text.split("\n\n"):
        if event.startswith("data: "):
            payload = event[6:]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


async def test_sdk_chat_non_streaming(stack):
    t = await stack.tenant()
    client = stack.openai(t)
    raw = await client.chat.completions.with_raw_response.create(
        model=stack.small, messages=MESSAGES, max_tokens=50
    )
    resp = raw.parse()
    assert resp.choices[0].message.content == "tok1 tok2 tok3 tok4 tok5 "
    u = resp.usage
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (11, 5, 16)
    assert u.prompt_tokens_details.cached_tokens == 3
    cost = cost_micros(QWEN, 11, 5, 3)
    assert u.model_extra["cost_micros"] == cost
    h = raw.headers
    assert h["x-loom-cost-micros"] == str(cost)
    assert h["x-loom-cost-usd"] == f"0.{cost:06d}"
    assert h["x-loom-price-version"] == QWEN.version
    rid = uuid.UUID(h["x-loom-request-id"])
    assert int(h["x-loom-max-cost-micros"]) > cost

    [event] = await stack.wait_events(1)
    assert event.id == rid
    assert (event.org_id, event.api_key_id) == (t.org_id, t.key_id)
    assert (event.prompt_tokens, event.completion_tokens, event.cached_prompt_tokens) == (11, 5, 3)
    assert event.status == "ok" and event.cost_micros == cost
    assert event.org_usage_total_micros == cost
    assert event.model_id == stack.small and event.cloud == "local"
    assert await stack.reserved(t) == (0, 0)
    assert await stack.usage(t) == cost


async def test_sdk_chat_streaming_with_usage(stack):
    t = await stack.tenant()
    client = stack.openai(t)
    chunks = [
        c
        async for c in await client.chat.completions.create(
            model=stack.small,
            messages=MESSAGES,
            stream=True,
            stream_options={"include_usage": True},
        )
    ]
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == "tok1 tok2 tok3 tok4 tok5 "
    # Usage is null on every chunk but the last (OpenAI semantics), which has no choices.
    assert all(c.usage is None for c in chunks[:-1])
    last = chunks[-1]
    assert last.choices == []
    assert (last.usage.prompt_tokens, last.usage.completion_tokens) == (11, 5)
    assert last.usage.prompt_tokens_details.cached_tokens == 3
    assert last.usage.model_extra["cost_micros"] == cost_micros(QWEN, 11, 5, 3)
    [event] = await stack.events()  # settled before [DONE] reached the client
    assert event.status == "ok" and event.completion_tokens == 5 and event.ttft_ms is not None
    assert event.cost_micros == cost_micros(QWEN, 11, 5, 3)


async def test_stream_without_include_usage_shows_no_usage(stack):
    t = await stack.tenant()
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": MESSAGES, "stream": True},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "x-loom-cost-micros" not in r.headers
    assert "x-loom-max-cost-micros" in r.headers
    objs = sse_objects(r.text)
    assert objs[-1] == "[DONE]"
    assert all("usage" not in o for o in objs[:-1])
    assert all(o["choices"] for o in objs[:-1])
    # The engine was asked for usage on every chunk all the same, and it was billed.
    sent = stack.engine.requests[0]
    assert sent["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    [event] = await stack.events()
    assert event.completion_tokens == 5 and event.status == "ok"


async def test_stream_continuous_usage_when_asked(stack):
    t = await stack.tenant()
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={
                "model": stack.small,
                "messages": MESSAGES,
                "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            },
        )
    objs = sse_objects(r.text)
    content = [o for o in objs if isinstance(o, dict) and o["choices"]]
    assert [o["usage"]["completion_tokens"] for o in content] == [0, 1, 2, 3, 4, 5]
    final = objs[-2]
    assert final["choices"] == [] and final["usage"]["cost_micros"] == cost_micros(QWEN, 11, 5, 3)


async def test_upstream_request_rewrite(stack):
    """max_tokens is always explicit; no client header reaches the pod; the pod token
    does; model stays the registry id."""
    t = await stack.tenant()
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {t.key}",
                "X-Forwarded-For": "1.2.3.4",
                "Cookie": "secret=1",
                "OpenAI-Organization": "org-other",
            },
            json={"model": stack.small, "messages": MESSAGES, "stream_options": {"x": 1}},
        )
    assert r.status_code == 200
    body, headers = stack.engine.requests[0], stack.engine.headers[0]
    assert body["max_tokens"] == stack.gateway.settings.max_tokens_cap
    assert body["model"] == stack.small
    assert "stream_options" not in body  # vLLM refuses it on non-streaming requests
    assert headers["x-loom-pod-token"]
    for h in ("authorization", "x-forwarded-for", "cookie", "openai-organization"):
        assert h not in headers
    # The reservation covered the cap.
    spec = stack.gateway.registry.get(stack.small)
    upper = prompt_tokens_upper_bound(len(r.request.content), spec.max_context)
    expected = max_cost_micros(QWEN, upper, 16384, spec.max_context)
    assert int(r.headers["x-loom-max-cost-micros"]) == expected


async def test_max_completion_tokens_is_capped_and_mirrored(stack):
    t = await stack.tenant()
    client = stack.openai(t)
    await client.chat.completions.create(
        model=stack.small, messages=MESSAGES, max_completion_tokens=7, max_tokens=9
    )
    body = stack.engine.requests[0]
    assert body["max_tokens"] == 7 and body["max_completion_tokens"] == 7


async def test_tools_and_response_format_pass_through(stack):
    t = await stack.tenant()
    client = stack.openai(t)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    resp = await client.chat.completions.create(
        model=stack.small, messages=MESSAGES, tools=tools, tool_choice="auto"
    )
    call = resp.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}
    assert stack.engine.requests[0]["tools"] == tools
    assert stack.engine.requests[0]["tool_choice"] == "auto"

    # Streaming tool calls.
    stream = await client.chat.completions.create(
        model=stack.small, messages=MESSAGES, tools=tools, stream=True
    )
    args = ""
    async for c in stream:
        for tc in (c.choices[0].delta.tool_calls or []) if c.choices else []:
            args += tc.function.arguments or ""
    assert json.loads(args) == {"city": "Paris"}

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object", "properties": {"answer": {"type": "integer"}}},
        },
    }
    resp = await client.chat.completions.create(
        model=stack.small, messages=MESSAGES, response_format=schema
    )
    assert json.loads(resp.choices[0].message.content) == {"answer": 42}
    assert stack.engine.requests[-1]["response_format"] == schema
    assert len(await stack.events()) == 3


async def test_sdk_completions(stack):
    t = await stack.tenant()
    client = stack.openai(t)
    resp = await client.completions.create(model=stack.big, prompt="Once upon a time")
    assert resp.choices[0].text == "tok1 tok2 tok3 tok4 tok5 "
    assert resp.usage.completion_tokens == 5
    # Explicit max_tokens: OpenAI's default for completions.
    assert stack.engine.requests[0]["max_tokens"] == 16

    stream = await client.completions.create(
        model=stack.big,
        prompt="Once upon a time",
        max_tokens=3,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = [c async for c in stream]
    assert "".join(c.choices[0].text for c in chunks if c.choices) == "tok1 tok2 tok3 "
    assert chunks[-1].usage.completion_tokens == 3
    events = await stack.events()
    big = Price(571_000, 571_000, 797_000)
    assert [e.cost_micros for e in events] == [
        cost_micros(big, 11, 5, 3),
        cost_micros(big, 11, 3, 3),
    ]
    assert all(e.price_version == big.version for e in events)


async def test_models_endpoint(stack, redis_client):
    await set_pool(redis_client, PoolState(stack.big, "unavailable", 300))
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{stack.url}/v1/models")
    assert r.status_code == 200
    data = {m["id"]: m for m in r.json()["data"]}
    assert set(data) == {stack.small, stack.big}  # the unpriced row is not offered
    small = data[stack.small]
    assert small["context_length"] == 32768
    assert small["status"] == "preview"
    assert small["pricing"]["input_per_mtok_micros"] == 77_000
    assert small["pricing"]["output_per_mtok_micros"] == 368_000
    assert small["pricing"]["input_per_mtok_usd"] == "0.077000"
    assert small["availability"] == {"state": "ready", "ready_replicas": 1, "retry_after_s": None}
    assert data[stack.big]["availability"]["state"] == "unavailable"

    # The SDK lists them too (with a key, as it always sends one).
    t = await stack.tenant()
    page = await stack.openai(t).models.list()
    assert {m.id for m in page.data} == {stack.small, stack.big}


async def test_models_endpoint_real_registry():
    """The default app lists config/models.yaml's servable rows."""
    from loom_bench.registry import load_registry
    from loom_platform.pricing import servable

    ids = {m.id for m in servable(load_registry())}
    assert ids == {"qwen3-8b", "llama-3.3-70b-instruct-fp8"}


# ------------------------------------------------------------------ errors


async def _post(stack, key: str | None, body: dict, **kw) -> httpx.Response:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient() as c:
        return await c.post(f"{stack.url}/v1/chat/completions", headers=headers, json=body, **kw)


def _err(r: httpx.Response) -> dict:
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    return body["error"]


async def test_401_shapes(stack):
    body = {"model": stack.small, "messages": MESSAGES}
    for key in (None, "nope", "loom_sk_" + "a" * 12 + "_" + "b" * 40):
        r = await _post(stack, key, body)
        assert r.status_code == 401
        assert _err(r)["code"] == "invalid_api_key"
        assert "x-loom-request-id" in r.headers
    with pytest.raises(openai.AuthenticationError):
        await stack.openai("loom_sk_" + "a" * 12 + "_" + "b" * 40).chat.completions.create(
            model=stack.small, messages=MESSAGES
        )
    assert await stack.events() == []


async def test_403_org_not_active(stack):
    pending = await stack.tenant(status="pending_approval")
    suspended = await stack.tenant(status="suspended")
    with pytest.raises(openai.PermissionDeniedError) as e:
        await stack.openai(pending).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert e.value.code == "org_pending_approval"
    with pytest.raises(openai.PermissionDeniedError) as e:
        await stack.openai(suspended).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert e.value.code == "org_suspended"
    assert await stack.events() == []


async def test_404_unknown_and_unpriced_models(stack):
    t = await stack.tenant()
    for model in ("gpt-4o", stack.unpriced):
        with pytest.raises(openai.NotFoundError) as e:
            await stack.openai(t).chat.completions.create(model=model, messages=MESSAGES)
        assert e.value.code == "model_not_found"
        assert e.value.param == "model"
    assert await stack.events() == []


async def test_400_validation(stack):
    t = await stack.tenant()
    cases = [
        {"model": stack.small, "messages": MESSAGES, "max_tokens": 16385},
        {"model": stack.small, "messages": MESSAGES, "max_completion_tokens": 0},
        {"model": stack.small, "messages": []},
        {"model": stack.small, "messages": MESSAGES, "n": 17},
        {"model": stack.small, "messages": MESSAGES, "stream": "yes"},
        {"messages": MESSAGES},
    ]
    for body in cases:
        r = await _post(stack, t.key, body)
        assert r.status_code == 400, body
        assert _err(r)["type"] == "invalid_request_error"
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            content=b"{not json",
        )
    assert r.status_code == 400
    with pytest.raises(openai.BadRequestError):
        await stack.openai(t).chat.completions.create(
            model=stack.small, messages=MESSAGES, max_tokens=100_000
        )
    assert stack.engine.requests == []
    assert await stack.events() == []


async def test_413_body_too_large(make_stack):
    stack = await make_stack(max_body_bytes=2048)
    t = await stack.tenant()
    big = [{"role": "user", "content": "x" * 4096}]
    r = await _post(stack, t.key, {"model": stack.small, "messages": big})
    assert r.status_code == 413
    assert _err(r)["code"] == "request_too_large"
    # Checked before the key: no lookup for an oversized body.
    r = await _post(stack, None, {"model": stack.small, "messages": big})
    assert r.status_code == 413


async def test_402_insufficient_balance_and_spend_limit(stack):
    poor = await stack.tenant(credit_micros=100)
    with pytest.raises(openai.APIStatusError) as e:
        await stack.openai(poor).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert e.value.status_code == 402
    assert e.value.code == "insufficient_balance"
    capped = await stack.tenant(spend_limit_micros=1000)
    with pytest.raises(openai.APIStatusError) as e:
        await stack.openai(capped).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert e.value.status_code == 402 and e.value.code == "key_spend_limit"
    # A small max_tokens fits both.
    await stack.openai(poor).chat.completions.create(
        model=stack.small, messages=MESSAGES, max_tokens=1
    )
    assert stack.engine.requests[0]["max_tokens"] == 1
    assert len(await stack.events()) == 1


async def test_429_rpm_and_concurrency(stack):
    t = await stack.tenant(key_limits={"rpm_limit": 2})
    client = stack.openai(t)
    for _ in range(2):
        await client.chat.completions.create(model=stack.small, messages=MESSAGES, max_tokens=5)
    with pytest.raises(openai.RateLimitError) as e:
        await client.chat.completions.create(model=stack.small, messages=MESSAGES, max_tokens=5)
    assert e.value.code == "rate_limit_exceeded"
    assert int(e.value.response.headers["retry-after"]) >= 1
    assert len(await stack.events()) == 2

    # Concurrency: one slot, held by an open stream.
    c = await stack.tenant(key_limits={"concurrency_limit": 1})
    stack.engine.behavior.token_delay_s = 0.2
    async with (
        httpx.AsyncClient() as http,
        http.stream(
            "POST",
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {c.key}"},
            json={"model": stack.small, "messages": MESSAGES, "stream": True},
        ) as first,
    ):
        assert first.status_code == 200
        r = await _post(stack, c.key, {"model": stack.small, "messages": MESSAGES})
        assert r.status_code == 429
        assert _err(r)["code"] == "concurrency_limit_exceeded"
        assert "retry-after" in r.headers
        await first.aread()


async def test_503_pool_states(stack, redis_client):
    t = await stack.tenant()
    client = stack.openai(t)
    # cold (no hash at all): wake requested, Retry-After = the cold-start estimate
    await redis_client.delete(f"loom:pool:{stack.big}")
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("loom:pool:wake")
    await pubsub.get_message(timeout=1)  # subscribe confirmation
    with pytest.raises(openai.InternalServerError) as e:
        await client.chat.completions.create(model=stack.big, messages=MESSAGES)
    assert e.value.status_code == 503
    assert e.value.code == "model_unavailable"
    assert e.value.response.headers["retry-after"] == "450"
    assert await redis_client.exists(wake_key(stack.big))
    msg = await pubsub.get_message(timeout=2)
    assert msg is not None and msg["data"] in (stack.big, stack.big.encode())
    await pubsub.aclose()

    # warming: Retry-After from the hash
    await set_pool(redis_client, PoolState(stack.big, "warming", 37))
    r = await _post(stack, t.key, {"model": stack.big, "messages": MESSAGES})
    assert r.status_code == 503 and r.headers["retry-after"] == "37"
    assert "starting up" in _err(r)["message"]

    # unavailable: no capacity
    await set_pool(redis_client, PoolState(stack.big, "unavailable", 300))
    r = await _post(stack, t.key, {"model": stack.big, "messages": MESSAGES})
    assert r.status_code == 503 and r.headers["retry-after"] == "300"
    assert "no GPU capacity" in _err(r)["message"]

    # ready with no replica counts as cold
    await set_pool(redis_client, PoolState(stack.big, "ready", 0, ()))
    r = await _post(stack, t.key, {"model": stack.big, "messages": MESSAGES})
    assert r.status_code == 503

    # No fallback to the other model, no reservation, no usage event.
    assert stack.engine.requests == []
    assert await stack.events() == []
    assert await stack.reserved(t) == (0, 0)

    await set_pool(
        redis_client,
        PoolState(
            stack.big,
            "ready",
            0,
            (
                Replica(
                    "e1", f"https://127.0.0.1:{stack.ports['e1']}", "mock-pod-1", "pod-token-test"
                ),
            ),
        ),
    )
    await client.chat.completions.create(model=stack.big, messages=MESSAGES)


async def test_engine_4xx_passes_through_unbilled(stack):
    t = await stack.tenant()
    stack.engine.behavior.fail_status = 400
    stack.engine.behavior.fail_message = (
        "This model's maximum context length is 32768 tokens. However, you requested 40000."
    )
    with pytest.raises(openai.BadRequestError) as e:
        await stack.openai(t).chat.completions.create(model=stack.small, messages=MESSAGES)
    assert e.value.code == "context_length_exceeded"
    assert "maximum context length" in e.value.message
    r = await _post(stack, t.key, {"model": stack.small, "messages": MESSAGES, "stream": True})
    assert r.status_code == 400
    events = await stack.events()
    assert [(e.status, e.cost_micros) for e in events] == [("error", 0), ("error", 0)]
    assert await stack.reserved(t) == (0, 0)
    assert len(stack.engine.requests) == 2  # never retried


async def test_cors_exposes_cost_headers(stack):
    async with httpx.AsyncClient() as c:
        r = await c.options(
            f"{stack.url}/v1/chat/completions",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
        r = await c.get(f"{stack.url}/v1/models", headers={"Origin": "http://localhost:3000"})
        exposed = r.headers["access-control-expose-headers"].lower()
        assert "x-loom-cost-micros" in exposed and "x-loom-request-id" in exposed
        r = await c.options(
            f"{stack.url}/v1/chat/completions",
            headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
        )
        assert "access-control-allow-origin" not in r.headers


async def test_unknown_route_is_openai_shaped(stack):
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{stack.url}/v1/embeddings", json={})
    assert r.status_code == 404
    _err(r)
