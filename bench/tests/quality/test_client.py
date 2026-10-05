import json

import httpx
import pytest

from loom_bench.quality.client import EvalClient, EvalRequestError


def chat_body(content="ok", finish="stop", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": {}}


def make_client(handler, **kw):
    kw.setdefault("backoff_s", 0.0)
    return EvalClient("http://test/v1", "m", transport=httpx.MockTransport(handler), **kw)


async def test_deterministic_params_and_extra_body():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json=chat_body())

    extra = {"chat_template_kwargs": {"enable_thinking": False}, "top_p": 1.0}
    async with make_client(handler, seed=7, extra_body=extra) as c:
        res = await c.chat([{"role": "user", "content": "hi"}], max_tokens=5, top_p=0.5)
    assert res.text == "ok" and res.finish_reason == "stop"
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body["temperature"] == 0.0 and body["seed"] == 7 and body["model"] == "m"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["top_p"] == 0.5  # per-call params override extra_body
    assert body["stream"] is False and body["max_tokens"] == 5


async def test_retries_5xx_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=chat_body())

    async with make_client(handler, max_retries=4) as c:
        res = await c.chat([{"role": "user", "content": "x"}], max_tokens=1)
    assert res.text == "ok" and len(calls) == 3


async def test_retries_timeouts_then_gives_up():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=request)

    async with make_client(handler, max_retries=2) as c:
        with pytest.raises(EvalRequestError, match="after 3 attempts, last: timeout") as e:
            await c.complete("x", max_tokens=1)
    assert len(calls) == 3 and e.value.attempts == 3


async def test_4xx_is_not_retried_and_hides_prompt():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "max_tokens too large"}})

    async with make_client(handler) as c:
        with pytest.raises(EvalRequestError) as e:
            await c.chat([{"role": "user", "content": "SECRET PROMPT"}], max_tokens=1)
    assert len(calls) == 1 and e.value.status == 400
    assert "max_tokens too large" in str(e.value)
    assert "SECRET" not in str(e.value)


async def test_tool_calls_and_completions_logprobs():
    def handler(request):
        if request.url.path.endswith("/chat/completions"):
            call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            return httpx.Response(200, json=chat_body(None, "tool_calls", [call]))
        lp = {"tokens": ["a"], "token_logprobs": [None], "top_logprobs": [None], "text_offset": [0]}
        return httpx.Response(
            200, json={"choices": [{"text": "a b", "finish_reason": "length", "logprobs": lp}]}
        )

    async with make_client(handler) as c:
        chat = await c.chat([{"role": "user", "content": "x"}], max_tokens=8, tools=[])
        comp = await c.complete("a", max_tokens=1, echo=True, logprobs=5)
    assert chat.text == "" and chat.tool_calls[0].name == "f"
    assert comp.text == "a b" and comp.logprobs["tokens"] == ["a"]


async def test_concurrency_limit():
    import asyncio

    active = peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(200, json=chat_body())

    async with make_client(handler, concurrency=3) as c:
        await asyncio.gather(
            *(c.chat([{"role": "user", "content": "x"}], max_tokens=1) for _ in range(12))
        )
    assert peak == 3
