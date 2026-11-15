"""A valid OpenAI chat request with a long prompt and no `max_tokens` must succeed, as it
does on OpenAI, without reserving more than before (docs/phase1-contracts.md §6 step 2).

Through the real mock engine (loom_bench.mock), which, like vLLM, refuses an explicit
`max_tokens` that does not fit the context next to the prompt."""

from __future__ import annotations

import json

import httpx
import pytest
from gateway_testkit import mtls_server_config, replica, serve, set_ready

from loom_bench.mock.config import MockConfig
from loom_bench.mock.content import TOKENIZER, render_chat_prompt
from loom_bench.mock.server import create_app as mock_app
from loom_gateway.request_body import prepare
from loom_platform.pricing import Price

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

QWEN = Price(77_000, 77_000, 368_000)
CONTEXT = 32_768


async def _mock_stack(make_stack, certs):
    stack = await make_stack(ready=False)
    cfg = MockConfig(models=[stack.small, stack.big], time_scale=0.01, max_model_len=CONTEXT)
    port = await stack.track_server(serve(mock_app(cfg), **mtls_server_config(certs)))
    await set_ready(stack.redis, stack.small, [replica("mock", port)])
    return stack


def _messages(words: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": " ".join(["word"] * words)}]


def _tokens(messages: list[dict[str, str]]) -> int:
    return len(list(TOKENIZER.pieces(render_chat_prompt(messages, None))))


@pytest.mark.parametrize("stream", [False, True])
async def test_long_prompt_without_max_tokens_succeeds(make_stack, certs, stream):
    stack = await _mock_stack(make_stack, certs)
    t = await stack.tenant()
    messages = _messages(20_000)
    prompt = _tokens(messages)
    assert 20_000 <= prompt < CONTEXT - 1 and prompt + 16_384 > CONTEXT
    body = {"model": stack.small, "messages": messages}
    if stream:
        body |= {"stream": True, "stream_options": {"include_usage": True}}
    # What the reservation was before this change, for the same request: unchanged.
    raw = json.dumps(body).encode()
    before = prepare(
        "chat",
        body,
        body_bytes=len(raw),
        model_id=stack.small,
        price=QWEN,
        max_context=CONTEXT,
        max_tokens_cap=16_384,
    ).max_cost_micros
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            content=raw,
        )
    assert r.status_code == 200, r.text
    assert int(r.headers["X-Loom-Max-Cost-Micros"]) == before
    if stream:
        chunks = [json.loads(e[6:]) for e in r.text.split("\n\n") if e.startswith("data: {")]
        usage = chunks[-1]["usage"]
        assert r.text.rstrip().endswith("data: [DONE]")
    else:
        usage = r.json()["usage"]
    assert usage["prompt_tokens"] == prompt
    assert 1 <= usage["completion_tokens"] <= CONTEXT - prompt
    [ev] = await stack.wait_events(1)
    assert ev.status == "ok" and ev.prompt_tokens == prompt
    assert ev.cost_micros == usage["cost_micros"] <= before
    assert await stack.reserved(t) == (0, 0)
    fitted = stack.gateway.metrics.max_tokens_fitted.labels(stack.small)._value.get()
    assert fitted == 1


async def test_prompt_over_the_context_is_still_refused(make_stack, certs):
    stack = await _mock_stack(make_stack, certs)
    t = await stack.tenant()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": _messages(40_000)},
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"
    [ev] = await stack.wait_events(1)
    assert ev.status == "error" and ev.cost_micros == 0


async def test_explicit_max_tokens_is_never_rewritten(make_stack, certs):
    """The client asked for this many: too many for its prompt is its 400, as on vLLM."""
    stack = await _mock_stack(make_stack, certs)
    t = await stack.tenant()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": _messages(20_000), "max_tokens": 16_000},
        )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"
