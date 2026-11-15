"""`n > 1` against the real mock engine (loom_bench.mock), which streams the choices
interleaved with vLLM's per-choice continuous usage: each chunk's `completion_tokens` is
its own choice's running count; only the final usage chunk sums the choices."""

from __future__ import annotations

import json

import httpx
import pytest
from gateway_testkit import mtls_server_config, replica, serve, set_ready

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app as mock_app
from loom_platform.pricing import Price, cost_micros

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

QWEN = Price(77_000, 77_000, 368_000)
STORY = [{"role": "user", "content": "Write a long story about batching."}]
CONTINUOUS = {"include_usage": True, "continuous_usage_stats": True}


async def _mock_stack(make_stack, certs, **config):
    stack = await make_stack(ready=False)
    cfg = MockConfig(models=[stack.small, stack.big], time_scale=0.01, **config)
    port = await stack.track_server(serve(mock_app(cfg), **mtls_server_config(certs)))
    await set_ready(stack.redis, stack.small, [replica("mock", port)])
    return stack


def _per_choice_counts(chunks: list[dict]) -> dict[int, int]:
    """Highest running completion count the client saw, per choice."""
    out: dict[int, int] = {}
    for c in chunks:
        if c.get("choices") and c.get("usage"):
            i = c["choices"][0]["index"]
            out[i] = max(out.get(i, 0), c["usage"]["completion_tokens"])
    return out


async def test_n_choices_stream_lost_mid_way_bills_every_choice(make_stack, certs):
    # Seed 3 drops this stream mid-way, with all three choices well under way.
    stack = await _mock_stack(make_stack, certs, abort_rate=1.0, seed=3)
    t = await stack.tenant()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={
                "model": stack.small,
                "messages": STORY,
                "n": 3,
                "max_tokens": 300,
                "ignore_eos": True,
                "stream": True,
                "stream_options": CONTINUOUS,
            },
        )
    events = [json.loads(e[6:]) for e in r.text.split("\n\n") if e.startswith("data: {")]
    assert events[-1]["error"]["code"] == "upstream_lost"
    seen = _per_choice_counts(events[:-1])
    assert sorted(seen) == [0, 1, 2] and min(seen.values()) > 5
    prompt = events[0]["usage"]["prompt_tokens"]

    [event] = await stack.wait_events(1)
    assert event.status == "aborted"
    # The prompt once, plus every choice's tokens: not just the last chunk's choice.
    assert (event.prompt_tokens, event.completion_tokens) == (prompt, sum(seen.values()))
    assert event.cost_micros == cost_micros(QWEN, prompt, sum(seen.values()), 0)
    assert await stack.reserved(t) == (0, 0)


async def test_n_choices_client_abort_bills_at_least_what_was_received(make_stack, certs):
    stack = await _mock_stack(make_stack, certs, abort_rate=0.0)
    t = await stack.tenant()
    received: list[dict] = []
    async with (
        httpx.AsyncClient(timeout=30) as c,
        c.stream(
            "POST",
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={
                "model": stack.small,
                "messages": STORY,
                "n": 4,
                "max_tokens": 2000,
                "ignore_eos": True,
                "stream": True,
                "stream_options": CONTINUOUS,
            },
        ) as r,
    ):
        async for line in r.aiter_lines():
            if line.startswith("data: {"):
                received.append(json.loads(line[6:]))
            if len(received) >= 120:
                break  # leaving the block closes the connection mid-stream
    seen = _per_choice_counts(received)
    assert sorted(seen) == [0, 1, 2, 3]
    [event] = await stack.wait_events(1)
    assert event.status == "aborted"
    # The gateway may have forwarded a little more than the client read, never less.
    assert event.completion_tokens >= sum(seen.values())
    assert event.completion_tokens < 4 * 2000


async def test_n_choices_complete_stream_and_non_stream(make_stack, certs):
    stack = await _mock_stack(make_stack, certs, abort_rate=0.0)
    t = await stack.tenant()
    headers = {"Authorization": f"Bearer {t.key}"}
    body = {"model": stack.small, "messages": STORY, "n": 3, "max_tokens": 200}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{stack.url}/v1/chat/completions", headers=headers, json=body)
        assert r.status_code == 200
        data = r.json()
        s = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers=headers,
            json={**body, "stream": True, "stream_options": {"include_usage": True}},
        )
    assert len(data["choices"]) == 3
    usage = data["usage"]
    final = [json.loads(e[6:]) for e in s.text.split("\n\n") if e.startswith("data: {")][-1]
    assert final["choices"] == [] and final["usage"]["completion_tokens"] > 0
    events = await stack.wait_events(2)
    non_stream, stream = sorted(events, key=lambda e: e.created_at)
    assert (non_stream.prompt_tokens, non_stream.completion_tokens) == (
        usage["prompt_tokens"],
        usage["completion_tokens"],
    )
    assert int(r.headers["X-Loom-Cost-Micros"]) == non_stream.cost_micros
    assert stream.status == "ok"
    assert stream.completion_tokens == final["usage"]["completion_tokens"]
    assert stream.cost_micros == final["usage"]["cost_micros"]
