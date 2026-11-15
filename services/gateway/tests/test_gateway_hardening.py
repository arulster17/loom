"""Request hardening: the engine parameter allowlist, unknown keys refused before the
body is read, the unknown-key cache, and `api_keys.last_used_at`."""

from __future__ import annotations

import asyncio
import time

import httpx
import psycopg
import pytest

import loom_gateway.authn as authn
from loom_platform.apikeys import generate_api_key

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

MSG = [{"role": "user", "content": "hi"}]


async def _post(stack, key: str, body: dict, path: str = "/v1/chat/completions"):
    async with httpx.AsyncClient(timeout=30) as c:
        return await c.post(
            f"{stack.url}{path}", headers={"Authorization": f"Bearer {key}"}, json=body
        )


@pytest.mark.parametrize(
    ("extra", "param"),
    [
        ({"chat_template": "{{ 7*7 }}"}, "chat_template"),
        ({"guided_json": {"type": "object"}}, "guided_json"),
        ({"logits_processors": ["x.y"]}, "logits_processors"),
        ({"structured_outputs": {"regex": ".*"}}, "structured_outputs"),
        ({"documents": [{"text": "x"}]}, "documents"),
        ({"mm_processor_kwargs": {}}, "mm_processor_kwargs"),
        ({"chat_template_kwargs": {"tools_in_user_message": True}},
         "chat_template_kwargs.tools_in_user_message"),
        ({"chat_template_kwargs": {"enable_thinking": "yes"}},
         "chat_template_kwargs.enable_thinking"),
    ],
)  # fmt: skip
async def test_engine_extras_outside_the_allowlist_are_refused(stack, extra, param):
    t = await stack.tenant()
    r = await _post(stack, t.key, {"model": stack.small, "messages": MSG, **extra})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == param
    assert err["code"] in ("unsupported_parameter", "invalid_request")
    assert stack.engine.requests == []  # never reached the engine
    assert await stack.events() == []  # refused before reserve: nothing billed


async def test_allowlisted_parameters_pass_through(stack):
    t = await stack.tenant()
    body = {
        "model": stack.small,
        "messages": MSG,
        "max_tokens": 5,
        "temperature": 0.2,
        "top_p": 0.9,
        "seed": 7,
        "stop": ["\n"],
        "logprobs": True,
        "top_logprobs": 2,
        "user": "u1",
        "response_format": {"type": "json_object"},
        "top_k": 20,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = await _post(stack, t.key, body)
    assert r.status_code == 200, r.text
    sent = stack.engine.requests[-1]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["top_k"] == 20 and sent["ignore_eos"] is True
    r = await _post(
        stack,
        t.key,
        {"model": stack.small, "prompt": "hi", "chat_template_kwargs": {}},
        "/v1/completions",
    )
    assert r.status_code == 400 and r.json()["error"]["param"] == "chat_template_kwargs"


async def test_unknown_key_is_refused_before_the_body_is_read(stack):
    """A 1 MB body announced but never sent: the 401 comes back at once."""
    host, port = stack.url.removeprefix("http://").split(":")
    reader, writer = await asyncio.open_connection(host, int(port))
    unknown = generate_api_key().plaintext
    writer.write(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        b"Authorization: Bearer " + unknown.encode() + b"\r\n"
        b"Content-Length: 1000000\r\n\r\n" + b'{"model":'
    )
    await writer.drain()
    start = time.monotonic()
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
    assert head.startswith(b"HTTP/1.1 401")
    assert time.monotonic() - start < 5
    writer.close()


async def test_unknown_key_is_looked_up_once_per_cache_ttl(stack, monkeypatch):
    calls = 0
    real = authn.load_auth_record

    async def counting(conn, key_hash):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return await real(conn, key_hash)

    monkeypatch.setattr(authn, "load_auth_record", counting)
    unknown = generate_api_key().plaintext
    for _ in range(20):
        r = await _post(stack, unknown, {"model": stack.small, "messages": MSG})
        assert r.status_code == 401
    assert calls == 1


async def test_last_used_at_is_written_at_most_once_a_minute(stack):
    t = await stack.tenant()

    async def last_used():  # type: ignore[no-untyped-def]
        async with await psycopg.AsyncConnection.connect(stack.pg_url) as conn:
            cur = await conn.execute("SELECT last_used_at FROM api_keys WHERE id = %s", (t.key_id,))
            return (await cur.fetchone())[0]

    assert await last_used() is None
    assert (await _post(stack, t.key, {"model": stack.small, "messages": MSG})).status_code == 200
    first = None
    for _ in range(100):
        first = await last_used()
        if first is not None:
            break
        await asyncio.sleep(0.02)
    assert first is not None
    for _ in range(3):
        assert (
            await _post(stack, t.key, {"model": stack.small, "messages": MSG})
        ).status_code == 200
    await asyncio.sleep(0.2)
    assert await last_used() == first  # throttled
