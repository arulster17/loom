"""Helpers for the before-M3 review-fix tests (docs/m3-review-fixes-spec.md, "Gateway"):
engines the shared `FakeEngine` cannot play (one that records when each request reached
it and then stalls, one that floods a stream with large chunks), routing a model to them,
and small polling and metric helpers."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from gateway_testkit import replica, serve, set_ready
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from loom_platform.limiter import api_key_key, org_key

MESSAGES = [{"role": "user", "content": "Count."}]


@dataclass
class SlowEngine:
    """A non-streaming engine that answers only after `delay_s` (long enough that the
    gateway's request timeout cuts it). `arrivals` holds `time.perf_counter()` at the
    moment each request reached the handler: the gateway dispatched it no later."""

    delay_s: float = 60.0
    arrivals: list[float] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/chat/completions", self._handle, methods=["POST"]),
                Route("/v1/completions", self._handle, methods=["POST"]),
                Route("/health", lambda r: Response(), methods=["GET"]),
            ]
        )

    async def _handle(self, request: Request) -> Response:
        self.arrivals.append(time.perf_counter())
        body = json.loads(await request.body())
        self.bodies.append(body)
        await asyncio.sleep(self.delay_s)
        chat = request.url.path.endswith("/chat/completions")
        choice: dict[str, Any] = {"index": 0, "logprobs": None, "finish_reason": "stop"}
        if chat:
            choice["message"] = {"role": "assistant", "content": "late"}
        else:
            choice["text"] = "late"
        return JSONResponse(
            {
                "id": f"cmpl-{uuid.uuid4().hex}",
                "object": "chat.completion" if chat else "text_completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [choice],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        )


@dataclass
class FloodEngine:
    """A streaming chat engine that sends large content chunks as fast as the gateway
    takes them (each with vLLM's continuous usage: prompt 11, completion = chunk number),
    then hangs. Behind a client that stopped reading, the gateway's sends block once the
    socket buffers are full. `sent` counts content chunks handed to the server."""

    chunk_bytes: int = 32 * 1024
    max_chunks: int = 5000
    prompt_tokens: int = 11
    sent: int = 0

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
                Route("/health", lambda r: Response(), methods=["GET"]),
            ]
        )

    async def _chat(self, request: Request) -> Response:
        body = json.loads(await request.body())
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        cap = int(body.get("max_tokens") or 16)
        text = "x" * self.chunk_bytes

        def sse(delta: dict[str, Any], completion: int) -> bytes:
            obj = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": None}],
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": completion,
                    "total_tokens": self.prompt_tokens + completion,
                },
            }
            return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"

        async def events() -> AsyncIterator[bytes]:
            yield sse({"role": "assistant", "content": ""}, 0)
            for i in range(1, min(cap, self.max_chunks) + 1):
                self.sent = i
                yield sse({"content": text}, i)
                await asyncio.sleep(0)
            await asyncio.sleep(3600)  # never finishes: the deadline must cut it

        return StreamingResponse(events(), media_type="text/event-stream")


async def route_to(stack: Any, app: Starlette, model: str, replica_id: str) -> str:
    """Serve `app` over plain HTTP for the stack's lifetime and make it `model`'s only
    ready replica (the stack must be built with `tls=False`). Returns the replica id."""
    port = await stack.track_server(serve(app))
    await set_ready(stack.redis, model, [replica(replica_id, port, tls=False)])
    return replica_id


async def wait_until(
    check: Callable[[], Awaitable[bool]], timeout_s: float, interval_s: float = 0.02
) -> bool:
    """Poll `check` until it is true or `timeout_s` has passed; returns the last result."""
    deadline = time.perf_counter() + timeout_s
    while True:
        if await check():
            return True
        if time.perf_counter() >= deadline:
            return False
        await asyncio.sleep(interval_s)


async def held(stack: Any, tenant: Any) -> tuple[int, int, int]:
    """(org rsum, org reservations in flight, key in-flight count) in Redis."""
    rsum, n = await stack.reserved(tenant)
    inflight = await stack.redis.get(api_key_key(tenant.org_id, tenant.key_id, "inflight"))
    return rsum, n, int(inflight or 0)


def done_marker(org_id: uuid.UUID, request_id: uuid.UUID | str) -> str:
    """The `:done:<rid>` key that marks a request id as finished (contracts §6)."""
    return org_key(org_id, f"done:{request_id}")


def sample(stack: Any, name: str, labels: dict[str, str]) -> float:
    """A metric sample from the gateway's registry; 0 when absent."""
    v = stack.gateway.metrics.registry.get_sample_value(name, labels)
    return float(v or 0.0)
