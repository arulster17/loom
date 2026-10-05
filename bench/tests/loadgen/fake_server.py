"""Fake OpenAI-compatible streaming server built on httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "http://fake/v1"


def sse(obj: Any) -> bytes:
    return f"data: {json.dumps(obj)}\n\n".encode()


def chat_chunk(content: str | None = None, finish_reason: str | None = None, **delta) -> bytes:
    if content is not None:
        delta["content"] = content
    return sse({"choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})


def usage_chunk(prompt: int, completion: int, cached: int | None = None) -> bytes:
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return sse({"choices": [], "usage": usage})


DONE = b"data: [DONE]\n\n"


@dataclass
class FakeServer:
    """Streams `tokens` chat chunks after `ttft_s`, `itl_s` apart; tracks concurrency."""

    ttft_s: float = 0.0
    itl_s: float = 0.0
    tokens: int = 4
    stall: bool = False
    inflight: int = 0
    max_inflight: int = 0
    bodies: list[dict[str, Any]] = field(default_factory=list)
    arrivals: list[float] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.arrivals.append(asyncio.get_running_loop().time())
        return httpx.Response(200, content=self._stream())

    async def _stream(self) -> AsyncIterator[bytes]:
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.ttft_s)
            if self.stall:
                await asyncio.Event().wait()
            for i in range(self.tokens):
                if i:
                    await asyncio.sleep(self.itl_s)
                last = i == self.tokens - 1
                yield chat_chunk("tok", "length" if last else None)
            yield usage_chunk(10, self.tokens)
            yield DONE
        finally:
            self.inflight -= 1


def scripted(chunks: list[bytes | float], status: int = 200) -> Callable[[httpx.Request], Any]:
    """Handler that yields `chunks` in order; a float means sleep that many seconds."""

    async def body() -> AsyncIterator[bytes]:
        for c in chunks:
            if isinstance(c, float):
                await asyncio.sleep(c)
            else:
                yield c

    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body())

    return handle
