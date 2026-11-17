"""Helpers for `test_gateway_m3_review_gaps.py` (docs/m3-review-fixes-spec.md, G2 and G4):
a reserve whose reply is lost, a `Limiter.release` that never succeeds and records when it
was called, exact-bytes posting and the gateway's pending asyncio tasks."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from loom_platform.limiter import Limiter


def lose_admitted_reply(monkeypatch: pytest.MonkeyPatch, limiter: Limiter) -> list[Any]:
    """The first admitted reserve takes effect in Redis, but its reply is lost: the call
    raises a socket timeout. Returns the request ids whose reply was lost."""
    real = limiter.reserve
    lost: list[Any] = []

    async def reserve(**kw: Any) -> Any:
        result = await real(**kw)
        if result.admitted and not lost:
            lost.append(kw["request_id"])
            raise RedisTimeoutError("Timeout reading from socket")
        return result

    monkeypatch.setattr(limiter, "reserve", reserve)
    return lost


def always_failing_release(
    monkeypatch: pytest.MonkeyPatch, limiter: Limiter
) -> list[tuple[float, Any]]:
    """`Limiter.release` always raises a connection reset (Redis never comes back).
    Returns `(time.perf_counter(), request_id)` for every call."""
    calls: list[tuple[float, Any]] = []

    async def release(org_id: Any, request_id: Any, *args: Any, **kw: Any) -> Any:
        calls.append((time.perf_counter(), request_id))
        raise RedisConnectionError("Connection reset by peer")

    monkeypatch.setattr(limiter, "release", release)
    return calls


async def post_bytes(
    stack: Any, key: str, path: str, body: bytes
) -> tuple[httpx.Response, float, float]:
    """POST exact body bytes; returns the response, the send time and the response time."""
    async with httpx.AsyncClient(timeout=30) as c:
        t_send = time.perf_counter()
        r = await c.post(
            f"{stack.url}{path}",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            content=body,
        )
        t_resp = time.perf_counter()
    return r, t_send, t_resp


def pending_tasks() -> set[asyncio.Task[Any]]:
    """Every unfinished task in this event loop (the gateway runs in it), except the
    caller's own."""
    me = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not me and not t.done()}


def usd(micros: int) -> str:
    return f"{micros // 1_000_000}.{micros % 1_000_000:06d}"
