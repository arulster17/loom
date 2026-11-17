"""G5 (R4): a settled non-streaming request is never answered 502
(docs/m3-review-fixes-spec.md, "Gateway")."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import pytest
from gateway_m3_kit import MESSAGES, held

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TIMEOUT_S = 2


def _slow_settle(monkeypatch: pytest.MonkeyPatch, limiter: Any, delay_s: float) -> list[float]:
    """`Limiter.settle` takes `delay_s` longer (Redis slow) before doing its work.
    Returns the perf_counter times at which each settle finished."""
    real = limiter.settle
    finished: list[float] = []

    async def settle(event: Any) -> Any:
        await asyncio.sleep(delay_s)
        result = await real(event)
        finished.append(time.perf_counter())
        return result

    monkeypatch.setattr(limiter, "settle", settle)
    return finished


async def test_g5_settle_finishing_after_the_deadline_still_answers_200(make_stack, monkeypatch):
    """Done when: a settle that finishes after the deadline yields 200 with the body and
    one `ok` usage event, and no 502."""
    stack = await make_stack(request_timeout_s=TIMEOUT_S)
    t = await stack.tenant()
    # The engine's 200 body arrives well inside the deadline; settle ends well past it.
    stack.engine.behavior.first_token_delay_s = 0.5
    stack.engine.behavior.tokens = 5
    finished = _slow_settle(monkeypatch, stack.gateway.limiter, delay_s=TIMEOUT_S + 0.5)

    async with httpx.AsyncClient(timeout=30) as c:
        t_send = time.perf_counter()
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json={"model": stack.small, "messages": MESSAGES, "max_tokens": 5},
        )

    assert r.status_code == 200, r.text
    assert finished, "the 200 came before its settle finished"
    assert finished[0] > t_send + TIMEOUT_S, "settle ended inside the deadline (test setup)"
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "tok1 tok2 tok3 tok4 tok5 "
    assert body["usage"]["completion_tokens"] == 5

    events = await stack.wait_events(1)
    await asyncio.sleep(0.3)
    events = await stack.events()
    assert len(events) == 1
    [event] = events
    assert event.status == "ok"
    assert event.completion_tokens == 5
    assert r.headers["x-loom-cost-micros"] == str(event.cost_micros)
    assert await stack.usage(t) == event.cost_micros
    assert await held(stack, t) == (0, 0, 0)
    assert stack.gateway.metrics.registry.get_sample_value(
        "loom_gateway_requests_total",
        {"endpoint": "chat", "model": stack.small, "status": "502"},
    ) in (None, 0.0)
