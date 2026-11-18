"""G4 (R3): a non-streaming request cut at the timeout is billed an estimate
(docs/m3-review-fixes-spec.md, "Gateway").

The estimate depends on `t`, the seconds from the first dispatch to the cut, which the
gateway measures. The tests bound it from times observed in this process (the gateway,
the engine and the client share `time.perf_counter()`):

- lower: the cut comes no earlier than receipt + timeout, receipt is after the client's
  send, and the dispatch is no later than the engine's arrival time;
- upper: the cut is no later than the client's response, and the dispatch is no earlier
  than the client's send.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

import httpx
import pytest
from gateway_m3_kit import MESSAGES, SlowEngine, held, route_to, sample

from loom_platform.pricing import cost_micros

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TIMEOUT_S = 2
DEFAULT_R = 10  # LOOM_TIMEOUT_BILL_TOKENS_PER_S default
EPS_S = 0.005  # timer granularity


async def _post_bytes(stack: Any, key: str, path: str, body: bytes) -> tuple[httpx.Response, float]:
    """POST exact body bytes; returns the response and the send time."""
    async with httpx.AsyncClient(timeout=30) as c:
        t_send = time.perf_counter()
        r = await c.post(
            f"{stack.url}{path}",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            content=body,
        )
    return r, t_send


def _t_bounds(t_send: float, t_resp: float, arrival: float) -> tuple[float, float]:
    lo = t_send + TIMEOUT_S - arrival - EPS_S
    hi = t_resp - t_send
    assert 0 < lo <= hi
    return lo, hi


def _completion_bounds(
    sequences: int, max_tokens: int, rate: float, lo: float, hi: float
) -> tuple[int, int]:
    return (
        sequences * min(max_tokens, math.floor(lo * rate)),
        sequences * min(max_tokens, math.floor(hi * rate)),
    )


def _usd(micros: int) -> str:
    return f"{micros // 1_000_000}.{micros % 1_000_000:06d}"


async def _slow_stack(make_stack: Any) -> tuple[Any, SlowEngine]:
    stack = await make_stack(tls=False, request_timeout_s=TIMEOUT_S)
    engine = SlowEngine()
    await route_to(stack, engine.app(), stack.small, "slow")
    return stack, engine


def _check_502(r: httpx.Response, cost: int) -> None:
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "upstream_lost"
    assert r.headers["x-loom-cost-micros"] == str(cost)
    assert r.headers["x-loom-cost-usd"] == _usd(cost)


async def test_g4_chat_cut_at_the_timeout_is_billed_the_estimate(make_stack, caplog):
    """completion = sequences x min(max_tokens, floor(t x R)) with R = 10 by default;
    prompt = prompts x min(prompt_upper, ceil(body bytes / 4)); cached = 0. The client
    gets 502 upstream_lost with the cost headers; the event counts in
    usage_estimated and not in usage_missing; the request log line says estimated."""
    stack, engine = await _slow_stack(make_stack)
    t = await stack.tenant()
    m = stack.small
    price = stack.gateway.lookup_model(m)[1]
    body = json.dumps(
        {"model": m, "messages": MESSAGES, "max_tokens": 1000, "n": 2},
        separators=(",", ":"),
    ).encode()
    estimated0 = sample(stack, "loom_gateway_usage_estimated_total", {"model": m})
    missing0 = sample(stack, "loom_gateway_usage_missing_total", {"model": m})

    with caplog.at_level(logging.INFO, logger="loom_gateway"):
        r, t_send = await _post_bytes(stack, t.key, "/v1/chat/completions", body)
        t_resp = time.perf_counter()
        [event] = await stack.wait_events(1)

    assert len(engine.arrivals) == 1
    lo, hi = _t_bounds(t_send, t_resp, engine.arrivals[0])
    c_lo, c_hi = _completion_bounds(2, 1000, DEFAULT_R, lo, hi)
    assert event.status == "aborted"
    assert c_lo <= event.completion_tokens <= c_hi, (event.completion_tokens, c_lo, c_hi)
    assert event.completion_tokens > 0
    assert event.prompt_tokens == math.ceil(len(body) / 4)
    assert event.cached_prompt_tokens == 0
    assert event.cost_micros == cost_micros(price, event.prompt_tokens, event.completion_tokens, 0)
    assert 0 < event.cost_micros <= int(r.headers["x-loom-max-cost-micros"])
    _check_502(r, event.cost_micros)

    assert await held(stack, t) == (0, 0, 0)
    assert await stack.usage(t) == event.cost_micros
    assert sample(stack, "loom_gateway_usage_estimated_total", {"model": m}) == estimated0 + 1
    assert sample(stack, "loom_gateway_usage_missing_total", {"model": m}) == missing0

    request_lines = [
        getattr(rec, "fields", {})
        for rec in caplog.records
        if rec.name.startswith("loom_gateway") and rec.getMessage() == "request"
    ]
    mine = [f for f in request_lines if f.get("request_id") == str(event.id)]
    assert mine, "no request log line for the cut request"
    assert mine[-1].get("estimated") is True


async def test_g4_completions_estimate_counts_prompts_sequences_and_caps_at_max_tokens(
    make_stack,
):
    """Two prompts, n = 2 (4 sequences), max_tokens = 3: floor(t x R) is well above 3, so
    completion = 4 x 3 exactly; prompt = 2 x ceil(body bytes / 4)."""
    stack, _ = await _slow_stack(make_stack)
    t = await stack.tenant()
    m = stack.small
    price = stack.gateway.lookup_model(m)[1]
    body = json.dumps(
        {"model": m, "prompt": ["Count to ten.", "Name a colour."], "n": 2, "max_tokens": 3},
        separators=(",", ":"),
    ).encode()

    r, _ = await _post_bytes(stack, t.key, "/v1/completions", body)
    [event] = await stack.wait_events(1)

    assert event.status == "aborted"
    assert event.completion_tokens == 4 * 3
    assert event.prompt_tokens == 2 * math.ceil(len(body) / 4)
    assert event.cached_prompt_tokens == 0
    assert event.cost_micros == cost_micros(price, event.prompt_tokens, 12, 0)
    assert event.cost_micros <= int(r.headers["x-loom-max-cost-micros"])
    _check_502(r, event.cost_micros)
    assert await held(stack, t) == (0, 0, 0)


async def test_g4_rate_setting_scales_the_estimate(make_stack, monkeypatch):
    """R comes from LOOM_TIMEOUT_BILL_TOKENS_PER_S."""
    rate = 40
    monkeypatch.setenv("LOOM_TIMEOUT_BILL_TOKENS_PER_S", str(rate))
    stack, engine = await _slow_stack(make_stack)
    t = await stack.tenant()
    body = json.dumps(
        {"model": stack.small, "messages": MESSAGES, "max_tokens": 1000},
        separators=(",", ":"),
    ).encode()

    r, t_send = await _post_bytes(stack, t.key, "/v1/chat/completions", body)
    t_resp = time.perf_counter()
    [event] = await stack.wait_events(1)

    lo, hi = _t_bounds(t_send, t_resp, engine.arrivals[0])
    c_lo, c_hi = _completion_bounds(1, 1000, rate, lo, hi)
    assert event.status == "aborted"
    assert c_lo <= event.completion_tokens <= c_hi, (event.completion_tokens, c_lo, c_hi)
    _check_502(r, event.cost_micros)


async def test_g4_rate_zero_bills_nothing(make_stack, monkeypatch):
    """LOOM_TIMEOUT_BILL_TOKENS_PER_S=0 turns estimating off: today's zero bill."""
    monkeypatch.setenv("LOOM_TIMEOUT_BILL_TOKENS_PER_S", "0")
    stack, engine = await _slow_stack(make_stack)
    t = await stack.tenant()
    body = json.dumps(
        {"model": stack.small, "messages": MESSAGES, "max_tokens": 1000},
        separators=(",", ":"),
    ).encode()

    r, _ = await _post_bytes(stack, t.key, "/v1/chat/completions", body)
    [event] = await stack.wait_events(1)

    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_lost"
    assert engine.arrivals, "the request was never dispatched (test setup)"
    assert event.status == "aborted"
    assert (event.prompt_tokens, event.completion_tokens, event.cost_micros) == (0, 0, 0)
    assert await stack.usage(t) == 0
    assert await held(stack, t) == (0, 0, 0)


async def test_g4_cut_before_any_dispatch_bills_nothing(make_stack):
    """A request whose deadline passes before any replica was dispatched (here: the body
    arrives slower than LOOM_REQUEST_TIMEOUT_S, so the deadline is gone before dispatch)
    is billed nothing: no estimate. (Today the gateway may still open the upstream call
    for the last millisecond and settle `aborted`; the spec says `error`. Either status
    is accepted here; the bill must be zero.)"""
    stack, _ = await _slow_stack(make_stack)
    t = await stack.tenant()
    body = json.dumps(
        {"model": stack.small, "messages": MESSAGES, "max_tokens": 1000},
        separators=(",", ":"),
    ).encode()
    half = len(body) // 2

    async def trickle() -> Any:
        yield body[:half]
        await asyncio.sleep(TIMEOUT_S + 0.5)
        yield body[half:]

    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}", "Content-Type": "application/json"},
            content=trickle(),
        )
    [event] = await stack.wait_events(1)

    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "upstream_lost"
    assert event.status in ("error", "aborted")
    assert (event.prompt_tokens, event.completion_tokens, event.cost_micros) == (0, 0, 0)
    assert await stack.usage(t) == 0
    assert await held(stack, t) == (0, 0, 0)
