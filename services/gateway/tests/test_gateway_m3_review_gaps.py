"""Regression tests for gaps a review found in the before-M3 gateway tests
(docs/m3-review-fixes-spec.md, "Gateway", G2 and G4).

G4: the estimate's clamp to the reservation, rate 0 as exactly today's behaviour, and a
cut before dispatch settling `error`. G2: the `:done:<rid>` marker's TTL after a
two-argument `Limiter.release`, and the background release giving up after the grace.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from typing import Any

import httpx
import pytest
from gateway_m3_gaps_kit import (
    always_failing_release,
    lose_admitted_reply,
    pending_tasks,
    post_bytes,
    usd,
)
from gateway_m3_kit import MESSAGES, SlowEngine, done_marker, held, route_to, sample, wait_until

from loom_platform.limiter import NO_LIMITS
from loom_platform.pricing import cost_micros, prompt_tokens_upper_bound

TIMEOUT_S = 2
EPS_S = 0.005  # timer granularity

# ------------------------------------------------------------------ G4


async def _slow_stack(make_stack: Any) -> tuple[Any, SlowEngine]:
    stack = await make_stack(tls=False, request_timeout_s=TIMEOUT_S)
    engine = SlowEngine()
    await route_to(stack, engine.app(), stack.small, "slow")
    return stack, engine


@pytest.mark.postgres
@pytest.mark.redis
async def test_g4_estimate_over_the_reservation_is_clamped_to_it(make_stack, monkeypatch):
    """G4: the cost is clamped to the reservation (`max_cost_micros`). A chat body above
    4 x the model's context (under the body limit) with a high rate and max_tokens = 1000
    makes the unclamped estimate exceed the reservation; the event and the 502's cost
    headers must name exactly the reservation."""
    rate = 1000
    monkeypatch.setenv("LOOM_TIMEOUT_BILL_TOKENS_PER_S", str(rate))
    stack, engine = await _slow_stack(make_stack)
    t = await stack.tenant()
    m = stack.small
    spec, price = stack.gateway.lookup_model(m)
    ctx = spec.max_context

    filler = "x" * 200_000
    body = json.dumps(
        {"model": m, "messages": [{"role": "user", "content": filler}], "max_tokens": 1000},
        separators=(",", ":"),
    ).encode()
    assert len(body) > 4 * ctx, (len(body), ctx)
    assert len(body) < stack.gateway.settings.max_body_bytes

    r, t_send, _ = await post_bytes(stack, t.key, "/v1/chat/completions", body)
    [event] = await stack.wait_events(1)

    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "upstream_lost"
    reservation = int(r.headers["x-loom-max-cost-micros"])

    # The estimate the formula gives before the clamp, at the shortest t the gateway can
    # have measured: it must exceed the reservation, or this test checks nothing.
    assert len(engine.arrivals) == 1, "the request was never dispatched (test setup)"
    t_lo = t_send + TIMEOUT_S - engine.arrivals[0] - EPS_S
    assert t_lo > 0
    prompt_upper = prompt_tokens_upper_bound(len(body), ctx)
    unclamped = cost_micros(
        price,
        min(prompt_upper, math.ceil(len(body) / 4)),
        min(1000, math.floor(t_lo * rate)),
        0,
    )
    assert unclamped > reservation, (unclamped, reservation)

    assert event.status == "aborted"
    assert event.cost_micros == reservation, (event.cost_micros, reservation, unclamped)
    assert r.headers["x-loom-cost-micros"] == str(reservation)
    assert r.headers["x-loom-cost-usd"] == usd(reservation)
    assert await stack.usage(t) == reservation
    assert await held(stack, t) == (0, 0, 0)


@pytest.mark.postgres
@pytest.mark.redis
async def test_g4_rate_zero_is_exactly_todays_behaviour(make_stack, monkeypatch):
    """G4 (guard: passes before the fix too): with LOOM_TIMEOUT_BILL_TOKENS_PER_S=0 a cut
    settles `aborted` at zero cost, counts in usage_missing and not in usage_estimated,
    and the 502 carries no cost headers."""
    monkeypatch.setenv("LOOM_TIMEOUT_BILL_TOKENS_PER_S", "0")
    stack, engine = await _slow_stack(make_stack)
    t = await stack.tenant()
    m = stack.small
    body = json.dumps(
        {"model": m, "messages": MESSAGES, "max_tokens": 1000}, separators=(",", ":")
    ).encode()
    missing0 = sample(stack, "loom_gateway_usage_missing_total", {"model": m})
    estimated0 = sample(stack, "loom_gateway_usage_estimated_total", {"model": m})

    r, _, _ = await post_bytes(stack, t.key, "/v1/chat/completions", body)
    [event] = await stack.wait_events(1)

    assert engine.arrivals, "the request was never dispatched (test setup)"
    assert r.status_code == 502, r.text
    assert r.json()["error"]["code"] == "upstream_lost"
    assert "x-loom-cost-micros" not in r.headers
    assert "x-loom-cost-usd" not in r.headers
    assert event.status == "aborted"
    assert (event.prompt_tokens, event.completion_tokens, event.cost_micros) == (0, 0, 0)
    assert sample(stack, "loom_gateway_usage_missing_total", {"model": m}) == missing0 + 1
    assert sample(stack, "loom_gateway_usage_estimated_total", {"model": m}) == estimated0
    assert await stack.usage(t) == 0
    assert await held(stack, t) == (0, 0, 0)


@pytest.mark.postgres
@pytest.mark.redis
async def test_g4_cut_before_any_dispatch_settles_error(make_stack):
    """G4: a request whose deadline passed before it was dispatched bills nothing and
    settles `error`, even if the code gives the attempt a last millisecond. The body
    arrives slower than LOOM_REQUEST_TIMEOUT_S, so the deadline is gone before dispatch."""
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
    assert event.status == "error", event.status
    assert (event.prompt_tokens, event.completion_tokens, event.cost_micros) == (0, 0, 0)
    assert await stack.usage(t) == 0
    assert await held(stack, t) == (0, 0, 0)


# ------------------------------------------------------------------ G2

TTL_SLACK_MS = 5_000  # time between the release and the PTTL read, under CPU load


@pytest.mark.postgres
@pytest.mark.redis
async def test_g2_release_marks_done_for_at_least_the_reservation_ttl(make_stack):
    """G2: `Limiter.release(org_id, request_id)` (two arguments) marks the request id done
    for at least the reservation TTL (LOOM_REQUEST_TIMEOUT_S + LOOM_RESERVATION_GRACE_S),
    both when the release finds the reservation and when it runs before the reserve."""
    stack = await make_stack(request_timeout_s=700, reservation_grace_s=90)
    settings = stack.gateway.settings
    ttl_ms = (settings.request_timeout_s + settings.reservation_grace_s) * 1000
    assert ttl_ms == 790_000
    limiter = stack.gateway.limiter
    t = await stack.tenant()
    await limiter.init_org(
        org_id=t.org_id,
        key_id=t.key_id,
        credit_total=5_000_000,
        usage_total=0,
        key_usage_total=0,
    )

    # The release runs before the (delayed) reserve: no reservation to find.
    early = uuid.uuid4()
    await limiter.release(t.org_id, early)
    # The release after an admitted reserve (the 503 path).
    late = uuid.uuid4()
    res = await limiter.reserve(
        org_id=t.org_id,
        key_id=t.key_id,
        request_id=late,
        max_cost_micros=1_000,
        tpm_estimate=100,
        limits=NO_LIMITS,
        ttl_ms=ttl_ms,
    )
    assert res.admitted, res
    await limiter.release(t.org_id, late)

    for rid in (early, late):
        pttl = await stack.redis.pttl(done_marker(t.org_id, rid))
        assert pttl != -2, f"no :done: marker for {rid}"
        assert pttl != -1, "the :done: marker must expire"
        assert pttl >= ttl_ms - TTL_SLACK_MS, (pttl, ttl_ms)


GRACE_S = 4  # reservation_grace_s for the give-up test
GIVE_UP_MARGIN_S = 5.0  # max retry spacing (2 s) plus slack for CPU load
QUIET_S = 3.0  # longer than the max retry spacing: a live loop would call again


@pytest.mark.postgres
@pytest.mark.redis
async def test_g2_background_release_gives_up_after_the_grace(make_stack, monkeypatch):
    """G2: when Redis never answers, the background release of a lost reserve is retried
    until LOOM_RESERVATION_GRACE_S has passed, then given up: no more calls after the
    grace plus a margin, and no task left running. The 503 is answered at once."""
    stack = await make_stack(reservation_grace_s=GRACE_S)
    t = await stack.tenant()
    limiter = stack.gateway.limiter
    lost = lose_admitted_reply(monkeypatch, limiter)
    calls = always_failing_release(monkeypatch, limiter)
    body = json.dumps(
        {"model": stack.small, "messages": MESSAGES, "max_tokens": 5}, separators=(",", ":")
    ).encode()

    before = pending_tasks()
    r, t_send, t_resp = await post_bytes(stack, t.key, "/v1/chat/completions", body)

    assert lost, "the reserve never took effect (test setup)"
    assert r.status_code == 503, r.text
    assert r.json()["error"]["code"] == "service_unavailable"
    assert r.headers["retry-after"] == "1"
    assert t_resp - t_send < GRACE_S - 1, "the 503 waited for the background release"

    give_up_by = t_resp + GRACE_S + GIVE_UP_MARGIN_S
    await asyncio.sleep(max(0.0, give_up_by - time.perf_counter()))
    n = len(calls)
    assert n >= 2, f"the release was not retried ({n} calls)"
    assert {rid for _, rid in calls} == {lost[0]}
    last = calls[-1][0]
    assert last >= t_send + GRACE_S - 2.5, (
        f"gave up {last - t_send:.2f} s after the request, well before the {GRACE_S} s grace"
    )

    await asyncio.sleep(QUIET_S)
    assert len(calls) == n, (
        f"still retrying {calls[-1][0] - t_send:.2f} s after the request "
        f"(grace {GRACE_S} s): {len(calls) - n} more calls"
    )

    async def nothing_left() -> bool:
        return not (pending_tasks() - before)

    assert await wait_until(nothing_left, 3.0), (
        f"tasks still running after the give-up: {pending_tasks() - before}"
    )
    assert await stack.events() == []
