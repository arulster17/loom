"""G1 (R1): a stream past its deadline settles even if the client stopped reading
(docs/m3-review-fixes-spec.md, "Gateway")."""

from __future__ import annotations

import asyncio
import json
import socket
import time

import pytest
from gateway_m3_kit import MESSAGES, FloodEngine, held, route_to

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TIMEOUT_S = 2
# "within about 1 s of the deadline": the trailing writes are bounded at 1 s, plus some
# room for the settle itself on a loaded machine.
SLACK_S = 2.0


async def _open_stalled_stream(url: str, key: str, body: bytes) -> tuple[socket.socket, float]:
    """POST a streaming request over a raw socket with a tiny receive buffer, read the
    response head, then never read again. Returns the socket and the send time."""
    host, port = "127.0.0.1", int(url.rsplit(":", 1)[1])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    await asyncio.to_thread(sock.connect, (host, port))
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Authorization: Bearer {key}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    t_send = time.perf_counter()
    await asyncio.to_thread(sock.sendall, head + body)
    got = b""
    while b"\r\n\r\n" not in got:
        chunk = await asyncio.to_thread(sock.recv, 1024)
        assert chunk, "the gateway closed the connection before the response head"
        got += chunk
    assert got.startswith(b"HTTP/1.1 200"), got[:200]
    return sock, t_send


async def test_g1_stream_to_a_client_that_stopped_reading_settles_at_the_deadline(
    make_stack,
):
    """Done when: a client that stops reading mid-stream (its socket buffer full) gets
    its usage event, `aborted`, with the tokens it was sent, within about 1 s of the
    deadline, and the reservation and in-flight counts are released by then."""
    stack = await make_stack(tls=False, request_timeout_s=TIMEOUT_S)
    engine = FloodEngine()
    rep = await route_to(stack, engine.app(), stack.small, "flood")
    t = await stack.tenant()
    body = json.dumps(
        {
            "model": stack.small,
            "messages": MESSAGES,
            "stream": True,
            "max_tokens": 10_000,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
    ).encode()

    sock, t_send = await _open_stalled_stream(stack.url, t.key, body)
    try:
        # The gateway received the request no earlier than t_send, so its deadline is
        # no earlier than t_send + TIMEOUT_S.
        budget = t_send + TIMEOUT_S + SLACK_S - time.perf_counter()
        events = await stack.wait_events(1, timeout_s=max(budget, 0.0))
        assert events, (
            f"no usage event within {SLACK_S} s of the {TIMEOUT_S} s deadline while the "
            "client stopped reading (the trailing writes are not bounded)"
        )
        assert engine.sent > 0
        # Released by the time the event exists.
        assert await held(stack, t) == (0, 0, 0)
        assert stack.gateway.router.inflight(rep) == 0
        [event] = events
        assert event.status == "aborted"
        assert event.prompt_tokens == engine.prompt_tokens
        # Billed for the chunks forwarded before the stall: at least one, never more
        # than the engine produced.
        assert 1 <= event.completion_tokens <= engine.sent
        assert event.cost_micros > 0
        assert await stack.usage(t) == event.cost_micros
    finally:
        sock.close()
    # Exactly one event, even once the connection is gone.
    await asyncio.sleep(0.3)
    assert len(await stack.events()) == 1
