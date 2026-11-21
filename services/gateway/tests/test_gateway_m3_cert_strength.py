"""A27, the gateway's half: what the other reload tests leave open (docs/phase1-m3-spec.md
§7; contracts §2 `LOOM_GATEWAY_CLIENT_CERT`/`_KEY`, `LOOM_POD_CA_CERT`,
`LOOM_GATEWAY_CERT_CHECK_S`, §9).

- "open streams keep theirs": with two streams open on the old TLS context, a reload
  leaves both on their connections until each finishes (the first finishing does not
  close the second's).
- "checks the file ... every `LOOM_GATEWAY_CERT_CHECK_S` seconds": two successive
  changes, each followed by more than the interval with no upstream request, are each in
  use for the very next request (the check runs before an upstream request, at most this
  often, contracts §2).
- "on a change it builds a new TLS context": with no change, requests keep using the
  same pooled upstream connection (seen by the pod as one client address and port).
- G3: a client certificate whose `notAfter` is after 2049 (DER GeneralizedTime) loads,
  and `loom_gateway_client_cert_expiry_seconds` reports it.
- contracts §9: the gateway starts with its client-certificate file missing; upstream
  requests fail as connect errors (502 `upstream_lost`) until the file appears, then the
  next check loads it.

The interval is `CHECK_S` (0.3 s: a positive number, §6.13) to keep the waits short; the
client certificate is the pool manager's combined file (`loom-gateway.pem`, key and
certificate in one PEM), given as both `LOOM_GATEWAY_CLIENT_CERT` and `_KEY`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from gateway_m3_cert_kit import (
    MESSAGES,
    HeldEngine,
    Issued,
    OpenStream,
    bundle,
    cert_expiry_gauge,
    chat,
    expiry_left,
    issue,
    new_ca,
    pod,
    route,
    wait_until_ok,
    write_atomic,
)
from gateway_testkit import HIGH_LIMITS, FakeEngine, Tenant

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

CHECK_S = 0.3
# More than one interval since the last check: the next request runs one.
PAST_INTERVAL_S = 2 * CHECK_S
YEAR_DAYS = 365


@dataclass
class Setup:
    stack: Any  # conftest.Stack
    tenant: Tenant
    dir: Path
    ca1: Issued
    client_file: Path

    @property
    def model(self) -> str:
        return self.stack.small

    async def pod(
        self, *, client_cas: list[Issued], engine: FakeEngine | None = None
    ) -> tuple[int, FakeEngine]:
        """A pod presenting a ca1 server certificate, accepting clients of `client_cas`.
        At teardown, before the pod stops, the gateway's upstream client is closed: a
        pooled connection it keeps open would hold the pod's TLS server in its 5-s
        graceful shutdown (speed only; nothing is asserted then)."""
        out = await pod(
            self.stack,
            self.dir / "pods",
            server=issue(self.ca1, server=True),
            client_cas=client_cas,
            engine=engine,
        )
        assert self.stack.exit_stack is not None
        self.stack.exit_stack.push_async_callback(self.stack.gateway.client.aclose)
        return out

    async def chat(self) -> int:
        return await chat(self.stack, self.tenant.key, self.model)

    async def chat_body(self) -> tuple[int, dict[str, Any]]:
        async with httpx.AsyncClient(base_url=self.stack.url, timeout=30, trust_env=False) as c:
            r = await c.post(
                "/v1/chat/completions",
                json={"model": self.model, "messages": MESSAGES, "max_tokens": 5},
                headers={"Authorization": f"Bearer {self.tenant.key}"},
            )
        return r.status_code, r.json()


Build = Callable[..., Awaitable[Setup]]


@pytest.fixture
def build(make_stack: Any, tmp_path: Path) -> Build:
    async def make(*, client: Issued | None, ca1: Issued | None = None) -> Setup:
        """The gateway with the pod CA `ca1` and `client` in its combined file (None:
        the file does not exist yet)."""
        ca1 = ca1 or new_ca("pod-ca-1")
        gw = tmp_path / "gateway-cert"
        gw.mkdir()
        client_file = gw / "loom-gateway.pem"
        if client is not None:
            write_atomic(client_file, client.combined())
        ca_file = tmp_path / "pod-ca.pem"
        write_atomic(ca_file, bundle(ca1))
        stack = await make_stack(
            engines=(),
            ready=False,
            gateway_client_cert=str(client_file),
            gateway_client_key=str(client_file),
            pod_ca_cert=str(ca_file),
            gateway_cert_check_s=CHECK_S,
        )
        tenant = await stack.tenant(key_limits=HIGH_LIMITS, org_limits=HIGH_LIMITS)
        return Setup(stack, tenant, tmp_path, ca1, client_file)

    return make


async def _wait(event: asyncio.Event, timeout_s: float, message: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout_s)
    except TimeoutError:
        raise AssertionError(message) from None


async def open_held(s: Setup, name: str) -> tuple[OpenStream, HeldEngine]:
    """A stream on a pod of its own (accepting ca1's clients), held after its first token."""
    engine = HeldEngine(name)
    port, _ = await s.pod(client_cas=[s.ca1], engine=engine)
    await route(s.stack, s.model, port)
    stream = await OpenStream(s.stack, s.tenant.key, s.model).start()
    assert stream.status == 200, stream.status
    await _wait(engine.holding, 10, f"stream {name} never reached its engine")
    return stream, engine


def assert_complete(text: str, which: str) -> None:
    for i in range(1, 6):
        assert f"tok{i} " in text, f"token {i} missing from {which}:\n{text}"
    assert text.rstrip().endswith("data: [DONE]"), f"{which}: {text[-500:]}"
    assert '"error"' not in text, f"{which}: {text[-500:]}"


# ------------------------------------------------------------------ open streams


@pytest.mark.timeout(60)
async def test_item5_two_open_streams_both_finish_on_the_old_context_after_a_reload(
    build: Build,
) -> None:
    """§7 "open streams keep theirs": two streams open on the old context (two pods);
    the client certificate changes and the new one is in use (a pod accepting only it
    answers); the first stream is released and completes; the second, still open on the
    old context after that, then completes too, every token and `[DONE]`, no error."""
    ca1 = new_ca("pod-ca-1")
    s = await build(client=issue(ca1, server=False), ca1=ca1)
    first, first_engine = await open_held(s, "held-1")
    second, second_engine = await open_held(s, "held-2")
    try:
        ca_b = new_ca("client-ca-b")
        port, _ = await s.pod(client_cas=[ca_b])
        await route(s.stack, s.model, port)
        assert await s.chat() != 200, "the pod accepted the old client certificate"
        write_atomic(s.client_file, issue(ca_b, server=False, days=7).combined())
        await wait_until_ok(s.stack, s.tenant.key, s.model)

        assert not first.done() and not second.done(), "a held stream ended at the reload"
        first_engine.gate.set()
        assert_complete(await first.finish(), "the first stream")
        await asyncio.sleep(0.3)  # anything the first one's end set off has run
        assert not second.done(), f"the second stream ended with the first:\n{second.text()}"
        second_engine.gate.set()
        assert_complete(await second.finish(), "the second stream")
    finally:
        first_engine.gate.set()
        second_engine.gate.set()
        await first.aclose()
        await second.aclose()


# ------------------------------------------------------------------ the check interval


@pytest.mark.timeout(60)
async def test_item6_two_successive_changes_are_each_used_by_the_next_request(
    build: Build,
) -> None:
    """The client certificate changes twice (to one from CA a, then from CA b). After
    each change, with no upstream request for more than `LOOM_GATEWAY_CERT_CHECK_S`, the
    very next request reaches a pod that accepts only the new certificate."""
    ca1 = new_ca("pod-ca-1")
    s = await build(client=issue(ca1, server=False), ca1=ca1)
    ca_a, ca_b = new_ca("client-ca-a"), new_ca("client-ca-b")
    port_a, _ = await s.pod(client_cas=[ca_a])
    port_b, _ = await s.pod(client_cas=[ca_b])

    await route(s.stack, s.model, port_a)
    assert await s.chat() != 200, "pod a accepted the start's client certificate"
    for n, (ca, port) in enumerate([(ca_a, port_a), (ca_b, port_b)], 1):
        await route(s.stack, s.model, port)
        write_atomic(s.client_file, issue(ca, server=False, days=7).combined())
        await asyncio.sleep(PAST_INTERVAL_S)
        status = await s.chat()
        assert status == 200, (
            f"change {n}: the first request {PAST_INTERVAL_S} s after it got {status} "
            f"(LOOM_GATEWAY_CERT_CHECK_S={CHECK_S})"
        )


# ------------------------------------------------------------------ no change, one pool


@dataclass
class PeerEngine(FakeEngine):
    """`FakeEngine` that records each HTTP request's client (address, port): one upstream
    connection is one port."""

    peers: list[tuple[str, int]] = field(default_factory=list)

    def app(self) -> Any:
        inner = super().app()

        async def app(scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "http":
                client = scope.get("client")
                self.peers.append(tuple(client) if client else ("?", 0))
            await inner(scope, receive, send)

        return app


@pytest.mark.timeout(60)
async def test_item7_without_a_change_requests_reuse_one_upstream_connection(
    build: Build,
) -> None:
    """§7 "on a change it builds a new TLS context": with no change, successive requests
    (across several check intervals) keep using the pooled connection: the pod sees them
    all from one client port."""
    ca1 = new_ca("pod-ca-1")
    s = await build(client=issue(ca1, server=False), ca1=ca1)
    engine = PeerEngine("peers")
    port, _ = await s.pod(client_cas=[ca1], engine=engine)
    await route(s.stack, s.model, port)
    for i in range(4):
        assert await s.chat() == 200, f"request {i}"
        if i == 1:
            await asyncio.sleep(PAST_INTERVAL_S)  # a check runs and finds no change
    assert len(engine.peers) == 4, engine.peers
    assert len(set(engine.peers)) == 1, f"new upstream connections without a change: {engine.peers}"


# ------------------------------------------------------------------ G3 after 2049


@pytest.mark.timeout(60)
async def test_item8_a_client_cert_expiring_after_2049_loads_and_the_gauge_reports_it(
    build: Build,
) -> None:
    """A client certificate valid 30 years (its `notAfter` after 2049, so DER
    GeneralizedTime): the gateway loads it (a pod accepting it answers), and
    `loom_gateway_client_cert_expiry_seconds` is the time it has left."""
    ca1 = new_ca("pod-ca-1", days=40 * YEAR_DAYS)
    client = issue(ca1, server=False, days=30 * YEAR_DAYS)
    year = client.cert.not_valid_after_utc.year
    assert year >= 2050, year
    der = client.cert.public_bytes(serialization.Encoding.DER)
    assert b"\x18\x0f" + str(year).encode() in der, "notAfter is not a GeneralizedTime"
    s = await build(client=client, ca1=ca1)
    port, _ = await s.pod(client_cas=[ca1])
    await route(s.stack, s.model, port)
    assert await s.chat() == 200
    got = cert_expiry_gauge(s.stack)
    assert got is not None, "no loom_gateway_client_cert_expiry_seconds"
    assert abs(got - expiry_left(client.cert)) < 120, (got, expiry_left(client.cert))


# ------------------------------------------------------------------ missing at start


@pytest.mark.timeout(60)
async def test_item9_starts_without_the_client_cert_then_loads_it_when_it_appears(
    build: Build,
) -> None:
    """Contracts §9: the client-certificate file does not exist when the gateway starts.
    The gateway starts; requests fail as connect errors (502 `upstream_lost`), also
    after a check interval; once the manager writes the file, the next check loads it and
    a request succeeds, and the expiry gauge reports the certificate."""
    ca1 = new_ca("pod-ca-1")
    s = await build(client=None, ca1=ca1)
    assert not s.client_file.exists()
    port, _ = await s.pod(client_cas=[ca1])
    await route(s.stack, s.model, port)
    for _ in range(2):
        status, body = await s.chat_body()
        assert status == 502, (status, body)
        assert body.get("error", {}).get("code") == "upstream_lost", body
        await asyncio.sleep(PAST_INTERVAL_S)
    assert cert_expiry_gauge(s.stack) is None

    client = issue(ca1, server=False, days=7)
    write_atomic(s.client_file, client.combined())
    await asyncio.sleep(PAST_INTERVAL_S)
    assert await s.chat() == 200, "the file appeared, but the next check did not load it"
    got = cert_expiry_gauge(s.stack)
    assert got is not None and abs(got - expiry_left(client.cert)) < 120, got


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
