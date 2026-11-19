"""A27, the gateway's half: client-certificate rotation and the CA trust bundle
(docs/phase1-m3-spec.md §7; contracts §2 `LOOM_GATEWAY_CLIENT_CERT`/`_KEY`,
`LOOM_POD_CA_CERT`, §9).

The gateway checks its client-certificate file and `LOOM_POD_CA_CERT` every 60 s; on a
change it builds a new TLS context for new upstream connections while open streams keep
theirs; a file that fails to load keeps the old context and logs an error;
`LOOM_POD_CA_CERT` may hold several CAs, any of them trusted; it exports
`loom_gateway_client_cert_expiry_seconds` (alarm G3).

The client certificate is the pool manager's combined file (key and certificate in one
PEM, renamed into place), given as both `LOOM_GATEWAY_CLIENT_CERT` and `_KEY`. Pods are
local mTLS servers (`gateway_m3_cert_kit.pod`); what the gateway's context holds is seen
from which pods it can reach: a pod that accepts only the new client certificate, or
presents a certificate only the new CA signs, answers once the change is picked up.

The spec names no setting for the 60 s interval, so these tests wait for it in real time
(each up to `PICKUP_BOUND_S`, 90 s) and carry their own timeout.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from gateway_m3_cert_kit import (
    DAY_S,
    PICKUP_BOUND_S,
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
from gateway_testkit import HIGH_LIMITS, Tenant

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


@dataclass
class Setup:
    stack: Any  # conftest.Stack
    tenant: Tenant
    dir: Path
    ca1: Issued
    client: Issued  # the gateway's client certificate at start (1 day, from ca1)
    client_file: Path
    ca_file: Path

    @property
    def model(self) -> str:
        return self.stack.small

    async def pod(
        self, *, server_ca: Issued, client_cas: list[Issued], engine: HeldEngine | None = None
    ) -> int:
        port, _ = await pod(
            self.stack,
            self.dir / "pods",
            server=issue(server_ca, server=True),
            client_cas=client_cas,
            engine=engine,
        )
        return port

    async def chat(self) -> int:
        return await chat(self.stack, self.tenant.key, self.model)


@pytest.fixture
async def setup(make_stack: Any, tmp_path: Path) -> Setup:
    ca1 = new_ca("pod-ca-1")
    client = issue(ca1, server=False, days=1)
    gw = tmp_path / "gateway-cert"
    gw.mkdir()
    client_file = gw / "loom-gateway.pem"
    write_atomic(client_file, client.combined())
    ca_file = tmp_path / "pod-ca.pem"
    write_atomic(ca_file, bundle(ca1))
    stack = await make_stack(
        engines=(),
        ready=False,
        gateway_client_cert=str(client_file),
        gateway_client_key=str(client_file),
        pod_ca_cert=str(ca_file),
    )
    tenant = await stack.tenant(key_limits=HIGH_LIMITS, org_limits=HIGH_LIMITS)
    return Setup(stack, tenant, tmp_path, ca1, client, client_file, ca_file)


async def open_held_stream(s: Setup) -> tuple[OpenStream, HeldEngine]:
    """A stream on a pod of ca1 (accepting ca1's clients), held after its first token."""
    engine = HeldEngine("held")
    port = await s.pod(server_ca=s.ca1, client_cas=[s.ca1], engine=engine)
    await route(s.stack, s.model, port)
    stream = await OpenStream(s.stack, s.tenant.key, s.model).start()
    assert stream.status == 200, stream.status
    await _wait(engine.holding, 10, "the stream never reached the engine")
    return stream, engine


async def finish_held_stream(stream: OpenStream, engine: HeldEngine) -> None:
    """Release the held stream: it completes, every token and `[DONE]`, no error."""
    try:
        assert not stream.done(), f"the open stream ended before it was released:\n{stream.text()}"
        engine.gate.set()
        text = await stream.finish()
    finally:
        await stream.aclose()
    for i in range(1, 6):
        assert f"tok{i} " in text, f"token {i} missing from the open stream:\n{text}"
    assert text.rstrip().endswith("data: [DONE]"), text[-500:]
    assert '"error"' not in text, text[-500:]


async def _wait(event: asyncio.Event, timeout_s: float, message: str) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout_s)
    except TimeoutError:
        raise AssertionError(message) from None


# ------------------------------------------------------------------ the gauge


async def test_client_cert_expiry_gauge_at_start(setup: Setup) -> None:
    """`loom_gateway_client_cert_expiry_seconds` is the time left on the client
    certificate the gateway loaded."""
    port = await setup.pod(server_ca=setup.ca1, client_cas=[setup.ca1])
    await route(setup.stack, setup.model, port)
    assert await setup.chat() == 200
    got = cert_expiry_gauge(setup.stack)
    assert got is not None, "no loom_gateway_client_cert_expiry_seconds"
    assert abs(got - expiry_left(setup.client.cert)) < 120, (got, expiry_left(setup.client.cert))


# ------------------------------------------------------------------ rotation


@pytest.mark.slow
@pytest.mark.timeout(300)
async def test_new_client_cert_is_used_for_new_connections_while_an_open_stream_finishes(
    setup: Setup,
) -> None:
    """The manager renames a new certificate (from another CA here, so a pod can tell
    them apart) into the gateway's file: within the check interval a pod that accepts
    only the new one answers, the gauge shows the new certificate's 7 days, and a
    stream opened before the change completes on its old connection."""
    s = setup
    stream, engine = await open_held_stream(s)
    try:
        ca_b = new_ca("client-ca-b")
        new_client = issue(ca_b, server=False, days=7)
        port = await s.pod(server_ca=s.ca1, client_cas=[ca_b])
        await route(s.stack, s.model, port)
        assert await s.chat() != 200, "the pod accepted the old client certificate"

        write_atomic(s.client_file, new_client.combined())
        await wait_until_ok(s.stack, s.tenant.key, s.model)
        got = cert_expiry_gauge(s.stack)
        assert got is not None and abs(got - expiry_left(new_client.cert)) < 120, (
            got,
            expiry_left(new_client.cert),
        )
        assert got > 6 * DAY_S
    finally:
        await finish_held_stream(stream, engine)


@pytest.mark.slow
@pytest.mark.timeout(300)
async def test_new_ca_bundle_is_trusted_for_new_connections_while_an_open_stream_finishes(
    setup: Setup,
) -> None:
    """`LOOM_POD_CA_CERT` rewritten as old + new CA: within the check interval a pod whose
    certificate the new CA signs answers, a new connection to a pod of the old CA still
    does (any CA in the bundle is trusted), and a stream opened before completes."""
    s = setup
    stream, engine = await open_held_stream(s)
    try:
        ca2 = new_ca("pod-ca-2")
        port = await s.pod(server_ca=ca2, client_cas=[s.ca1])
        await route(s.stack, s.model, port)
        assert await s.chat() != 200, "a pod of a CA not in the bundle was trusted"

        write_atomic(s.ca_file, bundle(s.ca1, ca2))
        await wait_until_ok(s.stack, s.tenant.key, s.model)

        port = await s.pod(server_ca=s.ca1, client_cas=[s.ca1])  # a new connection
        await route(s.stack, s.model, port)
        assert await s.chat() == 200, "the old CA is no longer trusted with the bundle"
    finally:
        await finish_held_stream(stream, engine)


# ------------------------------------------------------------------ a broken file


@pytest.mark.slow
@pytest.mark.timeout(300)
@pytest.mark.parametrize("which", ["client_cert", "ca_bundle"])
async def test_a_file_that_fails_to_load_keeps_the_old_context_and_logs_an_error(
    setup: Setup, which: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The client file with a key and no certificate (or a CA file that is not PEM):
    once the gateway has checked it (it logs an error), requests still succeed, a new
    connection to a pod of the old CA still works with the old client certificate, and
    the gauge still shows the old certificate."""
    s = setup
    port = await s.pod(server_ca=s.ca1, client_cas=[s.ca1])
    await route(s.stack, s.model, port)
    assert await s.chat() == 200
    caplog.set_level(logging.ERROR)
    if which == "client_cert":
        write_atomic(s.client_file, issue(s.ca1, server=False, days=7).key_pem())
    else:
        write_atomic(s.ca_file, b"-----BEGIN CERTIFICATE-----\nnot a certificate\n")

    def gateway_errors() -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.ERROR and r.name.startswith("loom_gateway")
        ]

    start = time.monotonic()
    while not gateway_errors():
        assert time.monotonic() - start < PICKUP_BOUND_S, (
            f"no error logged within {PICKUP_BOUND_S} s of the broken {which} file"
        )
        assert await s.chat() == 200, "a request failed while the file was broken"
        await asyncio.sleep(1)

    port = await s.pod(server_ca=s.ca1, client_cas=[s.ca1])  # a new connection
    await route(s.stack, s.model, port)
    assert await s.chat() == 200, f"the broken {which} file replaced the working context"
    got = cert_expiry_gauge(s.stack)
    assert got is not None and abs(got - expiry_left(s.client.cert)) < 120, got
