"""Helpers for the gateway's certificate-reload tests (docs/phase1-m3-spec.md §7, §13
A27; contracts §2 `LOOM_GATEWAY_CLIENT_CERT`/`_KEY`, `LOOM_POD_CA_CERT`, §9).

- CAs and certificates of any validity (`new_ca`, `issue`), written as the pool manager
  writes the gateway's client certificate: key and certificate in one PEM, renamed into
  place (`write_atomic`). The gateway is given that one file as both
  `LOOM_GATEWAY_CLIENT_CERT` and `LOOM_GATEWAY_CLIENT_KEY`.
- `pod`: a local pod (the shared `FakeEngine`, or `HeldEngine`) behind uvicorn's mTLS
  with a server certificate and the client CAs it accepts, on a free port, for the
  stack's lifetime. A new pod means a new upstream connection: the gateway cannot reuse
  a pooled one for it.
- `HeldEngine`: a streaming engine that stops after its first content token until the
  test releases it, so a stream stays open across a reload.
- `chat` / `wait_until_ok`: requests through the gateway (the observable result of the
  gateway's TLS context).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import ipaddress
import os
import ssl
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from gateway_testkit import SERVER_NAME, FakeEngine, replica, serve, set_ready
from starlette.responses import Response

DAY_S = 86400
MESSAGES = [{"role": "user", "content": "Count."}]
# The gateway checks its files every 60 s (spec §7); a change must be in use for new
# connections within that, plus slack for a loaded CI runner.
PICKUP_BOUND_S = 90.0


@dataclass(frozen=True)
class Issued:
    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey

    def cert_pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)

    def key_pem(self) -> bytes:
        return self.key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    def combined(self) -> bytes:
        """Key and certificate in one PEM (the pool manager's gateway file)."""
        return self.key_pem() + self.cert_pem()


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def new_ca(cn: str = "loom-m3-test-ca", *, days: float = 30) -> Issued:
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    name = _name(f"{cn}-{uuid.uuid4().hex[:6]}")  # self-signed: issuer = subject
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(ski, critical=False)
    )
    return Issued(cert.sign(key, hashes.SHA256()), key)


def issue(ca: Issued, *, server: bool, days: float = 1, cn: str | None = None) -> Issued:
    """A leaf from `ca`: a pod server certificate (SAN `mock-pod-1` and 127.0.0.1,
    serverAuth) or a gateway client certificate (clientAuth), valid `days`."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    b = (
        x509.CertificateBuilder()
        .subject_name(_name(cn or (SERVER_NAME if server else "loom-gateway")))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()),
            critical=False,
        )
    )
    if server:
        b = b.add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(SERVER_NAME), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        ).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    else:
        b = b.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
    return Issued(b.sign(ca.key, hashes.SHA256()), key)


def bundle(*cas: Issued) -> bytes:
    return b"".join(ca.cert_pem() for ca in cas)


def write_atomic(path: Path, data: bytes) -> None:
    """Write to a temporary name and rename into place (as the manager does)."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def expiry_left(cert: x509.Certificate) -> float:
    return cert.not_valid_after_utc.timestamp() - time.time()


@dataclass
class HeldEngine(FakeEngine):
    """`FakeEngine` whose streams stop after `hold_after` SSE events (the role chunk and
    the first token) until `gate` is set; `holding` is set when one is held."""

    hold_after: int = 2
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    holding: asyncio.Event = field(default_factory=asyncio.Event)

    def _stream(self, *args: Any, **kwargs: Any) -> Response:
        resp = super()._stream(*args, **kwargs)
        inner = resp.body_iterator  # type: ignore[attr-defined]

        async def held() -> AsyncIterator[Any]:
            n = 0
            async for part in inner:
                yield part
                n += 1
                if n == self.hold_after:
                    self.holding.set()
                    await self.gate.wait()

        resp.body_iterator = held()  # type: ignore[attr-defined]
        return resp


def _pod_files(directory: Path, server: Issued, client_cas: list[Issued]) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    crt, key, cas = (directory / f"{tag}.{n}" for n in ("crt", "key", "cas"))
    crt.write_bytes(server.cert_pem())
    key.write_bytes(server.key_pem())
    cas.write_bytes(bundle(*client_cas))
    return crt, key, cas


async def pod(
    stack: Any,
    directory: Path,
    *,
    server: Issued,
    client_cas: list[Issued],
    engine: FakeEngine | None = None,
) -> tuple[int, FakeEngine]:
    """A pod for the stack's lifetime: `engine` behind mTLS presenting `server` and
    accepting client certificates from `client_cas`. Returns (port, engine)."""
    crt, key, cas = _pod_files(directory, server, client_cas)
    engine = engine or FakeEngine(f"pod-{crt.stem}")
    port = await stack.track_server(
        serve(
            engine.app(),
            ssl_certfile=str(crt),
            ssl_keyfile=str(key),
            ssl_ca_certs=str(cas),
            ssl_cert_reqs=ssl.CERT_REQUIRED,
        )
    )
    return port, engine


async def route(stack: Any, model: str, port: int, rid: str | None = None) -> None:
    """Make the pod on `port` `model`'s only ready replica."""
    await set_ready(stack.redis, model, [replica(rid or f"r-{port}", port)])


async def chat(stack: Any, key: str, model: str, *, timeout_s: float = 30) -> int:
    """A non-streaming chat request through the gateway; its status."""
    async with httpx.AsyncClient(base_url=stack.url, timeout=timeout_s, trust_env=False) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": model, "messages": MESSAGES, "max_tokens": 5},
            headers={"Authorization": f"Bearer {key}"},
        )
    return r.status_code


async def wait_until_ok(
    stack: Any, key: str, model: str, *, timeout_s: float = PICKUP_BOUND_S, every_s: float = 1.0
) -> float:
    """Ask once a second until a request succeeds; seconds it took. Fails after
    `timeout_s`."""
    start = time.monotonic()
    statuses: list[int] = []
    while True:
        status = await chat(stack, key, model)
        if status == 200:
            return time.monotonic() - start
        statuses.append(status)
        if time.monotonic() - start > timeout_s:
            raise AssertionError(
                f"no successful request within {timeout_s} s (statuses {sorted(set(statuses))})"
            )
        await asyncio.sleep(every_s)


class OpenStream:
    """A streaming chat request through the gateway, read as it arrives."""

    def __init__(self, stack: Any, key: str, model: str) -> None:
        self._client = httpx.AsyncClient(base_url=stack.url, timeout=300, trust_env=False)
        self._req = self._client.build_request(
            "POST",
            "/v1/chat/completions",
            json={"model": model, "messages": MESSAGES, "max_tokens": 5, "stream": True},
            headers={"Authorization": f"Bearer {key}"},
        )
        self.lines: list[str] = []
        self.status: int | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> OpenStream:
        resp = await self._client.send(self._req, stream=True)
        self.status = resp.status_code

        async def read() -> None:
            try:
                async for line in resp.aiter_lines():
                    if line:
                        self.lines.append(line)
            finally:
                await resp.aclose()

        self._task = asyncio.ensure_future(read())
        return self

    def text(self) -> str:
        return "\n".join(self.lines)

    def done(self) -> bool:
        return self._task is not None and self._task.done()

    async def finish(self, timeout_s: float = 30) -> str:
        assert self._task is not None
        await asyncio.wait_for(self._task, timeout_s)
        return self.text()

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        await self._client.aclose()


def cert_expiry_gauge(stack: Any) -> float | None:
    """`loom_gateway_client_cert_expiry_seconds` from the gateway's registry."""
    v = stack.gateway.metrics.registry.get_sample_value(
        "loom_gateway_client_cert_expiry_seconds", {}
    )
    return None if v is None else float(v)
