"""Test helpers for the gateway: an in-process fake engine (vLLM-shaped, honours
`continuous_usage_stats`, injectable faults), uvicorn servers on free ports, dev mTLS
certificates, tenants and pools."""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import ipaddress
import json
import socket
import ssl
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import psycopg
import uvicorn
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from redis.asyncio import Redis
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from loom_platform.ledger import approve_org
from loom_platform.pool import PoolState, Replica, set_pool
from loom_platform.testing import create_api_key, create_org, create_user
from loom_platform.usage import UsageEvent

POD_TOKEN = "pod-token-test"
SERVER_NAME = "mock-pod-1"

# ------------------------------------------------------------------ fake engine


@dataclass
class Behavior:
    # Completion tokens generated: a number (capped at max_tokens) or "max".
    tokens: int | Literal["max"] = 5
    token_delay_s: float = 0.0
    first_token_delay_s: float = 0.0
    prompt_tokens: int = 11
    cached_tokens: int = 3
    fail_status: int | None = None  # answer this status (with an error body) instead
    fail_message: str = "engine failure"
    drop_after: int | None = None  # stream: kill the connection after this many tokens
    continuous_usage: bool = True  # honour stream_options.continuous_usage_stats


class EngineDropped(Exception):
    """Raised inside a stream to make uvicorn drop the connection mid-body."""


@dataclass
class FakeEngine:
    name: str = "engine"
    behavior: Behavior = field(default_factory=Behavior)
    requests: list[dict[str, Any]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)  # tokens sent, per stream
    aborted: int = 0

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/chat/completions", self._chat, methods=["POST"]),
                Route("/v1/completions", self._completions, methods=["POST"]),
                Route("/v1/models", self._models, methods=["GET"]),
                Route("/health", lambda r: Response(), methods=["GET"]),
            ]
        )

    async def _models(self, request: Request) -> Response:
        return JSONResponse({"object": "list", "data": []})

    async def _begin(self, request: Request) -> tuple[dict[str, Any], Response | None]:
        body = json.loads(await request.body())
        self.requests.append(body)
        self.headers.append({k.lower(): v for k, v in request.headers.items()})
        if request.headers.get("x-loom-pod-token") != POD_TOKEN:
            return body, JSONResponse({"error": {"message": "bad pod token"}}, status_code=403)
        b = self.behavior
        if b.fail_status is not None:
            return body, JSONResponse(
                {
                    "error": {
                        "message": b.fail_message,
                        "type": "BadRequestError",
                        "param": None,
                        "code": b.fail_status,
                    }
                },
                status_code=b.fail_status,
            )
        return body, None

    def _n(self, body: dict[str, Any]) -> int:
        cap = body.get("max_completion_tokens") or body.get("max_tokens") or 16
        t = self.behavior.tokens
        return cap if t == "max" else min(t, cap)

    def _usage(self, completion: int, *, details: bool = True) -> dict[str, Any]:
        p = self.behavior.prompt_tokens
        u: dict[str, Any] = {
            "prompt_tokens": p,
            "completion_tokens": completion,
            "total_tokens": p + completion,
        }
        if details:
            u["prompt_tokens_details"] = {"cached_tokens": self.behavior.cached_tokens}
        return u

    @staticmethod
    def _content(body: dict[str, Any], n: int) -> list[str]:
        rf = body.get("response_format") or {}
        if rf.get("type") in ("json_object", "json_schema"):
            text = '{"answer": 42}'
            # Spread the JSON over n pieces (n >= 1).
            k = max(1, -(-len(text) // max(n, 1)))
            pieces = [text[i : i + k] for i in range(0, len(text), k)]
            return pieces + [""] * (n - len(pieces))
        return [f"tok{i} " for i in range(1, n + 1)]

    async def _chat(self, request: Request) -> Response:
        body, err = await self._begin(request)
        if err is not None:
            return err
        n = self._n(body)
        model = body["model"]
        tool = None
        if body.get("tools") and body.get("tool_choice") != "none":
            tool = body["tools"][0]["function"]["name"]
        pieces = self._content(body, n)
        rid = f"chatcmpl-{uuid.uuid4().hex}"
        finish = "tool_calls" if tool else ("length" if n == self._cap(body) else "stop")
        if not body.get("stream"):
            await asyncio.sleep(self.behavior.first_token_delay_s)
            if tool:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": tool, "arguments": '{"city": "Paris"}'},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "".join(pieces)}
            return JSONResponse(
                {
                    "id": rid,
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {"index": 0, "message": message, "logprobs": None, "finish_reason": finish}
                    ],
                    "usage": self._usage(n),
                }
            )

        def delta(i: int) -> dict[str, Any]:
            if tool:
                args = '{"city": "Paris"}' if i == 1 else ""
                return {"tool_calls": [{"index": 0, "function": {"arguments": args}}]}
            return {"content": pieces[i - 1]}

        def chunk(d: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
            return {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model,
                "choices": [
                    {"index": 0, "delta": d, "logprobs": None, "finish_reason": finish_reason}
                ],
            }

        first: dict[str, Any] = {"role": "assistant", "content": ""}
        if tool:
            first = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": tool, "arguments": ""},
                    }
                ],
            }
        return self._stream(
            body,
            n,
            lambda i: chunk(first if i == 0 else delta(i), finish if i == n else None),
            rid,
            model,
            "chat.completion.chunk",
        )

    @staticmethod
    def _cap(body: dict[str, Any]) -> int:
        return int(body.get("max_completion_tokens") or body.get("max_tokens") or 16)

    async def _completions(self, request: Request) -> Response:
        body, err = await self._begin(request)
        if err is not None:
            return err
        n = self._n(body)
        model = body["model"]
        rid = f"cmpl-{uuid.uuid4().hex}"
        pieces = [f"tok{i} " for i in range(1, n + 1)]
        finish = "length" if n == self._cap(body) else "stop"
        if not body.get("stream"):
            return JSONResponse(
                {
                    "id": rid,
                    "object": "text_completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "text": "".join(pieces),
                            "logprobs": None,
                            "finish_reason": finish,
                        }
                    ],
                    "usage": self._usage(n),
                }
            )

        def chunk(i: int) -> dict[str, Any]:
            return {
                "id": rid,
                "object": "text_completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": pieces[i - 1],
                        "logprobs": None,
                        "finish_reason": finish if i == n else None,
                    }
                ],
            }

        return self._stream(body, n, chunk, rid, model, "text_completion", role_chunk=False)

    def _stream(
        self,
        body: dict[str, Any],
        n: int,
        make: Any,
        rid: str,
        model: str,
        obj: str,
        *,
        role_chunk: bool = True,
    ) -> Response:
        opts = body.get("stream_options") or {}
        include = bool(opts.get("include_usage"))
        continuous = (
            include
            and bool(opts.get("continuous_usage_stats"))
            and (self.behavior.continuous_usage)
        )
        b = self.behavior
        slot = len(self.generated)
        self.generated.append(0)

        def sse(o: dict[str, Any]) -> bytes:
            return b"data: " + json.dumps(o, separators=(",", ":")).encode() + b"\n\n"

        async def events() -> AsyncIterator[bytes]:
            finished = False
            try:
                await asyncio.sleep(b.first_token_delay_s)
                start = 0 if role_chunk else 1
                for i in range(start, n + 1):
                    if i > 1:
                        await asyncio.sleep(b.token_delay_s)
                    if b.drop_after is not None and i > b.drop_after:
                        raise EngineDropped
                    c = make(i)
                    if continuous:
                        c["usage"] = self._usage(i, details=False)
                    self.generated[slot] = i
                    yield sse(c)
                if include:
                    yield sse(
                        {
                            "id": rid,
                            "object": obj,
                            "created": 0,
                            "model": model,
                            "choices": [],
                            "usage": self._usage(n),
                        }
                    )
                yield b"data: [DONE]\n\n"
                finished = True
            finally:
                if not finished:
                    self.aborted += 1

        return StreamingResponse(events(), media_type="text/event-stream")


# ------------------------------------------------------------------ servers


class TcpProxy:
    """A TCP forwarder in front of the shared Redis, to simulate an outage without
    touching the server: `stop()` refuses and drops every connection (Redis down),
    `freeze()` accepts and forwards nothing (Redis frozen), `start()` restores it on the
    same port."""

    def __init__(self, host: str, port: int) -> None:
        self.target = (host, port)
        self.port = 0
        self.frozen = False
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()

    async def start(self) -> None:
        self.frozen = False
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        self.port = self._server.sockets[0].getsockname()[1]

    def freeze(self) -> None:
        self.frozen = True

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        for w in list(self._writers):
            w.close()
        self._writers.clear()
        if self._server is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server.wait_closed(), 2)
            self._server = None

    async def _handle(self, r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
        self._writers.add(w)
        try:
            ur, uw = await asyncio.open_connection(*self.target)
        except OSError:
            w.close()
            return
        self._writers.add(uw)

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while data := await src.read(65536):
                    if self.frozen:
                        continue  # swallowed: the caller waits until its socket timeout
                    dst.write(data)
                    await dst.drain()
            except (OSError, asyncio.IncompleteReadError):
                pass
            finally:
                dst.close()

        await asyncio.gather(pipe(r, uw), pipe(ur, w), return_exceptions=True)


@contextlib.asynccontextmanager
async def serve(app: Any, **config: Any) -> AsyncIterator[int]:
    """Run an ASGI app with uvicorn in this event loop on a free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            lifespan="on",
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=5,
            **config,
        )
    )
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("server exited during startup")
            await asyncio.sleep(0.005)
        yield port
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 10)
        sock.close()


@dataclass(frozen=True)
class Certs:
    ca: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path
    rogue_client_cert: Path
    rogue_client_key: Path


def _key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def _write_key(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )


def _cert(
    subject: str,
    key: ec.EllipticCurvePrivateKey,
    issuer: x509.Name | None,
    issuer_key: ec.EllipticCurvePrivateKey,
    *,
    ca: bool = False,
    server: bool = False,
) -> x509.Certificate:
    now = dt.datetime.now(dt.UTC)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, subject)])
    b = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(issuer or name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if ca:
        b = b.add_extension(
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
    elif server:
        b = b.add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(subject), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        ).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
    else:
        b = b.add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
    return b.sign(issuer_key, hashes.SHA256())


def make_certs(directory: Path) -> Certs:
    """A dev CA, a `mock-pod-1` server certificate and a gateway client certificate (plus
    a client certificate from another CA, which the pod must refuse)."""
    pem = serialization.Encoding.PEM
    ca_key = _key()
    ca = _cert("loom-test-ca", ca_key, None, ca_key, ca=True)
    srv_key, cli_key = _key(), _key()
    srv = _cert(SERVER_NAME, srv_key, ca.subject, ca_key, server=True)
    cli = _cert("loom-gateway", cli_key, ca.subject, ca_key)
    rogue_ca_key, rogue_key = _key(), _key()
    rogue_ca = _cert("rogue-ca", rogue_ca_key, None, rogue_ca_key, ca=True)
    rogue = _cert("loom-gateway", rogue_key, rogue_ca.subject, rogue_ca_key)
    out = Certs(
        ca=directory / "ca.pem",
        server_cert=directory / "server.pem",
        server_key=directory / "server.key",
        client_cert=directory / "client.pem",
        client_key=directory / "client.key",
        rogue_client_cert=directory / "rogue.pem",
        rogue_client_key=directory / "rogue.key",
    )
    out.ca.write_bytes(ca.public_bytes(pem))
    out.server_cert.write_bytes(srv.public_bytes(pem))
    _write_key(out.server_key, srv_key)
    out.client_cert.write_bytes(cli.public_bytes(pem))
    _write_key(out.client_key, cli_key)
    out.rogue_client_cert.write_bytes(rogue.public_bytes(pem))
    _write_key(out.rogue_client_key, rogue_key)
    return out


def mtls_server_config(certs: Certs) -> dict[str, Any]:
    return {
        "ssl_certfile": str(certs.server_cert),
        "ssl_keyfile": str(certs.server_key),
        "ssl_ca_certs": str(certs.ca),
        "ssl_cert_reqs": ssl.CERT_REQUIRED,
    }


# ------------------------------------------------------------------ tenants, pools, events


@dataclass(frozen=True)
class Tenant:
    org_id: uuid.UUID
    key_id: uuid.UUID
    key: str


async def make_tenant(
    pg_url: str,
    *,
    credit_micros: int = 5_000_000,
    status: str = "active",
    spend_limit_micros: int | None = None,
    key_limits: dict[str, int] | None = None,
    org_limits: dict[str, int] | None = None,
) -> Tenant:
    """An org (approved with `credit_micros` of signup credit unless pending) and a key."""
    conn = await psycopg.AsyncConnection.connect(pg_url)
    try:
        admin = await create_user(conn, role="admin")
        org = await create_org(conn, owner=admin)
        if status != "pending_approval":
            await approve_org(conn, org, admin, credit_micros)
            if status != "active":
                await conn.execute("UPDATE orgs SET status = %s WHERE id = %s", (status, org))
        key_id, key = await create_api_key(conn, org, spend_limit_micros=spend_limit_micros)
        for col, v in (key_limits or {}).items():
            await conn.execute(
                f"UPDATE api_keys SET {col} = %s WHERE id = %s",  # type: ignore[arg-type]
                (v, key_id),
            )
        for col, v in (org_limits or {}).items():
            await conn.execute(
                f"UPDATE orgs SET {col} = %s WHERE id = %s",  # type: ignore[arg-type]
                (v, org),
            )
        await conn.commit()
    finally:
        await conn.close()
    return Tenant(org, key_id, key.plaintext)


HIGH_LIMITS = {"rpm_limit": 100_000, "tpm_limit": 100_000_000, "concurrency_limit": 1000}


def replica(rid: str, port: int, *, tls: bool = True) -> Replica:
    scheme = "https" if tls else "http"
    return Replica(
        id=rid, url=f"{scheme}://127.0.0.1:{port}", server_name=SERVER_NAME, token=POD_TOKEN
    )


async def set_ready(redis: Redis, model_id: str, replicas: list[Replica]) -> None:
    await set_pool(redis, PoolState(model_id, "ready", 0, tuple(replicas)))


async def usage_events(redis: Redis, stream: str) -> list[UsageEvent]:
    entries = await redis.xrange(stream)
    return [UsageEvent.from_stream_fields(fields) for _, fields in entries]


async def wait_for_events(
    redis: Redis, stream: str, count: int, timeout_s: float = 5.0
) -> list[UsageEvent]:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        events = await usage_events(redis, stream)
        if len(events) >= count or asyncio.get_running_loop().time() > deadline:
            return events
        await asyncio.sleep(0.02)
