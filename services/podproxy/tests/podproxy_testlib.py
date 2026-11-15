"""Helpers for the pod proxy tests (imported by conftest.py and the test modules)."""

from __future__ import annotations

import asyncio
import socket
import ssl
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import uvicorn

POD = "mock-pod-1"


def token(certs: Path, pod: str = POD) -> str:
    return (certs / f"{pod}.token").read_text().strip()


def client_ssl(certs: Path, stem: str | None = "gateway", **kw: Any) -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=certs / "ca.crt")
    if stem is not None:
        ctx.load_cert_chain(certs / f"{stem}.crt", certs / f"{stem}.key")
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


@contextmanager
def serve_asgi(app: Any) -> Iterator[int]:
    """Run an ASGI app with uvicorn on a free loopback port in a background thread."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="auto"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        assert time.monotonic() < deadline, "upstream did not start"
        time.sleep(0.005)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


class Upstream:
    """A scripted engine: records what reaches it, streams `chunks` for the stream path,
    can hold the stream until `release` is set, and notes client disconnects."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.chunks: list[bytes] = [b"data: one\n\n", b"data: two\n\n"]
        self.hold_after: int | None = None
        self.release = threading.Event()
        self.disconnected = threading.Event()
        self.chunk_delay_s = 0.0

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while (await receive())["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.startup.complete"})
            await send({"type": "lifespan.shutdown.complete"})
            return
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        self.requests.append(
            {
                "method": scope["method"],
                "path": scope["path"],
                "headers": {k.decode(): v.decode() for k, v in scope["headers"]},
                "body": body,
            }
        )

        async def watch() -> None:
            while (await receive())["type"] != "http.disconnect":
                pass
            self.disconnected.set()

        watcher = asyncio.create_task(watch())
        try:
            if scope["path"] == "/v1/chat/completions":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream"),
                            (b"x-engine-internal", b"secret"),
                        ],
                    }
                )
                for i, chunk in enumerate(self.chunks):
                    if self.hold_after is not None and i == self.hold_after:
                        await asyncio.to_thread(self.release.wait, 10)
                    if self.disconnected.is_set():
                        return
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                    if self.chunk_delay_s:
                        await asyncio.sleep(self.chunk_delay_s)
                await send({"type": "http.response.body", "body": b""})
            else:
                payload = b'{"ok":true}'
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": payload})
        finally:
            watcher.cancel()


def gateway_client(certs: Path, port: int, stem: str | None = "gateway", **kw: Any) -> Any:
    return httpx.AsyncClient(
        base_url=f"https://127.0.0.1:{port}",
        verify=client_ssl(certs, stem, **kw),
        timeout=10,
        trust_env=False,
    )


def pod_ext(pod: str = POD) -> dict[str, Any]:
    """Connect to 127.0.0.1 but verify the certificate names `pod`, as the gateway does
    with `replica.server_name`."""
    return {"sni_hostname": pod}
