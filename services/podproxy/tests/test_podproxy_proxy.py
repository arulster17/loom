from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import ssl
import subprocess
import sys
import time

import httpx
import pytest
from podproxy_testlib import POD, client_ssl, gateway_client, pod_ext, serve_asgi, token

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app
from loom_podproxy.proxy import ProxyConfig, parse_networks

CHAT = "/v1/chat/completions"


def auth(certs, pod: str = POD) -> dict[str, str]:
    return {"X-Loom-Pod-Token": token(certs, pod)}


# ---------------------------------------------------------------- streaming


async def test_streams_byte_for_byte_without_buffering(certs, upstream, start_proxy):
    app, upstream_port = upstream
    # Odd boundaries, a multi-byte character split across chunks, an empty-looking line.
    app.chunks = [
        b'data: {"a":"caf',
        "é".encode()[:1],
        "é".encode()[1:] + b'"}\n\n',
        b": keepalive\n\n",
        b"data: [DONE]\n\n",
    ]
    app.hold_after = 1  # the second chunk waits until the test has seen the first
    _, port = await start_proxy(upstream_port)
    received: list[bytes] = []
    async with (
        gateway_client(certs, port) as http,
        http.stream("POST", CHAT, json={"x": 1}, headers=auth(certs), extensions=pod_ext()) as r,
    ):
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/event-stream"
        assert "x-engine-internal" not in r.headers
        chunks = r.aiter_raw()
        first = await asyncio.wait_for(anext(chunks), 5)
        received.append(first)
        # Only now may the upstream continue: had the proxy buffered, we would hang.
        assert first == app.chunks[0]
        app.release.set()
        async for chunk in chunks:
            received.append(chunk)
    assert b"".join(received) == b"".join(app.chunks)


async def test_forwards_only_content_type_and_accept(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    headers = {
        **auth(certs),
        "Authorization": "Bearer loom_sk_client",
        "Cookie": "session=1",
        "X-Forwarded-For": "1.2.3.4",
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    async with gateway_client(certs, port) as http:
        r = await http.post(
            "/v1/completions", content=b'{"prompt":"hi"}', headers=headers, extensions=pod_ext()
        )
    assert r.status_code == 200 and r.json() == {"ok": True}
    (seen,) = app.requests
    assert seen["path"] == "/v1/completions" and seen["body"] == b'{"prompt":"hi"}'
    got = seen["headers"]
    assert got["content-type"] == "application/json"
    assert got["accept"] == "text/event-stream"
    for name in ("authorization", "cookie", "x-forwarded-for", "x-loom-pod-token"):
        assert name not in got


async def test_keepalive_serves_many_requests_on_one_connection(certs, upstream, start_proxy):
    app, upstream_port = upstream
    proxy, port = await start_proxy(upstream_port)
    async with gateway_client(certs, port) as http:
        for _ in range(5):
            r = await http.get("/v1/models", headers=auth(certs), extensions=pod_ext())
            assert r.status_code == 200
        assert len(proxy._tasks) == 1  # one client connection
    assert len(app.requests) == 5


# ---------------------------------------------------------------- refusals


async def test_requires_pod_token(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    async with gateway_client(certs, port) as http:
        for headers in ({}, {"X-Loom-Pod-Token": "wrong"}, auth(certs, "mock-pod-2")):
            r = await http.post(CHAT, json={}, headers=headers, extensions=pod_ext())
            assert r.status_code == 403
            assert r.json()["error"]["type"] == "permission_error"
    assert app.requests == []


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", CHAT),
        ("POST", "/v1/models"),
        ("GET", "/version"),
        ("POST", "/v1/embeddings"),
        ("GET", "/v1/chat/completions/../../admin"),
        ("DELETE", "/health"),
    ],
)
async def test_other_routes_are_404(certs, upstream, start_proxy, method, path):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    async with gateway_client(certs, port) as http:
        r = await http.request(method, path, headers=auth(certs), extensions=pod_ext())
    assert r.status_code == 404
    assert app.requests == []


async def test_query_string_is_dropped(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    async with gateway_client(certs, port) as http:
        r = await http.get("/metrics?debug=1", headers=auth(certs), extensions=pod_ext())
    assert r.status_code == 200
    assert app.requests[0]["path"] == "/metrics"


async def test_body_too_large(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port, max_body_bytes=1000)
    async with gateway_client(certs, port) as http:
        r = await http.post(CHAT, content=b"x" * 5000, headers=auth(certs), extensions=pod_ext())
    assert r.status_code == 413
    assert app.requests == []


async def _refused(certs, port: int, stem: str | None, **ctx_overrides) -> None:
    async with gateway_client(certs, port, stem, **ctx_overrides) as http:
        with pytest.raises(httpx.TransportError):
            await http.get("/health", headers=auth(certs), extensions=pod_ext())


async def test_refuses_client_without_certificate(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    await _refused(certs, port, None)
    assert app.requests == []


async def test_refuses_certificate_from_another_ca(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    await _refused(certs, port, "rogue")
    assert app.requests == []


async def test_refuses_server_certificate_used_as_client(certs, upstream, start_proxy):
    """A pod's own certificate (serverAuth only) is not a gateway client certificate."""
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    await _refused(certs, port, "mock-pod-2")
    assert app.requests == []


async def test_refuses_tls_below_1_3(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    await _refused(certs, port, "gateway", maximum_version=ssl.TLSVersion.TLSv1_2)
    assert app.requests == []


async def test_gateway_detects_wrong_pod_name(certs, upstream, start_proxy):
    """The gateway's side of the check: the certificate must name the expected pod."""
    _, upstream_port = upstream
    _, port = await start_proxy(upstream_port)
    async with gateway_client(certs, port) as http:
        with pytest.raises(httpx.ConnectError, match=r"CERTIFICATE_VERIFY_FAILED|mismatch"):
            await http.get("/health", headers=auth(certs), extensions=pod_ext("mock-pod-2"))


async def test_source_address_allowlist(certs, upstream, start_proxy):
    app, upstream_port = upstream
    _, refused_port = await start_proxy(
        upstream_port, allowed_networks=parse_networks(["10.0.0.0/8"])
    )
    # Closed before the TLS handshake: no TLS alert, the TCP connection just ends.
    raw = socket.create_connection(("127.0.0.1", refused_port), timeout=5)
    raw.settimeout(5)
    assert await asyncio.to_thread(raw.recv, 1) == b""
    raw.close()
    await _refused(certs, refused_port, "gateway")

    _, allowed_port = await start_proxy(
        upstream_port, allowed_networks=parse_networks(["10.0.0.0/8,127.0.0.1/32"])
    )
    async with gateway_client(certs, allowed_port) as http:
        r = await http.get("/health", headers=auth(certs), extensions=pod_ext())
    assert r.status_code == 200
    assert len(app.requests) == 1


def test_ip_allowed_rules():
    cfg = ProxyConfig(token="t", allowed_networks=parse_networks(["203.0.113.7", "10.1.0.0/16"]))
    assert cfg.ip_allowed("203.0.113.7")
    assert cfg.ip_allowed("::ffff:203.0.113.7")  # IPv4-mapped, as a dual-stack socket reports
    assert cfg.ip_allowed("10.1.200.3")
    assert not cfg.ip_allowed("203.0.113.8")
    assert not cfg.ip_allowed("10.2.0.1")
    assert not cfg.ip_allowed("not-an-ip")
    assert ProxyConfig(token="t").ip_allowed("198.51.100.1")  # no allowlist: any source
    with pytest.raises(ValueError):
        ProxyConfig(token="")
    assert parse_networks(["2001:db8::/32"]) == (ipaddress.ip_network("2001:db8::/32"),)


# ---------------------------------------------------------------- the mock engine behind it


@pytest.fixture
def mock_engine():
    config = MockConfig(models=["qwen3-8b"], time_scale=0.02, abort_rate=0.0)
    with serve_asgi(create_app(config)) as port:
        yield port


@pytest.fixture
def aborting_engine():
    config = MockConfig(models=["qwen3-8b"], time_scale=0.001, abort_rate=1.0)
    with serve_asgi(create_app(config)) as port:
        yield port


def engine_running(port: int) -> float:
    text = httpx.get(f"http://127.0.0.1:{port}/metrics").text
    for line in text.splitlines():
        if line.startswith("vllm:num_requests_running"):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError("no running gauge")


async def test_chat_through_proxy_with_continuous_usage(certs, mock_engine, start_proxy):
    _, port = await start_proxy(mock_engine)
    body = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "Hello there"}],
        "max_tokens": 12,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    events = []
    async with (
        gateway_client(certs, port) as http,
        http.stream("POST", CHAT, json=body, headers=auth(certs), extensions=pod_ext()) as r,
    ):
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(line[6:])
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    final = chunks[-1]["usage"]
    assert chunks[-2]["usage"]["completion_tokens"] == final["completion_tokens"] == 12

    async with gateway_client(certs, port) as http:
        models = await http.get("/v1/models", headers=auth(certs), extensions=pod_ext())
        health = await http.get("/health", headers=auth(certs), extensions=pod_ext())
        tok = await http.post(
            "/tokenize",
            json={"model": "qwen3-8b", "messages": body["messages"]},
            headers=auth(certs),
            extensions=pod_ext(),
        )
    assert [m["id"] for m in models.json()["data"]] == ["qwen3-8b"]
    assert health.status_code == 200
    # The gateway's count for fitting an omitted max_tokens (contracts §6, §9).
    assert tok.status_code == 200
    assert tok.json()["count"] == final["prompt_tokens"]


async def test_client_abort_cancels_the_engine_request(certs, mock_engine, start_proxy):
    _, port = await start_proxy(mock_engine)
    body = {
        "model": "qwen3-8b",
        "messages": [{"role": "user", "content": "long"}],
        "max_tokens": 4000,
        "ignore_eos": True,
        "stream": True,
    }
    async with (
        gateway_client(certs, port) as http,
        http.stream("POST", CHAT, json=body, headers=auth(certs), extensions=pod_ext()) as r,
    ):
        lines = r.aiter_lines()
        await anext(lines)
        assert await asyncio.to_thread(engine_running, mock_engine) == 1
        # Leaving the block closes the stream; the client gives up mid-response.
    deadline = time.monotonic() + 5
    while await asyncio.to_thread(engine_running, mock_engine) != 0:
        assert time.monotonic() < deadline, "engine still running the aborted request"
        await asyncio.sleep(0.02)


async def test_client_abort_reaches_scripted_upstream(certs, upstream, start_proxy):
    app, upstream_port = upstream
    app.chunks = [b"data: x\n\n"] * 400
    app.chunk_delay_s = 0.02
    _, port = await start_proxy(upstream_port)
    async with (
        gateway_client(certs, port) as http,
        http.stream("POST", CHAT, json={}, headers=auth(certs), extensions=pod_ext()) as r,
    ):
        await anext(r.aiter_raw())
    assert await asyncio.to_thread(app.disconnected.wait, 5)


async def test_engine_lost_mid_stream_is_not_a_clean_end(certs, aborting_engine, start_proxy):
    _, port = await start_proxy(aborting_engine)
    body = {
        "model": "qwen3-8b",
        "prompt": "hi",
        "max_tokens": 50,
        "ignore_eos": True,
        "stream": True,
    }
    async with gateway_client(certs, port) as http:
        for _ in range(3):
            with pytest.raises(httpx.RemoteProtocolError):
                async with http.stream(
                    "POST", "/v1/completions", json=body, headers=auth(certs), extensions=pod_ext()
                ) as r:
                    async for _line in r.aiter_lines():
                        pass


async def test_health_is_503_when_engine_is_down(certs, start_proxy):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]
    _, port = await start_proxy(dead_port)
    async with gateway_client(certs, port) as http:
        health = await http.get("/health", headers=auth(certs), extensions=pod_ext())
        chat = await http.post(CHAT, json={}, headers=auth(certs), extensions=pod_ext())
    assert health.status_code == 503
    assert chat.status_code == 502
    assert chat.json()["error"]["code"] == "upstream_unavailable"


# ---------------------------------------------------------------- CLI


def test_cli_serves(certs, tmp_path):
    with serve_asgi(create_app(MockConfig(models=["m"], time_scale=0.001))) as engine_port:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cmd = [
            sys.executable,
            "-m",
            "loom_podproxy.proxy",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--upstream",
            f"http://127.0.0.1:{engine_port}",
            "--cert",
            str(certs / f"{POD}.crt"),
            "--key",
            str(certs / f"{POD}.key"),
            "--client-ca",
            str(certs / "ca.crt"),
            "--token-file",
            str(certs / f"{POD}.token"),
            "--allow-ip",
            "127.0.0.1",
            "--log-level",
            "warning",
        ]
        proc = subprocess.Popen(cmd)
        try:
            ctx = client_ssl(certs)
            deadline = time.monotonic() + 15
            with httpx.Client(verify=ctx, timeout=5, trust_env=False) as http:
                while True:
                    try:
                        r = http.get(
                            f"https://127.0.0.1:{port}/v1/models",
                            headers=auth(certs),
                            extensions=pod_ext(),
                        )
                        break
                    except httpx.TransportError:
                        assert time.monotonic() < deadline and proc.poll() is None
                        time.sleep(0.05)
            assert r.status_code == 200
            assert [m["id"] for m in r.json()["data"]] == ["m"]
        finally:
            proc.terminate()
            assert proc.wait(timeout=10) == 0
