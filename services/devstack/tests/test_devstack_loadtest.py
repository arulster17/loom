from __future__ import annotations

import json
import socket
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import uvicorn

from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import create_app
from loom_devstack.loadtest import LoadTestConfig, main, prepare, render, report, run_load


class FakeGateway:
    """Bearer-key check in front of the mock: 'limited' gets 429, unknown keys 401.
    Records each request's key and body."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.seen: list[tuple[str, dict[str, Any]]] = []
        self.lock = threading.Lock()

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        key = headers.get(b"authorization", b"").decode().removeprefix("Bearer ")
        chunks = []
        while True:
            msg = await receive()
            chunks.append(msg.get("body", b""))
            if not msg.get("more_body"):
                break
        body = b"".join(chunks)
        with self.lock:
            self.seen.append((key, json.loads(body) if body else {}))
        if key not in ("k1", "k2", "limited"):
            return await _json(send, 401, {"error": {"message": "bad key", "code": "x"}})
        if key == "limited":
            return await _json(send, 429, {"error": {"message": "slow down", "code": "y"}})
        sent = False

        async def replay() -> dict[str, Any]:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)


async def _json(send: Any, status: int, obj: dict[str, Any]) -> None:
    data = json.dumps(obj).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": data})


@contextmanager
def serve(app: Any) -> Iterator[int]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


@pytest.fixture
def gateway():
    fake = FakeGateway(create_app(MockConfig(models=["qwen3-8b"], time_scale=0.01)))
    with serve(fake) as port:
        yield fake, f"http://127.0.0.1:{port}/v1"


def test_prepare_strips_engine_only_fields():
    cfg = LoadTestConfig(base_url="x", api_keys=("k",), model="m", requests=3, warmup=1)
    reqs = prepare(cfg)
    assert len(reqs) == 4
    assert all("ignore_eos" not in r.payload for r in reqs)
    assert all(r.endpoint == "chat" and r.payload["messages"] for r in reqs)
    kept = prepare(
        LoadTestConfig(
            base_url="x", api_keys=("k",), model="m", requests=2, engine_params=True, max_tokens=7
        )
    )
    assert all(r.payload["ignore_eos"] and r.payload["max_tokens"] == 7 for r in kept)
    lab = prepare(
        LoadTestConfig(base_url="x", api_keys=("k",), model="m", workload="fixed-128-128")
    )
    assert lab[0].endpoint == "completions" and "prompt" in lab[0].payload


async def test_run_reports_ttft_and_tokens_per_second(gateway):
    fake, base_url = gateway
    cfg = LoadTestConfig(
        base_url=base_url,
        api_keys=("k1", "k2"),
        model="qwen3-8b",
        concurrency=4,
        requests=20,
        warmup=2,
        max_tokens=16,
    )
    records = await run_load(cfg)
    rep = report(records)
    s = rep.summary
    assert s.n_total == 20 and s.counts["ok"] == 20 and s.error_rate == 0
    assert rep.http_status == {"200": 20}
    assert s.ttft_ms.n == 20 and s.ttft_ms.p50 is not None and s.ttft_ms.p50 > 0
    assert s.per_request_output_tok_s.p50 is not None and s.throughput.output_tok_s > 0
    assert s.requests_missing_usage == 0
    keys = Counter(k for k, _ in fake.seen)
    assert set(keys) == {"k1", "k2"} and sum(keys.values()) == 22
    body = fake.seen[0][1]
    assert body["model"] == "qwen3-8b" and body["stream"] is True
    assert body["stream_options"]["include_usage"] is True
    text = render(rep, cfg)
    assert "TTFT ms" in text and "output tok/s" in text


async def test_rate_limited_and_bad_keys_are_counted(gateway):
    _, base_url = gateway
    cfg = LoadTestConfig(
        base_url=base_url,
        api_keys=("k1", "limited", "nope"),
        model="qwen3-8b",
        concurrency=3,
        requests=12,
        warmup=0,
        max_tokens=4,
    )
    rep = report(await run_load(cfg))
    assert set(rep.http_status) == {"200", "401", "429"}
    assert rep.summary.counts["error"] == rep.http_status["401"] + rep.http_status["429"]
    assert any("slow down" in e for e in rep.errors)


def test_cli_json_and_error_gate(gateway, tmp_path, capsys):
    _, base_url = gateway
    out = tmp_path / "report.json"
    common = ["--base-url", base_url, "--requests", "6", "--warmup", "0", "--max-tokens", "4"]
    main([*common, "--api-key", "k1,k2", "--json", str(out), "--max-error-rate", "0"])
    assert "TTFT ms" in capsys.readouterr().out
    data = json.loads(out.read_text())
    assert data["summary"]["counts"]["ok"] == 6 and data["http_status"] == {"200": 6}
    with pytest.raises(SystemExit) as e:
        main([*common, "--api-key", "limited", "--max-error-rate", "0.5"])
    assert e.value.code == 1
