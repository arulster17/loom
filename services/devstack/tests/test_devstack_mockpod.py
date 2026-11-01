from __future__ import annotations

import signal
import socket
import ssl
import subprocess
import sys
import time

import httpx

from loom_bench.registry import load_registry
from loom_devstack.devcerts import ensure_dev_certs
from loom_devstack.mockpod import ENGINE_PROFILES, engine_config, healthcheck
from loom_platform.pricing import servable


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_engine_profiles_cover_servable_models():
    ids = {m.id for m in servable(load_registry())}
    assert ids <= set(ENGINE_PROFILES)
    cfg = engine_config(["qwen3-8b"], [("time_scale", 0.5)])
    assert cfg["models"] == ["qwen3-8b"] and cfg["max_model_len"] == 32768
    assert cfg["time_scale"] == 0.5 and cfg["step_base_ms"] == 20.0


def test_mock_pod_serves_over_mtls_only(tmp_path):
    ensure_dev_certs(tmp_path, ("mock-pod-9",))
    port, engine_port = free_port(), free_port()
    cmd = [
        sys.executable,
        "-m",
        "loom_devstack.mockpod",
        "--pod-id",
        "mock-pod-9",
        "--model",
        "qwen3-8b",
        "--certs-dir",
        str(tmp_path),
        "--port",
        str(port),
        "--engine-port",
        str(engine_port),
        "--engine-set",
        "time_scale=0.01",
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd)
    try:
        deadline = time.monotonic() + 30
        while healthcheck(port, engine_port, pod_id="mock-pod-9", certs=tmp_path) != 0:
            assert proc.poll() is None, "mock pod exited"
            assert time.monotonic() < deadline, "mock pod not healthy"
            time.sleep(0.1)
        ctx = ssl.create_default_context(cafile=tmp_path / "ca.crt")
        ctx.load_cert_chain(tmp_path / "gateway.crt", tmp_path / "gateway.key")
        token = (tmp_path / "mock-pod-9.token").read_text().strip()
        with httpx.Client(verify=ctx, timeout=10, trust_env=False) as http:
            r = http.post(
                f"https://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": "qwen3-8b",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                },
                headers={"X-Loom-Pod-Token": token},
                extensions={"sni_hostname": "mock-pod-9"},
            )
        assert r.status_code == 200
        assert r.json()["usage"]["completion_tokens"] <= 5
        # The engine itself is on loopback only, at its own port.
        assert httpx.get(f"http://127.0.0.1:{engine_port}/health").status_code == 200
    finally:
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 0
