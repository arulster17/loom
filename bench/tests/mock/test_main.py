import socket
import subprocess
import sys
import time

import httpx
import pytest

from loom_bench.mock.config import MockConfig


def test_config_from_yaml(tmp_path):
    path = tmp_path / "mock.yaml"
    path.write_text("models: [a]\nmax_num_seqs: 4\ndegrade: 0.25\n")
    config = MockConfig.from_yaml(path)
    assert (config.models, config.max_num_seqs, config.degrade) == (["a"], 4, 0.25)
    path.write_text("max_num_seqs: 0\n")
    with pytest.raises(ValueError):
        MockConfig.from_yaml(path)


def test_module_entry_point_serves(tmp_path):
    config = tmp_path / "mock.yaml"
    config.write_text("models: [cli-model]\ntime_scale: 0.001\n")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cmd = [sys.executable, "-m", "loom_bench.mock", "--port", str(port), "--config", str(config)]
    proc = subprocess.Popen([*cmd, "--log-level", "warning"])
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                models = httpx.get(f"http://127.0.0.1:{port}/v1/models").json()
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline and proc.poll() is None
                time.sleep(0.05)
        assert [m["id"] for m in models["data"]] == ["cli-model"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_cli_models_and_overrides(tmp_path):
    from loom_bench.mock.__main__ import _override, build_config

    path = tmp_path / "mock.yaml"
    path.write_text("models: [a]\nstep_base_ms: 3\n")
    config = build_config(
        path, ["x", "y"], [_override("step_base_ms=20"), _override("byte_level=true")]
    )
    assert config.models == ["x", "y"]
    assert config.step_base_ms == 20 and config.byte_level is True
    assert build_config(None, None, []).models == ["mock-model"]
    with pytest.raises(ValueError):
        build_config(None, None, [_override("no_such_field=1")])
