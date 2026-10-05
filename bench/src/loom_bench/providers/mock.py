"""Mock provider: every host is an in-process mock backend on a free port.

Hosts live in a module-level table so `reap` finds servers whose runner died
without tearing down. Mock servers cannot outlive their process (daemon
threads), so a host recorded by a dead process is already gone.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from loom_bench import __version__
from loom_bench.experiment import mock_config_from_launch
from loom_bench.jobexec import execute_load_job
from loom_bench.jobs import EvalJob, EvalJobResult, LoadJob, LoadJobResult
from loom_bench.mock.config import MockConfig
from loom_bench.providers.base import Endpoint, EngineLaunch, Host, HostRequest
from loom_bench.quality.runner import execute_eval_job

SERVER_START_TIMEOUT_S = 10.0


@dataclass
class _Server:
    server: object  # uvicorn.Server
    thread: threading.Thread
    sock: socket.socket
    port: int


@dataclass
class _MockHost:
    host: Host
    server: _Server | None = None


_LIVE: dict[str, _MockHost] = {}


def live_host_ids() -> list[str]:
    return sorted(_LIVE)


def _start_server(config: MockConfig) -> _Server:
    import uvicorn

    from loom_bench.mock.server import create_app

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(config), log_level="critical"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [sock]}, daemon=True, name=f"mock-{port}"
    )
    thread.start()
    return _Server(server=server, thread=thread, sock=sock, port=port)


def _stop_server(s: _Server) -> None:
    s.server.should_exit = True  # type: ignore[attr-defined]
    s.thread.join(timeout=10)
    s.sock.close()


class MockProvider:
    name = "mock"

    def __init__(self, hourly_micros: int = 0) -> None:
        self.hourly_micros = hourly_micros

    async def provision(self, req: HostRequest) -> Host:
        now = datetime.now(UTC)
        host = Host(
            provider=self.name,
            host_id=f"mock-{uuid.uuid4().hex[:12]}",
            request=req,
            hourly_micros=self.hourly_micros,
            launched_at=now,
            ttl_at=now + timedelta(seconds=req.ttl_s),
        )
        _LIVE[host.host_id] = _MockHost(host)
        return host

    def _state(self, host: Host) -> _MockHost:
        try:
            return _LIVE[host.host_id]
        except KeyError:
            raise RuntimeError(f"mock host {host.host_id} is gone") from None

    async def start_engine(self, host: Host, launch: EngineLaunch, *, warm: bool) -> Endpoint:
        state = self._state(host)
        if state.server is not None:
            await self.stop_engine(host)
        config = mock_config_from_launch(launch)
        if launch.served_model not in config.models:
            raise ValueError(f"mock does not serve {launch.served_model!r}")
        # Cold stages count from launch (provisioning included); warm ones from now.
        t_ref = time.monotonic() - (
            0.0 if warm else (datetime.now(UTC) - host.launched_at).total_seconds()
        )
        stages: dict[str, float] = {} if warm else {"instance_running": 0.0}

        srv = await asyncio.to_thread(_start_server, config)
        state.server = srv
        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        while not srv.server.started:  # type: ignore[attr-defined]
            if time.monotonic() > deadline or not srv.thread.is_alive():
                raise RuntimeError("mock server did not start")
            await asyncio.sleep(0.005)
        stages["server_listening"] = time.monotonic() - t_ref

        root = f"http://127.0.0.1:{srv.port}"
        deadline = time.monotonic() + launch.ready_timeout_s
        async with httpx.AsyncClient(base_url=root, timeout=10, trust_env=False) as http:
            while (await http.get("/health")).status_code != 200:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"mock not healthy after {launch.ready_timeout_s}s")
                await asyncio.sleep(0.01)
            stages["engine_healthy"] = time.monotonic() - t_ref
            resp = await http.post(
                "/v1/completions",
                json={"model": launch.served_model, "prompt": "warm up", "max_tokens": 1},
            )
            resp.raise_for_status()
            stages["first_token"] = time.monotonic() - t_ref

        return Endpoint(
            base_url=f"{root}/v1",
            metrics_url=f"{root}/metrics",
            engine="mock",
            served_model=launch.served_model,
            start_stages=stages,
            warm=warm,
            system={
                "engine": "mock",
                "engine_version": __version__,
                "gpu_name": "simulated",
                "gpu_count": host.request.gpus,
            },
        )

    async def stop_engine(self, host: Host) -> None:
        state = self._state(host)
        if state.server is not None:
            srv, state.server = state.server, None
            await asyncio.to_thread(_stop_server, srv)

    async def run_job(self, host: Host, job: LoadJob) -> LoadJobResult:
        self._state(host)
        return await execute_load_job(job)

    async def run_eval(self, host: Host, job: EvalJob) -> EvalJobResult:
        self._state(host)
        return await execute_eval_job(job)

    async def teardown(self, host: Host) -> None:
        state = _LIVE.pop(host.host_id, None)
        if state is not None and state.server is not None:
            await asyncio.to_thread(_stop_server, state.server)

    async def reap(self, now: datetime) -> list[str]:
        expired = [s.host for s in _LIVE.values() if s.host.ttl_at <= now]
        for host in expired:
            await self.teardown(host)
        return [h.host_id for h in expired]
