"""A real `python -m loom_gateway` process stopped with SIGTERM while a stream is open
(docs/phase1-contracts.md §6 "Shutdown"): the stream is cut with an error chunk and
`[DONE]`, the tokens already streamed are billed (status `aborted`), the reservation and
the in-flight counts are released at once (not after the 660 s TTL), and the process
exits within its shutdown budget."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest
import yaml
from gateway_testkit import (
    Behavior,
    FakeEngine,
    make_tenant,
    replica,
    serve,
    set_ready,
    usage_events,
)
from redis.asyncio import Redis

from loom_bench.registry import Registry
from loom_platform.limiter import api_key_key, org_key
from loom_platform.pool import pool_key
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _wait_ready(url: str, proc: subprocess.Popen[bytes], timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"gateway exited during startup ({proc.returncode})")
            try:
                if (await c.get(f"{url}/ready", timeout=1)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.1)
    raise TimeoutError("gateway did not become ready")


@pytest.mark.timeout(120)
async def test_sigterm_mid_stream_settles_and_releases(
    pg_url: str, redis_client: Redis, test_registry: Registry, tmp_path: Path
) -> None:
    small = test_registry.models[0].id
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(yaml.safe_dump(test_registry.model_dump(mode="json")))
    stream = f"loom:test:usage:{uuid.uuid4().hex}"
    port = _free_port()
    engine = FakeEngine("e1", Behavior(tokens="max", token_delay_s=0.05))
    tenant = await make_tenant(pg_url)
    env = {
        **os.environ,
        "LOOM_ENV": "dev",
        "LOOM_DATABASE_URL": pg_url,
        "LOOM_REDIS_URL": os.environ.get(REDIS_ENV) or DEFAULT_REDIS,
        "LOOM_USAGE_STREAM": stream,
        "LOOM_MODELS_YAML": str(models_yaml),
        "LOOM_GATEWAY_HOST": "127.0.0.1",
        "LOOM_GATEWAY_PORT": str(port),
        "LOOM_GATEWAY_METRICS_PORT": str(_free_port()),
        "LOOM_SHUTDOWN_GRACE_S": "1",
        "LOOM_LOG_LEVEL": "WARNING",
    }
    url = f"http://127.0.0.1:{port}"
    async with serve(engine.app()) as engine_port:
        await set_ready(redis_client, small, [replica("e1", engine_port, tls=False)])
        proc = subprocess.Popen([sys.executable, "-m", "loom_gateway"], env=env)  # noqa: ASYNC220
        try:
            await _wait_ready(url, proc)
            body = {
                "model": small,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 2000,  # 100 s of tokens at 50 ms each
                "stream": True,
            }
            content_chunks = 0
            payloads: list[str] = []
            sent_at: float | None = None
            async with (
                httpx.AsyncClient(timeout=60) as client,
                client.stream(
                    "POST",
                    f"{url}/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {tenant.key}"},
                ) as resp,
            ):
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    payloads.append(data)
                    if data != "[DONE]":
                        chunk = json.loads(data)
                        if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                            content_chunks += 1
                    if content_chunks == 5 and sent_at is None:
                        sent_at = time.monotonic()
                        proc.send_signal(signal.SIGTERM)
            assert sent_at is not None
            # Cut within the grace (1 s) plus a little, not after the 100 s of tokens.
            assert time.monotonic() - sent_at < 10
            assert payloads[-1] == "[DONE]"
            err = json.loads(payloads[-2])["error"]
            assert err["code"] == "server_shutdown"
            # uvicorn re-raises the captured SIGTERM once its graceful shutdown is done.
            assert proc.wait(timeout=20) in (0, -signal.SIGTERM)

            events = await usage_events(redis_client, stream)
            assert len(events) == 1
            ev = events[0]
            assert ev.status == "aborted"
            assert ev.completion_tokens == content_chunks
            assert ev.prompt_tokens == engine.behavior.prompt_tokens
            assert ev.cost_micros > 0
            # Released now, not when the reservation's TTL runs out.
            assert int(await redis_client.get(org_key(tenant.org_id, "rsum")) or 0) == 0
            assert await redis_client.zcard(org_key(tenant.org_id, "rsv:exp")) == 0
            inflight = api_key_key(tenant.org_id, tenant.key_id, "inflight")
            assert int(await redis_client.get(inflight) or 0) == 0
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            keys = [k async for k in redis_client.scan_iter(f"loom:org:{tenant.org_id}:*")]
            await redis_client.delete(stream, pool_key(small), *keys)
