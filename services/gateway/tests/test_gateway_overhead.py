"""Micro-benchmark: the latency the gateway adds (brief §8.4: p50 < 10 ms).

A zero-latency fake engine answers a small chat completion. The same request goes to the
engine directly and through the gateway, interleaved, over kept-alive connections; the
overhead is the difference of the medians. Upstream runs over mTLS as in production (the
direct baseline uses the same client certificate). Everything runs in one uvloop event
loop, as the gateway does in production, so the number includes the extra HTTP hop and
both Redis calls (reserve, settle) but no network latency.

Run alone for a clean number:
    uv run pytest services/gateway/tests/test_gateway_overhead.py -s

Noisy runners: CI runs this next to the rest of the suite (`pytest -n auto`) on shared
cores, where CPU contention can only ever add latency, never remove it. So the test
measures up to ATTEMPTS times and passes on the first attempt under the target: one clean
median shows the gateway meets it. A real regression raises every attempt and still
fails; the bound stays absolute (10 ms, the brief's number) rather than relative to a
baseline that a slow machine would inflate as well.
"""

from __future__ import annotations

import os
import ssl
import statistics
import time
import uuid

import httpx
import pytest
import redis
import uvloop
from gateway_testkit import (
    SERVER_NAME,
    Certs,
    FakeEngine,
    make_tenant,
    mtls_server_config,
    replica,
    serve,
    set_ready,
)
from redis.asyncio import Redis

from loom_bench.registry import Registry
from loom_gateway.app import create_app
from loom_gateway.settings import GatewaySettings
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV

pytestmark = [pytest.mark.postgres, pytest.mark.redis, pytest.mark.slow]

WARMUP = 50
PAIRS = 400
TARGET_P50_MS = 10.0
ATTEMPTS = 3


def _redis_url() -> str:
    explicit = os.environ.get(REDIS_ENV)
    url = explicit or DEFAULT_REDIS
    try:
        redis.Redis.from_url(url, socket_connect_timeout=2).ping()
    except redis.ConnectionError as e:
        if explicit:
            raise
        pytest.skip(f"Redis not reachable: {e}")
    return url


async def _bench(pg_url: str, redis_url: str, certs: Certs, registry: Registry) -> dict:
    model = registry.models[0].id
    engine = FakeEngine()
    rds = Redis.from_url(redis_url)
    stream = f"loom:test:usage:{uuid.uuid4().hex}"
    settings = GatewaySettings(
        database_url=pg_url,
        redis_url=redis_url,
        usage_stream=stream,
        log_level="WARNING",
        gateway_client_cert=str(certs.client_cert),
        gateway_client_key=str(certs.client_key),
        pod_ca_cert=str(certs.ca),
    )
    limits = {"rpm_limit": 1_000_000, "tpm_limit": 1_000_000_000, "concurrency_limit": 1000}
    tenant = await make_tenant(pg_url, key_limits=limits, org_limits=limits)
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello there."}],
        "max_tokens": 5,
    }
    ctx = ssl.create_default_context(cafile=str(certs.ca))
    ctx.load_cert_chain(str(certs.client_cert), str(certs.client_key))
    try:
        app = create_app(settings, registry=registry)
        async with (
            serve(engine.app(), **mtls_server_config(certs)) as engine_port,
            serve(app) as gw_port,
            httpx.AsyncClient(verify=ctx, timeout=10) as direct,
            httpx.AsyncClient(timeout=10) as client,
        ):
            await set_ready(rds, model, [replica("e1", engine_port)])
            direct_url = f"https://127.0.0.1:{engine_port}/v1/chat/completions"
            gw_url = f"http://127.0.0.1:{gw_port}/v1/chat/completions"
            pod_headers = {"X-Loom-Pod-Token": "pod-token-test"}
            auth = {"Authorization": f"Bearer {tenant.key}"}
            ext = {"sni_hostname": SERVER_NAME}

            async def via_engine() -> float:
                t0 = time.perf_counter()
                r = await direct.post(direct_url, json=body, headers=pod_headers, extensions=ext)
                assert r.status_code == 200
                return time.perf_counter() - t0

            async def via_gateway() -> float:
                t0 = time.perf_counter()
                r = await client.post(gw_url, json=body, headers=auth)
                assert r.status_code == 200, r.text
                return time.perf_counter() - t0

            for _ in range(WARMUP):
                await via_engine()
                await via_gateway()
            d, g = [], []
            for _ in range(PAIRS):
                d.append(await via_engine())
                g.append(await via_gateway())
    finally:
        await rds.delete(stream, f"loom:pool:{model}")
        await rds.aclose()
    ms = 1000.0
    reg = app.state.gateway.metrics.registry

    def mean_phase(phase: str) -> float:
        total = reg.get_sample_value("loom_gateway_overhead_seconds_sum", {"phase": phase})
        n = reg.get_sample_value("loom_gateway_overhead_seconds_count", {"phase": phase})
        return float(total or 0) / float(n or 1) * ms

    return {
        "pre_dispatch_mean_ms": mean_phase("pre_dispatch"),
        "settle_mean_ms": mean_phase("settle"),
        "direct_p50_ms": statistics.median(d) * ms,
        "gateway_p50_ms": statistics.median(g) * ms,
        "overhead_p50_ms": (statistics.median(g) - statistics.median(d)) * ms,
        "overhead_p90_ms": (statistics.quantiles(g, n=10)[-1] - statistics.quantiles(d, n=10)[-1])
        * ms,
    }


def test_gateway_p50_overhead(pg_url, certs, test_registry, record_property):
    redis_url = _redis_url()
    attempts: list[float] = []
    for attempt in range(1, ATTEMPTS + 1):
        result = uvloop.run(_bench(pg_url, redis_url, certs, test_registry))
        attempts.append(result["overhead_p50_ms"])
        print(
            "\ngateway overhead (attempt {n}): p50 {overhead_p50_ms:.2f} ms, "
            "p90 {overhead_p90_ms:.2f} ms "
            "(direct p50 {direct_p50_ms:.2f} ms, through the gateway {gateway_p50_ms:.2f} ms; "
            "in the gateway: pre-dispatch {pre_dispatch_mean_ms:.2f} ms, "
            "settle {settle_mean_ms:.2f} ms mean)".format(n=attempt, **result)
        )
        if result["overhead_p50_ms"] < TARGET_P50_MS:
            break
    for k, v in result.items():
        record_property(k, round(v, 3))
    record_property("attempts_overhead_p50_ms", [round(a, 3) for a in attempts])
    assert min(attempts) < TARGET_P50_MS, (
        f"p50 overhead {', '.join(f'{a:.2f}' for a in attempts)} ms in {len(attempts)} "
        f"attempts, all above {TARGET_P50_MS} ms"
    )
