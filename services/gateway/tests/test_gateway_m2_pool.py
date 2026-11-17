"""The gateway's M2 changes (docs/phase1-m2-spec.md §7, §8, §13 item 3): cold-start
estimates from `config/serving.yaml`, a wake on `unavailable`, and the pool `reason` on
503s in logs and `loom_gateway_model_unavailable_total{model,state,reason}`."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from gateway_testkit import make_tenant, replica, serve
from prometheus_client.parser import text_string_to_metric_families

from loom_bench.registry import Registry
from loom_gateway import settings as gateway_settings
from loom_gateway.app import create_app
from loom_gateway.logs import JsonFormatter
from loom_gateway.metrics import GatewayMetrics
from loom_gateway.routing import Router
from loom_gateway.settings import DEFAULT_COLD_START_S, GatewaySettings
from loom_platform.pool import WAKE_TTL_S, PoolState, pool_key, set_pool, wake_key
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV, wait_for_message

MESSAGES = [{"role": "user", "content": "Say hello."}]
SERVING_ENV = "LOOM_SERVING_YAML"
SMALL = "qwen3-8b"
BIG = "llama-3.3-70b-instruct-fp8"

SERVING: dict[str, Any] = {
    "version": 1,
    "provider": "fake",
    "spend": {"monthly_cap_usd": 100, "alert_fractions": [0.5, 0.8, 1.0], "max_price_ratio": 1.25},
    "placement": {
        "data_center_prefixes": ["US-"],
        "allowed_cuda_versions": ["13.0"],
        "container_disk_gb": {SMALL: 80, BIG: 120},
    },
    "network": {"gateway_egress_cidrs": [], "proxy_port": 8443},
    "lifecycle": {
        "reconcile_every_s": 15,
        "lease_ttl_s": 3600,
        "lease_renew_every_s": 300,
        "drain_max_s": 600,
        "server_cert_days": 7,
        "server_cert_renew_before_s": 172800,
        "max_replacements_per_hour": 3,
    },
    "health": {
        "pod_state_every_s": 30,
        "proxy_health_every_s": 10,
        "proxy_health_failures": 3,
        "unhealthy_grace_s": 120,
        "progress_stall_s": 60,
        "bootstrap_stall_s": 600,
        "bootstrap_timeout_s": 1800,
    },
    "capacity": {"backoff_s": [30, 60, 120, 240], "unavailable_retry_after_s": 300},
    "models": {
        SMALL: {
            "min_replicas": 0,
            "max_replicas": 1,
            "idle_stop_s": 1800,
            "cold_start_estimate_s": 123,
            "engine_env": {},
        },
        BIG: {
            "min_replicas": 0,
            "max_replicas": 1,
            "idle_stop_s": 1800,
            "cold_start_estimate_s": 456,
            "engine_env": {"NCCL_P2P_LEVEL": "NVL"},
        },
    },
}


def _router() -> Router:
    return Router(None, GatewaySettings(), GatewayMetrics())  # type: ignore[arg-type]


# ------------------------------------------------------------------ cold-start estimates


def test_cold_start_estimates_come_from_serving_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(SERVING), encoding="utf-8")
    monkeypatch.setenv(SERVING_ENV, str(path))
    r = _router()
    assert r.cold_start_estimate(SMALL) == 123
    assert r.cold_start_estimate(BIG) == 456
    # Not configured there (qwen3-8b-fp8 had a gateway constant in M1): the default.
    assert r.cold_start_estimate("qwen3-8b-fp8") == DEFAULT_COLD_START_S
    assert r.cold_start_estimate("unknown") == DEFAULT_COLD_START_S


def test_a_model_left_out_of_serving_yaml_gets_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = copy.deepcopy(SERVING)
    del doc["models"][BIG]
    del doc["placement"]["container_disk_gb"][BIG]
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setenv(SERVING_ENV, str(path))
    r = _router()
    assert r.cold_start_estimate(SMALL) == 123
    assert r.cold_start_estimate(BIG) == DEFAULT_COLD_START_S


def test_without_serving_yaml_every_model_gets_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§8: "When the file is absent (gateway tests), the fallback is DEFAULT_COLD_START_S"
    (in M1 the 8B and 70B had their own gateway constants, 240 and 450)."""
    monkeypatch.setenv(SERVING_ENV, str(tmp_path / "no-such-serving.yaml"))
    r = _router()
    for model in (SMALL, BIG, "unknown"):
        assert r.cold_start_estimate(model) == DEFAULT_COLD_START_S


def test_gateway_cold_start_constants_are_gone() -> None:
    assert not hasattr(gateway_settings, "COLD_START_ESTIMATE_S")


# ------------------------------------------------------------------ 503s with the stack


async def _post(stack: Any, key: str, model: str) -> httpx.Response:
    async with httpx.AsyncClient() as c:
        return await c.post(
            f"{stack.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": MESSAGES},
            timeout=30,
        )


def _err(r: httpx.Response) -> dict[str, Any]:
    body = r.json()
    assert set(body) == {"error"}, body
    return body["error"]


async def _set_unavailable(redis: Any, model: str, reason: str | None) -> None:
    if reason is None:
        # Exactly what M1 code writes: no `reason` field at all.
        await redis.delete(pool_key(model))
        await redis.hset(
            pool_key(model),
            mapping={
                "state": "unavailable",
                "retry_after_s": 300,
                "replicas": "[]",
                "updated_at_ms": int(time.time() * 1000),
            },
        )
    else:
        await set_pool(redis, PoolState(model, "unavailable", 300, reason=reason))


@pytest.mark.postgres
@pytest.mark.redis
@pytest.mark.parametrize("reason", ["no_capacity", "spend_cap", None])
async def test_unavailable_wakes_once_per_ttl(stack, redis_client, reason: str | None) -> None:
    """§7: `check_routable` calls `request_wake` on `unavailable` too, whatever the reason
    (the manager decides which reasons a wake can clear). The client still gets 503
    "no GPU capacity" with the pool's Retry-After, and nothing is reserved or billed."""
    t = await stack.tenant()
    big = stack.big
    await redis_client.delete(wake_key(big))
    await _set_unavailable(redis_client, big, reason)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("loom:pool:wake")
    await pubsub.get_message(timeout=1)  # subscribe confirmation
    try:
        r = await _post(stack, t.key, big)
        assert r.status_code == 503 and r.headers["retry-after"] == "300"
        assert _err(r)["code"] == "model_unavailable"
        assert "no GPU capacity" in _err(r)["message"]
        assert await redis_client.exists(wake_key(big))
        assert 0 < await redis_client.ttl(wake_key(big)) <= WAKE_TTL_S
        assert await wait_for_message(pubsub, big)

        # Within the wake's TTL: another 503, no second publish.
        r = await _post(stack, t.key, big)
        assert r.status_code == 503 and "no GPU capacity" in _err(r)["message"]
        assert not await wait_for_message(pubsub, big, within_s=0.5)

        # The wake key gone (its TTL ran out): the next 503 publishes again.
        await redis_client.delete(wake_key(big))
        r = await _post(stack, t.key, big)
        assert r.status_code == 503
        assert await redis_client.exists(wake_key(big))
        assert await wait_for_message(pubsub, big)
    finally:
        await pubsub.aclose()

    assert stack.engine.requests == []
    assert await stack.events() == []
    assert await stack.reserved(t) == (0, 0)


@pytest.mark.postgres
@pytest.mark.redis
async def test_unavailable_wake_needs_balance(stack, redis_client) -> None:
    """Rate-limited like any 503: the wake comes after reserve, so an org without the
    balance gets 402 and wakes nothing."""
    big = stack.big
    await redis_client.delete(wake_key(big))
    await set_pool(redis_client, PoolState(big, "unavailable", 300, reason="no_capacity"))
    broke = await stack.tenant(credit_micros=1)
    r = await _post(stack, broke.key, big)
    assert r.status_code == 402 and _err(r)["code"] == "insufficient_balance"
    assert not await redis_client.exists(wake_key(big))


def _unavailable_counts(stack: Any, model: str) -> dict[tuple[str, str], float]:
    """(state, reason) -> value of loom_gateway_model_unavailable_total for `model`."""
    out: dict[tuple[str, str], float] = {}
    text = stack.gateway.metrics.render().decode()
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if s.name == "loom_gateway_model_unavailable_total" and s.labels.get("model") == model:
                out[(s.labels["state"], s.labels["reason"])] = s.value
    return out


@pytest.mark.postgres
@pytest.mark.redis
async def test_503s_count_and_log_the_pool_reason(stack, redis_client, caplog) -> None:
    t = await stack.tenant()
    big = stack.big
    with caplog.at_level(logging.DEBUG):
        await set_pool(redis_client, PoolState(big, "unavailable", 300, reason="no_capacity"))
        for _ in range(2):
            assert (await _post(stack, t.key, big)).status_code == 503
        await set_pool(redis_client, PoolState(big, "unavailable", 300, reason="spend_cap"))
        assert (await _post(stack, t.key, big)).status_code == 503
        await set_pool(redis_client, PoolState(big, "warming", 37))
        assert (await _post(stack, t.key, big)).status_code == 503
        await redis_client.delete(pool_key(big), wake_key(big))
        assert (await _post(stack, t.key, big)).status_code == 503

    counts = _unavailable_counts(stack, big)
    assert counts[("unavailable", "no_capacity")] == 2
    assert counts[("unavailable", "spend_cap")] == 1
    assert sum(v for (state, _), v in counts.items() if state == "warming") == 1
    assert sum(v for (state, _), v in counts.items() if state == "cold") == 1
    assert sum(counts.values()) == 5

    fmt = JsonFormatter()
    lines = [fmt.format(r) for r in caplog.records if r.name.startswith("loom_gateway")]
    for reason in ("no_capacity", "spend_cap"):
        hits = [ln for ln in lines if reason in ln]
        assert hits, f"no gateway log line names the pool reason {reason!r}"
        assert any(big in ln for ln in hits)
        for ln in hits:
            json.loads(ln)  # still one JSON object per line


# ------------------------------------------------------------------ through create_app


def _serving_for(models: dict[str, int]) -> dict[str, Any]:
    """SERVING with `models` (id -> cold_start_estimate_s) as its model entries."""
    doc = copy.deepcopy(SERVING)
    doc["models"] = {
        m: {
            "min_replicas": 0,
            "max_replicas": 1,
            "idle_stop_s": 1800,
            "cold_start_estimate_s": s,
            "engine_env": {},
        }
        for m, s in models.items()
    }
    doc["placement"]["container_disk_gb"] = {m: 80 for m in models}
    return doc


@contextlib.asynccontextmanager
async def _app(
    *,
    pg_url: str,
    registry: Registry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    serving: dict[str, Any] | None,
    cold_start_s: dict[str, int] | None = None,
) -> AsyncIterator[str]:
    """A gateway built by `create_app` with `LOOM_SERVING_YAML` naming `serving` (None: a
    file that does not exist). The test registry is also written out as
    `LOOM_MODELS_YAML`, so a loader that checks the registry sees these model ids."""
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(yaml.safe_dump(registry.model_dump(mode="json")), encoding="utf-8")
    monkeypatch.setenv("LOOM_MODELS_YAML", str(models_yaml))
    serving_yaml = tmp_path / ("serving.yaml" if serving is not None else "missing.yaml")
    if serving is not None:
        serving_yaml.write_text(yaml.safe_dump(serving), encoding="utf-8")
    monkeypatch.setenv(SERVING_ENV, str(serving_yaml))
    settings = GatewaySettings(
        database_url=pg_url,
        redis_url=os.environ.get(REDIS_ENV) or DEFAULT_REDIS,
        usage_stream=f"loom:test:usage:{uuid.uuid4().hex}",
        pool_cache_ttl_s=0,
        log_level="WARNING",
        env="dev",
    )
    kw: dict[str, Any] = {} if cold_start_s is None else {"cold_start_s": cold_start_s}
    app = create_app(settings, registry=registry, **kw)
    async with serve(app) as port:
        yield f"http://127.0.0.1:{port}"


async def _cold_503(url: str, key: str, model: str) -> httpx.Response:
    async with httpx.AsyncClient() as c:
        r = await c.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": MESSAGES},
            timeout=30,
        )
    assert r.status_code == 503 and _err(r)["code"] == "model_unavailable", r.text
    assert "starting up" in _err(r)["message"]
    return r


@pytest.fixture
async def cold_models(test_registry: Registry, redis_client) -> AsyncIterator[list[str]]:
    """The test registry's priced models, with no pool hash (cold), cleaned up after."""
    models = [m.id for m in test_registry.models if m.pricing is not None]
    keys = [k for m in models for k in (pool_key(m), wake_key(m))]
    await redis_client.delete(*keys)
    yield models
    await redis_client.delete(*keys)


@pytest.mark.postgres
@pytest.mark.redis
async def test_app_cold_start_retry_after_comes_from_serving_yaml(
    pg_url, test_registry, cold_models, tmp_path, monkeypatch
) -> None:
    """§8 item 1 through `create_app` with no `cold_start_s`: a cold model's 503 carries
    the serving.yaml estimate."""
    small, big = cold_models
    serving = _serving_for({small: 123, big: 456})
    t = await make_tenant(pg_url)
    async with _app(
        pg_url=pg_url,
        registry=test_registry,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        serving=serving,
    ) as url:
        assert (await _cold_503(url, t.key, small)).headers["retry-after"] == "123"
        assert (await _cold_503(url, t.key, big)).headers["retry-after"] == "456"


@pytest.mark.postgres
@pytest.mark.redis
async def test_app_cold_start_override_wins_over_serving_yaml(
    pg_url, test_registry, cold_models, tmp_path, monkeypatch
) -> None:
    """`create_app(cold_start_s=...)` overrides stay (§4) and beat serving.yaml for the
    same model; a model it leaves out still gets serving.yaml's value."""
    small, big = cold_models
    t = await make_tenant(pg_url)
    async with _app(
        pg_url=pg_url,
        registry=test_registry,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        serving=_serving_for({small: 123, big: 456}),
        cold_start_s={small: 77},
    ) as url:
        assert (await _cold_503(url, t.key, small)).headers["retry-after"] == "77"
        assert (await _cold_503(url, t.key, big)).headers["retry-after"] == "456"


@pytest.mark.postgres
@pytest.mark.redis
async def test_app_starts_and_answers_without_serving_yaml(
    pg_url, test_registry, cold_models, tmp_path, monkeypatch
) -> None:
    """§4: `LOOM_SERVING_YAML` naming a missing file: the gateway still starts, and every
    model gets DEFAULT_COLD_START_S."""
    small, big = cold_models
    t = await make_tenant(pg_url)
    async with _app(
        pg_url=pg_url,
        registry=test_registry,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        serving=None,
    ) as url:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{url}/v1/models", headers={"Authorization": f"Bearer {t.key}"})
        assert r.status_code == 200
        for model in (small, big):
            r = await _cold_503(url, t.key, model)
            assert r.headers["retry-after"] == str(DEFAULT_COLD_START_S)


@pytest.mark.postgres
@pytest.mark.redis
@pytest.mark.parametrize("leftover", ["no_capacity", "spend_cap", "crash_loop"])
async def test_ready_pool_with_a_leftover_reason_still_routes(
    stack, redis_client, leftover: str
) -> None:
    """§7: the gateway ignores `reason` except for logs and metrics, so a `ready` hash
    with replicas routes whatever reason it carries."""
    t = await stack.tenant()
    rep = replica("e1", stack.ports["e1"], tls=stack.tls)
    await set_pool(redis_client, PoolState(stack.big, "ready", 0, (rep,), reason=leftover))
    r = await _post(stack, t.key, stack.big)
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"]
    assert len(stack.engine.requests) == 1
    assert _unavailable_counts(stack, stack.big) == {}
