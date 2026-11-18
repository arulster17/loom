"""B3 after the second review (docs/m3-review-fixes-spec.md, "Billing", B3): "When the
deferred check fails, the service exits 2 through its normal shutdown path (the gateway
drains and settles its open streams as at SIGTERM), not `os._exit`."

A production-shaped gateway (`python -m loom_gateway`, `LOOM_ENV=prod`, every static
production problem cleared, `LOOM_SHUTDOWN_GRACE_S=1`) starts with Postgres unreachable
(a `QuietPgProxy` that refuses connections), so its role check is deferred to the gate.
Redis is reachable (`TlsRedis` in front of the test Redis). The tenant's key and org are
already cached in Redis (as a gateway that served them earlier would have left them:
`put_cached` and `Limiter.load_from_postgres`), so a request on that key needs no
Postgres: a stream starts from a fake engine (gateway_testkit's `FakeEngine` behind
mTLS, 50 ms per token, far longer than the test) while the gate is still closed.

Mid-stream the proxy opens and the deferred check fails:
- `superuser`: the configured login is a (fresh) superuser, a role problem;
- `wrong_password`: the server rejects the configured password, a misconfiguration.

The gateway must exit 2, and the stream must be settled first: its usage event written
to the usage stream (or spooled to `LOOM_SETTLE_SPOOL_PATH`), its reservation released
(org `rsum` 0, no reservation left in `rsv:exp`) and the key's in-flight count back to
0, as at SIGTERM, instead of waiting for the reservation's TTL."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import psycopg
import pytest
import yaml
from platform_m3_review2_kit import (
    PG_WARNING,
    REPO_ROOT,
    QuietPgProxy,
    free_port,
    superuser_problem,
)
from platform_m3_review3_kit import (
    GATEWAY,
    PASSWORD_FAILED,
    RedisFront,
    fresh_login,
    gateway_env,
    redis_front,
    wrong_password,
    wrong_password_rejected,
)
from platform_m3_review_proxy import Service, proxied_url
from platform_m3_roles import admin_info

from loom_bench.registry import Registry, load_registry
from loom_platform.apikeys import hash_api_key
from loom_platform.auth import auth_cache_key, load_auth_record, put_cached
from loom_platform.limiter import Limiter, api_key_key, org_key, org_prefix
from loom_platform.pool import pool_key

sys.path.insert(0, str(REPO_ROOT / "services" / "gateway" / "tests"))
from gateway_testkit import (
    Behavior,
    FakeEngine,
    make_certs,
    make_tenant,
    mtls_server_config,
    replica,
    serve,
    set_ready,
    usage_events,
)

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

# The waits add up to 95 s, under the test's 115 s timeout with setup and cleanup, so a
# failing wait ends in its assertion, not in the timeout.
WARN_S = 20  # the unreachable-at-start warning
HEALTH_S = 15
STREAM_S = 35  # the whole stream (the engine alone would take 100 s): gate retry (a few
# s) + the failed check + 1 s grace
EXIT_S = 25  # after the stream ended: the rest of the shutdown
CHUNKS_BEFORE_OPEN = 3


@pytest.fixture(scope="module")
def info() -> dict[str, str]:
    return admin_info()


@pytest.fixture
def rf(tmp_path: Path) -> Iterator[RedisFront]:
    with redis_front(tmp_path / "redis") as front:
        yield front


@pytest.fixture
def proxy(info) -> Iterator[QuietPgProxy]:
    p = QuietPgProxy(info.get("host", "localhost"), int(info.get("port", "5432")))
    try:
        yield p
    finally:
        p.close()


def _models_yaml(directory: Path) -> tuple[Path, str]:
    """config/models.yaml's qwen3-8b under an id unique to this test, so its pool hash
    (`loom:pool:<model>`) is never shared with another test on the same Redis."""
    base = load_registry()
    model = f"t{uuid.uuid4().hex[:8]}-qwen3-8b"
    reg = Registry(models=[base.get("qwen3-8b").model_copy(update={"id": model})])
    path = directory / "models.yaml"
    path.write_text(yaml.safe_dump(reg.model_dump(mode="json")))
    return path, model


async def _cache_tenant(pg_url: str, redis_client, stream: str, tenant) -> bytes:
    """What an earlier request on this key left in Redis: the auth record and the org's
    and key's balance state."""
    key_hash = hash_api_key(tenant.key)
    async with await psycopg.AsyncConnection.connect(pg_url) as conn:
        record = await load_auth_record(conn, key_hash)
        assert record is not None
        await put_cached(redis_client, key_hash, record, 600)
        await Limiter(redis_client, usage_stream=stream).load_from_postgres(
            conn, org_id=tenant.org_id, key_id=tenant.key_id
        )
    return key_hash


def _spooled(spool: Path, org: str) -> bool:
    return spool.exists() and org in spool.read_text()


async def _wait_healthy(base: str, svc: Service, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline and svc.returncode is None:
            try:
                if (await c.get(f"{base}/health", timeout=2)).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
    return False


async def _stream_until_cut(base: str, key: str, model: str, proxy: QuietPgProxy) -> dict:
    """Start a long stream; once a few content chunks arrived, open the proxy (the
    deferred check now fails); read until the gateway ends the stream or drops it."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 2000,  # 100 s of tokens at 50 ms each
        "stream": True,
    }
    seen = {"status": None, "chunks": 0, "opened": False, "error": None, "last": None}
    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(60, connect=10)) as client,
            client.stream(
                "POST",
                f"{base}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}"},
            ) as resp,
        ):
            seen["status"] = resp.status_code
            if resp.status_code != 200:
                seen["error"] = (await resp.aread()).decode(errors="replace")[:500]
                return seen
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                seen["last"] = data
                if data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                    seen["chunks"] += 1
                if seen["chunks"] >= CHUNKS_BEFORE_OPEN and not seen["opened"]:
                    seen["opened"] = True
                    proxy.open()
    except httpx.HTTPError as e:  # dropped mid-stream (the process died)
        seen["error"] = repr(e)
    return seen


@pytest.mark.timeout(115)
@pytest.mark.parametrize("failure", ["superuser", "wrong_password"])
async def test_item5_failed_deferred_check_settles_open_stream_then_exits_2(
    info, pg_url, redis_client, rf, proxy, tmp_path, failure
):
    """done-when B3 (clean exit on a failed deferred check, gateway): a stream in flight
    on a Redis-cached key while the gate is closed is settled (usage event written or
    spooled, reservation and in-flight count released) when the check later fails, and
    the gateway exits 2 naming the problem."""
    (tmp_path / "pod").mkdir()
    pod_certs = make_certs(tmp_path / "pod")
    models_yaml, model = _models_yaml(tmp_path)
    stream = rf.streams["LOOM_USAGE_STREAM"]
    spool = tmp_path / "spool.jsonl"
    dbname = psycopg.conninfo.conninfo_to_dict(pg_url)["dbname"]
    assert isinstance(dbname, str)
    tenant = await make_tenant(pg_url, credit_micros=50_000_000)
    key_hash = await _cache_tenant(pg_url, redis_client, stream, tenant)
    engine = FakeEngine("e1", Behavior(tokens="max", token_delay_s=0.05))
    try:
        with fresh_login(info, superuser=failure == "superuser") as login:
            if failure == "wrong_password":
                wrong_password_rejected(info, login)
            password = login.password if failure == "superuser" else wrong_password(login)
            url = proxied_url(proxy, dbname=dbname, user=login.name, password=password)
            async with serve(engine.app(), **mtls_server_config(pod_certs)) as engine_port:
                await set_ready(redis_client, model, [replica("e1", engine_port)])
                port = free_port()
                env = gateway_env(
                    rf,
                    url,
                    port=port,
                    spool=spool,
                    pod_ca=pod_certs.ca,
                    client_cert=pod_certs.client_cert,
                    client_key=pod_certs.client_key,
                    models_yaml=models_yaml,
                    usage_stream=stream,
                    grace_s="1",
                )
                svc = Service(GATEWAY, env)
                try:
                    await _check_settled_exit(
                        svc,
                        port,
                        tenant,
                        model,
                        proxy,
                        redis_client,
                        stream,
                        spool,
                        failure,
                        login.name,
                    )
                finally:
                    svc.stop()  # before the role is dropped
    finally:
        keys = [k async for k in redis_client.scan_iter(match=f"{org_prefix(tenant.org_id)}:*")]
        await redis_client.delete(pool_key(model), auth_cache_key(key_hash), *keys)


async def _check_settled_exit(
    svc: Service,
    port: int,
    tenant,
    model: str,
    proxy: QuietPgProxy,
    redis_client,
    stream: str,
    spool: Path,
    failure: str,
    login: str,
) -> None:
    base = f"http://127.0.0.1:{port}"
    warned = await asyncio.to_thread(svc.wait_for, PG_WARNING, WARN_S)
    assert svc.returncode is None and warned, (
        f"the gateway did not start with Postgres unreachable:\n{svc.output}"
    )
    assert await _wait_healthy(base, svc, HEALTH_S), (
        f"the gateway never answered /health:\n{svc.output}"
    )

    try:
        seen = await asyncio.wait_for(_stream_until_cut(base, tenant.key, model, proxy), STREAM_S)
    except TimeoutError:
        pytest.fail(
            f"the stream was still open {STREAM_S}s after it started ({failure}): once "
            "Postgres was reachable the deferred check should have failed and the gateway "
            f"cut the stream and exited 2 (exit code now {svc.returncode}):\n{svc.output}"
        )
    assert seen["status"] == 200, (
        f"the stream on a Redis-cached key did not start while the gate was closed: "
        f"{seen}\n{svc.output}"
    )
    assert seen["opened"], f"the stream ended before the proxy opened: {seen}"
    code = await asyncio.to_thread(svc.wait_exit, EXIT_S)
    out = svc.output
    assert code == 2, (
        f"the gateway exited {code} after the deferred check failed ({failure}), not 2:\n{out}"
    )
    if failure == "superuser":
        assert superuser_problem(out, login), out
    else:
        assert PASSWORD_FAILED.search(out), out

    events = [e for e in await usage_events(redis_client, stream) if e.org_id == tenant.org_id]
    spooled = await asyncio.to_thread(_spooled, spool, str(tenant.org_id))
    assert events or spooled, (
        "the open stream was not settled before the exit (no usage event in the stream, "
        f"nothing spooled): the process did not go through its normal shutdown. "
        f"Stream: {seen}\n{out}"
    )
    rsum = int(await redis_client.get(org_key(tenant.org_id, "rsum")) or 0)
    left = await redis_client.zcard(org_key(tenant.org_id, "rsv:exp"))
    assert (rsum, left) == (0, 0), (
        f"the reservation was not released at the exit (rsum {rsum}, {left} reservations "
        f"left): it would wait for its TTL.\n{out}"
    )
    inflight = api_key_key(tenant.org_id, tenant.key_id, "inflight")
    assert int(await redis_client.get(inflight) or 0) == 0, (
        f"the key's in-flight count was not released at the exit:\n{out}"
    )
