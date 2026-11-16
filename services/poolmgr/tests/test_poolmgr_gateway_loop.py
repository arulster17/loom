"""The gateway with the manager in the loop (docs/phase1-m2-spec.md §13 items 20-22; §1
"What M1 already covers"; §7; §8 "Nothing else changes").

A real gateway app (`loom_gateway.app.create_app`, real Redis and Postgres, served by
uvicorn on a loopback port) routes to pods that a real `PoolManager` created and
published in `loom:pool:<model>`. The pods are `poolmgr_fakes`' fake serving pods: a real
`loom-podproxy` (mTLS with the certificate the manager issued, the token it delivered) in
front of the bench mock engine. Nothing here writes the pool hash but the manager: the
gateway sees exactly what the manager publishes.

The gateway's client certificate and pod CA are the ones `Pki.generate` writes next to
the manager's (`loom-gateway.{crt,key}`, `ca.crt`), as `loom-poolmgr issue-ca` does in
dev. Each test has model ids of its own (`unique_model_id`), so parallel tests never share
a pool hash, a wake key or a usage stream.

These need Linux (the gateway imports `fcntl`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import psycopg
import pytest
import uvicorn
from poolmgr_fakes import (
    FakePod,
    FakeServing,
    PodBehaviour,
    bring_up,
    deployment,
    env_fixture,  # noqa: F401
    metric_value,
    proxy_get,
    serving_config,
    step,
    tick,
    unique_model_id,
    wait_until,
)
from redis.asyncio import Redis

from loom_bench.registry import Registry, load_registry
from loom_gateway.app import Gateway, create_app
from loom_gateway.settings import GatewaySettings
from loom_platform.ledger import approve_org
from loom_platform.limiter import org_key
from loom_platform.pool import PoolState, Replica, pool_key, wake_key
from loom_platform.serving import ServingConfig
from loom_platform.testing import (  # noqa: F401
    DEFAULT_REDIS,
    REDIS_ENV,
    create_api_key,
    create_org,
    create_user,
    pg_url,
    redis_client,
)
from loom_platform.usage import UsageEvent
from loom_poolmgr.manager import PoolManager
from loom_poolmgr.pki import GATEWAY_CN

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

MESSAGES = [{"role": "user", "content": "Count slowly."}]
# The gateway's own fallback estimate for these models: never what a 503 from a pool the
# manager wrote may carry (proves Retry-After comes from the pool hash).
GATEWAY_ESTIMATE_S = 999


# ------------------------------------------------------------------ gateway harness


@contextlib.asynccontextmanager
async def serve(app: Any) -> AsyncIterator[int]:
    """Run an ASGI app with uvicorn in this event loop on a free loopback port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            lifespan="on",
            log_level="warning",
            access_log=False,
            timeout_graceful_shutdown=5,
        )
    )
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        while not server.started:
            if task.done():
                task.result()
                raise RuntimeError("the gateway exited during startup")
            await asyncio.sleep(0.005)
        yield port
    finally:
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(task, 10)
        sock.close()


@dataclass(frozen=True)
class Tenant:
    org_id: uuid.UUID
    key_id: uuid.UUID
    key: str


async def make_tenant(database_url: str, *, credit_micros: int = 5_000_000) -> Tenant:
    """An approved org with `credit_micros` of credit and one API key."""
    conn = await psycopg.AsyncConnection.connect(database_url)
    try:
        admin = await create_user(conn, role="admin")
        org = await create_org(conn, owner=admin)
        await approve_org(conn, org, admin, credit_micros)
        key_id, key = await create_api_key(conn, org)
        await conn.commit()
    finally:
        await conn.close()
    return Tenant(org, key_id, key.plaintext)


@dataclass
class GatewayUnderTest:
    url: str
    app: Any
    redis: Redis
    stream: str
    database_url: str
    models: list[str]
    tenants: list[Tenant] = field(default_factory=list)

    @property
    def gateway(self) -> Gateway:
        gw: Gateway = self.app.state.gateway
        return gw

    async def tenant(self) -> Tenant:
        t = await make_tenant(self.database_url)
        self.tenants.append(t)
        return t

    async def events(self) -> list[UsageEvent]:
        entries = await self.redis.xrange(self.stream)
        return [UsageEvent.from_stream_fields(fields) for _, fields in entries]

    async def wait_events(self, count: int, timeout_s: float = 10.0) -> list[UsageEvent]:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            events = await self.events()
            if len(events) >= count or asyncio.get_running_loop().time() > deadline:
                return events
            await asyncio.sleep(0.02)

    async def reserved(self, t: Tenant) -> tuple[int, int]:
        """(org reservation sum, org in-flight reservations)."""
        rsum = await self.redis.get(org_key(t.org_id, "rsum"))
        n = await self.redis.zcard(org_key(t.org_id, "rsv:exp"))
        return int(rsum or 0), int(n)

    async def usage(self, t: Tenant) -> int:
        return int(await self.redis.get(org_key(t.org_id, "usage")) or 0)

    def metrics(self) -> str:
        return self.gateway.metrics.render().decode()

    async def post(self, t: Tenant, model: str, **body: Any) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60, trust_env=False) as c:
            return await c.post(
                f"{self.url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {t.key}"},
                json={"model": model, "messages": MESSAGES, **body},
            )


GatewayFactory = Callable[..., Awaitable[GatewayUnderTest]]


@pytest.fixture
async def make_gateway(
    env: FakeServing,
    pg_url: str,  # noqa: F811
    redis_client: Redis,  # noqa: F811
) -> AsyncIterator[GatewayFactory]:
    """Build a real gateway app serving `models` (copies of the registry's qwen3-8b row
    under the test's own ids), trusting the manager's pod CA with the dev gateway client
    certificate `Pki.generate` wrote."""
    base = load_registry().get("qwen3-8b")
    pki_dir = env.root / "pki"
    async with contextlib.AsyncExitStack() as es:
        made: list[GatewayUnderTest] = []

        async def build(models: list[str]) -> GatewayUnderTest:
            registry = Registry(models=[base.model_copy(update={"id": m}) for m in models])
            settings = GatewaySettings(
                database_url=pg_url,
                redis_url=os.environ.get(REDIS_ENV) or DEFAULT_REDIS,
                usage_stream=f"loom:test:usage:{uuid.uuid4().hex}",
                pool_cache_ttl_s=0,
                log_level="WARNING",
                env="dev",
                gateway_client_cert=str(pki_dir / f"{GATEWAY_CN}.crt"),
                gateway_client_key=str(pki_dir / f"{GATEWAY_CN}.key"),
                pod_ca_cert=str(env.pki.ca_cert_path),
            )
            app = create_app(
                settings,
                registry=registry,
                cold_start_s={m: GATEWAY_ESTIMATE_S for m in models},
            )
            port = await es.enter_async_context(serve(app))
            gw = GatewayUnderTest(
                url=f"http://127.0.0.1:{port}",
                app=app,
                redis=redis_client,
                stream=settings.usage_stream,
                database_url=pg_url,
                models=list(models),
            )
            made.append(gw)
            return gw

        try:
            yield build
        finally:
            await es.aclose()
            for gw in made:
                keys = [gw.stream]
                for m in gw.models:
                    keys += [pool_key(m), wake_key(m)]
                await redis_client.delete(*keys)
                for t in gw.tenants:
                    org_keys = [k async for k in redis_client.scan_iter(f"loom:org:{t.org_id}:*")]
                    if org_keys:
                        await redis_client.delete(*org_keys)


def loop_manager(env: FakeServing, config: ServingConfig, gw: GatewayUnderTest) -> PoolManager:
    """`env.manager(config)`, reading the gateway's usage stream (its demand signal, as
    `LOOM_USAGE_STREAM` wires it in compose)."""
    manager = PoolManager(
        config,
        deployments={m: deployment(config, m) for m in config.models},
        provider=env.provider(config),
        pod_exec=env.pods,
        redis=env.redis,
        database_url=env.database_url,
        leases=env.leases,
        pki=env.pki,
        ssh=env.ssh,
        bundle=env.bundle,
        clock=env.clock,
        sleep=env.clock.sleep,
        admin_queue=env.admin_queue,
        usage_stream=gw.stream,
    )
    env.managers.append(manager)
    return manager


# ------------------------------------------------------------------ helpers


def sse(text: str) -> list[dict[str, Any] | str]:
    """The `data:` payloads of an SSE body ("[DONE]" kept as a string)."""
    out: list[dict[str, Any] | str] = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


def content_of(events: list[dict[str, Any] | str]) -> str:
    return "".join(
        (e["choices"][0].get("delta") or {}).get("content") or ""
        for e in events
        if isinstance(e, dict) and e.get("choices")
    )


def error_of(r: httpx.Response) -> dict[str, Any]:
    body = r.json()
    assert set(body) == {"error"}, body
    err: dict[str, Any] = body["error"]
    return err


def retries(gw: GatewayUnderTest, model: str) -> float:
    prefix = f'loom_gateway_upstream_retries_total{{model="{model}"}} '
    for line in gw.metrics().splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix))
    return 0.0


async def row_of(env: FakeServing, pod: FakePod) -> dict[str, Any]:
    (row,) = [r for r in await env.rows(pod.model) if r["pod_name"] == pod.name]
    return row


async def nothing_billed(gw: GatewayUnderTest, t: Tenant) -> None:
    """No reservation left behind and no charge for this tenant."""
    assert await gw.reserved(t) == (0, 0)
    assert await gw.usage(t) == 0
    assert all(e.cost_micros == 0 for e in await gw.events() if e.org_id == t.org_id)


async def kill_engine(env: FakeServing, pod_id: str) -> None:
    """Kill a pod's engine (as `kill -9`) and wait until it is gone: its proxy then
    fails every request before the first byte."""
    engine = env.pods.pod(pod_id).engine
    assert engine is not None
    env.pods.kill_engine(pod_id)
    await wait_until(lambda: not engine.running)


async def engine_running(client: httpx.AsyncClient, replica: Replica) -> float:
    """`vllm:num_requests_running` of a replica's engine, read through its proxy."""
    r = await proxy_get(client, replica, "/metrics")
    assert r.status_code == 200, r.text
    return metric_value(r.text, "vllm:num_requests_running")


class GatewayStream:
    """A long streaming chat completion through the gateway, held open in the background
    (an open request on one replica) until `close()`."""

    def __init__(self, gw: GatewayUnderTest, t: Tenant, model: str) -> None:
        self.gw, self.t, self.model = gw, t, model
        self.first = asyncio.Event()
        self.stop = asyncio.Event()
        self.task: asyncio.Task[None] | None = None

    async def start(self, timeout_s: float = 15.0) -> GatewayStream:
        self.task = asyncio.ensure_future(self._run())
        waiter = asyncio.ensure_future(self.first.wait())
        await asyncio.wait({waiter, self.task}, timeout=timeout_s, return_when="FIRST_COMPLETED")
        waiter.cancel()
        if self.task.done():
            self.task.result()
        assert self.first.is_set(), "the busy stream sent no token"
        return self

    async def close(self) -> None:
        self.stop.set()
        if self.task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.task, 15)

    async def _run(self) -> None:
        body = {
            "model": self.model,
            "messages": MESSAGES,
            "max_tokens": 2000,
            "min_tokens": 1500,  # ~30 s on the mock: open for the whole test
            "stream": True,
        }
        async with (
            httpx.AsyncClient(timeout=60, trust_env=False) as c,
            c.stream(
                "POST",
                f"{self.gw.url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.t.key}"},
                json=body,
            ) as r,
        ):
            assert r.status_code == 200, await r.aread()
            async for line in r.aiter_lines():
                if line.startswith("data: {") and content_of(sse(line)):
                    self.first.set()
                if self.stop.is_set():
                    return


async def two_ready(env: FakeServing, manager: PoolManager, model: str) -> PoolState:
    """Tick until the manager publishes two ready replicas (min_replicas: 2)."""
    for _ in range(10):
        await tick(manager, env.clock, advance=1)
        pool = await env.pool(model)
        if pool.routable and len(pool.replicas) == 2:
            return pool
    raise AssertionError(f"{model} never got two ready replicas: {await env.pool(model)}")


# ------------------------------------------------------------------ item 20


async def test_engine_killed_mid_stream_bills_the_streamed_tokens_and_the_pod_is_replaced(
    env: FakeServing, make_gateway: GatewayFactory
) -> None:
    """Item 20: the pod's engine process is killed mid-stream. The client gets the tokens
    so far, an `upstream_lost` error chunk and `[DONE]`; the usage event bills exactly
    the tokens streamed (the last per-chunk usage the client saw); the manager marks the
    pod failed, terminates it and replaces it, and the replacement serves through the
    gateway."""
    model = unique_model_id()
    gw = await make_gateway([model])
    manager = loop_manager(env, serving_config([model]), gw)
    await manager.start()
    pod = await bring_up(env, manager, model)
    t = await gw.tenant()

    body = {
        "model": model,
        "messages": MESSAGES,
        "max_tokens": 400,
        "min_tokens": 200,  # ~4 s of tokens on the mock: the kill lands mid-stream
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    events: list[dict[str, Any] | str] = []
    content = 0
    killed = False
    async with (
        httpx.AsyncClient(timeout=60, trust_env=False) as c,
        c.stream(
            "POST",
            f"{gw.url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {t.key}"},
            json=body,
        ) as r,
    ):
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            ev: dict[str, Any] | str = data if data == "[DONE]" else json.loads(data)
            events.append(ev)
            if isinstance(ev, dict) and ev.get("choices"):
                if (ev["choices"][0].get("delta") or {}).get("content"):
                    content += 1
                if content == 10 and not killed:
                    env.pods.kill_engine(pod.pod_id)
                    killed = True
    assert killed, f"the stream ended after {content} content chunks, before the kill"

    assert events[-1] == "[DONE]"
    errors = [e for e in events if isinstance(e, dict) and "error" in e]
    assert errors and errors[-1]["error"]["code"] == "upstream_lost", errors
    usages = [e["usage"] for e in events if isinstance(e, dict) and e.get("usage")]
    assert usages, "no per-chunk usage reached the client"
    last = usages[-1]
    assert 10 <= last["completion_tokens"] < 200  # cut mid-way

    [event] = await gw.wait_events(1)
    assert event.status == "aborted"
    assert (event.prompt_tokens, event.completion_tokens) == (
        last["prompt_tokens"],
        last["completion_tokens"],
    )
    assert event.cost_micros > 0
    assert await gw.reserved(t) == (0, 0)
    assert await gw.usage(t) == event.cost_micros

    # The manager sees the dead engine on its health checks: failed, terminated, replaced.
    replacement: PoolState | None = None
    for _ in range(40):
        await env.wake(model)  # clients keep asking (as the gateway's 503 wakes)
        await tick(manager, env.clock, advance=5)
        pool = await env.pool(model)
        if pool.routable and all(rep.id != pod.pod_id for rep in pool.replicas):
            replacement = pool
            break
    assert replacement is not None, f"never replaced: {await env.pool(model)}"
    old = await row_of(env, pod)
    assert old["failure"], old
    assert old["state"] in ("failed", "terminated"), old
    assert not env.runpod.has(pod.pod_id), "the failed pod was not terminated"
    live = [p for p in env.pods.live() if p.model == model]
    assert [p.pod_id for p in live] == [replacement.replicas[0].id]
    assert len(env.runpod.creates()) == 2

    # The replacement serves through the gateway, billed normally.
    r2 = await gw.post(t, model, max_tokens=8)
    assert r2.status_code == 200, r2.text
    events2 = await gw.wait_events(2)
    assert len(events2) == 2 and events2[1].status == "ok"
    assert events2[1].completion_tokens == r2.json()["usage"]["completion_tokens"]


# ------------------------------------------------------------------ item 21


@pytest.mark.parametrize("stream", [False, True])
async def test_failure_before_first_byte_is_retried_once_on_the_second_replica(
    env: FakeServing, make_gateway: GatewayFactory, stream: bool
) -> None:
    """Item 21: two pods (`max_replicas: 2`). The gateway picks the replica with the
    fewest of its open requests: a long stream (another tenant's) keeps one replica busy,
    so the idle one is tried first, and its engine is dead, so its proxy fails before the
    first byte. The gateway retries once on the busy replica, the client gets a normal
    answer, and it is billed once."""
    model = unique_model_id()
    gw = await make_gateway([model])
    config = serving_config({model: {"min_replicas": 2, "max_replicas": 2}})
    manager = loop_manager(env, config, gw)
    await manager.start()
    pool = await two_ready(env, manager, model)
    assert pool.replicas[0].id != pool.replicas[1].id

    busy = await GatewayStream(gw, await gw.tenant(), model).start()
    try:
        client = env.client()
        running = {rep.id: await engine_running(client, rep) for rep in pool.replicas}
        loaded = [rid for rid, n in running.items() if n > 0]
        assert len(loaded) == 1, running
        (dead,) = [rep.id for rep in pool.replicas if rep.id not in loaded]
        await kill_engine(env, dead)
        t = await gw.tenant()

        r = await gw.post(
            t,
            model,
            max_tokens=8,
            stream=stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        )
    finally:
        await busy.close()
    assert r.status_code == 200, r.text
    request_id = r.headers["x-loom-request-id"]
    if stream:
        events = sse(r.text)
        assert events[-1] == "[DONE]"
        assert not any(isinstance(e, dict) and "error" in e for e in events), events
        assert content_of(events)
        usages = [e["usage"] for e in events if isinstance(e, dict) and e.get("usage")]
        completion = usages[-1]["completion_tokens"]
    else:
        assert r.json()["choices"][0]["message"]["content"]
        completion = r.json()["usage"]["completion_tokens"]

    assert retries(gw, model) == 1.0
    await gw.wait_events(2)  # this request's and the busy stream's
    await asyncio.sleep(0.5)
    mine = [e for e in await gw.events() if e.org_id == t.org_id]
    assert len(mine) == 1, mine  # billed once: the failed attempt is not an event
    [event] = mine
    assert str(event.id) == request_id
    assert event.status == "ok" and event.completion_tokens == completion
    assert event.cost_micros > 0
    assert await gw.reserved(t) == (0, 0)
    assert await gw.usage(t) == event.cost_micros


@pytest.mark.parametrize("stream", [False, True])
async def test_failure_before_first_byte_with_one_replica_is_502_and_unbilled(
    env: FakeServing, make_gateway: GatewayFactory, stream: bool
) -> None:
    """Item 21: with one replica there is nowhere to retry: 502 `upstream_lost`, nothing
    billed, the reservation released."""
    model = unique_model_id()
    gw = await make_gateway([model])
    manager = loop_manager(env, serving_config([model]), gw)
    await manager.start()
    pod = await bring_up(env, manager, model)
    await kill_engine(env, pod.pod_id)
    t = await gw.tenant()

    r = await gw.post(t, model, max_tokens=8, stream=stream)
    assert r.status_code == 502, r.text
    assert error_of(r)["code"] == "upstream_lost"
    assert retries(gw, model) == 0.0
    await gw.wait_events(1)
    await nothing_billed(gw, t)


# ------------------------------------------------------------------ item 22


async def test_cold_then_warming_is_503_with_the_pools_retry_after(
    env: FakeServing, make_gateway: GatewayFactory
) -> None:
    """Item 22, `cold` and `warming`: the manager publishes the model `cold` with its
    `cold_start_estimate_s`; the gateway answers 503 `model_unavailable` with that as
    Retry-After (not its own estimate) and sets the wake key; the manager starts a pod
    (held in bootstrap), publishes `warming`, and the next 503 carries the warming
    hash's `retry_after_s`. Nothing is reserved or billed."""
    model = unique_model_id()
    gw = await make_gateway([model])
    manager = loop_manager(env, serving_config({model: {"cold_start_estimate_s": 77}}), gw)
    env.pods.behaviour_for[model] = PodBehaviour(engine="hang")  # stays bootstrapping
    await manager.start()
    await gw.redis.delete(wake_key(model))
    await tick(manager, env.clock, advance=1)
    pool = await env.pool(model)
    assert (pool.state, pool.retry_after_s) == ("cold", 77)
    t = await gw.tenant()

    r = await gw.post(t, model, max_tokens=8)
    assert r.status_code == 503, r.text
    assert error_of(r)["code"] == "model_unavailable"
    assert r.headers["retry-after"] == "77"
    assert await gw.redis.exists(wake_key(model)), "the cold 503 did not wake the model"
    await nothing_billed(gw, t)

    for _ in range(10):
        await step(manager, env.clock, advance=1)
        pool = await env.pool(model)
        if pool.state == "warming":
            break
    assert pool.state == "warming", pool
    assert len(env.runpod.creates()) == 1  # the gateway's wake started exactly one pod

    r = await gw.post(t, model, max_tokens=8)
    assert r.status_code == 503, r.text
    assert error_of(r)["code"] == "model_unavailable"
    assert r.headers["retry-after"] == str(pool.retry_after_s)
    assert 15 <= pool.retry_after_s <= 77
    await nothing_billed(gw, t)


async def test_no_capacity_is_503_no_gpu_capacity_with_the_backoff_retry_after(
    env: FakeServing, make_gateway: GatewayFactory
) -> None:
    """Item 22, `unavailable`: a wake from the gateway's cold 503 meets no capacity (no
    US datacenter has stock); the manager publishes `unavailable` / `no_capacity` with
    the first backoff step, and the gateway answers 503 `model_unavailable` "no GPU
    capacity" with that Retry-After. Nothing is reserved or billed."""
    model = unique_model_id()
    gw = await make_gateway([model])
    manager = loop_manager(env, serving_config([model]), gw)  # backoff_s: [10, 20, 40]
    await manager.start()
    env.runpod.stock = [["EUR-IS-2"]]
    await gw.redis.delete(wake_key(model))
    await tick(manager, env.clock, advance=1)
    t = await gw.tenant()

    r = await gw.post(t, model, max_tokens=8)
    assert r.status_code == 503 and error_of(r)["code"] == "model_unavailable"
    assert await gw.redis.exists(wake_key(model))

    await tick(manager, env.clock, advance=1)
    pool = await env.pool(model)
    assert (pool.state, pool.reason, pool.retry_after_s) == ("unavailable", "no_capacity", 10)

    r = await gw.post(t, model, max_tokens=8)
    assert r.status_code == 503, r.text
    err = error_of(r)
    assert err["code"] == "model_unavailable"
    assert "no GPU capacity" in err["message"]
    assert r.headers["retry-after"] == "10"
    await nothing_billed(gw, t)
    assert env.runpod.creates() == [] and env.pods.of(model) == []
