from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from gateway_testkit import (
    Certs,
    FakeEngine,
    Tenant,
    make_certs,
    make_tenant,
    mtls_server_config,
    replica,
    serve,
    set_ready,
    usage_events,
    wait_for_events,
)
from openai import AsyncOpenAI
from redis.asyncio import Redis

from loom_bench.registry import Registry, load_registry
from loom_gateway.app import Gateway, create_app
from loom_gateway.settings import GatewaySettings
from loom_platform.limiter import org_key
from loom_platform.pool import pool_key, wake_key
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV, pg, pg_url, redis_client  # noqa: F401
from loom_platform.usage import UsageEvent


@pytest.fixture(scope="session")
def certs(tmp_path_factory: pytest.TempPathFactory) -> Certs:
    return make_certs(Path(tmp_path_factory.mktemp("certs")))


@pytest.fixture(scope="session")
def test_registry() -> Registry:
    """Copies of the real rows under ids unique to this session, so pool hashes
    (`loom:pool:<model>`) never collide with other sessions sharing the Redis server."""
    base = load_registry()
    tag = uuid.uuid4().hex[:8]
    small = base.get("qwen3-8b").model_copy(update={"id": f"t{tag}-qwen3-8b"})
    big = base.get("llama-3.3-70b-instruct-fp8").model_copy(
        update={"id": f"t{tag}-llama-70b-fp8", "base_model": None}
    )
    unpriced = base.get("qwen3-8b-fp8").model_copy(
        update={"id": f"t{tag}-unpriced", "base_model": None}
    )
    return Registry(models=[small, big, unpriced])


@dataclass
class Stack:
    url: str
    app: Any
    redis: Redis
    stream: str
    pg_url: str
    engines: dict[str, FakeEngine]
    ports: dict[str, int]
    small: str  # qwen3-8b copy
    big: str  # llama 70B FP8 copy
    unpriced: str
    tls: bool
    tenants: list[Tenant] = field(default_factory=list)
    clients: list[AsyncOpenAI] = field(default_factory=list)
    exit_stack: contextlib.AsyncExitStack | None = None

    async def track_server(self, server: contextlib.AbstractAsyncContextManager[int]) -> int:
        """Enter a `serve(...)` context for the lifetime of the stack; returns its port."""
        assert self.exit_stack is not None
        return await self.exit_stack.enter_async_context(server)

    @property
    def gateway(self) -> Gateway:
        gw: Gateway = self.app.state.gateway
        return gw

    @property
    def engine(self) -> FakeEngine:
        return next(iter(self.engines.values()))

    async def tenant(self, **kw: Any) -> Tenant:
        kw.setdefault("key_limits", None)
        t = await make_tenant(self.pg_url, **kw)
        self.tenants.append(t)
        return t

    def openai(self, tenant: Tenant | str) -> AsyncOpenAI:
        key = tenant if isinstance(tenant, str) else tenant.key
        return self.track(
            AsyncOpenAI(base_url=f"{self.url}/v1", api_key=key, max_retries=0, timeout=30)
        )

    def track(self, client: AsyncOpenAI) -> AsyncOpenAI:
        """Close this SDK client at teardown (inside the test's event loop)."""
        self.clients.append(client)
        return client

    async def events(self) -> list[UsageEvent]:
        return await usage_events(self.redis, self.stream)

    async def wait_events(self, count: int, timeout_s: float = 5.0) -> list[UsageEvent]:
        return await wait_for_events(self.redis, self.stream, count, timeout_s)

    async def reserved(self, t: Tenant) -> tuple[int, int]:
        """(org rsum, org in-flight reservations)."""
        rsum = await self.redis.get(org_key(t.org_id, "rsum"))
        n = await self.redis.zcard(org_key(t.org_id, "rsv:exp"))
        return int(rsum or 0), int(n)

    async def usage(self, t: Tenant) -> int:
        return int(await self.redis.get(org_key(t.org_id, "usage")) or 0)


StackFactory = Callable[..., Awaitable[Stack]]


@pytest.fixture
async def make_stack(
    pg_url: str,  # noqa: F811
    redis_client: Redis,  # noqa: F811
    certs: Certs,
    test_registry: Registry,
) -> AsyncIterator[StackFactory]:
    async with contextlib.AsyncExitStack() as es:
        stacks: list[Stack] = []

        async def build(
            *,
            engines: tuple[str, ...] = ("e1",),
            tls: bool = True,
            ready: bool = True,
            **overrides: Any,
        ) -> Stack:
            models = [m.id for m in test_registry.models]
            small, big, unpriced = models
            fakes = {name: FakeEngine(name) for name in engines}
            ports: dict[str, int] = {}
            for name, fake in fakes.items():
                cfg = mtls_server_config(certs) if tls else {}
                ports[name] = await es.enter_async_context(serve(fake.app(), **cfg))
            settings_kw: dict[str, Any] = {
                "database_url": pg_url,
                "redis_url": os.environ.get(REDIS_ENV) or DEFAULT_REDIS,
                "usage_stream": f"loom:test:usage:{uuid.uuid4().hex}",
                "pool_cache_ttl_s": 0,
                "log_level": "WARNING",
            }
            if tls:
                settings_kw |= {
                    "gateway_client_cert": str(certs.client_cert),
                    "gateway_client_key": str(certs.client_key),
                    "pod_ca_cert": str(certs.ca),
                }
            settings_kw |= overrides
            settings = GatewaySettings(**settings_kw)
            app = create_app(
                settings,
                registry=test_registry,
                cold_start_s={small: 240, big: 450},
            )
            port = await es.enter_async_context(serve(app))
            if ready:
                reps = [replica(name, ports[name], tls=tls) for name in engines]
                for m in (small, big):
                    await set_ready(redis_client, m, reps)
            stack = Stack(
                url=f"http://127.0.0.1:{port}",
                app=app,
                redis=redis_client,
                stream=settings.usage_stream,
                pg_url=pg_url,
                engines=fakes,
                ports=ports,
                small=small,
                big=big,
                unpriced=unpriced,
                tls=tls,
                exit_stack=es,
            )
            stacks.append(stack)
            return stack

        try:
            yield build
        finally:
            for s in stacks:
                for c in s.clients:
                    await c.close()
            await es.aclose()
            for s in stacks:
                keys = [s.stream]
                for m in (s.small, s.big, s.unpriced):
                    keys += [pool_key(m), wake_key(m)]
                await redis_client.delete(*keys)
                for t in s.tenants:
                    org_keys = [k async for k in redis_client.scan_iter(f"loom:org:{t.org_id}:*")]
                    if org_keys:
                        await redis_client.delete(*org_keys)


@pytest.fixture
async def stack(make_stack: StackFactory) -> Stack:
    return await make_stack()
