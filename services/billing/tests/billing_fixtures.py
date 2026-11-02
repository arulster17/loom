"""Billing test fixtures (re-exported by conftest.py) and helpers the tests import: a
pool on the session database, per-test stream names, factories for orgs, keys and usage
events. A uniquely named module, so it never collides with another service's conftest."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass

import httpx
import psycopg
import pytest
import redis.asyncio as aioredis
from psycopg_pool import AsyncConnectionPool

from loom_billing.consumer import UsageConsumer
from loom_billing.stripe_api import Transport
from loom_platform.db import open_pool
from loom_platform.ledger import STRIPE_CLEARING, Entry, org_credit_account, post_transaction
from loom_platform.pricing import Price, cost_micros
from loom_platform.testing import create_api_key, create_org, pg, pg_url, redis_client
from loom_platform.usage import UsageEvent

__all__ = ["pg", "pg_url", "redis_client"]

# Stub-mode settings for tests (LOOM_STRIPE_MODE=stub has no default secret).
STUB_WEBHOOK_SECRET = "whsec_test_stub_" + "s" * 24
STUB = {"stripe_mode": "stub", "stripe_stub_webhook_secret": STUB_WEBHOOK_SECRET}

# Qwen3-8B-like list price, micro-dollars per 1M (input, cached input, output).
PRICE = Price(77_000, 38_000, 308_000)
GROUP = "billing"


@dataclass(frozen=True)
class Tenant:
    org_id: uuid.UUID
    key_id: uuid.UUID


@pytest.fixture
async def pool(pg_url: str) -> AsyncIterator[AsyncConnectionPool]:
    p = await open_pool(pg_url, min_size=1, max_size=8)
    try:
        yield p
    finally:
        await p.close()


@pytest.fixture
async def stream(redis_client: aioredis.Redis) -> AsyncIterator[str]:
    name = f"loom:test:usage:{uuid.uuid4().hex[:12]}"
    try:
        yield name
    finally:
        await redis_client.delete(name, name + ":dead", name + ":notify")


@pytest.fixture
def make_tenant(pool: AsyncConnectionPool) -> Callable[..., Awaitable[Tenant]]:
    async def make(status: str = "active", credit_micros: int = 0) -> Tenant:
        async with pool.connection() as conn, conn.transaction():
            org = await create_org(conn, status=status)
            key, _ = await create_api_key(conn, org)
            if credit_micros:
                await purchase(conn, org, credit_micros)
        return Tenant(org, key)

    return make


async def purchase(conn: psycopg.AsyncConnection, org: uuid.UUID, micros: int) -> None:
    acct = await org_credit_account(conn, org)
    await post_transaction(
        conn,
        kind="purchase",
        idempotency_key=f"stripe:evt_test_{uuid.uuid4().hex}",
        org_id=org,
        entries=[Entry(acct, micros), Entry(STRIPE_CLEARING, -micros)],
    )


def make_event(
    t: Tenant,
    *,
    prompt: int = 1000,
    completion: int = 500,
    cached: int = 0,
    created_at: dt.datetime | None = None,
    status: str = "ok",
) -> UsageEvent:
    return UsageEvent(
        id=uuid.uuid4(),
        org_id=t.org_id,
        api_key_id=t.key_id,
        model_id="qwen3-8b",
        price_version=PRICE.version,
        cloud="local",
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_prompt_tokens=cached,
        latency_ms=120,
        ttft_ms=15,
        status=status,  # type: ignore[arg-type]
        cost_micros=cost_micros(PRICE, prompt, completion, cached),
        created_at=created_at or dt.datetime.now(dt.UTC),
    )


async def xadd_event(redis: aioredis.Redis, stream: str, ev: UsageEvent) -> str:
    f = ev.to_stream_fields()
    pairs = dict(zip(f[::2], f[1::2], strict=True))
    entry_id = await redis.xadd(stream, pairs)  # type: ignore[arg-type]
    return entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)


@pytest.fixture
def make_consumer(
    pool: AsyncConnectionPool, redis_client: aioredis.Redis, stream: str
) -> Callable[..., UsageConsumer]:
    def make(name: str = "c1", **kw: int) -> UsageConsumer:
        kw.setdefault("block_ms", 10)
        return UsageConsumer(
            pool,
            redis_client,
            stream=stream,
            group=GROUP,
            consumer=name,
            dead_stream=stream + ":dead",
            **kw,
        )

    return make


def asgi_transport(app: object) -> Transport:
    """A stripe_api.Transport that calls an ASGI app in-process."""
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))  # type: ignore[arg-type]

    async def call(
        method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        r = await client.request(method, url, headers=dict(headers), content=body)
        return r.status_code, r.content

    return call


async def ledger_is_balanced(conn: psycopg.AsyncConnection) -> bool:
    async with conn.cursor() as cur:
        await cur.execute("SELECT coalesce(sum(amount_micros), 0) FROM ledger_entries")
        row = await cur.fetchone()
        assert row is not None
        await cur.execute(
            "SELECT count(*) FROM (SELECT transaction_id FROM ledger_entries "
            "GROUP BY transaction_id HAVING sum(amount_micros) <> 0 OR count(*) < 2) x"
        )
        bad = await cur.fetchone()
        assert bad is not None
    return int(row[0]) == 0 and int(bad[0]) == 0
