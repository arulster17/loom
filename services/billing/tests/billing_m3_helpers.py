"""Helpers for the before-M3 reconcile tests (test_billing_m3_*.py, spec B1/B2)."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Any

import redis.asyncio as aioredis
from billing_fixtures import Tenant, make_event

from loom_platform.ledger import org_usage_total
from loom_platform.limiter import Limiter, org_key, script_source
from loom_platform.usage import UsageEvent, insert_usage_events

_SCRIPT_COMMANDS = {"EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"}
_SET_CREDIT = script_source("set_credit")
_SET_CREDIT_SHA = hashlib.sha1(_SET_CREDIT.encode()).hexdigest()


def _text(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return str(v)


class HealGate(aioredis.Redis):
    """A Redis client (sharing another client's pool) that holds the first server-side
    script call touching `org` (other than the credit refresh, set_credit.lua) until
    `release` is set. Given to a Reconciler, it stops that reconciler after it observed
    the org's usage and before its heal (spec B1: the heal is one atomic Lua call)."""

    def __init__(self, base: aioredis.Redis, org: uuid.UUID) -> None:
        super().__init__(connection_pool=base.connection_pool)
        self.org = str(org)
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self._held = False

    def _is_heal(self, args: tuple[Any, ...]) -> bool:
        if not args or _text(args[0]).upper() not in _SCRIPT_COMMANDS:
            return False
        texts = [_text(a) for a in args[1:]]
        if texts and (texts[0] == _SET_CREDIT_SHA or texts[0] == _SET_CREDIT):
            return False
        return any(self.org in t for t in texts[1:])

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        if not self._held and self._is_heal(args):
            self._held = True
            self.reached.set()
            await self.release.wait()
        return await super().execute_command(*args, **options)


async def wait_reached(gate: HealGate, task: asyncio.Task[Any], timeout_s: float = 15) -> None:
    """Wait until the gated reconciler is at its heal. Fails if it finished without one."""
    waiter = asyncio.ensure_future(gate.reached.wait())
    done, _ = await asyncio.wait({waiter, task}, timeout=timeout_s, return_when="FIRST_COMPLETED")
    if waiter not in done:
        waiter.cancel()
        if task in done:
            task.result()  # re-raise its error, if any
            raise AssertionError(
                "the reconciler observed redis_behind drift but finished without a "
                "server-side script call (spec B1: the heal is one atomic Lua call)"
            )
        raise AssertionError("the reconciler never reached its heal")


async def load(pool, redis_client, stream: str, t: Tenant) -> Limiter:
    limiter = Limiter(redis_client, usage_stream=stream)
    async with pool.connection() as conn:
        await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
    return limiter


async def redis_usage(redis_client, org: uuid.UUID) -> int:
    return int(await redis_client.get(org_key(org, "usage")))


async def pg_usage(pool, org: uuid.UUID) -> int:
    async with pool.connection() as conn:
        return await org_usage_total(conn, org)


async def lose_in_redis(pool, t: Tenant, prompt: int = 500_000) -> UsageEvent:
    """Usage Postgres has and Redis never counted (Redis lost it): `redis_behind`."""
    lost = make_event(t, prompt=prompt)
    async with pool.connection() as conn, conn.transaction():
        await insert_usage_events(conn, [lost])
    return lost


async def settle(limiter: Limiter, t: Tenant, prompt: int = 1000) -> int:
    ev = make_event(t, prompt=prompt)
    assert (await limiter.settle(ev)).settled
    return ev.cost_micros
