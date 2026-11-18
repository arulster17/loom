"""Helpers for the before-M3 reconcile tests (test_billing_m3_*.py, spec B1/B2)."""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from billing_fixtures import Tenant, make_event

from loom_platform.ledger import org_usage_total
from loom_platform.limiter import Limiter, org_key, script_source
from loom_platform.usage import UsageEvent, insert_usage_events

_SCRIPT_COMMANDS = {"EVAL", "EVALSHA", "EVAL_RO", "EVALSHA_RO", "FCALL", "FCALL_RO"}
# The limiter's own scripts (loading an org, reserve/settle/release, the credit refresh):
# none of them is the reconciler's heal, told apart by source or SHA.
_LIMITER_SCRIPTS = frozenset(
    v
    for name in ("init_org", "reserve", "settle", "release", "set_credit")
    for v in (script_source(name), hashlib.sha1(script_source(name).encode()).hexdigest())
)
CONTRACTS = Path(__file__).resolve().parents[3] / "docs" / "phase1-contracts.md"
_ORG_KEY = re.compile(r"^\|\s*`loom:org:<org>:([^`]+)`")


def documented_token_key() -> str | None:
    """The suffix of the heal-token key in contracts §6's key table (`loom:org:<org>:X`,
    the row naming a heal token), or None when §6 documents none."""
    text = CONTRACTS.read_text(encoding="utf-8")
    start = text.find("\n## 6.")
    end = text.find("\n## 7.", start + 1)
    section = text[start:end] if start >= 0 else ""
    for line in section.splitlines():
        m = _ORG_KEY.match(line)
        low = line.lower()
        if m and "heal" in low and "token" in low:
            return m.group(1)
    return None


def token_key(org: uuid.UUID) -> str:
    """The org's heal-token key, as contracts §6 documents it (fails the test when §6
    documents none)."""
    name = documented_token_key()
    assert name is not None, (
        "contracts §6 documents no heal token key (spec B1: a `loom:org:<org>:...` row "
        "for the random heal token, written by init_org and the heal)"
    )
    return org_key(org, name)


def _text(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return str(v)


def script_keys(args: tuple[Any, ...]) -> list[str] | None:
    """The KEYS of a server-side script call (EVAL/EVALSHA/FCALL and their _RO forms:
    `CMD script|sha|function numkeys key... arg...`), or None for any other command."""
    if len(args) < 3 or _text(args[0]).upper() not in _SCRIPT_COMMANDS:
        return None
    try:
        numkeys = int(_text(args[2]))
    except ValueError:
        return None
    return [_text(k) for k in args[3 : 3 + numkeys]]


class HealGate(aioredis.Redis):
    """A Redis client (sharing another client's pool) that holds the org's heal until
    `release` is set. Given to a Reconciler, it stops that reconciler after it observed
    the org's usage and before its heal (spec B1: the heal is one atomic Lua call).

    The heal is recognised by what the spec fixes about it: the server-side script call
    (EVAL, EVALSHA, FCALL, or their read-only forms) that adds to the org's `usage` and
    sets its heal token, so its KEYS include both the `usage` key and the heal-token key
    (contracts §6), and it is not one of the limiter's own scripts (`init_org`, which
    writes both keys too, reserve, settle, release, set_credit; told apart by SHA or
    source). Only the first such call is held; any other command, and any other script,
    goes through. The call is caught whether it is sent directly (`execute_command`,
    which `eval`, `evalsha`, `fcall` and registered `Script`s use) or through a
    `pipeline()`: queued in a pipeline, it is held when the pipeline is executed; in a
    watching pipeline (immediate mode), when it is sent."""

    def __init__(self, base: aioredis.Redis, org: uuid.UUID) -> None:
        super().__init__(connection_pool=base.connection_pool)
        self.usage_key = org_key(org, "usage")
        self.token_key = token_key(org)
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self._held = False

    def _is_heal(self, args: tuple[Any, ...]) -> bool:
        keys = script_keys(args)
        if keys is None:
            return False
        if _text(args[1]) in _LIMITER_SCRIPTS:
            return False
        return self.usage_key in keys and self.token_key in keys

    async def _hold(self, calls: list[tuple[Any, ...]]) -> None:
        if not self._held and any(self._is_heal(a) for a in calls):
            self._held = True
            self.reached.set()
            await self.release.wait()

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        await self._hold([args])
        return await super().execute_command(*args, **options)

    def pipeline(self, transaction: bool = True, shard_hint: Any = None) -> Any:
        pipe = super().pipeline(transaction=transaction, shard_hint=shard_hint)
        gate = self
        queue = pipe.execute_command
        run = pipe.execute

        def execute_command(*args: Any, **options: Any) -> Any:
            if not pipe.watching:  # queued: held at execute()
                return queue(*args, **options)

            async def immediate() -> Any:
                await gate._hold([args])
                return await queue(*args, **options)

            return immediate()

        async def execute(raise_on_error: bool = True) -> Any:
            await gate._hold([a for a, _ in pipe.command_stack])
            return await run(raise_on_error=raise_on_error)

        pipe.execute_command = execute_command  # type: ignore[method-assign]
        pipe.execute = execute  # type: ignore[method-assign]
        return pipe


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
