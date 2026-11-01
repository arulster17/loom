"""Credit reservation, rate limits and settlement in Redis: the Lua scripts in `lua/` and
thin async wrappers. The algorithm and key layout are in docs/phase1-contracts.md.

Per request the gateway makes one `reserve` call before dispatch and one `settle` call
after the response ends (success, error or abort). Balance state per org is two
counters loaded from Postgres once (`init_org`):

    available = credit - usage - reserved
    credit    = loom_org_credit_total(org)  (purchases, promos, refunds, adjustments)
    usage     = every settled request's cost (Redis is ahead of Postgres by the stream)

The per-minute ledger batch moves usage from unposted events to posted ledger entries;
both are already inside `usage`, so batching never touches Redis.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import cache
from importlib import resources

import psycopg
from redis.asyncio import Redis

from loom_platform.usage import UsageEvent

KEY_ROOT = "loom"
UNLIMITED = -1
# How long a settled request id is remembered, so a repeated settle is refused.
SETTLED_TTL_S = 24 * 3600


def org_prefix(org_id: uuid.UUID | str) -> str:
    return f"{KEY_ROOT}:org:{org_id}"


def org_key(org_id: uuid.UUID | str, name: str) -> str:
    """e.g. org_key(o, 'credit') -> 'loom:org:<o>:credit'."""
    return f"{org_prefix(org_id)}:{name}"


def api_key_key(org_id: uuid.UUID | str, key_id: uuid.UUID | str, name: str) -> str:
    """e.g. api_key_key(o, k, 'usage') -> 'loom:org:<o>:key:<k>:usage'."""
    return f"{org_prefix(org_id)}:key:{key_id}:{name}"


@cache
def _script_source(name: str) -> str:
    lua = resources.files("loom_platform.lua")
    body = lua.joinpath(f"{name}.lua").read_text(encoding="utf-8")
    if name in ("init_org", "set_credit"):
        return body
    return lua.joinpath("common.lua").read_text(encoding="utf-8") + "\n" + body


def script_source(name: str) -> str:
    """Full Lua source of 'reserve', 'settle', 'release', 'init_org' or 'set_credit'
    (common helpers prepended where used)."""
    return _script_source(name)


@dataclass(frozen=True)
class Limits:
    """Effective limits for one request; None = unlimited."""

    key_rpm: int | None
    key_tpm: int | None
    org_rpm: int | None
    org_tpm: int | None
    key_concurrency: int | None
    org_concurrency: int | None
    key_spend_limit_micros: int | None = None

    def argv(self) -> list[int]:
        return [
            UNLIMITED if v is None else v
            for v in (
                self.key_rpm,
                self.key_tpm,
                self.org_rpm,
                self.org_tpm,
                self.key_concurrency,
                self.org_concurrency,
                self.key_spend_limit_micros,
            )
        ]


NO_LIMITS = Limits(None, None, None, None, None, None, None)


@dataclass(frozen=True)
class ReserveResult:
    admitted: bool
    # ok | noinit | duplicate_request | insufficient_balance | key_spend_limit |
    # key_concurrency | org_concurrency | key_rpm | org_rpm | key_tpm | org_tpm
    reason: str
    retry_after_ms: int
    available_micros: int


@dataclass(frozen=True)
class SettleResult:
    settled: bool  # False: this request id was settled before; nothing changed
    # None when the org's state was not loaded (event appended, no counter bumped).
    org_usage_total_micros: int | None = None
    stream_id: str | None = None


def _str(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


class Limiter:
    """Async wrappers around the scripts. One instance per process; cheap to share."""

    def __init__(self, redis: Redis, *, usage_stream: str = "loom:usage") -> None:
        self.redis = redis
        self.usage_stream = usage_stream
        self._reserve = redis.register_script(script_source("reserve"))
        self._settle = redis.register_script(script_source("settle"))
        self._release = redis.register_script(script_source("release"))
        self._init = redis.register_script(script_source("init_org"))
        self._set_credit = redis.register_script(script_source("set_credit"))

    async def reserve(
        self,
        *,
        org_id: uuid.UUID,
        key_id: uuid.UUID,
        request_id: uuid.UUID,
        max_cost_micros: int,
        tpm_estimate: int,
        limits: Limits,
        ttl_ms: int,
    ) -> ReserveResult:
        res = await self._reserve(
            keys=[org_prefix(org_id)],
            args=[
                str(request_id),
                str(key_id),
                max_cost_micros,
                tpm_estimate,
                ttl_ms,
                *limits.argv(),
            ],
        )
        return ReserveResult(bool(res[0]), _str(res[1]), int(res[2]), int(res[3]))

    async def settle(self, event: UsageEvent) -> SettleResult:
        """Release the reservation, charge `event.cost_micros`, append the event."""
        res = await self._settle(
            keys=[org_prefix(event.org_id), self.usage_stream],
            args=[
                str(event.id),
                str(event.api_key_id),
                event.cost_micros,
                event.prompt_tokens + event.completion_tokens,
                SETTLED_TTL_S,
                *event.to_stream_fields(),
            ],
        )
        if not res[0]:
            return SettleResult(False)
        total = int(res[1])
        return SettleResult(True, total if total >= 0 else None, _str(res[2]))

    async def release(self, org_id: uuid.UUID, request_id: uuid.UUID) -> bool:
        """Drop a reservation that was never dispatched (503 after reserve): refund its tpm
        estimate, keep its rpm token spent, charge nothing, append no usage event."""
        return bool(await self._release(keys=[org_prefix(org_id)], args=[str(request_id)]))

    async def init_org(
        self,
        *,
        org_id: uuid.UUID,
        key_id: uuid.UUID,
        credit_total: int,
        usage_total: int,
        key_usage_total: int,
        credit_version: int = 0,
    ) -> tuple[bool, bool]:
        res = await self._init(
            keys=[org_prefix(org_id)],
            args=[credit_total, usage_total, str(key_id), key_usage_total, credit_version],
        )
        return bool(res[0]), bool(res[1])

    async def load_from_postgres(
        self, conn: psycopg.AsyncConnection, *, org_id: uuid.UUID, key_id: uuid.UUID
    ) -> tuple[bool, bool]:
        """Answer to a 'noinit' reserve: read the totals in one snapshot and init."""
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT loom_org_credit_total(%s), loom_org_usage_total(%s), "
                "loom_key_usage_total(%s), loom_org_credit_version(%s)",
                (org_id, org_id, key_id, org_id),
            )
            row = await cur.fetchone()
        assert row is not None
        credit, usage, key_usage, version = row
        return await self.init_org(
            org_id=org_id,
            key_id=key_id,
            credit_total=credit,
            usage_total=usage,
            key_usage_total=key_usage,
            credit_version=version,
        )

    async def set_credit(self, org_id: uuid.UUID, credit_total: int, version: int) -> bool:
        """Refresh an org's credit total from Postgres (on loom_credit_changed and in the
        minute reconcile) with its version (`ledger.org_credit_state`, one snapshot).
        Only updates an org that is already loaded (a lone credit key without its usage
        counter would never be read), and never with an older version than Redis holds:
        a reader whose snapshot predates another writer's cannot put back a stale total.
        True when Redis now holds this version's total."""
        res = await self._set_credit(keys=[org_prefix(org_id)], args=[credit_total, version])
        return int(res) == 1

    async def available(self, org_id: uuid.UUID) -> int | None:
        """credit - usage - reserved, or None when the org is not loaded."""
        credit, usage, rsum = await self.redis.mget(
            org_key(org_id, "credit"), org_key(org_id, "usage"), org_key(org_id, "rsum")
        )
        if credit is None or usage is None:
            return None
        return int(credit) - int(usage) - int(rsum or 0)
