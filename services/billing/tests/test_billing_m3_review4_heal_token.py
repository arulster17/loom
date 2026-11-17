"""B1 after the third review (docs/m3-review-fixes-spec.md, "Billing", B1): each loaded
org has a random heal token in Redis; every load sets a fresh one (`init_org.lua`, so
the gateway's `noinit` load too) and every heal sets a new one.

Redis state: the token's key is the one contracts §6 documents (the spec: "Contracts §6
(the token key, written by `init_org` and the heal)"). It is read from the §6 key table:
the `loom:org:<org>:...` row whose text names a heal token. The tests check only what the
spec fixes: a load or a heal leaves a non-empty token, and it differs from the one
before ("the only question asked of it is 'did it change'").

Behaviour: a reconciler held at its heal across a reload of the org must do nothing,
even when the reload came from an older Postgres snapshot (it has part of the missing
amount): with the token changed by the reload, the held heal adds nothing, so Redis
`usage` never ends above the true total."""

from __future__ import annotations

import asyncio
import contextlib
import re
import uuid
from pathlib import Path

import pytest
from billing_fixtures import GROUP
from billing_m3_helpers import (
    HealGate,
    load,
    lose_in_redis,
    pg_usage,
    redis_usage,
    settle,
    wait_reached,
)

from loom_billing.reconcile import Reconciler
from loom_platform.limiter import Limiter, org_key, org_prefix

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

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


async def _lose_org_keys(redis_client, org: uuid.UUID) -> None:
    """Redis loses everything it held for the org."""
    keys = [k async for k in redis_client.scan_iter(match=f"{org_prefix(org)}:*")]
    if keys:
        await redis_client.delete(*keys)
    assert await redis_client.exists(org_key(org, "usage")) == 0


async def _init_from_postgres(pool, limiter: Limiter, t, *, usage_total: int | None = None):
    """`init_org` with the totals read from Postgres (`usage_total` replaces the usage
    read, for a reload from an older snapshot)."""
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT loom_org_credit_total(%s), loom_org_usage_total(%s), "
            "loom_key_usage_total(%s), loom_org_credit_version(%s)",
            (t.org_id, t.org_id, t.key_id, t.org_id),
        )
        row = await cur.fetchone()
    assert row is not None
    credit, usage, key_usage, version = row
    org_new, _ = await limiter.init_org(
        org_id=t.org_id,
        key_id=t.key_id,
        credit_total=credit,
        usage_total=usage if usage_total is None else usage_total,
        key_usage_total=key_usage,
        credit_version=version,
    )
    assert org_new, "the org was not (re)initialised"


async def _load(how: str, pool, limiter: Limiter, t) -> None:
    if how == "noinit_load":
        async with pool.connection() as conn:
            org_new, _ = await limiter.load_from_postgres(conn, org_id=t.org_id, key_id=t.key_id)
        assert org_new, "the org was not (re)initialised"
    else:
        await _init_from_postgres(pool, limiter, t)


async def _token(redis_client, key: str) -> bytes | str | None:
    return await redis_client.get(key)


@pytest.mark.parametrize("how", ["init_org", "noinit_load"])
async def test_item2_every_load_writes_a_fresh_token(pool, redis_client, stream, make_tenant, how):
    """done-when B1 (token on load): initialising an org (`init_org`, and the gateway's
    `noinit` load) writes a heal token; two loads of the same org (Redis lost its keys
    in between) write different tokens."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = Limiter(redis_client, usage_stream=stream)
    key = token_key(t.org_id)
    seen: list[bytes | str] = []
    for n in range(3):
        if n:
            await _lose_org_keys(redis_client, t.org_id)
        await _load(how, pool, limiter, t)
        tok = await _token(redis_client, key)
        assert tok, f"load {n + 1} ({how}) wrote no heal token at {key}"
        seen.append(tok)
    assert len(set(seen)) == len(seen), (
        f"loads of the same org wrote a repeated heal token ({how}): {seen} (spec B1: every "
        "load sets a fresh token)"
    )


async def test_item2_heal_sets_a_new_token(pool, redis_client, stream, make_tenant, make_consumer):
    """done-when B1 (token on heal): a `redis_behind` heal adds the missing amount and
    sets a heal token different from the one read with U2."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    await settle(limiter, t)
    await consumer.drain()
    await lose_in_redis(pool, t)
    key = token_key(t.org_id)
    before = await _token(redis_client, key)
    p = await pg_usage(pool, t.org_id)
    assert await redis_usage(redis_client, t.org_id) < p

    await Reconciler(pool, redis_client, stream=stream, group=GROUP, drain_wait_s=0.2).run(
        [t.org_id]
    )

    assert await redis_usage(redis_client, t.org_id) == p, "the drift was not healed"
    after = await _token(redis_client, key)
    assert after, f"the heal left no heal token at {key}"
    assert after != before, f"the heal did not set a new token (still {after!r})"


async def test_item2_held_heal_across_older_snapshot_reload_does_nothing(
    pool, redis_client, stream, make_tenant, make_consumer
):
    """done-when B1 (reload between snapshot and heal): a reconciler observed
    `redis_behind` drift and is held at its heal. Redis loses the org's keys and the org
    is reloaded by `init_org` from an older Postgres snapshot, which has only part of
    the missing amount. The reload set a fresh token, so the held heal does nothing:
    Redis `usage` never ends above the true total."""
    t = await make_tenant(credit_micros=100_000_000)
    limiter = await load(pool, redis_client, stream, t)
    consumer = make_consumer()
    await consumer.setup()
    await settle(limiter, t)
    await consumer.drain()
    await lose_in_redis(pool, t)
    p = await pg_usage(pool, t.org_id)
    u2 = await redis_usage(redis_client, t.org_id)
    missing = p - u2
    assert missing > 1, "the setup made no redis_behind drift"

    gate = HealGate(redis_client, t.org_id)
    task = asyncio.create_task(
        Reconciler(pool, gate, stream=stream, group=GROUP, drain_wait_s=0.2).run([t.org_id])
    )
    try:
        await wait_reached(gate, task)
        await _lose_org_keys(redis_client, t.org_id)
        reloaded = u2 + missing // 2  # an older snapshot: part of the missing amount
        await _init_from_postgres(pool, limiter, t, usage_total=reloaded)
        assert await redis_usage(redis_client, t.org_id) == reloaded
    finally:
        gate.release.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(task), 15)
    await asyncio.wait_for(task, 15)

    true_total = await pg_usage(pool, t.org_id)
    usage = await redis_usage(redis_client, t.org_id)
    assert usage <= true_total, (
        f"over-healed after a reload: usage {usage} above the true total {true_total} by "
        f"{usage - true_total} (the held heal ran on a reloaded org; spec B1: every load "
        "sets a fresh token, so a heal read before the reload does nothing)"
    )
    assert usage == reloaded, f"usage {usage} moved off the reloaded value {reloaded}"
