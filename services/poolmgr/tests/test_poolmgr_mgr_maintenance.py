"""Per-pod SSH maintenance never stalls the tick (docs/phase1-m2-spec.md §11 "Fixed after
the final whole-branch review": lease renewal and certificate re-key run as their own
task, at most one in flight per pod, so one hung pod never stalls the tick).

Model A's pod is up; its maintenance call over SSH (a lease renewal, or a certificate
re-key) then hangs forever. Each tick must still return promptly, model B's wake must
still create a pod that becomes ready, the pool hashes keep being republished, and A
never has more than one maintenance call in flight however many renewal periods pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from poolmgr_fakes import (
    FakeServing,
    FakeServingExec,
    bring_up,
    env_fixture,  # noqa: F401
    serving_config,
    unique_model_id,
)

from loom_bench.providers.runpod_ssh import ExecResult, SshTarget
from loom_platform.pool import pool_key, wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import PoolManager

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

TICK_TIMEOUT_S = 10.0  # real seconds a tick may take: anything hung shows as a timeout


class HangingMaintenance:
    """Wraps the fake pod exec: for one pod, the chosen maintenance scripts never return
    (until `release`). Counts them, and the most in flight at once."""

    def __init__(self, env: FakeServing, pod_id: str, kinds: Callable[[str, Any], bool]):
        self.env = env
        self.pod_id = pod_id
        self.kinds = kinds
        self.release = asyncio.Event()
        self.started = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.maintenance_in_flight = 0
        self.max_maintenance_in_flight = 0
        self._run = env.pods.run

    def install(self) -> None:
        self.env.pods.run = self.run  # type: ignore[method-assign]

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        pod = next((p for p in self.env.pods.snapshot() if p.ssh_port == target.port), None)
        kind = FakeServingExec._classify(script)[0]
        mine = pod is not None and pod.pod_id == self.pod_id
        maintenance = mine and kind in ("serve_lease", "serve_bootstrap")
        if maintenance:
            self.maintenance_in_flight += 1
            self.max_maintenance_in_flight = max(
                self.max_maintenance_in_flight, self.maintenance_in_flight
            )
        try:
            if mine and self.kinds(kind, pod):
                self.started += 1
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
                try:
                    await self.release.wait()
                finally:
                    self.in_flight -= 1
            return await self._run(target, script, timeout_s=timeout_s)
        finally:
            if maintenance:
                self.maintenance_in_flight -= 1


async def tick_promptly(manager: PoolManager) -> None:
    await asyncio.wait_for(manager.reconcile_once(), TICK_TIMEOUT_S)


async def run_until(
    env: FakeServing,
    manager: PoolManager,
    done: Callable[[], Awaitable[bool]],
    *,
    wake: str,
    max_advance_s: float = 120.0,
) -> None:
    """Tick (each must return promptly), letting background work run and jumping the
    clock to the next sleeper, until `done()`."""
    start = env.clock.now
    for _ in range(200):
        if await done():
            return
        await env.wake(wake)
        await tick_promptly(manager)
        await asyncio.sleep(0.05)
        nxt = env.clock.next_deadline()
        env.clock.set(max(nxt, env.clock.now) if nxt is not None else env.clock.now + 1)
        assert env.clock.now - start <= max_advance_s, "not done within the fake time"
    raise AssertionError("not done")


async def updated_at(env: FakeServing, model: str) -> int:
    raw = await env.redis.hget(pool_key(model), "updated_at_ms")
    assert raw is not None, f"no pool hash for {model}"
    return int(raw)


def lease_hangs(kind: str, pod: Any) -> bool:
    return kind == "serve_lease"


def rekey_hangs(kind: str, pod: Any) -> bool:
    # The first serve_bootstrap is the bootstrap itself; later ones re-key.
    return kind == "serve_bootstrap" and len(pod.payloads) >= 1


CASES = {
    # Lease renewal every 60 s.
    "lease": (lease_hangs, {}, 61.0),
    # Server certificates live 1 day and are re-keyed 2 minutes after issue.
    "rekey": (
        rekey_hangs,
        {"lifecycle": {"server_cert_days": 1, "server_cert_renew_before_s": 86400 - 120}},
        125.0,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
async def test_a_hung_pod_maintenance_does_not_stall_the_tick(env: FakeServing, case: str) -> None:
    hangs, sections, due_after_s = CASES[case]
    a, b = unique_model_id(), unique_model_id()
    config = serving_config([a, b], **sections)
    manager = env.manager(config)
    await manager.start()
    pod_a = await bring_up(env, manager, a)
    await env.redis.delete(wake_key(b))

    hang = HangingMaintenance(env, pod_a.pod_id, hangs)
    hang.install()
    try:
        env.clock.advance(due_after_s)
        await env.wake(a)
        await tick_promptly(manager)  # A's maintenance is due: it starts, and hangs
        await asyncio.sleep(0.2)
        assert hang.started == 1, f"A's {case} never started (or more than once)"
        before = await updated_at(env, a)

        # B's wake still creates a pod that becomes ready.
        async def b_ready() -> bool:
            return (await env.pool(b)).routable

        await run_until(env, manager, b_ready, wake=b)
        assert len(env.pods.of(b)) == 1

        # More renewal periods pass while A's call still hangs: never a second one.
        for _ in range(3):
            env.clock.advance(due_after_s)
            await env.wake(a)
            await tick_promptly(manager)
            await asyncio.sleep(0.1)
        assert hang.started == 1
        assert hang.max_in_flight == 1
        assert hang.max_maintenance_in_flight <= 1
        assert await updated_at(env, a) > before, "A's pool hash was not republished"
        assert (await env.pool(b)).routable
    finally:
        hang.release.set()
        await asyncio.sleep(0.1)
