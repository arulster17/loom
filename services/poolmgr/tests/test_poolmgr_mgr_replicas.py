"""Replica caps and the crash loop (docs/phase1-m2-spec.md §13 items 9 and 11; §5 "Create":
"max_replicas counts every row not terminated, and is checked in the same transaction
that inserts the row"; "Health rules" last row; §7 `crash_loop`; §11 `reset`).

`reconcile_once()` is not re-entrant (§11 "Timing rules"), so ticks here are serial; the
race tested is a tick that runs while an earlier tick's create is still in flight
(`FakeServingRunpod.create_gate` holds POST /pods).
"""

from __future__ import annotations

import asyncio
import threading

from poolmgr_fakes import (
    FakeServing,
    bring_up,
    drive,
    env_fixture,  # noqa: F401
    serving_config,
    settle,
    step,
    tick,
    unique_model_id,
)

from loom_platform.pool import request_wake
from loom_platform.testing import pg_url, redis_client  # noqa: F401


async def live_rows(env: FakeServing, model: str) -> int:
    return sum(1 for r in await env.rows(model) if r["state"] != "terminated")


async def test_wakes_during_an_inflight_create_make_one_pod(env: FakeServing) -> None:
    """Item 9: many concurrent wakes, and ticks back to back while the first create is
    still waiting on RunPod, never exceed `max_replicas` (1)."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    gate = threading.Event()
    env.runpod.create_gate = gate  # POST /pods waits until the gate opens
    await asyncio.gather(*(request_wake(env.redis, model) for _ in range(10)))
    await manager.reconcile_once()
    for _ in range(6):
        await asyncio.gather(*(request_wake(env.redis, model) for _ in range(3)))
        await env.wake(model)
        await step(manager, env.clock, advance=5)
        assert await live_rows(env, model) <= 1
    assert len(env.runpod.creates()) == 0  # the one create is still in flight
    gate.set()
    await drive(env.clock, lambda: len(env.runpod.creates()) >= 1)
    await settle(manager, env.clock)
    for _ in range(4):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        assert await live_rows(env, model) <= 1
        assert len(env.runpod.of(model)) <= 1
    assert len(env.runpod.creates()) == 1
    assert (await env.pool(model)).routable


async def test_min_replicas_two_creates_exactly_two(env: FakeServing) -> None:
    """Item 9: the desired count is capped at max_replicas; never one more."""
    model = unique_model_id()
    manager = env.manager(serving_config({model: {"min_replicas": 2, "max_replicas": 2}}))
    await manager.start()
    await env.wake(model)
    for _ in range(6):
        await tick(manager, env.clock, advance=5)
        assert await live_rows(env, model) <= 2
    pool = await env.pool(model)
    assert pool.state == "ready" and len(pool.replicas) == 2
    assert len(env.runpod.creates()) == 2


async def test_failed_pod_counts_until_terminated(env: FakeServing) -> None:
    """Item 9, "in the moment a failed pod is being replaced": a failed pod whose terminate
    has not succeeded (REST DELETE and GraphQL podTerminate both refuse) still counts, so
    its replacement waits for it."""
    model = unique_model_id()
    manager = env.manager(serving_config([model], lifecycle={"max_replacements_per_hour": 10}))
    await manager.start()
    pod = await bring_up(env, manager, model)
    assert pod.engine is not None
    env.runpod.terminate_fails = True
    pod.engine.set_healthy(False)  # 3 failed checks, then the grace (30 s) runs out
    for _ in range(14):  # 70 s
        await env.wake(model)
        await step(manager, env.clock, advance=5)
        assert await live_rows(env, model) <= 1
        assert len(env.runpod.of(model)) <= 1
    (row,) = await env.rows(model)
    assert row["state"] == "failed", row["state"]
    assert env.runpod.has(pod.pod_id)
    assert len(env.runpod.creates()) == 1, "replaced while the failed pod still counts"

    env.runpod.terminate_fails = False
    for _ in range(6):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        assert await live_rows(env, model) <= 1
        if len(env.runpod.creates()) == 2:
            break
    assert not env.runpod.has(pod.pod_id)
    assert len(env.runpod.creates()) == 2
    states = [r["state"] for r in await env.rows(model)]
    assert states.count("terminated") == 1, states


async def test_crash_loop_stops_creates_until_reset(env: FakeServing) -> None:
    """Item 11: more than max_replacements_per_hour replacements: no further create, pool
    `unavailable` (`crash_loop`, unavailable_retry_after_s) whatever the wakes; `reset`
    clears it."""
    model = unique_model_id()
    manager = env.manager(serving_config([model], lifecycle={"max_replacements_per_hour": 2}))
    await manager.start()
    env.pods.behaviour.fail_bootstrap = True
    for _ in range(15):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    # The first pod and two replacements; a third replacement would be one too many.
    assert len(env.runpod.creates()) == 3
    pool = await env.pool(model)
    assert (pool.state, pool.reason, pool.retry_after_s) == ("unavailable", "crash_loop", 300)
    assert env.runpod.of(model) == [], "a failed pod was left running"
    for _ in range(4):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    assert len(env.runpod.creates()) == 3

    env.pods.behaviour.fail_bootstrap = False
    await env.admin({"op": "reset", "model": model})
    await env.wake(model)
    for _ in range(6):
        await tick(manager, env.clock, advance=1)
        if (await env.pool(model)).routable:
            break
    pool = await env.pool(model)
    assert pool.state == "ready" and pool.reason is None, pool
    assert len(env.runpod.creates()) == 4
