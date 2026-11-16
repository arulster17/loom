"""Idle stop, drain and always-on models (docs/phase1-m2-spec.md §13 items 7 and 8; §5
"Reconcile loop" step 1, "Idle stop and drain"; §11 "Timing rules" (idle time runs from
the last demand the manager observed; due at `>=`), `loom-poolmgr stop` and `drain`).

The drain is driven tick by tick (`step`: advance the fake clock, reconcile, let tasks
run) rather than with `settle`, so the test decides when time moves while a stream is
open on the draining pod.
"""

from __future__ import annotations

import time
from typing import Any

from poolmgr_fakes import (
    FakePod,
    FakeServing,
    ProxyStream,
    bring_up,
    env_fixture,  # noqa: F401
    serving_config,
    step,
    tick,
    unique_model_id,
    wait_until,
)

from loom_platform.pool import Replica, last_request_key, wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import PoolManager


def in_routing(replicas: tuple[Replica, ...], pod: FakePod) -> bool:
    return any(r.id == pod.pod_id for r in replicas)


def alive(env: FakeServing, pod: FakePod) -> bool:
    return env.runpod.has(pod.pod_id) and env.runpod.terminate_requests(pod.pod_id) == 0


async def row_of(env: FakeServing, pod: FakePod) -> dict[str, Any]:
    (row,) = [r for r in await env.rows(pod.model) if r["pod_name"] == pod.name]
    return row


async def ready_idle_pod(
    env: FakeServing, model: str, **config: Any
) -> tuple[PoolManager, FakePod, Replica, float]:
    """A ready pod whose last demand the manager saw at the returned time (a wake key
    seen on a tick, then deleted)."""
    manager = env.manager(serving_config({model: {"idle_stop_s": 60}}, **config))
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.wake(model)
    t_demand = env.clock()
    await step(manager, env.clock, advance=0)  # the manager observes this demand now
    await env.redis.delete(wake_key(model))  # no demand from here on
    (replica,) = (await env.pool(model)).replicas
    return manager, pod, replica, t_demand


async def drain_started(env: FakeServing, manager: PoolManager, model: str, pod: FakePod) -> float:
    """Tick every 5 s until the pod leaves the pool hash (at most 120 s); the time it did."""
    for _ in range(24):
        await step(manager, env.clock, advance=5)
        if not in_routing((await env.pool(model)).replicas, pod):
            return env.clock()
    raise AssertionError("the idle pod was never drained")


async def test_idle_pod_leaves_hash_first_and_waits_for_inflight(env: FakeServing) -> None:
    """Item 7: drained exactly idle_stop_s after the last demand (`>=`); removed from the
    hash before termination; terminated only after in-flight reaches 0; a stream open on
    it while it drains completes in full; the pool ends cold."""
    model = unique_model_id()
    manager, pod, replica, t_demand = await ready_idle_pod(
        env,
        model,
        lifecycle={"drain_max_s": 600},
        health={"progress_stall_s": 600},
    )
    drained_at = await drain_started(env, manager, model, pod)
    assert drained_at == t_demand + 60, drained_at - t_demand
    pool = await env.pool(model)
    assert pool.state == "cold" and pool.retry_after_s == 60 and pool.replicas == ()
    assert alive(env, pod), "terminated before draining"
    row = await row_of(env, pod)
    assert row["state"] == "draining" and row["drain_started_at"] is not None

    # A request routed from a cached pool hash arrives while the pod drains.
    stream = await ProxyStream(env.client(), replica, model, max_tokens=100).start()
    while not stream.finished:
        await step(manager, env.clock, advance=1)
        assert alive(env, pod) or stream.finished, "terminated with a request in flight"
    result = await stream.finish()
    assert result.complete, result
    assert result.completion_tokens == 100 and result.finish_reason == "length"

    for _ in range(30):
        await step(manager, env.clock, advance=1)
        if not env.runpod.has(pod.pod_id):
            break
    assert not env.runpod.has(pod.pod_id), "never terminated after in-flight reached 0"
    row = await row_of(env, pod)
    assert row["state"] == "terminated" and row["terminated_by"] == "manager"
    pool = await env.pool(model)
    assert (pool.state, pool.retry_after_s, pool.replicas) == ("cold", 60, ())


async def test_drain_terminates_at_drain_max_s(env: FakeServing) -> None:
    """Item 7: in-flight that never reaches 0 is cut at `drain_max_s`, not before."""
    model = unique_model_id()
    manager, pod, replica, _ = await ready_idle_pod(
        env,
        model,
        lifecycle={"drain_max_s": 30},
        health={"progress_stall_s": 600},
    )
    await drain_started(env, manager, model, pod)
    stream = await ProxyStream(env.client(), replica, model, max_tokens=5000).start()
    assert pod.engine is not None
    pod.engine.stall()  # the request never finishes
    started = (await row_of(env, pod))["drain_started_at"].timestamp()
    while env.clock() < started + 25:
        await step(manager, env.clock, advance=1)
    assert alive(env, pod), "terminated before drain_max_s"
    while env.clock() < started + 40 and env.runpod.has(pod.pod_id):
        await step(manager, env.clock, advance=1)
    assert not env.runpod.has(pod.pod_id), "not terminated at drain_max_s"
    await stream.cancel()
    assert (await row_of(env, pod))["state"] == "terminated"
    assert (await env.pool(model)).state == "cold"


async def test_settled_request_keeps_pod_ready(env: FakeServing) -> None:
    """Idle runs from the last demand observed: a change in the model's `last_request`
    stamp (the gateway's `settle.lua` writes it) restarts the idle_stop_s count."""
    model = unique_model_id()
    manager, pod, _, t_demand = await ready_idle_pod(env, model)
    while env.clock() < t_demand + 40:
        await step(manager, env.clock, advance=5)
    await env.redis.set(last_request_key(model), int(time.time() * 1000))
    t_used = env.clock()
    while env.clock() < t_used + 55:
        await step(manager, env.clock, advance=5)
        assert in_routing((await env.pool(model)).replicas, pod), env.clock() - t_used
    drained_at = await drain_started(env, manager, model, pod)
    # The stamp is seen at t_used or on the tick 5 s later; idle counts from there.
    assert t_used + 60 <= drained_at <= t_used + 65, drained_at - t_used
    await env.redis.delete(last_request_key(model))


async def test_stamp_from_before_start_is_not_demand(env: FakeServing) -> None:
    """Only a stamp that changes while the manager runs is demand: one left from before
    `start()` does not hold a ready pod past idle_stop_s."""
    model = unique_model_id()
    await env.redis.set(last_request_key(model), 1)
    try:
        manager, pod, _, t_demand = await ready_idle_pod(env, model)
        drained_at = await drain_started(env, manager, model, pod)
        assert t_demand + 60 <= drained_at <= t_demand + 65, drained_at - t_demand
    finally:
        await env.redis.delete(last_request_key(model))


async def test_min_replicas_one_starts_without_wake_and_never_idles(env: FakeServing) -> None:
    """Item 8: `min_replicas: 1` starts a pod without a wake and never idle-stops it."""
    model = unique_model_id()
    manager = env.manager(serving_config({model: {"min_replicas": 1, "idle_stop_s": 30}}))
    await manager.start()
    pod = await bring_up(env, manager, model, wake=False)
    assert not await env.redis.exists(wake_key(model))
    for _ in range(24):  # 120 s, four times idle_stop_s
        await tick(manager, env.clock, advance=5)
        pool = await env.pool(model)
        assert pool.state == "ready" and in_routing(pool.replicas, pod)
    assert alive(env, pod)
    assert len(env.runpod.creates()) == 1
    assert (await row_of(env, pod))["state"] == "ready"


async def test_admin_stop_drains_open_stream_then_terminates(env: FakeServing) -> None:
    """§11 `loom-poolmgr stop <model>` drains: a stream open when draining starts completes
    in full; the pod is terminated after it; the model stays cold, and a wake starts it."""
    model = unique_model_id()
    manager = env.manager(
        serving_config([model], lifecycle={"drain_max_s": 600}, health={"progress_stall_s": 600})
    )
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.redis.delete(wake_key(model))
    (replica,) = (await env.pool(model)).replicas
    stream = await ProxyStream(env.client(), replica, model, max_tokens=100).start()

    await env.admin({"op": "stop", "model": model})
    await step(manager, env.clock, advance=1)
    pool = await env.pool(model)
    assert not in_routing(pool.replicas, pod) and pool.state == "cold"
    while not stream.finished:
        await step(manager, env.clock, advance=1)
        assert alive(env, pod) or stream.finished, "terminated with a request in flight"
    result = await stream.finish()
    assert result.complete and result.completion_tokens == 100, result
    for _ in range(30):
        await step(manager, env.clock, advance=1)
        if not env.runpod.has(pod.pod_id):
            break
    assert not env.runpod.has(pod.pod_id)
    assert (await env.pool(model)).state == "cold"

    await env.wake(model)
    await tick(manager, env.clock, advance=1)
    await wait_until(lambda: len(env.runpod.creates()) == 2)


async def test_admin_drain_one_pod(env: FakeServing) -> None:
    """§11 `loom-poolmgr drain <pod id>`: that pod leaves the hash, drains and is
    terminated with `terminated_by = admin`."""
    model = unique_model_id()
    manager = env.manager(serving_config([model], health={"progress_stall_s": 600}))
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.redis.delete(wake_key(model))
    await env.admin({"op": "drain", "pod_id": pod.pod_id})
    await step(manager, env.clock, advance=1)
    assert not in_routing((await env.pool(model)).replicas, pod)
    for _ in range(30):
        await step(manager, env.clock, advance=1)
        if not env.runpod.has(pod.pod_id):
            break
    assert not env.runpod.has(pod.pod_id)
    row = await row_of(env, pod)
    assert row["state"] == "terminated" and row["terminated_by"] == "admin"
