"""Health rules (docs/phase1-m2-spec.md §13 item 10; §5 "Health rules"; §11 "Timing
rules"): each fires on its condition and on nothing else, and a replacement is created
only when there is demand.

Timings (poolmgr_fakes.serving_config): pod state and proxy health every 5 s, 3 failures
in a row, 30 s grace, progress stall 20 s, bootstrap stall 60 s. Demand is a wake key,
refreshed before each tick (or deleted for the no-demand cases).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from poolmgr_fakes import (
    FakePod,
    FakeServing,
    ProxyStream,
    bring_up,
    drive,
    env_fixture,  # noqa: F401
    serving_config,
    settle,
    step,
    tick,
    unique_model_id,
    wait_until,
)

from loom_platform.pool import Replica, wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr import scripts
from loom_poolmgr.manager import PoolManager


def in_routing(replicas: tuple[Replica, ...], pod: FakePod) -> bool:
    return any(r.id == pod.pod_id for r in replicas)


async def routed(env: FakeServing, model: str, pod: FakePod) -> bool:
    return in_routing((await env.pool(model)).replicas, pod)


async def row_of(env: FakeServing, pod: FakePod) -> dict[str, Any]:
    """The pod's serving_pods row (by name: written before RunPod is called)."""
    (row,) = [r for r in await env.rows(pod.model) if r["pod_name"] == pod.name]
    return row


async def set_demand(env: FakeServing, model: str, demand: bool) -> None:
    if demand:
        await env.wake(model)
    else:
        await env.redis.delete(wake_key(model))


async def ready(env: FakeServing, model: str, **config: Any) -> tuple[PoolManager, FakePod]:
    manager = env.manager(serving_config([model], **config))
    await manager.start()
    pod = await bring_up(env, manager, model)
    return manager, pod


async def replacements(
    env: FakeServing, manager: PoolManager, model: str, demand: bool, ticks: int = 4
) -> int:
    """Tick on with or without demand; the creates made after the first pod."""
    for _ in range(ticks):
        await set_demand(env, model, demand)
        await tick(manager, env.clock, advance=5)
    return len(env.runpod.creates()) - 1


# ----------------------------------------------------------------- pod state


@pytest.mark.parametrize("loss", ["gone", "exited"])
@pytest.mark.parametrize("demand", [True, False])
async def test_pod_lost_is_out_of_routing_and_replaced_only_with_demand(
    env: FakeServing, loss: str, demand: bool
) -> None:
    """Pod state rule: gone from the listing or EXITED: out of routing now,
    `terminated_by = runpod`; replaced if there is demand, not otherwise."""
    model = unique_model_id()
    manager, pod = await ready(env, model)
    await set_demand(env, model, demand)
    if loss == "gone":
        env.runpod.remove_pod(pod.pod_id)
    else:
        env.runpod.set_status(pod.pod_id, "EXITED")
    await set_demand(env, model, demand)
    await tick(manager, env.clock, advance=5)
    assert not await routed(env, model, pod)
    row = await row_of(env, pod)
    assert row["state"] == "terminated" and row["terminated_by"] == "runpod"
    assert await replacements(env, manager, model, demand) == (1 if demand else 0)
    if not demand:
        assert (await env.pool(model)).state == "cold"


async def test_running_pod_is_left_alone(env: FakeServing) -> None:
    """Nothing else: a RUNNING, healthy, idle pod survives a minute of ticks."""
    model = unique_model_id()
    manager, pod = await ready(env, model)
    for _ in range(12):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        assert await routed(env, model, pod)
    assert env.runpod.terminate_requests(pod.pod_id) == 0
    assert (await row_of(env, pod))["state"] == "ready"
    assert len(env.runpod.creates()) == 1


# ----------------------------------------------------------------- proxy health


async def test_two_failed_health_checks_change_nothing(env: FakeServing) -> None:
    model = unique_model_id()
    manager, pod = await ready(env, model)
    assert pod.engine is not None
    pod.engine.set_healthy(False)
    for _ in range(2):
        await tick(manager, env.clock, advance=5)
        assert await routed(env, model, pod)
    pod.engine.set_healthy(True)
    for _ in range(4):
        await tick(manager, env.clock, advance=5)
        assert await routed(env, model, pod)
    assert (await row_of(env, pod))["state"] == "ready"
    assert env.runpod.terminate_requests(pod.pod_id) == 0


async def test_three_failed_checks_then_healthy_within_grace_stays(env: FakeServing) -> None:
    """3 failures in a row: out of routing; healthy again within unhealthy_grace_s: back
    in routing, never failed or replaced."""
    model = unique_model_id()
    manager, pod = await ready(env, model)
    assert pod.engine is not None
    pod.engine.set_healthy(False)
    for _ in range(3):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    assert not await routed(env, model, pod)
    assert (await row_of(env, pod))["state"] not in ("failed", "terminated")
    pod.engine.set_healthy(True)
    await env.wake(model)
    await tick(manager, env.clock, advance=5)  # 20 s after the first failure
    for _ in range(3):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    assert await routed(env, model, pod)
    assert (await row_of(env, pod))["state"] == "ready"
    assert env.runpod.terminate_requests(pod.pod_id) == 0
    assert len(env.runpod.creates()) == 1


@pytest.mark.parametrize("demand", [True, False])
async def test_unhealthy_past_grace_fails_and_is_replaced_only_with_demand(
    env: FakeServing, demand: bool
) -> None:
    model = unique_model_id()
    manager, pod = await ready(env, model)
    assert pod.engine is not None
    pod.engine.set_healthy(False)
    for _ in range(12):  # 60 s: 3 failures (15 s) plus a 30 s grace, with room
        await set_demand(env, model, demand)
        await tick(manager, env.clock, advance=5)
    assert not env.runpod.has(pod.pod_id)
    row = await row_of(env, pod)
    assert row["state"] == "terminated" and row["failure"]
    assert await replacements(env, manager, model, demand) == (1 if demand else 0)
    if demand:
        for _ in range(4):
            if (await env.pool(model)).routable:
                break
            await tick(manager, env.clock, advance=5)
        assert (await env.pool(model)).routable


# ----------------------------------------------------------------- engine progress


@pytest.mark.parametrize("demand", [True, False])
async def test_progress_stall_fails_and_is_replaced_only_with_demand(
    env: FakeServing, demand: bool
) -> None:
    """Requests running and vllm:generation_tokens_total flat for progress_stall_s:
    failed; replaced only with demand."""
    model = unique_model_id()
    manager, pod = await ready(env, model)
    assert pod.engine is not None
    (replica,) = (await env.pool(model)).replicas
    stream = await ProxyStream(env.client(), replica, model, max_tokens=5000).start()
    pod.engine.stall()
    t_stall = env.clock()
    while env.clock() < t_stall + 15:
        await set_demand(env, model, demand)
        await step(manager, env.clock, advance=5)
    assert await routed(env, model, pod), "failed before progress_stall_s"
    while env.clock() < t_stall + 45 and env.runpod.has(pod.pod_id):
        await set_demand(env, model, demand)
        await step(manager, env.clock, advance=5)
    await wait_until(lambda: not env.runpod.has(pod.pod_id))
    await stream.cancel()
    assert (await row_of(env, pod))["failure"]
    assert await replacements(env, manager, model, demand) == (1 if demand else 0)


async def test_progressing_stream_is_not_a_stall(env: FakeServing) -> None:
    """Nothing else: a long stream that keeps producing tokens is never a stall."""
    model = unique_model_id()
    manager, pod = await ready(env, model)
    (replica,) = (await env.pool(model)).replicas
    stream = await ProxyStream(env.client(), replica, model, max_tokens=3000).start()
    for _ in range(10):  # 50 s of fake time, real time for tokens in between
        await env.wake(model)
        await step(manager, env.clock, advance=5)
        await asyncio.sleep(0.15)
        assert await routed(env, model, pod)
    assert not stream.finished or stream.result.error is None
    await stream.cancel()
    assert (await row_of(env, pod))["state"] == "ready"
    assert len(env.runpod.creates()) == 1


async def test_idle_engine_is_not_a_stall(env: FakeServing) -> None:
    """Nothing else: no request running and tokens flat is idle, not a stall."""
    model = unique_model_id()
    manager, pod = await ready(env, model)
    for _ in range(12):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
    assert await routed(env, model, pod)
    assert (await row_of(env, pod))["state"] == "ready"


# ----------------------------------------------------------------- bootstrap


@pytest.mark.parametrize("demand", [True, False])
async def test_bootstrap_stall_fails_and_is_replaced_only_with_demand(
    env: FakeServing, demand: bool
) -> None:
    """Bootstrap stall: serve_engine (rendered with ENGINE_STALL_S = bootstrap_stall_s)
    exits non-zero when the engine stops making progress: failed, terminated; replaced
    only with demand."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    env.pods.behaviour.engine = "hang"
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: any(p.engine_scripts for p in env.pods.snapshot()))
    pod = env.pods.only()
    assert scripts.script_var(pod.engine_scripts[0], "ENGINE_STALL_S") == "60"
    env.pods.behaviour.engine = "ok"  # a replacement boots normally
    await set_demand(env, model, demand)
    env.pods.release_engine(pod.pod_id, mode="fail")
    await settle(manager, env.clock)
    assert not env.runpod.has(pod.pod_id)
    row = await row_of(env, pod)
    assert row["state"] == "terminated" and row["failure"]
    assert await replacements(env, manager, model, demand) == (1 if demand else 0)


async def test_bootstrap_timeout_fails_only_after_bootstrap_timeout_s(env: FakeServing) -> None:
    """Bootstrap timeout: no first token within bootstrap_timeout_s (here 120 s) of the
    create (the row's created_at, §11): failed, terminated. A slow bootstrap below it is
    left alone."""
    model = unique_model_id()
    manager = env.manager(serving_config([model], health={"bootstrap_timeout_s": 120}))
    await manager.start()
    env.pods.behaviour.engine = "hang"
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: len(env.pods.snapshot()) == 1)
    pod = env.pods.only()
    t0 = (await row_of(env, pod))["created_at"].timestamp()
    while env.clock() < t0 + 110:
        await env.wake(model)
        await step(manager, env.clock, advance=10)
    assert env.runpod.has(pod.pod_id), "failed before bootstrap_timeout_s"
    assert (await row_of(env, pod))["state"] in ("creating", "bootstrapping")
    assert (await env.pool(model)).state == "warming"
    env.pods.behaviour.engine = "ok"
    while env.clock() < t0 + 140 and env.runpod.has(pod.pod_id):
        await env.wake(model)
        await step(manager, env.clock, advance=5)
    await wait_until(lambda: not env.runpod.has(pod.pod_id))
    row = await row_of(env, pod)
    assert row["state"] == "terminated" and row["failure"]
    for _ in range(4):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        if len(env.runpod.creates()) == 2:
            break
    assert len(env.runpod.creates()) == 2
