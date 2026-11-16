"""Failure rules (docs/phase1-m2-spec.md §11 "Failure and restart rules"; §5 "Health
rules", "Create", "Idle stop and drain"):

- a pod is lost only after two consecutive listings without it, taken after its id was
  known; a lost pod is terminated before its row is marked;
- an ambiguous create keeps its row counted until a listing shows no pod with its name or
  the pod found by name is terminated;
- an unexpected exception in a create or bootstrap fails the row and terminates the pod
  (by name if its id is unknown), so the model is not left `warming`;
- a failed /metrics read while draining means "try again next tick", until drain_max_s.

`max_replicas` (1 here) is checked after every tick both as live rows and as pods RunPod
still has.
"""

from __future__ import annotations

from typing import Any

import pytest
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
)
from poolmgr_faults import RunpodFaults, metrics_fail

from loom_platform.pool import wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import PoolManager


async def live_rows(env: FakeServing, model: str) -> int:
    return sum(1 for r in await env.rows(model) if r["state"] != "terminated")


async def within_max(env: FakeServing, model: str) -> None:
    assert await live_rows(env, model) <= 1, await env.rows(model)
    assert len(env.runpod.of(model)) <= 1, env.runpod.of(model)


async def row_named(env: FakeServing, model: str, name: str) -> dict[str, Any]:
    (row,) = [r for r in await env.rows(model) if r["pod_name"] == name]
    return row


def routed(pool_replicas: Any, pod: FakePod) -> bool:
    return any(r.id == pod.pod_id for r in pool_replicas)


async def ready_after(env: FakeServing, manager: PoolManager, model: str, ticks: int) -> bool:
    for _ in range(ticks):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        await within_max(env, model)
        if (await env.pool(model)).routable:
            return True
    return False


# ----------------------------------------------------------------- listing lag


async def test_new_pod_missing_from_a_lagging_listing_is_not_lost(env: FakeServing) -> None:
    """RunPod's listing trails the create by one listing: the new pod is not taken for
    lost, no second create is made, and it becomes ready."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    faults = RunpodFaults(env.runpod)
    faults.lag = 1
    for _ in range(10):
        await env.wake(model)
        await tick(manager, env.clock, advance=1)
        await within_max(env, model)
        if (await env.pool(model)).routable:
            break
    assert (await env.pool(model)).routable
    assert faults.listings >= 1, "no listing was lagging"
    assert len(env.runpod.creates()) == 1
    (row,) = await env.rows(model)
    assert row["state"] == "ready" and row["failure"] is None


async def test_pod_missing_from_two_listings_is_terminated_before_marked(env: FakeServing) -> None:
    """A ready pod missing from one listing stays; missing from a second consecutive one it
    is lost: out of routing and terminated, and its row is marked only once the terminate
    went through (with RunPod refusing it, the row stays unmarked and counted)."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    pod = await bring_up(env, manager, model)
    faults = RunpodFaults(env.runpod)
    faults.hidden[pod.pod_id] = -1  # RunPod still has it; the listing leaves it out
    env.runpod.terminate_fails = True

    await env.wake(model)
    await tick(manager, env.clock, advance=5)  # first listing without it
    assert routed((await env.pool(model)).replicas, pod), "lost after a single listing"
    assert (await row_named(env, model, pod.name))["state"] == "ready"

    await env.wake(model)
    await tick(manager, env.clock, advance=5)  # second consecutive listing without it
    assert not routed((await env.pool(model)).replicas, pod)
    assert env.runpod.terminate_requests(pod.pod_id) >= 1, "a lost pod was not terminated"
    assert (await row_named(env, model, pod.name))["state"] != "terminated", (
        "the row was marked terminated although RunPod refused the terminate"
    )
    await within_max(env, model)

    env.runpod.terminate_fails = False
    for _ in range(3):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        await within_max(env, model)
    assert not env.runpod.has(pod.pod_id)
    row = await row_named(env, model, pod.name)
    assert row["state"] == "terminated"


# ----------------------------------------------------------------- ambiguous creates


async def test_ambiguous_create_counts_until_resolved(env: FakeServing) -> None:
    """A create answers 502 but made the pod, and the name lookup cannot run (listings
    fail): its row stays counted, so no second create happens however long, until a
    listing finds the pod by name and it is terminated."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    faults = RunpodFaults(env.runpod)
    env.runpod.ambiguous = [502]
    faults.listing_fails = True
    await env.wake(model)
    await tick(manager, env.clock)
    (first,) = env.pods.snapshot()
    assert env.runpod.has(first.pod_id)
    for _ in range(40):  # 40 s: past every capacity.backoff_s step
        await env.wake(model)
        await step(manager, env.clock, advance=1)
        assert len(env.runpod.creates()) == 1, "created again while the first was unresolved"
        await within_max(env, model)

    faults.listing_fails = False
    for _ in range(12):
        await env.wake(model)
        await tick(manager, env.clock, advance=5)
        await within_max(env, model)
        if (await env.pool(model)).routable:
            break
    assert not env.runpod.has(first.pod_id)
    assert env.runpod.terminate_requests(first.pod_id) >= 1
    assert (await row_named(env, model, first.name))["state"] in ("failed", "terminated")
    assert (await env.pool(model)).routable
    assert len(env.runpod.creates()) == 2


@pytest.mark.parametrize("answer", [503, "timeout"])
async def test_ambiguous_create_with_lagging_name_lookup(
    env: FakeServing, answer: int | str
) -> None:
    """The pod an ambiguous create made is missing from the first listings (so the name
    lookup misses it): it is still terminated, max_replicas holds throughout, and the model
    ends with one ready pod."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    faults = RunpodFaults(env.runpod)
    faults.lag = 1
    env.runpod.ambiguous = [answer]
    await env.wake(model)
    await tick(manager, env.clock)
    (first,) = env.pods.snapshot()
    faults.lag = 0
    for _ in range(60):
        await env.wake(model)
        await tick(manager, env.clock, advance=1)
        await within_max(env, model)
        if (await env.pool(model)).routable and not env.runpod.has(first.pod_id):
            break
    assert not env.runpod.has(first.pod_id), "the ambiguous create's pod was left running"
    assert (await env.pool(model)).routable
    assert len(env.runpod.creates()) == 2


# ----------------------------------------------------------------- unexpected exceptions


@pytest.mark.parametrize("fault", ["create_raises", "db_error"])
async def test_unexpected_exception_fails_row_and_terminates_pod(
    env: FakeServing, fault: str
) -> None:
    """A create that raises something unexpected after RunPod made the pod (the pod id is
    never returned), or a database error right after the create answered (a price that
    overflows `cost_per_hr_micros`): the row ends failed/terminated, the pod is terminated
    (by name when its id is unknown), the model is not left warming, and a later wake
    creates again."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    if fault == "create_raises":
        RunpodFaults(env.runpod).raise_after_create = 1
    else:
        env.runpod.cost_per_hr = 1e14  # $1e14/h: 1e20 micros, beyond a bigint
    await env.wake(model)
    await tick(manager, env.clock)
    (first,) = env.pods.snapshot()
    env.runpod.cost_per_hr = 1.09
    await env.redis.delete(wake_key(model))
    for _ in range(6):
        await tick(manager, env.clock, advance=5)
        if not env.runpod.has(first.pod_id):
            break
    assert not env.runpod.has(first.pod_id), "the pod of the broken create was left running"
    assert (await row_named(env, model, first.name))["state"] in ("failed", "terminated")
    assert (await env.pool(model)).state != "warming", "stuck warming"
    assert await ready_after(env, manager, model, 12), "a later wake did not create again"
    assert len(env.runpod.creates()) == 2


# ----------------------------------------------------------------- drain


async def test_failed_metrics_read_while_draining_waits_until_drain_max(env: FakeServing) -> None:
    """While a pod drains, a /metrics read that fails is "try again next tick": a stream
    open on it completes, and the pod is terminated at drain_max_s, not before."""
    model = unique_model_id()
    manager = env.manager(
        serving_config([model], lifecycle={"drain_max_s": 60}, health={"progress_stall_s": 600})
    )
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.redis.delete(wake_key(model))
    assert pod.engine is not None
    (replica,) = (await env.pool(model)).replicas
    stream = await ProxyStream(env.client(), replica, model, max_tokens=150).start()
    metrics_fail(pod.engine)
    await env.admin({"op": "drain", "pod_id": pod.pod_id})
    await step(manager, env.clock, advance=0.5)
    assert not routed((await env.pool(model)).replicas, pod)
    started = (await row_named(env, model, pod.name))["drain_started_at"].timestamp()
    while not stream.finished:
        await step(manager, env.clock, advance=0.5)
        assert env.runpod.has(pod.pod_id) or stream.finished, "terminated mid-stream"
    result = await stream.finish()
    assert result.complete and result.completion_tokens == 150, result
    while env.clock() < started + 55:
        await step(manager, env.clock, advance=1)
        assert env.runpod.has(pod.pod_id), "terminated before drain_max_s"
    while env.clock() < started + 65 and env.runpod.has(pod.pod_id):
        await step(manager, env.clock, advance=1)
    assert not env.runpod.has(pod.pod_id), "not terminated at drain_max_s"
