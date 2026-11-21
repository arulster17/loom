"""No capacity, price refusal and ambiguous creates (docs/phase1-m2-spec.md §13 items 12
and 13; §5 "Create"; §7 rows `no_capacity`, `price_too_high` and `warming` in their
order; §11 `reset`).

Backoff in these tests: `capacity.backoff_s: [10, 20, 40]` (§11: the configured one); ticks
every fake second, so an attempt's time is exact to the second.
"""

from __future__ import annotations

import threading

import pytest
from poolmgr_fakes import (
    FakeServing,
    drive,
    env_fixture,  # noqa: F401
    serving_config,
    settle,
    step,
    tick,
    unique_model_id,
)

from loom_platform.pool import wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import PoolManager

NO_CAPACITY = "There are no instances currently available"


def no_capacity(env: FakeServing, how: str) -> None:
    """RunPod answers no capacity: no US datacenter lists stock, or the create is refused
    with RunPod's no-capacity message (RunpodApiError.no_capacity)."""
    if how == "stock":
        env.runpod.stock = [["EUR-IS-2"]]
    else:
        env.runpod.fail[("POST", "/v1/pods")] = [(500, NO_CAPACITY)] * 50


async def attempts(env: FakeServing, model: str) -> list[float]:
    """When each create attempt was made: one serving_pods row each (written before
    RunPod is called), at the fake clock."""
    return [r["created_at"].timestamp() for r in await env.rows(model)]


async def run_ticks(
    env: FakeServing, manager: PoolManager, model: str, seconds: int, *, wake: bool
) -> None:
    for _ in range(seconds):
        if wake:
            await env.wake(model)
        await tick(manager, env.clock, advance=1)


async def ready_within(env: FakeServing, manager: PoolManager, model: str, ticks: int) -> bool:
    for _ in range(ticks):
        await env.wake(model)
        await tick(manager, env.clock, advance=1)
        if (await env.pool(model)).routable:
            return True
    return False


async def test_min_replicas_is_demand_for_capacity_retries(env: FakeServing) -> None:
    """Item 12 / §5: retries continue on the backoff schedule with no wake key when
    `min_replicas >= 1` (always-on is demand)."""
    model = unique_model_id()
    manager = env.manager(serving_config({model: {"min_replicas": 1}}))
    await manager.start()
    no_capacity(env, "stock")
    t0 = env.clock()
    await tick(manager, env.clock)
    await run_ticks(env, manager, model, 75, wake=False)
    offsets = [round(t - t0) for t in await attempts(env, model)]
    assert offsets == [0, 10, 30, 70], offsets
    assert not await env.redis.exists(wake_key(model))


async def test_capacity_back_ends_the_hold(env: FakeServing) -> None:
    """After no capacity, the next attempt on the backoff schedule that finds a GPU makes a
    ready pod; the pool loses its reason."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    no_capacity(env, "stock")
    await env.wake(model)
    await tick(manager, env.clock)
    assert (await env.pool(model)).reason == "no_capacity"
    env.runpod.stock = [["US-TX-3"]]
    assert await ready_within(env, manager, model, 20)
    pool = await env.pool(model)
    assert pool.state == "ready" and pool.reason is None
    (body,) = env.runpod.creates()
    assert body["dataCenterIds"] == ["US-TX-3"]


async def test_create_in_flight_after_no_capacity_is_warming(env: FakeServing) -> None:
    """§7 order: `no_capacity` holds only while nothing is in flight; once the next
    attempt's create is under way the pool is `warming`."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    no_capacity(env, "stock")
    await env.wake(model)
    await tick(manager, env.clock)
    assert (await env.pool(model)).reason == "no_capacity"
    env.runpod.stock = [["US-TX-3"]]
    gate = threading.Event()
    env.runpod.create_gate = gate  # the next POST /pods waits: a create in flight
    for _ in range(12):
        await env.wake(model)
        await step(manager, env.clock, advance=1)
        if len(await env.rows(model)) == 2:
            break
    await env.wake(model)
    await step(manager, env.clock, advance=1)
    pool = await env.pool(model)
    assert (pool.state, pool.reason) == ("warming", None), pool
    gate.set()
    await drive(env.clock, lambda: len(env.runpod.creates()) == 1)
    await settle(manager, env.clock)
    assert await ready_within(env, manager, model, 5)


async def test_price_above_ratio_terminates_and_holds_until_reset(env: FakeServing) -> None:
    """Item 13: a pod priced above bench/prices.yaml x max_price_ratio is terminated, its
    row goes failed (`price_too_high`), and the pool `unavailable` (`price_too_high`):
    wakes do not create again until an admin `reset` (§11) clears the hold."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]), listed_usd_per_hr=1.0)  # ratio 1.25
    await manager.start()
    env.runpod.cost_per_hr = 2.0
    await env.wake(model)
    await tick(manager, env.clock)
    assert len(env.runpod.creates()) == 1
    (pod,) = env.pods.snapshot()
    assert not env.runpod.has(pod.pod_id)
    assert env.runpod.terminate_requests(pod.pod_id) >= 1
    (row,) = await env.rows(model)
    assert row["failure"] == "price_too_high"
    assert row["state"] in ("failed", "terminated")
    pool = await env.pool(model)
    assert (pool.state, pool.reason, pool.retry_after_s) == ("unavailable", "price_too_high", 300)
    env.runpod.cost_per_hr = 1.09  # back within the ratio: only the hold stops a create
    for _ in range(6):
        await env.wake(model)
        await tick(manager, env.clock, advance=10)
    assert len(env.runpod.creates()) == 1
    assert (await env.pool(model)).reason == "price_too_high"

    await env.admin({"op": "reset", "model": model})
    assert await ready_within(env, manager, model, 6)
    assert len(env.runpod.creates()) == 2
    assert (await env.pool(model)).reason is None


async def test_price_within_ratio_is_accepted(env: FakeServing) -> None:
    """Item 13, nothing else: 1.09 $/h against a listed 1.00 $/h (ratio 1.25) serves."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]), listed_usd_per_hr=1.0)
    await manager.start()
    assert await ready_within(env, manager, model, 6)
    assert len(env.runpod.creates()) == 1


@pytest.mark.parametrize("answer", [502, 503, "timeout"])
async def test_ambiguous_create_finds_pod_by_name_and_terminates_it(
    env: FakeServing, answer: int | str
) -> None:
    """Item 13: a create that times out or answers 5xx may still have made the pod: it is
    found by name and terminated, and no live row claims it."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))
    await manager.start()
    env.runpod.ambiguous = [answer]
    await env.wake(model)
    await tick(manager, env.clock)
    first = env.runpod.creates()[0]
    (made,) = [p for p in env.pods.snapshot() if p.name == first["name"]]
    assert not env.runpod.has(made.pod_id), "the pod an ambiguous create made was left"
    assert env.runpod.terminate_requests(made.pod_id) >= 1
    for row in await env.rows(model):
        if row["pod_name"] == first["name"]:
            assert row["state"] in ("failed", "terminated"), row
