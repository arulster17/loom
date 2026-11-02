from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from loom_devstack.poolsim import PoolSim, SimConfig, parse_replicas, servable_models
from loom_platform.pool import Replica, get_pool, pool_key, request_wake, wake_key

pytestmark = pytest.mark.redis


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
async def models(redis_client):
    """Random model ids and stream name: db 15 is shared with other test sessions."""
    tag = uuid.uuid4().hex[:8]
    ids = [f"sim-a-{tag}", f"sim-b-{tag}", f"sim-c-{tag}"]
    stream = f"loom:test:usage:{tag}"
    yield ids, stream
    await redis_client.delete(stream, *(pool_key(m) for m in ids), *(wake_key(m) for m in ids))


def replica(pod: str) -> Replica:
    return Replica(pod, f"https://{pod}:8443", pod, f"token-{pod}")


def make_sim(redis_client, models, clock, **kw) -> PoolSim:
    ids, stream = models
    config = SimConfig(
        replicas={ids[0]: (replica("pod-a"),), ids[1]: (replica("pod-b"),)},
        usage_stream=stream,
        cold_start_s=20,
        **kw,
    )
    return PoolSim(redis_client, ids, config, clock=clock)


async def test_seed_ready_with_replicas_else_unavailable(redis_client, models):
    ids, _ = models
    sim = make_sim(redis_client, models, Clock())
    await sim.seed()
    a, c = await get_pool(redis_client, ids[0]), await get_pool(redis_client, ids[2])
    assert a.routable and a.replicas == (replica("pod-a"),)
    assert c.state == "unavailable" and c.retry_after_s == 300 and not c.routable


async def test_cold_wake_warming_countdown_ready(redis_client, models):
    ids, _ = models
    clock = Clock()
    sim = make_sim(redis_client, models, clock)
    await sim.seed()
    await sim.set_state(ids[0], "cold")
    cold = await get_pool(redis_client, ids[0])
    assert cold.state == "cold" and cold.retry_after_s == 20 and not cold.replicas

    await sim.tick()  # no wake requested: stays cold
    assert (await get_pool(redis_client, ids[0])).state == "cold"

    # What the gateway does on a request to a cold model.
    assert await request_wake(redis_client, ids[0])
    await sim.tick()
    warming = await get_pool(redis_client, ids[0])
    assert warming.state == "warming" and warming.retry_after_s == 20 and not warming.routable

    clock.now += 7.5
    await sim.tick()
    assert (await get_pool(redis_client, ids[0])).retry_after_s == 13
    clock.now += 12.5
    await sim.tick()
    ready = await get_pool(redis_client, ids[0])
    assert ready.routable and ready.replicas == (replica("pod-a"),)
    # The wake request is consumed, so the next cold cycle can wake again.
    assert not await redis_client.exists(wake_key(ids[0]))


async def test_unavailable_ignores_wakes(redis_client, models):
    ids, _ = models
    sim = make_sim(redis_client, models, Clock())
    await sim.seed()
    await sim.set_state(ids[1], "unavailable")
    await request_wake(redis_client, ids[1])
    assert not await sim.wake(ids[1])
    await sim.tick()
    pool = await get_pool(redis_client, ids[1])
    assert pool.state == "unavailable" and pool.retry_after_s == 300
    await sim.set_state(ids[1], "unavailable", retry_after_s=42)
    assert (await get_pool(redis_client, ids[1])).retry_after_s == 42


async def test_model_without_replica_cannot_wake_or_be_ready(redis_client, models):
    ids, _ = models
    sim = make_sim(redis_client, models, Clock())
    await sim.seed()
    await sim.set_state(ids[2], "cold")
    assert not await sim.wake(ids[2])
    with pytest.raises(ValueError):
        await sim.set_state(ids[2], "ready")


async def test_idle_stop_uses_the_usage_stream(redis_client, models):
    ids, stream = models
    clock = Clock()
    await redis_client.xadd(stream, {"model_id": ids[0]})  # before start: ignored
    sim = make_sim(redis_client, models, clock, idle_stop_s=60)
    await sim.seed()
    await sim.tick()
    clock.now += 50
    await redis_client.xadd(stream, {"model_id": ids[0]})
    await sim.tick()  # ids[0] used at t+50; ids[1] idle since t
    clock.now += 20
    await sim.tick()
    assert (await get_pool(redis_client, ids[0])).state == "ready"
    assert (await get_pool(redis_client, ids[1])).state == "cold"
    clock.now += 50
    await sim.tick()
    assert (await get_pool(redis_client, ids[0])).state == "cold"


async def test_adopts_warming_set_elsewhere(redis_client, models):
    ids, _ = models
    clock = Clock()
    sim = make_sim(redis_client, models, clock)
    await sim.seed()
    other = make_sim(redis_client, models, clock)  # e.g. `loom-poolsim set ... warming`
    await other.set_state(ids[0], "warming", retry_after_s=5)
    await sim.tick()
    clock.now += 5
    await sim.tick()
    assert (await get_pool(redis_client, ids[0])).routable


async def test_run_reacts_to_published_wake(redis_client, models):
    ids, _ = models
    sim = make_sim(redis_client, models, asyncio.get_running_loop().time)
    sim.config.cold_start_s = 1
    sim.config.tick_s = 0.05
    stop = asyncio.Event()
    task = asyncio.create_task(sim.run(stop))
    try:
        for _ in range(100):
            if (await get_pool(redis_client, ids[0])).routable:
                break
            await asyncio.sleep(0.02)
        await sim.set_state(ids[0], "cold")
        await request_wake(redis_client, ids[0])
        states = set()
        for _ in range(200):
            pool = await get_pool(redis_client, ids[0])
            states.add(pool.state)
            if pool.routable:
                break
            await asyncio.sleep(0.02)
        assert {"warming", "ready"} <= states
    finally:
        stop.set()
        await asyncio.wait_for(task, 5)


def test_parse_replicas(tmp_path: Path):
    (tmp_path / "mock-pod-1.token").write_text("tok1\n")
    (tmp_path / "mock-pod-2.token").write_text("tok2")
    got = parse_replicas(
        ["qwen3-8b=mock-pod-1@https://mock-pod-1:8443,m2=mock-pod-2@https://mock-pod-2:8443"],
        tmp_path,
    )
    assert got == {
        "qwen3-8b": (Replica("mock-pod-1", "https://mock-pod-1:8443", "mock-pod-1", "tok1"),),
        "m2": (Replica("mock-pod-2", "https://mock-pod-2:8443", "mock-pod-2", "tok2"),),
    }
    with pytest.raises(ValueError):
        parse_replicas(["qwen3-8b=https://x"], tmp_path)


def test_servable_models_from_registry():
    assert servable_models() == ["qwen3-8b", "llama-3.3-70b-instruct-fp8"]
