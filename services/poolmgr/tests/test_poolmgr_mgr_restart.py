"""Single writer, restart and adoption (docs/phase1-m2-spec.md §13 item 15; §3 "Single
writer"; §5 "Restart and adoption", each row of its table).

A "restart" is `stop()` on one manager (pods keep running) and `start()` on a new one over
the same fakes, database and Redis. `start()` returns once adoption has finished.
"""

from __future__ import annotations

import logging

import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    metric_value,
    proxy_get,
    serving_config,
    tick,
    unique_model_id,
)

from loom_platform.pool import PoolState, Replica, set_pool
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import ManagerLocked
from loom_poolmgr.provider import pod_name


def add_listed_pods(env: FakeServing, model: str) -> dict[str, str]:
    """Pods in the listing that the manager did not make: an orphan serving pod, and three
    it must never touch."""
    t = int(env.clock())
    return {
        "orphan": env.runpod.add_pod(
            name=pod_name(model, t),
            desiredStatus="RUNNING",
            env={"LOOM_MANAGED": "true", "LOOM_ROLE": "serving", "LOOM_MODEL": model},
        )["id"],
        "bench": env.runpod.add_pod(
            name=f"loom-bench-exp-{t + 3600}-abcdef",
            desiredStatus="RUNNING",
            env={"LOOM_MANAGED": "true", "LOOM_EXPERIMENT": "exp"},
        )["id"],
        "unmanaged": env.runpod.add_pod(
            name=pod_name(model, t),  # an exact serving name, but no LOOM_MANAGED=true
            desiredStatus="RUNNING",
            env={"LOOM_MODEL": model},
        )["id"],
        "other": env.runpod.add_pod(name="jupyter-notebook", desiredStatus="RUNNING", env={})["id"],
    }


async def test_second_manager_cannot_start_while_first_holds_lock(env: FakeServing) -> None:
    """§3: `pg_try_advisory_lock(hashtext('loom-poolmgr'))`; a second manager fails to
    start (and does nothing) while the first holds it; it starts once the first stops."""
    model = unique_model_id()
    config = serving_config([model])
    first = env.manager(config)
    await first.start()
    pods = add_listed_pods(env, model)
    second = env.manager(config)
    with pytest.raises(ManagerLocked):
        await second.start()
    assert pods["orphan"] in env.runpod.pods, "a manager that lost the lock acted"
    assert env.runpod.terminate_requests(pods["orphan"]) == 0
    await first.stop()
    third = env.manager(config)
    await third.start()
    await third.stop()


async def test_ready_pod_is_adopted_and_rekeyed(env: FakeServing) -> None:
    """Item 15 / adoption row 1: after a restart a ready pod is re-keyed over SSH (new token
    and certificate, proxy reloaded, lease renewed), health-checked and probed, and put
    back in routing without a new create. The old token is refused, the new one works;
    the pool hash is rewritten from scratch."""
    model = unique_model_id()
    config = serving_config([model])
    first = env.manager(config)
    await first.start()
    pod = await bring_up(env, first, model)
    old = pod.replica()
    old_cert = pod.payloads[-1]["server.crt"]
    leases_before = len(pod.leases)
    client = env.client()
    probes_before = metric_value(
        (await proxy_get(client, old, "/metrics")).text, "vllm:request_success_total"
    )
    await first.stop()
    assert pod.pod_id in env.runpod.pods, "stop() must leave pods running"

    # A stale hash: adoption must not keep anything from it.
    stale = Replica(id="stale", url="https://127.0.0.1:1", server_name="stale", token="stale")
    await set_pool(env.redis, PoolState(model, "ready", 0, (stale, old)))

    second = env.manager(config)
    await second.start()
    assert pod.bootstraps[-1] == "reloaded"
    assert pod.token != old.token
    assert pod.payloads[-1]["server.crt"] != old_cert
    assert len(pod.leases) > leases_before
    pool = await env.pool(model)
    assert pool.state == "ready"
    assert pool.replicas == (pod.replica(),)
    assert (await proxy_get(client, old, "/health")).status_code == 403
    new = pool.replicas[0]
    assert (await proxy_get(client, new, "/health")).status_code == 200
    probes_after = metric_value(
        (await proxy_get(client, new, "/metrics")).text, "vllm:request_success_total"
    )
    assert probes_after > probes_before, "adoption did not probe the pod"
    assert len(env.runpod.creates()) == 1
    (row,) = await env.rows(model)
    assert row["state"] == "ready" and row["pod_id"] == pod.pod_id


async def test_orphan_terminated_and_other_pods_untouched(
    env: FakeServing, caplog: pytest.LogCaptureFixture
) -> None:
    """Item 15 / adoption row 2: a `loom-serve-*` pod with LOOM_MANAGED=true and no row is
    terminated (logged at error). A bench pod, a serving name without LOOM_MANAGED, and
    any other pod are never touched, at start or on later ticks."""
    caplog.set_level(logging.INFO)
    model = unique_model_id()
    pods = add_listed_pods(env, model)
    manager = env.manager(serving_config([model]))
    await manager.start()
    assert pods["orphan"] not in env.runpod.pods
    assert env.runpod.terminate_requests(pods["orphan"]) >= 1
    assert any(
        r.levelno >= logging.ERROR and pods["orphan"] in r.getMessage() for r in caplog.records
    ), "the orphan was not logged at error"
    for _ in range(4):
        await tick(manager, env.clock, advance=10)
    for kind in ("bench", "unmanaged", "other"):
        assert pods[kind] in env.runpod.pods, kind
        assert env.runpod.terminate_requests(pods[kind]) == 0, kind
    assert env.runpod.creates() == []


async def test_row_whose_pod_is_gone_is_marked_terminated_by_runpod(env: FakeServing) -> None:
    """Adoption row 3: a live row whose pod RunPod no longer lists becomes `terminated`,
    `terminated_by = runpod`; nothing routes to it."""
    model = unique_model_id()
    config = serving_config([model])
    first = env.manager(config)
    await first.start()
    pod = await bring_up(env, first, model)
    await first.stop()
    env.runpod.remove_pod(pod.pod_id)
    second = env.manager(config)
    await second.start()
    (row,) = await env.rows(model)
    assert row["state"] == "terminated" and row["terminated_by"] == "runpod"
    pool = await env.pool(model)
    assert pool.replicas == () and pool.state != "ready"


async def test_adoption_failure_fails_and_terminates(env: FakeServing) -> None:
    """Adoption row 1, "If any step fails: failed, terminate": a re-key that fails leaves
    no pod running and nothing routed."""
    model = unique_model_id()
    config = serving_config([model])
    first = env.manager(config)
    await first.start()
    pod = await bring_up(env, first, model)
    await first.stop()
    pod.behaviour.fail_bootstrap = True
    second = env.manager(config)
    await second.start()
    assert pod.pod_id not in env.runpod.pods
    (row,) = await env.rows(model)
    assert row["state"] in ("failed", "terminated") and row["failure"]
    pool = await env.pool(model)
    assert pool.replicas == ()
