"""Restart rules (docs/phase1-m2-spec.md §11 "Failure and restart rules"; §5 "Restart and
adoption"):

- an adopted pod counts as demand at adoption time: it is not idle-drained on the first
  ticks after a restart;
- holds survive a restart: `crash_loop` and `price_too_high` are recomputed from the last
  hour's rows, counting only failures after the model's last `reset`;
- adoption takes the model out of the pool hash before re-keying its pods, so the hash
  never offers a token the proxy refuses;
- outside `LOOM_ENV=dev`, `LOOM_POOLMGR_STATE_DIR` is required.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    proxy_get,
    serving_config,
    step,
    tick,
    unique_model_id,
)
from poolmgr_faults import BootstrapSpy

from loom_platform.pool import wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.settings import PoolmgrSettings


async def test_adopted_pod_is_not_idle_drained_right_after_restart(env: FakeServing) -> None:
    """The manager was down for longer than idle_stop_s; the adopted pod counts as demand
    at adoption, so it stays routed through the first idle_stop_s after the restart (and
    only then idles out)."""
    model = unique_model_id()
    config = serving_config({model: {"idle_stop_s": 60}})
    first = env.manager(config)
    await first.start()
    pod = await bring_up(env, first, model)
    await env.redis.delete(wake_key(model))
    await first.stop()
    env.clock.advance(300)  # down for five idle periods

    second = env.manager(config)
    await second.start()
    adopted_at = env.clock()
    while env.clock() < adopted_at + 55:
        await step(second, env.clock, advance=5)
        pool = await env.pool(model)
        assert [r.id for r in pool.replicas] == [pod.pod_id], (
            f"idle-drained {env.clock() - adopted_at:.0f} s after adoption"
        )
    for _ in range(4):
        await step(second, env.clock, advance=5)
    assert pod.pod_id not in [r.id for r in (await env.pool(model)).replicas]


async def test_reset_survives_a_restart(env: FakeServing) -> None:
    """After a crash loop and `reset`, a manager restarted within the hour does not put
    the model back into `crash_loop`: only failures after the reset count."""
    model = unique_model_id()
    config = serving_config([model], lifecycle={"max_replacements_per_hour": 2})
    first = env.manager(config)
    await first.start()
    env.pods.behaviour.fail_bootstrap = True
    for _ in range(15):
        await env.wake(model)
        await tick(first, env.clock, advance=5)
    assert (await env.pool(model)).reason == "crash_loop"
    creates = len(env.runpod.creates())
    env.pods.behaviour.fail_bootstrap = False
    await env.redis.delete(wake_key(model))
    await env.admin({"op": "reset", "model": model})
    await tick(first, env.clock, advance=5)
    assert (await env.pool(model)).reason is None
    await first.stop()

    second = env.manager(config)
    await second.start()
    assert (await env.pool(model)).reason != "crash_loop", "crash_loop came back on restart"
    for _ in range(8):
        await env.wake(model)
        await tick(second, env.clock, advance=5)
        if (await env.pool(model)).routable:
            break
    pool = await env.pool(model)
    assert pool.routable and pool.reason is None, pool
    assert len(env.runpod.creates()) == creates + 1


async def test_price_too_high_hold_survives_a_restart(env: FakeServing) -> None:
    """`price_too_high` is restored on restart: no new create until `reset`, even with the
    price back within the ratio."""
    model = unique_model_id()
    config = serving_config([model])
    first = env.manager(config, listed_usd_per_hr=1.0)
    await first.start()
    env.runpod.cost_per_hr = 2.0
    await env.wake(model)
    await tick(first, env.clock)
    assert (await env.pool(model)).reason == "price_too_high"
    await first.stop()
    env.runpod.cost_per_hr = 1.09

    second = env.manager(config, listed_usd_per_hr=1.0)
    await second.start()
    for _ in range(4):
        await env.wake(model)
        await tick(second, env.clock, advance=10)
    assert len(env.runpod.creates()) == 1, "created again after a restart without a reset"
    pool = await env.pool(model)
    assert (pool.state, pool.reason) == ("unavailable", "price_too_high")

    await env.admin({"op": "reset", "model": model})
    for _ in range(6):
        await env.wake(model)
        await tick(second, env.clock, advance=1)
        if (await env.pool(model)).routable:
            break
    assert (await env.pool(model)).routable
    assert len(env.runpod.creates()) == 2


async def test_adoption_takes_model_out_of_hash_before_rekey(env: FakeServing) -> None:
    """The hash left by the stopped manager offers the old token; adoption removes the
    pod from the hash before it re-keys it, so at no point (before or right after the
    proxy reloads) does the hash offer a token the proxy refuses."""
    model = unique_model_id()
    config = serving_config([model])
    first = env.manager(config)
    await first.start()
    pod = await bring_up(env, first, model)
    old_token = pod.token
    await first.stop()
    assert [r.token for r in (await env.pool(model)).replicas] == [old_token]

    spy = BootstrapSpy(env, model)
    second = env.manager(config)
    await second.start()
    assert spy.before and spy.after, "adoption did not re-key the pod"
    for pod_id, pool in spy.before:
        assert pod_id not in [r.id for r in pool.replicas], (
            "still in the pool hash when its re-key began"
        )
    for _, pool in spy.after:
        assert old_token not in [r.token for r in pool.replicas], (
            "the hash offered the old token after the proxy reloaded"
        )
    pool = await env.pool(model)
    assert [r.token for r in pool.replicas] == [pod.token] != [old_token]
    assert (await proxy_get(env.client(), pool.replicas[0], "/health")).status_code == 200


def _settings(tmp_path: Path, **kw: object) -> PoolmgrSettings:
    files = {
        name: tmp_path / name
        for name in (
            "poolmgr_ca_cert",
            "poolmgr_ca_key",
            "poolmgr_client_cert",
            "poolmgr_client_key",
            "poolmgr_ssh_key",
            "poolmgr_proxy_bundle",
        )
    }
    return PoolmgrSettings(poolmgr_lease_bucket="loom-leases", **files, **kw)  # type: ignore[arg-type]


def test_state_dir_required_outside_dev(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§11: outside LOOM_ENV=dev, a missing LOOM_POOLMGR_STATE_DIR is a production
    problem (pods' known_hosts must survive a restart); in dev it is not."""
    for name in list(os.environ):
        if name.startswith("LOOM_") or name == "RUNPOD_API_KEY":
            monkeypatch.delenv(name)
    missing = _settings(tmp_path, env="prod").production_problems()
    assert any("LOOM_POOLMGR_STATE_DIR" in p for p in missing), missing
    given = _settings(tmp_path, env="prod", poolmgr_state_dir=tmp_path).production_problems()
    assert not any("LOOM_POOLMGR_STATE_DIR" in p for p in given), given
    assert _settings(tmp_path, env="dev").production_problems() == []
