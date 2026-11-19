"""A26: the pool manager's M3 gauges (docs/phase1-m3-spec.md §7, §5.3 P2a/P2b/P3;
contracts §10).

- `loom_poolmgr_ca_expiry_seconds`: seconds until the pod CA's `notAfter`, set at start
  and each tick (by the manager's clock, the one its CA refusal uses).
- `loom_poolmgr_pool_reason{model,reason}`: 1 for each model's current pool-hash
  `reason`, 0 for the other reasons, rewritten each tick.
- Every counter an alarm reads exists at 0 for each label value from process start:
  `loom_poolmgr_creates_total{model,result}` (P2a reads `result="spend_cap"`).

The process-start checks run `loom-poolmgr run` (`poolmgr_m3_kit.PoolmgrRun`) and read the
metrics endpoint; the per-tick checks run a manager on the fakes of `poolmgr_fakes` with
its `metrics=` registry.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from decimal import Decimal

import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    delete_pod_rows,
    deployment,
    env_fixture,  # noqa: F401
    insert_pod_row,
    serving_config,
    tick,
    unique_model_id,
)
from poolmgr_m3_kit import (
    CREATE_RESULTS,
    REASONS,
    PoolmgrRun,
    model_keys_fixture,  # noqa: F401
    poolmgr_run_fixture,  # noqa: F401
    sample,
    write_ca,
)

from loom_platform.pool import PoolState
from loom_platform.serving import ServingConfig
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.manager import PoolManager
from loom_poolmgr.metrics import PoolmgrMetrics
from loom_poolmgr.pki import Pki

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

MID_NOVEMBER = dt.datetime(2026, 11, 15, 12, 0, tzinfo=dt.UTC)
HOUR = dt.timedelta(hours=1)


def metered_manager(
    env: FakeServing,
    config: ServingConfig,
    *,
    servable: tuple[str, ...] = (),
    listed_usd_per_hr: float = 1.0,
) -> tuple[PoolManager, PoolmgrMetrics]:
    """`env.manager(config)` with a metrics registry the test reads."""
    metrics = PoolmgrMetrics()
    manager = PoolManager(
        config,
        deployments={
            m: deployment(config, m, listed_usd_per_hr=listed_usd_per_hr) for m in config.models
        },
        provider=env.provider(config),
        pod_exec=env.pods,
        redis=env.redis,
        database_url=env.database_url,
        leases=env.leases,
        pki=env.pki,
        ssh=env.ssh,
        bundle=env.bundle,
        servable=servable,
        clock=env.clock,
        sleep=env.clock.sleep,
        admin_queue=env.admin_queue,
        metrics=metrics,
    )
    env.managers.append(manager)
    return manager, metrics


def value(metrics: PoolmgrMetrics, name: str, **labels: str) -> float | None:
    v = metrics.registry.get_sample_value(name, labels)
    return None if v is None else float(v)


def reason_gauges(metrics: PoolmgrMetrics, model: str) -> dict[str, float | None]:
    return {r: value(metrics, "loom_poolmgr_pool_reason", model=model, reason=r) for r in REASONS}


def assert_follows(metrics: PoolmgrMetrics, pool: PoolState) -> None:
    """The gauge says exactly what the pool hash says."""
    want = {r: 1.0 if pool.reason == r else 0.0 for r in REASONS}
    got = reason_gauges(metrics, pool.model_id)
    assert got == want, f"pool {pool.state}/{pool.reason}: gauges {got}"


def ca_seconds_left(env: FakeServing) -> float:
    return env.pki.ca.cert.not_valid_after_utc.timestamp() - env.clock()


# ------------------------------------------------------------------ process start


def test_at_start_the_gauges_exist_and_alarm_counters_are_zero(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """`loom-poolmgr run`, before anything happened: `ca_expiry_seconds` is the CA's time
    left, every `creates_total{model,result}` is 0 and the cold model's `pool_reason` is
    0 for every reason."""
    h = poolmgr_run
    ca = write_ca(h.tmp / "ca40", days=40)
    h.set_ca(ca)
    h.live(pg_url)
    h.serve_metrics()
    seen: dict[str, object] = {}

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        seen["metrics"] = await run.scrape()
        seen["at"] = time.time()

    h.scenario = scenario
    assert h.main("run") in (0, None)
    text = str(seen["metrics"])
    left = ca.ca.cert.not_valid_after_utc.timestamp() - float(seen["at"])  # type: ignore[arg-type]
    expiry = sample(text, "loom_poolmgr_ca_expiry_seconds")
    assert expiry is not None, "no loom_poolmgr_ca_expiry_seconds"
    assert abs(expiry - left) < 120, (expiry, left)
    creates = {
        r: sample(text, "loom_poolmgr_creates_total", model=h.model, result=r)
        for r in CREATE_RESULTS
    }
    assert creates == dict.fromkeys(CREATE_RESULTS, 0.0), creates
    reasons = {
        r: sample(text, "loom_poolmgr_pool_reason", model=h.model, reason=r) for r in REASONS
    }
    assert reasons == dict.fromkeys(REASONS, 0.0), reasons


# ------------------------------------------------------------------ CA expiry


async def test_ca_expiry_matches_not_after_at_start_and_each_tick(
    env: FakeServing, model_keys: list[str]
) -> None:
    model = unique_model_id()
    model_keys.append(model)
    files = write_ca(env.root / "ca40", days=40, now=env.clock())
    env.pki = Pki.load(files.ca_cert, files.ca_key, files.client_cert, files.client_key)
    manager, metrics = metered_manager(env, serving_config([model]))
    await manager.start()
    got = value(metrics, "loom_poolmgr_ca_expiry_seconds")
    assert got is not None, "loom_poolmgr_ca_expiry_seconds not set at start"
    assert abs(got - ca_seconds_left(env)) <= 2, (got, ca_seconds_left(env))
    assert abs(got - 40 * 86400) <= 600, got
    for advance in (3600, 86400, 5 * 86400):
        await tick(manager, env.clock, advance=advance)
        got = value(metrics, "loom_poolmgr_ca_expiry_seconds")
        assert got is not None
        assert abs(got - ca_seconds_left(env)) <= 2, (advance, got, ca_seconds_left(env))
    # Below alarm P3's 30 days now (40 - 6 days and an hour).
    assert got < 30 * 86400


# ------------------------------------------------------------------ pool reason


def no_stock(env: FakeServing) -> None:
    env.runpod.stock = [["EUR-IS-2"]]  # no US datacenter has the GPU


async def test_pool_reason_follows_the_pool_hash_each_tick(
    env: FakeServing, model_keys: list[str]
) -> None:
    """cold (no reason) -> `no_capacity` -> warming -> ready: on every tick the gauges of
    the model are 1 for the hash's reason and 0 for the rest; a servable model missing
    from the serving file is `not_configured`."""
    model = unique_model_id()
    missing = unique_model_id("llama")
    model_keys += [model, missing]
    config = serving_config([model], capacity={"backoff_s": [10], "unavailable_retry_after_s": 300})
    manager, metrics = metered_manager(env, config, servable=(model, missing))
    await manager.start()
    await tick(manager, env.clock, advance=1)
    assert (await env.pool(model)).state == "cold"
    assert_follows(metrics, await env.pool(model))
    assert_follows(metrics, await env.pool(missing))
    assert (await env.pool(missing)).reason == "not_configured"

    no_stock(env)
    for _ in range(3):
        await env.wake(model)
        await tick(manager, env.clock, advance=1)
        assert_follows(metrics, await env.pool(model))
    assert (await env.pool(model)).reason == "no_capacity"

    env.runpod.stock = None  # stock everywhere again
    await env.wake(model)
    await tick(manager, env.clock, advance=15)  # past the 10 s backoff
    seen = {(await env.pool(model)).state}
    assert_follows(metrics, await env.pool(model))
    await bring_up(env, manager, model)
    pool = await env.pool(model)
    seen.add(pool.state)
    assert pool.state == "ready", seen
    assert_follows(metrics, pool)
    assert_follows(metrics, await env.pool(missing))


async def test_spend_cap_reason_gauge_is_set_and_cleared(
    env: FakeServing, model_keys: list[str]
) -> None:
    """P2b's series: `pool_reason{reason="spend_cap"}` is 1 while the model is held at
    the cap and back to 0 once a reset lets it create again."""
    env.clock.now = MID_NOVEMBER.timestamp()
    model = unique_model_id()
    model_keys.append(model)
    other = f"history-{uuid.uuid4().hex[:8]}"
    await insert_pod_row(
        env.database_url,
        model_id=other,
        pod_id=f"hist{uuid.uuid4().hex[:12]}",
        state="terminated",
        cost_per_hr_micros=int(Decimal("0.9") * 1_000_000),
        created_at=MID_NOVEMBER - 2 * HOUR,
        terminated_at=MID_NOVEMBER - HOUR,
        terminated_by="manager",
    )
    config = serving_config(
        [model],
        spend={"monthly_cap_usd": 1.0},
        lifecycle={"lease_ttl_s": 3600, "lease_renew_every_s": 60},
    )
    manager, metrics = metered_manager(env, config, listed_usd_per_hr=1.0)
    await manager.start()
    await env.wake(model)
    await tick(manager, env.clock, advance=5)
    pool = await env.pool(model)
    assert pool.reason == "spend_cap"
    assert value(metrics, "loom_poolmgr_pool_reason", model=model, reason="spend_cap") == 1.0
    assert_follows(metrics, pool)

    await delete_pod_rows(env.database_url, other)
    await env.admin({"op": "reset", "model": model})
    await env.wake(model)
    await tick(manager, env.clock, advance=5)
    pool = await env.pool(model)
    assert pool.reason != "spend_cap", pool
    assert value(metrics, "loom_poolmgr_pool_reason", model=model, reason="spend_cap") == 0.0
    assert_follows(metrics, pool)
