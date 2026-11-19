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
    drive,
    env_fixture,  # noqa: F401
    insert_pod_row,
    serving_config,
    step,
    tick,
    unique_model_id,
    wait_until,
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


async def follows_after_next_tick(
    env: FakeServing,
    manager: PoolManager,
    metrics: PoolmgrMetrics,
    *models: str,
    wake: str | None = None,
) -> None:
    """One more tick (1 s on, `wake` woken first), then the gauges of each of `models`
    say what its pool hash says. The gauge is rewritten each tick: a hash written by the
    previous tick's work after that tick's rewrite shows from this tick on. The tick
    changes no reason (the states it follows are held: cold, in backoff, at the cap,
    warming or ready)."""
    if wake is not None:
        await env.wake(wake)
    await tick(manager, env.clock, advance=1)
    for model in models:
        assert_follows(metrics, await env.pool(model))


def ca_seconds_left(env: FakeServing) -> float:
    return env.pki.ca.cert.not_valid_after_utc.timestamp() - env.clock()


# ------------------------------------------------------------------ process start


def test_at_start_the_gauges_exist_and_alarm_counters_are_zero(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """`loom-poolmgr run`, before anything happened (scraped after the first tick, nothing
    woken): `ca_expiry_seconds` is the CA's time left, every `creates_total{model,result}`
    is 0 and the cold model's `pool_reason` is 0 for every reason. The model is the one
    the manager was built with (`PoolmgrRun.model`)."""
    h = poolmgr_run
    ca = write_ca(h.tmp / "ca40", days=40)
    h.set_ca(ca)
    h.live(pg_url)
    h.serve_metrics()
    seen: dict[str, object] = {}

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        # After the first tick: where the per-model series are created is the
        # implementation's choice (before or in the first tick).
        await run.wait_tick(0)
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
    for advance in (3600, 86400, 10 * 86400):
        await tick(manager, env.clock, advance=advance)
        got = value(metrics, "loom_poolmgr_ca_expiry_seconds")
        assert got is not None
        assert abs(got - ca_seconds_left(env)) <= 2, (advance, got, ca_seconds_left(env))
    # Below alarm P3's 30 days now: 40 days - (1 h + 1 day + 10 days) = 28 days 23 h, and
    # still far above the CA refusal at `server_cert_renew_before_s` (2 days).
    assert 2 * 86400 < got < 30 * 86400, got / 86400


# ------------------------------------------------------------------ pool reason


def no_stock(env: FakeServing) -> None:
    env.runpod.stock = [["EUR-IS-2"]]  # no US datacenter has the GPU


async def test_pool_reason_follows_the_pool_hash_each_tick(
    env: FakeServing, model_keys: list[str]
) -> None:
    """cold (no reason) -> `no_capacity` -> warming -> ready: in each state the gauges of
    the model are 1 for the hash's reason and 0 for the rest once a tick has run on it; a
    servable model missing from the serving file is `not_configured`."""
    model = unique_model_id()
    missing = unique_model_id("llama")
    model_keys += [model, missing]
    config = serving_config([model], capacity={"backoff_s": [10], "unavailable_retry_after_s": 300})
    manager, metrics = metered_manager(env, config, servable=(model, missing))
    await manager.start()
    await tick(manager, env.clock, advance=1)
    assert (await env.pool(model)).state == "cold"
    assert (await env.pool(missing)).reason == "not_configured"
    await follows_after_next_tick(env, manager, metrics, model, missing)

    no_stock(env)
    for _ in range(3):  # the first attempt finds no capacity, the others are in its backoff
        await env.wake(model)
        await tick(manager, env.clock, advance=1)
    assert (await env.pool(model)).reason == "no_capacity"
    await follows_after_next_tick(env, manager, metrics, model, wake=model)
    assert (await env.pool(model)).reason == "no_capacity"

    env.runpod.stock = [["US-TX-3"]]  # a US datacenter has the GPU again
    await env.wake(model)
    await tick(manager, env.clock, advance=15)  # past the 10 s backoff
    seen = {(await env.pool(model)).state}
    assert (await env.pool(model)).reason != "no_capacity"
    await follows_after_next_tick(env, manager, metrics, model)
    await bring_up(env, manager, model)
    pool = await env.pool(model)
    seen.add(pool.state)
    assert pool.state == "ready", seen
    await follows_after_next_tick(env, manager, metrics, model, missing)
    assert (await env.pool(model)).state == "ready"


async def test_spend_cap_reason_gauge_is_set_and_cleared(
    env: FakeServing, model_keys: list[str]
) -> None:
    """P2b's series: `pool_reason{reason="spend_cap"}` is 1 while the model is held at
    the cap and back to 0 once a reset lets it create again (each checked after one more
    tick: the gauge is rewritten each tick). The reset time the manager keeps in Redis
    (`loom:poolmgr:reset:<model>`) is deleted with the model's keys (`model_keys`)."""
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
    assert (await env.pool(model)).reason == "spend_cap"
    await follows_after_next_tick(env, manager, metrics, model)
    assert (await env.pool(model)).reason == "spend_cap"
    assert value(metrics, "loom_poolmgr_pool_reason", model=model, reason="spend_cap") == 1.0

    # After the reset the clock moves only until the create (as M2's spend tests do), and
    # the extra tick is a `step` that leaves the clock where it is: the pod's bootstrap
    # never reaches its health check, whose TLS check of the pod's 7-day server
    # certificate runs on real time, not on this test's fixed date.
    await delete_pod_rows(env.database_url, other)
    await env.admin({"op": "reset", "model": model})
    await env.wake(model)
    await manager.reconcile_once()
    await drive(env.clock, lambda: len(env.runpod.creates()) == 1, max_advance_s=1)

    async def cleared() -> bool:
        return (await env.pool(model)).reason != "spend_cap"

    await wait_until(cleared)
    await step(manager, env.clock)
    pool = await env.pool(model)
    assert pool.reason != "spend_cap", pool
    assert_follows(metrics, pool)
    assert value(metrics, "loom_poolmgr_pool_reason", model=model, reason="spend_cap") == 0.0
