"""`provider: local` costs nothing (docs/phase1-m2-spec.md §11 "Fixed after the final
whole-branch review": local deployments have no listed price, so no price check and no
spend accrual; the M2 acceptance's spend-cap check uses its own synthetic row).

- The deployments `loom-poolmgr run` builds for a local serving file have no listed price
  (the same file with `provider: runpod` has one: the control).
- A real local pod (engine plus proxy subprocesses, created and bootstrapped by the
  manager from those deployments) runs for a while and month-to-date spend stays 0.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
import yaml
from poolmgr_fakes import (
    FakeServing,
    env_fixture,  # noqa: F401
    pod_rows,
    unique_model_id,
)
from test_poolmgr_local_backend import local_backend, local_config, reconcile_until

from loom_bench.prices import load_prices
from loom_bench.registry import REPO_ROOT, Registry, load_registry
from loom_platform.db import libpq_url
from loom_platform.serving import load_serving
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr import cli, spend
from loom_poolmgr.manager import ManagerLocked, PoolManager, build_deployments

COMMITTED_SERVING = REPO_ROOT / "config" / "serving.yaml"


# ------------------------------------------------------------------ what `run` builds


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[list[dict[str, Any]]]:
    """`loom-poolmgr run` (dev, the committed local file) up to `PoolManager.start()`."""
    for var in list(os.environ):
        if var.startswith("LOOM_POOLMGR_") or var in ("LOOM_SERVING_YAML", "LOOM_ENV"):
            monkeypatch.delenv(var, raising=False)
    built: list[dict[str, Any]] = []

    class Recording(PoolManager):
        def __init__(self, config: Any, **kw: Any) -> None:
            built.append({"config": config, **kw})
            super().__init__(config, **kw)

        async def start(self) -> None:
            raise ManagerLocked("recorded (test)")

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(cli, "PoolManager", Recording)
    monkeypatch.setenv("LOOM_ENV", "dev")
    monkeypatch.setenv("LOOM_POOLMGR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOOM_POOLMGR_METRICS_PORT", "0")
    monkeypatch.setenv("LOOM_POOLMGR_SERVING_YAML", str(COMMITTED_SERVING))
    monkeypatch.setenv("LOOM_REDIS_URL", "redis://127.0.0.1:9/0")  # never connected
    yield built


def test_run_builds_local_deployments_without_a_listed_price(
    recorded: list[dict[str, Any]],
) -> None:
    with pytest.raises(SystemExit) as exit_:
        cli.main(["run"])
    assert exit_.value.code == 1  # ManagerLocked: the recording manager's start()
    [built] = recorded
    assert built["config"].provider == "local"
    deployments = built["deployments"]
    assert deployments
    for model_id, dep in deployments.items():
        assert dep.listed_price_per_hr_micros is None, (model_id, dep.listed_price_per_hr_micros)


def test_the_same_models_on_runpod_do_have_a_listed_price(tmp_path: Path) -> None:
    """The control: the price exists, local just does not use it."""
    doc = yaml.safe_load(COMMITTED_SERVING.read_text())
    doc["provider"] = "runpod"
    doc["network"]["gateway_egress_cidrs"] = ["203.0.113.10/32"]
    path = tmp_path / "serving-runpod.yaml"
    path.write_text(yaml.safe_dump(doc))
    registry = load_registry()
    config = load_serving(path, registry=registry)
    deployments = build_deployments(config, registry, load_prices())
    assert all(d.listed_price_per_hr_micros for d in deployments.values()), deployments


# ------------------------------------------------------------------ a running local pod


async def month_to_date(database_url: str) -> int:
    async with await psycopg.AsyncConnection.connect(libpq_url(database_url)) as conn:
        return await spend.month_to_date_micros(conn, dt.datetime.now(dt.UTC))


@pytest.mark.postgres
@pytest.mark.redis
async def test_a_running_local_pod_accrues_no_spend(env: FakeServing, tmp_path: Path) -> None:
    model = unique_model_id()
    config = local_config(model)
    row = load_registry().get("qwen3-8b").model_copy(update={"id": model})
    deployments = build_deployments(config, Registry(models=[row]), load_prices())
    assert await month_to_date(env.database_url) == 0
    async with local_backend(tmp_path) as local:
        manager = PoolManager(
            config,
            deployments=deployments,
            provider=local,
            pod_exec=local.exec,
            redis=env.redis,
            database_url=env.database_url,
            leases=env.leases,
            pki=env.pki,
            ssh=env.ssh,
            bundle=env.bundle,
            admin_queue=env.admin_queue,
        )  # real time: its pods are real processes
        env.managers.append(manager)
        await manager.start()
        pool = await reconcile_until(env, manager, model, lambda p: p.routable)
        assert len(pool.replicas) == 1  # no price check stood in the way
        await asyncio.sleep(3.0)  # the pod runs (a priced pod would accrue ~1 micro-$/3 ms)
        [pod_row] = await pod_rows(env.database_url, model)
        assert pod_row["state"] == "ready"
        assert not pod_row["cost_per_hr_micros"], pod_row["cost_per_hr_micros"]
        assert await month_to_date(env.database_url) == 0
