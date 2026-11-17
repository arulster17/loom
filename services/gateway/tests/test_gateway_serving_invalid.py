"""A present but invalid `config/serving.yaml` stops the gateway (docs/phase1-m2-spec.md
§11 "Fixed after the final whole-branch review": fail fast on a config error); only a
missing file falls back to `DEFAULT_COLD_START_S` (§4 "Gateway without the file").

"Refuses to start" means building the app (`create_app`) raises, or its startup (the
app's lifespan, which `uvicorn` runs before serving) raises. The missing-file and
committed-file cases start with the same settings, so a startup failure in the invalid
cases is the config's.
"""

from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import yaml

from loom_bench.registry import REPO_ROOT, load_registry
from loom_gateway.app import create_app
from loom_gateway.settings import DEFAULT_COLD_START_S, GatewaySettings
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV

SERVING_ENV = "LOOM_SERVING_YAML"
COMMITTED = yaml.safe_load((REPO_ROOT / "config" / "serving.yaml").read_text())

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


def settings(pg_url: str) -> GatewaySettings:
    return GatewaySettings(
        database_url=pg_url,
        redis_url=os.environ.get(REDIS_ENV) or DEFAULT_REDIS,
        usage_stream=f"loom:test:usage:{uuid.uuid4().hex}",
        log_level="WARNING",
        env="dev",
    )


def invalid_docs() -> dict[str, str]:
    unknown_key = copy.deepcopy(COMMITTED)
    unknown_key["surprise"] = 1
    bad_replicas = copy.deepcopy(COMMITTED)
    first = next(iter(bad_replicas["models"]))
    bad_replicas["models"][first]["min_replicas"] = 3
    bad_replicas["models"][first]["max_replicas"] = 1
    bad_type = copy.deepcopy(COMMITTED)
    bad_type["models"][first]["cold_start_estimate_s"] = "soon"
    return {
        "unknown_key": yaml.safe_dump(unknown_key),
        "min_above_max": yaml.safe_dump(bad_replicas),
        "bad_type": yaml.safe_dump(bad_type),
        "not_yaml": "version: 1\nmodels: [unclosed\n",
        "empty": "",
    }


async def start_error(s: GatewaySettings) -> Exception | None:
    """The error that stops the gateway from starting, or None when it starts (and is
    stopped again)."""
    try:
        app = create_app(s, registry=load_registry())
    except Exception as e:
        return e
    try:
        async with app.router.lifespan_context(app):
            return None
    except Exception as e:
        return e


@pytest.mark.parametrize("case", sorted(invalid_docs()))
async def test_an_invalid_serving_yaml_stops_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pg_url: str, case: str
) -> None:
    path = tmp_path / "serving.yaml"
    path.write_text(invalid_docs()[case], encoding="utf-8")
    monkeypatch.setenv(SERVING_ENV, str(path))
    error = await start_error(settings(pg_url))
    assert error is not None, f"the gateway started on an invalid serving.yaml ({case})"
    assert not isinstance(error, FileNotFoundError)


async def test_a_missing_serving_yaml_still_starts_with_the_default_estimates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pg_url: str
) -> None:
    monkeypatch.setenv(SERVING_ENV, str(tmp_path / "no-such-serving.yaml"))
    app = create_app(settings(pg_url), registry=load_registry())
    async with app.router.lifespan_context(app):
        router: Any = app.state.gateway.router
        for model in COMMITTED["models"]:
            assert router.cold_start_estimate(model) == DEFAULT_COLD_START_S


async def test_the_committed_serving_yaml_starts(
    monkeypatch: pytest.MonkeyPatch, pg_url: str
) -> None:
    monkeypatch.setenv(SERVING_ENV, str(REPO_ROOT / "config" / "serving.yaml"))
    app = create_app(settings(pg_url), registry=load_registry())
    async with app.router.lifespan_context(app):
        router: Any = app.state.gateway.router
        for model, entry in COMMITTED["models"].items():
            assert router.cold_start_estimate(model) == entry["cold_start_estimate_s"]
