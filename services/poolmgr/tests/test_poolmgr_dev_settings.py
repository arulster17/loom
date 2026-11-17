"""The compose-only knobs of the pool manager (docs/phase1-m2-spec.md §9 "Local backend",
§11 settings): `LOOM_POOLMGR_ALLOW_CIDRS`, `LOOM_POOLMGR_IDLE_STOP_S` and the
`loom-poolmgr local-capacity off|on` switch.

- `LOOM_POOLMGR_ALLOW_CIDRS`: comma-separated CIDRs the manager adds to the local pods'
  proxy allowlist. A bad CIDR is rejected when the settings load. Dev only: outside
  `LOOM_ENV=dev` it is a startup problem (`production_problems()`, which
  `enforce_environment` turns into a refusal).
- `LOOM_POOLMGR_IDLE_STOP_S`: replaces every model's `idle_stop_s`. Dev only, likewise.
- `loom-poolmgr local-capacity off|on`: dev only, queued on the admin queue, refused
  (non-zero exit, nothing queued) unless the serving file says `provider: local`.

How `loom-poolmgr run` wires the first two into the manager is checked by running the
real `main(["run"])` with `PoolManager` replaced by a subclass that records what it was
given and then stops (`ManagerLocked` from `start()`), so no database is touched.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import ipaddress
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from redis.asyncio import Redis

from loom_bench.registry import REPO_ROOT
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV, redis_client  # noqa: F401
from loom_poolmgr import cli
from loom_poolmgr.manager import ADMIN_QUEUE, ManagerLocked, PoolManager
from loom_poolmgr.settings import PoolmgrSettings

COMMITTED_SERVING = REPO_ROOT / "config" / "serving.yaml"
ALLOW = "LOOM_POOLMGR_ALLOW_CIDRS"
IDLE = "LOOM_POOLMGR_IDLE_STOP_S"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """No LOOM_POOLMGR_* / serving overrides from the caller's environment; dev."""
    for var in list(os.environ):
        if var.startswith("LOOM_POOLMGR_") or var in ("LOOM_SERVING_YAML", "LOOM_ENV"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOOM_ENV", "dev")
    yield monkeypatch


def nets(values: Any) -> set[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    return {ipaddress.ip_network(str(v), strict=False) for v in values}


def exit_code(argv: list[str]) -> int | str | None:
    """`loom-poolmgr <argv>`'s exit code (it always ends in SystemExit)."""
    with pytest.raises(SystemExit) as exit_:
        cli.main(argv)
    return exit_.value.code


def help_text(argv: list[str]) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = exit_code([*argv, "--help"])
    assert code in (0, None), f"`loom-poolmgr {' '.join(argv)} --help` exited {code}"
    return out.getvalue()


# ------------------------------------------------------------------ settings


def test_allow_cidrs_takes_a_comma_list_in_dev(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv(ALLOW, "10.1.0.0/24, 10.2.0.5,fd00::/64")
    settings = PoolmgrSettings()
    assert not [p for p in settings.production_problems() if ALLOW in p]


@pytest.mark.parametrize(
    "value", ["10.1.0.0/33", "not-a-cidr", "10.1.0.0/24,bogus", "10.1.0.0/24;10.2.0.0/24"]
)
def test_a_bad_allow_cidr_is_rejected(clean_env: pytest.MonkeyPatch, value: str) -> None:
    clean_env.setenv(ALLOW, value)
    with pytest.raises(ValueError):
        PoolmgrSettings()


@pytest.mark.parametrize(("var", "value"), [(ALLOW, "10.1.0.0/24"), (IDLE, "600")])
def test_the_dev_knobs_are_refused_outside_dev(
    clean_env: pytest.MonkeyPatch, var: str, value: str
) -> None:
    clean_env.setenv("LOOM_ENV", "prod")
    clean_env.setenv(var, value)
    try:
        settings = PoolmgrSettings()
    except ValueError as e:  # refused when the settings load: as good
        assert var in str(e)
        return
    problems = settings.production_problems()
    assert [p for p in problems if var in p], problems


def test_outside_dev_without_the_knobs_there_is_no_such_problem(
    clean_env: pytest.MonkeyPatch,
) -> None:
    clean_env.setenv("LOOM_ENV", "prod")
    problems = PoolmgrSettings().production_problems()
    assert not [p for p in problems if ALLOW in p or IDLE in p], problems


# ------------------------------------------------------------------ `run` wiring


@pytest.fixture
def recorded(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[list[tuple[Any, dict[str, Any]]]]:
    """`loom-poolmgr run` up to `PoolManager.start()`, recording the manager's arguments."""
    calls: list[tuple[Any, dict[str, Any]]] = []

    class Recording(PoolManager):
        def __init__(self, config: Any, **kw: Any) -> None:
            calls.append((config, kw))
            super().__init__(config, **kw)

        async def start(self) -> None:
            raise ManagerLocked("recorded (test)")

        async def stop(self) -> None:
            return None

    clean_env.setattr(cli, "PoolManager", Recording)
    clean_env.setenv("LOOM_POOLMGR_STATE_DIR", str(tmp_path / "state"))
    clean_env.setenv("LOOM_POOLMGR_METRICS_PORT", "0")
    clean_env.setenv("LOOM_POOLMGR_LOCAL_COLD_START_S", "0")
    clean_env.setenv("LOOM_POOLMGR_SERVING_YAML", str(COMMITTED_SERVING))  # provider local
    clean_env.setenv("LOOM_REDIS_URL", "redis://127.0.0.1:9/0")  # never connected
    yield calls


def run_main() -> None:
    assert exit_code(["run"]) == 1  # ManagerLocked: stopped before the first tick


def test_idle_stop_override_replaces_every_models_idle_stop(
    recorded: list[tuple[Any, dict[str, Any]]], clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv(IDLE, "42")
    run_main()
    [(config, _)] = recorded
    assert config.models, "no models configured"
    assert {m: s.idle_stop_s for m, s in config.models.items()} == {m: 42 for m in config.models}


def test_without_the_override_the_serving_files_idle_stop_applies(
    recorded: list[tuple[Any, dict[str, Any]]],
) -> None:
    committed = yaml.safe_load(COMMITTED_SERVING.read_text())["models"]
    run_main()
    [(config, kw)] = recorded
    assert {m: s.idle_stop_s for m, s in config.models.items()} == {
        m: v["idle_stop_s"] for m, v in committed.items()
    }
    assert not tuple(kw.get("extra_allow") or ())


def test_allow_cidrs_reach_the_manager_parsed(
    recorded: list[tuple[Any, dict[str, Any]]], clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv(ALLOW, " 10.9.0.0/24 ,10.9.1.7")
    run_main()
    [(_, kw)] = recorded
    assert nets(kw.get("extra_allow") or ()) == nets(["10.9.0.0/24", "10.9.1.7/32"])


# ------------------------------------------------------------------ local-capacity


def test_local_capacity_has_help() -> None:
    text = help_text(["local-capacity"])
    assert "off" in text and "on" in text


def runpod_serving(tmp_path: Path) -> Path:
    doc = yaml.safe_load(COMMITTED_SERVING.read_text())
    doc["provider"] = "runpod"
    doc["network"]["gateway_egress_cidrs"] = ["203.0.113.10/32"]
    path = tmp_path / "serving-runpod.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


@pytest.mark.redis
@pytest.mark.xdist_group("poolmgr-default-admin-queue")
@pytest.mark.parametrize("case", ["provider_runpod", "outside_dev"])
async def test_local_capacity_is_refused_unless_local_and_dev(
    clean_env: pytest.MonkeyPatch,
    tmp_path: Path,
    redis_client: Redis,  # noqa: F811
    case: str,
) -> None:
    help_text(["local-capacity"])  # the command exists: a refusal is not argparse's
    clean_env.setenv("LOOM_REDIS_URL", os.environ.get(REDIS_ENV) or DEFAULT_REDIS)
    if case == "provider_runpod":
        clean_env.setenv("LOOM_POOLMGR_SERVING_YAML", str(runpod_serving(tmp_path)))
    else:
        clean_env.setenv("LOOM_ENV", "prod")
        clean_env.setenv("LOOM_POOLMGR_SERVING_YAML", str(COMMITTED_SERVING))
    before = await redis_client.lrange(ADMIN_QUEUE, 0, -1)
    # The CLI runs its own event loop (asyncio.run): in a thread, as from a shell.
    code = await asyncio.to_thread(exit_code, ["local-capacity", "off"])
    assert code not in (0, None), "local-capacity was not refused"
    after = await redis_client.lrange(ADMIN_QUEUE, 0, -1)
    assert not [c for c in after if b"capacity" in c and c not in before], after
