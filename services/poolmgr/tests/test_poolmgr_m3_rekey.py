"""A27: `loom-poolmgr rekey [--model M]` re-issues every ready pod's server certificate now
(docs/phase1-m3-spec.md §7 "Admin re-key"; contracts §9: the SIGHUP path the manager
already uses near expiry).

The command runs as from a shell (`cli.main` in a thread, its own event loop) against
the test's database and Redis, while a manager on the fakes of `poolmgr_fakes` runs with the default
admin queue: whatever way the command reaches the running manager, it does through the
same Redis. Tests that use the default queue share the xdist group
`poolmgr-default-admin-queue` (as the existing `local-capacity` tests do).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import yaml
from cryptography import x509
from poolmgr_fakes import (
    FakePod,
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    peer_certificate,
    proxy_get,
    serving_config,
    tick,
    unique_model_id,
)
from poolmgr_m3_kit import model_keys_fixture  # noqa: F401

from loom_platform.serving import ServingConfig
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV, pg_url, redis_client  # noqa: F401
from loom_poolmgr import cli
from loom_poolmgr.manager import ADMIN_QUEUE, PoolManager

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.redis,
    pytest.mark.xdist_group("poolmgr-default-admin-queue"),
]


def cli_exit(*argv: str) -> int | str | None:
    try:
        cli.main(list(argv))
    except SystemExit as e:
        return e.code
    return 0


def cli_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: ServingConfig, database_url: str
) -> None:
    """The command's environment: dev, this test's serving file, database and Redis."""
    for var in list(os.environ):
        if var.startswith("LOOM_POOLMGR_") or var == "LOOM_SERVING_YAML":
            monkeypatch.delenv(var, raising=False)
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
    monkeypatch.setenv("LOOM_POOLMGR_SERVING_YAML", str(path))
    monkeypatch.setenv("LOOM_ENV", "dev")
    monkeypatch.setenv("LOOM_DATABASE_URL", database_url)
    monkeypatch.setenv("LOOM_REDIS_URL", os.environ.get(REDIS_ENV) or DEFAULT_REDIS)


async def serial(env: FakeServing, model: str) -> int:
    (replica,) = (await env.pool(model)).replicas
    der = await peer_certificate(replica, env.pki)
    return x509.load_der_x509_certificate(der).serial_number


async def two_ready_models(
    env: FakeServing, model_keys: list[str]
) -> tuple[PoolManager, ServingConfig, dict[str, FakePod]]:
    a, b = unique_model_id(), unique_model_id()
    model_keys += [a, b]
    env.admin_queue = ADMIN_QUEUE
    await env.redis.delete(ADMIN_QUEUE)
    # Long certificates: nothing is due for renewal on its own during the test.
    config = serving_config(
        [a, b], lifecycle={"server_cert_days": 7, "server_cert_renew_before_s": 86400}
    )
    manager = env.manager(config)
    await manager.start()
    pods = {a: await bring_up(env, manager, a), b: await bring_up(env, manager, b)}
    for pod in pods.values():
        assert pod.bootstraps == ["started"], pod.bootstraps
    return manager, config, pods


async def rekey_ticks(env: FakeServing, manager: PoolManager, done: dict[str, FakePod]) -> None:
    """A few ticks (the command is picked up on the next one) until the pods in `done`
    have reloaded."""
    for _ in range(3):
        await tick(manager, env.clock, advance=1)
        if all(p.bootstraps[-1] == "reloaded" for p in done.values()):
            return


async def test_rekey_reissues_every_ready_pods_certificate(
    env: FakeServing,
    model_keys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    try:
        manager, config, pods = await two_ready_models(env, model_keys)
        before = {m: await serial(env, m) for m in pods}
        cli_env(monkeypatch, tmp_path, config, env.database_url)
        code = await asyncio.to_thread(cli_exit, "rekey")
        assert code in (0, None), f"loom-poolmgr rekey exited {code}"
        await rekey_ticks(env, manager, pods)
        for model, pod in pods.items():
            assert pod.bootstraps == ["started", "reloaded"], (model, pod.bootstraps)
            assert await serial(env, model) != before[model], f"{model}: same certificate"
            (replica,) = (await env.pool(model)).replicas
            assert (await proxy_get(env.client(), replica, "/health")).status_code == 200
            assert (await env.pool(model)).routable
    finally:
        await env.redis.delete(ADMIN_QUEUE)


async def test_rekey_model_reissues_only_that_models_pods(
    env: FakeServing,
    model_keys: list[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    try:
        manager, config, pods = await two_ready_models(env, model_keys)
        a, b = list(pods)
        before = {m: await serial(env, m) for m in pods}
        cli_env(monkeypatch, tmp_path, config, env.database_url)
        code = await asyncio.to_thread(cli_exit, "rekey", "--model", a)
        assert code in (0, None), f"loom-poolmgr rekey --model exited {code}"
        await rekey_ticks(env, manager, {a: pods[a]})
        await tick(manager, env.clock, advance=1)
        assert pods[a].bootstraps == ["started", "reloaded"], pods[a].bootstraps
        assert await serial(env, a) != before[a]
        assert pods[b].bootstraps == ["started"], f"{b} was re-keyed too"
        assert await serial(env, b) == before[b]
    finally:
        await env.redis.delete(ADMIN_QUEUE)


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
