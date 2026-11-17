"""The `local` backend and the manager's compose-only knobs (docs/phase1-m2-spec.md §9
"Local backend", §4 `idle_stop_s: 0`, §11).

- `LocalServing` / `LocalPodExec`: a pod is the bench mock engine on a loopback port plus
  a real `loom-podproxy` in front of it, both subprocesses, set up by the manager's own
  create/bootstrap code (the SSH steps replaced by direct file writes). The pod answers
  over mTLS with the manager's client certificate and the token it delivered, and
  refuses a wrong token and a client without a certificate. `terminate` stops both
  processes. `no_capacity` makes a create raise `NoCapacity`.
- `loom-poolmgr local-capacity off|on` (admin queue) switches that from outside: off,
  a woken model goes `unavailable` / `no_capacity`; on, it comes up.
- `extra_allow` (`LOOM_POOLMGR_ALLOW_CIDRS`): added to every pod's proxy allowlist
  alongside the loopback default.
- `idle_stop_s: 0`: a ready pod is never idle-drained (spec §4 "Also": "`idle_stop_s: 0`
  means never idle-stopped (as poolsim's 0)").

The local-backend tests run on real time (real subprocesses; the manager's bootstrap
polls every 5 s), so each takes some seconds. They need Linux (`/proc`).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
import ssl
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from poolmgr_fakes import (
    FakeServing,
    bring_up,
    deployment,
    env_fixture,  # noqa: F401
    proxy_complete,
    proxy_get,
    serving_config,
    tick,
    unique_model_id,
)

from loom_platform.pool import PoolState, wake_key
from loom_platform.serving import ServingConfig
from loom_platform.testing import DEFAULT_REDIS, REDIS_ENV, pg_url, redis_client  # noqa: F401
from loom_poolmgr import cli
from loom_poolmgr.bootstrap import LOOPBACK_ALLOW
from loom_poolmgr.local import LocalServing
from loom_poolmgr.manager import ADMIN_QUEUE, PoolManager
from loom_poolmgr.provider import NoCapacity, pod_name

pytestmark = [pytest.mark.postgres, pytest.mark.redis]

REAL_TIMEOUT_S = 90.0


# ------------------------------------------------------------------ helpers


def local_config(model: str, **sections: Any) -> ServingConfig:
    return serving_config([model], provider="local", **sections)


def local_manager(
    env: FakeServing,
    config: ServingConfig,
    local: LocalServing,
    *,
    admin_queue: str | None = None,
) -> PoolManager:
    """A manager on the local backend, on real time (its pods are real processes)."""
    manager = PoolManager(
        config,
        deployments={m: deployment(config, m) for m in config.models},
        provider=local,
        pod_exec=local.exec,
        redis=env.redis,
        database_url=env.database_url,
        leases=env.leases,
        pki=env.pki,
        ssh=env.ssh,
        bundle=env.bundle,
        clock=time.time,
        sleep=asyncio.sleep,
        admin_queue=admin_queue or env.admin_queue,
    )
    env.managers.append(manager)
    return manager


async def reconcile_until(
    env: FakeServing,
    manager: PoolManager,
    model: str,
    done: Callable[[PoolState], bool],
    *,
    wake: bool = True,
    timeout_s: float = REAL_TIMEOUT_S,
) -> PoolState:
    """Tick (real time) until `done(pool)`, keeping the model woken."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        if wake:
            await env.wake(model)
        await manager.reconcile_once()
        await asyncio.wait_for(manager.wait_idle(), max(1.0, deadline - loop.time()))
        pool = await env.pool(model)
        if done(pool):
            return pool
        if loop.time() > deadline:
            raise AssertionError(f"{model}: not done within {timeout_s} s: {pool}")
        await asyncio.sleep(0.5)


def processes() -> list[tuple[int, list[str]]]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        with contextlib.suppress(OSError):
            args = (d / "cmdline").read_bytes().decode(errors="replace").split("\0")
            out.append((int(d.name), [a for a in args if a]))
    return out


def alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def arg_after(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def pod_processes(proxy_port: int) -> tuple[int, int]:
    """(proxy pid, engine pid) of the local pod whose proxy listens on `proxy_port`."""
    procs = processes()
    proxies = [
        (pid, args)
        for pid, args in procs
        if any("podproxy" in a for a in args) and arg_after(args, "--port") == str(proxy_port)
    ]
    assert len(proxies) == 1, proxies
    proxy_pid, proxy_args = proxies[0]
    upstream = arg_after(proxy_args, "--upstream")
    assert upstream, proxy_args
    engine_port = upstream.rstrip("/").rsplit(":", 1)[1]
    engines = [
        pid for pid, args in procs if pid != proxy_pid and arg_after(args, "--port") == engine_port
    ]
    assert len(engines) == 1, (engine_port, engines)
    return proxy_pid, engines[0]


def refused(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    except OSError:
        return True
    return False


async def eventually(cond: Callable[[], bool], timeout_s: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not cond():
        assert asyncio.get_running_loop().time() < deadline, "not within the timeout"
        await asyncio.sleep(0.1)


@contextlib.asynccontextmanager
async def local_backend(tmp_path: Path) -> AsyncIterator[LocalServing]:
    local = LocalServing(cold_start_s=0, root=tmp_path / "local-pods")
    try:
        yield local
    finally:
        await local.aclose()


# ------------------------------------------------------------------ LocalServing


async def test_a_local_pod_is_engine_plus_proxy_over_mtls_and_terminate_stops_both(
    env: FakeServing, tmp_path: Path
) -> None:
    model = unique_model_id()
    async with local_backend(tmp_path) as local:
        manager = local_manager(env, local_config(model), local)
        await manager.start()
        pool = await reconcile_until(env, manager, model, lambda p: p.routable)
        [replica] = pool.replicas
        port = int(replica.url.rsplit(":", 1)[1])
        assert replica.url.startswith("https://")

        client = env.client()
        assert (await proxy_get(client, replica, "/health")).status_code == 200
        r = await proxy_complete(client, replica, model, max_tokens=3)
        assert r.status_code == 200, r.text
        assert r.json()["choices"]
        bad = await proxy_get(client, replica, "/health", token="not-the-token")
        assert bad.status_code in (401, 403), bad.status_code
        no_cert = ssl.create_default_context(cafile=str(env.pki.ca_cert_path))
        async with httpx.AsyncClient(verify=no_cert, timeout=10, trust_env=False) as anon:
            with pytest.raises(httpx.HTTPError):
                r = await anon.get(
                    replica.url + "/health",
                    headers={"X-Loom-Pod-Token": replica.token},
                    extensions={"sni_hostname": replica.server_name},
                )
                r.raise_for_status()

        proxy_pid, engine_pid = pod_processes(port)
        assert alive(proxy_pid) and alive(engine_pid)
        await local.terminate(replica.id)
        await eventually(lambda: not alive(proxy_pid) and not alive(engine_pid))
        assert refused(port)
        assert replica.id not in {p.pod_id for p in await local.list()}
        await local.terminate(replica.id)  # idempotent


async def test_no_capacity_switch_makes_a_create_raise_and_back(tmp_path: Path) -> None:
    config = local_config(model := unique_model_id())
    dep = deployment(config, model)
    env_vars = {"LOOM_MODEL": model, "LOOM_MANAGED": "true", "LOOM_ROLE": "serving"}
    async with local_backend(tmp_path) as local:
        local.no_capacity = True
        with pytest.raises(NoCapacity):
            await local.create(dep, pod_name(model, time.time()), env_vars)
        assert await local.list() == []
        local.no_capacity = False
        info = await local.create(dep, pod_name(model, time.time()), env_vars)
        assert [p.pod_id for p in await local.list()] == [info.pod_id]


@pytest.mark.xdist_group("poolmgr-default-admin-queue")
async def test_local_capacity_off_and_on_from_the_cli(
    env: FakeServing, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`loom-poolmgr local-capacity off` (the running manager's admin queue): a woken
    model meets no capacity (pool `unavailable` / `no_capacity`); `on`: it comes up."""
    for var in list(os.environ):
        if var.startswith("LOOM_POOLMGR_") or var == "LOOM_SERVING_YAML":
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOOM_ENV", "dev")
    monkeypatch.setenv("LOOM_REDIS_URL", os.environ.get(REDIS_ENV) or DEFAULT_REDIS)
    monkeypatch.setenv("LOOM_DATABASE_URL", env.database_url)
    await env.redis.delete(ADMIN_QUEUE)

    def cli_exit(*argv: str) -> int | str | None:
        with pytest.raises(SystemExit) as exit_:
            cli.main(list(argv))
        return exit_.value.code

    model = unique_model_id()
    config = local_config(model, capacity={"backoff_s": [1], "unavailable_retry_after_s": 300})
    async with local_backend(tmp_path) as local:
        manager = local_manager(env, config, local, admin_queue=ADMIN_QUEUE)
        await manager.start()
        assert await asyncio.to_thread(cli_exit, "local-capacity", "off") in (0, None)
        pool = await reconcile_until(
            env, manager, model, lambda p: p.state == "unavailable", timeout_s=30
        )
        assert pool.reason == "no_capacity"
        assert await local.list() == []

        assert await asyncio.to_thread(cli_exit, "local-capacity", "on") in (0, None)
        pool = await reconcile_until(env, manager, model, lambda p: p.routable)
        assert len(pool.replicas) == 1
        await env.redis.delete(wake_key(model))


# ------------------------------------------------------------------ the allowlist


async def test_extra_allow_cidrs_join_the_loopback_default_in_every_pods_allowlist(
    env: FakeServing,
) -> None:
    model = unique_model_id()
    config = serving_config([model])  # gateway_egress_cidrs: [] (as the committed file)
    extra = ("10.20.0.0/24", "192.0.2.7/32")
    manager = PoolManager(
        config,
        deployments={model: deployment(config, model)},
        provider=env.provider(config),
        pod_exec=env.pods,
        redis=env.redis,
        database_url=env.database_url,
        leases=env.leases,
        pki=env.pki,
        ssh=env.ssh,
        bundle=env.bundle,
        clock=env.clock,
        sleep=env.clock.sleep,
        admin_queue=env.admin_queue,
        extra_allow=extra,
    )
    env.managers.append(manager)
    await manager.start()
    pod = await bring_up(env, manager, model)
    allow = pod.payloads[-1]["allow"].decode()
    lines = [x.strip() for x in allow.splitlines() if x.strip() and not x.startswith("#")]
    got = {ipaddress.ip_network(x, strict=False) for x in lines}
    want = {ipaddress.ip_network(x) for x in (*LOOPBACK_ALLOW, *extra)}
    assert got == want, lines
    # The proxy runs with it: the manager (from loopback) still gets through.
    assert (await env.pool(model)).routable


# ------------------------------------------------------------------ idle_stop_s: 0


async def test_idle_stop_zero_is_never_idle_drained(env: FakeServing) -> None:
    """Two models woken once, then left without demand: the one with `idle_stop_s: 60`
    is drained to cold; the one with `idle_stop_s: 0` stays ready on the same pod."""
    never, soon = unique_model_id(), unique_model_id()
    config = serving_config({never: {"idle_stop_s": 0}, soon: {"idle_stop_s": 60}})
    manager = env.manager(config)
    await manager.start()
    kept = await bring_up(env, manager, never)
    drained = await bring_up(env, manager, soon)
    for m in (never, soon):
        await env.redis.delete(wake_key(m))

    for _ in range(15):  # 1500 fake seconds without demand
        await tick(manager, env.clock, advance=100)

    pool = await env.pool(never)
    assert pool.state == "ready" and [r.id for r in pool.replicas] == [kept.pod_id]
    assert env.runpod.has(kept.pod_id) and env.runpod.terminate_requests(kept.pod_id) == 0
    assert (await env.pool(soon)).state == "cold"
    assert not env.runpod.has(drained.pod_id)
