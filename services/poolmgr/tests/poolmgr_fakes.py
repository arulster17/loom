"""Test doubles for the pool manager (docs/phase1-m2-spec.md §10, §11, §13). Tests only.

They live with the tests (spec §10/§11: `services/poolmgr/tests/poolmgr_fakes.py`).

§13's "fake": `FakeRunpod` plus `FakeServingExec` plus a fake clock, with pods whose proxy
is a real `loom-podproxy` in front of the bench mock engine on localhost (real mTLS, real
`/metrics`).

Pieces
------
- `FakeClock`: a manual epoch clock for `PoolManager(clock=..., sleep=...)`. `sleep(s)`
  returns once the clock has been advanced to its deadline, so nothing the manager waits
  for happens until a test moves time. `drive()` / `settle()` advance it for the test
  ("auto-jump"): whenever something sleeps on it and nothing else finishes for a moment,
  the clock jumps to the earliest deadline. `step()` advances, ticks and gives tasks a
  moment of real time without moving the clock for them.

  TLS uses real time: the manager issues pod certificates with `now=clock()`. The clock
  starts `CLOCK_LAG_S` (an hour) behind real time, so a test may advance it by about an
  hour before a certificate issued at fake "now" is not valid yet in real time.

- `FakeServingRunpod`: `FakeRunpod` (the bench's in-memory RunPod) whose created pods
  are RUNNING with public IP 127.0.0.1 and the port mappings `22` -> a fake SSH port and
  `<proxy_port>` -> a real loopback port reserved for the pod's proxy. It registers each
  created pod with the `FakeServingExec`, and stops the pod's proxy and engine when the
  pod is terminated (REST DELETE or GraphQL podTerminate) or a test removes it. Knobs:
  `mapping_delay` (listings/GETs before the mappings appear), `ambiguous` (creates that
  make the pod but answer a 5xx or time out), `create_gate` (a POST /pods waits for this
  event: a create in flight), `terminate_fails` (REST DELETE and GraphQL podTerminate both
  answer 500), plus FakeRunpod's own (`fail`, `stock`, `cost_per_hr`).

- `FakeServingExec`: `SshExec` answering the pod exec protocol of `loom_poolmgr.bootstrap`
  for those pods:

  - the SSH probe (`SSH_PROBE`): 255 while `offline_probes` lasts or once the pod is gone;
  - `serve_bootstrap` (script, `loom-payload-v1`, base64 tar): checks the payload as the
    script does (exactly `PAYLOAD_FILES`, none empty, the bundle's sha256 against
    `BUNDLE_SHA256`), writes the files into the pod's directory, and starts a real
    `PodProxy` from them (`loom-proxy started <pid>`) on the pod's reserved port, or, when
    one runs, reloads it with the proxy's own SIGHUP handler (`loom_podproxy.proxy.reload`:
    all or nothing, open connections kept) (`loom-proxy reloaded <pid>`);
  - `serve_engine` through `run_detached` (launch, poll, stderr tail, collect, kill, as
    the bench's `FakePodExec`): starts the bench mock engine (in-process, uvicorn in its
    own thread) on the pod's loopback engine port, serving `SERVED_MODEL` and the pod's
    `LOOM_MODEL`. Unless the engine is held (`hang`), a poll waits for the engine job to
    finish before answering, so a bootstrap costs exactly one `poll_s` of fake time
    whatever the real start-up speed;
  - `serve_lease`: records `LEASE_EPOCH` and answers `loom-lease <epoch>`.

  Failure injection, per pod (`FakePod.behaviour`), per model for pods created from now
  on (`FakeServingExec.behaviour_for[model]`), or for every new pod
  (`FakeServingExec.behaviour`): `offline_probes`, `fail_bootstrap`, `engine` ("ok" |
  "fail": serve_engine exits 1 as its stall detection does | "hang": it never finishes
  until `release_engine`), `fail_lease`. On a running pod's `FakeEngine`: `set_healthy`,
  `stall`/`resume` (the engine stops stepping while HTTP and /metrics still answer: a
  progress stall), `kill` (connections cut mid-stream, then gone).

- `FakeServing`: all of the above wired together, a fresh CA (`Pki.generate`), an SSH
  identity, a proxy bundle, `MemoryLeaseRecords` and an admin queue key of its own;
  `manager(config)` builds a `PoolManager` on them.

**Why `RunpodServing` over `FakeRunpod` rather than `LocalServing`:** it is the production
provider, and FakeRunpod records every request, so a test sees the exact create bodies,
the terminate calls and what was never touched. `LocalServing` runs real subprocesses for
compose; its pods are not reachable through `SshExec` at all.

Isolation: tests run in parallel against one Redis, so each uses its own model ids
(`unique_model_id()`), which scopes the pool hash, the wake key and the usage events, and
its own admin queue (`PoolManager(admin_queue=...)`, `push_admin(..., queue=...)`).

Fixture: `env` (import `env_fixture`, with `pg_url` and `redis_client` from
`loom_platform.testing`, into the test module: the tests may not add a conftest.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import itertools
import json
import math
import re
import socket
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import psycopg
import pytest
import uvicorn
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from psycopg.rows import dict_row
from redis.asyncio import Redis

from loom_bench.experiment import RUNPOD_GPU_TYPE_IDS
from loom_bench.mock.config import MockConfig
from loom_bench.mock.server import MockServer, create_app
from loom_bench.providers.runpod_fake import REST, FakeRunpod
from loom_bench.providers.runpod_ssh import ExecResult, SshTarget
from loom_bench.registry import Registry, load_registry
from loom_platform.db import libpq_url
from loom_platform.pool import WAKE_TTL_S, PoolState, Replica, get_pool, wake_key
from loom_platform.serving import ServingConfig
from loom_podproxy.proxy import PodProxy, ProxyConfig, ProxyFiles, reload
from loom_poolmgr import scripts
from loom_poolmgr.bootstrap import (
    LEASE_FILE,
    LEASE_POLL_S,
    LOOM_ROOT,
    PAYLOAD_FILES,
    SSH_DIR,
    SSH_PROBE,
    SSHD_CONFIG,
    STAGE_FILE,
    UPSTREAM,
    ProxyBundle,
    split_payload,
)
from loom_poolmgr.leases import LeaseRecords, MemoryLeaseRecords
from loom_poolmgr.manager import PoolManager, SshIdentity, push_admin
from loom_poolmgr.pki import Pki
from loom_poolmgr.provider import Deployment
from loom_poolmgr.runpod import USER_AGENT, RunpodServing
from loom_poolmgr.spend import usd_micros

# The fake clock starts this far behind real time (see the module docstring).
CLOCK_LAG_S = 3600.0
GRAPHQL_URL = "https://api.runpod.io/graphql"
TOKEN_HEADER = "X-Loom-Pod-Token"
# FakeRunpod's datacenters listing stock: two US ones and one elsewhere.
US_STOCK = ("US-TX-3", "US-TX-4", "EUR-IS-2")
# Mock engine timing: ~20 ms per token at low load, so a 100-token stream takes ~2 s.
ENGINE_DEFAULTS: dict[str, Any] = {
    "step_base_ms": 20.0,
    "decode_ms_per_seq": 0.25,
    "prefill_ms_per_token": 0.05,
    "mean_output_tokens": 200,
}
# Longest a poll waits for a (not held) engine job before answering.
ENGINE_JOB_WAIT_S = 30.0


# ------------------------------------------------------------------------------ time


class FakeClock:
    """A manual epoch clock. `sleep` waits until the clock reaches its deadline."""

    def __init__(self, start: float | None = None) -> None:
        self.now = time.time() - CLOCK_LAG_S if start is None else float(start)
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("the clock never goes back")
        self.set(self.now + seconds)

    def set(self, now: float) -> None:
        if now < self.now:
            raise ValueError("the clock never goes back")
        self.now = now
        for deadline, fut in list(self._sleepers):
            if deadline <= now and not fut.done():
                fut.set_result(None)

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        entry = (self.now + seconds, fut)
        self._sleepers.append(entry)
        try:
            await fut
        finally:
            self._sleepers.remove(entry)

    def next_deadline(self) -> float | None:
        pending = [d for d, f in self._sleepers if not f.done()]
        return min(pending) if pending else None


async def drive(
    clock: FakeClock,
    done: Callable[[], bool],
    *,
    max_advance_s: float = 300.0,
    timeout_s: float = 30.0,
    quiet_s: float = 0.02,
) -> None:
    """Let background work run until `done()`, jumping the clock to the next sleeper's
    deadline whenever nothing finishes for `quiet_s` real seconds. Fails after
    `max_advance_s` fake or `timeout_s` real seconds."""
    loop = asyncio.get_running_loop()
    start, real_deadline = clock.now, loop.time() + timeout_s
    while not done():
        if loop.time() > real_deadline:
            raise AssertionError(
                f"not done within {timeout_s} s real ({clock.now - start:.1f} fake s advanced)"
            )
        await asyncio.sleep(quiet_s)
        if done():
            return
        nxt = clock.next_deadline()
        if nxt is not None:
            if nxt - start > max_advance_s:
                raise AssertionError(
                    f"not done within {max_advance_s} fake s (next sleeper at +{nxt - start:.1f})"
                )
            clock.set(max(nxt, clock.now))


async def settle(
    manager: PoolManager,
    clock: FakeClock,
    *,
    max_advance_s: float = 300.0,
    timeout_s: float = 30.0,
) -> None:
    """`manager.wait_idle()`, driving the clock while its tasks sleep on it."""
    task = asyncio.ensure_future(manager.wait_idle())
    try:
        await drive(clock, task.done, max_advance_s=max_advance_s, timeout_s=timeout_s)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
    task.result()


async def tick(
    manager: PoolManager, clock: FakeClock, *, advance: float = 0.0, max_advance_s: float = 300.0
) -> None:
    """Advance the clock, run one reconcile tick, and let the work it started finish."""
    clock.advance(advance)
    await manager.reconcile_once()
    await settle(manager, clock, max_advance_s=max_advance_s)


async def step(
    manager: PoolManager, clock: FakeClock, *, advance: float = 0.0, real_s: float = 0.05
) -> None:
    """Advance the clock, run one reconcile tick, then give the tasks it started `real_s`
    of real time without moving the clock for them (a drain or a held bootstrap stays
    where the test put it)."""
    clock.advance(advance)
    await manager.reconcile_once()
    await asyncio.sleep(real_s)


async def wait_until(
    cond: Callable[[], Any], *, timeout_s: float = 10.0, interval_s: float = 0.02
) -> None:
    """Poll `cond` (sync, or returning an awaitable) in real time until it is truthy."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        value = cond()
        if asyncio.iscoroutine(value) or isinstance(value, asyncio.Future):
            value = await value
        if value:
            return
        if loop.time() > deadline:
            raise AssertionError(f"condition not met within {timeout_s} s")
        await asyncio.sleep(interval_s)


# ---------------------------------------------------------------------- mock engine


def _reserved_socket() -> socket.socket:
    """A loopback socket bound to a free port with SO_REUSEADDR, never listening: it
    keeps the kernel from handing the port to anyone else, while a listener that also
    sets SO_REUSEADDR (the pod proxy does) can still bind it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    return sock


class FakeEngine:
    """The bench mock engine (`loom_bench.mock`) on a loopback socket, served by uvicorn
    in its own thread and event loop."""

    def __init__(self, sock: socket.socket, models: list[str], **overrides: Any) -> None:
        self.port = int(sock.getsockname()[1])
        self.config = MockConfig(models=models, **{**ENGINE_DEFAULTS, **overrides})
        self.app = create_app(self.config)
        self.server = uvicorn.Server(
            uvicorn.Config(
                self.app, log_level="critical", lifespan="on", timeout_graceful_shutdown=1
            )
        )
        self.loop: asyncio.AbstractEventLoop | None = None
        self._sock = sock
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"engine:{self.port}")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        try:
            loop.run_until_complete(self.server.serve(sockets=[self._sock]))
        finally:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    async def start(self, timeout_s: float = 15.0) -> None:
        self._thread.start()
        deadline = time.monotonic() + timeout_s
        while not self.server.started:
            if not self._thread.is_alive():
                raise RuntimeError("the mock engine exited at start")
            if time.monotonic() > deadline:
                raise RuntimeError("the mock engine did not start")
            await asyncio.sleep(0.005)

    @property
    def mock(self) -> MockServer:
        server: MockServer = self.app.state.mock
        return server

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def set_healthy(self, healthy: bool) -> None:
        """Unhealthy: the engine answers /health (and generation) with 503."""
        self.mock.ready_at = 0.0 if healthy else math.inf

    def stall(self) -> None:
        """Stop the engine's step loop: requests stay running, no token is produced,
        /health and /metrics still answer (vllm:generation_tokens_total goes flat)."""
        assert self.loop is not None
        asyncio.run_coroutine_threadsafe(self.mock.engine.stop(), self.loop).result(5)

    def resume(self) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.mock.engine.start)

    def kill(self) -> None:
        """As `kill -9`: every open connection is cut mid-response, then the engine is
        gone (connections refused). Safe before the engine's loop exists: it then exits
        as soon as it starts."""
        self.server.force_exit = True
        self.server.should_exit = True
        loop = self.loop
        if loop is None or loop.is_closed():
            return

        def _abort() -> None:
            for conn in list(self.server.server_state.connections):
                transport = getattr(conn, "transport", None)
                if transport is not None:
                    transport.abort()

        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(_abort)

    async def stop(self) -> None:
        self.kill()
        if self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, 10)
        with contextlib.suppress(OSError):
            self._sock.close()


def metric_value(text: str, name: str) -> float:
    """Sum of a Prometheus metric's samples (0 when absent)."""
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+(\S+)", line)
        if m:
            total += float(m.group(1))
    return total


# ---------------------------------------------------------------------- fake pods

EngineMode = Literal["ok", "fail", "hang"]


@dataclass
class PodBehaviour:
    """Failure injection for one pod."""

    offline_probes: int = 0  # SSH probes answered 255 before sshd is "up"
    fail_bootstrap: bool = False  # serve_bootstrap exits 1
    engine: EngineMode = "ok"  # serve_engine: ok | fail (exit 1, a stall) | hang
    fail_lease: bool = False  # serve_lease: ssh fails (255)


@dataclass
class ExecCall:
    """One `SshExec.run` call as the fake saw it."""

    target: SshTarget
    stdin: str  # everything sent on stdin
    kind: str  # probe | serve_bootstrap | serve_lease | serve_engine | ctl | unknown
    pod_id: str | None
    at: float  # the fake clock at the call
    script: str  # stdin without a serve_bootstrap payload (what bash runs)


@dataclass
class FakePod:
    pod_id: str
    name: str
    env: dict[str, str]
    ssh_port: int
    proxy_port: int
    engine_port: int
    dir: Path
    behaviour: PodBehaviour
    created_at: float
    payloads: list[dict[str, bytes]] = field(default_factory=list)
    bootstraps: list[str] = field(default_factory=list)  # started | reloaded | failed
    lease_attempts: list[float] = field(default_factory=list)  # fake clock per attempt
    leases: list[tuple[float, int]] = field(default_factory=list)  # (at, epoch) accepted
    engine_scripts: list[str] = field(default_factory=list)
    served_model: str | None = None
    proxy: PodProxy | None = None
    engine: FakeEngine | None = None
    pid: int = 0
    gone: bool = False
    engine_held: bool = False  # serve_engine is held (`hang`) until released
    released: asyncio.Event = field(default_factory=asyncio.Event)
    proxy_sock: socket.socket | None = None
    engine_sock: socket.socket | None = None

    @property
    def model(self) -> str:
        return self.env.get("LOOM_MODEL", "")

    @property
    def token(self) -> str | None:
        """The token of the last payload delivered (what the proxy runs with, unless a
        reload failed)."""
        return self.payloads[-1]["token"].decode().strip() if self.payloads else None

    @property
    def lease_epoch(self) -> int | None:
        return self.leases[-1][1] if self.leases else None

    @property
    def files(self) -> ProxyFiles:
        d = self.dir / "proxy"
        return ProxyFiles(
            cert=d / "server.crt",
            key=d / "server.key",
            client_ca=d / "ca.crt",
            token_file=d / "token",
            allow_file=d / "allow",
        )

    def replica(self, token: str | None = None) -> Replica:
        """How the gateway would reach this pod (with `token`, else the current one)."""
        return Replica(
            id=self.pod_id,
            url=f"https://127.0.0.1:{self.proxy_port}",
            server_name=self.pod_id,
            token=token or self.token or "",
        )


@dataclass
class _Detached:
    pod: FakePod
    task: asyncio.Task[tuple[int, str, str]]


class FakeServingExec:
    """`SshExec` for fake serving pods (see the module docstring)."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.time,
        engine_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.clock = clock
        self.engine_overrides = dict(engine_overrides or {})
        self.behaviour = PodBehaviour()  # copied onto each pod created from now on
        self.behaviour_for: dict[str, PodBehaviour] = {}  # per LOOM_MODEL, wins over that
        self.pods: dict[str, FakePod] = {}
        self.calls: list[ExecCall] = []
        self.protocol_errors: list[str] = []
        self._by_ssh_port: dict[int, FakePod] = {}
        self._ctl: dict[str, _Detached] = {}
        self._ssh_ports = itertools.count(22001)
        self._pids = itertools.count(4001)
        self._lock = threading.Lock()
        self._cleanup: set[asyncio.Task[None]] = set()
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # ---------------------------------------------------------------- pods

    def register(self, pod_id: str, name: str, env: Mapping[str, str]) -> FakePod:
        """A new pod (FakeServingRunpod calls this on create; any thread). Reserves its
        loopback proxy and engine ports."""
        proxy_sock, engine_sock = _reserved_socket(), _reserved_socket()
        with self._lock:
            ssh_port = next(self._ssh_ports)
            template = self.behaviour_for.get(env.get("LOOM_MODEL", ""), self.behaviour)
            pod = FakePod(
                pod_id=pod_id,
                name=name,
                env=dict(env),
                ssh_port=ssh_port,
                proxy_port=int(proxy_sock.getsockname()[1]),
                engine_port=int(engine_sock.getsockname()[1]),
                dir=self.root / pod_id,
                behaviour=dataclasses.replace(template),
                created_at=self.clock(),
                proxy_sock=proxy_sock,
                engine_sock=engine_sock,
            )
            self.pods[pod_id] = pod
            self._by_ssh_port[ssh_port] = pod
        return pod

    def snapshot(self) -> list[FakePod]:
        """Every pod so far (a copy, safe while RunPod's worker threads add pods)."""
        with self._lock:
            return list(self.pods.values())

    def pod(self, pod_id: str) -> FakePod:
        return self.pods[pod_id]

    def only(self) -> FakePod:
        """The one pod there is (fails otherwise)."""
        pods = self.snapshot()
        assert len(pods) == 1, f"expected one pod, have {[p.pod_id for p in pods]}"
        return pods[0]

    def of(self, model_id: str) -> list[FakePod]:
        return [p for p in self.snapshot() if p.model == model_id]

    def live(self) -> list[FakePod]:
        return [p for p in self.snapshot() if not p.gone]

    def release_engine(self, pod_id: str, mode: EngineMode = "ok") -> None:
        """Let a pod's held (`hang`) serve_engine finish as `mode`."""
        pod = self.pods[pod_id]
        pod.behaviour.engine = mode
        pod.engine_held = mode == "hang"
        pod.released.set()

    def kill_engine(self, pod_id: str) -> None:
        engine = self.pods[pod_id].engine
        assert engine is not None, "the pod has no engine"
        engine.kill()

    def pod_gone(self, pod_id: str) -> None:
        """The pod no longer exists (terminated, removed, exited): SSH stops answering and
        its proxy and engine stop. Safe from any thread; unknown ids are ignored."""
        pod = self.pods.get(pod_id)
        if pod is None or pod.gone:
            return
        pod.gone = True
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._schedule_shutdown(pod)
        else:
            loop.call_soon_threadsafe(self._schedule_shutdown, pod)

    def _schedule_shutdown(self, pod: FakePod) -> None:
        task = asyncio.ensure_future(self._shutdown(pod))
        self._cleanup.add(task)
        task.add_done_callback(self._cleanup.discard)

    async def _shutdown(self, pod: FakePod) -> None:
        for ctl, job in list(self._ctl.items()):
            if job.pod is pod:
                job.task.cancel()
                self._ctl.pop(ctl, None)
        if pod.proxy is not None:
            await pod.proxy.close()
        if pod.engine is not None:
            await pod.engine.stop()
        for sock in (pod.proxy_sock, pod.engine_sock):
            if sock is not None:
                with contextlib.suppress(OSError):
                    sock.close()

    async def aclose(self) -> None:
        for pod in self.snapshot():
            pod.gone = True
            await self._shutdown(pod)
        if self._cleanup:
            await asyncio.gather(*self._cleanup, return_exceptions=True)

    # ---------------------------------------------------------------- SshExec

    def scripts(self, kind: str, pod_id: str | None = None) -> list[str]:
        """The scripts (payload stripped) of every call of `kind`."""
        return [
            c.script
            for c in list(self.calls)
            if c.kind == kind and (pod_id is None or c.pod_id == pod_id)
        ]

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        pod = self._by_ssh_port.get(target.port)
        kind, head, files = self._classify(script)
        call = ExecCall(
            target=target,
            stdin=script,
            kind=kind,
            pod_id=pod.pod_id if pod else None,
            at=self.clock(),
            script=head,
        )
        self.calls.append(call)
        if target.host != "127.0.0.1":
            self.protocol_errors.append(f"SSH to {target.host}: fake pods are on 127.0.0.1")
        if pod is None or pod.gone:
            return ExecResult(255, "", f"ssh: connect to host port {target.port}: refused\n")
        if kind == "probe":
            if pod.behaviour.offline_probes > 0:
                pod.behaviour.offline_probes -= 1
                return ExecResult(255, "", "ssh: connect to host: Connection timed out\n")
            return ExecResult(0, "loom-online\n", "")
        if kind == "ctl":
            return await self._control(pod, script, call)
        if kind == "serve_bootstrap":
            return await self._bootstrap(pod, head, files)
        if kind == "serve_lease":
            return self._lease(pod, head)
        self.protocol_errors.append(f"unexpected SSH call ({kind}): {head[:120]!r}")
        return ExecResult(2, "", "unexpected script\n")

    @staticmethod
    def _classify(script: str) -> tuple[str, str, dict[str, bytes]]:
        if script == SSH_PROBE:
            return "probe", script, {}
        # run_detached's own calls: launch starts `set -eu`; poll, collect and kill `d=`.
        if (
            script.startswith("set -eu\n")
            or script.startswith("d=")
            or re.match(r"^tail -c \d+ ", script)
        ):
            return "ctl", script, {}
        try:
            head, files = split_payload(script)
        except Exception:
            return "unknown", script, {}
        try:
            name = scripts.script_name(head)
        except ValueError:
            return "unknown", head, files
        return name, head, files

    # serve_bootstrap ---------------------------------------------------------

    async def _bootstrap(self, pod: FakePod, script: str, files: dict[str, bytes]) -> ExecResult:
        def fail(message: str) -> ExecResult:
            pod.bootstraps.append("failed")
            return ExecResult(1, "", f"fail: {message}\n")

        if pod.behaviour.fail_bootstrap:
            return fail("injected bootstrap failure")
        if not files:
            return fail("stdin does not carry a loom-payload-v1 payload")
        if set(files) != set(PAYLOAD_FILES):
            return fail(f"the payload holds {sorted(files)}, not {list(PAYLOAD_FILES)}")
        for name in PAYLOAD_FILES:
            if not files[name]:
                return fail(f"the payload has no {name}")
        try:
            sha = scripts.script_var(script, "BUNDLE_SHA256")
            upstream = scripts.script_var(script, "UPSTREAM")
        except KeyError as e:
            self.protocol_errors.append(f"serve_bootstrap without {e}")
            return fail(f"missing variable {e}")
        if hashlib.sha256(files["proxy-bundle"]).hexdigest() != sha:
            return fail("the proxy bundle does not match BUNDLE_SHA256")
        if upstream != UPSTREAM:
            self.protocol_errors.append(f"serve_bootstrap UPSTREAM={upstream!r}, not {UPSTREAM}")
        pod.payloads.append(dict(files))
        d = pod.dir / "proxy"
        d.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = d / name
            path.write_bytes(data)
            path.chmod(0o600)
        if pod.proxy is not None:
            reload(pod.proxy, pod.files)  # SIGHUP: all or nothing, open connections kept
            pod.bootstraps.append("reloaded")
            return ExecResult(0, f"loom-proxy reloaded {pod.pid}\n", "")
        try:
            token, networks, ctx = pod.files.load()
            config = ProxyConfig(
                token=token,
                upstream=f"http://127.0.0.1:{pod.engine_port}",
                allowed_networks=networks,
            )
        except Exception as e:
            return fail(f"the proxy exited at start: {type(e).__name__}")
        proxy = PodProxy(config, ctx)
        await proxy.start("127.0.0.1", pod.proxy_port)  # binds next to the reservation
        if pod.proxy_sock is not None:
            pod.proxy_sock.close()
            pod.proxy_sock = None
        pod.proxy = proxy
        pod.pid = next(self._pids)
        pod.bootstraps.append("started")
        return ExecResult(0, f"loom-proxy started {pod.pid}\n", "")

    # serve_lease -------------------------------------------------------------

    def _lease(self, pod: FakePod, script: str) -> ExecResult:
        pod.lease_attempts.append(self.clock())
        try:
            lease_file = scripts.script_var(script, "LEASE_FILE")
            epoch = int(str(scripts.script_var(script, "LEASE_EPOCH")))
        except (KeyError, ValueError) as e:
            self.protocol_errors.append(f"serve_lease: bad variables ({e})")
            return ExecResult(1, "", "fail: bad LEASE_EPOCH\n")
        if lease_file != LEASE_FILE:
            self.protocol_errors.append(f"serve_lease LEASE_FILE={lease_file!r}")
        if pod.behaviour.fail_lease:
            return ExecResult(255, "", "ssh: connect to host: Connection timed out\n")
        pod.leases.append((self.clock(), epoch))
        return ExecResult(0, f"loom-lease {epoch}\n", "")

    # serve_engine through run_detached ------------------------------------------

    async def _control(self, pod: FakePod, script: str, call: ExecCall) -> ExecResult:
        tail = re.match(r"^tail -c \d+ (\S+)/stderr", script)
        if tail:
            job = self._ctl.get(tail.group(1))
            if job is None or not job.task.done() or job.task.cancelled():
                return ExecResult(0, "", "")
            return ExecResult(0, job.task.result()[2], "")
        m = re.search(r"^d=(\S+)$", script, re.MULTILINE)
        if m is None:
            self.protocol_errors.append(f"unexpected control script {script[:120]!r}")
            return ExecResult(2, "", "unexpected\n")
        ctl = m.group(1)
        if "loom-launched" in script:
            body = script.split("<<'LOOM_RUN_EOF'\n", 1)[1].rsplit("\nLOOM_RUN_EOF\n", 1)[0]
            try:
                name = scripts.script_name(body)
            except ValueError:
                name = "unknown"
            if name != "serve_engine":
                self.protocol_errors.append(f"run_detached launched {name}, not serve_engine")
            call.kind = "serve_engine"
            call.script = body
            pod.engine_scripts.append(body)
            pod.engine_held = pod.behaviour.engine == "hang"
            self._ctl[ctl] = _Detached(pod, asyncio.ensure_future(self._engine_job(pod, body)))
            return ExecResult(0, "loom-launched\n", "")
        job = self._ctl.get(ctl)
        if "loom-running" in script:
            if job is not None and not job.task.done() and not pod.engine_held:
                # Not held: answer only once the job is done, so a bootstrap costs one
                # poll of fake time however long the real engine takes to start.
                await asyncio.wait({job.task}, timeout=ENGINE_JOB_WAIT_S)
            if job is None or not job.task.done():
                return ExecResult(0, "loom-running\n", "")
            if job.task.cancelled():
                return ExecResult(0, "loom-rc 137\n", "")
            return ExecResult(0, f"loom-rc {job.task.result()[0]}\n", "")
        if "kill -KILL" in script:
            if job is not None:
                job.task.cancel()
                self._ctl.pop(ctl, None)
            return ExecResult(0, "", "")
        # collect
        self._ctl.pop(ctl, None)
        if job is None or not job.task.done() or job.task.cancelled():
            return ExecResult(0, "", "")
        return ExecResult(0, job.task.result()[1], "")

    async def _engine_job(self, pod: FakePod, body: str) -> tuple[int, str, str]:
        while pod.behaviour.engine == "hang":
            await pod.released.wait()
            pod.released.clear()
        pod.engine_held = False
        if pod.behaviour.engine == "fail":
            stall = _var(body, "ENGINE_STALL_S")
            return 1, "", f"fail: the engine made no progress for {stall} s\n"
        served = _var(body, "SERVED_MODEL")
        pod.served_model = served
        models = list(dict.fromkeys(m for m in (served, pod.model) if m))
        sock, pod.engine_sock = pod.engine_sock, None
        if sock is None:
            return 1, "", "fail: the engine port is gone\n"
        engine = FakeEngine(sock, models, **self.engine_overrides)
        pod.engine = engine
        try:
            await engine.start()
        except RuntimeError as e:
            return 1, "", f"fail: {e}\n"
        now = self.clock()
        lines = [
            f"loom-stage image_pulled {pod.created_at + 1}",
            f"loom-stage sshd_ready {pod.created_at + 2}",
            f"loom-stage weights_ready {now + 0.1}",
            f"loom-stage engine_started {now + 0.2}",
            f"loom-stage engine_healthy {now + 0.3}",
            f"loom-stage first_token {now + 0.4}",
            "loom-sys gpus NVIDIA L40S",
            "loom-sys gpu_count 1",
            "loom-sys engine_isolation ok",
        ]
        return 0, "\n".join(lines) + "\n", ""


def _var(script: str, name: str) -> str:
    try:
        value = scripts.script_var(script, name)
    except KeyError:
        return ""
    return value if isinstance(value, str) else " ".join(value)


# ---------------------------------------------------------------------- fake RunPod


class FakeServingRunpod(FakeRunpod):
    """`FakeRunpod` whose created pods are fake serving pods (see the module docstring)."""

    def __init__(
        self,
        pods: FakeServingExec,
        *,
        clock: Callable[[], float] = time.time,
        proxy_port: int = 8443,
        cost_per_hr: float = 1.09,
        stock: Iterable[str] = US_STOCK,
    ) -> None:
        super().__init__(cost_per_hr=cost_per_hr)
        self.serving = pods
        self.clock = clock
        self.proxy_port = proxy_port
        self.stock = [list(stock)]
        self.stock_gpu = "NVIDIA L40S"
        # Listings/GETs answered before a created pod gets its IP and port mappings.
        self.mapping_delay = 0
        # Creates that make the pod but answer this status, or time out ("timeout").
        self.ambiguous: list[int | str] = []
        # When set: a POST /pods waits (in RunPod's worker thread) until the event is set.
        self.create_gate: threading.Event | None = None
        # REST DELETE and GraphQL podTerminate both answer 500 (nothing is terminated).
        self.terminate_fails = False
        self.create_times: list[float] = []  # the fake clock at each POST /pods
        self.terminated: list[str] = []  # pod ids terminated through the API
        self._pending: dict[str, tuple[int, dict[str, int]]] = {}
        self._rlock = threading.RLock()

    def handle(self, request: httpx.Request) -> httpx.Response:
        rest = request.url.path.removeprefix(httpx.URL(REST).path)
        gate = self.create_gate
        if gate is not None and request.method == "POST" and rest == "/pods":
            gate.wait(60)  # outside the lock: listings and terminates still answer
        with self._rlock:
            return self._handle(request)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        rest = request.url.path.removeprefix(httpx.URL(REST).path)
        on_rest = request.url.host == httpx.URL(REST).host
        terminate = (on_rest and request.method == "DELETE" and rest.startswith("/pods/")) or (
            not on_rest and b"podTerminate" in request.content
        )
        if terminate and self.terminate_fails:
            self.requests.append(request)
            return httpx.Response(500, json={"error": "injected: terminate failed"})
        if on_rest and request.method == "POST" and rest == "/pods":
            self.create_times.append(self.clock())
            mode = self.ambiguous.pop(0) if self.ambiguous else None
            response = super().handle(request)
            if response.status_code == 201:
                self._serve(self.pods[response.json()["id"]])
                if mode == "timeout":
                    raise httpx.ReadTimeout("injected: the create timed out", request=request)
                if isinstance(mode, int):
                    return httpx.Response(mode, json={"error": "injected", "status": mode})
            return response
        if on_rest and request.method == "GET" and (rest == "/pods" or rest.startswith("/pods/")):
            self._count_get()
        response = super().handle(request)
        if (
            on_rest
            and request.method == "DELETE"
            and rest.startswith("/pods/")
            and response.status_code < 300
        ):
            self._gone(rest.removeprefix("/pods/"))
        if not on_rest and b"podTerminate" in request.content:
            body = json.loads(request.content)
            self._gone(str((body.get("variables") or {}).get("podId")))
        return response

    def _serve(self, pod: dict[str, Any]) -> None:
        fake = self.serving.register(pod["id"], pod["name"], dict(pod.get("env") or {}))
        pod["desiredStatus"] = "RUNNING"
        dcs = pod.get("dataCenterIds") or []
        pod["machine"] = {**(pod.get("machine") or {}), "dataCenterId": dcs[0] if dcs else ""}
        mappings = {"22": fake.ssh_port, str(self.proxy_port): fake.proxy_port}
        if self.mapping_delay > 0:
            pod["publicIp"] = ""
            pod["portMappings"] = {}
            self._pending[pod["id"]] = (self.mapping_delay, mappings)
        else:
            pod["publicIp"] = "127.0.0.1"
            pod["portMappings"] = mappings

    def _count_get(self) -> None:
        for pod_id, (left, mappings) in list(self._pending.items()):
            pod = self.pods.get(pod_id)
            if pod is None:
                self._pending.pop(pod_id)
            elif left <= 1:
                pod["publicIp"] = "127.0.0.1"
                pod["portMappings"] = mappings
                self._pending.pop(pod_id)
            else:
                self._pending[pod_id] = (left - 1, mappings)

    def _gone(self, pod_id: str) -> None:
        self.terminated.append(pod_id)
        self.serving.pod_gone(pod_id)

    # Test controls --------------------------------------------------------------

    def snapshot(self) -> list[dict[str, Any]]:
        """Every pod RunPod lists now (a copy)."""
        with self._rlock:
            return [dict(p) for p in self.pods.values()]

    def has(self, pod_id: str) -> bool:
        with self._rlock:
            return pod_id in self.pods

    def of(self, model_id: str) -> list[str]:
        """Ids of the pods of `model_id` RunPod still lists."""
        return [
            p["id"] for p in self.snapshot() if (p.get("env") or {}).get("LOOM_MODEL") == model_id
        ]

    def remove_pod(self, pod_id: str) -> None:
        """RunPod lost the pod: it is no longer listed (not through the API)."""
        with self._rlock:
            self.pods.pop(pod_id, None)
        self.serving.pod_gone(pod_id)

    def set_status(self, pod_id: str, status: str) -> None:
        """E.g. EXITED: still listed, but its container (proxy, engine) is gone."""
        with self._rlock:
            self.pods[pod_id]["desiredStatus"] = status
        if status != "RUNNING":
            self.serving.pod_gone(pod_id)

    def creates(self) -> list[dict[str, Any]]:
        """Every POST /pods body."""
        return [b for b in self.bodies("POST", "/v1/pods") if isinstance(b, dict)]

    def terminate_requests(self, pod_id: str) -> int:
        """REST DELETEs and GraphQL podTerminates of `pod_id` (answered or refused)."""
        n = 0
        for r in list(self.requests):
            if (r.method == "DELETE" and r.url.path == f"/v1/pods/{pod_id}") or (
                r.method == "POST" and b"podTerminate" in r.content and pod_id.encode() in r.content
            ):
                n += 1
        return n


# ---------------------------------------------------------------------- config


def unique_model_id(base: str = "qwen3-8b") -> str:
    """A model id of its own for one test (`[a-z0-9.-]+`), so parallel tests sharing a
    Redis never share a pool hash, a wake key or usage events."""
    return f"{base}-t{uuid.uuid4().hex[:8]}"


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def serving_config(
    models: Mapping[str, Mapping[str, Any]] | Iterable[str], **sections: Any
) -> ServingConfig:
    """A `provider: fake` serving config with short timings for `models` (ids, or ids
    mapped to `ServedModel` overrides); `sections` override any other section's keys."""
    per_model = (
        {m: dict(v) for m, v in models.items()}
        if isinstance(models, Mapping)
        else {m: {} for m in models}
    )
    model_defaults = {
        "min_replicas": 0,
        "max_replicas": 1,
        "idle_stop_s": 3600,
        "cold_start_estimate_s": 60,
        "engine_env": {},
    }
    base: dict[str, Any] = {
        "version": 1,
        "provider": "fake",
        "spend": {
            "monthly_cap_usd": 100,
            "alert_fractions": [0.5, 0.8, 1.0],
            "max_price_ratio": 1.25,
        },
        "placement": {
            "data_center_prefixes": ["US-"],
            "allowed_cuda_versions": ["13.0"],
            "container_disk_gb": {m: 80 for m in per_model},
        },
        "network": {"gateway_egress_cidrs": [], "proxy_port": 8443},
        "lifecycle": {
            "reconcile_every_s": 5,
            "lease_ttl_s": 600,
            "lease_renew_every_s": 60,
            "drain_max_s": 60,
            "server_cert_days": 7,
            "server_cert_renew_before_s": 172800,
            "max_replacements_per_hour": 3,
        },
        "health": {
            "pod_state_every_s": 5,
            "proxy_health_every_s": 5,
            "proxy_health_failures": 3,
            "unhealthy_grace_s": 30,
            "progress_stall_s": 20,
            "bootstrap_stall_s": 60,
            "bootstrap_timeout_s": 600,
        },
        "capacity": {"backoff_s": [10, 20, 40], "unavailable_retry_after_s": 300},
        "models": {m: {**model_defaults, **v} for m, v in per_model.items()},
    }
    return ServingConfig.model_validate(_merge(base, sections))


_REGISTRY: list[Registry] = []


def registry() -> Registry:
    if not _REGISTRY:
        _REGISTRY.append(load_registry())
    return _REGISTRY[0]


def deployment(
    config: ServingConfig,
    model_id: str,
    *,
    base: str = "qwen3-8b",
    listed_usd_per_hr: float | None = 1.0,
) -> Deployment:
    """A `Deployment` of `model_id` (a test's own id) built on registry row `base`."""
    spec = registry().get(base)
    return Deployment(
        model_id=model_id,
        served=config.models[model_id],
        spec=spec,
        gpu_type_id=RUNPOD_GPU_TYPE_IDS[spec.hardware.gpu],
        gpu_count=spec.hardware.gpus_per_replica,
        container_disk_gb=config.placement.container_disk_gb[model_id],
        listed_price_per_hr_micros=(
            None if listed_usd_per_hr is None else usd_micros(listed_usd_per_hr)
        ),
    )


def start_command() -> str:
    """The rendered serve_start script (the pod's start command)."""
    return scripts.render(
        "serve_start",
        LEASE_FILE=LEASE_FILE,
        LEASE_POLL_S=LEASE_POLL_S,
        STAGE_FILE=STAGE_FILE,
        LOOM_ROOT=LOOM_ROOT,
        SSH_DIR=SSH_DIR,
        SSHD_CONFIG=SSHD_CONFIG,
        GRAPHQL_URL=GRAPHQL_URL,
    )


def make_ssh_identity(directory: Path) -> SshIdentity:
    """A fresh ed25519 key (OpenSSH format) and its public line."""
    directory.mkdir(parents=True, exist_ok=True)
    key = ed25519.Ed25519PrivateKey.generate()
    key_path = directory / "id_ed25519"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)
    public = key.public_key().public_bytes(
        serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
    )
    line = public.decode() + " loom-poolmgr"
    key_path.with_name("id_ed25519.pub").write_text(line + "\n")
    known = directory / "known_hosts"
    known.mkdir(exist_ok=True)
    return SshIdentity(key_path=key_path, public_key=line, known_hosts_dir=known)


def make_bundle() -> ProxyBundle:
    data = b"loom-podproxy test bundle " + uuid.uuid4().bytes
    return ProxyBundle(data=data, sha256=hashlib.sha256(data).hexdigest())


# ---------------------------------------------------------------------- the whole fake


class FakeServing:
    """Everything a `PoolManager` needs, faked (see the module docstring)."""

    def __init__(
        self,
        root: Path,
        *,
        redis: Redis,
        database_url: str,
        clock: FakeClock | None = None,
        proxy_port: int = 8443,
    ) -> None:
        self.root = root
        self.clock = clock or FakeClock()
        self.pods = FakeServingExec(root / "pods", clock=self.clock)
        self.runpod = FakeServingRunpod(self.pods, clock=self.clock, proxy_port=proxy_port)
        self.pki = Pki.generate(root / "pki")
        self.ssh = make_ssh_identity(root / "ssh")
        self.bundle = make_bundle()
        self.leases: MemoryLeaseRecords = MemoryLeaseRecords()
        self.redis = redis
        self.database_url = database_url
        # The admin queue key (spec §11): one per test, so parallel managers never pop
        # each other's commands.
        self.admin_queue = f"loom:poolmgr:admin:test:{uuid.uuid4().hex}"
        self.managers: list[PoolManager] = []
        self._clients: list[httpx.AsyncClient] = []

    def provider(self, config: ServingConfig) -> RunpodServing:
        return RunpodServing(
            self.runpod.api(user_agent=USER_AGENT),
            placement=config.placement,
            network=config.network,
            start_command=start_command(),
        )

    def manager(
        self,
        config: ServingConfig,
        *,
        deployments: Mapping[str, Deployment] | None = None,
        servable: tuple[str, ...] = (),
        leases: LeaseRecords | None = None,
        listed_usd_per_hr: float | None = 1.0,
    ) -> PoolManager:
        if deployments is None:
            deployments = {
                m: deployment(config, m, listed_usd_per_hr=listed_usd_per_hr) for m in config.models
            }
        manager = PoolManager(
            config,
            deployments=deployments,
            provider=self.provider(config),
            pod_exec=self.pods,
            redis=self.redis,
            database_url=self.database_url,
            leases=leases if leases is not None else self.leases,
            pki=self.pki,
            ssh=self.ssh,
            bundle=self.bundle,
            servable=servable,
            clock=self.clock,
            sleep=self.clock.sleep,
            admin_queue=self.admin_queue,  # type: ignore[call-arg]  # spec §11; phase B adds it
        )
        self.managers.append(manager)
        return manager

    async def admin(self, command: Mapping[str, str]) -> None:
        """`loom-poolmgr drain|stop|reset`, on this test's admin queue."""
        await push_admin(self.redis, command, queue=self.admin_queue)  # type: ignore[call-arg]

    def client(self) -> httpx.AsyncClient:
        """An mTLS client with the manager's client certificate (closed by `aclose`)."""
        c = proxy_client(self.pki)
        self._clients.append(c)
        return c

    async def wake(self, model_id: str) -> None:
        """Set (or refresh) the model's wake key, as `request_wake` does, without
        publishing on the shared wake channel."""
        await self.redis.set(wake_key(model_id), int(time.time() * 1000), ex=WAKE_TTL_S)

    async def pool(self, model_id: str) -> PoolState:
        return await get_pool(self.redis, model_id)

    async def rows(self, model_id: str | None = None) -> list[dict[str, Any]]:
        return await pod_rows(self.database_url, model_id)

    async def aclose(self) -> None:
        if self.runpod.create_gate is not None:
            self.runpod.create_gate.set()
        for manager in self.managers:
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(manager.stop(), 15)
        for c in self._clients:
            await c.aclose()
        await self.pods.aclose()


async def bring_up(
    env: FakeServing, manager: PoolManager, model_id: str, *, wake: bool = True, max_ticks: int = 8
) -> FakePod:
    """Wake the model (unless `wake` is False), tick until its pool is ready, and return
    its one live pod."""
    for _ in range(max_ticks):
        if wake:
            await env.wake(model_id)
        await tick(manager, env.clock, advance=1)
        if (await env.pool(model_id)).routable:
            break
    else:
        raise AssertionError(f"{model_id} never became ready: {await env.pool(model_id)}")
    pods = [p for p in env.pods.live() if p.model == model_id]
    assert len(pods) == 1, [p.pod_id for p in pods]
    return pods[0]


# ---------------------------------------------------------------------- talking to pods


def proxy_client(pki: Pki) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=pki.client_ssl_context(), timeout=10, trust_env=False)


async def proxy_get(
    client: httpx.AsyncClient, replica: Replica, path: str, *, token: str | None = None
) -> httpx.Response:
    return await client.get(
        replica.url + path,
        headers={TOKEN_HEADER: replica.token if token is None else token},
        extensions={"sni_hostname": replica.server_name},
    )


async def proxy_complete(
    client: httpx.AsyncClient,
    replica: Replica,
    model: str,
    *,
    token: str | None = None,
    max_tokens: int = 1,
) -> httpx.Response:
    return await client.post(
        replica.url + "/v1/completions",
        json={"model": model, "prompt": "hello", "max_tokens": max_tokens},
        headers={TOKEN_HEADER: replica.token if token is None else token},
        extensions={"sni_hostname": replica.server_name},
    )


async def peer_certificate(replica: Replica, pki: Pki) -> bytes:
    """The DER certificate a pod's proxy presents over mTLS (server name = replica's)."""
    host, port = replica.url.removeprefix("https://").rsplit(":", 1)
    _, writer = await asyncio.open_connection(
        host, int(port), ssl=pki.client_ssl_context(), server_hostname=replica.server_name
    )
    try:
        der: bytes = writer.get_extra_info("ssl_object").getpeercert(binary_form=True)
        return der
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


@dataclass
class StreamResult:
    status: int | None = None
    chunks: int = 0  # SSE events with a choice
    completion_tokens: int | None = None  # from the usage chunk
    finish_reason: str | None = None
    done: bool = False  # saw [DONE]
    error: str | None = None

    @property
    def complete(self) -> bool:
        return self.status == 200 and self.done and self.error is None


class ProxyStream:
    """A streaming completion through a pod's proxy, read in the background."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        replica: Replica,
        model: str,
        *,
        max_tokens: int = 100,
        token: str | None = None,
    ) -> None:
        self.client = client
        self.replica = replica
        self.model = model
        self.max_tokens = max_tokens
        self.token = replica.token if token is None else token
        self.result = StreamResult()
        self.first = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self, timeout_s: float = 10.0) -> ProxyStream:
        """Start, and return once the first token arrived."""
        self._task = asyncio.ensure_future(self._run())
        waiter = asyncio.ensure_future(self.first.wait())
        await asyncio.wait({waiter, self._task}, timeout=timeout_s, return_when="FIRST_COMPLETED")
        waiter.cancel()
        if not self.first.is_set():
            raise AssertionError(f"no first token: {self.result}")
        return self

    async def finish(self, timeout_s: float = 30.0) -> StreamResult:
        assert self._task is not None
        await asyncio.wait_for(asyncio.shield(self._task), timeout_s)
        return self.result

    @property
    def finished(self) -> bool:
        return self._task is not None and self._task.done()

    async def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task

    async def _run(self) -> None:
        body = {
            "model": self.model,
            "prompt": "count slowly",
            "max_tokens": self.max_tokens,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            async with self.client.stream(
                "POST",
                self.replica.url + "/v1/completions",
                json=body,
                headers={TOKEN_HEADER: self.token},
                extensions={"sni_hostname": self.replica.server_name},
                timeout=httpx.Timeout(10, read=120),
            ) as r:
                self.result.status = r.status_code
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        self.result.done = True
                        continue
                    event = json.loads(data)
                    for choice in event.get("choices") or ():
                        self.result.chunks += 1
                        self.first.set()
                        if choice.get("finish_reason"):
                            self.result.finish_reason = choice["finish_reason"]
                    usage = event.get("usage")
                    if usage:
                        self.result.completion_tokens = int(usage["completion_tokens"])
        except httpx.HTTPError as e:
            self.result.error = f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------- database


async def _connect(database_url: str, **kw: Any) -> psycopg.AsyncConnection[Any]:
    return await psycopg.AsyncConnection.connect(libpq_url(database_url), autocommit=True, **kw)


async def clear_serving_tables(database_url: str) -> None:
    """`serving_pods` and `serving_spend_alerts` emptied (adoption reads every live row,
    and month-to-date spend every row of the month)."""
    async with await _connect(database_url) as conn:
        await conn.execute("DELETE FROM serving_pods")
        await conn.execute("DELETE FROM serving_spend_alerts")


async def release_manager_locks(database_url: str) -> None:
    """End every other session holding an advisory lock in this (per-worker) database: a
    manager whose `stop()` hung must not leave the single-writer lock held, or every later
    test on the worker would fail with ManagerLocked."""
    async with await _connect(database_url) as conn:
        await conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_locks"
            " WHERE locktype = 'advisory' AND pid <> pg_backend_pid()"
            " AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
        )


async def pod_rows(database_url: str, model_id: str | None = None) -> list[dict[str, Any]]:
    """`serving_pods` rows (of one model), oldest first."""
    async with await _connect(database_url, row_factory=dict_row) as conn:
        if model_id is None:
            cur = await conn.execute("SELECT * FROM serving_pods ORDER BY created_at, pod_name")
        else:
            cur = await conn.execute(
                "SELECT * FROM serving_pods WHERE model_id = %s ORDER BY created_at, pod_name",
                (model_id,),
            )
        return list(await cur.fetchall())


async def insert_pod_row(database_url: str, **columns: Any) -> uuid.UUID:
    """A `serving_pods` row written directly (history for spend and adoption tests)."""
    columns.setdefault("id", uuid.uuid4())
    columns.setdefault("pod_name", f"loom-serve-history-{uuid.uuid4().hex[:12]}")
    if "stages" in columns:
        columns["stages"] = json.dumps(columns["stages"])
    names = ", ".join(columns)
    marks = ", ".join(["%s"] * len(columns))
    async with await _connect(database_url) as conn:
        await conn.execute(
            f"INSERT INTO serving_pods ({names}) VALUES ({marks})",  # type: ignore[arg-type]
            tuple(columns.values()),
        )
    row_id: uuid.UUID = columns["id"]
    return row_id


async def delete_pod_rows(database_url: str, model_id: str) -> None:
    async with await _connect(database_url) as conn:
        await conn.execute("DELETE FROM serving_pods WHERE model_id = %s", (model_id,))


async def spend_alerts(database_url: str) -> list[dict[str, Any]]:
    async with await _connect(database_url, row_factory=dict_row) as conn:
        cur = await conn.execute("SELECT * FROM serving_spend_alerts ORDER BY month, fraction")
        return list(await cur.fetchall())


async def dump_tables(database_url: str) -> str:
    """Every serving_pods and serving_spend_alerts row as JSON text (to search for
    secrets)."""
    out: list[str] = []
    async with await _connect(database_url) as conn:
        for table in ("serving_pods", "serving_spend_alerts"):
            cur = await conn.execute(
                f"SELECT row_to_json(t)::text FROM {table} t"  # type: ignore[arg-type]
            )
            out += [str(r[0]) for r in await cur.fetchall()]
    return "\n".join(out)


# ---------------------------------------------------------------------- fixture


@pytest.fixture(name="env")
async def env_fixture(
    tmp_path: Path, pg_url: str, redis_client: Redis
) -> AsyncIterator[FakeServing]:
    """The whole fake, on an emptied `serving_pods` and a fresh admin queue."""
    # Adoption reads every live row of the database, and spend every row of the month.
    await clear_serving_tables(pg_url)
    env = FakeServing(tmp_path, redis=redis_client, database_url=pg_url)
    try:
        yield env
    finally:
        try:
            await env.aclose()
        finally:
            await release_manager_locks(pg_url)
    assert not env.pods.protocol_errors, env.pods.protocol_errors
