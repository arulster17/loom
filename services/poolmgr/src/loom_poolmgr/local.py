"""The `local` backend (docs/phase1-m2-spec.md §9): pods as subprocess pairs, for compose.

Each "pod" is the bench mock engine (`python -m loom_bench.mock`) on a loopback port plus
a real `loom-podproxy` in front of it, with the certificate, token and allowlist the
manager delivers, both subprocesses of the manager. `LocalServing` is the
`ServingProvider`; `LocalServing.exec` is the matching `SshExec`, which answers the pod
exec protocol of `loom_poolmgr.bootstrap` without SSH: it writes the delivered files
into the pod's directory and starts (or SIGHUPs) the proxy, starts the engine for
serve_engine, and records serve_lease's epoch. The create, health, drain, idle and pool
hash code in the manager is the same as for RunPod.

`cold_start_s` (`LOOM_POOLMGR_LOCAL_COLD_START_S`) delays a pod's `RUNNING` status, as
an image pull would. `no_capacity` makes every create raise `NoCapacity` (the
acceptance's switch). The proxies listen on `bind` and are advertised at `host` (in
compose: `0.0.0.0` and the poolmgr service's name, with the compose network in
`gateway_egress_cidrs`, which `provider: local` allows).

Needs the bench's `mock` extra (FastAPI, uvicorn) installed next to the manager.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from loom_bench.providers.runpod_ssh import SSH_UNREACHABLE, ExecResult, SshTarget
from loom_poolmgr import scripts
from loom_poolmgr.bootstrap import PAYLOAD_FILES, SSH_PROBE, split_payload
from loom_poolmgr.provider import (
    POD_NAME_RE,
    Deployment,
    Endpoints,
    NoCapacity,
    PodInfo,
)

log = logging.getLogger(__name__)

_PROXY_FILES = ("server.crt", "server.key", "ca.crt", "token", "allow")
_HEREDOC_START = "<<'LOOM_RUN_EOF'\n"
_HEREDOC_END = "\nLOOM_RUN_EOF\n"


def _free_port(bind: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((bind, 0))
        return int(s.getsockname()[1])


@dataclass
class _Job:
    """serve_engine, launched: the engine process and when it was started."""

    started: float
    stdout: str = ""
    rc: int | None = None


@dataclass
class _LocalPod:
    pod_id: str
    name: str
    env: dict[str, str]
    created_at: float
    ssh_port: int  # a key for the exec, not a real port
    proxy_port: int
    engine_port: int
    dir: Path
    proxy: subprocess.Popen[bytes] | None = None
    engine: subprocess.Popen[bytes] | None = None
    jobs: dict[str, _Job] = field(default_factory=dict)
    lease_epoch: int | None = None


class LocalServing:
    """`provider: local` (compose, §9): each pod is a mock engine on a loopback port plus
    a `loom-podproxy` with a certificate the manager issued, as subprocesses of the
    manager. `cold_start_s` (LOOM_POOLMGR_LOCAL_COLD_START_S) delays the pod's RUNNING
    status. `no_capacity` makes every create raise `NoCapacity` (the acceptance's switch).
    Its pod exec (`exec`) writes the delivered files directly instead of using SSH."""

    def __init__(
        self,
        *,
        cold_start_s: float,
        clock: Callable[[], float] = time.time,
        root: Path | None = None,
        host: str = "127.0.0.1",
        bind: str = "127.0.0.1",
        python: str = sys.executable,
    ) -> None:
        self.cold_start_s = cold_start_s
        self.clock = clock
        self.no_capacity = False
        self.root = root or Path(tempfile.mkdtemp(prefix="loom-local-pods-"))
        self.host = host
        self.bind = bind
        self.python = python
        self.pods: dict[str, _LocalPod] = {}
        self._ssh_ports = itertools.count(30001)
        self.exec = LocalPodExec(self)

    def _status(self, pod: _LocalPod) -> str:
        return "RUNNING" if self.clock() >= pod.created_at + self.cold_start_s else "CREATED"

    def _info(self, pod: _LocalPod) -> PodInfo:
        return PodInfo(
            pod_id=pod.pod_id,
            name=pod.name,
            status=self._status(pod),
            env=dict(pod.env),
            cost_per_hr_micros=None,
            data_center="local",
            raw={"id": pod.pod_id, "name": pod.name},
        )

    async def create(self, model: Deployment, name: str, env: dict[str, str]) -> PodInfo:
        if self.no_capacity:
            raise NoCapacity("local backend: no capacity (switched off)")
        if not POD_NAME_RE.match(name):
            raise ValueError(f"not a serving pod name: {name!r}")
        pod_id = f"local{secrets.token_hex(6)}"
        pod = _LocalPod(
            pod_id=pod_id,
            name=name,
            env=dict(env),
            created_at=self.clock(),
            ssh_port=next(self._ssh_ports),
            proxy_port=_free_port(self.bind),
            engine_port=_free_port("127.0.0.1"),
            dir=self.root / pod_id,
        )
        pod.dir.mkdir(parents=True, mode=0o700)
        self.pods[pod_id] = pod
        log.info("local pod %s (%s) created", pod_id, name)
        return self._info(pod)

    async def list(self) -> list[PodInfo]:
        return [self._info(p) for p in self.pods.values()]

    async def get(self, pod_id: str) -> PodInfo | None:
        pod = self.pods.get(pod_id)
        return None if pod is None else self._info(pod)

    async def terminate(self, pod_id: str) -> None:
        pod = self.pods.pop(pod_id, None)
        if pod is None:
            return
        for proc in (pod.proxy, pod.engine):
            if proc is not None:
                await asyncio.to_thread(_stop, proc)
        shutil.rmtree(pod.dir, ignore_errors=True)
        log.info("local pod %s terminated", pod_id)

    def endpoints(self, pod: PodInfo) -> Endpoints | None:
        found = self.pods.get(pod.pod_id)
        if found is None or self._status(found) != "RUNNING":
            return None
        return Endpoints(ip=self.host, proxy_port=found.proxy_port, ssh_port=found.ssh_port)

    async def aclose(self) -> None:
        """Terminate every pod (the manager's own shutdown leaves them; compose does
        not keep them across a restart anyway)."""
        for pod_id in list(self.pods):
            await self.terminate(pod_id)


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5)


def _var(script: str, name: str) -> str:
    try:
        value = scripts.script_var(script, name)
    except KeyError:
        return ""
    return value if isinstance(value, str) else " ".join(value)


class LocalPodExec:
    """`SshExec` for `LocalServing` pods (the protocol in `loom_poolmgr.bootstrap`)."""

    def __init__(self, serving: LocalServing) -> None:
        self.serving = serving

    def _pod(self, target: SshTarget) -> _LocalPod | None:
        for pod in self.serving.pods.values():
            if pod.ssh_port == target.port:
                return pod
        return None

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        pod = self._pod(target)
        if pod is None or self.serving._status(pod) != "RUNNING":
            return ExecResult(SSH_UNREACHABLE, "", "local pod: not running\n")
        if script == SSH_PROBE:
            return ExecResult(0, "loom-online\n", "")
        if script.startswith("set -eu\n") or script.startswith("d=") or script.startswith("tail "):
            return await self._control(pod, script)
        head, files = split_payload(script)
        try:
            name = scripts.script_name(head)
        except ValueError:
            return ExecResult(2, "", "local pod: unexpected script\n")
        if name == "serve_lease":
            epoch = _var(head, "LEASE_EPOCH")
            if not epoch.isdigit():
                return ExecResult(1, "", "fail: LEASE_EPOCH is not an epoch\n")
            pod.lease_epoch = int(epoch)
            (pod.dir / "lease").write_text(epoch + "\n")
            return ExecResult(0, f"loom-lease {epoch}\n", "")
        if name == "serve_bootstrap":
            return await asyncio.to_thread(self._bootstrap, pod, head, files)
        return ExecResult(2, "", f"local pod: {name} is not run directly\n")

    def _bootstrap(self, pod: _LocalPod, script: str, files: dict[str, bytes]) -> ExecResult:
        if set(files) != set(PAYLOAD_FILES) or not all(files.values()):
            return ExecResult(1, "", "fail: the payload is incomplete\n")
        if hashlib.sha256(files["proxy-bundle"]).hexdigest() != _var(script, "BUNDLE_SHA256"):
            return ExecResult(1, "", "fail: the proxy bundle does not match BUNDLE_SHA256\n")
        proxy_dir = pod.dir / "proxy"
        proxy_dir.mkdir(mode=0o700, exist_ok=True)
        for name in _PROXY_FILES:
            tmp = proxy_dir / f".{name}.new"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(files[name])
            tmp.replace(proxy_dir / name)
        if pod.proxy is not None and pod.proxy.poll() is None:
            os.kill(pod.proxy.pid, signal.SIGHUP)
            return ExecResult(0, f"loom-proxy reloaded {pod.proxy.pid}\n", "")
        log_file = (pod.dir / "proxy.log").open("ab")
        pod.proxy = subprocess.Popen(
            [
                self.serving.python,
                "-m",
                "loom_podproxy.proxy",
                "--host",
                self.serving.bind,
                "--port",
                str(pod.proxy_port),
                "--upstream",
                f"http://127.0.0.1:{pod.engine_port}",
                "--cert",
                str(proxy_dir / "server.crt"),
                "--key",
                str(proxy_dir / "server.key"),
                "--client-ca",
                str(proxy_dir / "ca.crt"),
                "--token-file",
                str(proxy_dir / "token"),
                "--allow-ip-file",
                str(proxy_dir / "allow"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        time.sleep(1.0)
        if pod.proxy.poll() is not None:
            return ExecResult(1, "", "fail: the proxy exited at start\n")
        return ExecResult(0, f"loom-proxy started {pod.proxy.pid}\n", "")

    async def _control(self, pod: _LocalPod, script: str) -> ExecResult:
        tail = re.match(r"^tail -c \d+ (\S+)/stderr", script)
        if tail:
            job = pod.jobs.get(tail.group(1))
            if job is None or job.rc in (None, 0):
                return ExecResult(0, "", "")
            return ExecResult(0, _tail(pod.dir / "engine.log"), "")
        m = re.search(r"^d=(\S+)$", script, re.MULTILINE)
        if m is None:
            return ExecResult(2, "", "local pod: unexpected control script\n")
        ctl = m.group(1)
        if "loom-launched" in script:
            body = script.split(_HEREDOC_START, 1)[1].rsplit(_HEREDOC_END, 1)[0]
            pod.jobs[ctl] = self._start_engine(pod, body)
            return ExecResult(0, "loom-launched\n", "")
        job = pod.jobs.get(ctl)
        if "loom-running" in script:
            if job is None:
                return ExecResult(0, "loom-rc 1\n", "")
            if job.rc is None:
                await self._poll_engine(pod, job)
            return ExecResult(0, "loom-running\n" if job.rc is None else f"loom-rc {job.rc}\n", "")
        if "kill -KILL" in script:
            if pod.engine is not None:
                await asyncio.to_thread(_stop, pod.engine)
            pod.jobs.pop(ctl, None)
            return ExecResult(0, "", "")
        pod.jobs.pop(ctl, None)
        return ExecResult(0, "" if job is None else job.stdout, "")

    def _start_engine(self, pod: _LocalPod, body: str) -> _Job:
        served = _var(body, "SERVED_MODEL")
        models = list(dict.fromkeys(m for m in (served, pod.env.get("LOOM_MODEL", "")) if m))
        argv = [self.serving.python, "-m", "loom_bench.mock", "--host", "127.0.0.1"]
        argv += ["--port", str(pod.engine_port), "--log-level", "warning"]
        for model in models:
            argv += ["--model", model]
        log_file = (pod.dir / "engine.log").open("ab")
        pod.engine = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        return _Job(started=time.time())

    async def _poll_engine(self, pod: _LocalPod, job: _Job) -> None:
        if pod.engine is None or pod.engine.poll() is not None:
            job.rc = 1
            return
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                r = await client.get(f"http://127.0.0.1:{pod.engine_port}/health")
        except httpx.HTTPError:
            return
        if r.status_code != 200:
            return
        now = time.time()
        lines = [
            f"loom-stage image_pulled {pod.created_at + self.serving.cold_start_s}",
            f"loom-stage sshd_ready {pod.created_at + self.serving.cold_start_s}",
            f"loom-stage weights_ready {job.started}",
            f"loom-stage engine_started {job.started}",
            f"loom-stage engine_healthy {now}",
            f"loom-stage first_token {now}",
            "loom-sys engine_isolation ok",
        ]
        job.stdout = "\n".join(lines) + "\n"
        job.rc = 0


def _tail(path: Path, limit: int = 4000) -> str:
    with contextlib.suppress(OSError):
        return path.read_bytes()[-limit:].decode(errors="replace")
    return ""
