"""Bootstrapping a serving pod over SSH (docs/phase1-m2-spec.md §5 "Bootstrap", §9).

The pod exec
------------
The manager talks to a pod only through `loom_bench.providers.runpod_ssh.SshExec`:

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult

`script` is everything the pod's `bash -s` reads on stdin; the result is ssh's exit
status (255: could not connect), stdout and stderr. In production it is `OpenSshExec`;
the local backend has its own (it writes the delivered files directly); tests use
`loom_poolmgr.testing.FakeServingExec`. `target` is `SshTarget(host=<Endpoints.ip>,
port=<Endpoints.ssh_port>, key_path=<LOOM_POOLMGR_SSH_KEY>, known_hosts=<per pod>)`.

The kinds of call (a fake must answer each; the order is at the end):

1. **SSH probe**, until it answers: `script == SSH_PROBE` (`"echo loom-online\\n"`).
   Answer `(0, "loom-online\\n", "")` once sshd is up, `(255, "", ...)` before.
2. **serve_bootstrap**, a direct `run` (never detached), `timeout_s=BOOTSTRAP_CALL_TIMEOUT_S`.
   `script == bootstrap_stdin(rendered, files)`: the rendered `serve_bootstrap` script,
   then a line `PAYLOAD_MARKER` (`loom-payload-v1`), then base64 (76-column lines) of an
   uncompressed tar archive. `split_payload(script)` returns `(rendered script,
   {name: bytes})`. The archive holds exactly `PAYLOAD_FILES`:

   - `server.crt`: the pod's server certificate (PEM; SAN DNS = pod id, serverAuth);
   - `server.key`: its private key (PEM, PKCS#8);
   - `ca.crt`: the Loom pod CA certificate (client certificates must chain to it);
   - `token`: the pod token (`X-Loom-Pod-Token`), one line;
   - `allow`: the proxy allowlist, one CIDR per line: `gateway_egress_cidrs`, or, when
     that is empty (allowed only for `local`/`fake`), `LOOPBACK_ALLOW`;
   - `proxy-bundle`: the pinned `loom-podproxy` bundle tarball (sha256 =
     `BUNDLE_SHA256`).

   Success: rc 0 and a stdout line `loom-proxy started <pid>` (first run) or
   `loom-proxy reloaded <pid>` (the proxy was running: files replaced, SIGHUP sent).
   A fake starts (or reloads) a real `loom-podproxy` from the files here.
3. **serve_engine**, through `run_detached` (`ctl_root=CTL_ROOT`): the launch, poll,
   stderr-tail and collect calls exactly as the bench's `FakePodExec` answers them (the
   rendered script is the body of the launch call's `<<'LOOM_RUN_EOF'` heredoc). No
   secrets. Success: rc 0; stdout carries `loom-stage <name> <epoch>` lines
   (`weights_ready`, `engine_started`, `engine_healthy`, `first_token`, plus
   serve_start's `image_pulled`, `sshd_ready` replayed) and `loom-sys engine_isolation ok`.
4. **serve_lease**, a direct `run`, `timeout_s=LEASE_CALL_TIMEOUT_S`, the rendered script
   only (no payload); `script_var(script, "LEASE_EPOCH")` is the new epoch. Success: rc 0
   and `loom-lease <epoch>` on stdout. Also run every `lease_renew_every_s` after
   bootstrap.

**Re-key** runs serve_bootstrap again with a new payload (same bundle). Adoption after a
restart sends a new token and server certificate, then runs serve_lease; certificate
renewal sends a new certificate with the pod's current token, so the gateway's cached
pool hash stays valid across the reload. A fake must replace the files and reload its
proxy, keeping open connections.

`loom_poolmgr.scripts.script_name(script)` says which template a rendered script is;
`script_var(script, NAME)` reads one of its variables. Secrets appear only in the
serve_bootstrap payload: never in a rendered script's variables, a command line, the pod env, a log
line or Postgres.

Bootstrap order (§5): wait for RUNNING and the 22/proxy port mappings; probe SSH;
serve_lease (the first renewal: from now on the lease file exists); serve_bootstrap;
serve_engine; then, from the manager, proxy `GET /health` and a one-token
`POST /v1/completions` (`loom_poolmgr.health`); the row goes `ready`.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import re
import tarfile
import textwrap
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from loom_bench.engines import engine_process_argv, launch_weights, render_launch
from loom_bench.providers import runpod_layout
from loom_bench.providers.aws_ssm import parse_markers
from loom_bench.providers.base import EngineLaunch
from loom_bench.providers.runpod_api import GRAPHQL_URL
from loom_bench.providers.runpod_ssh import (
    DEFAULT_CTL_ROOT,
    SSH_UNREACHABLE,
    RemoteCommandError,
    SshExec,
    SshTarget,
    run_detached,
)
from loom_poolmgr import scripts
from loom_poolmgr.provider import Deployment

SSH_PROBE = "echo loom-online\n"
PAYLOAD_MARKER = "loom-payload-v1"
PAYLOAD_FILES = ("server.crt", "server.key", "ca.crt", "token", "allow", "proxy-bundle")
CTL_ROOT = DEFAULT_CTL_ROOT
BOOTSTRAP_CALL_TIMEOUT_S = 600.0
LEASE_CALL_TIMEOUT_S = 60.0
PROBE_CALL_TIMEOUT_S = 30.0
PROMPT_TOKENS_DETAILS = "--enable-prompt-tokens-details"

# Fixed pod paths and ids (the values rendered into the scripts).
LEASE_FILE = "/run/loom/lease"
LEASE_POLL_S = 30
LOOM_ROOT = "/var/lib/loom"
STAGE_FILE = "/var/lib/loom/stages"
SSH_DIR = "/root/.ssh"
SSHD_CONFIG = "/var/lib/loom/sshd_config"
PROXY_DIR = "/etc/loom/proxy"
PROXY_VENV = "/opt/loom/proxy-venv"
ENGINE_PORT = 8000
UPSTREAM = f"127.0.0.1:{ENGINE_PORT}"
PROXY_UID = 10002
ENGINE_UID = 10003
# Loopback only: the allowlist for local/fake providers with no gateway_egress_cidrs.
LOOPBACK_ALLOW = ("127.0.0.0/8", "::1/128")


@dataclass(frozen=True)
class ProxyBundle:
    """The pinned `loom-podproxy` bundle (`loom-podproxy-bundle`, §9) and its sha256 from
    `config/serving-artifacts.lock`."""

    data: bytes
    sha256: str

    @classmethod
    def load(cls, path: Path, sha256: str) -> ProxyBundle:
        """Read `path` and check it against `sha256`."""
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha256.strip().lower():
            raise ValueError(f"{path}: sha256 {actual} does not match the lock's {sha256}")
        return cls(data, actual)

    def __repr__(self) -> str:
        return f"ProxyBundle({len(self.data)} bytes, sha256={self.sha256})"


@dataclass(frozen=True)
class PodSecrets:
    """What serve_bootstrap delivers (PEM bytes and the token). Never logged or stored."""

    server_cert: bytes
    server_key: bytes
    ca_cert: bytes
    token: str
    allow: tuple[str, ...]

    def __repr__(self) -> str:
        return "PodSecrets(<redacted>)"


def payload_files(secrets: PodSecrets, bundle: ProxyBundle) -> dict[str, bytes]:
    """The payload members (`PAYLOAD_FILES`) for one bootstrap or re-key."""
    if not secrets.allow:
        raise ValueError("an empty allowlist would refuse every connection")
    return {
        "server.crt": secrets.server_cert,
        "server.key": secrets.server_key,
        "ca.crt": secrets.ca_cert,
        "token": secrets.token.encode() + b"\n",
        "allow": "".join(f"{cidr}\n" for cidr in secrets.allow).encode(),
        "proxy-bundle": bundle.data,
    }


def encode_payload(files: Mapping[str, bytes]) -> str:
    """Base64 (76-column lines) of an uncompressed tar archive of `files`, members in
    `PAYLOAD_FILES` order, mode 0600, owner root, mtime 0."""
    if set(files) != set(PAYLOAD_FILES):
        raise ValueError(f"payload must hold exactly {PAYLOAD_FILES}, got {sorted(files)}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in PAYLOAD_FILES:
            data = files[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))
    return "\n".join(textwrap.wrap(base64.b64encode(buf.getvalue()).decode(), 76)) + "\n"


def decode_payload(text: str) -> dict[str, bytes]:
    """`encode_payload`'s inverse."""
    raw = base64.b64decode("".join(text.split()), validate=True)
    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        for member in tar.getmembers():
            if not member.isfile() or "/" in member.name:
                raise ValueError(f"unexpected payload member {member.name!r}")
            f = tar.extractfile(member)
            assert f is not None
            out[member.name] = f.read()
    return out


def bootstrap_stdin(script: str, files: Mapping[str, bytes]) -> str:
    """What serve_bootstrap's `bash -s` reads: the rendered script, the marker line, the
    payload."""
    if not script.endswith("\n"):
        script += "\n"
    return f"{script}{PAYLOAD_MARKER}\n{encode_payload(files)}"


def split_payload(stdin: str) -> tuple[str, dict[str, bytes]]:
    """(rendered script, payload files) from a serve_bootstrap stdin; a stdin with no
    payload gives an empty dict."""
    marker = f"\n{PAYLOAD_MARKER}\n"
    head, sep, tail = stdin.partition(marker)
    if not sep:
        return stdin, {}
    return head + "\n", decode_payload(tail)


def ssh_target(endpoints_ip: str, ssh_port: int, key_path: Path, known_hosts: Path) -> SshTarget:
    """The `SshTarget` for a pod (no ControlMaster: calls are few and short)."""
    return SshTarget(host=endpoints_ip, port=ssh_port, key_path=key_path, known_hosts=known_hosts)


class BootstrapError(RuntimeError):
    """A bootstrap step failed (the message never holds a secret)."""


def bootstrap_script(bundle: ProxyBundle, *, proxy_port: int) -> str:
    """The rendered serve_bootstrap (no secrets: they follow it on stdin)."""
    return scripts.render(
        "serve_bootstrap",
        PROXY_DIR=PROXY_DIR,
        PROXY_VENV=PROXY_VENV,
        PROXY_PORT=proxy_port,
        UPSTREAM=UPSTREAM,
        BUNDLE_SHA256=bundle.sha256,
        PROXY_UID=PROXY_UID,
        ENGINE_UID=ENGINE_UID,
    )


def lease_script(lease_epoch: int) -> str:
    """The rendered serve_lease moving the pod's lease to `lease_epoch`."""
    return scripts.render("serve_lease", LEASE_FILE=LEASE_FILE, LEASE_EPOCH=int(lease_epoch))


def start_command(graphql_url: str = GRAPHQL_URL) -> str:
    """The rendered serve_start (the pod's start command, `dockerStartCmd`)."""
    return scripts.render(
        "serve_start",
        LEASE_FILE=LEASE_FILE,
        LEASE_POLL_S=LEASE_POLL_S,
        STAGE_FILE=STAGE_FILE,
        LOOM_ROOT=LOOM_ROOT,
        SSH_DIR=SSH_DIR,
        SSHD_CONFIG=SSHD_CONFIG,
        GRAPHQL_URL=graphql_url,
    )


def engine_launch(model: Deployment) -> EngineLaunch:
    """The registry row's launch with the serving config's `engine_env`."""
    launch = render_launch(model.spec)
    return launch.model_copy(update={"env": {**launch.env, **model.served.engine_env}})


def engine_script(model: Deployment, *, stall_s: int, ready_timeout_s: int) -> str:
    """The rendered serve_engine for `model` (no secrets): the bench's launch, bound to
    loopback, plus `--enable-prompt-tokens-details` for vLLM (plan §9)."""
    launch = engine_launch(model)
    argv = engine_process_argv(launch, host="127.0.0.1", weights_dir=runpod_layout.WEIGHTS_DIR)
    if launch.engine == "vllm" and PROMPT_TOKENS_DETAILS not in argv:
        argv.append(PROMPT_TOKENS_DETAILS)
    return scripts.render(
        "serve_engine",
        WARM=0,
        FETCH_WEIGHTS=[x for pair in launch_weights(launch) for x in pair],
        CACHED_WEIGHTS=[],
        WEIGHTS_DIR=runpod_layout.WEIGHTS_DIR,
        PORT=ENGINE_PORT,
        SERVED_MODEL=launch.served_model,
        READY_TIMEOUT_S=int(ready_timeout_s),
        ENGINE_STALL_S=int(stall_s),
        LOG_DIR=runpod_layout.LOG_DIR,
        STAGE_FILE=STAGE_FILE,
        ENGINE_ENV=[f"{k}={v}" for k, v in sorted(launch.env.items())],
        ENGINE_CMD=argv,
        ENGINE_PIDFILE=runpod_layout.ENGINE_PIDFILE,
        PROC1_ENVIRON=runpod_layout.PROC1_ENVIRON,
        SCAN_DIRS=list(runpod_layout.SECRET_SCAN_DIRS),
        ENGINE_UID=ENGINE_UID,
        PROXY_DIR=PROXY_DIR,
    )


async def wait_for_ssh(
    pod_exec: SshExec,
    target: SshTarget,
    *,
    timeout_s: float,
    poll_s: float,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    check: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Probe with `SSH_PROBE` until sshd answers; `TimeoutError` after `timeout_s` (on
    `clock`). `check` runs between probes and raises when the pod is lost."""
    deadline = clock() + timeout_s
    while True:
        res = await pod_exec.run(target, SSH_PROBE, timeout_s=PROBE_CALL_TIMEOUT_S)
        if res.returncode == 0 and "loom-online" in res.stdout:
            return
        if res.returncode != SSH_UNREACHABLE:
            raise BootstrapError(f"SSH probe exited {res.returncode}")
        if check is not None:
            await check()
        if clock() >= deadline:
            raise TimeoutError(f"SSH not reachable after {timeout_s:.0f} s")
        await sleep(poll_s)


_PROXY_RE = re.compile(r"^loom-proxy (started|reloaded) (\d+)$", re.MULTILINE)


async def deliver(
    pod_exec: SshExec,
    target: SshTarget,
    secrets: PodSecrets,
    bundle: ProxyBundle,
    *,
    proxy_port: int,
) -> str:
    """Run serve_bootstrap with the payload; returns `started` or `reloaded`. Raises
    `RemoteCommandError` on a non-zero exit (stderr is the script's own, never the
    payload)."""
    stdin = bootstrap_stdin(
        bootstrap_script(bundle, proxy_port=proxy_port), payload_files(secrets, bundle)
    )
    res = await pod_exec.run(target, stdin, timeout_s=BOOTSTRAP_CALL_TIMEOUT_S)
    if res.returncode != 0:
        raise RemoteCommandError("serve_bootstrap", res.returncode, res.stdout, res.stderr)
    m = _PROXY_RE.search(res.stdout)
    if m is None:
        raise RemoteCommandError("serve_bootstrap: no proxy line", 0, res.stdout, res.stderr)
    return m.group(1)


async def start_engine(
    pod_exec: SshExec,
    target: SshTarget,
    script: str,
    *,
    timeout_s: float,
    poll_s: float,
    check_host: Callable[[], Awaitable[None]],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict[str, float]:
    """Run the rendered serve_engine detached; returns the stage epochs it reported.

    `run_detached`'s own deadline is real time (`timeout_s`); callers on an injected clock
    enforce theirs in `check_host`, which runs every poll."""
    stdout = await run_detached(
        pod_exec,
        target,
        script,
        what="serve_engine",
        timeout_s=timeout_s,
        poll_s=poll_s,
        check_host=check_host,
        ctl_root=CTL_ROOT,
        sleep=sleep,
    )
    stages, system = parse_markers(stdout)
    if system.get("engine_isolation") != "ok":
        raise BootstrapError("serve_engine did not prove the engine's isolation")
    return stages
