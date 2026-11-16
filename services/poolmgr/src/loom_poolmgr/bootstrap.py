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

**Re-key** (adoption after a restart; certificate renewal) runs serve_bootstrap again
with a new payload (new token and server certificate, same bundle), then serve_lease. A
fake must replace the files and reload its proxy, keeping open connections.

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

import base64
import io
import tarfile
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loom_bench.providers.runpod_ssh import DEFAULT_CTL_ROOT, SshExec, SshTarget

SSH_PROBE = "echo loom-online\n"
PAYLOAD_MARKER = "loom-payload-v1"
PAYLOAD_FILES = ("server.crt", "server.key", "ca.crt", "token", "allow", "proxy-bundle")
CTL_ROOT = DEFAULT_CTL_ROOT
BOOTSTRAP_CALL_TIMEOUT_S = 600.0
LEASE_CALL_TIMEOUT_S = 60.0

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
        raise NotImplementedError("phase B")


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


async def wait_for_ssh(
    pod_exec: SshExec, target: SshTarget, *, timeout_s: float, poll_s: float
) -> None:
    """Probe with `SSH_PROBE` until sshd answers; `TimeoutError` after `timeout_s`."""
    raise NotImplementedError("phase B")


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
    raise NotImplementedError("phase B")


async def start_engine(
    pod_exec: SshExec,
    target: SshTarget,
    script: str,
    *,
    timeout_s: float,
    poll_s: float,
) -> dict[str, float]:
    """Run the rendered serve_engine detached; returns the stage epochs it reported."""
    raise NotImplementedError("phase B")
