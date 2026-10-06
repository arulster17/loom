"""Running scripts in a RunPod pod over direct SSH.

RunPod has no exec API, and its `ssh.runpod.io` proxy needs a PTY and reports no
exit status, so the provider uses OpenSSH to the pod's mapped TCP 22 (sshd is
installed by the pod start command). The remote command is always `bash -s` with
the script on stdin: presigned URLs never appear in argv on either side, in the
pod's environment, or in the RunPod API.

Like SSM, a script is launched detached and then polled: `run_detached` writes it
into a root-only control directory (`{ctl_root}/{uuid}`), starts it in its own
session, and polls for the exit-code file with short, fresh SSH calls. A dropped
connection therefore never kills a long job; consecutive connection failures
(ssh exit 255) are retried, and `check_host` runs every poll so a vanished pod is
reported as lost rather than as a failed command. On timeout the script's
process group is killed. The control directory (it holds the script, URLs and
all) is removed once its output has been read.
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

SSH_UNREACHABLE = 255  # OpenSSH's exit status for connection and auth failures
DEFAULT_CTL_ROOT = "/var/lib/loom/ctl"
_HEREDOC_END = "LOOM_RUN_EOF"
_STDERR_TAIL = 20_000
_HOST_RE = re.compile(r"^[A-Za-z0-9.:-]+$")
_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]*$")
_CTL_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")


@dataclass(frozen=True)
class SshTarget:
    host: str
    port: int
    key_path: Path
    known_hosts: Path
    user: str = "root"
    control_dir: Path | None = None  # ControlMaster sockets; None disables multiplexing

    def __post_init__(self) -> None:
        if not _HOST_RE.match(self.host):
            raise ValueError(f"invalid SSH host {self.host!r}")
        if not 0 < self.port < 65536:
            raise ValueError(f"invalid SSH port {self.port}")
        if not _USER_RE.match(self.user):
            raise ValueError(f"invalid SSH user {self.user!r}")


@dataclass(frozen=True)
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


class RemoteCommandError(RuntimeError):
    def __init__(self, what: str, returncode: int | None, stdout: str, stderr: str) -> None:
        status = "timed out" if returncode is None else f"exited {returncode}"
        super().__init__(f"{what} {status}: {stderr.strip()[-2000:]}")
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class SshExec(Protocol):
    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        """Run `script` with `bash -s` on the target; the exit status of ssh itself
        (255 when the connection fails)."""
        ...


def ssh_argv(target: SshTarget) -> list[str]:
    """OpenSSH argv for one non-interactive `bash -s` session.

    `-F /dev/null` ignores the user's config. Key-only, no agent or forwarding of
    any kind. The host key is trusted on first use and pinned in a per-pod
    `known_hosts` (RunPod publishes no host keys to verify against).
    """
    argv = [
        "ssh",
        "-F",
        "/dev/null",
        "-T",
        "-p",
        str(target.port),
        "-i",
        str(target.key_path),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ForwardX11=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={target.known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        "-o",
        "LogLevel=ERROR",
    ]
    if target.control_dir is not None:
        argv += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={target.control_dir}/%C",
            "-o",
            "ControlPersist=60",
        ]
    return [*argv, f"{target.user}@{target.host}", "bash -s"]


class OpenSshExec:
    """`SshExec` through the system `ssh` binary."""

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            *ssh_argv(target),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(script.encode()), timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(SSH_UNREACHABLE, "", f"ssh did not finish within {timeout_s}s")
        return ExecResult(
            proc.returncode if proc.returncode is not None else SSH_UNREACHABLE,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )


def generate_keypair(directory: Path) -> tuple[Path, str]:
    """A fresh ed25519 key pair in `directory` (made 0700); returns (private key path,
    public key line). The caller keeps the directory out of results and the repo."""
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    key = directory / "id_ed25519"
    if key.exists():
        raise FileExistsError(key)
    subprocess.run(
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "loom-bench", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    return key, key.with_suffix(".pub").read_text().strip()


def _launch_script(script: str, ctl: str) -> str:
    if _HEREDOC_END in script:
        raise ValueError("script contains the heredoc terminator")
    return (
        "set -eu\n"
        "umask 077\n"
        f"d={ctl}\n"
        'mkdir -p "$d"\n'
        f"cat >\"$d/run.sh\" <<'{_HEREDOC_END}'\n{script}\n{_HEREDOC_END}\n"
        # Its own session: the pid file holds the group to kill on timeout.
        'setsid bash -c \'echo $$ >"$1/pid"; bash "$1/run.sh" >"$1/stdout" 2>"$1/stderr"'
        ' </dev/null; echo $? >"$1/rc.tmp"; mv "$1/rc.tmp" "$1/rc"\' loom "$d"'
        " >/dev/null 2>&1 </dev/null &\n"
        'echo "loom-launched"\n'
    )


def _poll_script(ctl: str) -> str:
    return (
        f"d={ctl}\n"
        'if [ -f "$d/rc" ]; then echo "loom-rc $(cat "$d/rc")"; else echo loom-running; fi\n'
    )


def _stderr_script(ctl: str) -> str:
    return f"tail -c {_STDERR_TAIL} {ctl}/stderr 2>/dev/null || true\n"


def _collect_script(ctl: str) -> str:
    return f'd={ctl}\ncat "$d/stdout" 2>/dev/null || true\nrm -rf "$d"\n'


def _kill_script(ctl: str) -> str:
    return (
        f"d={ctl}\n"
        'if [ -f "$d/pid" ]; then\n'
        '  kill -KILL -- "-$(cat "$d/pid")" 2>/dev/null || true\n'
        "fi\n"
        'rm -rf "$d"\n'
    )


async def run_detached(
    ssh: SshExec,
    target: SshTarget,
    script: str,
    *,
    what: str,
    timeout_s: float,
    poll_s: float,
    check_host: Callable[[], Awaitable[None]],
    ctl_root: str = DEFAULT_CTL_ROOT,
    max_connection_failures: int = 5,
    call_timeout_s: float = 60.0,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Run `script` detached in the pod and return its stdout.

    Raises `RemoteCommandError` on a nonzero exit (with the stderr tail) or a
    timeout (after killing the script's process group); `check_host`'s exception
    (e.g. `HostLost`) wins over both when the pod itself is gone.
    """
    if not _CTL_RE.match(ctl_root):
        raise ValueError(f"invalid control root {ctl_root!r}")
    ctl = f"{ctl_root.rstrip('/')}/{uuid.uuid4().hex}"
    failures = 0

    async def call(body: str) -> ExecResult:
        nonlocal failures
        while True:
            res = await ssh.run(target, body, timeout_s=call_timeout_s)
            if res.returncode != SSH_UNREACHABLE:
                failures = 0
                return res
            failures += 1
            await check_host()
            if failures >= max_connection_failures:
                raise RemoteCommandError(f"{what}: ssh unreachable", None, "", res.stderr)
            await sleep(poll_s)

    launched = await call(_launch_script(script, ctl))
    if launched.returncode != 0 or "loom-launched" not in launched.stdout:
        await check_host()
        raise RemoteCommandError(f"{what}: launch", launched.returncode, "", launched.stderr)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        await sleep(poll_s)
        polled = await call(_poll_script(ctl))
        m = re.search(r"^loom-rc (-?\d+)$", polled.stdout, re.MULTILINE)
        if m is not None:
            rc = int(m.group(1))
            stderr = (await call(_stderr_script(ctl))).stdout
            stdout = (await call(_collect_script(ctl))).stdout
            if rc != 0:
                await check_host()
                raise RemoteCommandError(what, rc, stdout, stderr)
            return stdout
        await check_host()
        if loop.time() > deadline:
            stderr = (await call(_stderr_script(ctl))).stdout
            await call(_kill_script(ctl))
            raise RemoteCommandError(what, None, "", stderr)
