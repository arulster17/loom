import asyncio
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from loom_bench.providers.base import HostLost
from loom_bench.providers.runpod_ssh import (
    SSH_UNREACHABLE,
    ExecResult,
    RemoteCommandError,
    SshTarget,
    generate_keypair,
    run_detached,
    ssh_argv,
)

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not installed")


def stub_bin(tmp_path: Path) -> Path:
    """A PATH dir with a `setsid` (missing on macOS) that really starts a session."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    setsid = d / "setsid"
    setsid.write_text(
        "#!/bin/bash\n"
        'exec python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@"\n'
    )
    setsid.chmod(0o755)
    return d


class LocalExec:
    """Runs `bash -s` on this machine instead of over SSH. `drop` makes the next N
    calls fail like a broken connection (exit 255)."""

    def __init__(self, path_dir: Path) -> None:
        self.env = {**os.environ, "PATH": f"{path_dir}:{os.environ['PATH']}"}
        self.scripts: list[str] = []
        self.drop = 0

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        self.scripts.append(script)
        if self.drop:
            self.drop -= 1
            return ExecResult(SSH_UNREACHABLE, "", "Connection reset")
        assert BASH is not None
        proc = await asyncio.create_subprocess_exec(
            BASH,
            "-s",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        out, err = await asyncio.wait_for(proc.communicate(script.encode()), timeout_s)
        assert proc.returncode is not None
        return ExecResult(proc.returncode, out.decode(), err.decode())


def target(tmp_path: Path) -> SshTarget:
    return SshTarget(
        host="203.0.113.7",
        port=28813,
        key_path=tmp_path / "id_ed25519",
        known_hosts=tmp_path / "known_hosts",
    )


async def alive() -> None:
    return None


async def short_sleep(s: float) -> None:
    await asyncio.sleep(0.02)


def test_ssh_argv_is_key_only_without_forwarding(tmp_path: Path) -> None:
    t = SshTarget(
        host="203.0.113.7",
        port=28813,
        key_path=tmp_path / "k",
        known_hosts=tmp_path / "kh",
        control_dir=tmp_path / "ctl",
    )
    argv = ssh_argv(t)
    assert argv[:3] == ["ssh", "-F", "/dev/null"]
    assert argv[-2:] == ["root@203.0.113.7", "bash -s"]
    joined = " ".join(argv)
    for opt in [
        "-p 28813",
        f"-i {tmp_path / 'k'}",
        "BatchMode=yes",
        "IdentitiesOnly=yes",
        "ForwardAgent=no",
        "ClearAllForwardings=yes",
        "StrictHostKeyChecking=accept-new",
        f"UserKnownHostsFile={tmp_path / 'kh'}",
        "GlobalKnownHostsFile=/dev/null",
        "ConnectTimeout=10",
        "ControlMaster=auto",
        f"ControlPath={tmp_path / 'ctl'}/%C",
        "ControlPersist=60",
    ]:
        assert opt in joined
    assert "ControlMaster" not in " ".join(ssh_argv(target(tmp_path)))


@pytest.mark.parametrize(
    "kw",
    [{"host": "a b"}, {"host": "x;rm"}, {"port": 0}, {"port": 70000}, {"user": "Root$"}],
)
def test_target_validation(tmp_path: Path, kw: dict[str, object]) -> None:
    base: dict[str, object] = {
        "host": "203.0.113.7",
        "port": 22,
        "key_path": tmp_path / "k",
        "known_hosts": tmp_path / "kh",
    }
    with pytest.raises(ValueError):
        SshTarget(**{**base, **kw})  # type: ignore[arg-type]


@pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen not installed")
def test_generate_keypair(tmp_path: Path) -> None:
    key, pub = generate_keypair(tmp_path / "keys")
    assert pub.startswith("ssh-ed25519 ") and pub.endswith(" loom-bench")
    assert stat.S_IMODE((tmp_path / "keys").stat().st_mode) == 0o700
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        generate_keypair(tmp_path / "keys")


async def test_run_detached_returns_stdout_and_cleans_up(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    ctl = tmp_path / "ctl"
    out = await run_detached(
        ex,
        target(tmp_path),
        "echo 'loom-stage a 1.5'\necho \"url=$1\" >&2\necho done\n",
        what="test",
        timeout_s=10,
        poll_s=0,
        check_host=alive,
        ctl_root=str(ctl),
        sleep=short_sleep,
    )
    assert out == "loom-stage a 1.5\ndone\n"
    assert list(ctl.iterdir()) == []  # the control dir is gone
    # The script travels on stdin only: every call is a fixed `bash -s`.
    assert "echo done" in ex.scripts[0]


async def test_control_dir_is_private_while_running(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    ctl = tmp_path / "ctl"
    probe = 'ls -ld "$(dirname "$0")" 2>/dev/null; stat -c %a . 2>/dev/null; echo ok\n'
    script = f'd="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"\nls -l "$d/run.sh"\n{probe}'
    out = await run_detached(
        ex,
        target(tmp_path),
        script,
        what="t",
        timeout_s=10,
        poll_s=0,
        check_host=alive,
        ctl_root=str(ctl),
        sleep=short_sleep,
    )
    run_sh_line = out.splitlines()[0]
    assert run_sh_line.startswith("-rw-------")


async def test_nonzero_exit_raises_with_stderr_tail(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    with pytest.raises(RemoteCommandError) as e:
        await run_detached(
            ex,
            target(tmp_path),
            "echo partial\necho 'loom-error boom' >&2\nexit 3\n",
            what="start engine",
            timeout_s=10,
            poll_s=0,
            check_host=alive,
            ctl_root=str(tmp_path / "ctl"),
            sleep=short_sleep,
        )
    assert e.value.returncode == 3
    assert "loom-error boom" in str(e.value)
    assert e.value.stdout == "partial\n"
    assert list((tmp_path / "ctl").iterdir()) == []


async def test_timeout_kills_the_process_group(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    marker = tmp_path / "survived"
    script = f"(sleep 3; touch {marker}) &\nsleep 30\n"
    with pytest.raises(RemoteCommandError) as e:
        await run_detached(
            ex,
            target(tmp_path),
            script,
            what="job",
            timeout_s=0.5,
            poll_s=0,
            check_host=alive,
            ctl_root=str(tmp_path / "ctl"),
            sleep=short_sleep,
        )
    assert e.value.returncode is None
    assert "timed out" in str(e.value)
    await asyncio.sleep(3.5)
    assert not marker.exists()
    assert list((tmp_path / "ctl").iterdir()) == []


async def test_transient_connection_failures_are_retried(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    ex.drop = 2
    out = await run_detached(
        ex,
        target(tmp_path),
        "echo hi\n",
        what="t",
        timeout_s=10,
        poll_s=0,
        check_host=alive,
        ctl_root=str(tmp_path / "ctl"),
        sleep=short_sleep,
    )
    assert out == "hi\n"


async def test_persistent_connection_failure_raises(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    ex.drop = 100
    with pytest.raises(RemoteCommandError, match="ssh unreachable"):
        await run_detached(
            ex,
            target(tmp_path),
            "echo hi\n",
            what="t",
            timeout_s=10,
            poll_s=0,
            check_host=alive,
            ctl_root=str(tmp_path / "ctl"),
            max_connection_failures=3,
            sleep=short_sleep,
        )


async def test_host_lost_wins_over_command_failure(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    calls = 0

    async def gone_after_launch() -> None:
        nonlocal calls
        calls += 1
        raise HostLost(
            "pod-1",
            state="TERMINATED",
            reason_code=None,
            reason_message="pod is gone",
            detected_at=datetime.now(UTC),
            seconds_since_launch=1.0,
        )

    with pytest.raises(HostLost):
        await run_detached(
            ex,
            target(tmp_path),
            "sleep 5\n",
            what="t",
            timeout_s=10,
            poll_s=0,
            check_host=gone_after_launch,
            ctl_root=str(tmp_path / "ctl"),
            sleep=short_sleep,
        )
    with pytest.raises(HostLost):
        await run_detached(
            ex,
            target(tmp_path),
            "exit 4\n",
            what="t",
            timeout_s=10,
            poll_s=0,
            check_host=gone_after_launch,
            ctl_root=str(tmp_path / "ctl"),
            sleep=short_sleep,
        )


async def test_rejects_unsafe_input(tmp_path: Path) -> None:
    ex = LocalExec(stub_bin(tmp_path))
    with pytest.raises(ValueError, match="heredoc"):
        await run_detached(
            ex,
            target(tmp_path),
            "echo LOOM_RUN_EOF\n",
            what="t",
            timeout_s=1,
            poll_s=0,
            check_host=alive,
            ctl_root=str(tmp_path / "ctl"),
        )
    with pytest.raises(ValueError, match="control root"):
        await run_detached(
            ex,
            target(tmp_path),
            "true\n",
            what="t",
            timeout_s=1,
            poll_s=0,
            check_host=alive,
            ctl_root="/tmp/a b",
        )
    assert ex.scripts == []
