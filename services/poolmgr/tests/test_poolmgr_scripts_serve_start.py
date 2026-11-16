"""serve_start.sh, the serving pod's start command and lease watchdog
(docs/phase1-m2-spec.md §5 "Leases and the watchdog", §11 "Serve scripts", §13 item 18),
run under bash as RunPod runs it (`bash -c`) with stub binaries on PATH.

The stubs, as spec §11 says the script tests stub them:

- `date` answers from a fake clock: a file the test sets. The clock moves only when the
  test moves it.
- `sleep` logs its argument and returns after 50 ms, except `sleep infinity`, which
  blocks. So `LEASE_POLL_S` can be large while the tests take seconds.
- `python3` records each terminate call, the bench's `python3 -c <terminate code>
  "$GRAPHQL_URL"`, with the fake time; `$FAIL_TERMINATE` makes it fail.
- `sshd`, `ssh-keygen`, `apt-get`, and `mkdir`/`chmod`/`chown`/`install` on paths
  outside the test's directory stand in for what needs root (`/run/sshd`, host keys).
  `useradd` and friends log a call: there is no job user.

The script must therefore find `date`, `sleep`, `python3` and `sshd` through PATH.

The tests count the watchdog's `date` calls as its polls (`Pod.observe`): each time
the watchdog reads the clock it must also have re-read the lease file in that loop
iteration, so a renewal is seen within a poll or two (spec §5: `lease = max(...)`,
re-read every `LEASE_POLL_S`).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

BASH = shutil.which("bash")
pytestmark = [
    pytest.mark.skipif(BASH is None, reason="bash not installed"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups and stubs"),
]

T0 = 1_790_000_000
HOUR = 3600
DAY = 86_400
POLL = 7  # LEASE_POLL_S here; not 30, so a hard-coded interval shows
TERMINATE_RETRY_S = 30  # §11: retried every 30 s
POD_KEY = "rpa_PODSCOPEDFAKEKEYFAKEKEY"
POD_ID = "fakepod000001"
GRAPHQL_URL = "https://api.runpod.io/graphql"
MANAGER_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMANAGERFAKEKEYFAKEKEY loom-poolmgr"
ACCOUNT_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIACCOUNTFAKEKEYFAKEKEY account@runpod"
WAIT_S = 20.0


def q(x: object) -> str:
    return shlex.quote(str(x))


def real(name: str) -> str:
    path = shutil.which(name, path="/usr/bin:/bin")
    assert path is not None, name
    return path


def render(name: str, **values: Any) -> str:
    scripts = importlib.import_module("loom_poolmgr.scripts")
    out = scripts.render(name, **values)
    assert isinstance(out, str)
    return out


def stub(bin_dir: Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


class Pod:
    """One run of the rendered serve_start.sh against a fake clock."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.logs = tmp / "logs"
        self.bin.mkdir()
        self.logs.mkdir()
        self.clock = tmp / "clock"
        self.stop_flag = tmp / "stop"
        self.lease_file = tmp / "run" / "loom" / "lease"
        self.loom_root = tmp / "var" / "lib" / "loom"
        self.ssh_dir = tmp / "root" / ".ssh"
        self.values: dict[str, Any] = {
            "LEASE_FILE": str(self.lease_file),
            "LEASE_POLL_S": POLL,
            "STAGE_FILE": str(self.loom_root / "stages"),
            "LOOM_ROOT": str(self.loom_root),
            "SSH_DIR": str(self.ssh_dir),
            "SSHD_CONFIG": str(self.loom_root / "sshd_config"),
            "GRAPHQL_URL": GRAPHQL_URL,
        }
        self.proc: subprocess.Popen[bytes] | None = None
        self.set_clock(T0)
        self._stubs()

    # ------------------------------------------------------------------ stubs

    def _stubs(self) -> None:
        logs, clock, stop = q(self.logs), q(self.clock), q(self.stop_flag)
        die = f'[ -e {stop} ] && {{ kill -KILL "$PPID" 2>/dev/null; exit 1; }}\n'
        stub(
            self.bin,
            "date",
            die + f'echo "$*" >>{logs}/date\n'
            f'now="$(cat {clock})"\n'
            'for a in "$@"; do\n'
            '  case "$a" in\n'
            '    +%s) echo "$now"; exit 0 ;;\n'
            '    +%s.%N) echo "$now.000000000"; exit 0 ;;\n'
            "  esac\n"
            "done\n"
            f'exec {q(real("date"))} -d "@$now" "$@"\n',
        )
        stub(
            self.bin,
            "sleep",
            f'echo "sleep $*" >>{logs}/events\n' + die + 'case "$1" in infinity|inf) '
            f"exec {q(real('sleep'))} infinity ;; esac\n"
            f"exec {q(real('sleep'))} 0.05\n",
        )
        stub(
            self.bin,
            "python3",
            die + 'if [ "${1:-}" = -c ] && [[ "${2:-}" == *podTerminate* ]]; then\n'
            f'  printf "%s" "$2" >{logs}/terminate_code\n'
            f'  key=bad; [ "${{RUNPOD_API_KEY:-}}" = {q(POD_KEY)} ] && key=ok\n'
            f'  leak=no; case "$*" in *{q(POD_KEY)}*) leak=yes ;; esac\n'
            f'  echo "terminate at=$(cat {clock}) argc=$# url=${{3:-}} key=$key'
            f' pod=${{RUNPOD_POD_ID:-}} argv_leak=$leak" >>{logs}/events\n'
            f"  [ -e {q(self.tmp / 'FAIL_TERMINATE')} ] && exit 1\n"
            '  echo "terminate requested"; exit 0\n'
            "fi\n"
            f'echo "python3 $1" >>{logs}/python3_other\n',
        )
        stub(
            self.bin,
            "sshd",
            f'printf "%s\\n" "$@" >{logs}/sshd_args\n'
            f"[ -e {q(self.tmp / 'SSHD_FAILS')} ] && exit 1\n"
            f"[ -e {q(self.tmp / 'SSHD_HANGS')} ] && exec {q(real('sleep'))} 60\n"
            "exit 0\n",
        )
        for name in ("ssh-keygen", "apt-get", "apt"):
            stub(self.bin, name, f'echo "{name} $*" >>{logs}/pretend\nexit 0\n')
        stub(self.bin, "chown", f'echo "chown $*" >>{logs}/pretend\nexit 0\n')
        stub(self.bin, "setsid", 'exec "$@"\n')  # keep everything in our process group
        for name in ("useradd", "adduser", "groupadd", "usermod", "setpriv"):
            stub(self.bin, name, f'echo "{name} $*" >>{logs}/forbidden\nexit 0\n')
        # Root-only paths (outside this test's directory) are dropped and logged.
        for name in ("mkdir", "chmod", "install"):
            stub(
                self.bin,
                name,
                "args=(); n=0; skip=0\n"
                'for a in "$@"; do\n'
                '  if [ "$skip" = 1 ]; then skip=0; continue; fi\n'
                '  case "$a" in\n'
                f'    -o|-g) [ {name} = install ] && {{ skip=1; continue; }}; args+=("$a") ;;\n'
                f'    {q(self.tmp)}/*) args+=("$a"); n=$((n+1)) ;;\n'
                f'    /*) echo "{name} $a" >>{logs}/pretend ;;\n'
                '    *) args+=("$a") ;;\n'
                "  esac\n"
                "done\n"
                '[ "$n" -gt 0 ] || exit 0\n'
                f'exec {q(real(name))} "${{args[@]}}"\n',
            )

    # ------------------------------------------------------------------ control

    def set_clock(self, epoch: int) -> None:
        part = self.clock.with_name("clock.part")
        part.write_text(f"{epoch}\n")
        os.replace(part, self.clock)

    def write_lease(self, text: str) -> None:
        self.lease_file.parent.mkdir(parents=True, exist_ok=True)
        part = self.lease_file.with_name("lease.test-part")
        part.write_text(text)
        os.replace(part, self.lease_file)

    def flag(self, name: str) -> None:
        (self.tmp / name).touch()

    def start(
        self, lease_epoch: int | str | None, *, unset: tuple[str, ...] = (), **env: str
    ) -> None:
        assert BASH is not None
        script = render("serve_start", **self.values)
        full_env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.tmp / "root"),
            "LOOM_MANAGED": "true",
            "LOOM_ROLE": "serving",
            "LOOM_MODEL": "qwen3-8b",
            "LOOM_SSH_PUBKEY": MANAGER_KEY,
            "PUBLIC_KEY": ACCOUNT_KEY,  # RunPod's account key: never authorized
            "RUNPOD_API_KEY": POD_KEY,
            "RUNPOD_POD_ID": POD_ID,
            **env,
        }
        if lease_epoch is not None:
            full_env["LOOM_LEASE_EPOCH"] = str(lease_epoch)
        for name in unset:
            full_env.pop(name, None)
        with (self.logs / "stdout").open("wb") as out, (self.logs / "stderr").open("wb") as err:
            self.proc = subprocess.Popen(
                [BASH, "-c", script],
                env=full_env,
                cwd=self.tmp,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )

    def close(self) -> None:
        self.stop_flag.touch()
        if self.proc is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=10)

    # ------------------------------------------------------------------ observation

    def _lines(self, name: str) -> list[str]:
        path = self.logs / name
        return path.read_text().splitlines() if path.exists() else []

    def events(self) -> list[str]:
        return self._lines("events")

    def terminations(self) -> list[dict[str, str]]:
        out = []
        for line in self.events():
            if line.startswith("terminate "):
                out.append(dict(kv.split("=", 1) for kv in line.split()[1:]))
        return out

    def sleeps_before_first_terminate(self) -> list[str]:
        out = []
        for line in self.events():
            if line.startswith("terminate "):
                break
            if line.startswith("sleep "):
                out.append(line.removeprefix("sleep ").strip())
        return out

    def output(self) -> str:
        return (
            f"stdout:\n{(self.logs / 'stdout').read_text(errors='replace')[-2000:]}\n"
            f"stderr:\n{(self.logs / 'stderr').read_text(errors='replace')[-2000:]}\n"
            f"events:\n{chr(10).join(self.events()[-40:])}"
        )

    def assert_alive(self) -> None:
        assert self.proc is not None
        assert self.proc.poll() is None, f"the start command exited\n{self.output()}"

    def wait_for(self, pred: Callable[[], bool], what: str, timeout: float = WAIT_S) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return
            self.assert_alive()
            time.sleep(0.02)
        pytest.fail(f"timed out waiting for {what}\n{self.output()}")

    def wait_setup_done(self) -> None:
        """Setup has finished once the script reached `sleep infinity`."""
        self.wait_for(
            lambda: any(e in ("sleep infinity", "sleep inf") for e in self.events()),
            "setup to finish (sleep infinity)",
        )

    def observe(self, polls: int = 3) -> None:
        """Wait until the watchdog has read the clock `polls` more times."""
        seen = len(self._lines("date"))
        self.wait_for(
            lambda: len(self._lines("date")) >= seen + polls, f"{polls} more watchdog polls"
        )
        time.sleep(0.1)

    def at(self, epoch: int) -> None:
        """Move the clock to `epoch`; assert the watchdog saw it and did not terminate."""
        self.set_clock(epoch)
        self.observe()
        assert self.terminations() == [], f"terminated early at {epoch}\n{self.output()}"
        self.assert_alive()

    def expect_terminate_at(self, epoch: int) -> dict[str, str]:
        self.set_clock(epoch)
        self.wait_for(lambda: bool(self.terminations()), f"a terminate call at {epoch}")
        first = self.terminations()[0]
        assert int(first["at"]) >= epoch
        return first


@pytest.fixture
def pod(tmp_path: Path) -> Iterator[Pod]:
    p = Pod(tmp_path)
    try:
        yield p
    finally:
        p.close()


def assert_proper_terminate(call: dict[str, str]) -> None:
    # `python3 -c <code> "$GRAPHQL_URL"`, the pod-scoped key in its environment only.
    assert call["argc"] == "3", call
    assert call["url"] == GRAPHQL_URL
    assert call["key"] == "ok" and call["pod"] == POD_ID
    assert call["argv_leak"] == "no"


# ---------------------------------------------------------------------------- the lease


def test_terminates_at_the_env_epoch_without_a_lease_file(pod: Pod) -> None:
    lease = T0 + HOUR
    pod.start(lease)
    pod.wait_setup_done()
    assert not pod.lease_file.exists()
    pod.at(T0)
    pod.at(lease - POLL)
    pod.at(lease - 1)
    call = pod.expect_terminate_at(lease)
    assert_proper_terminate(call)
    # Terminating is a request to RunPod; the start command itself never exits.
    time.sleep(0.3)
    pod.assert_alive()


def test_a_later_lease_file_moves_the_deadline(pod: Pod) -> None:
    env_epoch, file_epoch = T0 + HOUR, T0 + 3 * HOUR
    pod.write_lease(f"{file_epoch}\n")
    pod.start(env_epoch)
    pod.wait_setup_done()
    pod.at(env_epoch)
    pod.at(env_epoch + 1)
    pod.at(file_epoch - 1)
    assert_proper_terminate(pod.expect_terminate_at(file_epoch))


def test_an_earlier_lease_file_never_shortens_the_env_epoch(pod: Pod) -> None:
    env_epoch, file_epoch = T0 + 3 * HOUR, T0 + HOUR
    pod.write_lease(f"{file_epoch}\n")
    pod.start(env_epoch)
    pod.wait_setup_done()
    pod.at(file_epoch)
    pod.at(file_epoch + 1)
    pod.at(env_epoch - 1)
    assert_proper_terminate(pod.expect_terminate_at(env_epoch))


@pytest.mark.parametrize(
    "contents",
    ["", "\n", "not-an-epoch\n", "12 34\n"],
    ids=["empty", "blank-line", "garbage", "two-numbers"],
)
def test_an_unreadable_lease_file_falls_back_to_the_env_epoch(pod: Pod, contents: str) -> None:
    # max(env, file) with no usable file is the env epoch: neither earlier, nor never.
    env_epoch = T0 + HOUR
    pod.write_lease(contents)
    pod.start(env_epoch)
    pod.wait_setup_done()
    pod.at(T0)
    pod.at(env_epoch - 1)
    assert_proper_terminate(pod.expect_terminate_at(env_epoch))


def test_the_lease_file_is_re_read_every_poll_and_there_is_no_absolute_ttl(pod: Pod) -> None:
    env_epoch = T0 + HOUR
    pod.start(env_epoch)
    pod.wait_setup_done()
    pod.at(T0 + 1800)

    def renew(epoch: int) -> None:
        # The manager's renewal, as it runs it: serve_lease.sh over SSH.
        # No directory made here: serve_lease creates /run/loom if it is missing.
        script = render("serve_lease", LEASE_FILE=str(pod.lease_file), LEASE_EPOCH=epoch)
        assert BASH is not None
        done = subprocess.run(
            [BASH, "-s"],
            input=script,
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert done.returncode == 0, done.stderr

    # Renewed while running: the new epoch is picked up (after a few polls, so the
    # watchdog has re-read the file), and the env epoch passes quietly.
    renew(T0 + 2 * HOUR)
    pod.observe()
    pod.at(env_epoch)
    pod.at(env_epoch + HOUR - 1)
    # Renewed out to 30 days: no TTL counted from the start ends it.
    end = T0 + 30 * DAY + HOUR
    renew(end)
    pod.observe()
    for day in (1, 2, 10, 30):
        pod.at(T0 + day * DAY)
    pod.at(end - 1)
    assert_proper_terminate(pod.expect_terminate_at(end))
    # Between reads the watchdog sleeps at most LEASE_POLL_S, never until the deadline.
    sleeps = [s for s in pod.sleeps_before_first_terminate() if s not in ("infinity", "inf")]
    assert sleeps, "the watchdog never slept between polls"
    for s in sleeps:
        assert float(s.removesuffix("s")) <= POLL, f"slept {s}s between lease reads"


@pytest.mark.parametrize("lease_epoch", [None, "", "soon", "1790003600x", "12 34"])
def test_a_missing_or_non_numeric_env_epoch_terminates_at_once(
    pod: Pod, lease_epoch: str | None
) -> None:
    # §5: fail closed. The clock never moves, so only this can end the pod.
    pod.start(lease_epoch, unset=("LOOM_LEASE_EPOCH",) if lease_epoch is None else ())
    pod.wait_for(lambda: bool(pod.terminations()), "a terminate call")
    assert_proper_terminate(pod.terminations()[0])
    assert int(pod.terminations()[0]["at"]) == T0
    time.sleep(0.3)
    pod.assert_alive()


def test_a_restart_after_the_lease_ended_terminates_at_once(pod: Pod) -> None:
    # A container restart re-runs the start command with the same absolute epoch.
    pod.set_clock(T0 + HOUR + 10)
    pod.start(T0 + HOUR)
    pod.wait_for(lambda: bool(pod.terminations()), "a terminate call")
    assert_proper_terminate(pod.terminations()[0])


def test_terminate_is_retried_while_it_fails(pod: Pod) -> None:
    pod.flag("FAIL_TERMINATE")
    pod.start(T0 + HOUR)
    pod.wait_setup_done()
    pod.expect_terminate_at(T0 + HOUR)
    pod.wait_for(lambda: len(pod.terminations()) >= 3, "terminate retries")
    events = pod.events()
    first = next(i for i, e in enumerate(events) if e.startswith("terminate "))
    retry_sleeps = []
    for e in events[first:]:
        if e.startswith("sleep "):
            with contextlib.suppress(ValueError):
                retry_sleeps.append(float(e.removeprefix("sleep ").strip().removesuffix("s")))
    assert TERMINATE_RETRY_S in retry_sleeps, f"retries are not 30 s apart: {retry_sleeps}"
    pod.assert_alive()


# ---------------------------------------------------------------------------- setup


def test_setup_failure_terminates_the_pod(pod: Pod) -> None:
    pod.flag("SSHD_FAILS")
    pod.start(T0 + HOUR)  # the lease is far off; the clock does not move
    pod.wait_for(lambda: bool(pod.terminations()), "a terminate call after setup failed")
    assert_proper_terminate(pod.terminations()[0])
    time.sleep(0.3)
    pod.assert_alive()


@pytest.mark.parametrize(
    "pubkey",
    [None, "", "not a key", "ssh-ed25519", "garbage; rm -rf /"],
    ids=["unset", "empty", "words", "type-only", "shell"],
)
def test_an_empty_or_invalid_ssh_pubkey_is_a_setup_failure(pod: Pod, pubkey: str | None) -> None:
    # §5: an empty or invalid LOOM_SSH_PUBKEY is a setup failure (terminate).
    if pubkey is None:
        pod.start(T0 + HOUR, unset=("LOOM_SSH_PUBKEY",))
    else:
        pod.start(T0 + HOUR, LOOM_SSH_PUBKEY=pubkey)
    pod.wait_for(lambda: bool(pod.terminations()), "a terminate call after setup failed")
    assert_proper_terminate(pod.terminations()[0])
    assert int(pod.terminations()[0]["at"]) == T0  # the lease is far off
    time.sleep(0.3)
    pod.assert_alive()


def test_the_watchdog_runs_while_setup_hangs(pod: Pod) -> None:
    pod.flag("SSHD_HANGS")
    pod.start(T0 + HOUR)
    pod.wait_for(lambda: (pod.logs / "sshd_args").exists(), "sshd to be started")
    assert pod.terminations() == []
    assert_proper_terminate(pod.expect_terminate_at(T0 + HOUR))


def sshd_options(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s*=\s*|\s+", line, maxsplit=1)
        value = parts[1].strip() if len(parts) > 1 else ""
        out.setdefault(parts[0].lower(), value)  # sshd: the first value wins
    return out


def test_sshd_is_key_only_root_only_without_forwarding(pod: Pod) -> None:
    pod.start(T0 + HOUR)
    pod.wait_setup_done()
    args = (pod.logs / "sshd_args").read_text().splitlines()
    assert pod.values["SSHD_CONFIG"] in args
    assert args[args.index(pod.values["SSHD_CONFIG"]) - 1] == "-f"
    opts = sshd_options(Path(pod.values["SSHD_CONFIG"]).read_text())
    assert opts.get("passwordauthentication") == "no"
    assert "no" in (
        opts.get("kbdinteractiveauthentication"),
        opts.get("challengeresponseauthentication"),
    )
    assert opts.get("pubkeyauthentication", "yes") == "yes"
    assert opts.get("permitrootlogin") in ("prohibit-password", "without-password")
    assert opts.get("allowusers") == "root"
    assert opts.get("permituserenvironment", "no") == "no"
    if opts.get("disableforwarding") != "yes":
        for key in ("allowtcpforwarding", "allowagentforwarding", "allowstreamlocalforwarding"):
            assert opts.get(key) == "no", key  # sshd's default for each is yes
        for key in ("x11forwarding", "permittunnel"):
            assert opts.get(key, "no") == "no", key
    assert pod.terminations() == []


def test_only_the_managers_key_is_authorized(pod: Pod) -> None:
    pod.start(T0 + HOUR)
    pod.wait_setup_done()
    opts = sshd_options(Path(pod.values["SSHD_CONFIG"]).read_text())
    # One keys file and no command that could authorize other keys.
    assert opts.get("authorizedkeyscommand", "none") == "none"
    entries = opts.get("authorizedkeysfile", ".ssh/authorized_keys").split()
    assert len(entries) == 1, entries
    keys_file = Path(entries[0])
    assert keys_file.is_absolute() and keys_file.parent == pod.ssh_dir
    keys = [k for k in keys_file.read_text().splitlines() if k.strip()]
    assert keys == [MANAGER_KEY]


def test_setup_records_the_sshd_ready_stage(pod: Pod) -> None:
    pod.start(T0 + HOUR)
    pod.wait_setup_done()
    stages = Path(pod.values["STAGE_FILE"]).read_text()
    assert re.search(r"^sshd_ready \d", stages, re.MULTILINE), stages


def test_no_job_user_and_the_pod_key_stays_out_of_files_and_output(pod: Pod) -> None:
    pod.start(T0 + HOUR)
    pod.wait_setup_done()
    pod.expect_terminate_at(T0 + HOUR)
    assert not (pod.logs / "forbidden").exists(), (pod.logs / "forbidden").read_text()
    for root in (pod.loom_root, pod.ssh_dir.parent, pod.lease_file.parent.parent):
        for f in root.rglob("*") if root.exists() else []:
            if f.is_file():
                assert POD_KEY not in f.read_text(errors="replace"), f
    for name in ("stdout", "stderr"):
        assert POD_KEY not in (pod.logs / name).read_text(errors="replace")


# ---------------------------------------------------------------------------- terminate call


def test_the_terminate_code_terminates_with_the_pod_key(pod: Pod) -> None:
    pod.start(T0 + HOUR)
    pod.wait_setup_done()
    pod.expect_terminate_at(T0 + HOUR)
    code = (pod.logs / "terminate_code").read_text()
    seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            seen.append(
                {
                    "auth": self.headers["Authorization"],
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            payload = json.dumps({"data": {"podTerminate": None}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a: Any) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/graphql"
        env = {"PATH": "/usr/bin:/bin", "RUNPOD_API_KEY": POD_KEY, "RUNPOD_POD_ID": POD_ID}
        done = subprocess.run(
            [sys.executable, "-c", code, url], env=env, capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 0, done.stderr
        (req,) = seen
        assert req["auth"] == f"Bearer {POD_KEY}"
        assert "podTerminate" in req["body"]["query"]
        assert "podStop" not in req["body"]["query"]
        assert req["body"]["variables"] == {"podId": POD_ID}
        missing = subprocess.run(
            [sys.executable, "-c", code, url],
            env={k: v for k, v in env.items() if k != "RUNPOD_API_KEY"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert missing.returncode != 0
        assert len(seen) == 1
    finally:
        server.shutdown()
