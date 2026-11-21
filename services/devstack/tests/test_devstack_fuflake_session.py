"""The PR job `flake-session`'s pytest step (plan R15, FU5): its real `run` text from
ci.yml, run with `bash -e` against a fake pytest, for each exit status pytest can give.
A failing test or an empty selection does not fail the job (flake-check judges); a
session that hits its 7-minute limit only warns; any other status fails the job."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CI = ROOT / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("bash"),
    reason="needs bash, as on the CI runner",
)

# What the fake `.venv/bin/python` must be called with in session 3 (the design's
# command line), in this order.
EXPECTED_ARGS = [
    "-m",
    "pytest",
    "-q",
    "-n",
    "auto",
    "--count",
    "2",
    "--repeat-scope",
    "session",
    "--junitxml=flake-3.xml",
    "tests/test_x.py",
]


def session_steps() -> list[dict[str, Any]]:
    job: dict[str, Any] = yaml.safe_load(CI.read_text())["jobs"]["flake-session"]
    return list(job.get("steps") or [])


def session_step() -> str:
    """The `run` text of flake-session's step that calls pytest."""
    steps = [s for s in session_steps() if "pytest" in str(s.get("run") or "")]
    assert len(steps) == 1, f"flake-session: expected one pytest step, got {len(steps)}"
    return str(steps[0]["run"])


def write_exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def run_session(tmp_path: Path, fake_exit: int) -> tuple[int, str, str, list[str]]:
    """Run the step in tmp_path with a fake pytest exiting `fake_exit`; returns
    (rc, stdout, stdout + stderr, the fake's arguments)."""
    script = session_step()
    bin_dir = tmp_path / "bin"
    # No busy loops in tests: `seq 0` prints nothing, so the step starts none.
    write_exe(bin_dir / "nproc", "echo 0\n")
    args_file = tmp_path / "args.txt"
    write_exe(
        tmp_path / ".venv" / "bin" / "python",
        'printf \'%s\\n\' "$@" > "$ARGS_FILE"\nexit "$FAKE_EXIT"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "SESSION": "3",
        "FLAKE_FILES": "tests/test_x.py",
        "FAKE_EXIT": str(fake_exit),
        "ARGS_FILE": str(args_file),
    }
    proc = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    args = args_file.read_text().splitlines() if args_file.exists() else []
    return proc.returncode, proc.stdout, proc.stdout + proc.stderr, args


@pytest.mark.parametrize("fake_exit", [0, 1, 5])
def test_pass_fail_or_nothing_to_repeat_passes_quietly(tmp_path: Path, fake_exit: int) -> None:
    rc, _, out, _ = run_session(tmp_path, fake_exit)
    assert rc == 0, f"pytest exit {fake_exit} must not fail the session job:\n{out}"
    assert "::warning::" not in out


@pytest.mark.parametrize("fake_exit", [124, 137])
def test_time_limit_warns_but_passes(tmp_path: Path, fake_exit: int) -> None:
    rc, stdout, out, _ = run_session(tmp_path, fake_exit)
    assert rc == 0, f"a session at its limit (exit {fake_exit}) must only warn:\n{out}"
    assert "::warning::7-minute limit hit in session 3" in stdout, out


@pytest.mark.parametrize("fake_exit", [2, 3])
def test_other_status_fails_the_job(tmp_path: Path, fake_exit: int) -> None:
    rc, _, out, _ = run_session(tmp_path, fake_exit)
    assert rc == fake_exit, out
    assert f"pytest exited {fake_exit}" in out


def test_pytest_gets_the_session_command_line(tmp_path: Path) -> None:
    _, _, out, args = run_session(tmp_path, 0)
    assert args, f"the step did not run .venv/bin/python:\n{out}"
    assert args[:2] == ["-m", "pytest"], args
    assert args[-1] == "tests/test_x.py", args
    # Each expected word, in order (other flags may sit between them).
    rest = iter(args)
    missing = [word for word in EXPECTED_ARGS if word not in rest]
    assert not missing, f"missing or out of order: {missing}; got {args}"


def test_upload_tolerates_a_missing_file_and_reruns() -> None:
    upload = "actions/upload-artifact@"
    uploads = [s for s in session_steps() if str(s.get("uses", "")).startswith(upload)]
    assert len(uploads) == 1, f"flake-session: expected one upload step, got {len(uploads)}"
    with_ = uploads[0].get("with") or {}
    assert with_.get("if-no-files-found") == "ignore", with_
    assert with_.get("overwrite") is True, with_
