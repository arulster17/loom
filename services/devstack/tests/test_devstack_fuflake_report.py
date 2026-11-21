"""The PR job `flake-check`'s report step (plan R15, FU5): its real `run` text from ci.yml,
run with `bash -e` over JUnit files like the 10 `flake-session` jobs upload, each one
pytest session of pytest-repeat's `--count 2`."""

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
    sys.platform == "win32" or not shutil.which("bash") or not shutil.which("python3"),
    reason="needs bash and python3, as on the CI runner",
)

TESTS = ("test_a", "test_b", "test_c")


def report_step() -> str:
    """The `run` text of flake-check's report step, which must read the downloaded
    session files under `junit/`."""
    check: dict[str, Any] = yaml.safe_load(CI.read_text())["jobs"]["flake-check"]
    steps = [s for s in check.get("steps") or [] if "flaky_report.py" in str(s.get("run") or "")]
    assert len(steps) == 1, f"flake-check: expected one flaky_report.py step, got {len(steps)}"
    run = str(steps[0]["run"])
    assert "junit/flake-*.xml" in run, "flake-check's report step does not read junit/flake-*.xml"
    return run


def session_file(path: Path, fail: tuple[str, int] | None = None) -> None:
    """One session's JUnit file: each test twice (`[1-2]`, `[2-2]`); `fail` = (test, copy)."""
    cases = []
    for name in TESTS:
        for copy in (1, 2):
            body = '<failure message="boom">x</failure>' if fail == (name, copy) else ""
            cases.append(
                f'<testcase classname="services.x.tests.test_x" name="{name}[{copy}-2]"'
                f' time="0.01">{body}</testcase>'
            )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        + "".join(cases)
        + "</testsuite></testsuites>"
    )


def run_report(
    tmp_path: Path, sessions: int, result: str = "success", fail_in: int | None = None
) -> tuple[int, str, str]:
    """Run the step in tmp_path with `sessions` JUnit files; returns (rc, output, summary)."""
    script = report_step()
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "flaky_report.py", tmp_path / "scripts" / "flaky_report.py")
    junit = tmp_path / "junit"
    junit.mkdir()
    for i in range(1, sessions + 1):
        session_file(junit / f"flake-{i}.xml", ("test_b", 2) if i == fail_in else None)
    summary = tmp_path / "summary.md"
    env = {**os.environ, "GITHUB_STEP_SUMMARY": str(summary), "SESSIONS": result}
    proc = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    text = summary.read_text() if summary.exists() else ""
    return proc.returncode, proc.stdout + proc.stderr, text


def test_no_junit_file_fails(tmp_path: Path) -> None:
    rc, out, _ = run_report(tmp_path, 0)
    assert rc == 1, out
    assert "every session was killed or failed" in out


def test_all_ten_sessions_give_20_runs_per_test(tmp_path: Path) -> None:
    rc, out, summary = run_report(tmp_path, 10)
    assert rc == 0, out
    assert "10 of 10 sessions" in out
    assert "each test ran 20 to 20 times" in out
    assert "**0 flaky**" in out
    assert "::warning::" not in out
    assert "each test ran 20 to 20 times" in summary, "report not added to the step summary"


def test_a_missing_session_warns_but_passes(tmp_path: Path) -> None:
    rc, out, summary = run_report(tmp_path, 9)
    assert rc == 0, out
    assert "::warning::only 9 of 10" in out
    assert "each test ran 18 to 18 times" in out
    assert f"{len(TESTS)} tests ran fewer than 20 times" in out
    assert "ran fewer than 20 times" in summary


def test_a_copy_failing_in_one_session_is_flaky(tmp_path: Path) -> None:
    rc, out, summary = run_report(tmp_path, 10, fail_in=4)
    assert rc == 1, out
    assert "**1 flaky**" in out
    assert "`services.x.tests.test_x::test_b`: 19 passed, 1 failed" in out
    assert "**1 flaky**" in summary


def test_a_failed_session_job_fails_the_check(tmp_path: Path) -> None:
    rc, out, _ = run_report(tmp_path, 10, result="failure")
    assert rc == 1, out
    assert "a flake-session job failed" in out
    assert "**0 flaky**" in out, "the report still runs over the files the sessions wrote"
