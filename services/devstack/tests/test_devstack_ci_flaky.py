"""Flake detection in CI (plan R15): `scripts/flaky_report.py`, the PR job `flake-check`,
the weekly `flake-hunt` workflow, and the acceptance job's shorter idle stop."""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"


def load(name: str) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load((WORKFLOWS / name).read_text())
    return doc


def runs(job: dict[str, Any]) -> str:
    return "\n".join(str(s.get("run") or "") for s in job.get("steps") or [])


@pytest.fixture(scope="module")
def report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("flaky_report", ROOT / "scripts/flaky_report.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclass looks its module up there
    spec.loader.exec_module(mod)
    return mod


def junit(path: Path, cases: list[tuple[str, str]]) -> Path:
    body = {"pass": "", "fail": '<failure message="boom">x</failure>', "skip": "<skipped/>"}
    xml = "".join(
        f'<testcase classname="m" name="{n}" time="0">{body[o]}</testcase>' for n, o in cases
    )
    path.write_text(f'<testsuites><testsuite name="pytest">{xml}</testsuite></testsuites>')
    return path


def test_strip_repeat_merges_only_pytest_repeat_copies(report: ModuleType) -> None:
    assert report.strip_repeat("t[3-20]", 20) == "t"
    assert report.strip_repeat("t[a-b-3-20]", 20) == "t[a-b]"
    assert report.strip_repeat("t[3-20]", None) == "t[3-20]"
    assert report.strip_repeat("t[1-2]", 20) == "t[1-2]"
    # xdist --dist loadgroup appends the xdist_group name
    assert report.strip_repeat("t[3-20]@quality-e2e", 20) == "t"
    assert report.strip_repeat("t[a-3-20]@g", 20) == "t[a]"
    assert report.strip_repeat("t[x]@g", None) == "t[x]"


def test_mixed_results_are_flaky_and_exit_1(
    report: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    f = junit(
        tmp_path / "a.xml",
        [
            ("t[1-3]", "pass"),
            ("t[2-3]", "fail"),
            ("t[3-3]", "skip"),
            ("u[1-3]", "fail"),
            ("v[1-3]", "pass"),
            ("", "pass"),
        ],
    )
    assert report.main(["--repeat-count", "3", "--expect-runs", "3", str(f)]) == 1
    out = capsys.readouterr().out
    assert "`m::t`: 1 passed, 1 failed" in out
    assert "**1 flaky**" in out and "1 failed every run" in out
    assert "ran fewer than 3 times" in out


def test_across_runs_without_repeat(report: ModuleType, tmp_path: Path) -> None:
    a = junit(tmp_path / "a.xml", [("t[x]", "pass"), ("u", "pass")])
    b = junit(tmp_path / "b.xml", [("t[x]", "fail"), ("u", "pass")])
    assert report.main([str(a), str(b)]) == 1
    assert report.main([str(a), str(a)]) == 0
    assert report.main([str(a), str(tmp_path / "missing.xml")]) == 2


def uses_step(job: dict[str, Any], action: str) -> dict[str, Any]:
    """The one step of `job` that uses `action` (e.g. `actions/upload-artifact`)."""
    found = [s for s in job.get("steps") or [] if str(s.get("uses", "")).startswith(action + "@")]
    assert len(found) == 1, f"expected one {action} step, found {len(found)}"
    return found[0]


def test_flake_files_lists_changed_tests_on_prs() -> None:
    files = load("ci.yml")["jobs"]["flake-files"]
    assert files["if"] == "github.event_name == 'pull_request'"
    assert "git diff" in runs(files) and "HEAD^1 HEAD" in runs(files)


def test_flake_sessions_run_one_session_per_job() -> None:
    """20 runs per test: 10 parallel jobs (each re-runs module fixtures on a fresh
    Postgres and Redis) x pytest-repeat's 2 copies, one pytest session per job."""
    jobs = load("ci.yml")["jobs"]
    assert "flake-session" in jobs, "no `flake-session` job in ci.yml"
    session = jobs["flake-session"]
    assert session["needs"] == "flake-files"
    assert session["if"] == "needs.flake-files.outputs.files != ''"
    assert {"postgres", "redis"} <= set(session["services"])
    assert session["strategy"]["matrix"]["session"] == list(range(1, 11))
    assert session["strategy"]["fail-fast"] is False
    assert session["env"]["SESSION"] == "${{ matrix.session }}"
    assert session["timeout-minutes"] <= 12
    cmd = runs(session)
    assert "for i in" not in cmd, "a session job runs one pytest session, not a loop"
    assert '--count 2 --repeat-scope session"' in cmd
    assert "-n auto $repeat" in cmd and "$FLAKE_FILES" in cmd
    assert '--junitxml="flake-$SESSION.xml"' in cmd
    assert "timeout -k 60 -s INT 420 " in cmd
    assert "nproc" in cmd, "no deliberate CPU load"
    upload = uses_step(session, "actions/upload-artifact")
    assert upload["if"] == "${{ !cancelled() }}"
    assert upload["with"]["name"] == "flake-session-${{ matrix.session }}"
    assert upload["with"]["path"] == "flake-${{ matrix.session }}.xml"


def test_flake_check_reports_over_all_sessions() -> None:
    check = load("ci.yml")["jobs"]["flake-check"]
    assert check["needs"] == ["flake-files", "flake-session"]
    assert check["if"] == "${{ !cancelled() && needs.flake-files.outputs.files != '' }}"
    assert "services" not in check, "the report job runs no tests: no Postgres or Redis"
    assert check["timeout-minutes"] <= 5
    download = uses_step(check, "actions/download-artifact")
    assert download["with"]["pattern"] == "flake-session-*"
    assert download["with"]["merge-multiple"] is True
    assert download["with"]["path"] == "junit"
    rep = runs(check)
    assert 'args="--repeat-count 2 --expect-runs 20"' in rep
    assert "flaky_report.py $args" in rep and "junit/flake-*.xml" in rep
    assert '[ "$SESSIONS" != success ]' in rep
    reporting = [s for s in check["steps"] if "flaky_report.py" in str(s.get("run") or "")]
    assert len(reporting) == 1
    assert (reporting[0].get("env") or {}).get("SESSIONS") == "${{ needs.flake-session.result }}"


def test_flake_jobs_pin_actions_by_commit() -> None:
    jobs = load("ci.yml")["jobs"]
    assert "flake-session" in jobs, "no `flake-session` job in ci.yml"
    for name in ("flake-session", "flake-check"):
        used = [str(s["uses"]) for s in jobs[name]["steps"] if "uses" in s]
        assert used, f"{name} uses no actions"
        for action in used:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), (
                f"{name}: {action} is not pinned by commit"
            )


def test_pytest_randomly_only_in_the_flake_hunt() -> None:
    """Installed, pytest-randomly shuffles every run; the hunt adds it with `--with`."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    names = {p["name"] for p in lock["package"]}
    assert "pytest-randomly" not in names
    assert "pytest-repeat" in names
    hunt = load("flake-hunt.yml")
    assert "schedule" in hunt[True] and "workflow_dispatch" in hunt[True]  # yaml: on -> True
    suite = hunt["jobs"]["suite"]
    assert suite["strategy"]["matrix"]["run"] == [1, 2, 3]
    assert "uv run --frozen --with pytest-randomly==" in runs(suite)
    assert "randomly" not in (WORKFLOWS / "ci.yml").read_text()
    report_job = hunt["jobs"]["report"]
    assert report_job["permissions"]["issues"] == "write"
    assert "gh issue create" in runs(report_job) and "gh issue edit" in runs(report_job)


def test_acceptance_idle_stop_outlasts_m1s_longest_request_gap() -> None:
    """M1's longest stretch without a request to a ready model is the wait for a killed
    pod's replacement, 183-196 s in CI; M2's idle_stop check accepts at most 900 s."""
    env = load("ci.yml")["jobs"]["acceptance"]["env"]
    assert 240 < int(env["LOOM_POOLMGR_IDLE_STOP_S"]) <= 900
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "LOOM_POOLMGR_IDLE_STOP_S: ${LOOM_POOLMGR_IDLE_STOP_S:-600}" in compose
