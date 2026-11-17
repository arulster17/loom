"""CI wiring for M2 (docs/phase1-m2-spec.md §12 `.github/workflows/ci.yml`, §9 "Bundle"
and "Local backend", §13 item 26, static part).

- mypy runs on `services/poolmgr/src` (CLAUDE.md's list "plus the new mypy path");
- `loom-podproxy-bundle --check` runs "as its own job with network access (not in the
  main test job)" (§9), so a commit whose proxy bundle no longer matches the committed
  `config/serving-artifacts.lock` fails CI. The main test job is the one running
  `pytest -q -n auto`.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]


def workflow() -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    return doc


def run_steps() -> list[tuple[str, str]]:
    """(job, command line) for every line of every `run:` step."""
    out: list[tuple[str, str]] = []
    for job_name, job in (workflow().get("jobs") or {}).items():
        for step in job.get("steps") or []:
            for line in str(step.get("run") or "").splitlines():
                if line.strip():
                    out.append((job_name, line.strip()))
    return out


def main_test_jobs() -> set[str]:
    return {
        job
        for job, cmd in run_steps()
        if "pytest" in shlex.split(cmd) and "-n" in shlex.split(cmd) and "auto" in cmd
    }


def test_ci_runs_mypy_on_the_pool_manager() -> None:
    lines = [cmd for _, cmd in run_steps() if "mypy" in shlex.split(cmd)]
    assert any("services/poolmgr/src" in shlex.split(cmd) for cmd in lines), lines


def test_ci_checks_the_proxy_bundle_against_its_lock_in_its_own_job() -> None:
    found = [
        (job, cmd)
        for job, cmd in run_steps()
        if "loom-podproxy-bundle" in cmd and "--check" in shlex.split(cmd)
    ]
    assert found, "no CI step runs `loom-podproxy-bundle --check`"
    main = main_test_jobs()
    assert main, "no CI job runs `pytest -q -n auto`"
    for job, cmd in found:
        assert "--out" in shlex.split(cmd), f"{cmd}: --out is required"
        assert job not in main, f"the bundle check runs in the main test job {job!r}"
