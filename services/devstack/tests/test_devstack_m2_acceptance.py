"""The M2 acceptance driver, `loom-m2-acceptance` (docs/phase1-m2-spec.md §2 item 8, §9,
§13 item 25).

Expected CLI (defined here; minimal, as `loom-m1-acceptance`):

    loom-m2-acceptance [--json PATH]        # exit 0 = every check passed, else non-zero

It drives the running compose stack (manager on the local backend) from the outside, and
with `--json PATH` writes a report shaped like M1's:

    {"items": [{"item": "<id>", "title": "...", "status": "PASS" | "FAIL",
                "numbers": {...}, "notes": [...]}, ...]}

with one item per §13 item 25 check, under these ids:

    cold_wake_ready   a cold model: 503 + Retry-After, wake, warming, then ready
    idle_stop         a ready model idle for its idle_stop_s (compose:
                      LOOM_POOLMGR_IDLE_STOP_S, 600 s) is drained to cold
    pod_kill_replaced a pod killed mid-stream: upstream_lost, then a new pod is ready
    no_capacity       `loom-poolmgr local-capacity off`: 503 "no GPU capacity"; `on` after
    spend_cap         a synthetic row over the monthly cap: the create is refused (pool
                      unavailable, spend_cap); the row and its alerts removed, then reset

Ports and the compose project come from the same variables as `loom-m1-acceptance`
(`LOOM_GATEWAY_PORT`, ..., `COMPOSE_PROJECT_NAME`, `LOOM_ACCEPT_*`).

The stack run is marked `acceptance` (deselected by default) and skips when the gateway
is not reachable:

    LOOM_ADMIN_EMAILS=admin@m1.test docker compose up -d --build --wait
    uv run pytest -m acceptance -s services/devstack/tests/test_devstack_m2_acceptance.py
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path

import httpx
import pytest

from loom_devstack import acceptance as m1

SCRIPT = "loom-m2-acceptance"
CHECKS = {
    "cold_wake_ready",
    "idle_stop",
    "pod_kill_replaced",
    "no_capacity",
    "spend_cap",
}


def entry_point() -> EntryPoint:
    found = [ep for ep in entry_points(group="console_scripts") if ep.name == SCRIPT]
    assert found, f"no `{SCRIPT}` console script is installed (project.scripts)"
    return found[0]


def test_the_cli_is_installed_and_takes_a_json_report_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main = entry_point().load()
    monkeypatch.setattr(sys, "argv", [SCRIPT, "--help"])  # as the console script calls it
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit) as exit_:
        main()
    assert exit_.value.code in (0, None)
    assert "--json" in out.getvalue()


@pytest.mark.acceptance
@pytest.mark.timeout(1800)
def test_m2_acceptance_on_the_compose_stack(tmp_path: Path) -> None:
    stack = m1.stack_from_env()
    url = f"{stack.gateway_root}/health"
    try:
        httpx.get(url, timeout=3, trust_env=False)
    except httpx.HTTPError as e:
        pytest.skip(f"compose stack not reachable ({url}): {e}")
    exe = shutil.which(SCRIPT) or shutil.which(SCRIPT, path=str(Path(sys.executable).parent))
    assert exe, f"`{SCRIPT}` is not on PATH (uv sync installs it with loom-devstack)"
    report_path = tmp_path / "m2-acceptance.json"
    proc = subprocess.run(
        [exe, "--json", str(report_path)],
        capture_output=True,
        text=True,
        timeout=1700,
    )
    tail = (proc.stdout + proc.stderr)[-4000:]
    assert report_path.is_file(), f"no JSON report written (exit {proc.returncode}):\n{tail}"
    report = json.loads(report_path.read_text())
    items = {i["item"]: i for i in report["items"]}
    missing = CHECKS - set(items)
    assert not missing, f"the report lacks {sorted(missing)}: {sorted(items)}"
    failed = {k: items[k].get("notes") for k in CHECKS if str(items[k]["status"]).upper() != "PASS"}
    assert not failed, failed
    assert proc.returncode == 0, tail
