"""The serve scripts' rendering seam (docs/phase1-m2-spec.md §11 "Serve scripts"):
`loom_poolmgr.scripts.render(name, **values)` and `required_vars(name)`, with the bench's
`render_script` contract, the required variables of the §11 table, valid bash, and no
secret ever passed as a variable (§5 "Bootstrap": secrets arrive only on SSH stdin).

`loom_poolmgr.scripts` is imported inside each test so that, until the package exists,
every test fails on its own with ModuleNotFoundError rather than one collection error.
"""

from __future__ import annotations

import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

BASH = shutil.which("bash")
pytestmark = [
    pytest.mark.skipif(BASH is None, reason="bash not installed"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX scripts and stubs"),
]

T0 = 1_790_000_000

# §11 table. serve_engine: the bench's start_engine requires line minus JOB_UID and
# JOB_HOME, plus ENGINE_UID and PROXY_DIR.
VALUES: dict[str, dict[str, Any]] = {
    "serve_start": {
        "LEASE_FILE": "/run/loom/lease",
        "LEASE_POLL_S": 30,
        "STAGE_FILE": "/var/lib/loom/stages",
        "LOOM_ROOT": "/var/lib/loom",
        "SSH_DIR": "/root/.ssh",
        "SSHD_CONFIG": "/var/lib/loom/sshd_config",
        "GRAPHQL_URL": "https://api.runpod.io/graphql",
    },
    "serve_lease": {
        "LEASE_FILE": "/run/loom/lease",
        "LEASE_EPOCH": T0 + 3600,
    },
    "serve_bootstrap": {
        "PROXY_DIR": "/etc/loom/proxy",
        "PROXY_VENV": "/opt/loom/proxy-venv",
        "PROXY_PORT": 8443,
        "UPSTREAM": "127.0.0.1:8000",
        "BUNDLE_SHA256": "a" * 64,
        "PROXY_UID": 10002,
        "ENGINE_UID": 10003,
    },
    "serve_engine": {
        "WARM": 0,
        "FETCH_WEIGHTS": ["Qwen/Qwen3-8B", "b" * 40],
        "CACHED_WEIGHTS": [],
        "WEIGHTS_DIR": "/opt/loom/hf",
        "PORT": 8000,
        "SERVED_MODEL": "qwen3-8b",
        "READY_TIMEOUT_S": 1800,
        "ENGINE_STALL_S": 600,
        "LOG_DIR": "/var/log/loom",
        "STAGE_FILE": "/var/lib/loom/stages",
        "ENGINE_ENV": ["VLLM_LOGGING_LEVEL=INFO"],
        "ENGINE_CMD": ["vllm", "serve", "Qwen/Qwen3-8B", "--host", "127.0.0.1"],
        "ENGINE_PIDFILE": "/var/lib/loom/engine.pid",
        "PROC1_ENVIRON": "/proc/1/environ",
        "SCAN_DIRS": ["/etc", "/opt/loom", "/var/lib/loom", "/tmp", "/root"],
        "ENGINE_UID": 10003,
        "PROXY_DIR": "/etc/loom/proxy",
    },
}
NAMES = sorted(VALUES)

# A variable with one of these in its name would carry a secret on the command line.
SECRET_NAME_RE = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE|CERT|(^|_)KEY($|_)")


def scripts() -> ModuleType:
    return importlib.import_module("loom_poolmgr.scripts")


def render(name: str, **over: Any) -> str:
    out = scripts().render(name, **{**VALUES[name], **over})
    assert isinstance(out, str)
    return out


def run_bash(script: str, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, "-s"],
        input=script,
        env={"PATH": "/usr/bin:/bin", "HOME": str(cwd)},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------- contract


@pytest.mark.parametrize("name", NAMES)
def test_required_vars_are_the_section_11_table(name: str) -> None:
    assert set(scripts().required_vars(name)) == set(VALUES[name])


@pytest.mark.parametrize("name", NAMES)
def test_render_takes_exactly_the_required_variables(name: str) -> None:
    values = VALUES[name]
    missing = sorted(values)[0]
    with pytest.raises(ValueError):
        scripts().render(name, **{k: v for k, v in values.items() if k != missing})
    with pytest.raises(ValueError):
        scripts().render(name, **values, UNEXPECTED_VAR="x")


@pytest.mark.parametrize("name", NAMES)
def test_rendered_script_is_valid_bash_with_the_common_helpers(name: str) -> None:
    script = render(name)
    assert script.startswith("#!/bin/bash\n")
    header = script.split("\n\n", 1)[0]
    assert "set -euo pipefail" in header
    # Every value is assigned in the header, quoted.
    for var in VALUES[name]:
        assert re.search(rf"^{var}=", header, re.MULTILINE), var
    # The package's own common.sh (stage, sysinfo, fail) is prepended.
    for helper in ("stage", "sysinfo", "fail"):
        assert re.search(rf"^(function\s+)?{helper}\s*(\(\))?\s*\{{", script, re.MULTILINE), helper
    # Never trace: these scripts run where secrets live.
    assert not re.search(r"^\s*set -[a-z]*x|xtrace", script, re.MULTILINE)
    assert BASH is not None
    subprocess.run([BASH, "-n"], input=script, text=True, check=True, timeout=30)


def test_values_are_quoted_not_interpreted(tmp_path: Path) -> None:
    hostile = tmp_path / "a b; touch pwned1; $(touch pwned2) `touch pwned3` 'q' \"d\""
    hostile.mkdir()
    lease = hostile / "lease"
    proc = run_bash(render("serve_lease", LEASE_FILE=str(lease), LEASE_EPOCH=T0), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert lease.read_text().strip() == str(T0)
    assert not [p for p in tmp_path.rglob("pwned*")]


# ---------------------------------------------------------------------------- secrets


@pytest.mark.parametrize("name", NAMES)
def test_no_secret_is_a_variable_or_a_placeholder(name: str) -> None:
    for var in scripts().required_vars(name):
        assert not SECRET_NAME_RE.search(var), f"{name} takes a secret-like variable {var}"
    script = render(name)
    assert "RUNPOD_SECRET" not in script  # no RunPod secret placeholders
    assert "AWS_ACCESS_KEY" not in script and "AWS_SECRET" not in script
    # The pod-scoped key is read by the terminate code from its own environment, never
    # expanded by the shell (which would put it on a command line).
    assert not re.search(r"\$\{?RUNPOD_API_KEY", script), name
    assert "PRIVATE KEY-----" not in script


# ---------------------------------------------------------------------------- serve_start


def test_serve_start_has_no_absolute_ttl_and_no_job_user() -> None:
    required = set(scripts().required_vars("serve_start"))
    for gone in ("TTL_EPOCH", "JOB_UID", "JOB_USER", "JOB_HOME", "AUTHORIZE_ACCOUNT_KEY"):
        assert gone not in required
    script = render("serve_start")
    # The template's own body (common.sh, prepended, may define generic helpers).
    body = script[script.index("# requires:") :]
    assert "useradd" not in body and "setpriv" not in body


def test_serve_start_terminates_and_never_stops_the_pod() -> None:
    script = render("serve_start")
    assert "podTerminate" in script
    assert "podStop" not in script  # a stopped pod keeps billing for its disk (§5)
    assert "/stop" not in script
