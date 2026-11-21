"""`loom-deploy`: standard library only, and the deploy order of P4 (M3 spec §6.6, A12).

`loom_devstack.deploy` is one module that runs on the box with the host's Python 3.12, so
it imports nothing outside the standard library (checked from its source and by importing
it with no site-packages at all).

The step list is the pure function `plan_steps(current, target, only)` (§6.6). The spec
fixes its name and arguments; the shapes below are this test's reading (defined here):

    current  the running release's manifest (a dict, as `releases/<sha>.json`, §6.1), or
             None before the first deploy
    target   the manifest of the release to apply (§6.12): {"sha": ..., "images":
             {<service>: "<repository>@sha256:<digest>", ...}, "migrations": n,
             "compose": {"path", "sha256"}, "bundle": {"path", "sha256"}}
    only     None (every service) or a collection of service names (`apply --only`)
    returns  a list of steps, each a tuple (action, *args); a service update (`up -d
             --no-deps <svc>`, then wait healthy) is ("up", "<svc>"), the migration
             ("migrate",)
    refusal  P4: `loom_devstack.deploy.Refused` (§6.12) whose message names the gateway

Order (§6.6 step 4): `redis` only if its image changed, then billing, gateway, poolmgr,
web, caddy, prometheus, grafana; the other services are updated on every apply, because
an apply of the same release re-renders their settings (§6.11 "Changing the name or the
domain"). Migrations run before any service is updated (§6.4).
"""

from __future__ import annotations

import ast
import copy
import importlib
import itertools
import os
import subprocess
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "services" / "devstack" / "src"
MODULE = SRC / "loom_devstack" / "deploy.py"

ORDER = ["redis", "billing", "gateway", "poolmgr", "web", "caddy", "prometheus", "grafana"]
ALWAYS = [s for s in ORDER if s != "redis"]


def deploy() -> Any:
    return importlib.import_module("loom_devstack.deploy")


def digest(release: str, service: str) -> str:
    import hashlib

    return hashlib.sha256(f"{release}/{service}".encode()).hexdigest()


def manifest(sha: str, images: dict[str, str] | None = None) -> dict[str, Any]:
    imgs = images or {
        s: f"123456789012.dkr.ecr.us-east-1.amazonaws.com/loom/{s}@sha256:{digest(sha, s)}"
        for s in [*ORDER, "migrate"]
    }
    return {
        "sha": sha,
        "images": imgs,
        "migrations": 12,
        "compose": {"path": f"releases/{sha}/compose.prod.yml", "sha256": digest(sha, "compose")},
        "bundle": {"path": f"releases/{sha}/deploy.tar.gz", "sha256": digest(sha, "bundle")},
    }


A = manifest("a" * 40)
B = manifest("b" * 40)


def ups(steps: Any) -> list[str]:
    steps = list(steps)
    for s in steps:
        assert isinstance(s, tuple) and s and isinstance(s[0], str), (
            f"step {s!r} is not (action, *args)"
        )
    return [s[1] for s in steps if s[0] == "up"]


def assert_migrate_first(steps: list[tuple[Any, ...]]) -> None:
    actions = [s[0] for s in steps]
    assert "migrate" in actions, f"no migrate step: {actions}"
    first_up = actions.index("up") if "up" in actions else len(actions)
    assert actions.index("migrate") < first_up, f"migrate after an update: {actions}"


def assert_refused(call: Any) -> None:
    refused = deploy().Refused
    with pytest.raises(Exception) as caught:
        call()
    err = caught.value
    assert isinstance(err, refused), f"not a deploy.Refused: {err!r}"
    assert "gateway" in str(err).lower(), f"the refusal does not name the gateway: {err}"


# ---------------------------------------------------------------- standard library only


def test_the_module_imports_only_the_standard_library() -> None:
    assert MODULE.is_file(), f"{MODULE.relative_to(ROOT)} does not exist"
    tree = ast.parse(MODULE.read_text(), filename=str(MODULE))
    outside = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                outside.append(f"relative import from {'.' * node.level}{node.module or ''}")
                continue
            names = [node.module or ""]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top != "__future__" and top not in sys.stdlib_module_names:
                outside.append(name)
    assert not outside, f"loom_devstack.deploy imports outside the standard library: {outside}"


def test_it_imports_and_runs_with_no_site_packages(tmp_path: Path) -> None:
    """The box's Python has nothing installed: `-S` drops site-packages (and the venv),
    so a dynamic or indirect third-party import fails here too."""
    assert MODULE.is_file(), f"{MODULE.relative_to(ROOT)} does not exist"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(SRC)}
    env["HOME"] = str(tmp_path)
    # Only the modules the import itself loads count: `__main__`, `builtins` and whatever
    # the interpreter loaded at start are in the snapshot taken before it.
    probe = (
        "import sys; before = set(sys.modules); import loom_devstack.deploy as d; "
        "bad = sorted(m for m in set(sys.modules) - before"
        " if m.split('.')[0] not in sys.stdlib_module_names"
        " and m.split('.')[0] != 'loom_devstack'); print(bad); assert not bad, bad"
    )
    r = subprocess.run(
        [sys.executable, "-S", "-c", probe], env=env, capture_output=True, text=True, timeout=60
    )
    assert r.returncode == 0, r.stdout + r.stderr
    h = subprocess.run(
        [sys.executable, "-S", "-m", "loom_devstack.deploy", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert h.returncode == 0, h.stdout + h.stderr
    for sub in ("secrets", "ses-smtp-password", "apply", "status", "rollback", "acceptance"):
        assert sub in h.stdout, f"`loom-deploy --help` does not list `{sub}`:\n{h.stdout}"
    a = subprocess.run(
        [sys.executable, "-S", "-m", "loom_devstack.deploy", "apply", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert a.returncode == 0, a.stdout + a.stderr
    assert "--release" in a.stdout and "--only" in a.stdout, a.stdout
    c = subprocess.run(
        [sys.executable, "-S", "-m", "loom_devstack.deploy", "acceptance", "--help"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert c.returncode == 0, c.stdout + c.stderr
    assert "--p4-release" in c.stdout, c.stdout


def test_loom_deploy_is_the_console_script_of_the_module() -> None:
    found = [ep for ep in entry_points(group="console_scripts") if ep.name == "loom-deploy"]
    assert found, "no `loom-deploy` console script is installed (project.scripts)"
    assert found[0].value.split(":")[0] == "loom_devstack.deploy", found[0].value


# ---------------------------------------------------------------- plan_steps: the order


def test_every_combination_of_changed_images_gives_the_p4_order() -> None:
    plan_steps = deploy().plan_steps
    checked = 0
    for n in range(len(ORDER) + 1):
        for changed in itertools.combinations(ORDER, n):
            images = dict(A["images"])
            for svc in changed:
                images[svc] = B["images"][svc]
            target = manifest("b" * 40, images)
            steps = list(plan_steps(A, target, None))
            want = (["redis"] if "redis" in changed else []) + ALWAYS
            assert ups(steps) == want, f"changed {changed}: {ups(steps)}"
            assert_migrate_first(steps)
            checked += 1
    assert checked == 2 ** len(ORDER)


def test_the_same_release_again_updates_every_service_but_redis() -> None:
    """An apply with the same release (a new domain or name, §6.11) still re-renders and
    updates the services; Redis keeps running."""
    steps = list(deploy().plan_steps(A, copy.deepcopy(A), None))
    assert ups(steps) == ALWAYS


def test_the_first_deploy_starts_everything_in_order() -> None:
    steps = list(deploy().plan_steps(None, A, None))
    assert ups(steps) == ORDER
    assert_migrate_first(steps)


def test_plan_steps_is_pure() -> None:
    plan_steps = deploy().plan_steps
    cur, tgt = copy.deepcopy(A), copy.deepcopy(B)
    first = list(plan_steps(cur, tgt, None))
    assert list(plan_steps(cur, tgt, None)) == first
    assert cur == A and tgt == B, "plan_steps changed its arguments"


# ---------------------------------------------------------------- P4: --only


def test_only_poolmgr_newer_than_the_running_gateway_is_refused() -> None:
    plan_steps = deploy().plan_steps
    assert_refused(lambda: plan_steps(A, B, ("poolmgr",)))
    assert_refused(lambda: plan_steps(A, B, ["poolmgr"]))


def test_only_poolmgr_with_another_service_but_not_the_gateway_is_refused() -> None:
    plan_steps = deploy().plan_steps
    assert_refused(lambda: plan_steps(A, B, ("poolmgr", "web")))
    assert_refused(lambda: plan_steps(A, B, ("billing", "poolmgr")))


def test_only_poolmgr_when_the_gateway_already_runs_the_release_is_allowed() -> None:
    assert ups(deploy().plan_steps(B, B, ("poolmgr",))) == ["poolmgr"]


@pytest.mark.parametrize("only", [("gateway", "poolmgr"), ("poolmgr", "gateway")])
def test_only_gateway_and_poolmgr_updates_the_gateway_first(only: tuple[str, ...]) -> None:
    assert ups(deploy().plan_steps(A, B, only)) == ["gateway", "poolmgr"]


@pytest.mark.parametrize(
    "only, want",
    [
        (("web",), ["web"]),
        (("gateway",), ["gateway"]),
        (("caddy", "billing"), ["billing", "caddy"]),
        (("grafana", "web", "gateway", "poolmgr"), ["gateway", "poolmgr", "web", "grafana"]),
    ],
)
def test_only_keeps_the_order_and_updates_nothing_else(
    only: tuple[str, ...], want: list[str]
) -> None:
    assert ups(deploy().plan_steps(A, B, only)) == want


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
