"""The image workflow and its GitHub environment (M3 spec §5.6, §6.1, §6.12).

The OIDC role trusts only the subject `repo:arulster17/loom:environment:platform-images`;
IAM cannot test a branch and an event together, so the environment (deployment branches:
`main` only) holds the branch rule and `images.yml`, the only workflow naming that
environment, runs on `workflow_dispatch` only. Read statically from
`.github/workflows/*.yml`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = ROOT / ".github" / "workflows"
IMAGES = WORKFLOWS / "images.yml"
ENV = "platform-images"


def load(path: Path) -> dict[Any, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), f"{path.name} is not a mapping"
    return doc


def triggers(doc: dict[Any, Any]) -> set[str]:
    # YAML 1.1 reads a bare `on:` key as the boolean True.
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return {str(x) for x in on}
    if isinstance(on, dict):
        return {str(k) for k in on}
    raise AssertionError(f"no `on:` triggers: {on!r}")


def environments(doc: dict[Any, Any]) -> list[str]:
    out = []
    for job in (doc.get("jobs") or {}).values():
        env = (job or {}).get("environment")
        if isinstance(env, dict):
            env = env.get("name")
        if env is not None:
            out.append(str(env))
    return out


def workflow_files() -> list[Path]:
    return sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])


def test_images_runs_on_workflow_dispatch_only() -> None:
    assert IMAGES.is_file(), ".github/workflows/images.yml does not exist"
    assert triggers(load(IMAGES)) == {"workflow_dispatch"}, triggers(load(IMAGES))


def test_images_uses_the_platform_images_environment() -> None:
    assert IMAGES.is_file(), ".github/workflows/images.yml does not exist"
    envs = environments(load(IMAGES))
    assert ENV in envs, f"no job of images.yml has `environment: {ENV}` ({envs})"
    assert set(envs) == {ENV}, f"images.yml names other environments: {envs}"


def test_no_other_workflow_names_the_environment() -> None:
    assert IMAGES.is_file(), ".github/workflows/images.yml does not exist"
    others = [p for p in workflow_files() if p.name != IMAGES.name]
    assert others, "no other workflows found: the scan would be vacuous"
    for p in others:
        assert ENV not in environments(load(p)), f"{p.name} uses environment {ENV}"
        assert not re.search(rf"(?<![\w-]){re.escape(ENV)}(?![\w-])", p.read_text()), (
            f"{p.name} names {ENV}"
        )


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
