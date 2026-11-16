"""`provider: runpod` never runs on dev placeholders (docs/phase1-m2-spec.md §11 "Fixed
after the final whole-branch review"): it requires a real proxy bundle
(`LOOM_POOLMGR_PROXY_BUNDLE`, whose sha256 matches `config/serving-artifacts.lock`) and a
lease bucket (`LOOM_POOLMGR_LEASE_BUCKET`) whatever `LOOM_ENV` is. Dev placeholders exist
only for `local`/`fake`.

Every case runs the real `loom-poolmgr run` (`cli.main(["run"])`) with `LOOM_ENV=dev`.
Nothing can reach RunPod: `cli.RunpodApi` is a stub that records and fails any use, and
`cli.PoolManager` records its construction and stops at `start()` (`ManagerLocked`, exit
1). "Refused" means the run ends (non-zero exit or an exception) before any manager is
built, saying why. Whether the check sits in the settings' `production_problems` or in
the CLI is left to the implementation.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml

from loom_bench.registry import REPO_ROOT, load_registry
from loom_platform.serving import load_serving
from loom_poolmgr import cli
from loom_poolmgr.manager import ManagerLocked, PoolManager
from loom_poolmgr.settings import PoolmgrSettings

COMMITTED_SERVING = REPO_ROOT / "config" / "serving.yaml"


class NoRunpod:
    """Stands in for `RunpodApi`: building it is harmless; any use is recorded (and
    fails the test)."""

    built: ClassVar[int] = 0
    used: ClassVar[list[str]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        NoRunpod.built += 1

    def __getattr__(self, name: str) -> Any:
        NoRunpod.used.append(name)
        raise RuntimeError(f"RunPod was called: RunpodApi.{name}")


@pytest.fixture
def run_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    for var in list(os.environ):
        if var.startswith("LOOM_POOLMGR_") or var in (
            "LOOM_SERVING_YAML",
            "LOOM_ENV",
            "RUNPOD_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
    built: list[dict[str, Any]] = []

    class Recording(PoolManager):
        def __init__(self, config: Any, **kw: Any) -> None:
            built.append({"config": config, **kw})
            super().__init__(config, **kw)

        async def start(self) -> None:
            raise ManagerLocked("recorded (test)")

        async def stop(self) -> None:
            return None

    NoRunpod.built = 0
    NoRunpod.used = []
    monkeypatch.setattr(cli, "PoolManager", Recording)
    monkeypatch.setattr(cli, "RunpodApi", NoRunpod)
    monkeypatch.setenv("LOOM_ENV", "dev")
    monkeypatch.setenv("LOOM_POOLMGR_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOOM_POOLMGR_METRICS_PORT", "0")
    monkeypatch.setenv("LOOM_REDIS_URL", "redis://127.0.0.1:9/0")  # never connected
    monkeypatch.setenv("RUNPOD_API_KEY", "test-not-a-key")
    yield {"monkeypatch": monkeypatch, "tmp": tmp_path, "built": built}
    assert NoRunpod.used == [], f"RunPod was called: {NoRunpod.used}"


def runpod_serving(tmp_path: Path) -> Path:
    doc = yaml.safe_load(COMMITTED_SERVING.read_text())
    doc["provider"] = "runpod"
    doc["network"]["gateway_egress_cidrs"] = ["203.0.113.10/32"]
    path = tmp_path / "serving-runpod.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def run(capsys: pytest.CaptureFixture[str]) -> tuple[int | str | None, str]:
    """`loom-poolmgr run`: (exit code, or "exception"; everything it said)."""
    code: int | str | None
    text = ""
    try:
        cli.main(["run"])
        code = 0
    except SystemExit as e:
        code = e.code
        if isinstance(e.code, str):
            text += e.code
    except Exception as e:  # a refusal by exception counts too
        code = "exception"
        text += f"{type(e).__name__}: {e}"
    out = capsys.readouterr()
    return code, text + out.out + out.err


def bundle_file(tmp_path: Path, data: bytes = b"not the pinned bundle") -> Path:
    path = tmp_path / "loom-podproxy-bundle.tar.gz"
    path.write_bytes(data)
    return path


def assert_refused(
    run_env: dict[str, Any], code: int | str | None, text: str, *needles: str
) -> None:
    assert NoRunpod.used == [], f"RunPod was called: {NoRunpod.used}"
    assert code not in (0, None, 1), f"not refused (exit {code}): {text}"
    assert run_env["built"] == [], "a manager was built: the run was not refused"
    assert any(n in text for n in needles), f"the refusal does not say why ({needles}): {text}"


def test_runpod_without_a_proxy_bundle_is_refused_in_dev(
    run_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    mp = run_env["monkeypatch"]
    mp.setenv("LOOM_POOLMGR_SERVING_YAML", str(runpod_serving(run_env["tmp"])))
    mp.setenv("LOOM_POOLMGR_LEASE_BUCKET", "loom-test-leases")
    code, text = run(capsys)
    assert_refused(run_env, code, text, "LOOM_POOLMGR_PROXY_BUNDLE")


def test_runpod_with_a_bundle_not_matching_the_lock_is_refused_in_dev(
    run_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    mp = run_env["monkeypatch"]
    mp.setenv("LOOM_POOLMGR_SERVING_YAML", str(runpod_serving(run_env["tmp"])))
    mp.setenv("LOOM_POOLMGR_LEASE_BUCKET", "loom-test-leases")
    mp.setenv("LOOM_POOLMGR_PROXY_BUNDLE", str(bundle_file(run_env["tmp"])))
    code, text = run(capsys)
    assert_refused(run_env, code, text, "sha256", "lock", "LOOM_POOLMGR_PROXY_BUNDLE")


def test_runpod_without_a_lease_bucket_is_refused_in_dev(
    run_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """With a bundle that matches the lock (the CLI's lock read is pointed at its
    sha256), only the lease bucket is missing."""
    mp = run_env["monkeypatch"]
    data = b"a bundle the lock names (test)"
    sha = hashlib.sha256(data).hexdigest()
    mp.setattr(cli, "read_lock", lambda *a, **k: {"sha256": sha}, raising=False)
    mp.setenv("LOOM_POOLMGR_SERVING_YAML", str(runpod_serving(run_env["tmp"])))
    mp.setenv("LOOM_POOLMGR_PROXY_BUNDLE", str(bundle_file(run_env["tmp"], data)))
    code, text = run(capsys)
    assert_refused(run_env, code, text, "LOOM_POOLMGR_LEASE_BUCKET")


def test_local_in_dev_still_starts_on_placeholders(
    run_env: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """No bundle, no lease bucket, no CA: `provider: local` (the committed file) builds
    its manager and reaches start()."""
    run_env["monkeypatch"].setenv("LOOM_POOLMGR_SERVING_YAML", str(COMMITTED_SERVING))
    code, text = run(capsys)
    assert code == 1, text  # ManagerLocked from the recording manager's start()
    [built] = run_env["built"]
    assert built["config"].provider == "local"
    assert NoRunpod.built == 0


@pytest.mark.parametrize("provider", ["local", "fake"])
def test_dev_placeholders_are_no_problem_for_local_and_fake(
    run_env: dict[str, Any], provider: str
) -> None:
    doc = yaml.safe_load(COMMITTED_SERVING.read_text())
    doc["provider"] = provider
    path = run_env["tmp"] / f"serving-{provider}.yaml"
    path.write_text(yaml.safe_dump(doc))
    config = load_serving(path, registry=load_registry())
    problems = PoolmgrSettings().production_problems(config)
    assert not [p for p in problems if "PROXY_BUNDLE" in p or "LEASE_BUCKET" in p], problems
