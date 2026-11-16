"""The pod proxy bundle (docs/phase1-m2-spec.md §9, §13 item 24).

`loom-podproxy-bundle --out DIR [--lock FILE]` builds, from the committed git blobs
(`git archive HEAD`), one tarball of the `loom-podproxy` wheel and its locked dependency
wheels, and records the tarball's sha256 in the lock (default
`config/serving-artifacts.lock`, committed). Bootstrap installs it into a fresh venv with
`pip install --no-index --find-links`. Built from git blobs, the hash does not depend on
the checkout (CRLF or not), so the committed lock must name this commit's bundle.

The tests never write the repo's lock: each build gets `--lock` in tmp_path, seeded with a
copy of the committed lock when there is one (the lock may be an input too).

Building fetches the dependency wheels (and the install test a CPython 3.12), so those
tests are `network`: run them with `-m network`.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
COMMITTED_LOCK = REPO / "config" / "serving-artifacts.lock"
POD_PYTHON = "3.12"  # vLLM images ship CPython 3.12


def _bin(venv: Path, name: str) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin") / name


def bundle_command() -> list[str]:
    script = shutil.which("loom-podproxy-bundle", path=str(Path(sys.executable).parent))
    assert script, "no loom-podproxy-bundle console script in this environment"
    return [script]


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_id(filename: str) -> tuple[str, str]:
    """(normalized project name, version) from a wheel file name."""
    name, version = Path(filename).name.split("-")[:2]
    return normalize(name), version


def locked_closure(root: str = "loom-podproxy") -> tuple[set[tuple[str, str]], set[str]]:
    """From uv.lock: (every (name, version) reachable from `root` under any marker, the
    names reachable without a marker)."""
    lock = tomllib.loads((REPO / "uv.lock").read_text(encoding="utf-8"))
    versions: dict[str, list[dict]] = {}
    for pkg in lock["package"]:
        versions.setdefault(normalize(pkg["name"]), []).append(pkg)

    def deps(name: str) -> list[dict]:
        out: list[dict] = []
        for pkg in versions[name]:
            out += pkg.get("dependencies", [])
        return out

    allowed: set[tuple[str, str]] = set()
    seen: set[str] = set()
    todo = [normalize(root)]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        allowed |= {(name, str(p["version"])) for p in versions[name]}
        todo += [normalize(d["name"]) for d in deps(name)]

    required: set[str] = set()
    todo = [normalize(root)]
    while todo:
        name = todo.pop()
        if name in required:
            continue
        required.add(name)
        todo += [normalize(d["name"]) for d in deps(name) if "marker" not in d]
    return allowed, required


@dataclass(frozen=True)
class Build:
    tarball: Path
    lock: Path
    sha256: str


def build(root: Path) -> Build:
    out = root / "out"
    out.mkdir(parents=True)
    lock = root / "serving-artifacts.lock"
    if COMMITTED_LOCK.is_file():
        shutil.copyfile(COMMITTED_LOCK, lock)
    done = subprocess.run(
        [*bundle_command(), "--out", str(out), "--lock", str(lock)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, f"bundle build failed:\n{done.stdout}\n{done.stderr}"
    tarballs = [p for p in out.rglob("*") if p.is_file() and tarfile.is_tarfile(p)]
    assert len(tarballs) == 1, f"expected one tarball in {out}: {sorted(out.rglob('*'))}"
    digest = hashlib.sha256(tarballs[0].read_bytes()).hexdigest()
    return Build(tarballs[0], lock, digest)


def wheels_in(tarball: Path) -> list[str]:
    with tarfile.open(tarball) as tar:
        return [m.name for m in tar.getmembers() if m.isfile() and m.name.endswith(".whl")]


# ---------------------------------------------------------------- no network


def test_bundle_cli_is_installed() -> None:
    done = subprocess.run(
        [*bundle_command(), "--help"], capture_output=True, text=True, timeout=60, cwd=REPO
    )
    assert done.returncode == 0, done.stderr
    assert "--out" in done.stdout and "--lock" in done.stdout


# ---------------------------------------------------------------- building (network)


@pytest.fixture(scope="module")
def builds(tmp_path_factory: pytest.TempPathFactory) -> tuple[Build, Build]:
    """Two independent builds from the same tree and lock."""
    return build(tmp_path_factory.mktemp("bundle-a")), build(tmp_path_factory.mktemp("bundle-b"))


@pytest.mark.network
@pytest.mark.timeout(1800)
@pytest.mark.xdist_group("podproxy-bundle")
def test_two_builds_have_the_same_sha256(builds: tuple[Build, Build]) -> None:
    a, b = builds
    assert a.sha256 == b.sha256
    assert a.tarball.read_bytes() == b.tarball.read_bytes()
    for one in builds:
        assert one.lock.is_file(), "the build recorded no lock file"
        assert one.sha256 in one.lock.read_text(), "the lock does not record the sha256"
    assert a.lock.read_text() == b.lock.read_text()


@pytest.mark.network
@pytest.mark.timeout(1800)
@pytest.mark.xdist_group("podproxy-bundle")
def test_bundle_holds_the_locked_wheels_and_nothing_else(builds: tuple[Build, Build]) -> None:
    allowed, required = locked_closure()
    wheels = wheels_in(builds[0].tarball)
    got = {wheel_id(w) for w in wheels}
    assert len(got) == len(wheels), f"duplicate wheels: {sorted(wheels)}"
    assert {name for name, _ in got} >= required, f"missing: {required - {n for n, _ in got}}"
    assert got <= allowed, f"not locked for loom-podproxy: {sorted(got - allowed)}"
    assert {"loom-podproxy", "h11", "httpx"} <= {name for name, _ in got}


@pytest.mark.network
@pytest.mark.timeout(1800)
@pytest.mark.xdist_group("podproxy-bundle")
def test_bundle_installs_offline_into_an_empty_venv(
    builds: tuple[Build, Build], tmp_path: Path
) -> None:
    uv = shutil.which("uv")
    assert uv, "needs uv for a CPython like the pods'"
    unpacked = tmp_path / "bundle"
    with tarfile.open(builds[0].tarball) as tar:
        tar.extractall(unpacked, filter="data")
    (own,) = list(unpacked.rglob("loom_podproxy-*.whl"))
    wheels = own.parent

    venv = tmp_path / "proxy-venv"
    subprocess.run(
        [uv, "venv", "--quiet", "--seed", "--python", POD_PYTHON, str(venv)],
        check=True,
        capture_output=True,
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PIP_", "UV_"))}
    env["PIP_CONFIG_FILE"] = os.devnull
    python = _bin(venv, "python")
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--no-index",
            "--find-links",
            str(wheels),
            "loom-podproxy",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    check = subprocess.run(
        [str(python), "-m", "pip", "check"], capture_output=True, text=True, env=env
    )
    assert check.returncode == 0, check.stdout + check.stderr

    helped = subprocess.run(
        [str(_bin(venv, "loom-podproxy")), "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=60,
    )
    assert helped.returncode == 0, helped.stderr
    assert "usage" in helped.stdout.lower() and "--cert" in helped.stdout


@pytest.mark.network
@pytest.mark.timeout(1800)
@pytest.mark.xdist_group("podproxy-bundle")
def test_committed_lock_records_this_tree_s_bundle(builds: tuple[Build, Build]) -> None:
    """Bootstrap checks the shipped tarball against the committed lock, so the committed
    lock must name what this commit builds (from git blobs: the same on any checkout)."""
    assert COMMITTED_LOCK.is_file(), f"{COMMITTED_LOCK} is not committed"
    assert builds[0].sha256 in COMMITTED_LOCK.read_text()
