"""`loom-podproxy-bundle` on a dirty tree (wave 2b review, docs/phase1-m2-spec.md §9).

The bundle is built from git HEAD, so a lock written from a tree with uncommitted changes
to `services/podproxy` or `uv.lock` would name a bundle that is not what the working tree
holds. Writing the lock then is refused (non-zero exit, lock untouched); `--check` only
reads the lock and still reports on a dirty tree. Dirt elsewhere in the repository does
not matter.

Each test works in a throwaway git repository holding the committed `services/podproxy`,
`uv.lock` and lock file, so the real tree is never dirtied. The CLI finds the repository
around its working directory. Building fetches the locked dependency wheels, so these
tests are `network`.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.network,
    pytest.mark.timeout(1800),
    pytest.mark.xdist_group("podproxy-bundle"),
]

REPO = Path(__file__).resolve().parents[3]
LOCK = "config/serving-artifacts.lock"
GIT = [
    "git",
    "-c",
    "user.name=loom-test",
    "-c",
    "user.email=test@example.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "core.autocrlf=false",
]
# The temp repositories must not inherit a GIT_DIR or index pointing at the real tree.
ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def bundle_command() -> list[str]:
    script = shutil.which("loom-podproxy-bundle", path=str(Path(sys.executable).parent))
    assert script, "no loom-podproxy-bundle console script in this environment"
    return [script]


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        [*GIT, *args], cwd=repo, capture_output=True, text=True, check=True, env=ENV
    )
    return done.stdout


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A git repository with the real HEAD's podproxy, uv.lock and lock (or, outside a
    git checkout, those files as they are), plus one file elsewhere, all committed."""
    repo = tmp_path_factory.mktemp("bundle-repo") / "repo"
    repo.mkdir()
    paths = ["services/podproxy", "uv.lock"]
    if (REPO / LOCK).is_file():
        paths.append(LOCK)
    archived = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "archive", "--format=tar", "HEAD", "--", *paths],
        cwd=REPO,
        capture_output=True,
        env=ENV,
    )
    if archived.returncode == 0:
        with tarfile.open(fileobj=io.BytesIO(archived.stdout)) as tar:
            tar.extractall(repo, filter="data")
    else:  # not a git checkout (e.g. a mirrored copy): the files as they are
        junk = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")
        for path in paths:
            src, dst = REPO / path, repo / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=junk)
            else:
                shutil.copyfile(src, dst)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("unrelated\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "test tree")
    return repo


@pytest.fixture
def repo(template: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "repo"
    shutil.copytree(template, copy, symlinks=True)
    return copy


def run_bundle(repo: Path, out: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*bundle_command(), "--out", str(out), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=900,
        env=ENV,
    )


def snapshot(path: Path) -> tuple[int, int, bytes] | None:
    """Enough to tell a rewrite (an atomic replace changes the inode) from no write."""
    if not path.exists():
        return None
    st = path.stat()
    return st.st_ino, st.st_mtime_ns, path.read_bytes()


def clean_build(repo: Path, tmp_path: Path) -> tuple[str, Path]:
    """Build on the still-clean tree into a scratch lock: (sha256, that lock)."""
    lock = tmp_path / "clean.lock"
    done = run_bundle(repo, tmp_path / "clean-out", "--lock", str(lock))
    assert done.returncode == 0, f"the clean tree did not build:\n{done.stdout}{done.stderr}"
    (tarball,) = [p for p in (tmp_path / "clean-out").rglob("*") if tarfile.is_tarfile(p)]
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert digest in lock.read_text()
    return digest, lock


def _append(path: Path, text: str) -> None:
    with path.open("a", newline="\n") as f:
        f.write(text)


def _edit_proxy_source(repo: Path) -> None:
    _append(repo / "services/podproxy/src/loom_podproxy/proxy.py", "# local edit\n")


def _stage_pyproject_edit(repo: Path) -> None:
    _append(repo / "services/podproxy/pyproject.toml", "# local edit\n")
    git(repo, "add", "services/podproxy/pyproject.toml")


def _delete_package_file(repo: Path) -> None:
    (repo / "services/podproxy/src/loom_podproxy/__init__.py").unlink()


def _edit_uv_lock(repo: Path) -> None:
    _append(repo / "uv.lock", "\n")


DIRT: dict[str, Callable[[Path], None]] = {
    "proxy-source-modified": _edit_proxy_source,
    "pyproject-staged": _stage_pyproject_edit,
    "package-file-deleted": _delete_package_file,
    "uv-lock-modified": _edit_uv_lock,
}


@pytest.mark.parametrize("dirty", DIRT.values(), ids=DIRT.keys())
@pytest.mark.parametrize("seeded", [True, False], ids=["lock-exists", "no-lock"])
def test_refuses_to_write_the_lock_on_a_dirty_tree(
    repo: Path, tmp_path: Path, dirty: Callable[[Path], None], seeded: bool
) -> None:
    _, clean_lock = clean_build(repo, tmp_path)  # the same tree builds while clean
    lock = tmp_path / "serving-artifacts.lock"
    if seeded:
        shutil.copyfile(clean_lock, lock)
    before = snapshot(lock)
    repo_lock_before = snapshot(repo / LOCK)
    dirty(repo)

    done = run_bundle(repo, tmp_path / "out", "--lock", str(lock))

    assert done.returncode != 0, f"wrote a lock from a dirty tree:\n{done.stdout}{done.stderr}"
    assert snapshot(lock) == before
    assert snapshot(repo / LOCK) == repo_lock_before


def test_the_default_lock_is_untouched_on_a_dirty_tree(repo: Path, tmp_path: Path) -> None:
    clean_build(repo, tmp_path)
    before = snapshot(repo / LOCK)
    _edit_proxy_source(repo)

    done = run_bundle(repo, tmp_path / "out")

    assert done.returncode != 0, done.stdout + done.stderr
    assert snapshot(repo / LOCK) == before


def test_dirt_outside_the_bundle_inputs_does_not_matter(repo: Path, tmp_path: Path) -> None:
    sha, _ = clean_build(repo, tmp_path)
    _append(repo / "docs" / "notes.md", "edited\n")
    (repo / "docs" / "untracked.md").write_text("new\n")
    lock = tmp_path / "serving-artifacts.lock"

    done = run_bundle(repo, tmp_path / "out", "--lock", str(lock))

    assert done.returncode == 0, done.stdout + done.stderr
    assert sha in lock.read_text()


@pytest.mark.parametrize("dirty", DIRT.values(), ids=DIRT.keys())
def test_check_still_reports_on_a_dirty_tree(
    repo: Path, tmp_path: Path, dirty: Callable[[Path], None]
) -> None:
    sha, matching = clean_build(repo, tmp_path)
    wrong = tmp_path / "wrong.lock"
    wrong.write_text(matching.read_text().replace(sha, "0" * 64))
    assert sha not in wrong.read_text()
    dirty(repo)
    before = snapshot(matching), snapshot(wrong)

    agrees = run_bundle(repo, tmp_path / "out-a", "--check", "--lock", str(matching))
    differs = run_bundle(repo, tmp_path / "out-b", "--check", "--lock", str(wrong))

    assert agrees.returncode == 0, f"--check refused a dirty tree:\n{agrees.stderr}"
    assert differs.returncode != 0, differs.stdout + differs.stderr
    assert (snapshot(matching), snapshot(wrong)) == before
