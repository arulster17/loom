"""serve_lease.sh, the renewal half that runs on the pod (docs/phase1-m2-spec.md §5
"Leases and the watchdog": "an atomic write of /run/loom/lease"; §11: requires
LEASE_FILE and LEASE_EPOCH; §13 item 17), run under bash.

Atomic means a temporary file in the lease file's own directory renamed over it: a
reader (the watchdog) sees the old epoch or the new one, never an empty or partial file,
and the file never goes missing.
"""

from __future__ import annotations

import importlib
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

BASH = shutil.which("bash")
MV = shutil.which("mv")
pytestmark = [
    pytest.mark.skipif(BASH is None, reason="bash not installed"),
    pytest.mark.skipif(sys.platform == "win32", reason="POSIX scripts and stubs"),
]

T0 = 1_790_000_000


def render_lease(lease_file: Path, epoch: int) -> str:
    scripts = importlib.import_module("loom_poolmgr.scripts")
    out = scripts.render("serve_lease", LEASE_FILE=str(lease_file), LEASE_EPOCH=epoch)
    assert isinstance(out, str)
    return out


def run_lease(
    lease_file: Path, epoch: int, *, path: str = "/usr/bin:/bin", **env: str
) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, "-s"],
        input=render_lease(lease_file, epoch),
        env={"PATH": path, "HOME": str(lease_file.parent), **env},
        capture_output=True,
        text=True,
        timeout=30,
    )


def lease_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run" / "loom"
    d.mkdir(parents=True)
    return d


def test_writes_the_epoch(tmp_path: Path) -> None:
    lease = lease_dir(tmp_path) / "lease"
    proc = run_lease(lease, T0 + 3600)
    assert proc.returncode == 0, proc.stderr
    assert lease.read_text().strip() == str(T0 + 3600)
    # Nothing left behind next to it.
    assert sorted(p.name for p in lease.parent.iterdir()) == ["lease"]


def test_creates_a_missing_lease_directory(tmp_path: Path) -> None:
    # On a fresh pod /run/loom does not exist (§5: creating its directory if missing).
    lease = tmp_path / "run" / "loom" / "lease"
    assert not lease.parent.parent.exists()
    proc = run_lease(lease, T0 + 3600)
    assert proc.returncode == 0, proc.stderr
    assert lease.read_text().strip() == str(T0 + 3600)
    assert sorted(p.name for p in lease.parent.iterdir()) == ["lease"]


def test_replaces_the_file_by_rename_in_its_own_directory(tmp_path: Path) -> None:
    d = lease_dir(tmp_path)
    lease = d / "lease"
    lease.write_text(f"{T0}\n")
    before = lease.stat().st_ino
    # A hard link keeps the old inode: written in place, it would show the new epoch.
    old = tmp_path / "old-inode"
    os.link(lease, old)
    # Log every `mv` (if the script uses one) on the way to the real one.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mv_log = tmp_path / "mv.log"
    assert MV is not None
    stub = bin_dir / "mv"
    stub.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\n" "$@" >>{shlex.quote(str(mv_log))}\n'
        f'printf -- "--end--\\n" >>{shlex.quote(str(mv_log))}\n'
        f'exec {shlex.quote(MV)} "$@"\n'
    )
    stub.chmod(0o755)
    # The temporary file is not under $TMPDIR (another filesystem in the pod: /tmp vs /run).
    proc = run_lease(
        lease, T0 + 7200, path=f"{bin_dir}:/usr/bin:/bin", TMPDIR=str(tmp_path / "no-such-dir")
    )
    assert proc.returncode == 0, proc.stderr
    assert lease.read_text().strip() == str(T0 + 7200)
    assert lease.stat().st_ino != before, "the lease file was written in place, not renamed"
    assert old.read_text().strip() == str(T0)
    assert sorted(p.name for p in d.iterdir()) == ["lease"]
    if mv_log.exists():
        for call in mv_log.read_text().split("--end--\n"):
            operands = [a for a in call.splitlines() if a and not a.startswith("-")]
            if len(operands) >= 2:
                src, dst = Path(operands[-2]), Path(operands[-1])
                assert src.parent == d, f"temporary file {src} is not in {d}"
                assert dst in (lease, d)


def test_a_concurrent_reader_never_sees_a_partial_or_missing_lease(tmp_path: Path) -> None:
    lease = lease_dir(tmp_path) / "lease"
    assert run_lease(lease, T0).returncode == 0
    written = {T0}
    bad: list[Any] = []
    reads = 0
    stop = threading.Event()

    def reader() -> None:
        nonlocal reads
        while not stop.is_set():
            try:
                text = lease.read_text()
            except FileNotFoundError:
                bad.append("<missing>")
                continue
            reads += 1
            try:
                value = int(text.strip())
            except ValueError:
                bad.append(text)
                continue
            if value not in written:
                bad.append(text)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for i in range(1, 61):
            epoch = T0 + 60 * i
            written.add(epoch)
            proc = run_lease(lease, epoch)
            assert proc.returncode == 0, proc.stderr
    finally:
        stop.set()
        thread.join(timeout=10)
    assert reads > 0
    assert not bad, f"reader saw {bad[:5]}"
    assert lease.read_text().strip() == str(T0 + 3600)


def test_a_failed_write_exits_non_zero(tmp_path: Path) -> None:
    # The manager retries a failed renewal each tick (§5), so it must see the failure.
    not_a_dir = tmp_path / "run"
    not_a_dir.write_text("a file where the lease directory should be\n")
    proc = run_lease(not_a_dir / "lease", T0 + 3600)
    assert proc.returncode != 0
