"""Pod proxy hardening from the wave 2b review (docs/phase1-m2-spec.md §9).

- An allowlist file whose last entry has no newline after it is a parse failure, like a
  blank one: a truncated `96.0.0.1/32` reads as the valid, much wider `96.0.0.1/3` (which
  covers 127.0.0.1). The proxy refuses to start on one, and a reload with one keeps the
  old allowlist and logs at error.
- A SIGHUP that arrives after the process has started but before it serves does not
  kill it. The test holds the proxy at startup by giving it its token through a FIFO, so
  the signal lands at a known point: inside `loom-podproxy`, before it listens. (A signal
  during the interpreter's own boot, before any Python runs, is out of any program's
  reach and is not tested.)

Uses the process harness of test_podproxy_sighup.py.
"""

from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from podproxy_testlib import Upstream
from test_podproxy_sighup import Files, ProxyProcess

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP and FIFOs are POSIX")


@pytest.fixture
def files(tmp_path: Path) -> Files:
    return Files(tmp_path)


@pytest.fixture
def start(
    files: Files, upstream: tuple[Upstream, int], tmp_path: Path
) -> Iterator[Callable[..., ProxyProcess]]:
    procs: list[ProxyProcess] = []

    def factory(*, allow_file: bool = False, wait: bool = True) -> ProxyProcess:
        proc = ProxyProcess(files, upstream[1], tmp_path / f"proxy-{len(procs)}.log", allow_file)
        procs.append(proc)  # registered before the wait: stopped even if never ready
        if wait:
            proc.ready()
        return proc

    yield factory
    for p in procs:
        p.stop()


def _exit_code(proxy: ProxyProcess, timeout: float = 15) -> int | None:
    try:
        return proxy.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


# ---------------------------------------------------------------- unterminated allowlist

# Each allows 127.0.0.1 once a newline ends it; without one, each is a truncated file.
UNTERMINATED = {
    "cut-cidr": "96.0.0.1/3",
    "cut-second-line": "10.0.0.0/8\n127.0.0.1/32",
    "address": "127.0.0.1",
    "trailing-spaces": "127.0.0.1/32   ",
}


@pytest.mark.parametrize("content", UNTERMINATED.values(), ids=UNTERMINATED.keys())
def test_an_unterminated_allowlist_refuses_to_start(
    files: Files, start: Callable[..., ProxyProcess], content: str
) -> None:
    files.allow.write_text(content + "\n")
    first = start(allow_file=True)  # the same entries, newline-terminated, serve
    first.stop()

    files.allow.write_text(content)
    proxy = start(allow_file=True, wait=False)
    code = _exit_code(proxy)
    assert code not in (None, 0), f"proxy started on an unterminated allowlist:\n{proxy.log()}"


def test_an_unterminated_allowlist_on_reload_keeps_the_old_allowlist(
    files: Files, start: Callable[..., ProxyProcess]
) -> None:
    proxy = start(allow_file=True)
    files.allow.write_text("10.0.0.0/8\n")
    proxy.hup()
    proxy.wait_for(proxy.closed_before_handshake, "127.0.0.1 to be refused")
    assert proxy.errors() == 0

    files.allow.write_text("10.0.0.0/8\n96.0.0.1/3")  # cut off from 96.0.0.1/32
    proxy.hup()

    proxy.wait_for(lambda: proxy.errors() > 0, "the failed reload to be logged at error")
    assert proxy.closed_before_handshake()  # the old allowlist: 127.0.0.1 still outside
    assert proxy.status(files.token()) is None
    assert proxy.proc.poll() is None

    # Control: the same entries with the newline are a valid, wider allowlist.
    files.allow.write_text("10.0.0.0/8\n96.0.0.1/3\n")
    proxy.hup()
    proxy.wait_for(lambda: proxy.status(files.token()) == 200, "the terminated file to apply")


# ---------------------------------------------------------------- SIGHUP before ready


def _open_writer(fifo: Path, proxy: ProxyProcess, timeout: float = 15) -> int:
    """Open the FIFO for writing once the proxy has opened it for reading."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno != errno.ENXIO:  # ENXIO: no reader yet
                raise
        code = proxy.proc.poll()
        assert code is None, f"proxy exited ({code}) before reading its token:\n{proxy.log()}"
        assert time.monotonic() < deadline, f"proxy never read its token:\n{proxy.log()}"
        time.sleep(0.01)


def test_a_sighup_before_the_proxy_is_ready_does_not_kill_it(
    files: Files, start: Callable[..., ProxyProcess]
) -> None:
    token = files.token()
    files.token_file.unlink()
    os.mkfifo(files.token_file)

    proxy = start(wait=False)
    fd = _open_writer(files.token_file, proxy)  # the proxy is now blocked reading it
    try:
        assert proxy.status(token) is None  # not serving yet
        os.kill(proxy.proc.pid, signal.SIGHUP)
        time.sleep(0.5)
        code = proxy.proc.poll()
        assert code is None, f"a SIGHUP before ready killed the proxy ({code}):\n{proxy.log()}"
        os.write(fd, (token + "\n").encode())
    finally:
        os.close(fd)
    # From here on the token is an ordinary file again, as on a pod.
    tmp = files.token_file.with_name("pod.token.tmp")
    tmp.write_text(token + "\n")
    os.replace(tmp, files.token_file)

    proxy.ready()

    # And its reload still works once it serves.
    files.write_token("rotated-" + token)
    proxy.hup()
    proxy.wait_for(lambda: proxy.status("rotated-" + token) == 200, "the reload after ready")
    assert proxy.status(token) == 403
