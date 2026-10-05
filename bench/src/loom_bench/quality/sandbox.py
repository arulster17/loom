"""Run untrusted, model-written Python under resource limits.

Each program runs in a fresh interpreter (`python -I -S`: isolated mode, no
site-packages) as its own process group, with an empty environment, a
throw-away working directory, and rlimits set by a small launcher before the
program is compiled: CPU seconds, file size, open files, no new processes, no
core dumps, and address space where the OS enforces it (Linux; macOS rejects
RLIMIT_AS, so memory is unbounded there). A wall-clock timeout kills the
whole process group.

This is a resource sandbox, not a security boundary: the program runs as the
current user and can open network sockets and read files the user can read.
Run code evals inside a disposable container or VM with no network egress and
no credentials, and only with `allow_code_exec=True`, which every entry point
requires explicitly.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

_SETUP_FAILED = 113

# Runs before the untrusted program; argv: program path, cpu_s, memory_bytes, file_bytes, nofile.
_LAUNCHER = f"""
import resource, sys
path, cpu, mem, fsize, nofile = sys.argv[1], *map(int, sys.argv[2:6])
def lim(name, value, required=True):
    try:
        resource.setrlimit(getattr(resource, name), (value, value))
    except (ValueError, OSError):
        if required:
            sys.stderr.write("sandbox: cannot set " + name + "\\n")
            sys.exit({_SETUP_FAILED})
lim("RLIMIT_CPU", cpu)
lim("RLIMIT_FSIZE", fsize)
lim("RLIMIT_NOFILE", nofile)
lim("RLIMIT_CORE", 0)
lim("RLIMIT_NPROC", 0, required=sys.platform == "linux")
if mem:
    lim("RLIMIT_AS", mem, required=sys.platform == "linux")
with open(path, encoding="utf-8") as f:
    source = f.read()
del resource, lim, f, cpu, mem, fsize, nofile
sys.argv = [path]
exec(compile(source, path, "exec"), {{"__name__": "__main__", "__builtins__": __builtins__}})
"""


class CodeExecDisabled(RuntimeError):
    """Raised when code execution was not explicitly enabled."""


class SandboxLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_s: Annotated[float, Field(gt=0)] = 10.0
    cpu_s: Annotated[int, Field(ge=1)] = 10
    memory_mb: Annotated[int, Field(ge=0)] = 1024  # 0 = no address-space limit
    max_file_mb: Annotated[int, Field(ge=0)] = 16
    max_open_files: Annotated[int, Field(ge=8)] = 64


ExecStatus = Literal["passed", "failed", "timeout"]


@dataclass(frozen=True, slots=True)
class ExecResult:
    status: ExecStatus
    returncode: int | None
    duration_s: float
    # Exception class name from the last stderr line ("AssertionError"), when there is one.
    error_type: str | None = None


def _error_type(stderr: str) -> str | None:
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    head = lines[-1].split(":", 1)[0].strip()
    name = head.rsplit(".", 1)[-1]
    return name if name.isidentifier() else None


def run_python(
    program: str, *, allow_code_exec: bool, limits: SandboxLimits | None = None
) -> ExecResult:
    """Execute `program`; passed means exit status 0 within the limits.

    Raises `CodeExecDisabled` unless enabled, and RuntimeError when a required
    limit cannot be applied (the result would not be comparable).
    """
    if not allow_code_exec:
        raise CodeExecDisabled(
            "refusing to execute model-generated code: pass allow_code_exec=True, "
            "and only inside an isolated container or VM"
        )
    lim = limits or SandboxLimits()
    with tempfile.TemporaryDirectory(prefix="loom-sandbox-") as tmp:
        path = Path(tmp) / "program.py"
        path.write_text(program, encoding="utf-8")
        argv = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _LAUNCHER,
            str(path),
            str(lim.cpu_s),
            str(lim.memory_mb * 2**20),
            str(lim.max_file_mb * 2**20),
            str(lim.max_open_files),
        ]
        start = time.monotonic()
        proc = subprocess.Popen(
            argv,
            cwd=tmp,
            env={},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, err = proc.communicate(timeout=lim.timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return ExecResult("timeout", None, time.monotonic() - start)
        finally:
            _kill_group(proc)
        duration = time.monotonic() - start
    stderr = err.decode("utf-8", errors="replace")[-2000:]
    if proc.returncode == 0:
        return ExecResult("passed", 0, duration)
    if proc.returncode == _SETUP_FAILED and stderr.startswith("sandbox:"):
        raise RuntimeError(stderr.strip())
    if proc.returncode == -signal.SIGXCPU or proc.returncode == -signal.SIGKILL:
        return ExecResult("timeout", proc.returncode, duration)
    return ExecResult("failed", proc.returncode, duration, _error_type(stderr))


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
