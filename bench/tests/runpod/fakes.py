import inspect
import re
import shlex
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from loom_bench.providers import runpod_fake
from loom_bench.providers.aws_ssm import required_vars
from loom_bench.providers.runpod_fake import FAKE_KEY, GRAPHQL, REST
from loom_bench.providers.runpod_ssh import ExecResult, SshTarget

__all__ = ["FAKE_KEY", "GRAPHQL", "REST", "FakeRunpod", "create_response_fixture"]

FIXTURES = Path(__file__).parent / "fixtures"


def create_response_fixture() -> dict[str, Any]:
    """Shape of a real `POST /pods` response (spike, 2026-10-06), ids and env redacted.
    Tests monkeypatch this name to change the shape of the pods `FakeRunpod` creates."""
    return runpod_fake.create_response_fixture()


class FakeRunpod(runpod_fake.FakeRunpod):
    """`loom_bench.providers.runpod_fake.FakeRunpod` (moved there so other packages can
    use it), with its pod template taken from this module's `create_response_fixture`."""

    def _pod_template(self) -> dict[str, Any]:
        return create_response_fixture()


# --- SSH into fake pods ------------------------------------------------------------

PodResult = tuple[int, str, str]  # (exit code, stdout, stderr) of one pod script
Handler = Callable[[SshTarget, str], PodResult | Awaitable[PodResult]]
TEMPLATES = ("pod_start", "start_engine", "stop_engine", "run_job", "collect_profile")
_REQUIRES = {
    " ".join(sorted(required_vars(name, template_dir="runpod_scripts"))): name for name in TEMPLATES
}


def template_of(script: str) -> str:
    """Which runpod_scripts template a rendered script came from (its `# requires:`)."""
    m = re.search(r"^# requires:(.*)$", script, re.MULTILINE)
    assert m, "not a rendered pod script"
    return _REQUIRES[" ".join(sorted(m.group(1).split()))]


def script_var(script: str, name: str) -> str:
    """A scalar variable as rendered into a pod script (`NAME='value'`)."""
    m = re.search(rf"^{name}=(.*)$", script, re.MULTILINE)
    assert m, name
    (value,) = shlex.split(m.group(1))
    return value


def script_array(script: str, name: str) -> list[str]:
    """An array variable as rendered into a pod script (`NAME=('a' 'b')`). An element may
    span lines inside its quotes (a quoted argument with newlines in it): the array ends
    at the first `)` closing a line outside any quote."""
    m = re.search(rf"^{name}=\(", script, re.MULTILINE)
    assert m, name
    for end in re.finditer(r"\)$", script[m.end() :], re.MULTILINE):
        try:
            return shlex.split(script[m.end() : m.end() + end.start()])
        except ValueError:  # that `)` sits inside a quoted element
            continue
    raise AssertionError(f"{name}: unterminated array")


def script_pairs(script: str, name: str) -> list[tuple[str, str]]:
    """A flat `repo revision ...` array variable as (repo, revision) pairs."""
    flat = script_array(script, name)
    return list(zip(flat[::2], flat[1::2], strict=True))


def engine_stdout(*, gpus: int = 1, isolation: bool = True) -> str:
    now = time.time()  # stages after the controller's own (ssh_online), in pod order
    lines = [
        f"loom-stage image_pulled {now + 1}",
        f"loom-stage sshd_ready {now + 2}",
        f"loom-stage weights_ready {now + 3}",
        f"loom-stage engine_started {now + 4}",
        f"loom-stage engine_healthy {now + 5}",
        f"loom-stage first_token {now + 5.5}",
        "loom-sys gpus " + ",".join(["NVIDIA L40S"] * gpus),
        f"loom-sys gpu_count {gpus}",
        "loom-sys driver_version 580.159.03",
        "loom-sys cuda_version 13.0",
    ]
    if isolation:
        lines.append("loom-sys job_isolation ok")
    return "\n".join(lines) + "\n"


def ok(stdout: str = "") -> PodResult:
    return 0, stdout, ""


def _default(target: SshTarget, script: str) -> PodResult:
    name = template_of(script)
    if name == "start_engine":
        return ok(engine_stdout())
    if name == "stop_engine":
        return ok("loom-stopped\n")
    if name == "run_job":
        return ok("loom-sys job_isolation ok\nloom-job-done\n")
    raise AssertionError(f"unexpected script {name}")


class FakePodExec:
    """`SshExec` for fake pods: answers the SSH probe and `run_detached`'s launch, poll,
    stderr and collect calls, handing each launched pod script to `handlers[template]`
    (sync or async) for its (rc, stdout, stderr). `offline` probes fail like a pod
    whose sshd is not up yet (exit 255)."""

    def __init__(self, handlers: Mapping[str, Handler] | None = None, *, offline: int = 0) -> None:
        self.handlers = dict(handlers or {})
        self.offline = offline
        self.probes = 0
        self.launched: list[tuple[SshTarget, str, str]] = []  # (target, template, script)
        self._results: dict[str, PodResult] = {}

    def scripts(self, template: str) -> list[str]:
        return [s for _, name, s in self.launched if name == template]

    async def run(self, target: SshTarget, script: str, *, timeout_s: float) -> ExecResult:
        if script == "echo loom-online\n":
            self.probes += 1
            if self.offline:
                self.offline -= 1
                return ExecResult(255, "", "Connection timed out")
            return ExecResult(0, "loom-online\n", "")
        tail = re.match(r"^tail -c \d+ (\S+)/stderr", script)
        if tail:
            return ExecResult(0, self._results[tail.group(1)][2], "")
        m = re.search(r"^d=(\S+)$", script, re.MULTILINE)
        assert m, f"unexpected SSH script: {script[:200]}"
        ctl = m.group(1)
        if "loom-launched" in script:
            body = script.split("<<'LOOM_RUN_EOF'\n", 1)[1].rsplit("\nLOOM_RUN_EOF\n", 1)[0]
            name = template_of(body)
            self.launched.append((target, name, body))
            out = self.handlers.get(name, _default)(target, body)
            result: PodResult = await out if inspect.isawaitable(out) else out
            self._results[ctl] = result
            return ExecResult(0, "loom-launched\n", "")
        if "loom-running" in script:
            return ExecResult(0, f"loom-rc {self._results[ctl][0]}\n", "")
        if "kill -KILL" in script:
            return ExecResult(0, "", "")
        return ExecResult(0, self._results.pop(ctl)[1], "")
