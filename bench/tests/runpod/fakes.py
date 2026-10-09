import copy
import inspect
import itertools
import json
import re
import shlex
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from loom_bench.providers.aws_ssm import required_vars
from loom_bench.providers.runpod_api import RunpodApi
from loom_bench.providers.runpod_ssh import ExecResult, SshTarget

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_KEY = "rpa_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
REST = "https://rest.runpod.test/v1"
GRAPHQL = "https://api.runpod.test/graphql"


def create_response_fixture() -> dict[str, Any]:
    """Shape of a real `POST /pods` response (spike, 2026-10-06), ids and env redacted."""
    data: dict[str, Any] = json.loads((FIXTURES / "create_pod_response.json").read_text())
    return data


class FakeRunpod:
    """In-memory RunPod behind `httpx.MockTransport`.

    REST: POST/GET /pods, GET/DELETE /pods/{id}. GraphQL: `podTerminate`. Any
    `/stop` route fails the test: Loom must never stop a pod (stopped pods bill for
    disk). `fail` maps (method, path) to queued status codes served before the real
    handler; `rest_delete_status` makes REST DELETE refuse, as for a pod-scoped key.
    """

    def __init__(self, *, cost_per_hr: float = 1.09, gets_before_ssh: int | None = None) -> None:
        self.pods: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        # Each entry is a status, or (status, error message) to imitate a specific refusal.
        self.fail: dict[tuple[str, str], list[int | tuple[int, str]]] = {}
        self.rest_delete_status: int | None = None
        self.cost_per_hr = cost_per_hr
        self.created_at = "2026-10-06 07:03:35.426 +0000 UTC"
        # GETs of a pod before RunPod assigns its public IP and 22/tcp mapping (None:
        # never, as in the create response fixture).
        self.gets_before_ssh = gets_before_ssh
        self._gets: dict[str, int] = {}
        self._ids = (f"fakepod{n:06d}" for n in itertools.count(1))

    def api(self, **kw: Any) -> RunpodApi:
        return RunpodApi(
            SecretStr(FAKE_KEY),
            rest_url=REST,
            graphql_url=GRAPHQL,
            transport=httpx.MockTransport(self.handle),
            sleep=lambda s: None,
            **kw,
        )

    def add_pod(self, **fields: Any) -> dict[str, Any]:
        pod = create_response_fixture()
        pod.update(
            id=next(self._ids),
            env={},
            createdAt=self.created_at,
            costPerHr=self.cost_per_hr,
        )
        pod.update(fields)
        self.pods[pod["id"]] = pod
        return pod

    def bodies(self, method: str, path: str) -> list[Any]:
        return [
            json.loads(r.content) if r.content else None
            for r in self.requests
            if r.method == method and r.url.path == path
        ]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if "/stop" in path:
            raise AssertionError(f"Loom must never stop a pod: {request.method} {path}")
        queued = self.fail.get((request.method, path))
        if queued:
            status, error = queued.pop(0), "injected"
            if isinstance(status, tuple):
                status, error = status
            return httpx.Response(status, json={"error": error, "status": status})
        if request.url.host == httpx.URL(GRAPHQL).host:
            return self._graphql(json.loads(request.content))
        rest = path.removeprefix(httpx.URL(REST).path)
        if rest == "/pods" and request.method == "POST":
            body = json.loads(request.content)
            pod = self.add_pod(
                **{k: copy.deepcopy(v) for k, v in body.items() if k != "env"},
                env=dict(body.get("env") or {}),
            )
            return httpx.Response(201, json=pod)
        if rest == "/pods" and request.method == "GET":
            return httpx.Response(200, json=list(self.pods.values()))
        if rest.startswith("/pods/"):
            pod_id = rest.removeprefix("/pods/")
            if request.method == "GET":
                pod = self.pods.get(pod_id)
                if pod is None:
                    return httpx.Response(404, json={"error": "pod not found"})
                self._assign_ssh(pod)
                return httpx.Response(200, json=pod)
            if request.method == "DELETE":
                if self.rest_delete_status is not None:
                    return httpx.Response(self.rest_delete_status, json={"error": "forbidden"})
                if self.pods.pop(pod_id, None) is None:
                    return httpx.Response(404, json={"error": "pod not found"})
                return httpx.Response(200)
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})

    def _assign_ssh(self, pod: dict[str, Any]) -> None:
        if pod.get("publicIp") or self.gets_before_ssh is None:
            return
        n = self._gets[pod["id"]] = self._gets.get(pod["id"], 0) + 1
        if n > self.gets_before_ssh:
            index = list(self.pods).index(pod["id"]) + 1
            pod["publicIp"] = f"203.0.113.{index}"
            pod["portMappings"] = {"22": 28800 + index}

    def _graphql(self, body: dict[str, Any]) -> httpx.Response:
        query = body.get("query", "")
        if "podTerminate" in query:
            pod_id = (body.get("variables") or {}).get("podId")
            self.pods.pop(str(pod_id), None)
            return httpx.Response(200, json={"data": {"podTerminate": None}})
        return httpx.Response(200, json={"errors": [{"message": "unsupported query"}]})


# --- SSH into fake pods ------------------------------------------------------------

PodResult = tuple[int, str, str]  # (exit code, stdout, stderr) of one pod script
Handler = Callable[[SshTarget, str], PodResult | Awaitable[PodResult]]
TEMPLATES = ("pod_start", "start_engine", "stop_engine", "run_job")
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
    """An array variable as rendered into a pod script (`NAME=('a' 'b')`)."""
    m = re.search(rf"^{name}=\((.*)\)$", script, re.MULTILINE)
    assert m, name
    return shlex.split(m.group(1))


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
