import copy
import itertools
import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr

from loom_bench.providers.runpod_api import RunpodApi

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

    def __init__(self, *, cost_per_hr: float = 1.09) -> None:
        self.pods: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        self.fail: dict[tuple[str, str], list[int]] = {}
        self.rest_delete_status: int | None = None
        self.cost_per_hr = cost_per_hr
        self.created_at = "2026-10-06 07:03:35.426 +0000 UTC"
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
            return httpx.Response(queued.pop(0), json={"error": "injected"})
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
                return httpx.Response(200, json=pod)
            if request.method == "DELETE":
                if self.rest_delete_status is not None:
                    return httpx.Response(self.rest_delete_status, json={"error": "forbidden"})
                if self.pods.pop(pod_id, None) is None:
                    return httpx.Response(404, json={"error": "pod not found"})
                return httpx.Response(200)
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})

    def _graphql(self, body: dict[str, Any]) -> httpx.Response:
        query = body.get("query", "")
        if "podTerminate" in query:
            pod_id = (body.get("variables") or {}).get("podId")
            self.pods.pop(str(pod_id), None)
            return httpx.Response(200, json={"data": {"podTerminate": None}})
        return httpx.Response(200, json={"errors": [{"message": "unsupported query"}]})
