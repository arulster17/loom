"""An in-memory RunPod for tests, importable by any package (the bench's RunPod tests and
the pool manager's, docs/phase1-m2-spec.md §10). Never used outside tests: it serves a
`RunpodApi` through `httpx.MockTransport` and makes no network calls."""

import copy
import itertools
import json
from importlib import resources
from typing import Any

import httpx
from pydantic import SecretStr

from loom_bench.providers.runpod_api import RunpodApi

FAKE_KEY = "rpa_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
REST = "https://rest.runpod.test/v1"
GRAPHQL = "https://api.runpod.test/graphql"


def create_response_fixture() -> dict[str, Any]:
    """Shape of a real `POST /pods` response (spike, 2026-10-06), ids and env redacted."""
    text = resources.files("loom_bench.providers").joinpath("runpod_fake_pod.json").read_text()
    data: dict[str, Any] = json.loads(text)
    return data


class FakeRunpod:
    """In-memory RunPod behind `httpx.MockTransport`.

    REST: POST/GET /pods, GET/DELETE /pods/{id}, GET /networkvolumes, DELETE
    /networkvolumes/{id}. GraphQL: `podTerminate`. Any
    `/stop` route fails the test: Loom must never stop a pod (stopped pods bill for
    disk). `fail` maps (method, path) to queued status codes served before the real
    handler; `rest_delete_status` makes REST DELETE refuse, as for a pod-scoped key.
    """

    def __init__(self, *, cost_per_hr: float = 1.09, gets_before_ssh: int | None = None) -> None:
        self.pods: dict[str, dict[str, Any]] = {}
        self.volumes: dict[str, dict[str, Any]] = {}
        self.requests: list[httpx.Request] = []
        # Each entry is a status, or (status, error message) to imitate a specific refusal.
        self.fail: dict[tuple[str, str], list[int | tuple[int, str]]] = {}
        self.rest_delete_status: int | None = None
        # GraphQL `dataCenters` availability, one entry per query (the last repeats): the
        # datacenter ids listing the GPU as available, or "error" for a failed query.
        # None: the query is unsupported (it errors), as before this fake knew it.
        self.stock: list[list[str] | str] | None = None
        self.stock_gpu = "NVIDIA L40S"
        self.stock_queries = 0
        self.cost_per_hr = cost_per_hr
        self.created_at = "2026-10-06 07:03:35.426 +0000 UTC"
        # GETs of a pod before RunPod assigns its public IP and 22/tcp mapping (None:
        # never, as in the create response fixture).
        self.gets_before_ssh = gets_before_ssh
        self._gets: dict[str, int] = {}
        self._ids = (f"fakepod{n:06d}" for n in itertools.count(1))
        self._volume_ids = (f"fakevol{n:06d}" for n in itertools.count(1))

    def api(self, **kw: Any) -> RunpodApi:
        return RunpodApi(
            SecretStr(FAKE_KEY),
            rest_url=REST,
            graphql_url=GRAPHQL,
            transport=httpx.MockTransport(self.handle),
            sleep=lambda s: None,
            **kw,
        )

    def _pod_template(self) -> dict[str, Any]:
        """A fresh pod record to fill in; a subclass may change its shape."""
        return create_response_fixture()

    def add_pod(self, **fields: Any) -> dict[str, Any]:
        pod = self._pod_template()
        pod.update(
            id=next(self._ids),
            env={},
            createdAt=self.created_at,
            costPerHr=self.cost_per_hr,
        )
        pod.update(fields)
        self.pods[pod["id"]] = pod
        return pod

    def add_volume(self, **fields: Any) -> dict[str, Any]:
        """A network volume shaped like REST `GET /networkvolumes` items."""
        vol: dict[str, Any] = {
            "id": next(self._volume_ids),
            "name": "volume",
            "size": 250,
            "dataCenterId": "US-MO-1",
        }
        vol.update(fields)
        self.volumes[vol["id"]] = vol
        return vol

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
                found = self.pods.get(pod_id)
                if found is None:
                    return httpx.Response(404, json={"error": "pod not found"})
                self._assign_ssh(found)
                return httpx.Response(200, json=found)
            if request.method == "DELETE":
                if self.rest_delete_status is not None:
                    return httpx.Response(self.rest_delete_status, json={"error": "forbidden"})
                if self.pods.pop(pod_id, None) is None:
                    return httpx.Response(404, json={"error": "pod not found"})
                return httpx.Response(200)
        if rest == "/networkvolumes" and request.method == "GET":
            return httpx.Response(200, json=list(self.volumes.values()))
        if rest.startswith("/networkvolumes/") and request.method == "DELETE":
            if self.volumes.pop(rest.removeprefix("/networkvolumes/"), None) is None:
                return httpx.Response(404, json={"error": "network volume not found"})
            return httpx.Response(204)
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
        if "dataCenters" in query and self.stock is not None:
            self.stock_queries += 1
            entry = self.stock.pop(0) if len(self.stock) > 1 else self.stock[0]
            if entry == "error":
                return httpx.Response(200, json={"errors": [{"message": "upstream timeout"}]})
            all_dcs = ["US-TX-3", "US-TX-4", "OC-AU-1", "EUR-IS-2", "EU-NL-1"]
            centers = [
                {
                    "id": dc,
                    "gpuAvailability": [
                        {
                            "gpuTypeId": self.stock_gpu,
                            "stockStatus": "Low" if dc in entry else None,
                            "available": dc in entry,
                        },
                        {
                            "gpuTypeId": "NVIDIA H100 80GB HBM3",
                            "stockStatus": "High",
                            "available": True,
                        },
                    ],
                }
                for dc in all_dcs
            ]
            return httpx.Response(200, json={"data": {"dataCenters": centers}})
        return httpx.Response(200, json={"errors": [{"message": "unsupported query"}]})
