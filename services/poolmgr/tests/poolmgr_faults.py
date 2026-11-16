"""Extra fault injection for the manager's failure and restart rules (docs/phase1-m2-spec.md
§11 "Failure and restart rules"), layered on poolmgr_fakes without changing it: each helper
wraps one fake's instance method.

- `RunpodFaults(env.runpod)`: RunPod's pod listing lags or fails; a create makes the pod
  and then raises something that is not an HTTP error.
- `metrics_fail(engine)`: the engine's /metrics answers 500 (through the real proxy).
- `BootstrapSpy(env, model)`: the pool hash as it was right before and right after each
  serve_bootstrap (re-key) the fake pods receive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
from poolmgr_fakes import FakeEngine, FakePod, FakeServing, FakeServingRunpod
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from loom_bench.providers.runpod_fake import REST
from loom_bench.providers.runpod_ssh import ExecResult
from loom_platform.pool import PoolState

_REST_PATH = httpx.URL(REST).path
_REST_HOST = httpx.URL(REST).host


class RunpodFaults:
    """Listing faults on a `FakeServingRunpod`:

    - `lag`: each pod created from now on is missing from its first `lag` listings
      (GET /pods), as RunPod's listing can trail a create; GET /pods/{id} sees it;
    - `hidden[pod_id]`: listings that still leave the pod out (negative: all of them);
    - `listing_fails`: GET /pods answers 503 (RunpodApi retries, then raises);
    - `raise_after_create`: that many creates make the pod, then raise a RuntimeError
      (not an HTTP error) out of the API call.
    """

    def __init__(self, runpod: FakeServingRunpod) -> None:
        self.runpod = runpod
        self.lag = 0
        self.hidden: dict[str, int] = {}
        self.listing_fails = False
        self.raise_after_create = 0
        self.listings = 0  # listings answered
        self._orig: Callable[[httpx.Request], httpx.Response] = runpod._handle
        runpod._handle = self._handle  # type: ignore[method-assign]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        on_rest = request.url.host == _REST_HOST
        rest = request.url.path.removeprefix(_REST_PATH)
        if on_rest and request.method == "GET" and rest == "/pods":
            if self.listing_fails:
                self.runpod.requests.append(request)
                return httpx.Response(503, json={"error": "injected: listing unavailable"})
            response = self._orig(request)
            visible: list[Any] = []
            for pod in response.json():
                left = self.hidden.get(pod["id"], 0)
                if left == 0:
                    visible.append(pod)
                elif left > 0:
                    self.hidden[pod["id"]] = left - 1
            self.listings += 1
            return httpx.Response(200, json=visible)
        if on_rest and request.method == "POST" and rest == "/pods":
            before = set(self.runpod.pods)
            try:
                response = self._orig(request)
            finally:
                for pod_id in set(self.runpod.pods) - before:
                    if self.lag:
                        self.hidden[pod_id] = self.lag
            if response.status_code == 201 and self.raise_after_create > 0:
                self.raise_after_create -= 1
                raise RuntimeError("injected: an unexpected error after the pod was made")
            return response
        return self._orig(request)


def metrics_fail(engine: FakeEngine, fail: bool = True) -> None:
    """The engine's /metrics answers 500 (or, with `fail=False`, answers again)."""
    routes = engine.app.router.routes

    async def broken(_: Request) -> Response:
        return PlainTextResponse("injected: metrics unavailable", status_code=500)

    for route in list(routes):
        if getattr(route, "name", None) == "loom-test-broken-metrics":
            routes.remove(route)
    if fail:
        routes.insert(0, Route("/metrics", broken, name="loom-test-broken-metrics"))


class BootstrapSpy:
    """Records `model`'s pool hash right before and right after every serve_bootstrap a
    fake pod receives (a start or a re-key)."""

    def __init__(self, env: FakeServing, model: str) -> None:
        self.env = env
        self.model = model
        self.before: list[tuple[str, PoolState]] = []  # (pod id, hash) before delivery
        self.after: list[tuple[str, PoolState]] = []  # (pod id, hash) after the reload
        self._orig = env.pods._bootstrap
        env.pods._bootstrap = self._bootstrap  # type: ignore[method-assign]

    async def _bootstrap(self, pod: FakePod, script: str, files: dict[str, bytes]) -> ExecResult:
        self.before.append((pod.pod_id, await self.env.pool(self.model)))
        result = await self._orig(pod, script, files)
        self.after.append((pod.pod_id, await self.env.pool(self.model)))
        return result
