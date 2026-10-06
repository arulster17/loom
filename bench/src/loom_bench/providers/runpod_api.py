"""RunPod REST and GraphQL client, and loading the account API key.

The account key comes from `RUNPOD_API_KEY`, falling back to the macOS Keychain
(service `RUNPOD_API_KEY`). It is held as a `SecretStr`, sent only in the
`Authorization` header (never a URL parameter), and redacted from every error.

Pods are created and read through REST (`rest.runpod.io/v1`). Terminating goes
through REST `DELETE /pods/{id}` and falls back to GraphQL `podTerminate`, which is
also the only route the pod-scoped key RunPod injects into a pod may use. Pods are
never stopped: a stopped pod keeps billing for its disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from decimal import ROUND_CEILING, Decimal
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import SecretStr

from loom_bench.money import Micros

REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
KEY_ENV = "RUNPOD_API_KEY"
KEYCHAIN_SERVICE = "RUNPOD_API_KEY"
POD_ID_RE = re.compile(r"^[a-z0-9]{6,32}$")
# Idempotent requests are retried on these statuses and on transport errors.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
RETRIES = 3
TERMINATE_MUTATION = "mutation Terminate($podId: String!) { podTerminate(input: {podId: $podId}) }"

log = logging.getLogger(__name__)


class RunpodApiError(RuntimeError):
    def __init__(self, method: str, path: str, status: int | None, detail: str) -> None:
        where = f"{method} {path}"
        super().__init__(f"RunPod {where} failed ({status or 'no response'}): {detail[:500]}")
        self.method = method
        self.path = path
        self.status = status


def _keychain_lookup() -> str | None:
    """The key from the macOS login Keychain, or None (also off macOS)."""
    if sys.platform != "darwin" or shutil.which("security") is None:
        return None
    user = os.environ.get("USER", "")
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-a", user, "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = proc.stdout.strip()
    return value if proc.returncode == 0 and value else None


def load_runpod_api_key(
    env: Mapping[str, str] | None = None,
    *,
    keychain: Callable[[], str | None] | None = None,
) -> SecretStr | None:
    """The account API key from `env` (default `os.environ`), then the Keychain."""
    env = os.environ if env is None else env
    value = env.get(KEY_ENV, "").strip()
    if not value:
        value = ((keychain or _keychain_lookup)() or "").strip()
    return SecretStr(value) if value else None


def usd_to_micros(value: float | int | str | Decimal) -> Micros:
    """A dollar amount from the API (e.g. `costPerHr` 1.09) in micros, rounded up.

    Parsed through `str` so 1.09 is exactly 1_090_000, not a binary-float neighbour.
    """
    micros = (Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING)
    if micros < 0:
        raise ValueError(f"negative amount {value!r}")
    return int(micros)


def check_pod_id(pod_id: str) -> str:
    if not POD_ID_RE.match(pod_id):
        raise ValueError(f"not a RunPod pod id: {pod_id!r}")
    return pod_id


def public_ssh(pod: Mapping[str, Any]) -> tuple[str, int] | None:
    """(public IP, port) mapped to the pod's 22/tcp, or None until RunPod assigns them."""
    ip = pod.get("publicIp") or ""
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") if isinstance(mappings, Mapping) else None
    if not ip or port is None:
        return None
    return str(ip), int(port)


class RunpodApi:
    """Synchronous client; async callers wrap calls in `asyncio.to_thread`."""

    def __init__(
        self,
        api_key: SecretStr,
        *,
        rest_url: str = REST_URL,
        graphql_url: str = GRAPHQL_URL,
        transport: httpx.BaseTransport | None = None,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("empty RunPod API key")
        self._key = api_key
        self.rest_url = rest_url.rstrip("/")
        self.graphql_url = graphql_url
        self._sleep = sleep
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "User-Agent": "loom-bench",
            },
            transport=transport,
            timeout=timeout_s,
        )

    def __repr__(self) -> str:
        return f"RunpodApi(rest_url={self.rest_url!r}, graphql_url={self.graphql_url!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _redact(self, text: str) -> str:
        key = self._key.get_secret_value()
        return text.replace(key, "***") if key else text

    def _request(
        self,
        method: str,
        url: str,
        path: str,
        *,
        body: Any = None,
        ok_404: bool = False,
        retry: bool,
    ) -> Any:
        attempts = RETRIES if retry else 1
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.request(method, url, json=body)
            except httpx.TransportError as e:
                if attempt == attempts:
                    raise RunpodApiError(method, path, None, self._redact(repr(e))) from None
                self._sleep(2.0**attempt)
                continue
            if resp.status_code in RETRY_STATUSES and attempt < attempts:
                self._sleep(2.0**attempt)
                continue
            if resp.status_code == 404 and ok_404:
                return None
            if resp.status_code >= 400:
                raise RunpodApiError(method, path, resp.status_code, self._redact(resp.text))
            if not resp.content:
                return None
            try:
                return resp.json()
            except json.JSONDecodeError:
                return resp.text
        raise AssertionError("unreachable")

    def _rest(
        self, method: str, path: str, *, body: Any = None, ok_404: bool = False, retry: bool
    ) -> Any:
        return self._request(
            method, self.rest_url + path, path, body=body, ok_404=ok_404, retry=retry
        )

    def graphql(self, query: str, variables: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The `data` of a GraphQL call; GraphQL `errors` raise."""
        body: dict[str, Any] = {"query": query}
        if variables:
            body["variables"] = dict(variables)
        out = self._request("POST", self.graphql_url, "graphql", body=body, retry=False)
        if not isinstance(out, dict):
            raise RunpodApiError("POST", "graphql", 200, "non-JSON response")
        if out.get("errors"):
            messages = "; ".join(str(e.get("message", e)) for e in out["errors"])
            raise RunpodApiError("POST", "graphql", 200, self._redact(messages))
        data = out.get("data")
        return data if isinstance(data, dict) else {}

    def create_pod(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """POST /pods. Not retried: the caller adopts or deletes a pod that a failed
        request may still have created (see `find_pods_by_name`)."""
        out = self._rest("POST", "/pods", body=dict(body), retry=False)
        if not isinstance(out, dict) or not out.get("id"):
            raise RunpodApiError("POST", "/pods", 200, "response has no pod id")
        return out

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        """GET /pods/{id}; None once the pod is gone."""
        out = self._rest("GET", f"/pods/{check_pod_id(pod_id)}", ok_404=True, retry=True)
        return out if isinstance(out, dict) else None

    def list_pods(self) -> list[dict[str, Any]]:
        out = self._rest("GET", "/pods", retry=True)
        if not isinstance(out, list):
            raise RunpodApiError("GET", "/pods", 200, "expected a list of pods")
        return [p for p in out if isinstance(p, dict)]

    def find_pods_by_name(self, name: str) -> list[dict[str, Any]]:
        return [p for p in self.list_pods() if p.get("name") == name]

    def delete_pod(self, pod_id: str) -> None:
        """DELETE /pods/{id} (terminate). A pod that is already gone is fine."""
        self._rest("DELETE", f"/pods/{check_pod_id(pod_id)}", ok_404=True, retry=True)

    def terminate_pod(self, pod_id: str) -> None:
        """Terminate via REST, falling back to GraphQL `podTerminate` when REST
        refuses the key (as it does for pod-scoped keys)."""
        try:
            self.delete_pod(pod_id)
            return
        except RunpodApiError as e:
            if e.status not in (401, 403):
                raise
            log.info("REST delete of %s refused (%s); using GraphQL podTerminate", pod_id, e.status)
        self.graphql(TERMINATE_MUTATION, {"podId": check_pod_id(pod_id)})
