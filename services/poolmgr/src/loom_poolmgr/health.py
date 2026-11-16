"""Talking to a pod's proxy as the gateway does (docs/phase1-m2-spec.md §5 "Bootstrap"
step 4, "Health rules").

Every call goes to `https://<ip>:<mapped proxy port>` with the manager's client
certificate (`Pki.client_ssl_context()`), `sni_hostname` = the pod id, and the pod's
`X-Loom-Pod-Token`:

- `GET /health`: the proxy health check (`proxy_health_failures` in a row take the pod
  out of routing);
- `GET /metrics`: the engine's Prometheus text, for drain (`vllm:num_requests_running`)
  and the progress rule (`vllm:generation_tokens_total` flat while requests run);
- `POST /v1/completions` with `max_tokens: 1`: the bootstrap probe.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import httpx

from loom_platform.pool import Replica

TOKEN_HEADER = "X-Loom-Pod-Token"
_SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)")


@dataclass(frozen=True)
class EngineMetrics:
    """The vLLM gauges and counters the manager reads (summed over label sets)."""

    requests_running: float
    requests_waiting: float
    generation_tokens_total: float


def parse_metrics(text: str) -> EngineMetrics:
    """Sum `vllm:num_requests_running`, `vllm:num_requests_waiting` and
    `vllm:generation_tokens_total` over their label sets. A metric that is absent is 0."""
    sums = {
        "vllm:num_requests_running": 0.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:generation_tokens_total": 0.0,
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if m is None or m.group("name") not in sums:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        if math.isfinite(value):
            sums[m.group("name")] += value
    return EngineMetrics(
        requests_running=sums["vllm:num_requests_running"],
        requests_waiting=sums["vllm:num_requests_waiting"],
        generation_tokens_total=sums["vllm:generation_tokens_total"],
    )


def _headers(replica: Replica) -> dict[str, str]:
    return {TOKEN_HEADER: replica.token}


def _ext(replica: Replica) -> dict[str, str]:
    return {"sni_hostname": replica.server_name}


async def proxy_health(client: httpx.AsyncClient, replica: Replica, *, timeout_s: float) -> bool:
    """Whether the proxy answers `GET /health` with 200."""
    try:
        r = await client.get(
            replica.url + "/health",
            headers=_headers(replica),
            extensions=_ext(replica),
            timeout=timeout_s,
        )
    except httpx.HTTPError:
        return False
    return r.status_code == 200


async def engine_metrics(
    client: httpx.AsyncClient, replica: Replica, *, timeout_s: float
) -> EngineMetrics:
    """The engine's metrics through the proxy. Raises `httpx.HTTPError` on failure."""
    r = await client.get(
        replica.url + "/metrics",
        headers=_headers(replica),
        extensions=_ext(replica),
        timeout=timeout_s,
    )
    r.raise_for_status()
    return parse_metrics(r.text)


class ProbeFailed(RuntimeError):
    """The one-token probe did not answer a 200 with a choice."""


async def probe(
    client: httpx.AsyncClient, replica: Replica, served_model: str, *, timeout_s: float
) -> None:
    """A one-token completion through the proxy. Raises on anything but a 200 with a
    choice."""
    r = await client.post(
        replica.url + "/v1/completions",
        json={"model": served_model, "prompt": "ping", "max_tokens": 1},
        headers=_headers(replica),
        extensions=_ext(replica),
        timeout=timeout_s,
    )
    if r.status_code != 200:
        raise ProbeFailed(f"probe answered {r.status_code}")
    try:
        choices = r.json().get("choices")
    except ValueError:
        choices = None
    if not choices:
        raise ProbeFailed("probe answered no choice")
