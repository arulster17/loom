"""`ServingProvider` on RunPod (docs/phase1-m2-spec.md §5 "Create", §10).

`RunpodServing` wraps the bench's synchronous `RunpodApi` (built with
`user_agent="loom-poolmgr"`, called through `asyncio.to_thread`). The create body is
the bench's `pod_body` shape with `ports: ["22/tcp", "<proxy port>/tcp"]`, no HF secret
(both checkpoints are ungated), `dataCenterIds` = the datacenters listing stock
(`gpu_stock`) whose id starts with a configured prefix, and the env the manager passes.
The ambiguous-create cleanup follows the bench's `RunpodProvider._create`: a create that
fails without a clear answer (timeout, 5xx) lists pods by name and terminates any it
finds. The price check (`max_price_ratio`) is the manager's, on `PodInfo`.

In tests the API is `FakeRunpod().api(user_agent="loom-poolmgr")`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loom_bench.providers.runpod_api import RunpodApi
from loom_platform.serving import Network, Placement
from loom_poolmgr.provider import Deployment, Endpoints, PodInfo

USER_AGENT = "loom-poolmgr"
SSH_PORT = 22


def pod_body(
    model: Deployment,
    name: str,
    env: Mapping[str, str],
    *,
    data_center_ids: list[str],
    placement: Placement,
    network: Network,
    start_command: str,
) -> dict[str, Any]:
    """The POST /pods body. `start_command` is the rendered serve_start script
    (`dockerStartCmd` runs it with `bash -c`); `env` holds no secret."""
    raise NotImplementedError("phase B")


def pod_info(pod: Mapping[str, Any]) -> PodInfo:
    """A RunPod pod JSON (REST) as `PodInfo` (`raw` keeps the JSON)."""
    raise NotImplementedError("phase B")


class RunpodServing:
    """`ServingProvider` for `provider: runpod`."""

    def __init__(
        self,
        api: RunpodApi,
        *,
        placement: Placement,
        network: Network,
        start_command: str,
    ) -> None:
        self.api = api
        self.placement = placement
        self.network = network
        self.start_command = start_command

    async def create(self, model: Deployment, name: str, env: dict[str, str]) -> PodInfo:
        raise NotImplementedError("phase B")

    async def list(self) -> list[PodInfo]:
        raise NotImplementedError("phase B")

    async def terminate(self, pod_id: str) -> None:
        raise NotImplementedError("phase B")

    def endpoints(self, pod: PodInfo) -> Endpoints | None:
        raise NotImplementedError("phase B")
