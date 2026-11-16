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

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from loom_bench.providers.runpod_api import (
    TERMINATE_MUTATION,
    RunpodApi,
    RunpodApiError,
    check_pod_id,
    public_port,
    usd_to_micros,
)
from loom_platform.serving import Network, Placement
from loom_poolmgr.provider import (
    POD_NAME_PREFIX,
    CreateFailed,
    Deployment,
    Endpoints,
    NoCapacity,
    PodInfo,
    ProviderError,
)

USER_AGENT = "loom-poolmgr"
SSH_PORT = 22
CLOUD_TYPE = "SECURE"

log = logging.getLogger(__name__)


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
    return {
        "name": name,
        "imageName": model.spec.engine.image,
        "cloudType": CLOUD_TYPE,
        "computeType": "GPU",
        "interruptible": False,
        "gpuTypeIds": [model.gpu_type_id],
        "gpuCount": model.gpu_count,
        "allowedCudaVersions": list(placement.allowed_cuda_versions),
        "containerDiskInGb": model.container_disk_gb,
        # Explicit: the API's default is a 20 GB pod volume, billed while stopped too.
        "volumeInGb": 0,
        "ports": [f"{SSH_PORT}/tcp", f"{network.proxy_port}/tcp"],
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [start_command],
        "dataCenterIds": list(data_center_ids),
        "env": dict(env),
    }


def _micros(value: object) -> int | None:
    if value in (None, 0, "") or not isinstance(value, int | float | str):
        return None
    try:
        return usd_to_micros(value)
    except (ValueError, ArithmeticError):
        return None


def pod_info(pod: Mapping[str, Any]) -> PodInfo:
    """A RunPod pod JSON (REST) as `PodInfo` (`raw` keeps the JSON)."""
    env = pod.get("env")
    raw_machine = pod.get("machine")
    machine: Mapping[str, Any] = raw_machine if isinstance(raw_machine, Mapping) else {}
    gpu_ids = pod.get("gpuTypeIds")
    gpu_type = machine.get("gpuTypeId") or (
        gpu_ids[0] if isinstance(gpu_ids, list) and gpu_ids else None
    )
    count = pod.get("gpuCount")
    return PodInfo(
        pod_id=str(pod.get("id") or ""),
        name=str(pod.get("name") or ""),
        status=str(pod.get("desiredStatus") or ""),
        env={str(k): str(v) for k, v in env.items()} if isinstance(env, Mapping) else {},
        cost_per_hr_micros=_micros(pod.get("costPerHr")),
        gpu_type_id=str(gpu_type) if gpu_type else None,
        gpu_count=int(count) if isinstance(count, int) else None,
        data_center=str(machine.get("dataCenterId") or "").strip() or None,
        raw=dict(pod),
    )


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

    def _allowed(self, data_center: str) -> bool:
        return any(data_center.startswith(p) for p in self.placement.data_center_prefixes)

    def _create(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """POST /pods; when it fails without a clear refusal (no response, a 5xx other
        than no capacity), terminate any pod it made anyway (found by its unique name)."""
        try:
            return self.api.create_pod(body)
        except RunpodApiError as e:
            if e.no_capacity:
                raise NoCapacity(f"RunPod has no {body['gpuTypeIds'][0]} free") from None
            if e.status is None or e.status >= 500:
                for pod in self.api.find_pods_by_name(str(body["name"])):
                    log.warning("terminating pod %s, made by an ambiguous create", pod.get("id"))
                    self._terminate(str(pod["id"]))
            raise CreateFailed(f"create {body['name']}: {e}") from None

    def _create_sync(self, model: Deployment, name: str, env: dict[str, str]) -> PodInfo:
        try:
            stocked = [dc for dc in self.api.gpu_stock(model.gpu_type_id) if self._allowed(dc)]
        except RunpodApiError as e:
            raise CreateFailed(f"RunPod availability query failed: {e}") from None
        if not stocked:
            raise NoCapacity(
                f"no {'/'.join(self.placement.data_center_prefixes)} datacenter lists "
                f"{model.gpu_type_id} in stock"
            )
        body = pod_body(
            model,
            name,
            env,
            data_center_ids=stocked,
            placement=self.placement,
            network=self.network,
            start_command=self.start_command,
        )
        try:
            pod = self._create(body)
        except RunpodApiError as e:  # the cleanup itself failed
            raise CreateFailed(f"create {name}: cleanup failed: {e}") from None
        info = pod_info(pod)
        if not info.pod_id:
            raise CreateFailed(f"create {name}: the answer has no pod id")
        if info.cost_per_hr_micros is None:
            try:
                fetched = self.api.get_pod(info.pod_id)
            except (RunpodApiError, ValueError):
                fetched = None
            if fetched is not None:
                info = pod_info(fetched)
        return info

    async def create(self, model: Deployment, name: str, env: dict[str, str]) -> PodInfo:
        return await asyncio.to_thread(self._create_sync, model, name, env)

    async def list(self) -> list[PodInfo]:
        try:
            pods = await asyncio.to_thread(self.api.list_pods)
        except RunpodApiError as e:
            raise ProviderError(f"RunPod pod listing failed: {e}") from None
        return [pod_info(p) for p in pods if str(p.get("name") or "").startswith(POD_NAME_PREFIX)]

    def _terminate(self, pod_id: str) -> None:
        check_pod_id(pod_id)
        try:
            self.api.terminate_pod(pod_id)
            return
        except RunpodApiError as e:
            log.warning("REST terminate of %s failed (%s); trying GraphQL", pod_id, e.status)
        self.api.graphql(TERMINATE_MUTATION, {"podId": pod_id})

    async def terminate(self, pod_id: str) -> None:
        try:
            await asyncio.to_thread(self._terminate, pod_id)
        except (RunpodApiError, ValueError) as e:
            raise ProviderError(f"terminate {pod_id}: {e}") from None

    def endpoints(self, pod: PodInfo) -> Endpoints | None:
        if pod.status != "RUNNING":
            return None
        ssh = public_port(pod.raw, SSH_PORT)
        proxy = public_port(pod.raw, self.network.proxy_port)
        if ssh is None or proxy is None or ssh[0] != proxy[0]:
            return None
        return Endpoints(ip=ssh[0], proxy_port=proxy[1], ssh_port=ssh[1])
