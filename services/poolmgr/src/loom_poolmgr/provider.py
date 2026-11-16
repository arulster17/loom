"""Where serving pods come from (docs/phase1-m2-spec.md §11): the `ServingProvider`
protocol and its types.

Implementations: `loom_poolmgr.runpod.RunpodServing` (RunPod's API; with `FakeRunpod`'s
transport in tests) and `loom_poolmgr.local.LocalServing` (compose: each "pod" is a mock
engine plus a `loom-podproxy` subprocess pair, §9). The create, health, drain, idle and
pool-hash code in the manager is the same for both; only the provider and the pod exec
differ.

A provider knows nothing about the manager's rows or states. It creates a pod under a
name the manager chose, lists the `loom-serve-*` pods it can see, terminates by id, and
says where a pod's proxy and SSH ports are.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from loom_bench.registry import ModelSpec
from loom_platform.serving import ServedModel

POD_NAME_PREFIX = "loom-serve-"
# `loom-serve-<model>-<created epoch>-<6 hex>`; the model id is kept whole and parsed
# from the right (the reaper's rule, §6, uses the same pattern).
POD_NAME_RE = re.compile(
    r"^loom-serve-(?P<model>[a-z0-9.-]+)-(?P<epoch>\d{10})-(?P<hex>[0-9a-f]{6})$"
)

# RunPod's desiredStatus values that mean the pod can no longer serve (bench `pod_loss`).
GONE_STATUSES = frozenset({"EXITED", "TERMINATED"})


def pod_name(model_id: str, created_epoch: float, suffix: str | None = None) -> str:
    """The name of a new serving pod; `suffix` (6 hex) is random unless given."""
    suffix = secrets.token_hex(3) if suffix is None else suffix
    name = f"{POD_NAME_PREFIX}{model_id}-{int(created_epoch):010d}-{suffix}"
    if not POD_NAME_RE.match(name):
        raise ValueError(f"not a serving pod name: {name!r}")
    return name


def parse_pod_name(name: str) -> tuple[str, int] | None:
    """(model id, created epoch) of a serving pod name, or None for any other name."""
    m = POD_NAME_RE.match(name)
    return None if m is None else (m.group("model"), int(m.group("epoch")))


@dataclass(frozen=True)
class Deployment:
    """Everything a provider needs to create one replica of a model: the serving config
    entry, the registry row (GPU type, engine, weights) and what the placement adds."""

    model_id: str
    served: ServedModel
    spec: ModelSpec
    gpu_type_id: str  # RunPod id, from RUNPOD_GPU_TYPE_IDS[spec.hardware.gpu]
    gpu_count: int  # spec.hardware.gpus_per_replica
    container_disk_gb: int  # placement.container_disk_gb[model_id]
    # bench/prices.yaml's price of spec.hardware.instance_types.runpod; None for local/fake
    # providers (no price check).
    listed_price_per_hr_micros: int | None
    # The container disk's price per hour (bench/prices.yaml storage), added to RunPod's
    # costPerHr for spend; 0 when unknown.
    disk_per_hr_micros: int = 0


@dataclass(frozen=True)
class Endpoints:
    """Where the manager reaches a pod: its proxy (mTLS) and its sshd."""

    ip: str
    proxy_port: int
    ssh_port: int


@dataclass(frozen=True)
class PodInfo:
    """One pod as the provider lists it."""

    pod_id: str
    name: str
    # RunPod's desiredStatus: CREATED, RUNNING, EXITED, TERMINATED (local: RUNNING/EXITED).
    status: str
    # The pod's env as the provider reports it (LOOM_MANAGED, LOOM_ROLE, LOOM_MODEL);
    # never holds a secret (the manager puts none in a pod's env).
    env: Mapping[str, str] = field(default_factory=dict)
    cost_per_hr_micros: int | None = None
    gpu_type_id: str | None = None
    gpu_count: int | None = None
    data_center: str | None = None
    # The provider's own record (RunPod's pod JSON: publicIp, portMappings, ...).
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def managed(self) -> bool:
        """A pod the manager may adopt or terminate as an orphan: an exact serving name
        and `LOOM_MANAGED=true`."""
        return POD_NAME_RE.match(self.name) is not None and self.env.get("LOOM_MANAGED") == "true"

    @property
    def gone(self) -> bool:
        return self.status in GONE_STATUSES


class ProviderError(RuntimeError):
    """A provider call failed."""


class NoCapacity(ProviderError):
    """No GPU of the type in any allowed datacenter (RunPod's no-capacity answer, or no
    US datacenter listing stock). Nothing was created."""


class CreateFailed(ProviderError):
    """A create failed. `ambiguous`: it failed without a clear answer (timeout, 5xx), so
    a pod may exist under its name; `cleaned` lists the pods found by that name and
    terminated before this was raised. An ambiguous create with nothing cleaned may still
    surface a pod later: the caller keeps the name accounted for until a later listing
    shows no pod with it (spec §11 "Failure and restart rules")."""

    def __init__(
        self, message: str, *, ambiguous: bool = False, cleaned: tuple[str, ...] = ()
    ) -> None:
        super().__init__(message)
        self.ambiguous = ambiguous
        self.cleaned = cleaned


class ServingProvider(Protocol):
    """Async; every call may raise `ProviderError`."""

    async def create(self, model: Deployment, name: str, env: dict[str, str]) -> PodInfo:
        """Create one pod named `name` with `env` (LOOM_MANAGED, LOOM_ROLE, LOOM_MODEL,
        LOOM_LEASE_EPOCH, LOOM_SSH_PUBKEY: no secrets). Raises `NoCapacity` or
        `CreateFailed`."""
        ...

    async def list(self) -> list[PodInfo]:
        """Every `loom-serve-*` pod the provider can see (no other pod)."""
        ...

    async def get(self, pod_id: str) -> PodInfo | None:
        """One pod by id (RunPod's GET /pods/{id}), or None once it is gone."""
        ...

    async def terminate(self, pod_id: str) -> None:
        """Terminate (never stop). Idempotent: an unknown or gone pod (404) is success."""
        ...

    def endpoints(self, pod: PodInfo) -> Endpoints | None:
        """The pod's public IP and mapped proxy and SSH ports, or None until assigned."""
        ...
