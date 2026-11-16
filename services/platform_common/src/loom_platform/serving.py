"""Serving deployment settings: `config/serving.yaml` (docs/phase1-m2-spec.md §4).

`config/models.yaml` stays the source of what is served and at what price; this file
says how the pool manager deploys it (provider, spend cap, placement, lifecycle and
health timings, per-model replica counts). It is kept out of the registry because the
registry's fields are part of every benchmark config hash.

Path: an explicit `load_serving(path)`, else `$LOOM_SERVING_YAML`, else the repo's
`config/serving.yaml`. Every section rejects unknown keys. A validation error is a
`ValueError` (Pydantic's `ValidationError` is one) whose message names the field.
"""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from loom_bench.engines import validate_engine_env
from loom_bench.registry import REPO_ROOT, Registry, load_registry, read_yaml
from loom_platform.pricing import servable

SERVING_YAML_ENV = "LOOM_SERVING_YAML"
DEFAULT_SERVING_YAML = REPO_ROOT / "config" / "serving.yaml"

Provider = Literal["runpod", "local", "fake"]
# Providers that run nothing on RunPod: no gateway egress allowlist needed.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"local", "fake"})

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
Fraction = Annotated[float, Field(gt=0, le=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Spend(_Strict):
    monthly_cap_usd: PositiveFloat
    # Alert once per month as month-to-date spend passes each fraction of the cap.
    alert_fractions: list[Fraction]
    # Refuse a pod priced above bench/prices.yaml x this.
    max_price_ratio: Annotated[float, Field(ge=1)]

    @field_validator("alert_fractions")
    @classmethod
    def _unique_fractions(cls, v: list[float]) -> list[float]:
        # Each fraction is a row key in serving_spend_alerts (month, fraction).
        dupes = sorted({x for x in v if v.count(x) > 1})
        if dupes:
            raise ValueError(f"alert_fractions must be unique, repeated: {dupes}")
        return v


class Placement(_Strict):
    data_center_prefixes: Annotated[list[NonEmptyStr], Field(min_length=1)]
    allowed_cuda_versions: Annotated[list[str], Field(min_length=1)]
    container_disk_gb: dict[str, PositiveInt]

    @field_validator("allowed_cuda_versions")
    @classmethod
    def _cuda_versions(cls, v: list[str]) -> list[str]:
        # The values RunPod's REST API accepts (imported here: loom_bench.experiment is
        # heavy and only this check needs it).
        from loom_bench.experiment import RunpodCudaVersion

        known = get_args(RunpodCudaVersion)
        bad = [x for x in v if x not in known]
        if bad:
            raise ValueError(f"not a RunPod CUDA version: {bad} (one of {list(known)})")
        return v


class Network(_Strict):
    # Source CIDRs the pod proxy allows (the gateway's and the manager's egress).
    gateway_egress_cidrs: list[str]
    proxy_port: Annotated[int, Field(gt=0, lt=65536)]

    @field_validator("gateway_egress_cidrs")
    @classmethod
    def _cidrs(cls, v: list[str]) -> list[str]:
        for cidr in v:
            ipaddress.ip_network(cidr)  # ValueError names the bad value
        return v


class Lifecycle(_Strict):
    reconcile_every_s: PositiveInt
    lease_ttl_s: PositiveInt
    lease_renew_every_s: PositiveInt
    drain_max_s: PositiveInt
    server_cert_days: PositiveInt
    server_cert_renew_before_s: PositiveInt
    max_replacements_per_hour: PositiveInt

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # A renewal must land before the lease runs out, or the watchdog kills the pod.
        if self.lease_renew_every_s >= self.lease_ttl_s:
            raise ValueError(
                f"lease_renew_every_s ({self.lease_renew_every_s}) must be < "
                f"lease_ttl_s ({self.lease_ttl_s})"
            )
        # Otherwise every certificate is due for renewal the moment it is issued.
        lifetime_s = self.server_cert_days * 86400
        if self.server_cert_renew_before_s >= lifetime_s:
            raise ValueError(
                f"server_cert_renew_before_s ({self.server_cert_renew_before_s}) must be < "
                f"server_cert_days x 86400 ({lifetime_s})"
            )
        return self


class Health(_Strict):
    pod_state_every_s: PositiveInt
    proxy_health_every_s: PositiveInt
    proxy_health_failures: PositiveInt
    unhealthy_grace_s: PositiveInt
    progress_stall_s: PositiveInt
    bootstrap_stall_s: PositiveInt
    bootstrap_timeout_s: PositiveInt

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        # A stall limit above the timeout could never fire first.
        if self.bootstrap_stall_s > self.bootstrap_timeout_s:
            raise ValueError(
                f"bootstrap_stall_s ({self.bootstrap_stall_s}) must be <= "
                f"bootstrap_timeout_s ({self.bootstrap_timeout_s})"
            )
        return self


class Capacity(_Strict):
    # Waits between create attempts after RunPod reports no capacity.
    backoff_s: Annotated[list[PositiveInt], Field(min_length=1)]
    unavailable_retry_after_s: PositiveInt


class ServedModel(_Strict):
    # 0 = on demand (started by a wake); >= 1 = always on.
    min_replicas: NonNegativeInt
    # Hard cap on the model's pods not terminated.
    max_replicas: PositiveInt
    # A ready pod with no demand this long is drained (0: never idle-stopped, as poolsim).
    idle_stop_s: NonNegativeInt
    # Retry-After of a request to the cold model.
    cold_start_estimate_s: PositiveInt
    # Added to the engine's environment (no secret names), on top of the registry row's.
    engine_env: dict[str, str] = Field(default_factory=dict)

    @field_validator("engine_env")
    @classmethod
    def _engine_env(cls, v: dict[str, str]) -> dict[str, str]:
        return validate_engine_env(v)

    @model_validator(mode="after")
    def _replicas(self) -> Self:
        if self.min_replicas > self.max_replicas:
            raise ValueError(
                f"min_replicas ({self.min_replicas}) > max_replicas ({self.max_replicas})"
            )
        return self


class ServingConfig(_Strict):
    version: Literal[1]
    provider: Provider
    spend: Spend
    placement: Placement
    network: Network
    lifecycle: Lifecycle
    health: Health
    capacity: Capacity
    # A servable model missing here is never started (its pool: unavailable,
    # not_configured).
    models: dict[str, ServedModel]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.provider not in LOCAL_PROVIDERS and not self.network.gateway_egress_cidrs:
            raise ValueError(
                f"network.gateway_egress_cidrs must not be empty with provider {self.provider}"
            )
        for model_id in self.models:
            if model_id not in self.placement.container_disk_gb:
                raise ValueError(f"placement.container_disk_gb has no entry for {model_id}")
        for model_id in self.placement.container_disk_gb:
            if model_id not in self.models:
                raise ValueError(f"placement.container_disk_gb.{model_id}: not in models")
        return self

    def check_registry(self, registry: Registry) -> None:
        """Every configured model is servable and has a RunPod instance type."""
        offered = {m.id for m in servable(registry)}
        known = {m.id for m in registry.models}
        for model_id in self.models:
            if model_id not in known:
                raise ValueError(f"models.{model_id}: not in the model registry")
            if model_id not in offered:
                raise ValueError(
                    f"models.{model_id}: not servable (the registry row has no pricing "
                    "or is disabled)"
                )
            if registry.get(model_id).hardware.instance_types.runpod is None:
                raise ValueError(f"models.{model_id}: no runpod entry in hardware.instance_types")


def serving_path(path: Path | str | None = None) -> Path:
    """The file `load_serving(path)` reads."""
    if path is None:
        path = os.environ.get(SERVING_YAML_ENV) or DEFAULT_SERVING_YAML
    return Path(path)


def load_serving(
    path: Path | str | None = None,
    *,
    registry: Registry | None = None,
    check_registry: bool = True,
) -> ServingConfig:
    """Load and validate `config/serving.yaml` (or `$LOOM_SERVING_YAML`, or `path`)
    against the model registry (`load_registry()` unless given). `check_registry=False`
    checks the file alone: the gateway reads only the cold-start estimates, and the
    registry it serves may differ (tests serve copies under other ids)."""
    cfg = ServingConfig.model_validate(read_yaml(serving_path(path)))
    if check_registry:
        cfg.check_registry(registry if registry is not None else load_registry())
    return cfg
