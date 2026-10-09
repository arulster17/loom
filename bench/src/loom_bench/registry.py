"""Model registry: `config/models.yaml`, validated at load time.

The registry is the single source of truth for what Loom can serve and how.
Adding a model is a YAML change; every rule below is enforced on load.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Hashable
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_YAML = REPO_ROOT / "config" / "models.yaml"
MODELS_YAML_ENV = "LOOM_MODELS_YAML"

# Money in config files is integer micro-dollars; strict so a float like 1.86 is an error.
MicrosField = Annotated[int, Field(strict=True, ge=0)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
GitSha = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
# Digest-pinned image reference: repo@sha256:<64 hex>. Tags are rejected because they move.
PinnedImage = Annotated[
    str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._/-]*[a-z0-9]@sha256:[0-9a-f]{64}$")
]
Cloud = Literal["aws", "gcp", "runpod"]
# Serving precision of the weights.
Quantization = Literal["none", "fp8", "awq", "gptq", "w4a16", "w8a8", "fp4", "nvfp4"]
# `quantization_config.quant_method` in a pre-quantized checkpoint's config.json.
QuantMethod = Literal["fp8", "compressed-tensors", "awq", "gptq", "modelopt"]
KvCacheDtype = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
Status = Literal["enabled", "preview", "disabled"]

# Serving precisions each checkpoint format provides. Unquantized weights (None) are
# served as they are or quantized to FP8 on load; the others are served as stored.
CHECKPOINT_PRECISIONS: dict[QuantMethod | None, frozenset[Quantization]] = {
    None: frozenset({"none", "fp8"}),
    "fp8": frozenset({"fp8"}),
    "compressed-tensors": frozenset({"fp8", "w8a8", "w4a16", "nvfp4"}),
    "awq": frozenset({"awq"}),
    "gptq": frozenset({"gptq"}),
    "modelopt": frozenset({"fp8", "nvfp4"}),
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of keeping the last one."""


def _construct_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    seen: set[Hashable] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def read_yaml(path: Path) -> Any:
    """Parse a config file with safe types only and no silently overwritten keys."""
    with path.open(encoding="utf-8") as f:
        return yaml.load(f, Loader=_UniqueKeyLoader)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrustRemoteCodeReview(StrictModel):
    reviewer: NonEmptyStr
    date: dt.date
    notes: NonEmptyStr


class HFSource(StrictModel):
    repo: Annotated[str, StringConstraints(pattern=r"^[\w.-]+/[\w.-]+$")]
    revision: GitSha
    license: NonEmptyStr
    gated: Literal[False, "auto", "manual"]
    size_bytes: Annotated[int, Field(strict=True, gt=0)]
    quant_method: QuantMethod | None = None  # None: unquantized (BF16/FP16) weights
    trust_remote_code: bool = False
    trust_remote_code_review: TrustRemoteCodeReview | None = None

    @model_validator(mode="after")
    def _remote_code_needs_review(self) -> Self:
        if self.trust_remote_code and self.trust_remote_code_review is None:
            raise ValueError("trust_remote_code: true requires trust_remote_code_review")
        return self


class Engine(StrictModel):
    name: Literal["vllm", "sglang"]
    version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+([.+-][\w.]+)?$")]
    image: PinnedImage
    args: dict[str, Any] = Field(default_factory=dict)
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class InstanceTypes(StrictModel):
    aws: NonEmptyStr | None = None
    gcp: NonEmptyStr | None = None
    runpod: NonEmptyStr | None = None  # a `bench/prices.yaml` runpod instance, e.g. l40s-x1


class Hardware(StrictModel):
    gpu: NonEmptyStr
    gpus_per_replica: Annotated[int, Field(ge=1)]
    nodes_per_replica: Annotated[int, Field(ge=1)] = 1
    instance_types: InstanceTypes

    @model_validator(mode="after")
    def _single_node_only(self) -> Self:
        if self.nodes_per_replica != 1:
            raise ValueError("nodes_per_replica > 1 is reserved; multi-node serving is not built")
        return self


class Parallelism(StrictModel):
    tp: Annotated[int, Field(ge=1)] = 1
    pp: Annotated[int, Field(ge=1)] = 1
    ep: Annotated[int, Field(ge=1)] = 1


class Pricing(StrictModel):
    """Public price in micro-dollars per 1M tokens."""

    input_per_mtok: MicrosField
    output_per_mtok: MicrosField
    cached_input_per_mtok: MicrosField

    @model_validator(mode="after")
    def _cached_not_above_input(self) -> Self:
        if self.cached_input_per_mtok > self.input_per_mtok:
            raise ValueError("cached_input_per_mtok must be <= input_per_mtok")
        return self


class Scaling(StrictModel):
    min_replicas: Annotated[int, Field(ge=0)]
    max_replicas: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _max_at_least_min(self) -> Self:
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")
        return self


class Capabilities(StrictModel):
    tools: bool = False
    json_schema: bool = False
    vision: bool = False
    reasoning: bool = False


class ModelSpec(StrictModel):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9.-]*[a-z0-9]$")]
    display_name: NonEmptyStr
    hf: HFSource
    engine: Engine
    hardware: Hardware
    parallelism: Parallelism
    max_context: Annotated[int, Field(ge=1)]
    quantization: Quantization
    kv_cache_dtype: KvCacheDtype = "auto"
    pricing: Pricing | None = None
    scaling: Scaling
    clouds: Annotated[list[Cloud], Field(min_length=1)]
    capabilities: Capabilities
    # Tool-call parser per engine (vLLM `--tool-call-parser`, SGLang `--tool-call-parser`).
    # Model-level, not under `engine`, so it survives a variant switching engines.
    # `capabilities.tools` needs one: without it the engine rejects `tool_choice: auto`.
    tool_call_parsers: dict[Literal["vllm", "sglang"], NonEmptyStr] = Field(default_factory=dict)
    routing_tier: Annotated[int, Field(ge=0)]
    status: Status
    # The registry entry this one is a quantization of: the same model served at another
    # precision (its own labeled row; the base stays the quality reference). Providers
    # list prices per model, not per checkpoint, so the competitiveness view compares
    # this row with the base entry's competitor listings (`Registry.market_model_id`).
    # Excluded from dumps: it says nothing about how the model is served, and the spec
    # dump is part of every config hash, which must not move for it.
    base_model: str | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        hw, par = self.hardware, self.parallelism
        gpus = hw.gpus_per_replica * hw.nodes_per_replica
        if gpus != par.tp * par.pp:
            raise ValueError(
                f"gpus_per_replica * nodes_per_replica ({gpus}) must equal tp * pp "
                f"({par.tp * par.pp})"
            )
        if (par.tp * par.pp) % par.ep != 0:
            raise ValueError(f"ep ({par.ep}) must divide tp * pp ({par.tp * par.pp})")
        if len(set(self.clouds)) != len(self.clouds):
            raise ValueError("clouds must not repeat")
        missing = [c for c in self.clouds if getattr(hw.instance_types, c) is None]
        if missing:
            raise ValueError(f"hardware.instance_types missing for clouds {missing}")
        allowed = CHECKPOINT_PRECISIONS[self.hf.quant_method]
        if self.quantization not in allowed:
            raise ValueError(
                f"quantization {self.quantization!r} cannot be served from "
                f"{self.hf.quant_method or 'unquantized'} weights "
                f"(hf.quant_method {self.hf.quant_method!r} allows {sorted(allowed)})"
            )
        if self.status == "enabled" and self.pricing is None:
            raise ValueError("status: enabled requires pricing")
        return self


class Registry(StrictModel):
    models: list[ModelSpec]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [m.id for m in self.models]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate model ids: {dupes}")
        # Checked here, not on ModelSpec, so model snapshots stored before the field
        # existed still load; rendering a launch checks it again for variants.
        for m in self.models:
            if m.capabilities.tools and m.engine.name not in m.tool_call_parsers:
                raise ValueError(
                    f"{m.id}: capabilities.tools needs tool_call_parsers.{m.engine.name} "
                    "(the engine rejects tool_choice: auto without a parser)"
                )
        by_id = {m.id: m for m in self.models}
        for m in self.models:
            if m.base_model is None:
                continue
            base = by_id.get(m.base_model)
            if base is None or base.id == m.id:
                raise ValueError(f"{m.id}: base_model {m.base_model!r} is not another entry")
            if base.base_model is not None:
                raise ValueError(
                    f"{m.id}: base_model {base.id!r} has a base_model of its own; name the "
                    f"root entry ({base.base_model!r})"
                )
            if base.quantization == m.quantization:
                raise ValueError(
                    f"{m.id}: base_model {base.id!r} is served at the same precision "
                    f"({m.quantization}); base_model names the entry this one quantizes"
                )
        return self

    def get(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"unknown model id {model_id!r}")

    def market_model_id(self, model_id: str) -> str:
        """The registry id whose public list prices apply to `model_id`: its base model
        for a quantized entry (providers price the model, whatever precision they serve
        it at), else the id itself."""
        return self.get(model_id).base_model or model_id


def load_registry(path: Path | str | None = None) -> Registry:
    """Load `config/models.yaml` (or `$LOOM_MODELS_YAML`, or `path`)."""
    if path is None:
        path = os.environ.get(MODELS_YAML_ENV) or DEFAULT_MODELS_YAML
    return Registry.model_validate(read_yaml(Path(path)))
