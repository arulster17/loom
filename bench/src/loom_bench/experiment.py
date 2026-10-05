"""Experiment specs (`bench/experiments/*.yaml`) and their expansion into cells.

A cell is one fully resolved serving configuration: the registry entry with a
variant's overrides and one sweep point applied, the rendered engine launch and
the hardware it runs on. Its `config_hash` identifies the setup across
experiments. Workloads, load points and repetitions are run inside a cell.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from loom_bench.cost import CostAllocation
from loom_bench.engines import render_launch
from loom_bench.jobs import TokenizerSpec
from loom_bench.loadgen.arrivals import parse_arrivals
from loom_bench.loadgen.base import LOAD_GENERATORS
from loom_bench.mock.config import MockConfig
from loom_bench.money import parse_usd
from loom_bench.provenance import canonical_json, config_hash
from loom_bench.providers.base import EngineLaunch
from loom_bench.records import LoadMode
from loom_bench.registry import REPO_ROOT, ModelSpec, Registry, read_yaml
from loom_bench.slo import Slo
from loom_bench.workloads import WorkloadProfile, load_profile

EXPERIMENTS_DIR = REPO_ROOT / "bench" / "experiments"

Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]*$")]


def _usd(value: Any) -> int:
    if isinstance(value, float):
        raise ValueError('write money as a string, e.g. "$1.50"')
    return parse_usd(value)


UsdMicros = Annotated[int, BeforeValidator(_usd), Field(ge=0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- providers ------------------------------------------------------------------


class MockProviderSpec(MockConfig):
    """Mock backend settings plus a simulated hourly price so cost math runs end to end."""

    kind: Literal["mock"]
    hourly_price: UsdMicros = 0

    def mock_config(self, overrides: Mapping[str, Any] | None = None) -> MockConfig:
        base = self.model_dump(exclude={"kind", "hourly_price"})
        return MockConfig.model_validate({**base, **(overrides or {})})


class LocalProviderSpec(_Strict):
    """An OpenAI-compatible endpoint you already run (any engine, or `bench mock-server`)."""

    kind: Literal["local"]
    base_url: str  # including the API prefix, e.g. http://127.0.0.1:8000/v1
    metrics_url: str | None = None
    engine: Literal["vllm", "sglang", "mock"]
    served_model: str
    tokenizer: Literal["hf", "simple"] = "hf"


class AwsEc2ProviderSpec(_Strict):
    kind: Literal["aws_ec2"]
    region: str = "us-east-1"
    instance_type: str | None = None  # default: the cell's hardware.instance_types.aws
    market: Literal["spot", "on_demand"] = "spot"
    disk_gb: PositiveInt = 200


ProviderSpec = Annotated[
    MockProviderSpec | LocalProviderSpec | AwsEc2ProviderSpec, Field(discriminator="kind")
]


# --- variants -------------------------------------------------------------------


class EnginePatch(_Strict):
    name: str | None = None
    version: str | None = None
    image: str | None = None
    # Merged over the registry args (null removes a key); reset when `name` switches engine.
    args: dict[str, Any] | None = None
    chat_template_kwargs: dict[str, Any] | None = None


class HFPatch(_Strict):
    """Point at a different checkpoint, e.g. a pre-quantized repo."""

    repo: str | None = None
    revision: str | None = None
    license: str | None = None
    gated: Literal[False, "auto", "manual"] | None = None
    size_bytes: int | None = None


class HardwarePatch(_Strict):
    gpu: str | None = None
    gpus_per_replica: int | None = None
    instance_types: dict[str, str] | None = None


class ParallelismPatch(_Strict):
    tp: int | None = None
    pp: int | None = None
    ep: int | None = None


class Variant(_Strict):
    """Named overrides of the registry entry; the patched spec is re-validated."""

    name: Slug
    engine: EnginePatch | None = None
    hf: HFPatch | None = None
    hardware: HardwarePatch | None = None
    parallelism: ParallelismPatch | None = None
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    max_context: int | None = None
    mock: dict[str, Any] = Field(default_factory=dict)  # MockConfig overrides, mock provider


class Sample(_Strict):
    """Random search: `random` points drawn without replacement from the sweep grid."""

    random: PositiveInt
    seed: int = 0


# --- workloads and load -----------------------------------------------------------

# Arrival kinds whose intensity is a single rate the sweep value replaces.
RATE_FIELDS = {"constant": "rate", "poisson": "rate", "gamma": "rate", "diurnal": "mean_rate"}


class LoadSearch(_Strict):
    """Bisect [lo, hi] for the highest load meeting the SLO (`slo.bisect_next_load`)."""

    lo: PositiveFloat
    hi: PositiveFloat
    rel_tol: PositiveFloat = 0.05
    max_points: Annotated[int, Field(ge=2)] = 8

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.lo >= self.hi:
            raise ValueError("search needs lo < hi")
        return self


class LoadSpec(_Strict):
    mode: LoadMode
    values: list[PositiveFloat] | None = None
    search: LoadSearch | None = None
    duration_s: PositiveFloat | None = None
    num_requests: PositiveInt | None = None
    warmup_s: NonNegativeFloat | None = None
    warmup_requests: Annotated[int, Field(ge=0)] | None = None
    arrival: dict[str, Any] | None = None  # open loop; `rate` comes from the load value
    request_timeout_s: PositiveFloat = 600.0
    drain_timeout_s: NonNegativeFloat = 60.0
    max_inflight: PositiveInt = 4096
    scrape_interval_s: PositiveFloat = 1.0

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.values is None) == (self.search is None):
            raise ValueError("give exactly one of load.values or load.search")
        if self.values is not None:
            if not self.values:
                raise ValueError("load.values must not be empty")
            if len(set(self.values)) != len(self.values):
                raise ValueError("load.values must not repeat")
        if self.mode is LoadMode.OPEN_LOOP:
            if self.duration_s is None or self.num_requests is not None:
                raise ValueError("open loop needs duration_s (num_requests is closed loop only)")
            if self.warmup_requests is not None:
                raise ValueError("open loop warms up by time: use warmup_s")
            if (self.warmup_s or 0.0) >= self.duration_s:
                raise ValueError("warmup_s must be shorter than duration_s (warmup is inside it)")
            kind = (self.arrival or {}).get("kind", "poisson")
            if kind not in RATE_FIELDS:
                raise ValueError(
                    f"arrival kind {kind!r} has no single rate to sweep; use one of "
                    f"{sorted(RATE_FIELDS)}"
                )
            if self.arrival and RATE_FIELDS[kind] in self.arrival:
                raise ValueError(f"arrival.{RATE_FIELDS[kind]} comes from the load value")
            self.arrival_for(1.0)
        else:
            if (self.duration_s is None) == (self.num_requests is None):
                raise ValueError("closed loop needs exactly one of duration_s or num_requests")
            if self.warmup_s is not None:
                raise ValueError("closed loop warms up by count: use warmup_requests")
            if self.arrival is not None:
                raise ValueError("arrival applies to open loop only")
            if self.values is not None and any(v != int(v) for v in self.values):
                raise ValueError("closed-loop values are concurrencies and must be integers")
        return self

    def arrival_for(self, rate: float) -> dict[str, Any]:
        """Resolved ArrivalSpec (as JSON) at `rate` requests/s."""
        template = dict(self.arrival or {"kind": "poisson"})
        template.setdefault("kind", "poisson")
        template[RATE_FIELDS[template["kind"]]] = rate
        return parse_arrivals(template).model_dump(mode="json")


class WorkloadEntry(_Strict):
    profile: str  # name in bench/workloads/ or a YAML path
    label: Slug | None = None  # defaults to the profile name; must be unique
    overrides: dict[str, Any] = Field(default_factory=dict)
    load: LoadSpec

    @property
    def name(self) -> str:
        return self.label or Path(self.profile).stem

    def resolve(self) -> WorkloadProfile:
        return load_profile(self.profile, self.overrides or None)


# --- budget, cost, quality --------------------------------------------------------


class BudgetSpec(_Strict):
    max_spend: UsdMicros
    ttl_minutes: PositiveFloat  # hard host lifetime; every host self-destructs at this age
    accrual_interval_s: PositiveFloat = 10.0  # guard accrual period = worst-case overshoot

    @property
    def ttl_s(self) -> int:
        return math.ceil(self.ttl_minutes * 60)


class CostAllocationSpec(_Strict):
    method: Literal["all_output", "all_input", "weighted"] = "all_output"
    output_input_ratio: str | int | None = None

    def allocation(self) -> CostAllocation:
        if self.method == "weighted":
            if self.output_input_ratio is None:
                raise ValueError("weighted allocation needs output_input_ratio")
            return CostAllocation.weighted(self.output_input_ratio)
        return CostAllocation(self.method)

    @model_validator(mode="after")
    def _valid(self) -> Self:
        self.allocation()
        return self


class QualitySpec(_Strict):
    suite: str
    baseline_variant: Slug
    gate: dict[str, Any] = Field(default_factory=dict)


# --- experiment -----------------------------------------------------------------


class Experiment(_Strict):
    name: Slug
    description: str
    model: str
    provider: ProviderSpec
    variants: Annotated[list[Variant], Field(min_length=1)]
    sweep: dict[str, Annotated[list[Any], Field(min_length=1)]] = Field(default_factory=dict)
    sample: Sample | None = None
    workloads: Annotated[list[WorkloadEntry], Field(min_length=1)]
    repetitions: PositiveInt = 3
    allow_single_run: bool = False
    slo: Slo | None = None
    cost_allocation: CostAllocationSpec = Field(default_factory=CostAllocationSpec)
    budget: BudgetSpec
    quality: QualitySpec | None = None
    loadgen: str = "native"
    seed: int = 0

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"variant names must be unique: {names}")
        labels = [w.name for w in self.workloads]
        if len(set(labels)) != len(labels):
            raise ValueError(f"workload labels must be unique (set `label`): {labels}")
        if self.repetitions == 1 and not self.allow_single_run:
            raise ValueError(
                "repetitions: 1 gives no confidence interval; set allow_single_run: true "
                "to run it anyway (results are flagged untrusted)"
            )
        if self.sample is not None and not self.sweep:
            raise ValueError("sample needs a sweep to draw from")
        if self.loadgen not in LOAD_GENERATORS:
            raise ValueError(f"unknown loadgen {self.loadgen!r}; known {sorted(LOAD_GENERATORS)}")
        if any(w.load.search is not None for w in self.workloads) and self.slo is None:
            raise ValueError("load.search needs an slo to search against")
        if self.quality is not None and self.quality.baseline_variant not in names:
            raise ValueError(f"quality.baseline_variant {self.quality.baseline_variant!r} unknown")
        is_mock = self.provider.kind == "mock"
        for v in self.variants:
            if v.mock and not is_mock:
                raise ValueError(f"variant {v.name}: `mock` overrides need provider kind mock")
        if not is_mock and any(k.startswith("mock.") for k in self.sweep):
            raise ValueError("mock.* sweep knobs need provider kind mock")
        return self

    @property
    def trusted(self) -> bool:
        return self.repetitions >= 2


def load_experiment(path: str | Path) -> Experiment:
    return Experiment.model_validate(read_yaml(Path(path)))


# --- expansion ------------------------------------------------------------------


class Cell(BaseModel):
    """One resolved serving configuration on one kind of host."""

    model_config = ConfigDict(frozen=True)

    key: str
    variant: str
    knobs: dict[str, Any]
    spec: ModelSpec
    mock: MockConfig | None
    launch: EngineLaunch
    host_key: str
    hardware: dict[str, Any]
    tokenizer: TokenizerSpec
    config: dict[str, Any]
    config_hash: str

    @property
    def gpus(self) -> int:
        return self.spec.hardware.gpus_per_replica


class ExpansionError(ValueError):
    pass


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _set_path(doc: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    out = dict(doc)
    if not rest:
        out[head] = value
        return out
    child = out.get(head)
    if child is None:
        child = {}
    if not isinstance(child, dict):
        raise ExpansionError(f"sweep knob {path!r}: {head!r} is not a mapping")
    out[head] = _set_path(child, rest, value)
    return out


def apply_variant(base: ModelSpec, variant: Variant) -> dict[str, Any]:
    """Registry entry with the variant's overrides, as an unvalidated dict.

    Same rules as `engines.apply_overrides`: switching engine needs its image and
    version and drops the other engine's args; an arg set to null is removed; the
    registry's max_context is a cap.
    """
    doc = base.model_dump(mode="json")
    patch = variant.model_dump(exclude_none=True, exclude={"name", "mock"})
    engine = patch.pop("engine", {})
    args = (variant.engine.args if variant.engine else None) or {}
    if "name" in engine and engine["name"] != base.engine.name:
        if "image" not in engine or "version" not in engine:
            raise ExpansionError(
                f"variant {variant.name}: switching engine to {engine['name']} needs image "
                "and version"
            )
        doc["engine"]["args"] = {}
    engine.pop("args", None)
    doc["engine"].update(engine)
    for key, value in args.items():
        if value is None:
            doc["engine"]["args"].pop(key, None)
        else:
            doc["engine"]["args"][key] = value
    if variant.max_context is not None and variant.max_context > base.max_context:
        raise ExpansionError(
            f"variant {variant.name}: max_context {variant.max_context} exceeds the registry "
            f"cap {base.max_context}"
        )
    return _merge(doc, patch)


def sweep_points(exp: Experiment) -> list[dict[str, Any]]:
    """Grid points in a stable order; `sample` draws a seeded subset of them."""
    if not exp.sweep:
        return [{}]
    keys = sorted(exp.sweep)
    grid = [
        dict(zip(keys, combo, strict=True))
        for combo in itertools.product(*(exp.sweep[k] for k in keys))
    ]
    if exp.sample is not None and exp.sample.random < len(grid):
        picked = random.Random(exp.sample.seed).sample(range(len(grid)), exp.sample.random)
        grid = [grid[i] for i in sorted(picked)]
    return grid


def _knob_label(value: Any) -> str:
    return value if isinstance(value, str) else canonical_json(value)


def cell_key(variant: str, knobs: Mapping[str, Any]) -> str:
    if not knobs:
        return variant
    inner = ",".join(f"{k}={_knob_label(knobs[k])}" for k in sorted(knobs))
    return f"{variant}[{inner}]"


MOCK_CONFIG_ARG = "--config-json"


def mock_launch(spec: ModelSpec, mock: MockConfig) -> EngineLaunch:
    """The mock backend stands in for the engine; its whole config is its command line."""
    return EngineLaunch(
        engine="mock",
        image="",
        model_repo=spec.hf.repo,
        model_revision=spec.hf.revision,
        served_model=spec.id,
        args=[MOCK_CONFIG_ARG, canonical_json(mock)],
        gpus=spec.hardware.gpus_per_replica,
    )


def mock_config_from_launch(launch: EngineLaunch) -> MockConfig:
    try:
        raw = launch.args[launch.args.index(MOCK_CONFIG_ARG) + 1]
    except (ValueError, IndexError):
        raise ValueError(f"mock launch needs {MOCK_CONFIG_ARG} <json>") from None
    return MockConfig.model_validate_json(raw)


def local_launch(spec: ModelSpec, provider: LocalProviderSpec) -> EngineLaunch:
    """What is known about an endpoint someone else started: engine and model, no command."""
    return EngineLaunch(
        engine=provider.engine,
        image="",
        model_repo=spec.hf.repo,
        model_revision=spec.hf.revision,
        served_model=provider.served_model,
        args=[],
        gpus=spec.hardware.gpus_per_replica,
    )


def _launch(exp: Experiment, spec: ModelSpec, mock: MockConfig | None) -> EngineLaunch:
    if mock is not None:
        return mock_launch(spec, mock)
    if isinstance(exp.provider, LocalProviderSpec):
        return local_launch(spec, exp.provider)
    return render_launch(spec, None)


def hardware_for(exp: Experiment, spec: ModelSpec) -> tuple[str, dict[str, Any]]:
    """(host group key, hardware description) for a cell's spec."""
    p = exp.provider
    gpus = spec.hardware.gpus_per_replica
    if p.kind == "mock":
        hw = {"provider": "mock", "gpu": spec.hardware.gpu, "gpus": gpus}
        return f"mock/{gpus}x{spec.hardware.gpu}", hw
    if p.kind == "local":
        hw = {"provider": "local", "engine": p.engine, "served_model": p.served_model}
        return f"local/{p.base_url}", hw
    instance = p.instance_type or spec.hardware.instance_types.aws
    if instance is None:
        raise ExpansionError(f"{spec.id}: no aws instance type in the registry or provider")
    hw = {
        "provider": "aws_ec2",
        "cloud": "aws",
        "region": p.region,
        "instance_type": instance,
        "market": p.market,
        "disk_gb": p.disk_gb,
        "gpu": spec.hardware.gpu,
        "gpus": gpus,
    }
    return f"aws/{p.region}/{instance}/{p.market}/{p.disk_gb}gb", hw


def tokenizer_for(exp: Experiment, spec: ModelSpec) -> TokenizerSpec:
    p = exp.provider
    if p.kind == "mock" or (p.kind == "local" and p.tokenizer == "simple"):
        return TokenizerSpec(kind="simple")
    return TokenizerSpec(kind="hf", repo=spec.hf.repo, revision=spec.hf.revision)


def build_cell(
    exp: Experiment,
    *,
    variant: str,
    knobs: Mapping[str, Any],
    spec: ModelSpec,
    mock: MockConfig | None,
) -> Cell:
    launch = _launch(exp, spec, mock)
    host_key, hardware = hardware_for(exp, spec)
    config: dict[str, Any] = {
        "model": spec.model_dump(mode="json"),
        "launch": launch.model_dump(mode="json"),
        "hardware": hardware,
    }
    return Cell(
        key=cell_key(variant, knobs),
        variant=variant,
        knobs=dict(knobs),
        spec=spec,
        mock=mock,
        launch=launch,
        host_key=host_key,
        hardware=hardware,
        tokenizer=tokenizer_for(exp, spec),
        config=config,
        config_hash=config_hash(config),
    )


def expand(exp: Experiment, registry: Registry) -> list[Cell]:
    """Cells in a deterministic order: variants as declared, then sweep grid order."""
    try:
        base = registry.get(exp.model)
    except KeyError as e:
        raise ExpansionError(str(e)) from None
    cells: list[Cell] = []
    for variant in exp.variants:
        patched = apply_variant(base, variant)
        for knobs in sweep_points(exp):
            model_doc, mock_over = patched, dict(variant.mock)
            for path, value in sorted(knobs.items()):
                if path.startswith("mock."):
                    mock_over = _set_path(mock_over, path.removeprefix("mock."), value)
                else:
                    model_doc = _set_path(model_doc, path, value)
            key = cell_key(variant.name, knobs)
            try:
                spec = ModelSpec.model_validate(model_doc)
                mock = (
                    exp.provider.mock_config({"models": [spec.id], **mock_over})
                    if isinstance(exp.provider, MockProviderSpec)
                    else None
                )
                cells.append(
                    build_cell(exp, variant=variant.name, knobs=knobs, spec=spec, mock=mock)
                )
            except ValueError as e:  # pydantic ValidationError, or a launch that cannot render
                raise ExpansionError(f"cell {key}: invalid overrides:\n{e}") from None
    keys = [c.key for c in cells]
    if len(set(keys)) != len(keys):
        raise ExpansionError(f"duplicate cell keys: {keys}")
    if exp.provider.kind == "local" and len(cells) != 1:
        raise ExpansionError(
            "the local provider benchmarks one running endpoint: declare one variant, no sweep"
        )
    for w in exp.workloads:
        try:
            w.resolve()
        except (FileNotFoundError, ValidationError) as e:
            raise ExpansionError(f"workload {w.name}: {e}") from None
    return cells


def derive_seed(*parts: Any) -> int:
    """Stable 31-bit seed from experiment seed, workload, load value and repetition."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
