"""Workload profiles: pluggable YAML in `bench/workloads/`, validated by Pydantic.

`content` says whether prompts are synthetic or realistic text. Reports must
carry it: random text understates speculative decoding and prefix caching.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveInt,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from loom_bench.client.openai_stream import Endpoint
from loom_bench.loadgen.arrivals import TraceFormat

PROFILES_DIR = Path(__file__).resolve().parents[3] / "workloads"

RangeRatio = Annotated[float, Field(ge=0.0, lt=1.0)]


class DatasetProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source: str
    revision: str | None = None
    license: str
    license_url: str | None = None


class DatasetDownload(BaseModel):
    """A public dataset file pinned to a Hugging Face commit and its sha256, which a
    remote GPU host downloads for itself (the laptop's `path` does not exist there).
    Locally, `path` is used as given."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hf_dataset: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")]
    revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    filename: Annotated[str, StringConstraints(pattern=r"^[\w.-]+(/[\w.-]+)*$")]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: PositiveInt

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.hf_dataset}/resolve/"
            f"{self.revision}/{self.filename}"
        )


class _Profile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    content: Literal["synthetic", "realistic"]
    dataset: DatasetProvenance | None = None
    seed: int = 0
    temperature: NonNegativeFloat = 0.0


class SyntheticProfile(_Profile):
    """Random text of fixed (or uniformly jittered) length; `ignore_eos` pins output length."""

    kind: Literal["synthetic"] = "synthetic"
    endpoint: Endpoint = "completions"
    input_len: PositiveInt
    output_len: PositiveInt
    range_ratio: RangeRatio = 0.0  # lengths uniform in [len*(1-r), len*(1+r)]
    ignore_eos: bool = True


class SharedPrefixProfile(_Profile):
    """System prompt shared within `num_prefix_groups` groups, then a unique user turn."""

    kind: Literal["shared_prefix"] = "shared_prefix"
    endpoint: Literal["chat"] = "chat"
    input_len: PositiveInt
    output_len: PositiveInt
    range_ratio: RangeRatio = 0.0
    prefix_share: Annotated[float, Field(ge=0.0, le=0.95)]
    num_prefix_groups: PositiveInt = 1
    ignore_eos: bool = True

    @property
    def prefix_len(self) -> int:
        return round(self.input_len * self.prefix_share)

    @model_validator(mode="after")
    def _room_for_suffix(self) -> SharedPrefixProfile:
        if int(self.input_len * (1 - self.range_ratio)) - self.prefix_len < 1:
            raise ValueError("input_len too small for prefix_share: no room for a unique suffix")
        return self


class ChatDatasetProfile(_Profile):
    """ShareGPT-format conversations; output length follows the reference reply."""

    kind: Literal["chat_dataset"] = "chat_dataset"
    endpoint: Literal["chat"] = "chat"
    path: str  # user-supplied; `~` and `$VARS` expanded
    download: DatasetDownload | None = None  # how a remote host gets the file at `path`
    max_input_len: PositiveInt | None = None  # skip longer conversations
    min_output_len: PositiveInt = 4  # skip near-empty reference replies
    max_output_len: PositiveInt = 1024
    ignore_eos: bool = True


class NeedleProfile(_Profile):
    """Haystack of `context_len` tokens with one retrievable fact per request."""

    kind: Literal["long_context_needle"] = "long_context_needle"
    endpoint: Endpoint = "chat"
    context_len: Annotated[int, Field(ge=128)]
    depths: Annotated[list[Annotated[float, Field(ge=0.0, le=1.0)]], Field(min_length=1)] = [0.5]
    output_len: PositiveInt = 32


class CodeCompletionProfile(_Profile):
    """Python-like source cut mid-file; short, latency-sensitive completions."""

    kind: Literal["code_completion"] = "code_completion"
    endpoint: Literal["completions"] = "completions"
    input_len: PositiveInt
    output_len: PositiveInt
    range_ratio: RangeRatio = 0.0
    ignore_eos: bool = True


class LongGenerationProfile(_Profile):
    """Short essay instruction, long answer."""

    kind: Literal["long_generation"] = "long_generation"
    endpoint: Endpoint = "chat"
    input_len: Annotated[int, Field(ge=32)] = 256
    output_len: PositiveInt = 2048
    range_ratio: RangeRatio = 0.0
    ignore_eos: bool = True


class TraceProfile(_Profile):
    """Token lengths from a production trace; pair with `trace` arrivals on the same file
    (request i gets row i, arrival i is row i's time scaled to the load's rate)."""

    kind: Literal["trace"] = "trace"
    endpoint: Endpoint = "completions"
    path: str
    format: TraceFormat
    max_rows: PositiveInt | None = None
    max_input_len: PositiveInt | None = None  # clamp to fit the model's context
    max_output_len: PositiveInt | None = None
    ignore_eos: bool = True


WorkloadProfile = Annotated[
    SyntheticProfile
    | SharedPrefixProfile
    | ChatDatasetProfile
    | NeedleProfile
    | CodeCompletionProfile
    | LongGenerationProfile
    | TraceProfile,
    Field(discriminator="kind"),
]

_ADAPTER: TypeAdapter[WorkloadProfile] = TypeAdapter(WorkloadProfile)


def parse_profile(data: Mapping[str, Any]) -> WorkloadProfile:
    return _ADAPTER.validate_python(data)


def apply_overrides(profile: WorkloadProfile, overrides: Mapping[str, Any]) -> WorkloadProfile:
    """Experiment-level field overrides, deep-merged and re-validated."""
    return parse_profile(_merge(profile.model_dump(), overrides))


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    return sorted(p.stem for p in profiles_dir.glob("*.yaml"))


def load_profile(
    name_or_path: str | Path,
    overrides: Mapping[str, Any] | None = None,
    *,
    profiles_dir: Path = PROFILES_DIR,
) -> WorkloadProfile:
    """Load a profile by name (`fixed-1k-1k`) or YAML path, then apply `overrides`."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = profiles_dir / f"{name_or_path}.yaml"
    if not path.is_file():
        known = ", ".join(list_profiles(profiles_dir))
        raise FileNotFoundError(f"no workload profile {name_or_path!r} (known: {known})")
    profile = parse_profile(yaml.safe_load(path.read_text()))
    return apply_overrides(profile, overrides) if overrides else profile


def _merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out
