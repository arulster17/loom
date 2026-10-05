"""Provenance records: everything needed to reproduce one benchmark run.

Hashes are sha256 over a canonical JSON encoding, so the same config hashes the
same regardless of key order, container type or whether it arrived as a
Pydantic model, dataclass or plain dict. Fields nobody measured stay None;
builders never fill in a guess.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path, PurePath
from typing import Any
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from loom_bench import __version__
from loom_bench.records import LoadMode

SCHEMA_VERSION = 1


def to_jsonable(obj: Any) -> Any:
    """Reduce `obj` to JSON-native types (dict with str keys, list, str, int, float, bool, None)."""
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if obj is None or isinstance(obj, bool | int | str):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float has no canonical JSON form: {obj!r}")
        return obj
    if isinstance(obj, BaseModel):
        return to_jsonable(obj.model_dump(mode="json"))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = _key(k)
            if key in out:
                raise ValueError(f"keys collide after conversion to str: {key!r}")
            out[key] = to_jsonable(v)
        return out
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set | frozenset):
        return sorted((to_jsonable(v) for v in obj), key=canonical_json)
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            raise ValueError(f"naive datetime is ambiguous: {obj!r}")
        return obj.astimezone(UTC).isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, UUID | Decimal):
        return str(obj)
    raise TypeError(f"no canonical JSON form for {type(obj).__name__}")


def _key(k: Any) -> str:
    if isinstance(k, Enum):
        k = k.value
    if isinstance(k, str):
        return k
    if isinstance(k, bool | int | float):
        return json.dumps(to_jsonable(k))
    if isinstance(k, PurePath | UUID):
        return to_jsonable(k)
    raise TypeError(f"unsupported mapping key type {type(k).__name__}")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 as-is."""
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


class GitInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha: str | None = None
    dirty: bool | None = None
    branch: str | None = None  # None when detached


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(repo_dir: str | Path = ".") -> GitInfo:
    """Commit, dirtiness (tracked files only) and branch of `repo_dir`.

    Every field is None outside a git work tree or when git is missing; `sha`
    and `dirty` are None in a repo with no commits yet.
    """
    repo = Path(repo_dir)
    if _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
        return GitInfo()
    sha = _git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    if not sha:
        return GitInfo(branch=_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD") or None)
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    branch = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    return GitInfo(sha=sha, dirty=None if status is None else status != "", branch=branch or None)


class Market(StrEnum):
    SPOT = "spot"
    ON_DEMAND = "on_demand"
    LOCAL = "local"


class ContentKind(StrEnum):
    SYNTHETIC = "synthetic"
    REALISTIC = "realistic"


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LoadgenInfo(_Section):
    name: str | None = None
    version: str | None = None


class EngineInfo(_Section):
    name: str | None = None
    version: str | None = None
    image: str | None = None
    image_digest: str | None = None
    args: dict[str, Any] | None = None


class ModelInfo(_Section):
    repo: str | None = None
    revision: str | None = None
    quantization: str | None = None


class HardwareInfo(_Section):
    gpu_type: str | None = None
    gpu_count: int | None = None
    instance_type: str | None = None


class WorkloadInfo(_Section):
    name: str | None = None
    profile_hash: str | None = None
    content: ContentKind | None = None


class DatasetInfo(_Section):
    name: str | None = None
    source: str | None = None
    revision: str | None = None
    license: str | None = None
    license_url: str | None = None


class LoadInfo(_Section):
    mode: LoadMode | None = None
    value: float | None = None  # req/s for open loop, concurrency for closed loop
    seed: int | None = None


class HostInfo(_Section):
    python: str | None = None
    platform: str | None = None


class Provenance(_Section):
    schema_version: int = SCHEMA_VERSION
    created_at: AwareDatetime
    git: GitInfo = Field(default_factory=GitInfo)
    bench_version: str | None = None
    loadgen: LoadgenInfo = Field(default_factory=LoadgenInfo)
    engine: EngineInfo = Field(default_factory=EngineInfo)
    cuda_version: str | None = None
    driver_version: str | None = None
    model: ModelInfo = Field(default_factory=ModelInfo)
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    cloud: str | None = None
    region: str | None = None
    market: Market | None = None
    config_hash: str
    config: dict[str, Any]
    workload: WorkloadInfo = Field(default_factory=WorkloadInfo)
    dataset: DatasetInfo = Field(default_factory=DatasetInfo)
    load: LoadInfo = Field(default_factory=LoadInfo)
    repetition: int | None = None
    host: HostInfo = Field(default_factory=HostInfo)

    @field_validator("created_at")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        return v.astimezone(UTC)


def host_info() -> HostInfo:
    """The client host running the benchmark (not the GPU server)."""
    return HostInfo(python=platform.python_version(), platform=platform.platform())


def build_provenance(
    config: Any,
    *,
    repo_dir: str | Path | None = None,
    created_at: datetime | None = None,
    **sections: Any,
) -> Provenance:
    """Provenance for a run of `config`.

    Detects what the client can know for certain (git state of `repo_dir`,
    bench version, host); everything else comes from `sections`, keyed by
    Provenance field name, and stays None when not given.
    """
    resolved = to_jsonable(config)
    if not isinstance(resolved, dict):
        raise TypeError("config must serialize to a JSON object")
    for name in ("schema_version", "config_hash", "config", "created_at"):
        if name in sections:
            raise TypeError(f"{name} is derived, not passed")
    sections.setdefault("git", git_info(repo_dir) if repo_dir is not None else GitInfo())
    sections.setdefault("bench_version", __version__)
    sections.setdefault("host", host_info())
    return Provenance(
        created_at=created_at or datetime.now(UTC),
        config_hash=config_hash(resolved),
        config=resolved,
        **sections,
    )


def provenance_hash(p: Provenance) -> str:
    """Content hash of the whole record, created_at included.

    Identifies one stored record; compare setups with `config_hash` instead.
    """
    return config_hash(p)
