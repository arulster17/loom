"""GPU utilization, memory and power from periodic `nvidia-smi` samples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

NVIDIA_SMI_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
)
NVIDIA_SMI_ARGS = (
    "nvidia-smi",
    f"--query-gpu={','.join(NVIDIA_SMI_FIELDS)}",
    "--format=csv,noheader,nounits",
)

_UNAVAILABLE = {"[N/A]", "N/A", "[Not Supported]", "[Unknown Error]"}


@dataclass(frozen=True, slots=True)
class GpuSample:
    timestamp: str
    index: int
    utilization_pct: float | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    power_w: float | None


def _num(text: str) -> float | None:
    text = text.strip()
    if text in _UNAVAILABLE:
        return None
    return float(text)


def parse_nvidia_smi(text: str) -> list[GpuSample]:
    """Parse output of `NVIDIA_SMI_ARGS` (possibly many polls concatenated).

    Fields nvidia-smi reports as unavailable become None; any other malformed line
    raises ValueError.
    """
    samples: list[GpuSample] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != len(NVIDIA_SMI_FIELDS):
            raise ValueError(f"expected {len(NVIDIA_SMI_FIELDS)} fields: {raw!r}")
        try:
            samples.append(
                GpuSample(
                    timestamp=parts[0],
                    index=int(parts[1]),
                    utilization_pct=_num(parts[2]),
                    memory_used_mib=_num(parts[3]),
                    memory_total_mib=_num(parts[4]),
                    power_w=_num(parts[5]),
                )
            )
        except ValueError as e:
            raise ValueError(f"malformed nvidia-smi line: {raw!r}") from e
    return samples


class GpuStats(BaseModel):
    n_samples: int
    utilization_mean_pct: float | None
    utilization_p95_pct: float | None
    memory_peak_mib: float | None
    memory_total_mib: float | None
    power_mean_w: float | None


class PerGpuStats(GpuStats):
    index: int


class GpuMetrics(BaseModel):
    """`overall` pools samples from all GPUs, so its power is the per-GPU mean."""

    n_gpus: int
    overall: GpuStats
    gpus: list[PerGpuStats]


def _stats(samples: Sequence[GpuSample]) -> dict[str, float | int | None]:
    util = [s.utilization_pct for s in samples if s.utilization_pct is not None]
    mem = [s.memory_used_mib for s in samples if s.memory_used_mib is not None]
    total = [s.memory_total_mib for s in samples if s.memory_total_mib is not None]
    power = [s.power_w for s in samples if s.power_w is not None]
    return {
        "n_samples": len(samples),
        "utilization_mean_pct": math.fsum(util) / len(util) if util else None,
        "utilization_p95_pct": float(np.percentile(util, 95)) if util else None,
        "memory_peak_mib": max(mem) if mem else None,
        "memory_total_mib": max(total) if total else None,
        "power_mean_w": math.fsum(power) / len(power) if power else None,
    }


def summarize_gpu(samples: Sequence[GpuSample]) -> GpuMetrics:
    by_index: dict[int, list[GpuSample]] = {}
    for s in samples:
        by_index.setdefault(s.index, []).append(s)
    return GpuMetrics(
        n_gpus=len(by_index),
        overall=GpuStats.model_validate(_stats(samples)),
        gpus=[
            PerGpuStats.model_validate({"index": i, **_stats(by_index[i])})
            for i in sorted(by_index)
        ],
    )
