"""Open-loop arrival schedules.

A schedule is a sorted float64 array of arrival offsets (seconds) in
[0, duration_s). Random schedules are pure functions of their parameters and a
seeded `numpy.random.Generator`, so a run can be replayed exactly.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    model_validator,
)

RateFn = Callable[[np.ndarray], np.ndarray]


def constant(rate: float, duration_s: float) -> np.ndarray:
    """Evenly spaced arrivals, the first at t=0."""
    n = int(np.ceil(duration_s * rate))
    t = np.arange(n, dtype=np.float64) / rate
    return t[t < duration_s]


def poisson(rate: float, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Homogeneous Poisson process: exponential gaps with mean 1/rate."""
    return _renewal(lambda size: rng.exponential(1.0 / rate, size), rate, duration_s)


def gamma(
    rate: float, burstiness: float, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    """Gamma-distributed gaps with shape=burstiness and mean 1/rate (vLLM `--burstiness`).

    burstiness=1 is Poisson; below 1 is burstier (CV = 1/sqrt(burstiness)), above 1 smoother.
    """
    scale = 1.0 / (rate * burstiness)
    return _renewal(lambda size: rng.gamma(burstiness, scale, size), rate, duration_s)


def onoff_burst(
    rate_hi: float,
    rate_lo: float,
    t_on_s: float,
    t_off_s: float,
    duration_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Poisson at `rate_hi` for `t_on_s`, then `rate_lo` for `t_off_s`, repeating."""
    cycle = t_on_s + t_off_s

    def rate_fn(t: np.ndarray) -> np.ndarray:
        return np.where(np.mod(t, cycle) < t_on_s, rate_hi, rate_lo)

    return _thinning(rate_fn, max(rate_hi, rate_lo), duration_s, rng)


def ramp(
    rate_start: float, rate_end: float, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    """Poisson with rate moving linearly from `rate_start` to `rate_end` (up or down)."""

    def rate_fn(t: np.ndarray) -> np.ndarray:
        return rate_start + (rate_end - rate_start) * t / duration_s

    return _thinning(rate_fn, max(rate_start, rate_end), duration_s, rng)


def diurnal(
    mean_rate: float,
    amplitude: float,
    period_s: float,
    duration_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Poisson with rate `mean_rate * (1 - amplitude * cos(2*pi*t / period_s))`.

    Starts at the trough, peaks at period/2. `amplitude` is relative, in [0, 1].
    """

    def rate_fn(t: np.ndarray) -> np.ndarray:
        return mean_rate * (1.0 - amplitude * np.cos(2.0 * np.pi * t / period_s))

    return _thinning(rate_fn, mean_rate * (1.0 + amplitude), duration_s, rng)


def _renewal(draw: Callable[[int], np.ndarray], rate: float, duration_s: float) -> np.ndarray:
    parts: list[np.ndarray] = []
    t = 0.0
    while True:
        size = int(rate * (duration_s - t) * 1.1) + 16
        times = t + np.cumsum(draw(size))
        parts.append(times[times < duration_s])
        if times[-1] >= duration_s:
            return np.concatenate(parts)
        t = float(times[-1])


def _thinning(
    rate_fn: RateFn, rate_max: float, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    """Non-homogeneous Poisson by thinning (Lewis & Shedler) a rate_max process."""
    if rate_max <= 0:
        return np.empty(0, dtype=np.float64)
    candidates = poisson(rate_max, duration_s, rng)
    keep = rng.random(candidates.size) * rate_max < rate_fn(candidates)
    return candidates[keep]


# --- trace replay -------------------------------------------------------------

TraceFormat = Literal["azure", "burstgpt"]

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class TraceRow:
    timestamp_s: float  # seconds since the first row in the file
    input_tokens: int
    output_tokens: int


def data_path(path: str | Path) -> Path:
    """Expand `~` and `$VARS` in user-supplied dataset/trace paths."""
    return Path(os.path.expandvars(str(path))).expanduser()


def read_azure_trace(path: str | Path, max_rows: int | None = None) -> list[TraceRow]:
    """Azure LLM inference trace CSV: `TIMESTAMP,ContextTokens,GeneratedTokens`.

    TIMESTAMP is an ISO datetime (2023: ``2023-11-16 18:15:46.6805900``; 2024 adds
    ``+00:00``); naive values are taken as UTC.
    """

    def micros(value: str) -> int:
        dt = datetime.fromisoformat(value.strip())
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return (dt - _EPOCH) // timedelta(microseconds=1)

    return _read_csv(path, max_rows, ("TIMESTAMP", "ContextTokens", "GeneratedTokens"), micros)


def read_burstgpt_trace(path: str | Path, max_rows: int | None = None) -> list[TraceRow]:
    """BurstGPT CSV: `Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type`.

    Timestamp is seconds. Failed requests (0 response tokens) are kept: they are
    still arrivals.
    """
    return _read_csv(
        path,
        max_rows,
        ("Timestamp", "Request tokens", "Response tokens"),
        lambda v: float(v) * 1e6,
    )


def read_trace(path: str | Path, fmt: TraceFormat, max_rows: int | None = None) -> list[TraceRow]:
    readers = {"azure": read_azure_trace, "burstgpt": read_burstgpt_trace}
    return readers[fmt](path, max_rows)


def _read_csv(
    path: str | Path,
    max_rows: int | None,
    columns: tuple[str, str, str],
    parse_micros: Callable[[str], float],
) -> list[TraceRow]:
    # Microseconds keep sub-ms offsets exact; float epoch seconds lose them.
    ts_col, in_col, out_col = columns
    raw: list[tuple[float, int, int]] = []
    with data_path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing trace columns {sorted(missing)}")
        for row in reader:
            if max_rows is not None and len(raw) >= max_rows:
                break
            raw.append((parse_micros(row[ts_col]), int(row[in_col]), int(row[out_col])))
    raw.sort(key=lambda r: r[0])
    if not raw:
        return []
    t_first = raw[0][0]
    return [TraceRow((t - t_first) / 1e6, i, o) for t, i, o in raw]


def trace_offsets(
    rows: list[TraceRow], time_scale: float = 1.0, duration_s: float | None = None
) -> np.ndarray:
    """Arrival offsets from trace rows; `time_scale` 0.5 replays twice as fast."""
    t = np.array([r.timestamp_s for r in rows], dtype=np.float64) * time_scale
    return t if duration_s is None else t[t < duration_s]


def trace_offsets_at_rate(rows: list[TraceRow], rate: float, duration_s: float) -> np.ndarray:
    """The trace's first round(rate x duration_s) arrivals, stretched or compressed so the
    next one would land at `duration_s`: the gaps keep their proportions and the mean
    rate over the run is `rate`."""
    n = round(rate * duration_s)
    if n < 1:
        raise ValueError(f"trace at {rate:g}/s for {duration_s:g} s schedules no arrivals")
    if len(rows) <= n:
        raise ValueError(
            f"trace has {len(rows)} arrivals; {rate:g}/s for {duration_s:g} s needs {n + 1}"
        )
    t = np.array([r.timestamp_s for r in rows[: n + 1]], dtype=np.float64)
    if t[n] <= 0:
        raise ValueError(f"the trace's first {n + 1} arrivals share one timestamp")
    scaled = t[:n] * (duration_s / t[n])
    return scaled[scaled < duration_s]


# --- YAML specs ---------------------------------------------------------------


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConstantArrivals(_Spec):
    kind: Literal["constant"] = "constant"
    rate: PositiveFloat

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        return constant(self.rate, duration_s)


class PoissonArrivals(_Spec):
    kind: Literal["poisson"] = "poisson"
    rate: PositiveFloat

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        return poisson(self.rate, duration_s, np.random.default_rng(seed))


class GammaArrivals(_Spec):
    kind: Literal["gamma"] = "gamma"
    rate: PositiveFloat
    burstiness: PositiveFloat

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        return gamma(self.rate, self.burstiness, duration_s, np.random.default_rng(seed))


class OnOffBurstArrivals(_Spec):
    kind: Literal["onoff_burst"] = "onoff_burst"
    rate_hi: PositiveFloat
    rate_lo: NonNegativeFloat
    t_on_s: PositiveFloat
    t_off_s: PositiveFloat

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return onoff_burst(self.rate_hi, self.rate_lo, self.t_on_s, self.t_off_s, duration_s, rng)


class RampArrivals(_Spec):
    kind: Literal["ramp"] = "ramp"
    rate_start: NonNegativeFloat
    rate_end: NonNegativeFloat

    @model_validator(mode="after")
    def _not_idle(self) -> RampArrivals:
        if self.rate_start == 0 and self.rate_end == 0:
            raise ValueError("ramp needs rate_start or rate_end > 0")
        return self

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        return ramp(self.rate_start, self.rate_end, duration_s, np.random.default_rng(seed))


class DiurnalArrivals(_Spec):
    kind: Literal["diurnal"] = "diurnal"
    mean_rate: PositiveFloat
    amplitude: Annotated[float, Field(ge=0.0, le=1.0)]
    period_s: PositiveFloat

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return diurnal(self.mean_rate, self.amplitude, self.period_s, duration_s, rng)


class TraceArrivals(_Spec):
    """Replays a trace's arrival times. `time_scale` stretches them (0.5 replays twice as
    fast, default 1); `rate` instead sets the mean rate (`trace_offsets_at_rate`), which
    is how an experiment sweeps a trace."""

    kind: Literal["trace"] = "trace"
    path: str
    format: TraceFormat
    time_scale: PositiveFloat | None = None
    rate: PositiveFloat | None = None
    max_rows: PositiveInt | None = None

    @model_validator(mode="after")
    def _one_scale(self) -> TraceArrivals:
        if self.time_scale is not None and self.rate is not None:
            raise ValueError("trace arrivals take time_scale or rate, not both")
        return self

    def rows(self) -> list[TraceRow]:
        return read_trace(self.path, self.format, self.max_rows)

    def schedule(self, duration_s: float, seed: int) -> np.ndarray:
        if self.rate is not None:
            return trace_offsets_at_rate(self.rows(), self.rate, duration_s)
        return trace_offsets(self.rows(), self.time_scale or 1.0, duration_s)


ArrivalSpec = Annotated[
    ConstantArrivals
    | PoissonArrivals
    | GammaArrivals
    | OnOffBurstArrivals
    | RampArrivals
    | DiurnalArrivals
    | TraceArrivals,
    Field(discriminator="kind"),
]

_ADAPTER: TypeAdapter[ArrivalSpec] = TypeAdapter(ArrivalSpec)


def parse_arrivals(data: Mapping[str, Any]) -> ArrivalSpec:
    """Build an arrival spec from a YAML mapping such as ``{kind: poisson, rate: 4}``."""
    return _ADAPTER.validate_python(data)
