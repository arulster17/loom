"""Statistics shared by run metrics, repetition aggregates and the quality gate.

Everything here is deterministic: bootstrap resampling takes an explicit seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field
from scipy import stats as sps

# A numpy-style reduction that accepts `axis` (np.mean, np.median,
# functools.partial(np.percentile, q=95), ...). Bootstrap applies it row-wise.
Statistic = Callable[..., np.ndarray | float]

# Upper bound on floats held per bootstrap chunk (n_boot rows x n samples).
_CHUNK_ELEMENTS = 4_000_000


class Estimate(BaseModel):
    """Mean of repeated measurements with a Student-t confidence interval.

    With fewer than two values there is no interval: `lo`/`hi`/`std` are None and
    `trusted` is False. A single run is never trusted.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    lo: float | None
    hi: float | None
    n: int
    std: float | None  # sample standard deviation (ddof=1)
    confidence: float = 0.95

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trusted(self) -> bool:
        return self.n >= 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cv(self) -> float | None:
        """Coefficient of variation, std / |mean|."""
        if self.std is None or self.mean == 0:
            return None
        return self.std / abs(self.mean)


class Interval(BaseModel):
    """Point estimate with a percentile-bootstrap confidence interval."""

    model_config = ConfigDict(frozen=True)

    point: float
    lo: float | None
    hi: float | None
    n: int
    confidence: float
    n_boot: int
    seed: int


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D sequence of values")
    if arr.size == 0:
        raise ValueError("no values")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    """q-th percentile (0-100) with numpy's default linear interpolation."""
    return float(np.percentile(_as_array(values), q, method="linear"))


def mean_ci(values: Sequence[float] | np.ndarray, confidence: float = 0.95) -> Estimate:
    """Mean with a two-sided Student-t interval: mean ± t(n-1) · s / √n."""
    _check_confidence(confidence)
    arr = _as_array(values)
    n = int(arr.size)
    mean = float(arr.mean())
    if n < 2:
        return Estimate(mean=mean, lo=None, hi=None, n=n, std=None, confidence=confidence)
    std = float(arr.std(ddof=1))
    half = float(sps.t.ppf((1 + confidence) / 2, n - 1)) * std / math.sqrt(n)
    return Estimate(mean=mean, lo=mean - half, hi=mean + half, n=n, std=std, confidence=confidence)


def _bootstrap_distribution(
    n: int, n_boot: int, seed: int, fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Draw `n_boot` index resamples of size n and evaluate `fn` on each row block."""
    rng = np.random.default_rng(seed)
    rows = max(1, _CHUNK_ELEMENTS // n)
    out = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, rows):
        stop = min(n_boot, start + rows)
        idx = rng.integers(0, n, size=(stop - start, n))
        out[start:stop] = fn(idx)
    return out


def _interval(
    point: float,
    boot: np.ndarray | None,
    n: int,
    confidence: float,
    n_boot: int,
    seed: int,
) -> Interval:
    if boot is None:
        return Interval(
            point=point, lo=None, hi=None, n=n, confidence=confidence, n_boot=n_boot, seed=seed
        )
    alpha = 1 - confidence
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(
        point=point,
        lo=float(lo),
        hi=float(hi),
        n=n,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Statistic = np.mean,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI for `statistic(values)`. No interval when n < 2."""
    _check_confidence(confidence)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    arr = _as_array(values)
    n = int(arr.size)
    point = float(statistic(arr, axis=-1))
    if n < 2:
        return _interval(point, None, n, confidence, n_boot, seed)
    boot = _bootstrap_distribution(n, n_boot, seed, lambda idx: statistic(arr[idx], axis=-1))
    return _interval(point, boot, n, confidence, n_boot, seed)


def paired_bootstrap_delta(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    statistic: Statistic = np.mean,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> Interval:
    """CI for statistic(a) - statistic(b) where a[i] and b[i] score the same item.

    Items are resampled jointly, so per-item difficulty cancels out. Typical use:
    a = candidate per-sample scores, b = baseline; the gate checks `lo`.
    """
    _check_confidence(confidence)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    xa, xb = _as_array(a), _as_array(b)
    if xa.size != xb.size:
        raise ValueError(f"paired samples differ in length: {xa.size} vs {xb.size}")
    n = int(xa.size)
    point = float(statistic(xa, axis=-1)) - float(statistic(xb, axis=-1))
    if n < 2:
        return _interval(point, None, n, confidence, n_boot, seed)
    boot = _bootstrap_distribution(
        n,
        n_boot,
        seed,
        lambda idx: statistic(xa[idx], axis=-1) - statistic(xb[idx], axis=-1),
    )
    return _interval(point, boot, n, confidence, n_boot, seed)
