"""Statistics shared by run metrics, repetition aggregates and the quality gate.

Everything here is deterministic: bootstrap resampling takes an explicit seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, computed_field
from scipy import stats as sps

# A numpy-style reduction that accepts `axis` (np.mean, np.median,
# functools.partial(np.percentile, q=95), ...). Bootstrap applies it row-wise.
Statistic = Callable[..., Any]

# Upper bound on floats held per bootstrap chunk (n_boot rows x n samples).
_CHUNK_ELEMENTS = 4_000_000


CiMethod = Literal["t", "t_clipped", "log_t"]

# What each `Estimate.method` means; reports quote these.
CI_METHOD_TEXT: dict[CiMethod, str] = {
    "t": "arithmetic mean with a two-sided Student-t interval, mean ± t(n-1)·s/√n",
    "t_clipped": (
        "arithmetic mean with a two-sided Student-t interval, mean ± t(n-1)·s/√n, "
        "with both bounds clipped to the metric's range"
    ),
    "log_t": (
        "geometric mean with a Student-t interval of the log values, exponentiated: "
        "exp(m ± t(n-1)·s_log/√n), where m and s_log are the mean and standard deviation "
        "of ln(value); multiplicative and always positive"
    ),
}


class Estimate(BaseModel):
    """Point estimate of repeated measurements with a confidence interval.

    `method` says how `mean`, `lo` and `hi` were computed (see `CI_METHOD_TEXT`):

    - "t": `mean` is the arithmetic mean; `lo`/`hi` its Student-t interval.
    - "t_clipped": the same, with `lo`/`hi` clipped to the metric's range (a
      proportion to [0, 1], a count to >= 0).
    - "log_t": `mean` is the geometric mean; `lo`/`hi` are the Student-t interval of
      ln(values), exponentiated, so 0 < lo <= mean <= hi. For strictly positive,
      right-skewed quantities (latency, throughput, rates).

    `std` is always the sample standard deviation (ddof=1) of the raw values;
    `log_std` that of ln(values), for "log_t" only. With fewer than two values there is
    no interval: `lo`/`hi`/`std` are None and `trusted` is False. A single run is
    never trusted.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    lo: float | None
    hi: float | None
    n: int
    std: float | None  # sample standard deviation of the values (ddof=1)
    confidence: float = 0.95
    method: CiMethod = "t"
    log_std: float | None = None  # sample standard deviation of ln(values); log_t only

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trusted(self) -> bool:
        return self.n >= 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cv(self) -> float | None:
        """Run-to-run coefficient of variation.

        std / |mean| for arithmetic estimates; for "log_t" the geometric CV,
        sqrt(exp(s_log²) - 1), which is std / mean of a lognormal and matches the
        interval's scale.
        """
        if self.method == "log_t" and self.log_std is not None:
            return math.sqrt(math.expm1(self.log_std**2))
        if self.std is None or self.mean == 0:
            return None
        return self.std / abs(self.mean)

    def scaled(self, k: float) -> Estimate:
        """The estimate of k x the quantity (k > 0); the method and log spread carry over."""
        if k <= 0:
            raise ValueError("scale factor must be positive")
        return self.model_copy(
            update={
                "mean": self.mean * k,
                "lo": None if self.lo is None else self.lo * k,
                "hi": None if self.hi is None else self.hi * k,
                "std": None if self.std is None else self.std * k,
            }
        )


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


def _t_quantile(confidence: float, n: int) -> float:
    return float(sps.t.ppf((1 + confidence) / 2, n - 1))


def mean_ci(
    values: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> Estimate:
    """Mean with a two-sided Student-t interval: mean ± t(n-1) · s / √n.

    With `lower` and/or `upper` (the metric's range, e.g. 0 and 1 for a proportion)
    the bounds are clipped to it and the method is "t_clipped".
    """
    _check_confidence(confidence)
    arr = _as_array(values)
    n = int(arr.size)
    mean = float(arr.mean())
    method: CiMethod = "t" if lower is None and upper is None else "t_clipped"
    if n < 2:
        return Estimate(
            mean=mean, lo=None, hi=None, n=n, std=None, confidence=confidence, method=method
        )
    std = float(arr.std(ddof=1))
    half = _t_quantile(confidence, n) * std / math.sqrt(n)
    lo, hi = mean - half, mean + half
    if lower is not None:
        lo, hi = max(lo, lower), max(hi, lower)
    if upper is not None:
        lo, hi = min(lo, upper), min(hi, upper)
    return Estimate(mean=mean, lo=lo, hi=hi, n=n, std=std, confidence=confidence, method=method)


def log_mean_ci(values: Sequence[float] | np.ndarray, confidence: float = 0.95) -> Estimate:
    """Geometric mean with a Student-t interval on the log scale, exponentiated.

    With m and s the mean and sample standard deviation of ln(values):
    point exp(m), interval exp(m ± t(n-1) · s / √n). Both bounds are positive and the
    interval is multiplicative around the point (geometric mean ×/÷ a factor), the
    right shape for strictly positive, right-skewed quantities. Every value must be
    positive.
    """
    _check_confidence(confidence)
    arr = _as_array(values)
    if np.any(arr <= 0):
        raise ValueError("log-scale interval needs strictly positive values")
    n = int(arr.size)
    logs = np.log(arr)
    m = float(logs.mean())
    # Identical values give that value exactly (exp(ln x) can be off in the last bit).
    constant = bool(np.all(arr == arr[0]))
    mean = float(arr[0]) if constant else math.exp(m)
    if n < 2:
        return Estimate(
            mean=mean, lo=None, hi=None, n=n, std=None, confidence=confidence, method="log_t"
        )
    log_std = 0.0 if constant else float(logs.std(ddof=1))
    half = _t_quantile(confidence, n) * log_std / math.sqrt(n)
    return Estimate(
        mean=mean,
        lo=mean if constant else math.exp(m - half),
        hi=mean if constant else math.exp(m + half),
        n=n,
        std=float(arr.std(ddof=1)),
        confidence=confidence,
        method="log_t",
        log_std=log_std,
    )


def _bootstrap_distribution(
    n: int, n_boot: int, seed: int, fn: Callable[[np.ndarray], Any]
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
