"""Combine repetitions of one load point into mean ± t-CI per metric.

Each run contributes one value per metric (its p95 TTFT, its output tok/s, ...);
the aggregate is the across-run mean with a Student-t interval, so the CI reflects
run-to-run variance. A metric missing from some runs is aggregated over the runs
that have it, and its `n` says how many.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from loom_bench.metrics.summary import RunSummary, flatten_metrics
from loom_bench.stats import Estimate, mean_ci

# Metrics whose run-to-run coefficient of variation is checked against `cv_warn`.
HEADLINE_METRICS = (
    "ttft_ms.p50",
    "ttft_ms.p95",
    "tpot_ms.p50",
    "tpot_ms.p95",
    "e2e_ms.p95",
    "throughput.request_rate",
    "throughput.output_tok_s",
)


class AggregateSummary(BaseModel):
    n_runs: int
    confidence: float
    trusted: bool  # False with fewer than two runs
    warnings: list[str]
    metrics: dict[str, Estimate]  # keys as in `flatten_metrics`

    def get(self, key: str) -> Estimate | None:
        return self.metrics.get(key)


def aggregate_runs(
    runs: Sequence[RunSummary], *, confidence: float = 0.95, cv_warn: float = 0.10
) -> AggregateSummary:
    if not runs:
        raise ValueError("no runs to aggregate")
    values: dict[str, list[float]] = {}
    for run in runs:
        for key, v in flatten_metrics(run).items():
            values.setdefault(key, []).append(v)
    metrics = {key: mean_ci(vs, confidence) for key, vs in sorted(values.items())}

    warnings: list[str] = []
    if len(runs) < 2:
        warnings.append(
            f"only {len(runs)} run: no confidence interval; a single run is never trusted"
        )
    for key in HEADLINE_METRICS:
        est = metrics.get(key)
        if est is not None and est.cv is not None and est.cv > cv_warn:
            warnings.append(f"{key}: run-to-run CV {est.cv:.1%} exceeds {cv_warn:.0%}")

    return AggregateSummary(
        n_runs=len(runs),
        confidence=confidence,
        trusted=len(runs) >= 2,
        warnings=warnings,
        metrics=metrics,
    )
