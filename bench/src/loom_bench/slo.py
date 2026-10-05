"""Latency/error SLOs, the conservative pass rule, and goodput search over a load sweep.

An SLO is declared in experiment YAML, e.g.

    slo: {ttft_ms: {p95: 1000}, tpot_ms: {p95: 50}, max_error_rate: 0.01}

Run-level check (`slo_met`): a load point meets the SLO only if, for every target,
the upper end of the 95% CI across repetitions is within the target. With a single
repetition the mean is used and the verdict is flagged untrusted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat

from loom_bench.records import LoadMode, RequestRecord
from loom_bench.stats import Estimate

if TYPE_CHECKING:
    from loom_bench.metrics.aggregate import AggregateSummary

LatencyMetric = Literal["ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"]
Pctl = Literal["p50", "p90", "p95", "p99"]

LATENCY_METRICS: tuple[LatencyMetric, ...] = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms")
PERCENTILES: dict[Pctl, float] = {"p50": 50.0, "p90": 90.0, "p95": 95.0, "p99": 99.0}

Targets = dict[Pctl, PositiveFloat]


class Slo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ttft_ms: Targets | None = None
    tpot_ms: Targets | None = None
    itl_ms: Targets | None = None
    e2e_ms: Targets | None = None
    max_error_rate: float = Field(ge=0, le=1)

    @classmethod
    def from_yaml(cls, text: str) -> Slo:
        return cls.model_validate(yaml.safe_load(text))

    def targets(self) -> list[tuple[LatencyMetric, Pctl, float]]:
        out: list[tuple[LatencyMetric, Pctl, float]] = []
        for metric in LATENCY_METRICS:
            for pctl, target in sorted(
                (getattr(self, metric) or {}).items(), key=lambda kv: PERCENTILES[kv[0]]
            ):
                out.append((metric, pctl, target))
        return out

    def request_meets(self, r: RequestRecord) -> bool:
        """Per-request check used for SLO attainment and request goodput.

        A request is good when it succeeded and, for TTFT / TPOT / E2E, its value is
        within the loosest listed target of that metric (the tail bound every good
        request must respect, as in vLLM's `--goodput`). For ITL, each listed
        percentile of the request's own token gaps must be within its target.
        TPOT and ITL are vacuously met by requests with fewer than two tokens.
        """
        if not r.ok:
            return False
        scalars: dict[str, float | None] = {
            "ttft_ms": r.ttft_s,
            "tpot_ms": r.tpot_s,
            "e2e_ms": r.e2e_s,
        }
        for metric, value_s in scalars.items():
            targets: dict[Pctl, float] | None = getattr(self, metric)
            if not targets:
                continue
            if value_s is None:
                if metric == "tpot_ms":
                    continue
                return False
            if value_s * 1000 > max(targets.values()):
                return False
        if self.itl_ms and r.itl_s:
            gaps_ms = np.asarray(r.itl_s, dtype=float) * 1000
            for pctl, target in self.itl_ms.items():
                if float(np.percentile(gaps_ms, PERCENTILES[pctl])) > target:
                    return False
        return True


class SloCheck(BaseModel):
    metric: str  # aggregate key, e.g. "ttft_ms.p95" or "error_rate"
    target: float
    observed: float | None  # CI upper bound, or the mean when there is no CI
    used_upper_bound: bool
    passed: bool


class SloVerdict(BaseModel):
    met: bool
    trusted: bool  # False when any check had no CI (single repetition)
    checks: list[SloCheck]


def _check(metric: str, target: float, est: Estimate | None) -> SloCheck:
    if est is None:
        return SloCheck(
            metric=metric, target=target, observed=None, used_upper_bound=False, passed=False
        )
    bound = est.hi if est.hi is not None else est.mean
    return SloCheck(
        metric=metric,
        target=target,
        observed=bound,
        used_upper_bound=est.hi is not None,
        passed=bound <= target,
    )


def slo_met(slo: Slo, agg: AggregateSummary) -> SloVerdict:
    """Conservative run-level verdict; a missing metric fails its check."""
    checks = [
        _check(f"{metric}.{pctl}", target, agg.get(f"{metric}.{pctl}"))
        for metric, pctl, target in slo.targets()
    ]
    checks.append(_check("error_rate", slo.max_error_rate, agg.get("error_rate")))
    return SloVerdict(
        met=all(c.passed for c in checks),
        trusted=agg.trusted and all(c.used_upper_bound for c in checks),
        checks=checks,
    )


class GoodputPoint(BaseModel):
    load: float
    met: bool
    trusted: bool


class GoodputResult(BaseModel):
    """Highest load meeting the SLO and the throughput measured there.

    `load` is requests/s for open loop and concurrency for closed loop. Throughput
    fields are per replica, mean ± CI across repetitions at that load point.
    `bracketed` is False when no tested load failed, i.e. goodput may be higher.
    """

    load_mode: LoadMode
    slo: Slo
    max_load: float | None
    first_failing_load: float | None
    bracketed: bool
    trusted: bool
    request_rate: Estimate | None
    output_tok_s: Estimate | None
    input_tok_s: Estimate | None
    total_tok_s: Estimate | None
    max_sustainable_concurrency: float | None  # closed loop only
    points: list[GoodputPoint]


def find_goodput(
    slo: Slo,
    points: Sequence[tuple[float, AggregateSummary]],
    load_mode: LoadMode,
) -> GoodputResult:
    """Goodput = the highest passing load below the first failing one.

    Loads above the first failure never count, even if they happen to pass.
    """
    ordered = sorted(points, key=lambda p: p[0])
    loads = [load for load, _ in ordered]
    if len(set(loads)) != len(loads):
        raise ValueError("duplicate load values in sweep")

    graded: list[GoodputPoint] = []
    best: tuple[float, AggregateSummary, SloVerdict] | None = None
    first_fail: float | None = None
    for load, agg in ordered:
        verdict = slo_met(slo, agg)
        graded.append(GoodputPoint(load=load, met=verdict.met, trusted=verdict.trusted))
        if first_fail is not None:
            continue
        if verdict.met:
            best = (load, agg, verdict)
        else:
            first_fail = load

    if best is None:
        return GoodputResult(
            load_mode=load_mode,
            slo=slo,
            max_load=None,
            first_failing_load=first_fail,
            bracketed=first_fail is not None,
            trusted=False,
            request_rate=None,
            output_tok_s=None,
            input_tok_s=None,
            total_tok_s=None,
            max_sustainable_concurrency=None,
            points=graded,
        )
    load, agg, verdict = best
    return GoodputResult(
        load_mode=load_mode,
        slo=slo,
        max_load=load,
        first_failing_load=first_fail,
        bracketed=first_fail is not None,
        trusted=verdict.trusted,
        request_rate=agg.get("throughput.request_rate"),
        output_tok_s=agg.get("throughput.output_tok_s"),
        input_tok_s=agg.get("throughput.input_tok_s"),
        total_tok_s=agg.get("throughput.total_tok_s"),
        max_sustainable_concurrency=load if load_mode is LoadMode.CLOSED_LOOP else None,
        points=graded,
    )


def bisect_next_load(
    history: Sequence[tuple[float, bool]], lo: float, hi: float, rel_tol: float = 0.05
) -> float | None:
    """Next load to test when searching [lo, hi] for the highest load meeting the SLO.

    `history` holds (load, met) for tested points; points outside [lo, hi] are
    ignored. Tests `lo`, then `hi`, then bisects between the highest pass below the
    lowest failure and that failure. Returns None when done: `lo` fails, `hi` passes,
    or the bracket is within `rel_tol` of the passing load.
    """
    if not 0 < lo < hi:
        raise ValueError("need 0 < lo < hi")
    if rel_tol <= 0:
        raise ValueError("rel_tol must be positive")
    results = {load: met for load, met in history if lo <= load <= hi}
    if lo not in results:
        return lo
    if not results[lo]:
        return None
    failures = [load for load, met in results.items() if not met]
    if not failures:
        return hi if hi not in results else None
    fail = min(failures)
    passing = max(load for load, met in results.items() if met and load < fail)
    if fail - passing <= rel_tol * passing:
        return None
    return (passing + fail) / 2
