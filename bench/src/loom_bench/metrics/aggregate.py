"""Combine repetitions of one load point into a point estimate ± CI per metric.

Each run contributes one value per metric (its p95 TTFT, its output tok/s, ...);
the aggregate combines them across runs, so the CI reflects run-to-run variance. A
metric missing from some runs is aggregated over the runs that have it, and its `n`
says how many.

How each metric is combined is decided in one place, `CI_RULES`:

- strictly positive, right-skewed quantities (latencies, throughputs, rates) use
  the geometric mean with a Student-t interval on the log scale ("log_t"), so both
  bounds stay positive. If any repetition is not positive (a throughput of 0), the
  metric falls back to the arithmetic interval clipped at 0 ("t_clipped").
- proportions (error rate, SLO attainment, cache hit rates) use the arithmetic
  mean of the per-run proportions with a Student-t interval clipped to [0, 1]. The
  interval is the run-to-run spread of each run's proportion (what the conservative
  SLO check needs), not a binomial interval over pooled requests: with no errors in
  any repetition it is [0, 0].
- counts and other non-negative quantities use the arithmetic interval clipped at 0.
- anything not listed uses the plain arithmetic interval ("t").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

from pydantic import BaseModel

from loom_bench.metrics.summary import RunSummary, flatten_metrics
from loom_bench.stats import Estimate, log_mean_ci, mean_ci

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

Scale = Literal["positive", "proportion", "percent", "non_negative", "unbounded"]


@dataclass(frozen=True)
class CiRule:
    """How one family of metrics is combined across repetitions."""

    scale: Scale
    description: str  # the metric family, for the methodology footer

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        return {
            "positive": (0.0, None),
            "proportion": (0.0, 1.0),
            "percent": (0.0, 100.0),
            "non_negative": (0.0, None),
            "unbounded": (None, None),
        }[self.scale]


_LATENCY = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "queue_delay_ms")
_DIST_STATS = ("mean", "min", "p50", "p90", "p95", "p99", "max")

LATENCY = CiRule("positive", "latency percentiles and means (ms)")
THROUGHPUT = CiRule("positive", "throughput and goodput (tok/s, req/s)")
OTHER_POSITIVE = CiRule("positive", "server queue time, measurement window")
PREFILL = CiRule("positive", "requests in prefill (request rate × mean TTFT)")
PROPORTION = CiRule("proportion", "error rate, SLO attainment, cache fractions and hit rates")
PERCENT = CiRule("percent", "GPU utilization (%)")
COUNT = CiRule("non_negative", "request counts and sample sizes")
GAUGE = CiRule("non_negative", "other server and GPU gauges")
UNBOUNDED = CiRule("unbounded", "anything else")

# Ordered (glob pattern, rule) over `flatten_metrics` keys; the first match wins.
CI_RULES: tuple[tuple[str, CiRule], ...] = (
    ("*.n", COUNT),
    *((f"{m}.{s}", LATENCY) for m in _LATENCY for s in _DIST_STATS),
    *((f"per_request_output_tok_s.{s}", THROUGHPUT) for s in _DIST_STATS),
    ("throughput.*", THROUGHPUT),
    ("goodput.request_rate", THROUGHPUT),
    ("goodput.output_tok_s", THROUGHPUT),
    ("prefill_in_flight", PREFILL),
    ("server.queue_time_mean_ms", OTHER_POSITIVE),
    ("window_s", OTHER_POSITIVE),
    ("error_rate", PROPORTION),
    ("goodput.slo_attainment", PROPORTION),
    ("cached_prompt_fraction", PROPORTION),
    ("server.kv_usage_*", PROPORTION),
    ("server.prefix_cache_hit_rate", PROPORTION),
    ("gpu.*_pct", PERCENT),
    ("counts.*", COUNT),
    ("n_total", COUNT),
    ("n_warmup_excluded", COUNT),
    ("requests_missing_usage", COUNT),
    ("goodput.good_requests", COUNT),
    ("server.*", GAUGE),
    ("gpu.*", GAUGE),
)


def ci_rule(key: str) -> CiRule:
    """The rule for an aggregate key as produced by `flatten_metrics`."""
    return next((rule for pattern, rule in CI_RULES if fnmatchcase(key, pattern)), UNBOUNDED)


def estimate_metric(key: str, values: Sequence[float], confidence: float = 0.95) -> Estimate:
    """Combine one metric's per-run values with the method `ci_rule(key)` assigns."""
    rule = ci_rule(key)
    if rule.scale == "positive" and all(v > 0 for v in values):
        return log_mean_ci(values, confidence)
    lower, upper = rule.bounds
    return mean_ci(values, confidence, lower=lower, upper=upper)


def ci_rule_summary() -> list[tuple[str, str]]:
    """(metric family, method) pairs in rule order, for methodology text."""
    method = {
        "positive": "geometric mean, Student-t interval on the log scale (log_t); "
        "arithmetic interval clipped at 0 if a repetition is 0",
        "proportion": "arithmetic mean of per-run proportions, Student-t interval "
        "clipped to [0, 1] (t_clipped)",
        "percent": "arithmetic mean, Student-t interval clipped to [0, 100] (t_clipped)",
        "non_negative": "arithmetic mean, Student-t interval clipped at 0 (t_clipped)",
        "unbounded": "arithmetic mean, Student-t interval (t)",
    }
    out: dict[str, str] = {}
    for _, rule in (*CI_RULES, ("", UNBOUNDED)):
        out.setdefault(rule.description, method[rule.scale])
    return list(out.items())


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
    metrics = {key: estimate_metric(key, vs, confidence) for key, vs in sorted(values.items())}

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
