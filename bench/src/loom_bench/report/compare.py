"""Compare two experiments or two configs, load point by load point.

Sweeps are matched by config hash (default: a rerun of the same config), cell key
(the same matrix cell, whose config hash must also match), or workload alone (two
different configs on the same workload; each side must then hold one config per
workload). Within a matched sweep, points are matched by load value.

Verdict per metric, on the repetition means (delta = B - A):

1. within tolerance: |delta| <= rel_tol * |A| (absolute tolerance for metrics in
   `abs_tol`, e.g. error rate) → within normal variance;
2. otherwise, Welch's t-test on the two sets of repetitions: p >= alpha means the
   difference is not distinguishable from run-to-run noise → within normal variance.
   Metrics aggregated on the log scale (latency, throughput: `Estimate.method`
   "log_t") are tested on ln(values), i.e. on the ratio of geometric means;
3. otherwise, or with fewer than two repetitions on either side → outside.

Cost at SLO has no per-run samples, so rule 2 is replaced by overlap of the two
cost CIs. Goodput is shown with its search bracket; when the two brackets overlap the
search cannot separate the configs (`resolution.brackets_overlap`) and the comparison
says so. The comparison as a whole is within normal variance only if every
metric is, every sweep and point matched, and (except when matching by workload)
config hashes are identical.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from scipy import stats as sps

from loom_bench.cost import MicrosRange
from loom_bench.metrics.aggregate import HEADLINE_METRICS
from loom_bench.records import LoadMode
from loom_bench.report.analyze import COMPLETED, ConfigResult, LoadPoint, load_points
from loom_bench.report.format import goodput_bracket, md_table, num, pct, usd
from loom_bench.report.resolution import brackets_overlap
from loom_bench.stats import Estimate
from loom_bench.store.models import BenchRun

MatchBy = Literal["config_hash", "cell_key", "workload"]
DEFAULT_METRICS = (*HEADLINE_METRICS, "error_rate")
DEFAULT_ABS_TOL: Mapping[str, float] = {"error_rate": 0.01}
COST_METRICS = ("output_per_mtok", "input_per_mtok", "total_per_mtok")


class MetricDelta(BaseModel):
    metric: str
    a: Estimate | None
    b: Estimate | None
    delta: float | None  # b - a, on repetition means
    delta_lo: float | None  # Welch confidence interval of the difference
    delta_hi: float | None
    rel_delta: float | None  # delta / |a|
    p_value: float | None  # Welch two-sided
    within_normal_variance: bool
    reason: str


class CostDelta(BaseModel):
    metric: str
    a: MicrosRange | None
    b: MicrosRange | None
    delta_micros: int | None
    rel_delta: float | None
    within_normal_variance: bool
    reason: str


class PointComparison(BaseModel):
    workload: str
    load_mode: LoadMode
    load: float
    name_a: str
    name_b: str
    config_hash_a: str
    config_hash_b: str
    config_hash_match: bool
    metrics: list[MetricDelta]
    within_normal_variance: bool


class ConfigComparison(BaseModel):
    workload: str
    load_mode: LoadMode
    name_a: str
    name_b: str
    config_hash_a: str
    config_hash_b: str
    config_hash_match: bool
    goodput_load_a: float | None
    goodput_load_b: float | None
    goodput_bracket_a: str | None = None  # `format.goodput_bracket`
    goodput_bracket_b: str | None = None
    goodput_tie: bool = False  # brackets overlap: a tie within the search resolution
    metrics: list[MetricDelta]
    cost: list[CostDelta]
    within_normal_variance: bool


class Comparison(BaseModel):
    label_a: str
    label_b: str
    match_by: MatchBy
    rel_tol: float
    abs_tol: dict[str, float]
    alpha: float
    confidence: float
    configs: list[ConfigComparison]
    points: list[PointComparison]
    only_in_a: list[str]
    only_in_b: list[str]
    within_normal_variance: bool


@dataclass(frozen=True)
class _Sweep:
    config_hash: str
    name: str
    workload: str
    load_mode: LoadMode
    points: list[LoadPoint]
    result: ConfigResult | None


def _sweeps(side: Sequence[ConfigResult] | Iterable[BenchRun]) -> list[_Sweep]:
    items = list(side)
    if not items:
        raise ValueError("nothing to compare: empty input")
    if all(isinstance(i, ConfigResult) for i in items):
        return [
            _Sweep(r.config_hash, r.name, r.workload, r.load_mode, r.points, r)
            for r in items
            if isinstance(r, ConfigResult)
        ]
    if not all(isinstance(i, BenchRun) for i in items):
        raise TypeError("compare takes a list of ConfigResult or a list of BenchRun rows")
    runs = [r for r in items if isinstance(r, BenchRun)]
    cells: dict[str, set[str]] = defaultdict(set)
    for r in runs:
        if r.status == COMPLETED and r.cell_key:
            cells[r.config_hash].add(r.cell_key)
    return [
        _Sweep(
            key.config_hash,
            next(iter(cells[key.config_hash]))
            if len(cells[key.config_hash]) == 1
            else key.config_hash[:12],
            key.workload,
            key.load_mode,
            points,
            None,
        )
        for key, points in sorted(load_points(runs).items())
    ]


def _key(s: _Sweep, by: MatchBy) -> tuple[str, ...]:
    if by == "config_hash":
        return (s.config_hash, s.workload, s.load_mode.value)
    if by == "cell_key":
        return (s.name, s.workload, s.load_mode.value)
    return (s.workload, s.load_mode.value)


def _index(sweeps: list[_Sweep], by: MatchBy, label: str) -> dict[tuple[str, ...], _Sweep]:
    out: dict[tuple[str, ...], _Sweep] = {}
    for s in sweeps:
        k = _key(s, by)
        if k in out:
            raise ValueError(f"{label}: more than one sweep matches {k} when matching by {by}")
        out[k] = s
    return out


def _welch_moments(
    ma: float, sa: float, na: int, mb: float, sb: float, nb: int, confidence: float
) -> tuple[float, float, float]:
    """(lo, hi, p) of Welch's test and interval for mb - ma."""
    va, vb = sa**2 / na, sb**2 / nb
    d = mb - ma
    se = math.sqrt(va + vb)
    if se == 0:
        return d, d, 1.0 if d == 0 else 0.0
    df = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    p = float(2 * sps.t.sf(abs(d) / se, df))
    half = float(sps.t.ppf((1 + confidence) / 2, df)) * se
    return d - half, d + half, p


def _welch(
    a: Estimate, b: Estimate, confidence: float
) -> tuple[float | None, float | None, float | None]:
    """(delta_lo, delta_hi, p) for b - a; all None without two repetitions per side.

    Estimates are compared on the scale their intervals use. Two "log_t" estimates
    are tested on ln(values), i.e. on the ratio of geometric means; the ratio's
    interval [r_lo, r_hi] is reported as the difference a.mean x (r - 1). Estimates
    combined by different methods (one side had a zero) are not tested.
    """
    if a.n < 2 or b.n < 2 or a.std is None or b.std is None:
        return None, None, None
    if (a.method == "log_t") != (b.method == "log_t"):
        return None, None, None
    if a.method == "log_t":
        if a.log_std is None or b.log_std is None:
            return None, None, None
        lo, hi, p = _welch_moments(
            math.log(a.mean), a.log_std, a.n, math.log(b.mean), b.log_std, b.n, confidence
        )
        return a.mean * math.expm1(lo), a.mean * math.expm1(hi), p
    return _welch_moments(a.mean, a.std, a.n, b.mean, b.std, b.n, confidence)


def _within_tol(
    metric: str, a: float, d: float, rel_tol: float, abs_tol: Mapping[str, float]
) -> tuple[bool, str]:
    if metric in abs_tol:
        return abs(d) <= abs_tol[
            metric
        ], f"|Δ| {abs(d):.4g} vs absolute tolerance {abs_tol[metric]:g}"
    if a == 0:
        return d == 0, "A is zero; only an exact match is within tolerance"
    return abs(d) <= rel_tol * abs(a), f"|Δ| {pct(abs(d / a))} vs tolerance {pct(rel_tol)}"


def compare_metric(
    metric: str,
    a: Estimate | None,
    b: Estimate | None,
    *,
    rel_tol: float,
    abs_tol: Mapping[str, float],
    alpha: float,
    confidence: float,
) -> MetricDelta:
    if a is None or b is None:
        side = "A" if a is None else "B"
        return MetricDelta(
            metric=metric,
            a=a,
            b=b,
            delta=None,
            delta_lo=None,
            delta_hi=None,
            rel_delta=None,
            p_value=None,
            within_normal_variance=False,
            reason=f"missing in {side}",
        )
    d = b.mean - a.mean
    lo, hi, p = _welch(a, b, confidence)
    ok, tol_text = _within_tol(metric, a.mean, d, rel_tol, abs_tol)
    if ok:
        within, reason = True, f"within tolerance ({tol_text})"
    elif p is None:
        why = (
            "CI methods differ"
            if a.n >= 2 and b.n >= 2 and a.method != b.method
            else "fewer than 2 repetitions"
        )
        within, reason = False, f"outside tolerance ({tol_text}); {why}, no test"
    elif p >= alpha:
        within, reason = (
            True,
            f"outside tolerance ({tol_text}) but not significant (Welch p={p:.3f})",
        )
    else:
        within, reason = False, f"outside tolerance ({tol_text}) and significant (Welch p={p:.3g})"
    return MetricDelta(
        metric=metric,
        a=a,
        b=b,
        delta=d,
        delta_lo=lo,
        delta_hi=hi,
        rel_delta=None if a.mean == 0 else d / abs(a.mean),
        p_value=p,
        within_normal_variance=within,
        reason=reason,
    )


def compare_cost(
    metric: str, a: MicrosRange | None, b: MicrosRange | None, rel_tol: float
) -> CostDelta:
    if a is None or a.value is None or b is None or b.value is None:
        both_missing = (a is None or a.value is None) and (b is None or b.value is None)
        return CostDelta(
            metric=metric,
            a=a,
            b=b,
            delta_micros=None,
            rel_delta=None,
            within_normal_variance=both_missing,
            reason="no cost on either side" if both_missing else "cost missing on one side",
        )
    d = b.value - a.value
    rel = None if a.value == 0 else d / a.value
    if (d == 0) or (rel is not None and abs(rel) <= rel_tol):
        within, reason = True, f"within tolerance ({pct(abs(rel or 0))} vs {pct(rel_tol)})"
    else:
        inf = 2**63
        overlap = (a.lo or 0) <= (inf if b.hi is None else b.hi) and (b.lo or 0) <= (
            inf if a.hi is None else a.hi
        )
        within = overlap
        reason = (
            "outside tolerance but the cost CIs overlap"
            if overlap
            else ("outside tolerance and the cost CIs do not overlap")
        )
    return CostDelta(
        metric=metric,
        a=a,
        b=b,
        delta_micros=d,
        rel_delta=rel,
        within_normal_variance=within,
        reason=reason,
    )


def compare(
    a: Sequence[ConfigResult] | Iterable[BenchRun],
    b: Sequence[ConfigResult] | Iterable[BenchRun],
    *,
    match_by: MatchBy = "config_hash",
    metrics: Sequence[str] = DEFAULT_METRICS,
    rel_tol: float = 0.10,
    abs_tol: Mapping[str, float] = DEFAULT_ABS_TOL,
    alpha: float = 0.05,
    confidence: float = 0.95,
    label_a: str = "A",
    label_b: str = "B",
) -> Comparison:
    if rel_tol < 0 or not 0 < alpha < 1:
        raise ValueError("need rel_tol >= 0 and 0 < alpha < 1")
    side_a = _index(_sweeps(a), match_by, label_a)
    side_b = _index(_sweeps(b), match_by, label_b)
    need_hash = match_by != "workload"
    opts: dict[str, Any] = dict(
        rel_tol=rel_tol, abs_tol=abs_tol, alpha=alpha, confidence=confidence
    )

    only_a = [
        f"{s.name} {s.workload} {s.load_mode.value}" for k, s in side_a.items() if k not in side_b
    ]
    only_b = [
        f"{s.name} {s.workload} {s.load_mode.value}" for k, s in side_b.items() if k not in side_a
    ]
    configs: list[ConfigComparison] = []
    points: list[PointComparison] = []
    for k in sorted(set(side_a) & set(side_b)):
        sa, sb = side_a[k], side_b[k]
        hash_match = sa.config_hash == sb.config_hash
        pb = {p.load: p for p in sb.points}
        pa = {p.load: p for p in sa.points}
        unit = "req/s" if sa.load_mode is LoadMode.OPEN_LOOP else "concurrent"
        only_a += [f"{sa.name} {sa.workload} @ {load:g} {unit}" for load in pa if load not in pb]
        only_b += [f"{sb.name} {sb.workload} @ {load:g} {unit}" for load in pb if load not in pa]
        for load in sorted(set(pa) & set(pb)):
            deltas = [
                compare_metric(m, pa[load].aggregate.get(m), pb[load].aggregate.get(m), **opts)
                for m in metrics
                if pa[load].aggregate.get(m) is not None or pb[load].aggregate.get(m) is not None
            ]
            points.append(
                PointComparison(
                    workload=sa.workload,
                    load_mode=sa.load_mode,
                    load=load,
                    name_a=sa.name,
                    name_b=sb.name,
                    config_hash_a=sa.config_hash,
                    config_hash_b=sb.config_hash,
                    config_hash_match=hash_match,
                    metrics=deltas,
                    within_normal_variance=all(d.within_normal_variance for d in deltas)
                    and (hash_match or not need_hash),
                )
            )
        ra, rb = sa.result, sb.result
        if ra is not None and rb is not None:
            gmetrics = [
                compare_metric(
                    "goodput.output_tok_s", ra.goodput.output_tok_s, rb.goodput.output_tok_s, **opts
                )
            ]
            costs = [
                compare_cost(
                    f"cost.{m}",
                    getattr(ra.cost, m) if ra.cost else None,
                    getattr(rb.cost, m) if rb.cost else None,
                    rel_tol,
                )
                for m in COST_METRICS
            ]
            if ra.goodput.output_tok_s is None and rb.goodput.output_tok_s is None:
                gmetrics = []
            configs.append(
                ConfigComparison(
                    workload=ra.workload,
                    load_mode=ra.load_mode,
                    name_a=ra.name,
                    name_b=rb.name,
                    config_hash_a=ra.config_hash,
                    config_hash_b=rb.config_hash,
                    config_hash_match=hash_match,
                    goodput_load_a=ra.goodput.max_load,
                    goodput_load_b=rb.goodput.max_load,
                    goodput_bracket_a=goodput_bracket(ra.goodput),
                    goodput_bracket_b=goodput_bracket(rb.goodput),
                    goodput_tie=brackets_overlap(ra.goodput, rb.goodput),
                    metrics=gmetrics,
                    cost=costs,
                    within_normal_variance=all(m.within_normal_variance for m in gmetrics)
                    and all(c.within_normal_variance for c in costs)
                    and (hash_match or not need_hash),
                )
            )

    return Comparison(
        label_a=label_a,
        label_b=label_b,
        match_by=match_by,
        rel_tol=rel_tol,
        abs_tol=dict(abs_tol),
        alpha=alpha,
        confidence=confidence,
        configs=configs,
        points=points,
        only_in_a=only_a,
        only_in_b=only_b,
        within_normal_variance=bool(points)
        and not only_a
        and not only_b
        and all(p.within_normal_variance for p in points)
        and all(c.within_normal_variance for c in configs),
    )


def _verdict(ok: bool) -> str:
    return "within" if ok else "OUTSIDE"


def _est(e: Estimate | None) -> str:
    if e is None:
        return "n/a"
    if e.lo is None or e.hi is None:
        return f"{num(e.mean, 3)} (n={e.n})"
    return f"{num(e.mean, 3)} [{num(e.lo, 3)}, {num(e.hi, 3)}]"


def _delta(d: MetricDelta) -> str:
    if d.delta is None:
        return "n/a"
    rel = "" if d.rel_delta is None else f" ({d.rel_delta:+.1%})"
    ci = "" if d.delta_lo is None else f" [{num(d.delta_lo, 3)}, {num(d.delta_hi, 3)}]"
    return f"{d.delta:+,.3f}{ci}{rel}"


def _cost(r: MicrosRange | None) -> str:
    if r is None or r.value is None:
        return "n/a"
    return f"{usd(r.value)} [{usd(r.lo)}, {usd(r.hi)}]"


METRIC_HEADERS = ("Metric", "A", "B", "Δ = B − A [CI] (rel)", "Welch p", "Verdict", "Reason")


def _metric_rows(deltas: Iterable[MetricDelta]) -> list[list[Any]]:
    return [
        [
            d.metric,
            _est(d.a),
            _est(d.b),
            _delta(d),
            "n/a" if d.p_value is None else f"{d.p_value:.3g}",
            _verdict(d.within_normal_variance),
            d.reason,
        ]
        for d in deltas
    ]


def render_markdown(c: Comparison) -> str:
    n_bad = (
        sum(not d.within_normal_variance for p in c.points for d in p.metrics)
        + sum(not d.within_normal_variance for cfg in c.configs for d in cfg.metrics)
        + sum(not e.within_normal_variance for cfg in c.configs for e in cfg.cost)
    )
    head = (
        "**Verdict: within normal variance**"
        if c.within_normal_variance
        else f"**Verdict: outside normal variance** ({n_bad} metrics outside; "
        f"{len(c.only_in_a) + len(c.only_in_b)} unmatched)"
    )
    abs_tol = ", ".join(f"{k} ±{v:g}" for k, v in c.abs_tol.items())
    parts = [
        f"# Comparison: {c.label_a} vs {c.label_b}",
        "",
        head,
        "",
        f"Matched by {c.match_by}. A metric is within normal variance when |B − A| is within "
        f"{pct(c.rel_tol)} of A (absolute: {abs_tol or 'none'}), or otherwise when Welch's "
        f"t-test on the repetitions gives p ≥ {c.alpha:g} (on ln(values), the ratio of "
        "geometric means, for latency and throughput). Cost at SLO uses CI overlap instead "
        f"of the t-test. Intervals are {c.confidence:.0%} CIs."
        + ("" if c.match_by == "workload" else " Config hashes must match exactly."),
    ]
    for cfg in c.configs:
        parts += [
            "",
            f"## {cfg.name_a} vs {cfg.name_b}: {cfg.workload} ({cfg.load_mode.value}) at goodput",
            "",
            f"Config hash {'matches' if cfg.config_hash_match else 'differs'}: "
            f"`{cfg.config_hash_a}` vs `{cfg.config_hash_b}`. Goodput load "
            f"{cfg.goodput_bracket_a or 'none'} vs {cfg.goodput_bracket_b or 'none'}"
            + (
                ": the brackets overlap, a tie within the search resolution; goodput and "
                "cost below come from the same grid point, so compare latency at equal load"
                if cfg.goodput_tie
                else ""
            )
            + f". Verdict: {_verdict(cfg.within_normal_variance)}.",
            "",
            md_table(METRIC_HEADERS, _metric_rows(cfg.metrics)),
            "",
            md_table(
                ("Cost", "A", "B", "Δ micros (rel)", "Verdict", "Reason"),
                [
                    [
                        d.metric,
                        _cost(d.a),
                        _cost(d.b),
                        "n/a"
                        if d.delta_micros is None
                        else f"{d.delta_micros:+d}"
                        + ("" if d.rel_delta is None else f" ({d.rel_delta:+.1%})"),
                        _verdict(d.within_normal_variance),
                        d.reason,
                    ]
                    for d in cfg.cost
                ],
            ),
        ]
    for p in c.points:
        unit = "req/s" if p.load_mode is LoadMode.OPEN_LOOP else "concurrent"
        parts += [
            "",
            f"## {p.name_a} vs {p.name_b}: {p.workload} @ {p.load:g} {unit} — "
            f"{_verdict(p.within_normal_variance)}",
            "",
        ]
        if not p.config_hash_match:
            parts += [f"Config hash differs: `{p.config_hash_a}` vs `{p.config_hash_b}`.", ""]
        parts.append(md_table(METRIC_HEADERS, _metric_rows(p.metrics)))
    for label, missing in ((c.label_a, c.only_in_a), (c.label_b, c.only_in_b)):
        if missing:
            parts += ["", f"**Only in {label}:** " + "; ".join(missing)]
    return "\n".join(parts) + "\n"


def render_json(c: Comparison) -> str:
    return c.model_dump_json(indent=2) + "\n"
