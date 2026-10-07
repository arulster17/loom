"""Latency of configs at the same offered load, where goodput cannot separate them.

Goodput ties within the search resolution (`resolution`) leave throughput and cost
equal, but the configs still differ in how fast they serve the same load. For every
load all configs on a board ran (same workload and load mode), this lines up their
latency percentiles: geometric means across repetitions with log-scale t intervals,
as everywhere else (`metrics.aggregate`).

A config is called lower on a metric only when its confidence interval lies
entirely below every other config's interval at that load; otherwise the verdict is
"no significant difference". Every metric here is lower-is-better.

Which loads are shown (`shown_loads`): every common load at or below the lowest
goodput, where all configs are compared inside the SLO, plus the highest common
load, which shows how they degrade past it. Without a goodput on every side, every
common load is shown.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from loom_bench.records import LoadMode
from loom_bench.report.analyze import ConfigResult, LoadPoint
from loom_bench.report.format import est, load_value, md_table
from loom_bench.slo import GoodputResult
from loom_bench.stats import Estimate

# (aggregate key, label); every one is lower-is-better.
EQUAL_LOAD_METRICS: tuple[tuple[str, str], ...] = (
    ("ttft_ms.p50", "TTFT p50"),
    ("ttft_ms.p95", "TTFT p95"),
    ("tpot_ms.p50", "TPOT p50"),
    ("tpot_ms.p95", "TPOT p95"),
    ("e2e_ms.p95", "E2E p95"),
)
NO_SIGNIFICANT_DIFFERENCE = "no significant difference"

# Loads from the same search grid are equal floats; rounding guards against noise in
# how a load was stored.
_LOAD_DIGITS = 9


class EqualLoadSide(BaseModel):
    """One config's sweep, from a ConfigResult or from raw runs (no SLO verdict)."""

    name: str
    points: list[LoadPoint]
    goodput: GoodputResult | None = None


class EqualLoadCell(BaseModel):
    """One config at one load: its latency estimates and the SLO verdict there."""

    name: str
    metrics: dict[str, Estimate | None]  # keyed as EQUAL_LOAD_METRICS
    slo_met: bool | None  # None when no SLO verdict is known


class EqualLoadPoint(BaseModel):
    load: float
    cells: list[EqualLoadCell]  # one per config, in board order
    lower: dict[str, str | None]  # metric -> the config significantly lower, else None

    def verdict(self, metric: str) -> str:
        name = self.lower.get(metric)
        return NO_SIGNIFICANT_DIFFERENCE if name is None else f"{name} lower"


class EqualLoadTable(BaseModel):
    workload: str
    load_mode: LoadMode
    names: list[str]
    points: list[EqualLoadPoint]


def side_of(r: ConfigResult) -> EqualLoadSide:
    return EqualLoadSide(name=r.name, points=r.points, goodput=r.goodput)


def _key(load: float) -> float:
    return round(load, _LOAD_DIGITS)


def shown_loads(sides: Sequence[EqualLoadSide]) -> list[float]:
    """Common loads at or below the lowest goodput, plus the highest common load; every
    common load when some side has no goodput."""
    if not sides:
        return []
    common = set.intersection(*({_key(p.load) for p in s.points} for s in sides))
    if not common:
        return []
    goodputs = [s.goodput.max_load if s.goodput else None for s in sides]
    if any(g is None for g in goodputs):
        return sorted(common)
    floor = min(_key(g) for g in goodputs if g is not None)
    return sorted({load for load in common if load <= floor} | {max(common)})


def disjoint_lower(estimates: Sequence[Estimate | None]) -> int | None:
    """Index of the estimate whose CI lies entirely below every other's, else None.

    Needs an interval on every side: a single repetition is never significant.
    """
    if len(estimates) < 2:
        return None
    ests: list[Estimate] = []
    for e in estimates:
        if e is None or e.lo is None or e.hi is None:
            return None
        ests.append(e)
    best = min(range(len(ests)), key=lambda i: ests[i].mean)
    hi = ests[best].hi
    assert hi is not None
    others = [e.lo for i, e in enumerate(ests) if i != best]
    return best if all(lo is not None and hi < lo for lo in others) else None


def _slo_met(side: EqualLoadSide, load: float) -> bool | None:
    if side.goodput is None:
        return None
    return next((p.met for p in side.goodput.points if _key(p.load) == load), None)


def equal_load(
    sides: Sequence[EqualLoadSide], workload: str, load_mode: LoadMode
) -> EqualLoadTable | None:
    """Latency of every side at the loads they all ran; None with fewer than two sides
    or no common load."""
    if len(sides) < 2:
        return None
    loads = shown_loads(sides)
    if not loads:
        return None
    by_load = [{_key(p.load): p for p in s.points} for s in sides]
    points = []
    for load in loads:
        cells = [
            EqualLoadCell(
                name=s.name,
                metrics={m: idx[load].aggregate.get(m) for m, _ in EQUAL_LOAD_METRICS},
                slo_met=_slo_met(s, load),
            )
            for s, idx in zip(sides, by_load, strict=True)
        ]
        lower: dict[str, str | None] = {}
        for metric, _ in EQUAL_LOAD_METRICS:
            best = disjoint_lower([c.metrics[metric] for c in cells])
            lower[metric] = None if best is None else cells[best].name
        points.append(EqualLoadPoint(load=load, cells=cells, lower=lower))
    return EqualLoadTable(
        workload=workload, load_mode=load_mode, names=[s.name for s in sides], points=points
    )


def equal_load_for(results: Sequence[ConfigResult]) -> EqualLoadTable | None:
    """The equal-load table of one leaderboard board (same workload and load mode)."""
    if not results:
        return None
    first = results[0]
    return equal_load([side_of(r) for r in results], first.workload, first.load_mode)


EQUAL_LOAD_NOTE = (
    "Latency at equal load: each config at the loads every config on this board ran (the "
    "loads at or below the lowest goodput, plus the highest common load). Geometric means "
    "across repetitions with 95% CIs; a config is lower on a metric only when its CI lies "
    "entirely below every other config's, otherwise there is no significant difference."
)
# Decimals per equal-load metric: TPOT is a few tens of ms, the rest hundreds.
EQUAL_LOAD_PLACES = {"tpot_ms.p50": 1, "tpot_ms.p95": 1}


def slo_text(met: bool | None) -> str:
    return "n/a" if met is None else ("met" if met else "failed")


def equal_load_rows(t: EqualLoadTable) -> list[list[str]]:
    """Markdown rows: per load, the SLO verdict, then one row per metric."""
    out: list[list[str]] = []
    for p in t.points:
        load = load_value(p.load, t.load_mode)
        out.append([load, "SLO at this load", *(slo_text(c.slo_met) for c in p.cells), ""])
        for metric, label in EQUAL_LOAD_METRICS:
            places = EQUAL_LOAD_PLACES.get(metric, 0)
            cells = []
            for c in p.cells:
                text = est(c.metrics.get(metric), places, "ms")
                cells.append(f"**{text}**" if p.lower[metric] == c.name else text)
            out.append([load, label, *cells, p.verdict(metric)])
    return out


def equal_load_markdown(t: EqualLoadTable) -> str:
    return md_table(["Load", "Metric", *t.names, "Verdict"], equal_load_rows(t))
