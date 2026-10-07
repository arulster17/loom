"""The search resolution of goodput: its bracket, and when two configs tie inside it.

Goodput is found by a search over a grid of loads, so it is only known to lie in a
bracket: at least the highest passing load, below the lowest failing one
(`goodput_bounds`). Below saturation a server delivers what it is offered, so two
configs whose brackets overlap get the same goodput, throughput and cost at SLO from
the search alone. That is a tie within the search resolution, not a measured
equality (`brackets_overlap`); latency at equal load (`equal_load`) can still tell
such configs apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from loom_bench.report.analyze import ConfigResult
from loom_bench.slo import GoodputResult


def goodput_bounds(g: GoodputResult) -> tuple[float, float] | None:
    """[low, high) that the true goodput lies in; None when no tested load met the SLO.

    High is the lowest failing load, or infinity when no tested load failed.
    """
    if g.max_load is None:
        return None
    high = math.inf if g.first_failing_load is None else g.first_failing_load
    return g.max_load, high


def brackets_overlap(a: GoodputResult, b: GoodputResult) -> bool:
    """Whether the search cannot tell the two goodputs apart: both have a goodput and
    their brackets share some load."""
    ba, bb = goodput_bounds(a), goodput_bounds(b)
    if ba is None or bb is None:
        return False
    return ba[0] < bb[1] and bb[0] < ba[1]


def goodput_ties(result: ConfigResult, others: Sequence[ConfigResult]) -> list[ConfigResult]:
    """The configs in `others` on the same workload and load mode whose goodput
    bracket overlaps `result`'s."""
    return [
        o
        for o in others
        if o is not result
        and o.workload == result.workload
        and o.load_mode is result.load_mode
        and brackets_overlap(result.goodput, o.goodput)
    ]
