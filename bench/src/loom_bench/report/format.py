"""Display formatting shared by every report. Money stays integer micros until here."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from functools import cache
from typing import Any

from jinja2 import BaseLoader, Environment, PackageLoader, StrictUndefined, select_autoescape

from loom_bench.cost import MicrosRange
from loom_bench.money import Micros, format_usd
from loom_bench.records import LoadMode
from loom_bench.slo import GoodputResult, Slo
from loom_bench.stats import Estimate

NA = "n/a"
CSV_USD_PLACES = 6  # exact: one micro-dollar
CSV_FLOAT_PLACES = 4
MIN_SIG_FIGS = 2  # small values never round to "0"
MAX_PLACES = 6

# Explains the "+" goodput notation; shown once per table, not per row.
UNBRACKETED_NOTE = (
    "+ after a goodput load: not bracketed. Every tested load met the SLO, so goodput is at "
    "least that load (it may be higher) and cost at SLO is an upper bound."
)


def usd(m: Micros | None, places: int = 4) -> str:
    return NA if m is None else format_usd(m, places)


def usd_ci(r: MicrosRange | None) -> str:
    """Value with its CI, e.g. `$1.2924 [1.2610, 1.3252]`; a missing high bound is unbounded.

    A price the cost allocation does not apply to reads `n/a (<reason>)`, never $0.
    """
    if r is not None and r.na_reason:
        return f"{NA} ({r.na_reason})"
    if r is None or r.value is None:
        return NA
    if r.lo is None and r.hi is None:
        return f"{usd(r.value)} (no CI)"
    lo = NA if r.lo is None else usd(r.lo).lstrip("$")
    hi = "unbounded" if r.hi is None else usd(r.hi).lstrip("$")
    return f"{usd(r.value)} [{lo}, {hi}]"


def num(x: float | None, places: int = 0) -> str:
    return NA if x is None else f"{x:,.{places}f}"


def places_for(values: Iterable[float | None], places: int = 0) -> int:
    """At least `places` decimals, more if needed to show the smallest non-zero value
    with MIN_SIG_FIGS significant figures (0.23 s, not 0 s); at most MAX_PLACES."""
    smallest = min((abs(v) for v in values if v is not None and v != 0), default=None)
    if smallest is None:
        return places
    needed = MIN_SIG_FIGS - 1 - math.floor(math.log10(smallest))
    return min(max(places, needed), max(places, MAX_PLACES))


def sig(x: float | None, places: int = 0) -> str:
    """`num` with at least MIN_SIG_FIGS significant figures: 0.23, 1.3, 95, 1,152."""
    return NA if x is None else num(x, places_for([x], places))


def est(e: Estimate | None, places: int = 0, unit: str = "") -> str:
    """Point estimate with its CI, e.g. `354 [338, 372] ms`; a single run shows
    `(n=1, no CI)`. All three numbers share the decimals that give the smallest of them
    two significant figures (`0.130 [0.091, 0.190]`), so nothing rounds to 0."""
    if e is None:
        return NA
    suffix = f" {unit}" if unit else ""
    p = places_for([e.mean, e.lo, e.hi], places)
    if e.lo is None or e.hi is None:
        return f"{num(e.mean, p)}{suffix} (n={e.n}, no CI)"
    return f"{num(e.mean, p)} [{num(e.lo, p)}, {num(e.hi, p)}]{suffix}"


def pct(x: Fraction | float) -> str:
    """Whole percent from 10% up, one decimal below."""
    value = float(x) * 100
    return f"{value:.0f}%" if abs(value) >= 10 else f"{value:.1f}%"


def load_unit(mode: LoadMode) -> str:
    return "req/s" if mode is LoadMode.OPEN_LOOP else "concurrency"


def load_value(load: float | None, mode: LoadMode, *, at_least: bool = False) -> str:
    """`8 req/s` or `8 concurrent`; `at_least` marks an unbracketed goodput: `8+ req/s`."""
    if load is None:
        return NA
    value = f"{load:g}{'+' if at_least else ''}"
    return f"{value} req/s" if mode is LoadMode.OPEN_LOOP else f"{value} concurrent"


def goodput_load(g: GoodputResult) -> str:
    """The goodput load, with `+` when no tested load failed (see UNBRACKETED_NOTE)."""
    if g.max_load is None:
        return "none met the SLO"
    return load_value(g.max_load, g.load_mode, at_least=not g.bracketed)


def any_unbracketed(goodputs: Iterable[GoodputResult]) -> bool:
    return any(g.max_load is not None and not g.bracketed for g in goodputs)


def load_mode_label(mode: LoadMode) -> str:
    if mode is LoadMode.OPEN_LOOP:
        return "open loop (requests arrive on a fixed schedule at a set rate, req/s)"
    return "closed loop (a fixed number of concurrent clients, each sending on completion)"


def describe_slo(slo: Slo) -> str:
    parts = [
        f"{metric.removesuffix('_ms').upper()} {pctl} ≤ {target:g} ms"
        for metric, pctl, target in slo.targets()
    ]
    parts.append(f"error rate ≤ {slo.max_error_rate:.2%}")
    return ", ".join(parts)


def md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(md_cell(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(md_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(lines)


def to_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if row.get(k) is None else row[k] for k in columns})
    return buf.getvalue()


def micros_columns(prefix: str, r: MicrosRange | None) -> dict[str, Any]:
    """`<prefix>_micros`, `_lo_micros`, `_hi_micros`, the same as formatted USD, and
    `<prefix>_na_reason` (why the price is not applicable; its micros are then empty)."""
    out: dict[str, Any] = {}
    for end in ("", "_lo", "_hi"):
        value = None if r is None else getattr(r, end.lstrip("_") or "value")
        out[f"{prefix}{end}_micros"] = value
        out[f"{prefix}{end}_usd"] = None if value is None else usd(value, CSV_USD_PLACES)
    out[f"{prefix}_na_reason"] = None if r is None else r.na_reason
    return out


def micros_column_names(prefix: str) -> list[str]:
    return [
        *(f"{prefix}{end}_{kind}" for end in ("", "_lo", "_hi") for kind in ("micros", "usd")),
        f"{prefix}_na_reason",
    ]


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, CSV_FLOAT_PLACES)


def estimate_columns(prefix: str, e: Estimate | None) -> dict[str, Any]:
    """Point, bounds and `<prefix>_ci_method` (`Estimate.method`: t, t_clipped, log_t)."""
    return {
        prefix: None if e is None else _round(e.mean),
        f"{prefix}_lo": None if e is None else _round(e.lo),
        f"{prefix}_hi": None if e is None else _round(e.hi),
        f"{prefix}_ci_method": None if e is None else e.method,
    }


def estimate_column_names(prefix: str) -> list[str]:
    return [prefix, f"{prefix}_lo", f"{prefix}_hi", f"{prefix}_ci_method"]


# Jinja filters and globals every HTML page (reports and the site) uses.
FILTERS: dict[str, Any] = {
    "usd": usd,
    "usd_ci": usd_ci,
    "est": est,
    "num": num,
    "sig": sig,
    "pct": pct,
}
GLOBALS: dict[str, Any] = {
    "describe_slo": describe_slo,
    "load_mode_label": load_mode_label,
    "load_value": load_value,
    "goodput_load": goodput_load,
    "places_for": places_for,
    "any_unbracketed": any_unbracketed,
    "unbracketed_note": UNBRACKETED_NOTE,
}


def jinja_env(loader: BaseLoader) -> Environment:
    """An HTML environment with the shared settings, FILTERS and GLOBALS."""
    env = Environment(
        loader=loader,
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(FILTERS)
    env.globals.update(GLOBALS)
    return env


@cache
def html_env() -> Environment:
    return jinja_env(PackageLoader("loom_bench.report", "templates"))
