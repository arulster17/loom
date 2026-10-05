"""Display formatting shared by every report. Money stays integer micros until here."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from functools import cache
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape

from loom_bench.cost import MicrosRange
from loom_bench.money import Micros, format_usd
from loom_bench.records import LoadMode
from loom_bench.slo import Slo
from loom_bench.stats import Estimate

NA = "n/a"
CSV_USD_PLACES = 6  # exact: one micro-dollar
CSV_FLOAT_PLACES = 4


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


def est(e: Estimate | None, places: int = 0, unit: str = "") -> str:
    """Mean with its CI, e.g. `354 [338, 372] ms`; a single run shows `(n=1, no CI)`."""
    if e is None:
        return NA
    suffix = f" {unit}" if unit else ""
    if e.lo is None or e.hi is None:
        return f"{num(e.mean, places)}{suffix} (n={e.n}, no CI)"
    return f"{num(e.mean, places)} [{num(e.lo, places)}, {num(e.hi, places)}]{suffix}"


def pct(x: Fraction | float) -> str:
    """Whole percent from 10% up, one decimal below."""
    value = float(x) * 100
    return f"{value:.0f}%" if abs(value) >= 10 else f"{value:.1f}%"


def load_unit(mode: LoadMode) -> str:
    return "req/s" if mode is LoadMode.OPEN_LOOP else "concurrency"


def load_value(load: float | None, mode: LoadMode) -> str:
    if load is None:
        return NA
    return f"{load:g} req/s" if mode is LoadMode.OPEN_LOOP else f"{load:g} concurrent"


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


@cache
def html_env() -> Environment:
    env = Environment(
        loader=PackageLoader("loom_bench.report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    env.filters.update(usd=usd, usd_ci=usd_ci, est=est, num=num, pct=pct)
    return env
