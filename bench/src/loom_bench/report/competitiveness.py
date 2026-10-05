"""Competitiveness view: our cost at SLO and planned price against public list prices.

Per registry model and workload, "our cost" is the cost at SLO of the leaderboard's
top-ranked config (trusted, not failing the quality gate). Competitor prices come
from `bench/competitors.yaml`: public list prices only, never measured. Aggregators
and entries whose availability is unverified are listed but, by default, left out
of the market comparison behind the `assess()` flags.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from pydantic import BaseModel

from loom_bench.competitiveness import CostPerMtok, assess
from loom_bench.cost import MicrosRange
from loom_bench.money import Micros
from loom_bench.prices import Competitors, PriceBook
from loom_bench.records import LoadMode
from loom_bench.registry import ModelSpec, Pricing, Registry
from loom_bench.report.analyze import ConfigResult
from loom_bench.report.format import (
    html_env,
    md_table,
    micros_column_names,
    micros_columns,
    pct,
    to_csv,
    usd,
    usd_ci,
)
from loom_bench.report.leaderboard import RowStatus, rank
from loom_bench.report.methodology import Methodology, methodology, methodology_markdown

PUBLIC_PRICES_NOTE = (
    "Public list prices only: competitor numbers are the per-token prices each provider "
    "publishes on its pricing page, with the source and the date it was checked. No "
    "competitor endpoint was called or benchmarked, and list prices say nothing about a "
    "provider's latency, quality or quantization unless the page discloses it."
)


class CompetitorPrice(BaseModel):
    provider: str
    provider_model: str | None
    aggregator: bool
    availability: str
    input_per_mtok: Micros
    output_per_mtok: Micros
    quantization: str | None
    source: str
    last_checked: dt.date
    notes: str | None
    in_comparison: bool  # counted by the price_above_market flag


class FlagView(BaseModel):
    kind: str
    side: str | None
    message: str


class Margin(BaseModel):
    """Our price minus our cost at SLO, per 1M tokens; `worst` uses the cost CI high bound."""

    value: Micros
    worst: Micros | None
    fraction_of_price: float | None


class CompetitivenessRow(BaseModel):
    model_id: str
    display_name: str
    workload: str | None
    load_mode: LoadMode | None
    best_config: str | None
    best_config_hash: str | None
    cost_input: MicrosRange | None
    cost_output: MicrosRange | None
    price: Pricing | None
    margin_input: Margin | None
    margin_output: Margin | None
    competitors: list[CompetitorPrice]
    flags: list[FlagView]


class NotOfferedView(BaseModel):
    name: str
    reason: str
    url: str | None
    last_checked: dt.date


class CompetitivenessReport(BaseModel):
    title: str
    note: str
    rows: list[CompetitivenessRow]
    not_offered: list[NotOfferedView]
    competitors_last_checked: dt.date
    include_aggregators: bool
    include_unverified: bool
    methodology: Methodology


def _margin(price: Micros, cost: MicrosRange | None) -> Margin | None:
    if cost is None or cost.value is None:
        return None
    return Margin(
        value=price - cost.value,
        worst=None if cost.hi is None else price - cost.hi,
        fraction_of_price=None if price == 0 else float(Fraction(price - cost.value, price)),
    )


def _competitors(
    spec: ModelSpec, competitors: Competitors, include_aggregators: bool, include_unverified: bool
) -> list[CompetitorPrice]:
    return [
        CompetitorPrice(
            provider=p.name,
            provider_model=e.provider_model,
            aggregator=p.aggregator,
            availability=e.availability,
            input_per_mtok=e.input_per_mtok,
            output_per_mtok=e.output_per_mtok,
            quantization=e.quantization,
            source=str(e.source),
            last_checked=p.last_checked,
            notes=e.notes,
            in_comparison=(include_aggregators or not p.aggregator)
            and (include_unverified or e.availability == "listed"),
        )
        for p, e in competitors.entries_for(spec.id)
    ]


def _row(
    spec: ModelSpec,
    workload: str | None,
    mode: LoadMode | None,
    best: ConfigResult | None,
    competitors: Competitors,
    include_aggregators: bool,
    include_unverified: bool,
) -> CompetitivenessRow:
    cost = best.cost if best else None
    cin, cout = (cost.input_per_mtok, cost.output_per_mtok) if cost else (None, None)
    # A side the allocation does not price (na_reason) has no cost and no margin; a
    # side that should be priced but is not (zero throughput) means no measurement.
    priced = [r for r in (cin, cout) if r is not None and not r.na_reason]
    measured = (
        CostPerMtok(
            input=None if cin is None else cin.value,
            output=None if cout is None else cout.value,
        )
        if priced and all(r.value is not None for r in priced)
        else None
    )
    flags = assess(
        spec.id,
        spec.pricing,
        measured,
        competitors,
        include_aggregators=include_aggregators,
        include_unverified=include_unverified,
    )
    price = spec.pricing
    return CompetitivenessRow(
        model_id=spec.id,
        display_name=spec.display_name,
        workload=workload,
        load_mode=mode,
        best_config=best.name if best else None,
        best_config_hash=best.config_hash if best else None,
        cost_input=cin,
        cost_output=cout,
        price=price,
        margin_input=_margin(price.input_per_mtok, cin) if price else None,
        margin_output=_margin(price.output_per_mtok, cout) if price else None,
        competitors=_competitors(spec, competitors, include_aggregators, include_unverified),
        flags=[FlagView(kind=f.kind.value, side=f.side, message=f.message) for f in flags],
    )


def build_competitiveness(
    results: Sequence[ConfigResult],
    registry: Registry,
    competitors: Competitors,
    *,
    price_book: PriceBook | None = None,
    include_aggregators: bool = False,
    include_unverified: bool = False,
    title: str = "Loom competitiveness: cost at SLO vs public list prices",
) -> CompetitivenessReport:
    """One row per registry model and benchmarked workload (one row if never benchmarked).

    Results are matched to registry models by Hugging Face repo.
    """
    names = {r.config_hash: r.name for r in results}
    rows: list[CompetitivenessRow] = []
    used: list[ConfigResult] = []
    args = (competitors, include_aggregators, include_unverified)
    for spec in registry.models:
        groups: dict[tuple[str, LoadMode], list[ConfigResult]] = defaultdict(list)
        for r in results:
            if r.model_repo == spec.hf.repo:
                groups[(r.workload, r.load_mode)].append(r)
        if not groups:
            rows.append(_row(spec, None, None, None, *args))
        for (workload, mode), members in sorted(groups.items()):
            ranked = [row for row in rank(members, names=names) if row.status is RowStatus.RANKED]
            best = ranked[0].result if ranked else None
            if best is not None:
                used.append(best)
            rows.append(_row(spec, workload, mode, best, *args))
    return CompetitivenessReport(
        title=title,
        note=PUBLIC_PRICES_NOTE,
        rows=rows,
        not_offered=[
            NotOfferedView(
                name=n.name,
                reason=n.reason,
                url=None if n.url is None else str(n.url),
                last_checked=n.last_checked,
            )
            for n in competitors.not_offered
        ],
        competitors_last_checked=competitors.last_checked,
        include_aggregators=include_aggregators,
        include_unverified=include_unverified,
        methodology=methodology(used, price_book),
    )


def margin_text(m: Margin | None, cost: MicrosRange | None = None) -> str:
    """Margin with its share of price; `cost` explains an n/a margin (allocation)."""
    if m is None:
        return f"n/a ({cost.na_reason})" if cost is not None and cost.na_reason else "n/a"
    share = "" if m.fraction_of_price is None else f" ({pct(m.fraction_of_price)} of price)"
    worst = "" if m.worst is None else f"; {usd(m.worst)} at cost CI high"
    return f"{usd(m.value)}{share}{worst}"


def price_text(p: Pricing | None, side: str) -> str:
    return "not set" if p is None else usd(getattr(p, f"{side}_per_mtok"))


def row_title(r: CompetitivenessRow) -> str:
    if r.workload is None:
        return "not benchmarked"
    return f"{r.workload} ({r.load_mode.value.replace('_', ' ') if r.load_mode else ''})"


def comparison_scope(report: CompetitivenessReport) -> str:
    left_out = []
    if not report.include_aggregators:
        left_out.append("aggregators")
    if not report.include_unverified:
        left_out.append("entries whose availability is unverified")
    if not left_out:
        return "Flags compare against every listed entry."
    return f"Flags leave out {' and '.join(left_out)}; they are listed for reference."


OURS_HEADERS = (
    "Workload",
    "Best config at SLO",
    "Our cost $/1M in",
    "Our cost $/1M out",
    "Our price $/1M in",
    "Our price $/1M out",
    "Margin in",
    "Margin out",
)
COMPETITOR_HEADERS = (
    "Provider",
    "Provider model",
    "$/1M in",
    "$/1M out",
    "Quantization",
    "Availability",
    "In flag comparison",
    "Source",
    "Last checked",
)


def _ours_md(r: CompetitivenessRow) -> list[Any]:
    return [
        row_title(r),
        r.best_config or "none ranked",
        usd_ci(r.cost_input),
        usd_ci(r.cost_output),
        price_text(r.price, "input"),
        price_text(r.price, "output"),
        margin_text(r.margin_input, r.cost_input),
        margin_text(r.margin_output, r.cost_output),
    ]


def _competitor_md(c: CompetitorPrice) -> list[Any]:
    return [
        c.provider + (" (aggregator)" if c.aggregator else ""),
        c.provider_model or "",
        usd(c.input_per_mtok),
        usd(c.output_per_mtok),
        c.quantization or "not disclosed",
        c.availability,
        "yes" if c.in_comparison else "no",
        c.source,
        c.last_checked.isoformat(),
    ]


def render_markdown(report: CompetitivenessReport) -> str:
    parts = [f"# {report.title}", "", f"> {report.note}", ""]
    parts.append(
        f"Competitor prices last checked {report.competitors_last_checked.isoformat()}. "
        f"{comparison_scope(report)} Margin is our price minus our cost at SLO under the "
        "stated cost allocation."
    )
    by_model: dict[str, list[CompetitivenessRow]] = defaultdict(list)
    for r in report.rows:
        by_model[r.model_id].append(r)
    for model_id, rows in by_model.items():
        first = rows[0]
        parts += ["", f"## {first.display_name} (`{model_id}`)", ""]
        parts.append(md_table(OURS_HEADERS, map(_ours_md, rows)))
        parts += ["", "**Public list prices**", ""]
        if first.competitors:
            parts.append(md_table(COMPETITOR_HEADERS, map(_competitor_md, first.competitors)))
        else:
            parts.append("No public list price recorded for this model.")
        flags = [(r, f) for r in rows for f in r.flags]
        parts += ["", "**Flags**", ""]
        if flags:
            parts += [f"- [{row_title(r)}] `{f.kind}`: {f.message}" for r, f in flags]
        else:
            parts.append("- none")
    if report.not_offered:
        parts += ["", "## Providers with no public per-token price", ""]
        parts += [
            f"- {n.name}: {n.reason} (checked {n.last_checked.isoformat()})"
            for n in report.not_offered
        ]
    parts += ["", methodology_markdown(report.methodology)]
    return "\n".join(parts)


def render_html(report: CompetitivenessReport) -> str:
    by_model: dict[str, list[CompetitivenessRow]] = defaultdict(list)
    for r in report.rows:
        by_model[r.model_id].append(r)
    return (
        html_env()
        .get_template("competitiveness.html.j2")
        .render(
            report=report,
            by_model=by_model,
            scope=comparison_scope(report),
            row_title=row_title,
            margin_text=margin_text,
            price_text=price_text,
        )
    )


CSV_COLUMNS = [
    "model_id",
    "workload",
    "load_mode",
    "best_config",
    "best_config_hash",
    *micros_column_names("cost_input_per_mtok"),
    *micros_column_names("cost_output_per_mtok"),
    "price_input_per_mtok_micros",
    "price_input_per_mtok_usd",
    "price_output_per_mtok_micros",
    "price_output_per_mtok_usd",
    "margin_input_micros",
    "margin_input_worst_micros",
    "margin_output_micros",
    "margin_output_worst_micros",
    "flags",
    "competitor",
    "competitor_model",
    "competitor_aggregator",
    "competitor_availability",
    "competitor_in_comparison",
    "competitor_input_per_mtok_micros",
    "competitor_input_per_mtok_usd",
    "competitor_output_per_mtok_micros",
    "competitor_output_per_mtok_usd",
    "competitor_quantization",
    "competitor_source",
    "competitor_last_checked",
    "price_basis",
    "public_list_prices_only",
]


def render_csv(report: CompetitivenessReport) -> str:
    bases = {c.config_hash: c.price.basis for c in report.methodology.configs}
    out = []
    for r in report.rows:
        base = {
            "model_id": r.model_id,
            "workload": r.workload,
            "load_mode": r.load_mode.value if r.load_mode else None,
            "best_config": r.best_config,
            "best_config_hash": r.best_config_hash,
            **micros_columns("cost_input_per_mtok", r.cost_input),
            **micros_columns("cost_output_per_mtok", r.cost_output),
            "price_input_per_mtok_micros": r.price.input_per_mtok if r.price else None,
            "price_input_per_mtok_usd": usd(r.price.input_per_mtok, 6) if r.price else None,
            "price_output_per_mtok_micros": r.price.output_per_mtok if r.price else None,
            "price_output_per_mtok_usd": usd(r.price.output_per_mtok, 6) if r.price else None,
            "margin_input_micros": r.margin_input.value if r.margin_input else None,
            "margin_input_worst_micros": r.margin_input.worst if r.margin_input else None,
            "margin_output_micros": r.margin_output.value if r.margin_output else None,
            "margin_output_worst_micros": r.margin_output.worst if r.margin_output else None,
            "flags": "; ".join(f.kind + (f"({f.side})" if f.side else "") for f in r.flags),
            "price_basis": bases.get(r.best_config_hash or ""),
            "public_list_prices_only": True,
        }
        entries: list[CompetitorPrice | None] = [*r.competitors] or [None]
        for c in entries:
            row = dict(base)
            if c is not None:
                row |= {
                    "competitor": c.provider,
                    "competitor_model": c.provider_model,
                    "competitor_aggregator": c.aggregator,
                    "competitor_availability": c.availability,
                    "competitor_in_comparison": c.in_comparison,
                    "competitor_input_per_mtok_micros": c.input_per_mtok,
                    "competitor_input_per_mtok_usd": usd(c.input_per_mtok, 6),
                    "competitor_output_per_mtok_micros": c.output_per_mtok,
                    "competitor_output_per_mtok_usd": usd(c.output_per_mtok, 6),
                    "competitor_quantization": c.quantization,
                    "competitor_source": c.source,
                    "competitor_last_checked": c.last_checked,
                }
            out.append(row)
    return to_csv(out, CSV_COLUMNS)
