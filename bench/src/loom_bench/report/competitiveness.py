"""Competitiveness view: our cost at SLO and planned price against public list prices.

Per registry model and workload, "our cost" is the cost at SLO of the board's headline
config (`leaderboard.headline_row`: trusted first, then quality verified, then rank;
never one that failed the quality gate) at the on-demand price, split between input
and output by measured prefill time (`cost.py`, prefill_time), plus the blended cost
of the workload's own token mix. Margins and break-even prices are therefore
reproducible from the price book. When no config has a cost at the declared SLO and
an alternative-SLO analysis is given, its figure is added as a separate row, labelled
as not the declared SLO.

Competitor prices come from `bench/competitors.yaml`: public list prices only, never
measured. Each is also shown at every workload's token mix, so blended compares like
with like. Aggregators and entries whose availability is unverified are listed but, by
default, left out of the market comparison behind the `assess()` flags.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from pydantic import BaseModel

from loom_bench.competitiveness import CostPerMtok, assess, like_for_like, price_at_mix
from loom_bench.cost import MicrosRange
from loom_bench.money import Micros
from loom_bench.prices import Competitors, PriceBook
from loom_bench.records import LoadMode
from loom_bench.registry import ModelSpec, Pricing, Registry
from loom_bench.report.analyze import ConfigResult
from loom_bench.report.format import (
    describe_slo,
    html_env,
    md_table,
    micros_column_names,
    micros_columns,
    pct,
    to_csv,
    usd,
    usd_ci,
)
from loom_bench.report.leaderboard import (
    LeaderboardRow,
    headline_row,
    no_headline_reason,
    rank,
    workload_order,
)
from loom_bench.report.methodology import Methodology, methodology, methodology_markdown

PUBLIC_PRICES_NOTE = (
    "Public list prices only: competitor numbers are the per-token prices each provider "
    "publishes on its pricing page, with the source and the date it was checked. No "
    "competitor endpoint was called or benchmarked, and list prices say nothing about a "
    "provider's latency, quality or quantization unless the page discloses it."
)

OUR_COST_NOTE = (
    "Our cost is the headline config's cost at SLO (see the summary) at the on-demand list "
    "price, storage included: $/1M input and $/1M output split the replica's cost by "
    "measured prefill time, and $/1M blended is its cost over all tokens at the workload's "
    "own input:output mix. Each public price is also shown at every workload's mix, so "
    "blended compares like with like. With no price set, the break-even price is the lowest "
    "price that covers our cost: at the point estimate, and at the cost CI high bound."
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
    in_comparison: bool  # counted by the *_above_market flags
    blended_at_mix: Micros | None = None  # this price list at the row's token mix


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
    cost_blended: MicrosRange | None = None
    basis: str | None = None  # why this config, or why there is no cost
    slo: str | None = None
    alternative_slo: bool = False  # the figure is under an alternative SLO, not the declared one
    tokens_in: float | None = None  # mean per request
    tokens_out: float | None = None
    quantization: str | None = None  # ours, from the config label
    # Set when the list prices are another registry entry's: the base model this
    # quantized entry is compared with (`ModelSpec.base_model`).
    market_model_id: str | None = None

    @property
    def input_share(self) -> Fraction | None:
        """Input tokens' share of the workload's tokens."""
        if self.tokens_in is None or self.tokens_out is None:
            return None
        total = self.tokens_in + self.tokens_out
        return None if total <= 0 else Fraction(self.tokens_in) / Fraction(total)


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
    market_model_id: str,
    competitors: Competitors,
    include_aggregators: bool,
    include_unverified: bool,
    input_share: Fraction | None,
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
            blended_at_mix=None
            if input_share is None
            else price_at_mix(e.input_per_mtok, e.output_per_mtok, input_share),
        )
        for p, e in competitors.entries_for(market_model_id)
    ]


def quantization_text(q: str | None) -> str:
    if q is None:
        return "not recorded"
    return "unquantized" if q == "none" else q


def basis_text(row: LeaderboardRow) -> str:
    """Why the quoted config: its leaderboard standing and, if untrusted, why."""
    r = row.result
    if row.rank is not None:
        text = f"leaderboard rank {row.rank}"
    else:
        reasons = sorted({w.label for w in r.warnings if w.kind.value != "no_goodput"})
        text = f"{row.status.value}, not ranked" + (f": {', '.join(reasons)}" if reasons else "")
    if row.goodput_ties:
        text += f"; tied with {', '.join(row.goodput_ties)}"
    return text


def _row(
    spec: ModelSpec,
    workload: str | None,
    mode: LoadMode | None,
    rows: Sequence[LeaderboardRow],
    competitors: Competitors,
    include_aggregators: bool,
    include_unverified: bool,
    *,
    alternative_slo: bool = False,
    market_model_id: str | None = None,
) -> CompetitivenessRow:
    """`market_model_id`: the registry entry whose list prices apply (`spec.base_model`
    for a quantized entry, `Registry.market_model_id`); by default `spec.id`."""
    market_id = market_model_id or spec.id
    head = headline_row(rows)
    best = head.result if head else None
    split = best.split_cost if best else None
    cin, cout = (split.input_per_mtok, split.output_per_mtok) if split else (None, None)
    blended = split.total_per_mtok if split else None
    shape = best.tokens_per_request() if best else None
    tokens_in, tokens_out = shape if shape else (None, None)
    share = (
        Fraction(tokens_in) / Fraction(tokens_in + tokens_out)
        if tokens_in is not None and tokens_out is not None and tokens_in + tokens_out > 0
        else None
    )
    # A side the split could not price (na_reason) has no cost of its own; the blended
    # cost needs no split, so it is compared even then.
    priced = [r for r in (cin, cout) if r is not None and not r.na_reason]
    has_sides = bool(priced) and all(r.value is not None for r in priced)
    has_blended = blended is not None and blended.value is not None and share is not None
    measured = (
        CostPerMtok(
            input=cin.value if has_sides and cin is not None else None,
            output=cout.value if has_sides and cout is not None else None,
            blended=blended.value if has_blended and blended is not None else None,
            input_share=share if has_blended else None,
        )
        if has_sides or has_blended
        else None
    )
    flags = assess(
        spec.id,
        spec.pricing,
        measured,
        competitors,
        include_aggregators=include_aggregators,
        include_unverified=include_unverified,
        market_model_id=market_id,
    )
    price = spec.pricing
    if head is not None:
        basis = basis_text(head)
        if alternative_slo:
            basis = f"alternative SLO {describe_slo(head.result.goodput.slo)}: {basis}"
    elif rows:
        basis = no_headline_reason(rows)
    else:
        basis = "not benchmarked"
    slos = {describe_slo(row.result.goodput.slo) for row in rows}
    return CompetitivenessRow(
        model_id=spec.id,
        display_name=spec.display_name,
        workload=workload,
        load_mode=mode,
        best_config=best.name if best else None,
        best_config_hash=best.config_hash if best else None,
        cost_input=cin,
        cost_output=cout,
        cost_blended=blended,
        price=price,
        margin_input=_margin(price.input_per_mtok, cin) if price else None,
        margin_output=_margin(price.output_per_mtok, cout) if price else None,
        competitors=_competitors(
            market_id, competitors, include_aggregators, include_unverified, share
        ),
        market_model_id=None if market_id == spec.id else market_id,
        flags=[FlagView(kind=f.kind.value, side=f.side, message=f.message) for f in flags],
        basis=basis,
        slo=slos.pop() if len(slos) == 1 else None,
        alternative_slo=alternative_slo,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        quantization=best.label.quantization if best else None,
    )


def _groups(
    spec: ModelSpec, results: Sequence[ConfigResult]
) -> dict[tuple[str, LoadMode], list[ConfigResult]]:
    groups: dict[tuple[str, LoadMode], list[ConfigResult]] = defaultdict(list)
    for r in results:
        if r.model_repo == spec.hf.repo:
            groups[(r.workload, r.load_mode)].append(r)
    return groups


def build_competitiveness(
    results: Sequence[ConfigResult],
    registry: Registry,
    competitors: Competitors,
    *,
    price_book: PriceBook | None = None,
    include_aggregators: bool = False,
    include_unverified: bool = False,
    title: str = "Loom competitiveness: cost at SLO vs public list prices",
    alt_results: Sequence[ConfigResult] | None = None,
    benchmarked_only: bool = False,
) -> CompetitivenessReport:
    """One row per registry model and benchmarked workload (one row if never benchmarked,
    none with `benchmarked_only`), plus, where no config has a cost at the declared SLO,
    the `alt_results` figure (an alternative-SLO analysis of the same runs) as a
    separately labelled row.

    Results are matched to registry models by Hugging Face repo.
    """
    names = {r.config_hash: r.name for r in [*results, *(alt_results or [])]}
    rows: list[CompetitivenessRow] = []
    used: list[ConfigResult] = []
    args = (competitors, include_aggregators, include_unverified)
    for spec in registry.models:
        groups = _groups(spec, results)
        alt_groups = _groups(spec, alt_results or [])
        market = registry.market_model_id(spec.id)
        if not groups and not benchmarked_only:
            rows.append(_row(spec, None, None, [], *args, market_model_id=market))
        ordered = sorted(groups.items(), key=lambda kv: (workload_order(kv[0][0]), kv[0][1]))
        for (workload, mode), members in ordered:
            ranked = rank(members, names=names)
            row = _row(spec, workload, mode, ranked, *args, market_model_id=market)
            rows.append(row)
            if (head := headline_row(ranked)) is not None:
                used.append(head.result)
            alt_members = alt_groups.get((workload, mode))
            if row.cost_blended is None and alt_members:
                alt_ranked = rank(alt_members, names=names)
                alt_row = _row(
                    spec,
                    workload,
                    mode,
                    alt_ranked,
                    *args,
                    alternative_slo=True,
                    market_model_id=market,
                )
                if alt_row.cost_blended is not None:
                    rows.append(alt_row)
                    if (alt_head := headline_row(alt_ranked)) is not None:
                        used.append(alt_head.result)
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
    """Margin with its share of price; `cost` explains an n/a margin."""
    if m is None:
        return f"n/a ({cost.na_reason})" if cost is not None and cost.na_reason else "n/a"
    share = "" if m.fraction_of_price is None else f" ({pct(m.fraction_of_price)} of price)"
    worst = "" if m.worst is None else f"; {usd(m.worst)} at cost CI high"
    return f"{usd(m.value)}{share}{worst}"


def price_text(p: Pricing | None, side: str) -> str:
    return "no price set" if p is None else usd(getattr(p, f"{side}_per_mtok"))


def break_even_text(cost: MicrosRange | None) -> str:
    """The lowest price covering our cost: at the point estimate and the CI high bound."""
    if cost is None or cost.value is None:
        return f"n/a ({cost.na_reason})" if cost is not None and cost.na_reason else "n/a"
    high = "unbounded" if cost.hi is None else usd(cost.hi)
    return f"{usd(cost.value)} (CI high {high})"


def row_title(r: CompetitivenessRow) -> str:
    if r.workload is None:
        return "not benchmarked"
    mode = r.load_mode.value.replace("_", " ") if r.load_mode else ""
    title = f"{r.workload} ({mode})"
    if r.alternative_slo:
        title += " at the ALTERNATIVE SLO, not the declared one"
    return title


def config_text(r: CompetitivenessRow) -> str:
    if r.best_config is None:
        return r.basis or "none"
    return f"{r.best_config} ({r.basis})" if r.basis else r.best_config


def shape_text(r: CompetitivenessRow) -> str:
    if r.tokens_in is None or r.tokens_out is None:
        return "n/a"
    return f"{r.tokens_in:,.0f} / {r.tokens_out:,.0f}"


def mix_header(r: CompetitivenessRow) -> str:
    alt = ", alt SLO" if r.alternative_slo else ""
    return f"At {r.workload} mix ({shape_text(r)}{alt})"


def mix_cell(c: CompetitorPrice, r: CompetitivenessRow) -> str:
    """A public price at this row's mix and our blended cost as a multiple of it."""
    if c.blended_at_mix is None:
        return "n/a"
    text = usd(c.blended_at_mix)
    ours = r.cost_blended.value if r.cost_blended else None
    if ours is not None and c.blended_at_mix > 0:
        text += f" (our cost {ours / c.blended_at_mix:.2f}×)"
    return text


def precision_text(c: CompetitorPrice, ours: str | None) -> str:
    """The competitor's disclosed precision, marked against ours when ours is known."""
    if c.quantization is None:
        return "not disclosed"
    if ours is None:
        return c.quantization
    return f"{c.quantization} ({like_for_like(c.quantization, ours)})"


def market_note(r: CompetitivenessRow) -> str | None:
    """Why a row's list prices are another registry entry's."""
    if r.market_model_id is None:
        return None
    return (
        f"Public list prices are those of {r.market_model_id}, the model this entry "
        "serves at another precision: providers price the model, and the precision column "
        "says which listings match ours."
    )


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
    "Our config at SLO (basis)",
    "Tokens per request in / out",
    "Our cost $/1M in",
    "Our cost $/1M out",
    "Our cost $/1M blended",
    "Our price $/1M in",
    "Our price $/1M out",
    "Break-even $/1M in",
    "Break-even $/1M out",
    "Margin in",
    "Margin out",
)


def competitor_headers(rows: Sequence[CompetitivenessRow]) -> list[str]:
    return [
        "Provider",
        "Provider model",
        "$/1M in",
        "$/1M out",
        "Precision",
        *(mix_header(r) for r in _priced(rows)),
        "Availability",
        "In flag comparison",
        "Source",
        "Last checked",
    ]


def _priced(rows: Sequence[CompetitivenessRow]) -> list[CompetitivenessRow]:
    """Rows with a blended cost: one mix column each in the public-price table."""
    return [r for r in rows if r.cost_blended is not None and r.cost_blended.value is not None]


def _ours_md(r: CompetitivenessRow) -> list[Any]:
    return [
        row_title(r),
        config_text(r),
        shape_text(r),
        usd_ci(r.cost_input),
        usd_ci(r.cost_output),
        usd_ci(r.cost_blended),
        price_text(r.price, "input"),
        price_text(r.price, "output"),
        break_even_text(r.cost_input),
        break_even_text(r.cost_output),
        margin_text(r.margin_input, r.cost_input),
        margin_text(r.margin_output, r.cost_output),
    ]


def _competitor_md(
    i: int, c: CompetitorPrice, rows: Sequence[CompetitivenessRow], ours: str | None
) -> list[Any]:
    return [
        c.provider + (" (aggregator)" if c.aggregator else ""),
        c.provider_model or "",
        usd(c.input_per_mtok),
        usd(c.output_per_mtok),
        precision_text(c, ours),
        *(mix_cell(r.competitors[i], r) for r in _priced(rows)),
        c.availability,
        "yes" if c.in_comparison else "no",
        c.source,
        c.last_checked.isoformat(),
    ]


def flag_groups(rows: Sequence[CompetitivenessRow]) -> list[tuple[list[str], FlagView]]:
    """Each distinct flag once, with the rows (workloads) it was raised on."""
    groups: dict[tuple[str, str | None, str], tuple[list[str], FlagView]] = {}
    for r in rows:
        for f in r.flags:
            titles, _ = groups.setdefault((f.kind, f.side, f.message), ([], f))
            titles.append(row_title(r))
    return list(groups.values())


def by_model(report: CompetitivenessReport) -> dict[str, list[CompetitivenessRow]]:
    out: dict[str, list[CompetitivenessRow]] = defaultdict(list)
    for r in report.rows:
        out[r.model_id].append(r)
    return out


def our_quantization(rows: Sequence[CompetitivenessRow]) -> str | None:
    return next((r.quantization for r in rows if r.quantization is not None), None)


def lede(report: CompetitivenessReport) -> str:
    return (
        f"Competitor prices last checked {report.competitors_last_checked.isoformat()}. "
        f"{comparison_scope(report)} {OUR_COST_NOTE}"
    )


def section_markdown(report: CompetitivenessReport, level: int = 2) -> str:
    """The per-model tables and flags, headings starting at `level` (embedded in the
    leaderboard at 2, as the body of the standalone report at 2 as well)."""
    h = "#" * level
    parts = [f"> {report.note}", "", lede(report)]
    for model_id, rows in by_model(report).items():
        first = rows[0]
        ours = our_quantization(rows)
        parts += [
            "",
            f"{h} {first.display_name} (`{model_id}`), ours {quantization_text(ours)}",
            "",
        ]
        parts.append(md_table(OURS_HEADERS, map(_ours_md, rows)))
        parts += ["", "**Public list prices** (per 1M tokens)", ""]
        if note := market_note(first):
            parts += [note, ""]
        if first.competitors:
            table = [_competitor_md(i, c, rows, ours) for i, c in enumerate(first.competitors)]
            parts.append(md_table(competitor_headers(rows), table))
        else:
            parts.append("No public list price recorded for this model.")
        parts += ["", "**Flags**", ""]
        groups = flag_groups(rows)
        if groups:
            parts += [f"- [{'; '.join(titles)}] `{f.kind}`: {f.message}" for titles, f in groups]
        else:
            parts.append("- none")
    if report.not_offered:
        parts += ["", f"{h} Providers with no public per-token price", ""]
        parts += [
            f"- {n.name}: {n.reason} (checked {n.last_checked.isoformat()})"
            for n in report.not_offered
        ]
    return "\n".join(parts)


def render_markdown(report: CompetitivenessReport) -> str:
    parts = [f"# {report.title}", "", section_markdown(report)]
    parts += ["", methodology_markdown(report.methodology)]
    return "\n".join(parts)


def _html_context(report: CompetitivenessReport) -> dict[str, Any]:
    return {
        "comp": report,
        "comp_by_model": by_model(report),
        "comp_lede": lede(report),
        "row_title": row_title,
        "config_text": config_text,
        "shape_text": shape_text,
        "margin_text": margin_text,
        "price_text": price_text,
        "break_even_text": break_even_text,
        "mix_header": mix_header,
        "mix_cell": mix_cell,
        "precision_text": precision_text,
        "market_note": market_note,
        "quantization_text": quantization_text,
        "our_quantization": our_quantization,
        "priced_rows": _priced,
        "flag_groups": flag_groups,
    }


def section_html(report: CompetitivenessReport, heading: str) -> str:
    """The section as an HTML fragment, under an <h2> `heading` (for the leaderboard)."""
    return (
        html_env()
        .get_template("_competitiveness_section.html.j2")
        .render(heading=heading, **_html_context(report))
    )


def render_html(report: CompetitivenessReport) -> str:
    return (
        html_env()
        .get_template("competitiveness.html.j2")
        .render(report=report, **_html_context(report))
    )


CSV_COLUMNS = [
    "model_id",
    "workload",
    "load_mode",
    "slo",
    "alternative_slo",
    "best_config",
    "best_config_hash",
    "basis",
    "tokens_in_per_request",
    "tokens_out_per_request",
    *micros_column_names("cost_input_per_mtok"),
    *micros_column_names("cost_output_per_mtok"),
    *micros_column_names("cost_blended_per_mtok"),
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
    "competitor_listing_model_id",
    "competitor_aggregator",
    "competitor_availability",
    "competitor_in_comparison",
    "competitor_input_per_mtok_micros",
    "competitor_input_per_mtok_usd",
    "competitor_output_per_mtok_micros",
    "competitor_output_per_mtok_usd",
    "competitor_blended_at_mix_micros",
    "competitor_blended_at_mix_usd",
    "our_blended_cost_vs_competitor",
    "competitor_quantization",
    "our_quantization",
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
            "slo": r.slo,
            "alternative_slo": r.alternative_slo,
            "best_config": r.best_config,
            "best_config_hash": r.best_config_hash,
            "basis": r.basis,
            "tokens_in_per_request": None if r.tokens_in is None else round(r.tokens_in, 1),
            "tokens_out_per_request": None if r.tokens_out is None else round(r.tokens_out, 1),
            **micros_columns("cost_input_per_mtok", r.cost_input),
            **micros_columns("cost_output_per_mtok", r.cost_output),
            **micros_columns("cost_blended_per_mtok", r.cost_blended),
            "price_input_per_mtok_micros": r.price.input_per_mtok if r.price else None,
            "price_input_per_mtok_usd": usd(r.price.input_per_mtok, 6) if r.price else None,
            "price_output_per_mtok_micros": r.price.output_per_mtok if r.price else None,
            "price_output_per_mtok_usd": usd(r.price.output_per_mtok, 6) if r.price else None,
            "margin_input_micros": r.margin_input.value if r.margin_input else None,
            "margin_input_worst_micros": r.margin_input.worst if r.margin_input else None,
            "margin_output_micros": r.margin_output.value if r.margin_output else None,
            "margin_output_worst_micros": r.margin_output.worst if r.margin_output else None,
            "flags": "; ".join(f.kind + (f"({f.side})" if f.side else "") for f in r.flags),
            "our_quantization": r.quantization,
            "price_basis": bases.get(r.best_config_hash or ""),
            "public_list_prices_only": True,
        }
        ours = r.cost_blended.value if r.cost_blended else None
        entries: list[CompetitorPrice | None] = [*r.competitors] or [None]
        for c in entries:
            row = dict(base)
            if c is not None:
                mix = c.blended_at_mix
                row |= {
                    "competitor": c.provider,
                    "competitor_model": c.provider_model,
                    # The registry entry the listing is recorded under: the base model
                    # for a quantized entry (`ModelSpec.base_model`).
                    "competitor_listing_model_id": r.market_model_id or r.model_id,
                    "competitor_aggregator": c.aggregator,
                    "competitor_availability": c.availability,
                    "competitor_in_comparison": c.in_comparison,
                    "competitor_input_per_mtok_micros": c.input_per_mtok,
                    "competitor_input_per_mtok_usd": usd(c.input_per_mtok, 6),
                    "competitor_output_per_mtok_micros": c.output_per_mtok,
                    "competitor_output_per_mtok_usd": usd(c.output_per_mtok, 6),
                    "competitor_blended_at_mix_micros": mix,
                    "competitor_blended_at_mix_usd": None if mix is None else usd(mix, 6),
                    "our_blended_cost_vs_competitor": (
                        round(ours / mix, 4) if ours is not None and mix else None
                    ),
                    "competitor_quantization": c.quantization,
                    "competitor_source": c.source,
                    "competitor_last_checked": c.last_checked,
                }
            out.append(row)
    return to_csv(out, CSV_COLUMNS)
