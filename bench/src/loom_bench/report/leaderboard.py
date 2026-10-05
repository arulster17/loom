"""Per-model leaderboard ranked by $/1M output tokens at SLO, at the on-demand price.

Under the all_input cost allocation output tokens have no price of their own, so
those results rank by $/1M input tokens instead (`ranking_cost`). The on-demand
price (`cost.replica_prices`) is the public list price, so the ranking is
reproducible from the price book; spot, committed-1y and as-run costs of the same
side are shown next to it where a board has them.

One board per (model, workload, load mode): costs measured on different workloads
are not comparable. Rows are ranked cheapest first. Configs that failed the
quality gate, are untrusted, or have no cost at SLO are still listed, after the
ranked rows and without a rank, with the reason in their status. Recommendations
are generated from the numbers alone, so the same data always gives the same text.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from fractions import Fraction
from typing import Any

from pydantic import BaseModel

from loom_bench.cost import PRICE_COLUMN_LABELS, MicrosRange, PriceColumn
from loom_bench.prices import PriceBook
from loom_bench.provenance import ContentKind
from loom_bench.records import LoadMode
from loom_bench.report.analyze import UNTRUSTING, ColdStartStat, ConfigResult
from loom_bench.report.format import (
    UNBRACKETED_NOTE,
    any_unbracketed,
    describe_slo,
    est,
    estimate_column_names,
    estimate_columns,
    goodput_load,
    html_env,
    load_value,
    md_table,
    micros_column_names,
    micros_columns,
    pct,
    sig,
    to_csv,
    usd,
    usd_ci,
)
from loom_bench.report.methodology import (
    DatasetRef,
    Methodology,
    methodology,
    methodology_markdown,
)


class RowStatus(StrEnum):
    RANKED = "ranked"
    GATE_FAILED = "quality gate failed"
    UNTRUSTED = "untrusted"
    NO_COST = "no cost at SLO"


_STATUS_ORDER = list(RowStatus)


class LeaderboardRow(BaseModel):
    rank: int | None  # None: not ranked, listed after the ranked rows
    status: RowStatus
    result: ConfigResult
    cold_start: ColdStartStat | None
    recommendation: str


# Price columns shown next to the on-demand ranking cost, when any row has one.
EXTRA_PRICE_COLUMNS = (PriceColumn.SPOT, PriceColumn.COMMITTED_1Y, PriceColumn.AS_RUN)


class Leaderboard(BaseModel):
    model: str
    workload: str
    load_mode: LoadMode
    content: ContentKind | None
    rows: list[LeaderboardRow]
    price_columns: list[PriceColumn]  # extra cost columns with a value in some row

    @property
    def side(self) -> str:
        """The token side every row is ranked by: "in" under all_input, else "out"."""
        return ranking_side(row.result for row in self.rows)


class LeaderboardReport(BaseModel):
    title: str
    boards: list[Leaderboard]
    methodology: Methodology


def ranking_cost(
    r: ConfigResult, column: PriceColumn = PriceColumn.ON_DEMAND
) -> MicrosRange | None:
    """The side a result is ranked by ($/1M output, or $/1M input under all_input) at
    one price column; the on-demand column is the ranking itself."""
    cost = r.cost_at(column)
    if cost is None:
        return None
    return cost.input_per_mtok if r.allocation == "all_input" else cost.output_per_mtok


def ranking_side(results: Iterable[ConfigResult]) -> str:
    return "in" if any(r.allocation == "all_input" for r in results) else "out"


def price_columns(results: Sequence[ConfigResult]) -> list[PriceColumn]:
    """The extra price columns at least one of `results` has a cost in."""
    return [
        c
        for c in EXTRA_PRICE_COLUMNS
        if any((rng := ranking_cost(r, c)) is not None and rng.value is not None for r in results)
    ]


def price_header(column: PriceColumn, side: str) -> str:
    return f"$/1M {side} at SLO, {PRICE_COLUMN_LABELS[column]}"


def _out_cost(r: ConfigResult) -> int | None:
    rng = ranking_cost(r)
    return None if rng is None else rng.value


def status_of(r: ConfigResult) -> RowStatus:
    if _out_cost(r) is None:
        return RowStatus.NO_COST
    if r.quality is not None and r.quality.gate == "fail":
        return RowStatus.GATE_FAILED
    if not r.trusted:
        return RowStatus.UNTRUSTED
    return RowStatus.RANKED


def _rel(a: int, b: int) -> Fraction:
    """a relative to b: (a - b) / b."""
    return Fraction(a - b, b)


def _overlap(a: ConfigResult, b: ConfigResult) -> bool:
    """Whether the ranking-cost CIs overlap; a missing high bound is unbounded."""
    ra, rb = ranking_cost(a), ranking_cost(b)
    if ra is None or rb is None:
        return False
    inf = 2**63
    a_lo, a_hi = ra.lo or 0, inf if ra.hi is None else ra.hi
    b_lo, b_hi = rb.lo or 0, inf if rb.hi is None else rb.hi
    return a_lo <= b_hi and b_lo <= a_hi


def _quality_clause(r: ConfigResult, names: Mapping[str, str]) -> str:
    q = r.quality
    if q is None or q.gate is None:
        return "quality not evaluated"
    if q.gate == "baseline":
        return "quality baseline"
    base = names.get(q.baseline_config_hash or "", (q.baseline_config_hash or "?")[:12])
    if q.gate == "pass":
        return f"passed the quality gate vs {base}"
    worst = q.worst()
    detail = f" (worst: {worst.task} {worst.delta:+.3f})" if worst and worst.delta else ""
    if q.gate == "fail":
        return f"failed the quality gate vs {base}{detail}"
    return f"quality gate inconclusive vs {base}{detail}"


def recommend(
    r: ConfigResult, status: RowStatus, ranked: Sequence[ConfigResult], names: Mapping[str, str]
) -> str:
    cost = _out_cost(r)
    leader = ranked[0] if ranked else None
    lead_cost = _out_cost(leader) if leader else None
    quality = _quality_clause(r, names)

    if status is RowStatus.NO_COST:
        if r.goodput.max_load is None:
            return "No cost at SLO: no tested load met the SLO; test lower loads"
        if r.hourly_micros is None:
            return f"No cost at SLO: {r.prices.missing or 'no on-demand price for this replica'}"
        return "No cost at SLO: no token throughput measured at the goodput load"
    if status in (RowStatus.GATE_FAILED, RowStatus.UNTRUSTED):
        if status is RowStatus.GATE_FAILED:
            text = f"Not ranked: {quality}"
        else:
            reasons = sorted({w.label for w in r.warnings if w.kind in UNTRUSTING})
            text = f"Not ranked: untrusted ({', '.join(reasons)}); rerun before relying on it"
        if leader is not None and lead_cost is not None and cost is not None and cost < lead_cost:
            text += f"; would be {pct(-_rel(cost, lead_cost))} cheaper than {leader.name}"
        return text

    assert cost is not None and leader is not None and lead_cost is not None
    if r is leader:
        if len(ranked) == 1:
            return f"Only ranked config at SLO; {quality}"
        runner = ranked[1]
        runner_cost = _out_cost(runner)
        assert runner_cost is not None
        text = f"Cheapest at SLO; {pct(-_rel(cost, runner_cost))} cheaper than {runner.name}"
        if _overlap(r, runner):
            text += " (cost CIs overlap)"
        return f"{text}; {quality}"

    text = f"{pct(_rel(cost, lead_cost))} more expensive than {leader.name} at SLO"
    if _overlap(r, leader):
        text += " (cost CIs overlap)"
    if r.ttft_p95_ms and leader.ttft_p95_ms and r.ttft_p95_ms.mean < leader.ttft_p95_ms.mean:
        gain = 1 - r.ttft_p95_ms.mean / leader.ttft_p95_ms.mean
        text += f"; {pct(gain)} lower p95 TTFT at goodput"
    return f"{text}; {quality}"


def _sort_key(r: ConfigResult) -> tuple[int, int, str]:
    cost = _out_cost(r)
    return (_STATUS_ORDER.index(status_of(r)), cost if cost is not None else 2**63, r.name)


def rank(
    results: Sequence[ConfigResult],
    cold_starts: Mapping[str, ColdStartStat] | None = None,
    names: Mapping[str, str] | None = None,
) -> list[LeaderboardRow]:
    """Rows for one board, ranked rows first (cheapest first), then unranked by status."""
    names = names or {r.config_hash: r.name for r in results}
    ordered = sorted(results, key=_sort_key)
    ranked = [r for r in ordered if status_of(r) is RowStatus.RANKED]
    rows = []
    for r in ordered:
        status = status_of(r)
        rows.append(
            LeaderboardRow(
                rank=ranked.index(r) + 1 if status is RowStatus.RANKED else None,
                status=status,
                result=r,
                cold_start=(cold_starts or {}).get(r.config_hash),
                recommendation=recommend(r, status, ranked, names),
            )
        )
    return rows


def build_leaderboard(
    results: Sequence[ConfigResult],
    *,
    cold_starts: Mapping[str, ColdStartStat] | None = None,
    price_book: PriceBook | None = None,
    title: str = "Loom leaderboard: cost at SLO",
) -> LeaderboardReport:
    names = {r.config_hash: r.name for r in results}
    groups: dict[tuple[str, str, LoadMode], list[ConfigResult]] = defaultdict(list)
    for r in results:
        groups[(r.model_repo or "unknown model", r.workload, r.load_mode)].append(r)
    boards = []
    for (model, workload, mode), members in sorted(groups.items()):
        contents = {r.content for r in members}
        boards.append(
            Leaderboard(
                model=model,
                workload=workload,
                load_mode=mode,
                content=contents.pop() if len(contents) == 1 else None,
                rows=rank(members, cold_starts, names),
                price_columns=price_columns(members),
            )
        )
    ordered = [row.result for b in boards for row in b.rows]
    return LeaderboardReport(
        title=title, boards=boards, methodology=methodology(ordered, price_book)
    )


def quality_text(r: ConfigResult) -> str:
    q = r.quality
    if q is None or q.gate is None:
        return "not evaluated"
    worst = q.worst()
    if worst is None or worst.delta is None:
        return q.gate
    return f"{worst.delta:+.3f} ({worst.task}) · {q.gate}"


def cold_text(c: ColdStartStat | None) -> str:
    return "n/a" if c is None else f"{sig(c.median_s)} s (median of {c.n})"


def board_title(b: Leaderboard) -> str:
    content = f", {b.content.value} content" if b.content else ""
    return f"{b.model}: {b.workload} ({b.load_mode.value.replace('_', ' ')}{content})"


MD_HEAD = ("#", "Config", "$/1M out at SLO, on-demand", "$/1M in at SLO, on-demand")
MD_TAIL = (
    "Goodput out tok/s per replica",
    "per GPU",
    "Load at goodput",
    "p95 TTFT at goodput",
    "p95 TPOT at goodput",
    "Raw peak out tok/s (no SLO)",
    "Quality Δ / gate",
    "Cold start",
    "Status",
    "Recommendation",
)


def md_headers(b: Leaderboard) -> list[str]:
    return [*MD_HEAD, *(price_header(c, b.side) for c in b.price_columns), *MD_TAIL]


def _md_row(row: LeaderboardRow, columns: Sequence[PriceColumn]) -> list[Any]:
    r = row.result
    cost = r.cost
    return [
        row.rank if row.rank is not None else "–",
        f"**{r.name}**<br>{r.label.text}",
        usd_ci(cost.output_per_mtok if cost else None),
        usd_ci(cost.input_per_mtok if cost else None),
        *(usd_ci(ranking_cost(r, c)) for c in columns),
        est(r.goodput.output_tok_s, 1),
        est(r.goodput_output_tok_s_per_gpu, 1),
        goodput_load(r.goodput),
        est(r.ttft_p95_ms, 0, "ms"),
        est(r.tpot_p95_ms, 1, "ms"),
        f"{est(r.peak_output_tok_s, 1)} at {load_value(r.peak_load, r.load_mode)}",
        quality_text(r),
        cold_text(row.cold_start),
        row.status.value,
        row.recommendation,
    ]


def render_markdown(report: LeaderboardReport) -> str:
    parts = [f"# {report.title}", ""]
    parts.append(
        "Ranked by $/1M output tokens at SLO (input tokens under the all_input cost "
        "allocation) at the on-demand list price, cheapest first; spot, committed-1y and "
        "as-run costs of the same tokens are shown where available. Every price includes "
        "the replica's block storage. Values are point estimates (geometric means for "
        "latency and throughput) with 95% confidence intervals in brackets; the methodology "
        "below says how each is computed. Goodput is the highest tested load that met the "
        "SLO; raw peak throughput ignores the SLO and is not goodput. Unranked rows (quality "
        "gate failed, untrusted, or no cost) are listed last."
    )
    if any_unbracketed(row.result.goodput for b in report.boards for row in b.rows):
        parts += ["", UNBRACKETED_NOTE]
    for b in report.boards:
        rows = [_md_row(row, b.price_columns) for row in b.rows]
        parts += ["", f"## {board_title(b)}", "", md_table(md_headers(b), rows)]
        warned = [row.result for row in b.rows if row.result.warnings]
        if warned:
            parts += ["", "**Warnings**", ""]
            for r in warned:
                parts += [f"- {r.name}: {w.message}" for w in r.warnings]
    parts += ["", methodology_markdown(report.methodology)]
    return "\n".join(parts)


def render_html(report: LeaderboardReport) -> str:
    return (
        html_env()
        .get_template("leaderboard.html.j2")
        .render(
            report=report,
            board_title=board_title,
            price_header=price_header,
            ranking_cost=ranking_cost,
            quality_text=quality_text,
            cold_text=cold_text,
            unbracketed=any_unbracketed(
                row.result.goodput for b in report.boards for row in b.rows
            ),
        )
    )


CSV_COLUMNS = [
    "model",
    "workload",
    "load_mode",
    "content",
    "rank",
    "status",
    "config",
    "config_hash",
    "label",
    *(
        name
        for column in PriceColumn
        for name in (
            *micros_column_names(f"{column.value}_output_per_mtok"),
            *micros_column_names(f"{column.value}_input_per_mtok"),
            *micros_column_names(f"{column.value}_blended_per_mtok"),
            f"{column.value}_hourly_micros",
            f"{column.value}_hourly_usd",
        )
    ),
    "storage_gb",
    "allocation",
    "goodput_load",
    "first_failing_load",
    *estimate_column_names("goodput_output_tok_s"),
    *estimate_column_names("goodput_output_tok_s_per_gpu"),
    *estimate_column_names("goodput_request_rate"),
    *estimate_column_names("ttft_p95_ms"),
    *estimate_column_names("tpot_p95_ms"),
    *estimate_column_names("peak_output_tok_s"),
    "peak_load",
    "quality_gate",
    "quality_baseline_config_hash",
    "quality_worst_task",
    "quality_worst_delta",
    "cold_start_median_s",
    "cold_start_n",
    "trusted",
    "warnings",
    "recommendation",
    "slo",
    "repetitions",
    "ci_method",
    "engine",
    "image_digest",
    "model_revision",
    "gpu",
    "location",
    "git_shas",
    "dataset",
    "dataset_source",
    "price_basis",
    "as_run_price_basis",
    "price_sources",
    "price_last_checked",
    "provenance_digests",
    "reproduce_command",
]


def _price_csv(r: ConfigResult, column: PriceColumn) -> dict[str, Any]:
    cost, hourly, name = r.cost_at(column), r.prices.get(column), column.value
    return {
        **micros_columns(f"{name}_output_per_mtok", cost.output_per_mtok if cost else None),
        **micros_columns(f"{name}_input_per_mtok", cost.input_per_mtok if cost else None),
        **micros_columns(f"{name}_blended_per_mtok", cost.total_per_mtok if cost else None),
        f"{name}_hourly_micros": hourly,
        f"{name}_hourly_usd": None if hourly is None else usd(hourly, 6),
    }


def _csv_row(b: Leaderboard, row: LeaderboardRow, m: Methodology) -> dict[str, Any]:
    r = row.result
    prov = next(c for c in m.configs if c.key == r.key)
    q = r.quality
    worst = q.worst() if q else None
    dataset = DatasetRef.model_validate(r.provenance.get("dataset") or {})
    return {
        "model": b.model,
        "workload": b.workload,
        "load_mode": b.load_mode.value,
        "content": r.content.value if r.content else None,
        "rank": row.rank,
        "status": row.status.value,
        "config": r.name,
        "config_hash": r.config_hash,
        "label": r.label.text,
        **{k: v for column in PriceColumn for k, v in _price_csv(r, column).items()},
        "storage_gb": r.prices.storage_gb,
        "allocation": r.allocation,
        "goodput_load": r.goodput.max_load,
        "first_failing_load": r.goodput.first_failing_load,
        **estimate_columns("goodput_output_tok_s", r.goodput.output_tok_s),
        **estimate_columns("goodput_output_tok_s_per_gpu", r.goodput_output_tok_s_per_gpu),
        **estimate_columns("goodput_request_rate", r.goodput.request_rate),
        **estimate_columns("ttft_p95_ms", r.ttft_p95_ms),
        **estimate_columns("tpot_p95_ms", r.tpot_p95_ms),
        **estimate_columns("peak_output_tok_s", r.peak_output_tok_s),
        "peak_load": r.peak_load,
        "quality_gate": q.gate if q else None,
        "quality_baseline_config_hash": q.baseline_config_hash if q else None,
        "quality_worst_task": worst.task if worst else None,
        "quality_worst_delta": worst.delta if worst else None,
        "cold_start_median_s": row.cold_start.median_s if row.cold_start else None,
        "cold_start_n": row.cold_start.n if row.cold_start else None,
        "trusted": r.trusted,
        "warnings": "; ".join(w.message for w in r.warnings),
        "recommendation": row.recommendation,
        "slo": describe_slo(r.goodput.slo),
        "repetitions": m.repetitions,
        "ci_method": m.ci_method,
        "engine": prov.engine,
        "image_digest": prov.image_digest,
        "model_revision": r.model_revision,
        "gpu": prov.gpu,
        "location": prov.location,
        "git_shas": " ".join(prov.git_shas),
        "dataset": dataset.text,
        "dataset_source": dataset.source,
        "price_basis": prov.price.basis,
        "as_run_price_basis": prov.price.as_run,
        "price_sources": " ".join(prov.price.urls),
        "price_last_checked": prov.price.last_checked,
        "provenance_digests": " ".join(r.provenance_digests),
        "reproduce_command": prov.reproduce,
    }


def render_csv(report: LeaderboardReport) -> str:
    rows = [_csv_row(b, row, report.methodology) for b in report.boards for row in b.rows]
    return to_csv(rows, CSV_COLUMNS)
