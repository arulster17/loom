"""Reports from the results store: leaderboard, competitiveness view and compare.

Each `render_*` returns `{format: text}` with the format as the file extension
("md", "html", "csv", "json"; the leaderboard adds "equal_load.csv", its latency at
equal load in long form); `write_reports` writes them as `<name>.<format>`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from loom_bench.prices import Competitors, PriceBook
from loom_bench.registry import Registry
from loom_bench.report import compare as _compare
from loom_bench.report import competitiveness as _competitiveness
from loom_bench.report import leaderboard as _leaderboard
from loom_bench.report import summary as _summary
from loom_bench.report.analyze import (
    ColdStartStat,
    ConfigResult,
    analyze_runs,
    cold_starts_by_config,
    default_price_resolver,
    quality_for,
    with_quality,
)
from loom_bench.store.models import BenchRun

Rendered = dict[str, str]


def render_leaderboard(
    results: Sequence[ConfigResult],
    *,
    cold_starts: Mapping[str, ColdStartStat] | None = None,
    price_book: PriceBook | None = None,
    title: str = "Loom leaderboard: cost at SLO",
    registry: Registry | None = None,
    competitors: Competitors | None = None,
    alt_results: Sequence[ConfigResult] | None = None,
    view_note: str | None = None,
) -> Rendered:
    """The leaderboard with its summary on top. With `registry` and `competitors` it also
    has a competitiveness section (and "competitiveness.csv"). `alt_results`, the same
    runs judged against an alternative SLO, are quoted, labelled, only where no config
    has a cost at the declared SLO. `view_note` opens the summary (an alternative view)."""
    report = _leaderboard.build_leaderboard(
        results, cold_starts=cold_starts, price_book=price_book, title=title
    )
    alt = _leaderboard.build_leaderboard(alt_results) if alt_results else None
    summary = _summary.build_summary(
        report, alt=alt, competitors=competitors, registry=registry, view_note=view_note
    )
    top_md, top_html = [_summary.summary_markdown(summary)], [_summary.summary_html(summary)]
    bottom_md: list[str] = []
    bottom_html: list[str] = []
    out: Rendered = {}
    if registry is not None and competitors is not None:
        comp = _competitiveness.build_competitiveness(
            results,
            registry,
            competitors,
            price_book=price_book,
            alt_results=alt_results,
            benchmarked_only=True,
        )
        heading = "Competitiveness: our cost at SLO vs public list prices"
        bottom_md.append(f"## {heading}\n\n{_competitiveness.section_markdown(comp, level=3)}")
        bottom_html.append(_competitiveness.section_html(comp, heading))
        out["competitiveness.csv"] = _competitiveness.render_csv(comp)
    return {
        "md": _leaderboard.render_markdown(report, top=top_md, bottom=bottom_md),
        "html": _leaderboard.render_html(report, top=top_html, bottom=bottom_html),
        "csv": _leaderboard.render_csv(report),
        "equal_load.csv": _leaderboard.render_equal_load_csv(report),
        **out,
    }


def render_competitiveness(
    results: Sequence[ConfigResult],
    registry: Registry,
    competitors: Competitors,
    *,
    price_book: PriceBook | None = None,
    include_aggregators: bool = False,
    include_unverified: bool = False,
) -> Rendered:
    report = _competitiveness.build_competitiveness(
        results,
        registry,
        competitors,
        price_book=price_book,
        include_aggregators=include_aggregators,
        include_unverified=include_unverified,
    )
    return {
        "md": _competitiveness.render_markdown(report),
        "html": _competitiveness.render_html(report),
        "csv": _competitiveness.render_csv(report),
    }


def render_compare(
    a: Sequence[ConfigResult] | Iterable[BenchRun],
    b: Sequence[ConfigResult] | Iterable[BenchRun],
    **options: Any,
) -> Rendered:
    """`options` are passed to `compare.compare` (match_by, rel_tol, alpha, labels, ...)."""
    comparison = _compare.compare(a, b, **options)
    return {"md": _compare.render_markdown(comparison), "json": _compare.render_json(comparison)}


def write_reports(out_dir: str | Path, **reports: Mapping[str, str]) -> list[Path]:
    """Write each `name={format: text}` to `out_dir/<name>.<format>`; returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, formats in reports.items():
        for fmt, text in formats.items():
            path = out / f"{name}.{fmt}"
            path.write_text(text, encoding="utf-8")
            paths.append(path)
    return paths


__all__ = [
    "ConfigResult",
    "analyze_runs",
    "cold_starts_by_config",
    "default_price_resolver",
    "quality_for",
    "render_compare",
    "render_competitiveness",
    "render_leaderboard",
    "with_quality",
    "write_reports",
]
