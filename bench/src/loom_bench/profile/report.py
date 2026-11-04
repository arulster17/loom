"""`bench roofline`: the floors in theory, and stored runs set against them.

Writes `roofline.md` (tables and the plain-words reading), `roofline.json` (every
number behind them, with run ids) and `roofline.svg` (where a decode step's time
goes at each config's goodput: the floor's parts, and what is left over).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from loom_bench.profile.measured import (
    PointSummary,
    StepFit,
    default_roots,
    fit_steps,
    load_runs,
    observe_runs,
    summarize_points,
)
from loom_bench.profile.roofline import ServingSetup, decode_step, max_batch_at, prefill

THEORY_BATCHES = (1, 8, 32, 64, 128)
THEORY_CONTEXT = 1536  # fixed-1k-1k's mean context mid-generation
TPOT_SLO_MS = 50.0


def theory_rows(
    setups: Sequence[ServingSetup],
    batches: Sequence[int] = THEORY_BATCHES,
    context: int = THEORY_CONTEXT,
) -> list[dict[str, Any]]:
    rows = []
    for s in setups:
        row: dict[str, Any] = {**s.describe(), "context": context}
        row["step_floor_ms"] = {b: decode_step(s, b, b * context).total_ms for b in batches}
        row["max_batch_at_slo"] = max_batch_at(s, TPOT_SLO_MS, context)
        row["prefill_1k_ms"] = prefill(s, 1024).total_ms
        rows.append(row)
    return rows


def goodput_loads(
    roots: Sequence[Path], experiments: Sequence[str]
) -> dict[tuple[str, str, str], float]:
    """(experiment id, cell, workload) -> the load at goodput, from each experiment's
    results/<id>/goodput.json when present."""
    out: dict[tuple[str, str, str], float] = {}
    for exp in experiments:
        for root in roots:
            for path in (root / exp / "goodput.json", root / "results" / exp / "goodput.json"):
                if path.exists():
                    for row in json.loads(path.read_text()):
                        if row.get("max_load") is not None:
                            out[(exp, row["cell"], row["workload"])] = float(row["max_load"])
                    break
    return out


@dataclass
class RatioExplanation:
    """Why config B serves `measured_ratio` times config A's goodput on one workload.

    `decode_ratio`: how many more sequences B decodes than A within the step time A
    runs at its goodput (both fits), i.e. the gain if decode were the only work. The
    rest is what prefill and the latency tail take back."""

    workload: str
    a: str
    b: str
    load_a: float
    load_b: float
    step_a_ms: float
    batch_a_fit: float
    batch_b_fit: float
    fixed_a_ms: float
    fixed_b_ms: float
    per_seq_a_ms: float
    per_seq_b_ms: float
    tail_a: float  # TPOT p95 / ITL p50 at goodput
    tail_b: float
    prefill_floor_a: float
    prefill_floor_b: float
    prefill_implied_a: float  # 1 - ITL p50 / ITL mean at goodput
    prefill_implied_b: float
    # The next load each searched past its goodput: what failed there, how full the KV
    # cache was and how many sequences were preempted; and the cache's estimated size.
    fail_a: dict[str, Any] | None = None
    fail_b: dict[str, Any] | None = None
    kv_capacity_a: float | None = None
    kv_capacity_b: float | None = None

    @property
    def measured_ratio(self) -> float:
        return self.load_b / self.load_a

    @property
    def decode_ratio(self) -> float:
        return self.batch_b_fit / self.batch_a_fit

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "measured_ratio": self.measured_ratio,
            "decode_ratio": self.decode_ratio,
        }


def _fail_facts(p: PointSummary | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {
        "load": p.load_value,
        "failing": p.failing() or ["its CI upper bound only (the mean passes)"],
        "kv_usage": p.kv_usage,
        "preemptions": p.preemptions,
        "ttft_p95_ms": p.ttft_p95_ms,
        "tpot_p95_ms": p.tpot_p95_ms,
    }


def _capacity(points: Sequence[PointSummary], ref: PointSummary) -> float | None:
    caps = [
        p.kv_capacity_tokens
        for p in points
        if p.experiment == ref.experiment and p.cell == ref.cell and p.kv_capacity_tokens
    ]
    return max(caps) if caps else None


def _next_point(points: Sequence[PointSummary], ref: PointSummary) -> PointSummary | None:
    above = [
        p
        for p in points
        if p.experiment == ref.experiment
        and p.cell == ref.cell
        and p.workload == ref.workload
        and p.load_value > ref.load_value
    ]
    return min(above, key=lambda p: p.load_value) if above else None


def explain_ratio(
    a: PointSummary,
    b: PointSummary,
    fit_a: StepFit,
    fit_b: StepFit,
    points: Sequence[PointSummary] = (),
) -> RatioExplanation:
    step = a.itl_p50_ms
    return RatioExplanation(
        workload=a.workload,
        a=a.cell,
        b=b.cell,
        load_a=a.load_value,
        load_b=b.load_value,
        step_a_ms=step,
        batch_a_fit=fit_a.batch_at(step),
        batch_b_fit=fit_b.batch_at(step),
        fixed_a_ms=fit_a.fixed_ms,
        fixed_b_ms=fit_b.fixed_ms,
        per_seq_a_ms=fit_a.per_seq_ms,
        per_seq_b_ms=fit_b.per_seq_ms,
        tail_a=(a.tpot_p95_ms or float("nan")) / a.itl_p50_ms,
        tail_b=(b.tpot_p95_ms or float("nan")) / b.itl_p50_ms,
        prefill_floor_a=a.prefill_share,
        prefill_floor_b=b.prefill_share,
        prefill_implied_a=1 - a.itl_p50_ms / a.itl_mean_ms,
        prefill_implied_b=1 - b.itl_p50_ms / b.itl_mean_ms,
        fail_a=_fail_facts(_next_point(points, a)),
        fail_b=_fail_facts(_next_point(points, b)),
        kv_capacity_a=_capacity(points, a),
        kv_capacity_b=_capacity(points, b),
    )


@dataclass
class RooflineReport:
    theory: list[dict[str, Any]]
    points: list[PointSummary]
    fits: list[StepFit]
    goodput: dict[tuple[str, str, str], float]
    ratios: list[RatioExplanation]
    skipped: list[str]

    def at_goodput(self) -> list[PointSummary]:
        return [
            p
            for p in self.points
            if self.goodput.get((p.experiment, p.cell, p.workload)) == p.load_value
        ]

    def to_json(self) -> dict[str, Any]:
        return {
            "theory": self.theory,
            "points": [{**p.as_dict(), "at_goodput": p in self.at_goodput()} for p in self.points],
            "fits": [f.as_dict() for f in self.fits],
            "ratios": [r.as_dict() for r in self.ratios],
            "skipped": self.skipped,
        }


def build_report(
    db_url: str,
    experiments: Sequence[str],
    setups: Sequence[ServingSetup],
    *,
    compare: Sequence[tuple[str, str, str]] = (),
    roots: Sequence[Path] | None = None,
) -> RooflineReport:
    """`compare`: (workload, cell A, cell B) pairs whose goodput ratio to explain."""
    roots = list(roots) if roots is not None else default_roots(db_url)
    runs = load_runs(db_url, experiments) if experiments else []
    obs, skipped = observe_runs(runs, roots=roots)
    points = summarize_points(obs)
    fits = fit_steps(points)
    exp_ids = sorted({r.experiment_id for r in runs})
    goodput = goodput_loads(roots, exp_ids)
    ratios = []
    for workload, cell_a, cell_b in compare:
        # Both sides from the same experiment: cells of different experiments share names
        # (e.g. a profiling run's fixed load points next to a sweep's searched goodput).
        found = False
        for exp in exp_ids:
            pa = _goodput_point(points, goodput, cell_a, workload, exp)
            pb = _goodput_point(points, goodput, cell_b, workload, exp)
            fa = _fit(fits, exp, cell_a, workload)
            fb = _fit(fits, exp, cell_b, workload)
            if not (pa and pb and fa and fb):
                continue
            found = True
            if pa.load_mode != "open_loop" or pb.load_mode != "open_loop":
                # A closed-loop "goodput" is a concurrency, not a request rate.
                skipped.append(f"ratio {cell_b}/{cell_a} on {workload} in {exp[:8]}: closed loop")
                continue
            ratios.append(explain_ratio(pa, pb, fa, fb, points))
        if not found:
            skipped.append(f"ratio {cell_b}/{cell_a} on {workload}: missing goodput or fit")
    return RooflineReport(theory_rows(setups), points, fits, goodput, ratios, skipped)


def _goodput_point(
    points: Sequence[PointSummary],
    goodput: dict[tuple[str, str, str], float],
    cell: str,
    wl: str,
    exp: str,
) -> PointSummary | None:
    for p in points:
        if (
            p.experiment == exp
            and p.cell == cell
            and p.workload == wl
            and goodput.get((exp, cell, wl)) == p.load_value
        ):
            return p
    return None


def _fit(fits: Sequence[StepFit], exp: str, cell: str, wl: str) -> StepFit | None:
    return next(
        (f for f in fits if f.experiment == exp and f.cell == cell and f.workload == wl), None
    )


# -- rendering -------------------------------------------------------------------------


def _f(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def render_markdown(rep: RooflineReport) -> str:
    out = ["# Roofline: decode-step floors and stored runs", ""]
    out += [
        "Generated by `bench roofline`. Floors use spec-sheet peaks (`loom_bench.profile.hardware`);",  # noqa: E501
        "measured figures trace to the run ids in `roofline.json`. Explainer:",
        "[docs/perf/roofline-and-profiling.md](../../docs/perf/roofline-and-profiling.md).",
        "",
        "## Floors in theory",
        "",
        f"Decode-step floor (ms) by batch, every sequence holding {THEORY_CONTEXT} tokens; the "
        f"largest batch whose floor fits a {TPOT_SLO_MS:.0f} ms step; a 1,024-token prefill.",
        "",
    ]
    if rep.theory:
        bs = list(rep.theory[0]["step_floor_ms"])
        out.append(
            "| Config | GPU x TP | Weights read / GPU / step | KV / token / GPU | "
            + " | ".join(f"B={b}" for b in bs)
            + " | Max batch at 50 ms | 1k prefill |"
        )
        out.append("|---|---|---|---|" + "---|" * len(bs) + "---|---|")
        for r in rep.theory:
            out.append(
                f"| {r['label']} ({r['weights']} weights, {r['kv_cache']} KV) | {r['gpu']} x{r['tp']} "  # noqa: E501
                f"| {r['weight_gb_per_gpu_per_step']:.2f} GB | {r['kv_kib_per_token_per_gpu']:.0f} KiB | "  # noqa: E501
                + " | ".join(_f(r["step_floor_ms"][b]) for b in bs)
                + f" | {r['max_batch_at_slo']:.0f} | {_f(r['prefill_1k_ms'])} ms |"
            )
        out.append("")
    if rep.points:
        out += [
            "## Stored runs against the floor",
            "",
            "Batch and context are rebuilt from each run's per-request rows (token-weighted means "
            "over the measurement window); repetitions are averaged. *Achieved* = floor / measured "
            "ITL p50 (a typical decode step). *Achieved (mean)* adds the least time prefill takes "
            "and compares with the ITL mean. Rows marked **G** are each config's goodput.",
            "",
            "| Exp | Cell | Workload | Load | Batch | Context | Floor (w / kv / comm) ms | "
            "ITL p50 | ITL mean | TPOT p95 | Achieved | Achieved (mean) | Over floor |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        good = rep.at_goodput()
        for p in rep.points:
            fp = p.floor_parts
            ach = "n/a (spec.)" if p.speculative else _pct(p.achieved_step)
            achm = "n/a (spec.)" if p.speculative else _pct(p.achieved_mean)
            mark = " **G**" if p in good else ""
            out.append(
                f"| {p.experiment[:8]} | {p.cell} | {p.workload} | {p.load_value:.3g}{mark} | "
                f"{p.batch:.1f} | {p.context:.0f} | {p.floor_ms:.1f} ({fp['weights_ms']:.1f} / "
                f"{fp['kv_ms']:.1f} / {fp['comm_ms']:.1f}) | {p.itl_p50_ms:.1f} | {p.itl_mean_ms:.1f} | "  # noqa: E501
                f"{_f(p.tpot_p95_ms)} | {ach} | {achm} | {p.overhead_ms:.1f} |"
            )
        out.append("")
    if rep.fits:
        out += [
            "## Fixed and per-sequence step cost",
            "",
            "ITL p50 (less the modelled all-reduce) fitted as fixed + per-sequence x batch over each "  # noqa: E501
            "cell's load points. The fixed part's floor is the weight read; the per-sequence "
            "part's is that sequence's KV read.",
            "",
            "| Exp | Cell | Workload | Points | Fixed ms (floor) | Per sequence ms (floor) | "
            "Per-seq / floor | R² |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in rep.fits:
            out.append(
                f"| {f.experiment[:8]} | {f.cell} | {f.workload} | {f.points} | "
                f"{f.fixed_ms:.1f} ({f.weights_floor_ms:.1f}) | {f.per_seq_ms:.3f} "
                f"({f.per_seq_floor_ms:.3f}) | {f.per_seq_stretch:.2f}x | {f.r2:.3f} |"
            )
        out.append("")
    ttft = ttft_rows(rep.points)
    if ttft:
        out += [
            "## Time to first token at the lightest load",
            "",
            "TTFT p50 at each cell's lowest load point against the prefill floor for the mean "
            "prompt (FLOPs at the tensor peak plus TP all-reduce). The engine was still decoding "
            "other requests at these loads (batch shown), so a prompt shares its steps with them "
            "and the gap is not all prefill inefficiency. Workloads with prefix caching "
            "(shared-prefix) are left out: their uncached share varies per request.",
            "",
            "| Exp | Cell | Workload | Load | Decode batch | Prompt | Floor ms | TTFT p50 ms | Floor / TTFT |",  # noqa: E501
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in ttft:
            out.append(
                f"| {r['experiment'][:8]} | {r['cell']} | {r['workload']} | {r['load']:.3g} | "
                f"{r['batch']:.1f} | {r['prompt']:.0f} | {r['floor_ms']:.0f} | {r['ttft_p50_ms']:.0f} | "  # noqa: E501
                f"{_pct(r['floor_ms'] / r['ttft_p50_ms'])} |"
            )
        out.append("")
    for x in rep.ratios:
        out += [
            f"## Why {x.b} serves {x.measured_ratio:.2f}x {x.a} on {x.workload}",
            "",
            f"- Goodput: {x.a} {x.load_a:.3g} req/s, {x.b} {x.load_b:.3g} req/s.",
            f"- A step costs {x.fixed_a_ms:.1f} ms fixed + {x.per_seq_a_ms:.3f} ms per sequence on "
            f"{x.a}, {x.fixed_b_ms:.1f} + {x.per_seq_b_ms:.3f} on {x.b}.",
            f"- Within {x.a}'s step at goodput ({x.step_a_ms:.1f} ms ITL p50), that is "
            f"{x.batch_a_fit:.0f} sequences on {x.a} and {x.batch_b_fit:.0f} on {x.b}: "
            f"**{x.decode_ratio:.2f}x** if decode were the only work.",
            f"- The latency tail takes part back: TPOT p95 / ITL p50 is {x.tail_a:.2f} on {x.a} and "  # noqa: E501
            f"{x.tail_b:.2f} on {x.b}. Prefill's floor is {_pct(x.prefill_floor_a)} vs "
            f"{_pct(x.prefill_floor_b)} of the time; the mean-vs-median ITL gap implies "
            f"{_pct(x.prefill_implied_a)} vs {_pct(x.prefill_implied_b)}.",
        ]
        for cell, fail, cap in ((x.a, x.fail_a, x.kv_capacity_a), (x.b, x.fail_b, x.kv_capacity_b)):
            if fail is None:
                continue
            kv = f"KV cache {_pct(fail['kv_usage'])} full" if fail["kv_usage"] is not None else ""
            pre = f", {fail['preemptions']:.0f} preemptions per run" if fail["preemptions"] else ""
            size = f"; the cache holds ~{cap / 1000:.0f}k tokens" if cap else ""
            out.append(
                f"- {cell} fails at {fail['load']:.3g} req/s on {'; '.join(fail['failing'])} "
                f"({kv}{pre}{size})."
            )
        out.append("")
    if rep.skipped:
        out += ["## Skipped", ""] + [f"- {s}" for s in rep.skipped] + [""]
    return "\n".join(out)


def ttft_rows(points: Sequence[PointSummary]) -> list[dict[str, Any]]:
    lowest: dict[tuple[str, str, str], PointSummary] = {}
    for p in points:
        if p.workload == "shared-prefix" or p.ttft_p50_ms is None:
            continue
        key = (p.experiment, p.cell, p.workload)
        if key not in lowest or p.load_value < lowest[key].load_value:
            lowest[key] = p
    rows = []
    for (exp, cell, wl), p in sorted(lowest.items()):
        floor = prefill(p.setup, round(p.prompt_mean))
        assert p.ttft_p50_ms is not None
        rows.append(
            {
                "experiment": exp,
                "cell": cell,
                "workload": wl,
                "load": p.load_value,
                "prompt": p.prompt_mean,
                "batch": p.batch,
                "floor_ms": floor.total_ms,
                "ttft_p50_ms": p.ttft_p50_ms,
                "run_ids": p.run_ids,
            }
        )
    return rows


# Categorical slots 1-3 of the dataviz reference palette (validated all-pairs) and a
# neutral for "beyond the floor"; dark-mode steps under prefers-color-scheme.
_SVG_STYLE = """
.s1{fill:#2a78d6}.s2{fill:#eb6834}.s3{fill:#1baf7a}.s4{fill:#b9b8b3}
.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.ax{stroke:#d6d5d0;stroke-width:1}
text{font-family:system-ui,-apple-system,sans-serif;font-size:12px}
@media (prefers-color-scheme: dark){
.s1{fill:#3987e5}.s2{fill:#d95926}.s3{fill:#199e70}.s4{fill:#5d5c58}
.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.ax{stroke:#3a3a37}}
"""
_PARTS = (
    ("weights_ms", "s1", "Weight read (floor)"),
    ("kv_ms", "s2", "KV-cache read (floor)"),
    ("comm_ms", "s3", "All-reduce (model)"),
    ("over", "s4", "Beyond the floor (measured - floor)"),
)


def render_svg(points: Sequence[PointSummary]) -> str:
    """Horizontal stacked bars: a typical decode step (ITL p50) split into the floor's
    parts and the remainder, one bar per config at its goodput."""
    rows = [p for p in points if not p.speculative]
    label_w, bar_w, row_h, top = 350, 460, 30, 56
    height = top + row_h * len(rows) + 40
    width = label_w + bar_w + 160
    scale_max = max((p.itl_p50_ms for p in rows), default=1.0)
    x_of = lambda ms: label_w + ms / scale_max * bar_w  # noqa: E731
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t">',
        f"<style>{_SVG_STYLE}</style>",
        f'<rect class="bg" width="{width}" height="{height}"/>',
        '<title id="t">Where a typical decode step goes, at each config\'s goodput</title>',
        '<text class="t1" x="12" y="20" style="font-size:14px;font-weight:600">'
        "Where a typical decode step (ITL p50) goes, at goodput</text>",
    ]
    lx = 12
    for _, cls, name in _PARTS:
        out.append(f'<rect class="{cls}" x="{lx}" y="32" width="10" height="10" rx="2"/>')
        out.append(f'<text class="t2" x="{lx + 14}" y="41">{escape(name)}</text>')
        lx += 14 + 7 * len(name) + 18
    for i, p in enumerate(rows):
        y = top + i * row_h
        label = f"{p.experiment[:8]} {p.cell} · {p.workload} @ {p.load_value:.3g}"
        out.append(
            f'<text class="t1" x="{label_w - 8}" y="{y + 15}" text-anchor="end">{escape(label)}</text>'  # noqa: E501
        )
        parts = dict(p.floor_parts)
        floor_mem = parts["weights_ms"] + parts["kv_ms"] + parts["comm_ms"]
        parts["over"] = max(0.0, p.itl_p50_ms - floor_mem)
        x = float(label_w)
        for key, cls, name in _PARTS:
            w = parts[key] / scale_max * bar_w
            if w <= 0.2:
                continue
            gap = 2 if key != "over" else 0
            out.append(
                f'<rect class="{cls}" x="{x:.1f}" y="{y + 4}" width="{max(w - gap, 0.5):.1f}" '
                f'height="18" rx="2"><title>{escape(name)}: {parts[key]:.1f} ms</title></rect>'
            )
            x += w
        out.append(
            f'<text class="t2" x="{x_of(p.itl_p50_ms) + 6:.1f}" y="{y + 17}">'
            f"{p.itl_p50_ms:.1f} ms · {_pct(p.achieved_step)}</text>"
        )
    axis_y = top + row_h * len(rows) + 4
    out.append(
        f'<line class="ax" x1="{label_w}" x2="{label_w + bar_w}" y1="{axis_y}" y2="{axis_y}"/>'
    )
    for tick in range(0, int(scale_max) + 1, 10):
        tx = x_of(tick)
        out.append(
            f'<text class="t2" x="{tx:.1f}" y="{axis_y + 16}" text-anchor="middle">{tick}</text>'
        )
    out.append(
        f'<text class="t2" x="{label_w + bar_w / 2}" y="{axis_y + 32}" text-anchor="middle">'
        "ms per decode step (labels: measured ITL p50 · share of it the floor explains)</text>"
    )
    out.append("</svg>")
    return "\n".join(out)


def write_report(rep: RooflineReport, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "roofline.md"
    js = out_dir / "roofline.json"
    svg = out_dir / "roofline.svg"
    text = render_markdown(rep)
    if rep.points:
        text = text.replace(
            "## Stored runs against the floor",
            "![Where a decode step goes](roofline.svg)\n\n## Stored runs against the floor",
            1,
        )
    md.write_text(text)
    js.write_text(json.dumps(rep.to_json(), indent=2, default=str))
    svg.write_text(render_svg(rep.at_goodput()))
    return [md, js, svg]
