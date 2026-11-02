"""Draw the README charts from stored results.

Every number on the charts comes from a committed file, never from this script:

- `reports/8b-config-sweep-v1/leaderboard.csv` (experiment 7a8237d0), written by
  `bench report`.
- `docs/img/data/70b-h100-9f0853d7-load-sweep.csv`: per-load latency of experiment
  9f0853d7, which `bench report` does not export (its CSV holds the goodput point only).
  The `extract` step writes it from the results database with the Lab's own aggregation
  (`loom_bench.report.analyze.load_points`, `loom_bench.slo.slo_met`) and checks the
  goodput rows against `reports/70b-h100-v2/leaderboard.csv`.

Usage (matplotlib is not a project dependency, so it is added for the run):

    uv run --with matplotlib python scripts/readme_charts.py            # draw from the CSVs
    uv run --with matplotlib python scripts/readme_charts.py extract \\
        --db "sqlite:///file:results/loom.db?mode=ro&uri=true"          # refresh the 70B CSV
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "docs" / "img"
SWEEP_8B_CSV = ROOT / "reports" / "8b-config-sweep-v1" / "leaderboard.csv"
REPORT_70B_CSV = ROOT / "reports" / "70b-h100-v2" / "leaderboard.csv"
LOADS_70B_CSV = IMG / "data" / "70b-h100-9f0853d7-load-sweep.csv"
EXPERIMENT_70B = "9f0853d7"

# Chart chrome: a solid light surface, so the PNGs read the same in GitHub's light and
# dark themes. Series colours are the first two slots of a CVD-checked categorical
# palette; grey marks configs that did not ship.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
NOT_SHIPPED = "#b5b3ab"

CONFIG_NAMES_8B = {
    "bf16": "BF16 (reference)",
    "bf16-kv8": "BF16 + FP8 KV cache",
    "fp8": "FP8 weights",
    "fp8-kv8": "FP8 + FP8 KV cache",
    "fp8-kv8-mbt1024": "FP8 + FP8 KV cache,\nmax batched tokens 1024",
}
CONFIG_NAMES_70B = {"vllm-tp2": "BF16", "vllm-tp2-fp8": "FP8"}
# Where each goodput label sits relative to its point (offset points), clear of the lines.
LABEL_OFFSETS_70B = {
    ("vllm-tp2", "fixed-1k-1k"): (10, 50),
    ("vllm-tp2-fp8", "fixed-1k-1k"): (12, 34),
    ("vllm-tp2", "shared-prefix"): (-12, 38),
    ("vllm-tp2-fp8", "shared-prefix"): (-14, 50),
}
LOAD_FIELDS = [
    "experiment",
    "config",
    "workload",
    "load_req_s",
    "n_runs",
    "tpot_p95_ms",
    "tpot_p95_ms_lo",
    "tpot_p95_ms_hi",
    "ttft_p95_ms",
    "ttft_p95_ms_lo",
    "ttft_p95_ms_hi",
    "error_rate",
    "slo_met",
]


def _style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
        }
    )


def _usd(v: str) -> float:
    return float(v.lstrip("$"))


def chart_8b_sweep(out: Path) -> None:
    """Goodput and $/1M blended per Qwen3-8B config on chat-sharegpt, gate marked."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows = {
        r["config"]: r
        for r in csv.DictReader(SWEEP_8B_CSV.open())
        if r["workload"] == "chat-sharegpt"
    }
    order = [c for c in CONFIG_NAMES_8B if c in rows]
    if len(order) != len(CONFIG_NAMES_8B):
        sys.exit(f"{SWEEP_8B_CSV}: expected configs {list(CONFIG_NAMES_8B)}, got {list(rows)}")

    fig, (ax_g, ax_c) = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    y = list(range(len(order)))[::-1]
    colours, gate_text = [], []
    for c in order:
        gate = rows[c]["quality_gate"]
        shipped = gate in ("baseline", "pass")
        colours.append(BLUE if shipped else NOT_SHIPPED)
        gate_text.append("reference" if gate == "baseline" else f"gate {gate}")

    goodput = [float(rows[c]["goodput_load"]) for c in order]
    failing = [float(rows[c]["first_failing_load"]) for c in order]
    ax_g.barh(y, goodput, height=0.55, color=colours)
    for yi, g, f in zip(y, goodput, failing, strict=True):
        # The search bracket: goodput is at least g and below the load that failed.
        ax_g.plot([g, f], [yi, yi], color=INK_2, lw=1.2)
        ax_g.plot([f], [yi], marker="|", ms=9, color=INK_2)
        ax_g.text(f + 0.12, yi, f"{g:.2f}", va="center", color=INK, fontsize=10)
    ax_g.set_xlim(0, max(failing) * 1.18)
    ax_g.set_title("Goodput at SLO (req/s)")
    ax_g.set_xlabel("highest load meeting the SLO; whisker = load that failed")
    ax_g.grid(axis="y", visible=False)

    cost = [_usd(rows[c]["split_blended_per_mtok_usd"]) for c in order]
    lo = [_usd(rows[c]["split_blended_per_mtok_lo_usd"]) for c in order]
    hi = [_usd(rows[c]["split_blended_per_mtok_hi_usd"]) for c in order]
    ax_c.barh(y, cost, height=0.55, color=colours)
    ax_c.errorbar(
        cost,
        y,
        xerr=[
            [c - lo_ for c, lo_ in zip(cost, lo, strict=True)],
            [hi_ - c for c, hi_ in zip(cost, hi, strict=True)],
        ],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1.2,
        capsize=3,
    )
    for yi, c, h, g in zip(y, cost, hi, gate_text, strict=True):
        ax_c.text(h + 0.003, yi, f"${c:.4f}  ·  {g}", va="center", color=INK, fontsize=10)
    ax_c.set_xlim(0, max(hi) * 1.75)
    ax_c.set_title("Cost per 1M tokens, blended (USD)")
    ax_c.set_xlabel("on-demand $1.10/h; whisker = 95% CI")
    ax_c.grid(axis="y", visible=False)

    ax_g.set_yticks(y, [CONFIG_NAMES_8B[c] for c in order])
    ax_g.tick_params(axis="y", length=0)
    ax_c.tick_params(axis="y", length=0)
    fig.legend(
        handles=[
            Patch(color=BLUE, label="ships (BF16 reference)"),
            Patch(
                color=NOT_SHIPPED,
                label="faster, but the quality gate vs BF16 is inconclusive: not shipped",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        ncol=2,
    )
    fig.suptitle(
        "Qwen3-8B on 1x L40S, vLLM 0.30, chat-sharegpt (experiment 7a8237d0)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.895,
        "SLO: TTFT p95 <= 1 s, TPOT p95 <= 50 ms, judged on the 95% CI upper bound over 3 "
        "repetitions per load",
        color=INK_2,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.2, right=0.985, top=0.78, bottom=0.2, wspace=0.06)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _read_loads() -> list[dict[str, str]]:
    if not LOADS_70B_CSV.exists():
        sys.exit(f"{LOADS_70B_CSV} is missing: run `extract --db ...` first")
    return list(csv.DictReader(LOADS_70B_CSV.open()))


def chart_70b_tpot(out: Path) -> None:
    """TPOT p95 against offered load, 70B BF16 vs FP8 on 2x H100 SXM, SLO line."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _read_loads()
    workloads = [
        ("fixed-1k-1k", "fixed-1k-1k (1,024 in / 1,024 out)"),
        ("shared-prefix", "shared-prefix (2,083 in / 128 out)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, (wl, title) in zip(axes, workloads, strict=True):
        for config, colour in (("vllm-tp2", BLUE), ("vllm-tp2-fp8", ORANGE)):
            pts = sorted(
                (r for r in rows if r["workload"] == wl and r["config"] == config),
                key=lambda r: float(r["load_req_s"]),
            )
            x = [float(r["load_req_s"]) for r in pts]
            m = [float(r["tpot_p95_ms"]) for r in pts]
            lo = [float(r["tpot_p95_ms_lo"]) for r in pts]
            hi = [float(r["tpot_p95_ms_hi"]) for r in pts]
            ax.plot(x, m, color=colour, lw=2, solid_capstyle="round", zorder=3)
            ax.errorbar(
                x,
                m,
                yerr=[
                    [a - b for a, b in zip(m, lo, strict=True)],
                    [b - a for a, b in zip(m, hi, strict=True)],
                ],
                fmt="none",
                ecolor=colour,
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
            for r, xi, mi in zip(pts, x, m, strict=True):
                met = r["slo_met"] == "true"
                ax.plot(
                    [xi],
                    [mi],
                    marker="o",
                    ms=7.5,
                    mfc=colour if met else SURFACE,
                    mec=colour,
                    mew=2,
                    zorder=4,
                )
            best = [xi for r, xi in zip(pts, x, strict=True) if r["slo_met"] == "true"]
            if best:
                g = max(best)
                gm = m[x.index(g)]
                ax.annotate(
                    f"{CONFIG_NAMES_70B[config]} goodput\n{g:g} req/s",
                    (g, gm),
                    xytext=LABEL_OFFSETS_70B[(config, wl)],
                    textcoords="offset points",
                    ha="left" if LABEL_OFFSETS_70B[(config, wl)][0] > 0 else "right",
                    color=INK,
                    fontsize=9.5,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8},
                )
        ax.axhline(50, color=INK_2, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.set_xscale("log", base=2)
        ticks = sorted({float(r["load_req_s"]) for r in rows if r["workload"] == wl})
        # Label the round loads only; the bisected ones (0.595, 0.707) stay unlabelled ticks.
        ax.set_xticks(ticks, [f"{t:g}" if math.log2(t) % 0.5 == 0 else "" for t in ticks])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("offered load (req/s, open loop, log scale)")
    axes[0].set_ylabel("TPOT p95 (ms)")
    axes[0].set_ylim(0, 260)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, marker="o", mfc=BLUE, mec=BLUE, label="BF16"),
            Line2D([], [], color=ORANGE, lw=2, marker="o", mfc=ORANGE, mec=ORANGE, label="FP8"),
            Line2D([], [], color=INK_2, lw=1.2, ls=(0, (4, 3)), label="SLO: TPOT p95 50 ms"),
            Line2D([], [], ls="", marker="o", mfc=INK_2, mec=INK_2, label="load met the SLO"),
            Line2D(
                [],
                [],
                ls="",
                marker="o",
                mfc=SURFACE,
                mec=INK_2,
                mew=2,
                label="load failed (TPOT or TTFT)",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        ncol=5,
    )
    fig.suptitle(
        f"Llama 3.3 70B on 2x H100 SXM, TP=2, vLLM 0.30 (experiment {EXPERIMENT_70B})",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        0.895,
        "Geometric mean of 3 repetitions per load, whisker = 95% CI. A load passes only if "
        "TTFT p95 <= 1 s and TPOT p95 <= 50 ms at the CI upper bound.",
        color=INK_2,
        fontsize=9.5,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.2, wspace=0.06)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def extract(db: str) -> None:
    """Write the 70B per-load CSV from the results database, with the Lab's aggregation."""
    import uuid

    from sqlalchemy import select

    from loom_bench.report.analyze import load_points
    from loom_bench.slo import Slo, slo_met
    from loom_bench.store.db import session_scope
    from loom_bench.store.models import BenchExperiment
    from loom_bench.store.repo import list_runs

    with session_scope(db) as s:
        exps = [
            e
            for e in s.scalars(select(BenchExperiment))
            if str(e.id).replace("-", "").startswith(EXPERIMENT_70B)
        ]
        if len(exps) != 1:
            sys.exit(f"expected one experiment {EXPERIMENT_70B} in {db}, found {len(exps)}")
        exp = exps[0]
        slo = Slo.model_validate(exp.spec["slo"])
        runs = list_runs(s, experiment_id=uuid.UUID(str(exp.id)))
        cell_of = {r.config_hash: r.cell_key for r in runs}
        points = load_points(runs)

    out_rows: list[dict[str, str]] = []
    for key, pts in sorted(points.items()):
        for p in pts:
            agg = p.aggregate
            tpot, ttft, err = agg.get("tpot_ms.p95"), agg.get("ttft_ms.p95"), agg.get("error_rate")
            if tpot is None or ttft is None or err is None:
                sys.exit(f"missing latency metrics at {key} {p.load}")
            out_rows.append(
                {
                    "experiment": str(exp.id),
                    "config": cell_of[key.config_hash] or key.config_hash,
                    "workload": key.workload,
                    "load_req_s": f"{p.load:.6g}",
                    "n_runs": str(agg.n_runs),
                    "tpot_p95_ms": f"{tpot.mean:.2f}",
                    "tpot_p95_ms_lo": f"{tpot.lo:.2f}",
                    "tpot_p95_ms_hi": f"{tpot.hi:.2f}",
                    "ttft_p95_ms": f"{ttft.mean:.1f}",
                    "ttft_p95_ms_lo": f"{ttft.lo:.1f}",
                    "ttft_p95_ms_hi": f"{ttft.hi:.1f}",
                    "error_rate": f"{err.mean:.4f}",
                    "slo_met": "true" if slo_met(slo, agg).met else "false",
                }
            )
    _check_against_report(out_rows)
    LOADS_70B_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LOADS_70B_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOAD_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {LOADS_70B_CSV.relative_to(ROOT)}: {len(out_rows)} load points")


def _check_against_report(rows: list[dict[str, str]]) -> None:
    """The highest passing load and its TPOT must match the committed report."""
    for r in csv.DictReader(REPORT_70B_CSV.open()):
        mine = [x for x in rows if x["config"] == r["config"] and x["workload"] == r["workload"]]
        passing = [x for x in mine if x["slo_met"] == "true"]
        if not passing:
            sys.exit(
                f"{r['config']} {r['workload']}: no passing load, report says {r['goodput_load']}"
            )
        top = max(passing, key=lambda x: float(x["load_req_s"]))
        if not math.isclose(float(top["load_req_s"]), float(r["goodput_load"]), rel_tol=1e-4):
            sys.exit(f"{r['config']} {r['workload']}: goodput {top['load_req_s']} != report")
        if not math.isclose(float(top["tpot_p95_ms"]), float(r["tpot_p95_ms"]), abs_tol=0.01):
            sys.exit(f"{r['config']} {r['workload']}: TPOT {top['tpot_p95_ms']} != report")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    ex = sub.add_parser("extract", help="refresh the 70B per-load CSV from a results DB")
    ex.add_argument("--db", required=True, help="results database URL (read only)")
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.db)
        return
    _style()
    IMG.mkdir(parents=True, exist_ok=True)
    chart_8b_sweep(IMG / "8b-config-sweep.png")
    chart_70b_tpot(IMG / "70b-h100-tpot.png")
    print("wrote docs/img/8b-config-sweep.png, docs/img/70b-h100-tpot.png")


if __name__ == "__main__":
    main()
