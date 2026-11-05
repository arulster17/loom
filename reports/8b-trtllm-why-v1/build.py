"""Rebuild this report's breakdowns and tables from the stored traces and runs (free, no GPU).

    uv run python reports/8b-trtllm-why-v1/build.py /path/to/results /path/to/loom.db

1. Every profile window of the experiments below goes through
   `loom_bench.profile.trace.analyze_path` (what `bench profile analyze` runs, with this
   commit's TensorRT-LLM step labels and kernel categories) and lands in
   `breakdowns/<experiment>_<cell>_<workload>_<load>_<run id>.json`. vLLM's 14d959e5
   windows are re-analysed with the same code, so both engines are split alike.
2. `tables.md` is written from those files and from the load runs in the results DB
   (opened read-only).
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

from loom_bench.profile.trace import analyze_path

HERE = Path(__file__).parent
OUT = HERE / "breakdowns"
CATS = ("gemm", "attention", "quant", "sampling", "memcpy", "other", "idle")
TRT_BF16 = "71e2c32c-ed94-4e09-9142-015b96b6745f"  # qwen3-8b-trtllm-bf16-profile-aws
TRT_FP8 = "ad5b1408-beb3-48e3-8572-bb81fd582339"  # qwen3-8b-trtllm-fp8-cudagraph-aws
SMOKE = "d35e631b-3bf4-4b08-af5b-cd7cd155a817"  # aws-smoke-trtllm-why
VLLM = "14d959e5-5a13-4939-a56d-71a8d87d9d2a"  # qwen3-8b-profile-aws (vLLM v0.30.0)
STOCK_FP8_EARLIER = "08204560-e609-4ef0-aa0e-9577153bd656"  # qwen3-8b-trtllm-fp8-aws
LIGHT_STOCK = "e3694f45-17cc-4233-a107-d0f1faf34d3e"  # aws-smoke-trtllm-fp8 (stock FP8)
EXPERIMENTS = (TRT_BF16, TRT_FP8, SMOKE, VLLM)
# Core decode-attention kernels: TensorRT-LLM's XQA (`kernel_mha`) and masked multi-head
# attention, vLLM's FlashAttention split-KV kernel and its combine step.
CORE_ATTN = re.compile(r"^kernel_mha$|masked_multihead_attention|flash_fwd_splitkv")
# TensorRT-LLM 1.4.0rc2's default decode CUDA-graph batch sizes (enable_padding false).
TRT_GRAPHS = set(range(1, 33)) | {64, 128}
HBM = 864e9
WEIGHT_BYTES_BF16 = 15.14e9


def build(results: Path) -> None:
    OUT.mkdir(exist_ok=True)
    for exp in EXPERIMENTS:
        for w in sorted((results / exp / "profiles").glob("*/window.json")):
            doc = json.loads(w.read_text())
            if exp == VLLM and doc["cell"] != "bf16":
                continue
            res = analyze_path(w.parent / "trace.tar.gz")
            keep = ("run_id", "cell", "workload", "load_value", "client_latency", "capture")
            name = (
                f"{exp[:8]}_{doc['cell']}_{doc['workload']}_{doc['load_value']:g}_"
                f"{doc['run_id'][:8]}.json"
            )
            body = {
                "experiment": exp,
                "engine": "vllm" if exp == VLLM else "trtllm",
                "window": {k: doc[k] for k in keep},
                "trace_bytes": doc["trace_bytes"],
            }
            (OUT / name).write_text(json.dumps({**body, **res.as_dict()}, indent=1))


def load() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]


def find(rows: list[dict], exp: str, cell: str, workload: str, load: float) -> dict:
    (d,) = [
        r
        for r in rows
        if r["experiment"] == exp
        and r["window"]["cell"] == cell
        and r["window"]["workload"] == workload
        and r["window"]["load_value"] == load
    ]
    return d


def decode(d: dict, batch: int) -> dict | None:
    for g in d["ranks"][0]["step_shapes"]:
        if g["kind"] == "decode" and g["batch"] == batch:
            return g
    return None


def core_attn_per_step(d: dict) -> float:
    r = d["ranks"][0]
    total = sum(k["total_ms"] for k in r["top_kernels"] if CORE_ATTN.search(k["name"]))
    return total / r["steps"]


def f(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def wls(points: list[tuple[float, float, int]]) -> tuple[float, float]:
    """Weighted least squares y = a + b x over (x, y, weight)."""
    sw = sum(w for _, _, w in points)
    mx = sum(x * w for x, _, w in points) / sw
    my = sum(y * w for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    b = sxy / sxx
    return my - b * mx, b


def windows_table(rows: list[dict]) -> list[str]:
    out = [
        "## Windows",
        "",
        "Every profile window analysed here. ITL = the client's inter-token gap inside the "
        "window and just before it (same job); slowdown = their ratio. TensorRT-LLM records "
        "`num_steps` steps (400; 32 in the smoke) and stops by itself; vLLM v0.30 records "
        "the whole 20 s.",
        "",
        "| Experiment | Engine | Cell | Workload | Load | Run | Steps | Window s | Trace MB | ITL p50 profiled / before (ms) | Slowdown |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in rows:
        w, r = d["window"], d["ranks"][0]
        lat = w["client_latency"]
        p, b = lat["profiled"]["itl_p50_ms"], lat["before"]["itl_p50_ms"]
        out.append(
            f"| {d['experiment'][:8]} | {d['engine']} | {w['cell']} | {w['workload']} | "
            f"{w['load_value']:g} | {w['run_id'][:8]} | {r['steps']} | "
            f"{r['window_ms'] / 1000:.1f} | {d['trace_bytes'] / 1e6:.1f} | "
            f"{f(p)} / {f(b)} | {f(lat.get('profiler_slowdown'), 3)} |"
        )
    return out


def matched_table(rows: list[dict]) -> list[str]:
    trt16 = find(rows, TRT_BF16, "trtllm-bf16", "fixed-1k-1k", 16)
    trt64 = find(rows, TRT_BF16, "trtllm-bf16", "fixed-1k-1k", 64)
    trtc = find(rows, TRT_BF16, "trtllm-bf16", "chat-sharegpt", 2.25)
    trt1 = find(rows, SMOKE, "trtllm-bf16", "chat-sharegpt", 1)
    v1 = find(rows, VLLM, "bf16", "fixed-1k-1k", 1)
    v16 = find(rows, VLLM, "bf16", "fixed-1k-1k", 16)
    v64 = find(rows, VLLM, "bf16", "fixed-1k-1k", 64)
    vc = find(rows, VLLM, "bf16", "chat-sharegpt", 2.213)
    pairs: list[tuple[str, dict, dict, int]] = [
        ("1 (TRT: smoke chat 1.0; vLLM: 1k-1k)", trt1, v1, 1),
        ("16 (1k-1k, 16 in flight)", trt16, v16, 16),
    ]
    pairs += [(f"{b} (chat ~2.2 req/s)", trtc, vc, b) for b in range(21, 28)]
    pairs.append(("64 (1k-1k, 64 in flight)", trt64, v64, 64))
    out = [
        "## BF16 decode steps at matched batch sizes: TensorRT-LLM 1.4.0rc2 vs vLLM v0.30.0",
        "",
        "Pure-decode steps only, ms per step (step start to next step start). T = "
        "TensorRT-LLM (71e2c32c; batch 1 from smoke d35e631b), V = vLLM (14d959e5), same "
        "g6e.xlarge type, BF16 weights and KV. Attention includes RoPE and KV-cache writes; "
        "normalization kernels are `other` in both. Context lengths are not in the step "
        "labels: the 1k-1k windows are closed loops whose requests start together, so their "
        "contexts depend on where in the requests' life each window fell; the chat windows "
        "are open-loop steady states at near-equal load, the fairest pairing.",
        "",
        "| Batch | n T / V | Step p50 T / V | Step mean T / V | Δ mean | GEMM T / V | Attention T / V | Other T / V | Idle T / V |",  # noqa: E501
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, t, v, b in pairs:
        gt, gv = decode(t, b), decode(v, b)
        if gt is None or gv is None:
            continue
        ct, cv = gt["category_ms"], gv["category_ms"]
        out.append(
            f"| {label} | {gt['steps']} / {gv['steps']} | "
            f"{f(gt['step_ms_p50'])} / {f(gv['step_ms_p50'])} | "
            f"{f(gt['step_ms_mean'])} / {f(gv['step_ms_mean'])} | "
            f"{gt['step_ms_mean'] - gv['step_ms_mean']:+.2f} | "
            f"{f(ct['gemm'])} / {f(cv['gemm'])} | {f(ct['attention'])} / {f(cv['attention'])} | "
            f"{f(ct['other'])} / {f(cv['other'])} | {f(ct['idle'])} / {f(cv['idle'])} |"
        )
    out += [
        "",
        "### Per extra sequence (chat windows, decode steps weighted by count)",
        "",
        "Least-squares line of each figure on the batch size over the chat windows' "
        "pure-decode steps (TensorRT-LLM batches 20-27, vLLM 19-29). Not reliable: the "
        "range is 7-10 batches wide and, inside one window, the context mix changes with "
        "the batch, so these slopes even disagree in sign with the 16-to-64 figures below "
        "and with the client fits (roofline/). The level gap at each batch above is the "
        "usable result.",
        "",
        "| Figure | TensorRT-LLM ms per sequence | vLLM ms per sequence | Ratio |",
        "|---|---|---|---|",
    ]
    for key, name in (
        ("step_ms_mean", "Step"),
        ("gemm", "GEMM"),
        ("attention", "Attention"),
    ):
        slopes = []
        for d in (trtc, vc):
            pts = []
            for g in d["ranks"][0]["step_shapes"]:
                if g["kind"] != "decode":
                    continue
                y = g[key] if key == "step_ms_mean" else g["category_ms"][key]
                pts.append((g["batch"], y, g["steps"]))
            slopes.append(wls(pts)[1])
        out.append(f"| {name} | {slopes[0]:.3f} | {slopes[1]:.3f} | {slopes[0] / slopes[1]:.2f} |")
    out += [
        "",
        "### Attention from 16 to 64 in flight (1k-1k)",
        "",
        "| | TensorRT-LLM | vLLM |",
        "|---|---|---|",
    ]
    a = {
        k: decode(d, b)["category_ms"]["attention"]
        for k, d, b in (("t16", trt16, 16), ("t64", trt64, 64), ("v16", v16, 16), ("v64", v64, 64))  # type: ignore[index]
    }
    out.append(f"| Attention at 16 (ms) | {f(a['t16'])} | {f(a['v16'])} |")
    out.append(f"| Attention at 64 (ms) | {f(a['t64'])} | {f(a['v64'])} |")
    out.append(
        f"| Per sequence, 16 to 64 (ms) | {(a['t64'] - a['t16']) / 48:.3f} | "
        f"{(a['v64'] - a['v16']) / 48:.3f} |"
    )
    out += [
        "",
        "### Core decode-attention kernel per step (whole window)",
        "",
        "TensorRT-LLM: `kernel_mha` (XQA); vLLM: `flash_fwd_splitkv_kernel` and its combine "
        "kernel. Whole-window totals over steps, so steps with prefill are in the divisor.",
        "",
        "| Window | TensorRT-LLM ms / step | vLLM ms / step | Ratio |",
        "|---|---|---|---|",
    ]
    for label, t, v in (
        ("1k-1k, 16", trt16, v16),
        ("1k-1k, 64", trt64, v64),
        ("chat ~2.2 req/s", trtc, vc),
    ):
        ct, cv = core_attn_per_step(t), core_attn_per_step(v)
        out.append(f"| {label} | {f(ct)} | {f(cv)} | {ct / cv:.2f} |")
    return out


def gemm_kernel_table(rows: list[dict]) -> list[str]:
    out = [
        "## Which GEMM kernels ran (BF16)",
        "",
        "Top GEMM kernels of each window by total time, as a share of the window's GEMM "
        "kernel time. Both engines call cuBLAS / CUTLASS through PyTorch for BF16 linear "
        "layers; the kernel cuBLAS picks depends on the batch (the M dimension).",
        "",
        "| Engine | Window | Kernel | Share of GEMM time |",
        "|---|---|---|---|",
    ]
    for exp, cell, wl, load in (
        (TRT_BF16, "trtllm-bf16", "fixed-1k-1k", 16),
        (VLLM, "bf16", "fixed-1k-1k", 16),
        (TRT_BF16, "trtllm-bf16", "chat-sharegpt", 2.25),
        (VLLM, "bf16", "chat-sharegpt", 2.213),
        (TRT_BF16, "trtllm-bf16", "fixed-1k-1k", 64),
        (VLLM, "bf16", "fixed-1k-1k", 64),
    ):
        d = find(rows, exp, cell, wl, load)
        r = d["ranks"][0]
        gemm_total = r["category_ms"]["gemm"]
        top = [k for k in r["top_kernels"] if k["category"] == "gemm"][:3]
        for k in top:
            name = re.sub(r"^void cutlass::Kernel2<|\(.*$", "", k["name"])[:80]
            share = 100 * k["total_ms"] / gemm_total
            out.append(f"| {d['engine']} | {wl} {load:g} | `{name}` | {share:.0f}% |")
    t16 = decode(find(rows, TRT_BF16, "trtllm-bf16", "fixed-1k-1k", 16), 16)
    v1 = decode(find(rows, VLLM, "bf16", "fixed-1k-1k", 1), 1)
    t1 = decode(find(rows, SMOKE, "trtllm-bf16", "chat-sharegpt", 1), 1)
    out += [
        "",
        "GEMM bandwidth (weights read per step over GEMM time per step, 15.14 GB):",
        "",
        "| Step | GEMM ms | Achieved GB/s | % of 864 GB/s |",
        "|---|---|---|---|",
    ]
    for label, g in (("TRT batch 1", t1), ("vLLM batch 1", v1), ("TRT batch 16", t16)):
        assert g is not None
        ms = g["category_ms"]["gemm"]
        gbs = WEIGHT_BYTES_BF16 / (ms / 1000) / 1e9
        out.append(f"| {label} | {f(ms)} | {gbs:.0f} | {100 * gbs * 1e9 / HBM:.0f}% |")
    return out


def eager_table(rows: list[dict]) -> list[str]:
    d = find(rows, TRT_BF16, "trtllm-bf16", "fixed-1k-1k", 64)
    out = [
        "## TensorRT-LLM decode steps with and without a CUDA graph (BF16, 1k-1k 64 window)",
        "",
        "While the 64-request batch refilled, single decode-only steps ran at the batch "
        "sizes below. TensorRT-LLM 1.4.0rc2 captures decode graphs for 1-32, 64 and 128 by "
        "default, padding off: other sizes run eagerly, launched kernel by kernel from "
        "Python. One step each except 1 and 64, so these are indications, not measurements "
        "of a distribution.",
        "",
        "| Batch | CUDA graph | Steps | Step ms | GPU busy ms | GPU idle ms |",
        "|---|---|---|---|---|---|",
    ]
    for g in d["ranks"][0]["step_shapes"]:
        if g["kind"] != "decode":
            continue
        idle = g["category_ms"]["idle"]
        out.append(
            f"| {g['batch']} | {'yes' if g['batch'] in TRT_GRAPHS else 'no'} | {g['steps']} | "
            f"{f(g['step_ms_mean'])} | {f(g['step_ms_mean'] - idle)} | {f(idle)} |"
        )
    return out


def fp8_windows_table(rows: list[dict]) -> list[str]:
    pad = find(rows, TRT_FP8, "trtllm-fp8-cudagraph-pad", "chat-sharegpt", 3.375)
    stock = find(rows, TRT_FP8, "trtllm-fp8", "chat-sharegpt", 3.375)
    out = [
        "## FP8 at chat 3.375 req/s: padded vs stock profile windows (ad5b1408)",
        "",
        "Both windows ran after their cell's measured runs, at batches 15-30 (neither in "
        "the slow state). At these batches every decode step has a graph in both configs, "
        "and the steps match to within 0.1 ms.",
        "",
        "| Batch | n pad / stock | Step mean pad / stock | GEMM pad / stock | Attention pad / stock | Idle pad / stock |",  # noqa: E501
        "|---|---|---|---|---|---|",
    ]
    for g in pad["ranks"][0]["step_shapes"]:
        if g["kind"] != "decode":
            continue
        s = decode(stock, g["batch"])
        if s is None or g["steps"] < 10 or s["steps"] < 10:
            continue
        cp, cs = g["category_ms"], s["category_ms"]
        out.append(
            f"| {g['batch']} | {g['steps']} / {s['steps']} | {f(g['step_ms_mean'])} / "
            f"{f(s['step_ms_mean'])} | {f(cp['gemm'])} / {f(cs['gemm'])} | "
            f"{f(cp['attention'])} / {f(cs['attention'])} | {f(cp['idle'])} / {f(cs['idle'])} |"
        )
    return out


def smoke_bs4_table(rows: list[dict]) -> list[str]:
    out = [
        "## The batch-4 attention anomaly (FP8 with FP8 KV cache, smoke d35e631b)",
        "",
        "Decode steps of the padded FP8 cell by batch, with the attention kernel that ran.",
        "",
        "| Window | Batch | Steps | Step mean ms | Attention ms | Attention kernels in window (ms total) |",  # noqa: E501
        "|---|---|---|---|---|---|",
    ]
    for wl, load in (("fixed-1k-1k", 4), ("chat-sharegpt", 1)):
        d = find(rows, SMOKE, "trtllm-fp8-cudagraph-pad", wl, load)
        r = d["ranks"][0]
        kern = "; ".join(
            f"`{'MMHA' if 'masked_multihead' in k['name'] else k['name'][:24]}` {k['total_ms']:.0f}"
            for k in r["top_kernels"]
            if CORE_ATTN.search(k["name"])
        )
        for g in r["step_shapes"]:
            if g["kind"] != "decode":
                continue
            out.append(
                f"| {wl} {load:g} | {g['batch']} | {g['steps']} | {f(g['step_ms_mean'])} | "
                f"{f(g['category_ms']['attention'])} | {kern} |"
            )
    bf = find(rows, SMOKE, "trtllm-bf16", "fixed-1k-1k", 4)
    g = decode(bf, 4)
    assert g is not None
    out.append(
        f"| BF16 (BF16 KV) fixed-1k-1k 4 | 4 | {g['steps']} | {f(g['step_ms_mean'])} | "
        f"{f(g['category_ms']['attention'])} | `kernel_mha` (XQA) |"
    )
    return out


def fp8_runs_table(db: Path) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select experiment_id, config_hash, load_value, repetition, summary from bench_runs"
        " where workload = 'chat-sharegpt' and cell_key like 'trtllm-fp8%'"
        " and experiment_id in (?, ?) and status = 'completed'",
        (TRT_FP8.replace("-", ""), STOCK_FP8_EARLIER.replace("-", "")),
    ).fetchall()
    con.close()
    runs: dict[tuple[str, float, int], dict[str, Any]] = {}
    for exp, h, load, rep, summ in rows:
        s = json.loads(summ)
        srv = s.get("server") or {}
        if h.startswith("2d964304"):
            who = "pad"
        elif exp.startswith(TRT_FP8[:8]):
            who = "stock-ad5b"
        else:
            who = "stock-0820"
        runs[(who, round(load, 3), rep)] = {
            "itl": s["itl_ms"]["p50"],
            "tpot": s["tpot_ms"]["p95"],
            "run_mean": srv.get("running_mean"),
            "run_max": srv.get("running_max"),
            "rate": s["throughput"]["request_rate"],
        }
    out = [
        "## FP8 chat runs: padded vs stock on identical arrivals",
        "",
        "Each load and repetition replays the same seeded arrival trace in every cell "
        "(the measured request rates match to the digit), so each row is a paired "
        "comparison. Stock = 55b35655 (default CUDA graphs), in 08204560 (2026-11-04) and "
        "again in ad5b1408 (2026-11-05, same host as the padded cell). Padded = 2d964304 "
        "(`enable_padding: true, max_batch_size: 128`). ITL p50 / TPOT p95 in ms; running "
        "= the engine's running-request gauge, mean / max over the run. **Bold**: the "
        "run's steps went slow (TPOT p95 > 40 ms).",
        "",
        "| Load | Rep | Stock 08204560 | Stock ad5b1408 | Padded ad5b1408 |",
        "|---|---|---|---|---|",
    ]

    def cell(r: dict[str, Any] | None) -> str:
        if r is None:
            return "n/a"
        txt = (
            f"{r['itl']:.1f} / {r['tpot']:.1f} ms, running {r['run_mean']:.1f} / {r['run_max']:.0f}"
        )
        return f"**{txt}**" if r["tpot"] > 40 else txt

    slow = {"pad": 0, "stock-ad5b": 0, "stock-0820": 0}
    for load in (2.619, 3.375):
        for rep in range(3):
            row = {w: runs.get((w, load, rep)) for w in slow}
            for w, r in row.items():
                if r is not None and r["tpot"] > 40:
                    slow[w] += 1
            out.append(
                f"| {load:g} | {rep} | {cell(row['stock-0820'])} | {cell(row['stock-ad5b'])} | "
                f"{cell(row['pad'])} |"
            )
    out += [
        "",
        f"Slow runs out of 6: stock 08204560 {slow['stock-0820']}, stock ad5b1408 "
        f"{slow['stock-ad5b']}, padded {slow['pad']}.",
    ]
    return out


def light_load_table(db: Path) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select experiment_id, cell_key, workload, load_mode, load_value, repetition, summary"
        " from bench_runs where experiment_id in (?, ?) and status = 'completed'"
        " order by experiment_id, cell_key, workload, load_value, repetition",
        (LIGHT_STOCK.replace("-", ""), SMOKE.replace("-", "")),
    ).fetchall()
    con.close()
    out = [
        "## FP8 (FP8 KV) at light load: stock and padded alike",
        "",
        "Smoke runs at small batches. e3694f45 = aws-smoke-trtllm-fp8 (stock 55b35655, "
        "2026-11-03); d35e631b = aws-smoke-trtllm-why. ITL p50 and the mean running batch.",
        "",
        "| Experiment | Cell | Workload | Load | Rep | ITL p50 ms | Running mean |",
        "|---|---|---|---|---|---|---|",
    ]
    for exp, cell, wl, mode, load, rep, summ in rows:
        s = json.loads(summ)
        run = (s.get("server") or {}).get("running_mean")
        out.append(
            f"| {exp[:8]} | {cell} | {wl} ({mode.replace('_', ' ')}) | {load:g} | {rep} | "
            f"{s['itl_ms']['p50']:.1f} | {f(run, 1)} |"
        )
    return out


def tables(rows: list[dict], db: Path) -> str:
    out = ['# TensorRT-LLM "why" tables (generated by build.py; do not edit)', ""]
    for part in (
        windows_table(rows),
        matched_table(rows),
        gemm_kernel_table(rows),
        eager_table(rows),
        fp8_runs_table(db),
        fp8_windows_table(rows),
        smoke_bs4_table(rows),
        light_load_table(db),
    ):
        out += [*part, ""]
    return "\n".join(out)


if __name__ == "__main__":
    results, db = Path(sys.argv[1]), Path(sys.argv[2])
    build(results)
    (HERE / "tables.md").write_text(tables(load(), db))
    print(f"wrote {HERE / 'tables.md'}")
