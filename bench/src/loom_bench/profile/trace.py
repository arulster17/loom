"""Break an engine trace's GPU time into what the GPU was doing.

Input: PyTorch profiler traces (Chrome trace JSON, `.json` or `.json.gz`, as vLLM's
`--profiler-config` writes them, one file per GPU worker), or a `.tar.gz` holding
them (what a Loom profile window uploads).

Every GPU kernel is put in one category by its name (`CATEGORIES`, first match
wins). The trace's time on each GPU is then split exclusively: where kernels of
different categories overlap (e.g. NCCL on its own stream under a GEMM), the time
goes to the higher-priority one, so only *exposed* communication counts as
communication. Time with no kernel running is `idle`: the GPU waiting on the CPU
(scheduling, sampling bookkeeping, Python, launch latency). The categories and
`idle` sum to the window.

Per-step figures divide by the number of engine steps in the window: from the
trace's step annotations when it has them, else the `steps` the capture asked for
(`max_iterations`), else they are left out.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# (category, kernel-name pattern); first match wins. Exclusive-time priority is the
# order of PRIORITY below, not this list.
CATEGORIES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "allreduce",
        re.compile(
            r"nccl|all_?reduce|cross_device_reduce|allgather|all_gather|reduce_scatter", re.I
        ),
    ),
    (
        "attention",
        re.compile(
            r"flash|fmha|attention|attn|paged|reshape_and_cache|flashinfer|BatchDecode|BatchPrefill"
            r"|merge_state|rotary|rope",
            re.I,
        ),
    ),
    (
        "gemm",
        re.compile(
            r"gemm|cutlass|cublas|xmma|matmul|gemv|wgmma|marlin|scaled_mm|_mm_|_mm$|mm_kernel|Kernel2|sm\d\d_|splitk",
            re.I,
        ),
    ),
    ("quant", re.compile(r"quant|dequant|fp8|e4m3|scaled_conv|per_token_group", re.I)),
    (
        "sampling",
        re.compile(
            r"sampl|top_?k|top_?p|argmax|softmax|gumbel|multinomial|penalt|logit|radix|sort", re.I
        ),
    ),
)
ORDER = ("gemm", "attention", "allreduce", "quant", "sampling", "memcpy", "other")
# Highest first: what a busy GPU is credited with when categories overlap.
PRIORITY = ("gemm", "attention", "quant", "sampling", "other", "memcpy", "allreduce")
LABELS = {
    "gemm": "GEMM (linear layers, LM head)",
    "attention": "Attention (incl. RoPE, KV-cache writes)",
    "allreduce": "All-reduce / NCCL (exposed)",
    "quant": "Quantize / dequantize",
    "sampling": "Sampling",
    "memcpy": "Memory copies",
    "other": "Other kernels (norms, activations, elementwise)",
    "idle": "GPU idle: CPU / scheduler gaps",
}
STEP_ANNOTATION = re.compile(r"execute_model|model_runner|_model_forward|gpu_model_runner", re.I)


def categorize(name: str) -> str:
    for cat, pat in CATEGORIES:
        if pat.search(name):
            return cat
    return "other"


@dataclass
class RankBreakdown:
    """One GPU's window, split exclusively by category (milliseconds)."""

    source: str
    device: int
    window_ms: float
    category_ms: dict[str, float]
    kernels: int
    steps: int | None
    step_source: str | None
    top_kernels: list[dict[str, Any]] = field(default_factory=list)

    @property
    def busy_ms(self) -> float:
        return self.window_ms - self.category_ms.get("idle", 0.0)

    def per_step_ms(self) -> dict[str, float] | None:
        if not self.steps:
            return None
        return {k: v / self.steps for k, v in self.category_ms.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "device": self.device,
            "window_ms": self.window_ms,
            "busy_ms": self.busy_ms,
            "kernels": self.kernels,
            "steps": self.steps,
            "step_source": self.step_source,
            "category_ms": self.category_ms,
            "category_share": {k: v / self.window_ms for k, v in self.category_ms.items()}
            if self.window_ms
            else {},
            "per_step_ms": self.per_step_ms(),
            "top_kernels": self.top_kernels,
        }


@dataclass
class TraceBreakdown:
    ranks: list[RankBreakdown]

    def as_dict(self) -> dict[str, Any]:
        return {"ranks": [r.as_dict() for r in self.ranks]}


def _events(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict):
        return list(doc.get("traceEvents") or [])
    if isinstance(doc, list):
        return doc
    raise ValueError("not a Chrome trace (no traceEvents)")


def _device(ev: dict[str, Any]) -> int:
    args = ev.get("args") or {}
    dev = args.get("device", ev.get("pid", 0))
    try:
        return int(dev)
    except (TypeError, ValueError):
        return 0


def _exclusive(intervals: list[tuple[float, float, str]]) -> tuple[dict[str, float], float, float]:
    """Exclusive time per category over [first start, last end], plus idle."""
    if not intervals:
        return {}, 0.0, 0.0
    start = min(s for s, _, _ in intervals)
    end = max(e for _, e, _ in intervals)
    bounds: list[tuple[float, int, str]] = []
    for s, e, c in intervals:
        if e > s:
            bounds.append((s, 1, c))
            bounds.append((e, -1, c))
    bounds.sort(key=lambda b: (b[0], b[1]))
    active: dict[str, int] = defaultdict(int)
    out: dict[str, float] = defaultdict(float)
    prev = start
    for t, delta, c in bounds:
        if t > prev:
            live = next((p for p in PRIORITY if active.get(p, 0) > 0), None)
            out[live or "idle"] += t - prev
            prev = t
        active[c] += delta
    return dict(out), start, end


def analyze_events(
    events: list[dict[str, Any]], *, source: str = "", steps: int | None = None
) -> list[RankBreakdown]:
    """One breakdown per GPU device in `events` (times in the trace's microseconds)."""
    by_dev: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    names: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    step_ranges: dict[int, list[tuple[float, float]]] = defaultdict(list)
    cpu_steps = 0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = str(ev.get("cat", "")).lower()
        ts, dur = float(ev.get("ts", 0.0)), float(ev.get("dur", 0.0))
        name = str(ev.get("name", ""))
        if cat == "kernel":
            dev = _device(ev)
            kind = categorize(name)
            by_dev[dev].append((ts, ts + dur, kind))
            agg = names[dev][name]
            agg[0] += 1
            agg[1] += dur
        elif cat in {"gpu_memcpy", "gpu_memset"}:
            by_dev[_device(ev)].append((ts, ts + dur, "memcpy"))
        elif cat == "gpu_user_annotation" and STEP_ANNOTATION.search(name):
            step_ranges[_device(ev)].append((ts, ts + dur))
        elif cat == "user_annotation" and STEP_ANNOTATION.search(name):
            cpu_steps += 1
    if not by_dev:
        raise ValueError("the trace has no GPU kernel events")
    out = []
    for dev in sorted(by_dev):
        excl, start, end = _exclusive(by_dev[dev])
        ms = {k: excl.get(k, 0.0) / 1000 for k in (*ORDER, "idle")}
        n_steps, how = None, None
        if step_ranges.get(dev):
            n_steps, how = len(step_ranges[dev]), "gpu annotations"
        elif cpu_steps:
            n_steps, how = cpu_steps, "cpu annotations"
        elif steps:
            n_steps, how = steps, "requested max_iterations"
        top = sorted(names[dev].items(), key=lambda kv: -kv[1][1])[:15]
        out.append(
            RankBreakdown(
                source=source,
                device=dev,
                window_ms=(end - start) / 1000,
                category_ms=ms,
                kernels=sum(int(v[0]) for v in names[dev].values()),
                steps=n_steps,
                step_source=how,
                top_kernels=[
                    {
                        "name": n[:160],
                        "category": categorize(n),
                        "count": int(c),
                        "total_ms": t / 1000,
                    }
                    for n, (c, t) in top
                ],
            )
        )
    return out


def _load_json(raw: bytes, name: str) -> Any:
    if name.endswith(".gz") or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _is_trace(name: str) -> bool:
    return name.endswith((".json", ".json.gz")) and "async_llm" not in name


def analyze_path(path: Path, *, steps: int | None = None) -> TraceBreakdown:
    """A trace file, or a tarball of them (each worker's file is one rank)."""
    ranks: list[RankBreakdown] = []
    name = path.name
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path) as tar:
            for m in sorted(tar.getmembers(), key=lambda m: m.name):
                if not m.isfile() or not _is_trace(m.name):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                doc = _load_json(f.read(), m.name)
                ranks += analyze_events(_events(doc), source=m.name, steps=steps)
        if not ranks:
            raise ValueError("no trace files in the archive")
    else:
        ranks = analyze_events(
            _events(_load_json(path.read_bytes(), name)), source=name, steps=steps
        )
    return TraceBreakdown(ranks)


def render_breakdown(result: TraceBreakdown) -> str:
    buf = io.StringIO()
    for r in result.ranks:
        per = r.per_step_ms()
        buf.write(
            f"{r.source} (GPU {r.device}): window {r.window_ms:.1f} ms, {r.kernels} kernels, "
            f"steps: {r.steps if r.steps else 'unknown'}"
            f"{f' ({r.step_source})' if r.step_source else ''}\n"
        )
        for k in (*ORDER, "idle"):
            v = r.category_ms.get(k, 0.0)
            share = v / r.window_ms if r.window_ms else 0.0
            step = f"  {per[k]:7.2f} ms/step" if per else ""
            buf.write(f"  {LABELS[k]:<48} {v:9.1f} ms  {100 * share:5.1f}%{step}\n")
    return buf.getvalue()
