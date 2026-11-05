"""Break an engine trace's GPU time into what the GPU was doing.

Input: PyTorch profiler traces (Chrome trace JSON, `.json` or `.json.gz`, as vLLM's
`--profiler-config` or TensorRT-LLM's runtime `/start_profile` writes them, one file per
GPU worker), or a `.tar.gz` holding them (what a Loom profile window uploads).

Every GPU kernel is put in one category by its name (`CATEGORIES`, first match
wins). The trace's time on each GPU is then split exclusively: where kernels of
different categories overlap (e.g. NCCL on its own stream under a GEMM), the time
goes to the higher-priority one, so only *exposed* communication counts as
communication. Time with no kernel running is `idle`: the GPU waiting on the CPU
(scheduling, sampling bookkeeping, Python, launch latency). The categories and
`idle` sum to the window.

Per-step figures divide by the number of engine steps in the window: from the
trace's step annotations when it has them, else the `steps` the capture asked for
(`max_iterations`), else they are left out. vLLM v0.30 names each step
`execute_context_<n>(<tokens>)_generation_<n>(<tokens>)` (prefill requests and tokens,
decode requests): those annotations also split the window step by step, so pure-decode
steps are broken down per batch size (`step_shapes`), each step measured from its start
to the next step's start (the LM head and sampling run after the annotated forward, and
the gap to the next step is the GPU waiting on the CPU). vLLM v0.30 ignores the
per-session `max_iterations` (smoke a183a857 recorded every step of the capture), so
the requested count is only a last resort.

TensorRT-LLM 1.4.0rc2 (PyTorch backend) labels each executor iteration once with a bare
`step[DECODE bs=<generation requests>]` or `step[EXTEND bs=<context requests>
toks=<context tokens>]` (`PyExecutor._build_step_scope_label`). With the overlap
scheduler the sampling of an iteration runs in a second scope with a suffix
(`step[...] sample`); only the bare labels are step boundaries (`TRTLLM_STEP`, anchored).
An EXTEND label does not say how many generation requests ride along, so those steps
form the `mixed` group, as vLLM's steps with prefill do. When a trace has TensorRT-LLM
step labels, only they count; vLLM's patterns are not looked at.
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
            r"|merge_state|rotary|rope"
            # TensorRT-LLM's decode attention: XQA and masked multi-head attention kernels
            r"|xqa|mmha|multihead|kernel_mha",
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
    "quant": "Quantize / dequantize (incl. fused norm+quant)",
    "sampling": "Sampling",
    "memcpy": "Memory copies",
    "other": "Other kernels (norms, activations, elementwise)",
    "idle": "GPU idle: CPU / scheduler gaps",
}
STEP_ANNOTATION = re.compile(
    r"execute_model|model_runner|_model_forward|gpu_model_runner|execute_context_\d", re.I
)
# vLLM v0.30's step name: prefill requests (tokens), decode requests (tokens).
STEP_SHAPE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
# TensorRT-LLM 1.4.0rc2's bare per-iteration label; `step[...] sample` does not match.
TRTLLM_STEP = re.compile(r"^step\[(DECODE|EXTEND) bs=(\d+)(?: toks=(\d+))?\]$")


def step_shape(name: str) -> tuple[int, int, int] | None:
    """(prefill requests, prefill tokens, decode requests) from a step label of either
    engine; None if it is not one. TensorRT-LLM's EXTEND label has no decode count (0)."""
    m = TRTLLM_STEP.match(name)
    if m:
        if m[1] == "DECODE":
            return 0, 0, int(m[2])
        return int(m[2]), int(m[3] or 0), 0
    v = STEP_SHAPE.search(name)
    if v:
        return int(v[1]), int(v[2]), int(v[3])
    return None


# torch.compile (Inductor) fuses FP8 activation quantization into the RMSNorm / SiLU
# kernel before each scaled GEMM: `triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_
# div_..._reciprocal_...` computes the per-token scale (abs, max, reciprocal, clamp) and
# casts. Inductor's pointwise and reduction kernels (`triton_poi_`, `triton_red_`,
# `triton_per_`) never run a matrix multiply: only its templates (`triton_tem_`) do.
INDUCTOR_NON_GEMM = re.compile(r"^triton_(poi|red|per)_")
INDUCTOR_QUANT = re.compile(r"scaled_mm|quant|fp8|e4m3|abs_clamp|clamp.*reciprocal", re.I)


# Normalization kernels are "other" (fused with a quantize: "quant") whatever library
# they come from: TensorRT-LLM 1.4.0rc2 runs FlashInfer's FusedAddRMSNorm, whose name the
# attention pattern ("flashinfer") would otherwise claim (71e2c32c: 0.2-0.8 ms a step).
NORM = re.compile(r"rms_?norm|layer_?norm", re.I)
QUANT = re.compile(r"quant|fp8|e4m3", re.I)


def categorize(name: str) -> str:
    inductor = bool(INDUCTOR_NON_GEMM.match(name))
    if inductor and INDUCTOR_QUANT.search(name):
        return "quant"
    if NORM.search(name):
        return "quant" if QUANT.search(name) else "other"
    for cat, pat in CATEGORIES:
        if inductor and cat == "gemm":
            continue
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
    step_shapes: list[dict[str, Any]] = field(default_factory=list)

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
            "step_shapes": self.step_shapes,
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


def _segments(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The window [first start, last end] cut into consecutive pieces, each credited to
    the highest-priority category running in it, or `idle`."""
    if not intervals:
        return []
    start = min(s for s, _, _ in intervals)
    bounds: list[tuple[float, int, str]] = []
    for s, e, c in intervals:
        if e > s:
            bounds.append((s, 1, c))
            bounds.append((e, -1, c))
    bounds.sort(key=lambda b: (b[0], b[1]))
    active: dict[str, int] = defaultdict(int)
    out: list[tuple[float, float, str]] = []
    prev = start
    for t, delta, c in bounds:
        if t > prev:
            live = next((p for p in PRIORITY if active.get(p, 0) > 0), None)
            out.append((prev, t, live or "idle"))
            prev = t
        active[c] += delta
    return out


def _exclusive(intervals: list[tuple[float, float, str]]) -> tuple[dict[str, float], float, float]:
    """Exclusive time per category over [first start, last end], plus idle."""
    if not intervals:
        return {}, 0.0, 0.0
    out: dict[str, float] = defaultdict(float)
    for s, e, c in _segments(intervals):
        out[c] += e - s
    start = min(s for s, _, _ in intervals)
    end = max(e for _, e, _ in intervals)
    return dict(out), start, end


def _by_period(
    segments: list[tuple[float, float, str]], bounds: list[float]
) -> list[dict[str, float]]:
    """Exclusive time per category inside each period [bounds[i], bounds[i+1])."""
    out: list[dict[str, float]] = [defaultdict(float) for _ in range(len(bounds) - 1)]
    j = 0
    for s, e, c in segments:
        while j < len(bounds) - 1 and bounds[j + 1] <= s:
            j += 1
        k = j
        while k < len(bounds) - 1 and bounds[k] < e:
            lo, hi = max(s, bounds[k]), min(e, bounds[k + 1])
            if hi > lo:
                out[k][c] += hi - lo
            if bounds[k + 1] >= e:
                break
            k += 1
    return [dict(d) for d in out]


def step_shapes(
    segments: list[tuple[float, float, str]], steps: list[tuple[float, str]]
) -> list[dict[str, Any]]:
    """Per-step breakdown from vLLM's step annotations (start time, name): each step runs
    from its start to the next step's start, so it includes the LM head, sampling and the
    GPU waiting for the next step. Pure-decode steps are grouped by batch size; steps
    carrying prefill form one `mixed` group. Milliseconds."""
    parsed = [(t, step_shape(n)) for t, n in sorted(steps)]
    if len(parsed) < 2 or not all(m for _, m in parsed):
        return []
    bounds = [t for t, _ in parsed]
    periods = _by_period(segments, bounds)
    groups: dict[tuple[str, int], list[tuple[float, dict[str, float], int]]] = defaultdict(list)
    for (t, m), (t_next, _), cats in zip(parsed, parsed[1:], periods, strict=False):
        assert m is not None
        prefill_reqs, prefill_tokens, decode_reqs = m
        key = ("decode", decode_reqs) if prefill_reqs == 0 else ("mixed", 0)
        groups[key].append(((t_next - t) / 1000, cats, prefill_tokens))
    out: list[dict[str, Any]] = []
    for (kind, batch), rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0] != "decode", kv[0][1])
    ):
        n = len(rows)
        periods_ms = sorted(r[0] for r in rows)
        cat_ms = {k: sum(r[1].get(k, 0.0) for r in rows) / n / 1000 for k in (*ORDER, "idle")}
        row: dict[str, Any] = {
            "kind": kind,
            "batch": batch if kind == "decode" else None,
            "steps": n,
            "step_ms_p50": periods_ms[n // 2],
            "step_ms_mean": sum(periods_ms) / n,
            "category_ms": cat_ms,
        }
        if kind == "mixed":
            row["prefill_tokens_mean"] = sum(r[2] for r in rows) / n
        out.append(row)
    return out


def analyze_events(
    events: list[dict[str, Any]], *, source: str = "", steps: int | None = None
) -> list[RankBreakdown]:
    """One breakdown per GPU device in `events` (times in the trace's microseconds)."""
    by_dev: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    names: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    # per labelling scheme ("trtllm", "vllm"): GPU step annotations by device, CPU count
    step_names: dict[str, dict[int, list[tuple[float, str]]]] = {
        "trtllm": defaultdict(list),
        "vllm": defaultdict(list),
    }
    cpu_steps = {"trtllm": 0, "vllm": 0}
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
        elif cat in {"gpu_user_annotation", "user_annotation"}:
            scheme = (
                "trtllm"
                if TRTLLM_STEP.match(name)
                else "vllm"
                if STEP_ANNOTATION.search(name)
                else None
            )
            if scheme is None:
                continue
            if cat == "gpu_user_annotation":
                step_names[scheme][_device(ev)].append((ts, name))
            else:
                cpu_steps[scheme] += 1
    if not by_dev:
        raise ValueError("the trace has no GPU kernel events")
    trt = bool(cpu_steps["trtllm"] or any(step_names["trtllm"].values()))
    gpu_steps = step_names["trtllm" if trt else "vllm"]
    cpu_count = cpu_steps["trtllm" if trt else "vllm"]
    out = []
    for dev in sorted(by_dev):
        segments = _segments(by_dev[dev])
        excl: dict[str, float] = defaultdict(float)
        for s, e, c in segments:
            excl[c] += e - s
        start = min(s for s, _, _ in by_dev[dev])
        end = max(e for _, e, _ in by_dev[dev])
        ms = {k: excl.get(k, 0.0) / 1000 for k in (*ORDER, "idle")}
        n_steps, how = None, None
        if gpu_steps.get(dev):
            n_steps, how = len(gpu_steps[dev]), "gpu annotations"
        elif cpu_count:
            n_steps, how = cpu_count, "cpu annotations"
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
                step_shapes=step_shapes(segments, gpu_steps.get(dev, [])),
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
        if r.step_shapes:
            cols = (*ORDER, "idle")
            buf.write("  steps by shape (ms per step, start to next start):\n")
            buf.write(
                "    shape        steps  p50   mean  " + " ".join(f"{c:>9}" for c in cols) + "\n"
            )
            for g in r.step_shapes:
                shape = f"decode x{g['batch']}" if g["kind"] == "decode" else "mixed"
                head = f"    {shape:<12} {g['steps']:5d} {g['step_ms_p50']:5.1f}"
                buf.write(
                    f"{head} {g['step_ms_mean']:6.1f}  "
                    + " ".join(f"{g['category_ms'][c]:9.2f}" for c in cols)
                    + "\n"
                )
    return buf.getvalue()
