"""Engine-side metrics from Prometheus `/metrics` scrapes taken during a run.

Metric names differ between engines and drift between engine versions, so each
quantity has an ordered list of candidate names; the first name present in any
scrape wins. Fix names here, in `ENGINE_METRICS`, and nowhere else.

Missing metrics yield None (listed in `ServerMetrics.missing`); nothing here raises
on unexpected exposition content.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel

_NAME_RE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_RE = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*(,|\})')
_TYPE_RE = re.compile(r"#\s*TYPE\s+(\S+)\s+(\S+)")
_UNESCAPE = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}

Labels = dict[str, str]


def _unescape(value: str) -> str:
    return re.sub(r"\\[\\\"n]", lambda m: _UNESCAPE[m.group(0)], value)


def _parse_labels(line: str, pos: int) -> tuple[Labels, int] | None:
    """Parse `{a="x",b="y"}` starting at `line[pos] == "{"`; return labels and end index."""
    labels: Labels = {}
    pos += 1
    if line[pos:].lstrip().startswith("}"):
        return labels, line.index("}", pos) + 1
    while True:
        m = _LABEL_RE.match(line, pos)
        if m is None:
            return None
        labels[m.group(1)] = _unescape(m.group(2))
        pos = m.end()
        if m.group(3) == "}":
            return labels, pos
        if line[pos:].lstrip().startswith("}"):
            return labels, line.index("}", pos) + 1


@dataclass(frozen=True, slots=True)
class Histogram:
    buckets: list[tuple[float, float]]  # (upper bound `le`, cumulative count), ascending
    sum: float
    count: float


@dataclass(slots=True)
class Scrape:
    """One parsed exposition: samples by sample name, and declared family types."""

    samples: dict[str, list[tuple[Labels, float]]] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)

    def has(self, name: str) -> bool:
        return name in self.samples

    def value(self, name: str, reduce: Literal["sum", "mean"] = "sum") -> float | None:
        """Combine all label sets of a sample name (e.g. several engines or models)."""
        values = [v for _, v in self.samples.get(name, []) if not math.isnan(v)]
        if not values:
            return None
        total = math.fsum(values)
        return total if reduce == "sum" else total / len(values)

    def histogram(self, base: str) -> Histogram | None:
        """Histogram `base` summed over all label sets other than `le`."""
        total, count = self.value(f"{base}_sum"), self.value(f"{base}_count")
        if total is None or count is None:
            return None
        by_le: dict[float, float] = {}
        for labels, v in self.samples.get(f"{base}_bucket", []):
            try:
                le = float(labels["le"])
            except (KeyError, ValueError):
                continue
            if not math.isnan(v):
                by_le[le] = by_le.get(le, 0.0) + v
        return Histogram(buckets=sorted(by_le.items()), sum=total, count=count)


def parse_prometheus(text: str) -> Scrape:
    """Parse Prometheus text exposition (0.0.4; OpenMetrics `# EOF` tolerated).

    Malformed lines are skipped.
    """
    scrape = Scrape()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = _TYPE_RE.match(line)
            if m:
                scrape.types[m.group(1)] = m.group(2)
            continue
        m = _NAME_RE.match(line)
        if m is None:
            continue
        name, pos = m.group(0), m.end()
        labels: Labels = {}
        if pos < len(line) and line[pos] == "{":
            parsed = _parse_labels(line, pos)
            if parsed is None:
                continue
            labels, pos = parsed
        rest = line[pos:].split()
        if not rest:
            continue
        try:
            value = float(rest[0])
        except ValueError:
            continue
        scrape.samples.setdefault(name, []).append((labels, value))
    return scrape


@dataclass(frozen=True)
class EngineMetricNames:
    """Candidate sample names per quantity, in preference order."""

    running: tuple[str, ...]  # gauge, requests being decoded/prefilled
    waiting: tuple[str, ...]  # gauge, requests queued in the engine
    kv_usage: tuple[str, ...]  # gauge, fraction 0..1 of KV cache in use
    preemptions: tuple[str, ...]  # counter
    prefix_queries: tuple[str, ...]  # counter, prompt tokens looked up in the prefix cache
    prefix_hits: tuple[str, ...]  # counter, of which hit
    prefix_hit_rate: tuple[str, ...]  # gauge fallback when the counters are absent
    queue_time: tuple[str, ...]  # histogram base name, seconds


VLLM_METRICS = EngineMetricNames(
    running=("vllm:num_requests_running",),
    waiting=("vllm:num_requests_waiting",),
    kv_usage=("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    preemptions=("vllm:num_preemptions_total",),
    prefix_queries=("vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"),
    prefix_hits=("vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"),
    prefix_hit_rate=("vllm:gpu_prefix_cache_hit_rate",),
    queue_time=("vllm:request_queue_time_seconds",),
)

SGLANG_METRICS = EngineMetricNames(
    running=("sglang:num_running_reqs",),
    waiting=("sglang:num_queue_reqs",),
    kv_usage=("sglang:token_usage",),
    preemptions=(),
    prefix_queries=(),
    prefix_hits=(),
    prefix_hit_rate=("sglang:cache_hit_rate",),
    queue_time=("sglang:queue_time_seconds",),
)

ENGINE_METRICS: dict[str, EngineMetricNames] = {
    "vllm": VLLM_METRICS,
    "sglang": SGLANG_METRICS,
    "mock": VLLM_METRICS,
}


class ServerMetrics(BaseModel):
    """Engine-reported metrics over the scrape window.

    Gauge means are plain averages over scrapes (assumes a regular scrape interval).
    Counter quantities are increases between the first and last scrape, with
    counter resets handled. Running/waiting are summed across label sets; KV usage
    and hit-rate gauges are averaged across label sets.
    """

    engine: str
    n_scrapes: int
    duration_s: float | None
    running_mean: float | None
    running_max: float | None
    waiting_mean: float | None
    waiting_max: float | None
    kv_usage_mean: float | None
    kv_usage_max: float | None
    preemptions: float | None
    prefix_cache_hit_rate: float | None
    queue_time_mean_ms: float | None
    missing: list[str]


def _pick(scrapes: Sequence[Scrape], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if any(s.has(name) for s in scrapes):
            return name
    return None


def _increase(values: Sequence[float]) -> float:
    """Counter increase across consecutive readings; a drop is a reset to zero."""
    total = 0.0
    for prev, cur in pairwise(values):
        total += cur - prev if cur >= prev else cur
    return total


def _gauge(
    scrapes: Sequence[Scrape], candidates: tuple[str, ...], reduce: Literal["sum", "mean"]
) -> tuple[float, float] | None:
    name = _pick(scrapes, candidates)
    if name is None:
        return None
    values = [v for s in scrapes if (v := s.value(name, reduce)) is not None]
    if not values:
        return None
    return math.fsum(values) / len(values), max(values)


def _counter_increase(scrapes: Sequence[Scrape], candidates: tuple[str, ...]) -> float | None:
    name = _pick(scrapes, candidates)
    if name is None:
        return None
    values = [v for s in scrapes if (v := s.value(name)) is not None]
    if len(values) < 2:
        return None
    return _increase(values)


def summarize_scrapes(engine: str, scrapes: Sequence[tuple[float, str]]) -> ServerMetrics:
    """Summarise `(t, exposition text)` scrapes of one engine taken during a run."""
    if engine not in ENGINE_METRICS:
        raise ValueError(f"unknown engine {engine!r}; known: {sorted(ENGINE_METRICS)}")
    names = ENGINE_METRICS[engine]
    timed = sorted(scrapes, key=lambda s: s[0])
    parsed = [parse_prometheus(text) for _, text in timed]
    missing: list[str] = []

    running = _gauge(parsed, names.running, "sum")
    waiting = _gauge(parsed, names.waiting, "sum")
    kv = _gauge(parsed, names.kv_usage, "mean")
    preemptions = _counter_increase(parsed, names.preemptions)

    hit_rate: float | None = None
    queries = _counter_increase(parsed, names.prefix_queries)
    hits = _counter_increase(parsed, names.prefix_hits)
    if queries is not None and hits is not None:
        hit_rate = hits / queries if queries > 0 else None
    else:
        gauge = _gauge(parsed, names.prefix_hit_rate, "mean")
        hit_rate = gauge[0] if gauge else None

    queue_ms: float | None = None
    queue_count = _pick(parsed, tuple(f"{base}_count" for base in names.queue_time))
    if queue_count is not None:
        base = queue_count.removesuffix("_count")
        hists = [h for s in parsed if (h := s.histogram(base)) is not None]
        if len(hists) >= 2:
            d_count = _increase([h.count for h in hists])
            d_sum = _increase([h.sum for h in hists])
            queue_ms = d_sum / d_count * 1000 if d_count > 0 else None

    for label, value in (
        ("running", running),
        ("waiting", waiting),
        ("kv_usage", kv),
        ("preemptions", preemptions),
        ("prefix_cache_hit_rate", hit_rate),
        ("queue_time", queue_ms),
    ):
        if value is None:
            missing.append(label)

    return ServerMetrics(
        engine=engine,
        n_scrapes=len(timed),
        duration_s=timed[-1][0] - timed[0][0] if timed else None,
        running_mean=running[0] if running else None,
        running_max=running[1] if running else None,
        waiting_mean=waiting[0] if waiting else None,
        waiting_max=waiting[1] if waiting else None,
        kv_usage_mean=kv[0] if kv else None,
        kv_usage_max=kv[1] if kv else None,
        preemptions=preemptions,
        prefix_cache_hit_rate=hit_rate,
        queue_time_mean_ms=queue_ms,
        missing=missing,
    )
