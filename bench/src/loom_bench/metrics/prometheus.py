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
    # Speculative decoding counters. A "draft" is one request's proposal in one engine
    # step (only steps that proposed at least one token count); draft tokens are the
    # tokens proposed, accepted tokens those the target model kept. The per-position
    # counter carries a `position` label 0..k-1: drafts whose token at that position was
    # accepted (so position i counts only when 0..i-1 were accepted too).
    spec_drafts: tuple[str, ...] = ()
    spec_draft_tokens: tuple[str, ...] = ()
    spec_accepted_tokens: tuple[str, ...] = ()
    spec_accepted_per_position: tuple[str, ...] = ()


# Label of the per-position acceptance counter.
SPEC_POSITION_LABEL = "position"

# Names verified against vllm/v1/metrics/loggers.py at v0.30.0 and
# sglang/srt/observability/metrics_collector.py at v0.5.21. The speculative-decoding
# counters are vllm/v1/spec_decode/metrics.py's `SpecDecodingProm` at v0.30.0
# (prometheus_client Counters, so the samples carry `_total`); vLLM only registers them
# when the engine runs with a speculative config.
VLLM_METRICS = EngineMetricNames(
    running=("vllm:num_requests_running",),
    waiting=("vllm:num_requests_waiting",),
    kv_usage=("vllm:kv_cache_usage_perc",),
    preemptions=("vllm:num_preemptions_total",),
    prefix_queries=("vllm:prefix_cache_queries_total",),
    prefix_hits=("vllm:prefix_cache_hits_total",),
    prefix_hit_rate=(),
    queue_time=("vllm:request_queue_time_seconds",),
    spec_drafts=("vllm:spec_decode_num_drafts_total",),
    spec_draft_tokens=("vllm:spec_decode_num_draft_tokens_total",),
    spec_accepted_tokens=("vllm:spec_decode_num_accepted_tokens_total",),
    spec_accepted_per_position=("vllm:spec_decode_num_accepted_tokens_per_pos_total",),
)

# SGLang's speculative-decoding metrics are not mapped (not verified at v0.5.21, and no
# SGLang speculative config has run), so an SGLang run's spec fields stay None.
SGLANG_METRICS = EngineMetricNames(
    running=("sglang:num_running_reqs",),
    waiting=("sglang:num_queue_reqs",),
    kv_usage=("sglang:token_usage",),
    # SGLang "retracts" requests under KV pressure; its preemption equivalent.
    preemptions=("sglang:num_retracted_requests_total",),
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
    # Speculative decoding over the window (None when the engine exposes no spec
    # counters, i.e. it was not speculating). Defaults keep runs stored before these
    # existed valid.
    spec_drafts: float | None = None
    spec_draft_tokens: float | None = None
    spec_accepted_tokens: float | None = None
    # accepted / drafted tokens: the share of proposed tokens the target kept.
    spec_acceptance_rate: float | None = None
    # 1 + accepted / drafts: tokens a request gains per verify step that drafted,
    # counting the target's own (bonus) token, as vLLM's log line defines it.
    spec_mean_accepted_length: float | None = None
    # Position ("0", "1", ...) -> drafts whose token there was accepted / drafts.
    spec_acceptance_by_position: dict[str, float] | None = None


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


def _labelled_increase(
    scrapes: Sequence[Scrape], candidates: tuple[str, ...], label: str
) -> dict[str, float] | None:
    """Counter increase per value of `label` (other labels summed); None when absent."""
    name = _pick(scrapes, candidates)
    if name is None:
        return None
    series: dict[str, list[float]] = {}
    for s in scrapes:
        totals: dict[str, float] = {}
        for labels, v in s.samples.get(name, []):
            if label in labels and not math.isnan(v):
                totals[labels[label]] = totals.get(labels[label], 0.0) + v
        for key, v in totals.items():
            series.setdefault(key, []).append(v)
    out = {k: _increase(vs) for k, vs in series.items() if len(vs) >= 2}
    return out or None


def _position_key(position: str) -> tuple[int, str]:
    return (int(position), position) if position.isdigit() else (1 << 30, position)


@dataclass(frozen=True, slots=True)
class _Spec:
    drafts: float | None
    draft_tokens: float | None
    accepted: float | None
    acceptance_rate: float | None
    mean_accepted_length: float | None
    by_position: dict[str, float] | None

    @property
    def complete(self) -> bool:
        """Every counter was scraped (a window with no draft has no per-position rates)."""
        counters = (self.drafts, self.draft_tokens, self.accepted)
        return None not in counters and (self.by_position is not None or self.drafts == 0)


def _spec(parsed: Sequence[Scrape], names: EngineMetricNames) -> _Spec:
    drafts = _counter_increase(parsed, names.spec_drafts)
    draft_tokens = _counter_increase(parsed, names.spec_draft_tokens)
    accepted = _counter_increase(parsed, names.spec_accepted_tokens)
    per_pos = _labelled_increase(parsed, names.spec_accepted_per_position, SPEC_POSITION_LABEL)
    rate = accepted / draft_tokens if accepted is not None and draft_tokens else None
    length = 1 + accepted / drafts if accepted is not None and drafts else None
    by_pos = (
        {p: per_pos[p] / drafts for p in sorted(per_pos, key=_position_key)}
        if per_pos is not None and drafts
        else None
    )
    return _Spec(drafts, draft_tokens, accepted, rate, length, by_pos)


def summarize_scrapes(
    engine: str, scrapes: Sequence[tuple[float, str]], *, speculative: bool = False
) -> ServerMetrics:
    """Summarise `(t, exposition text)` scrapes of one engine taken during a run.

    With `speculative` (the engine was launched with a speculative config), absent
    speculative-decoding counters are listed in `missing` as "spec_decode"; without it
    they are reported when present and otherwise left None silently."""
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
    spec = _spec(parsed, names)
    if speculative and not spec.complete:
        missing.append("spec_decode")

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
        spec_drafts=spec.drafts,
        spec_draft_tokens=spec.draft_tokens,
        spec_accepted_tokens=spec.accepted,
        spec_acceptance_rate=spec.acceptance_rate,
        spec_mean_accepted_length=spec.mean_accepted_length,
        spec_acceptance_by_position=spec.by_position,
    )
