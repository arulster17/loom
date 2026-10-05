"""Summarise one run (one load point x one repetition) from its RequestRecords.

Population and denominators:

- Warmup records are dropped before anything else. `n_total` counts the rest.
- Error rate = (error + timeout) / (ok + error + timeout). Aborted requests were
  cancelled by the client at the run deadline, so they are in neither term.
- Latency distributions (TTFT, TPOT, ITL, E2E, client queue delay, per-request
  output tok/s) use successful requests only. ITL pools every token gap of every
  successful request. Client queue delay exists for open loop only.
- Throughput = totals over successful requests / `window_s`, the measurement
  window. Pass the window the load generator measured (excluding warmup); when it
  is None the span from the first send to the last finish of measured requests is
  used. Tokens come from the server's usage block; successful requests without
  usage are counted in `requests_missing_usage` and contribute no tokens.
- Output tok/s per GPU = output tok/s / `gpus` (GPUs of the replica under test).
- Per-request output tok/s = completion tokens / E2E latency of that request.
- Cached-prompt fraction = Σ cached prompt tokens / Σ prompt tokens over successful
  requests that report cached tokens.
- SLO attainment = good requests / (ok + error + timeout), where "good" is
  `Slo.request_meets`. Request goodput = good requests (and their output tokens)
  / `window_s`, like vLLM's `--goodput`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel

from loom_bench.metrics.gpu import GpuMetrics
from loom_bench.metrics.prometheus import ServerMetrics
from loom_bench.records import RequestRecord, RequestStatus
from loom_bench.slo import PERCENTILES, Slo


class Distribution(BaseModel):
    """Summary of a sample; all statistics are None when n == 0."""

    n: int
    mean: float | None = None
    min: float | None = None
    p50: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    max: float | None = None

    @classmethod
    def of(cls, values: Iterable[float]) -> Distribution:
        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            return cls(n=0)
        pcts = np.percentile(arr, list(PERCENTILES.values()), method="linear")
        return cls(
            n=int(arr.size),
            mean=float(arr.mean()),
            min=float(arr.min()),
            max=float(arr.max()),
            **{k: float(v) for k, v in zip(PERCENTILES, pcts, strict=True)},
        )


class Throughput(BaseModel):
    request_rate: float  # successful requests / s
    output_tok_s: float
    input_tok_s: float
    total_tok_s: float
    output_tok_s_per_gpu: float


class RequestGoodput(BaseModel):
    slo_attainment: float | None
    good_requests: int
    request_rate: float
    output_tok_s: float


class RunSummary(BaseModel):
    n_total: int
    n_warmup_excluded: int
    counts: dict[str, int]
    error_rate: float | None
    window_s: float
    gpus: int
    ttft_ms: Distribution
    tpot_ms: Distribution
    itl_ms: Distribution
    e2e_ms: Distribution
    queue_delay_ms: Distribution
    per_request_output_tok_s: Distribution
    throughput: Throughput
    cached_prompt_fraction: float | None
    requests_missing_usage: int
    slo: Slo | None
    goodput: RequestGoodput | None
    server: ServerMetrics | None = None
    gpu: GpuMetrics | None = None


def _ms(values: Iterable[float | None]) -> list[float]:
    return [v * 1000 for v in values if v is not None]


def _span(records: list[RequestRecord]) -> float:
    ends = [r.finished_at_s for r in records if r.finished_at_s is not None]
    if not records or not ends:
        raise ValueError("window_s is None and no measured request finished")
    return max(ends) - min(r.sent_at_s for r in records)


def summarize_run(
    records: Iterable[RequestRecord],
    *,
    window_s: float | None,
    gpus: int,
    slo: Slo | None = None,
    server: ServerMetrics | None = None,
    gpu: GpuMetrics | None = None,
) -> RunSummary:
    if gpus < 1:
        raise ValueError("gpus must be >= 1")
    all_records = list(records)
    measured = [r for r in all_records if not r.warmup]
    window = _span(measured) if window_s is None else window_s
    if window <= 0:
        raise ValueError(f"measurement window must be positive, got {window}")

    counts = {s.value: 0 for s in RequestStatus}
    for r in measured:
        counts[r.status.value] += 1
    attempted = counts["ok"] + counts["error"] + counts["timeout"]
    failed = counts["error"] + counts["timeout"]

    ok = [r for r in measured if r.ok]
    out_tokens = sum(r.completion_tokens or 0 for r in ok)
    in_tokens = sum(r.prompt_tokens or 0 for r in ok)
    missing_usage = sum(1 for r in ok if r.completion_tokens is None or r.prompt_tokens is None)

    cached = [r for r in ok if r.cached_prompt_tokens is not None and r.prompt_tokens]
    prompt_with_cache = sum(r.prompt_tokens or 0 for r in cached)
    cached_fraction = (
        sum(r.cached_prompt_tokens or 0 for r in cached) / prompt_with_cache
        if prompt_with_cache
        else None
    )

    per_request_tps = [
        r.completion_tokens / e2e
        for r in ok
        if r.completion_tokens is not None and (e2e := r.e2e_s) is not None and e2e > 0
    ]

    goodput: RequestGoodput | None = None
    if slo is not None:
        good = [r for r in ok if slo.request_meets(r)]
        goodput = RequestGoodput(
            slo_attainment=len(good) / attempted if attempted else None,
            good_requests=len(good),
            request_rate=len(good) / window,
            output_tok_s=sum(r.completion_tokens or 0 for r in good) / window,
        )

    return RunSummary(
        n_total=len(measured),
        n_warmup_excluded=len(all_records) - len(measured),
        counts=counts,
        error_rate=failed / attempted if attempted else None,
        window_s=window,
        gpus=gpus,
        ttft_ms=Distribution.of(_ms(r.ttft_s for r in ok)),
        tpot_ms=Distribution.of(_ms(r.tpot_s for r in ok)),
        itl_ms=Distribution.of(_ms(g for r in ok for g in r.itl_s)),
        e2e_ms=Distribution.of(_ms(r.e2e_s for r in ok)),
        queue_delay_ms=Distribution.of(_ms(r.queue_delay_s for r in ok)),
        per_request_output_tok_s=Distribution.of(per_request_tps),
        throughput=Throughput(
            request_rate=len(ok) / window,
            output_tok_s=out_tokens / window,
            input_tok_s=in_tokens / window,
            total_tok_s=(in_tokens + out_tokens) / window,
            output_tok_s_per_gpu=out_tokens / window / gpus,
        ),
        cached_prompt_fraction=cached_fraction,
        requests_missing_usage=missing_usage,
        slo=slo,
        goodput=goodput,
        server=server,
        gpu=gpu,
    )


def flatten_metrics(summary: RunSummary) -> dict[str, float]:
    """Numeric leaves keyed by dotted path ("ttft_ms.p95", "throughput.output_tok_s",
    "server.kv_usage_mean", "gpu.overall.utilization_mean_pct", ...).

    Run configuration (`slo`, `gpus`), lists, strings and None values are left out.
    """
    out: dict[str, float] = {}

    def walk(prefix: str, value: object) -> None:
        if isinstance(value, BaseModel):
            value = value.model_dump()
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif (
            isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
        ):
            out[prefix] = float(value)

    walk("", summary.model_dump(exclude={"slo", "gpus"}))
    return out
