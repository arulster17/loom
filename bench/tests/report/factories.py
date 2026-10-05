"""Realistic store rows for report tests: 3 configs x 4 open-loop loads x 3 repetitions.

Latency is set per load so the SLO verdicts are known in advance:

- vllm-bf16 (on-demand g6e.xlarge): passes 2 and 4 req/s, fails at 6 → goodput 4.
- sglang-bf16 (on-demand g6e.xlarge): passes 2, 4, 6, fails at 8 → goodput 6, so its
  cost at SLO is 2/3 of vllm-bf16's (same instance price, 1.5x the tokens).
- vllm-awq (spot g6e.xlarge, cheapest): goodput 6 but fails the quality gate.

Each request asks for 200 prompt tokens and returns 100 output tokens, so at load L
req/s output throughput is 100·L tok/s (within the per-repetition window jitter).
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any

from loom_bench.metrics.summary import summarize_run
from loom_bench.provenance import (
    ContentKind,
    DatasetInfo,
    EngineInfo,
    GitInfo,
    HardwareInfo,
    HostInfo,
    LoadInfo,
    ModelInfo,
    WorkloadInfo,
    build_provenance,
)
from loom_bench.records import LoadMode, Market, RequestRecord, RequestStatus
from loom_bench.slo import Slo
from loom_bench.store.models import BenchEvalRun, BenchGateDecision, BenchRun

SLO = Slo(ttft_ms={"p95": 600}, tpot_ms={"p95": 60}, max_error_rate=0.01)
LOADS = (2.0, 4.0, 6.0, 8.0)
REPS = (1, 2, 3)
EXPERIMENT = uuid.UUID(int=1)
CREATED = datetime(2026, 10, 1, tzinfo=UTC)
GIT_SHA = "0123456789abcdef0123456789abcdef01234567"
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"
DIGEST = "sha256:" + "8a" * 32
PROMPT_TOKENS = 200
OUTPUT_TOKENS = 100
TPOT_MS = 30.0
WINDOW_S = 30.0

# base TTFT (ms) per load; the run's p95 is about 1.2x the base
TTFT_VLLM = {2.0: 200.0, 4.0: 300.0, 6.0: 700.0, 8.0: 900.0}
TTFT_SGLANG = {2.0: 150.0, 4.0: 200.0, 6.0: 350.0, 8.0: 800.0}


def _id(*parts: Any) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, "/".join(map(str, parts)))


def records(
    load: float, ttft_ms: float, *, tpot_ms: float = TPOT_MS, missing_usage: int = 0
) -> list[RequestRecord]:
    n = int(load * WINDOW_S)
    out = []
    for i in range(n):
        sent = i / load
        ttft = ttft_ms * (0.8 + 0.4 * i / (n - 1)) / 1000
        first = sent + ttft
        usage = i >= missing_usage
        out.append(
            RequestRecord(
                request_id=f"r{i}",
                status=RequestStatus.OK,
                sent_at_s=sent,
                scheduled_at_s=sent,
                first_token_at_s=first,
                finished_at_s=first + tpot_ms / 1000 * (OUTPUT_TOKENS - 1),
                prompt_tokens=PROMPT_TOKENS if usage else None,
                completion_tokens=OUTPUT_TOKENS if usage else None,
            )
        )
    return out


def make_runs(
    cell_key: str,
    *,
    engine: str = "vllm",
    quantization: str = "none",
    ttft: dict[float, float] = TTFT_VLLM,
    market: Market = Market.ON_DEMAND,
    instance_type: str | None = "g6e.xlarge",
    cloud: str | None = "aws",
    region: str | None = "us-east-1",
    hourly_micros: int | None = None,
    reps: tuple[int, ...] = REPS,
    loads: tuple[float, ...] = LOADS,
    experiment_id: uuid.UUID = EXPERIMENT,
    latency_scale: float = 1.0,
    rep_jitter: float = 0.02,
    missing_usage: int = 0,
    engine_args: dict[str, Any] | None = None,
) -> list[BenchRun]:
    args = engine_args if engine_args is not None else {"max_num_seqs": 256}
    config: dict[str, Any] = {
        "cell": cell_key,
        "engine": engine,
        "quantization": quantization,
        "engine_args": args,
    }
    runs = []
    for load in loads:
        for rep in reps:
            factor = 1 + rep_jitter * (rep - 2)
            summary = summarize_run(
                records(
                    load,
                    ttft[load] * factor * latency_scale,
                    tpot_ms=TPOT_MS * factor * latency_scale,
                    missing_usage=missing_usage,
                ),
                window_s=WINDOW_S * (1 + rep_jitter / 2 * (rep - 2)),
                gpus=1,
                slo=SLO,
            )
            prov = build_provenance(
                config,
                created_at=CREATED,
                hourly_micros=hourly_micros,
                git=GitInfo(sha=GIT_SHA, dirty=False, branch="main"),
                loadgen={"name": "native", "version": "0.1.0"},
                engine=EngineInfo(
                    name=engine,
                    version="0.30.0" if engine == "vllm" else "0.5.21",
                    image=f"example/{engine}@{DIGEST}",
                    image_digest=DIGEST,
                    args=args,
                ),
                cuda_version="13.0",
                driver_version="580.65",
                model=ModelInfo(repo="Qwen/Qwen3-8B", revision=QWEN_REV, quantization=quantization),
                hardware=HardwareInfo(gpu_type="L40S", gpu_count=1, instance_type=instance_type),
                cloud=cloud,
                region=region,
                market=market,
                workload=WorkloadInfo(name="chat", content=ContentKind.REALISTIC),
                dataset=DatasetInfo(
                    name="ShareGPT_V3",
                    source="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered",
                    license="apache-2.0",
                ),
                load=LoadInfo(mode=LoadMode.OPEN_LOOP, value=load, seed=7),
                repetition=rep,
                host=HostInfo(python="3.12.7", platform="test"),
            )
            runs.append(
                BenchRun(
                    id=_id(experiment_id, cell_key, load, rep, latency_scale),
                    experiment_id=experiment_id,
                    cell_key=cell_key,
                    config_hash=prov.config_hash,
                    workload="chat",
                    load_mode=LoadMode.OPEN_LOOP.value,
                    load_value=load,
                    repetition=rep,
                    status="completed",
                    provenance=prov.model_dump(mode="json"),
                    summary=summary.model_dump(mode="json"),
                )
            )
    return runs


def eval_row(config_hash: str, task: str, score: float) -> BenchEvalRun:
    return BenchEvalRun(
        id=_id("eval", config_hash, task),
        experiment_id=EXPERIMENT,
        config_hash=config_hash,
        task=task,
        task_version="1",
        n=500,
        score=score,
        ci_low=score - 0.02,
        ci_high=score + 0.02,
        provenance={},
        created_at=CREATED,
    )


def gate_row(baseline: str, candidate: str, decision: str) -> BenchGateDecision:
    return BenchGateDecision(
        id=_id("gate", baseline, candidate),
        experiment_id=EXPERIMENT,
        baseline_config_hash=baseline,
        candidate_config_hash=candidate,
        decision=decision,
        details={},
        created_at=CREATED,
    )


def assert_self_contained(html: str, allowed: tuple[str, ...]) -> None:
    assert html.startswith("<!doctype html>")
    for banned in ("<link", "<script", "<img", "<iframe", "@import", "url(", "src="):
        assert banned not in html, banned
    hrefs = re.findall(r'<a href="([^"]+)">', html)
    for url in re.findall(r"https?://[^\s\"<>]+", html):
        assert url in hrefs, f"{url} is not a link"
        assert url.startswith(allowed), url
