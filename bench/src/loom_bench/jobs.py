"""Serializable work units passed from the runner to wherever load is generated.

A `LoadJob` runs in-process for the mock/local providers and, for cloud
providers, on the GPU host itself (`bench job run job.json`) so client-observed
latency never includes a WAN round trip. Everything here must round-trip
through JSON.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from loom_bench.records import LoadMode, RequestRecord


class TokenizerSpec(BaseModel):
    """`simple` = loom_bench.tokenize.SimpleTokenizer; `hf` = HF tokenizer at a pinned revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(pattern="^(simple|hf)$")
    repo: str | None = None
    revision: str | None = None


class LoadJob(BaseModel):
    """One load point x one repetition against one endpoint."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    base_url: str  # as reachable from where the job executes (e.g. http://127.0.0.1:8000)
    metrics_url: str | None = None  # Prometheus endpoint to scrape during the run
    engine: str  # vllm | sglang | mock — selects the metric-name map
    served_model: str  # value for the request `model` field
    loadgen: str = "native"  # key in loom_bench.loadgen registry
    workload: dict[str, Any]  # resolved WorkloadProfile (model_dump(mode="json"))
    tokenizer: TokenizerSpec
    mode: LoadMode
    load_value: float  # requests/s (open loop) or concurrency (closed loop)
    arrival: dict[str, Any] | None = None  # resolved ArrivalSpec, open loop only
    duration_s: float | None = None
    num_requests: int | None = None
    warmup_s: float = 0.0
    warmup_requests: int = 0
    request_timeout_s: float = 600.0
    drain_timeout_s: float = 60.0
    max_inflight: int = 4096
    seed: int = 0
    scrape_interval_s: float = 1.0
    sample_gpu: bool = False  # run nvidia-smi alongside (only meaningful on a GPU host)
    keep_output: bool = False  # keep output text for sanity checks (synthetic prompts only)
    extra_body: dict[str, Any] = Field(default_factory=dict)  # e.g. chat_template_kwargs


class LoadJobResult(BaseModel):
    """Raw observations from one LoadJob; analysis happens on the runner side."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    mode: LoadMode
    load_value: float
    t_measure_start_s: float
    t_measure_end_s: float
    client_saturated_count: int = 0
    records: list[dict[str, Any]]  # RequestRecord.to_row() rows
    scrapes: list[tuple[float, str]] = Field(default_factory=list)  # (t, prometheus text)
    nvidia_smi_csv: str | None = None
    started_at: str  # ISO-8601 UTC wall clock
    finished_at: str
    meta: dict[str, Any] = Field(default_factory=dict)

    def request_records(self) -> list[RequestRecord]:
        return [RequestRecord.from_row(r) for r in self.records]
