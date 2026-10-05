"""Serializable work units passed from the runner to wherever they execute.

A `LoadJob` runs in-process for the mock/local providers and, for cloud
providers, on the GPU host itself (`bench job run job.json`) so client-observed
latency never includes a WAN round trip. An `EvalJob` runs a pinned quality
suite the same way (`bench quality job`): the engine on a cloud host listens on
its loopback only. Everything here must round-trip through JSON.
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from loom_bench.quality.divergence import DivergenceResult, ReferenceLogprobs
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks.base import ItemResult
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
    base_url: str  # with the API prefix, as reachable from where the job executes
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


class EvalJob(BaseModel):
    """A pinned eval suite (or a subset of its tasks) against one endpoint, plus this
    config's half of the logprob divergence: capture the reference (the baseline) or
    score against a captured one (a candidate)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    suite: Suite  # resolved, so the executing host needs no repo checkout
    tasks: list[str] | None = None  # subset of the suite's tasks; None runs all of them
    base_url: str  # with the API prefix, as reachable from where the job executes
    served_model: str
    extra_body: dict[str, Any] = Field(default_factory=dict)  # e.g. chat_template_kwargs
    allow_code_exec: bool = False  # code_exec tasks run model-written programs
    seed: int = 0
    concurrency: PositiveInt = 16  # in-flight requests for native tasks and divergence
    request_timeout_s: PositiveFloat = 300.0
    divergence: Literal["capture", "score"] | None = None
    reference: ReferenceLogprobs | None = None  # what "score" compares against

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        names = {t.name for t in self.suite.tasks}
        unknown = sorted(set(self.tasks or ()) - names)
        if unknown:
            raise ValueError(f"tasks {unknown} are not in suite {self.suite.suite}")
        if self.divergence is not None and self.suite.divergence is None:
            raise ValueError(f"suite {self.suite.suite} has no divergence section")
        if (self.divergence == "score") != (self.reference is not None):
            raise ValueError("a reference is needed by, and only by, divergence 'score'")
        return self


class EvalTaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    version: str  # what actually ran (harness task versions, data file version)
    items: list[ItemResult]  # per-item score and content hash, paired by item id in the gate
    provenance: dict[str, Any] = Field(default_factory=dict)
    seconds: float


class EvalJobResult(BaseModel):
    """Per-item scores, sanity rates and divergence; outputs themselves stay where the job
    ran (they are never persisted or logged)."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    suite: str
    model: str
    tasks: dict[str, EvalTaskResult]
    sanity: SanityResult
    reference: ReferenceLogprobs | None = None  # after divergence "capture"
    divergence: DivergenceResult | None = None  # after divergence "score"
    started_at: str  # ISO-8601 UTC wall clock
    finished_at: str
