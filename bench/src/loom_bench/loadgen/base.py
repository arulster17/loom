"""Load-generator contract.

`prepare_load` resolves a `LoadJob` once into a `LoadContext`: the parsed
workload and arrival spec, the load plan, and the prepared requests. Every
generator registered in `LOAD_GENERATORS` takes that context in `run` and
returns a `RunResult` of `RequestRecord`s, so metrics, storage and reports
don't depend on which generator produced the load.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol

from loom_bench.client.openai_stream import PreparedRequest
from loom_bench.jobs import LoadJob, Timeline, TokenizerSpec
from loom_bench.loadgen.arrivals import ArrivalSpec, parse_arrivals
from loom_bench.records import LoadMode, RequestRecord
from loom_bench.tokenize import HFTokenizer, SimpleTokenizer, Tokenizer
from loom_bench.workloads import WorkloadProfile, build_requests, parse_profile

# Duration-bound closed-loop runs need requests prepared up front: this many per
# worker per second of run is 40x headroom for a 1k-token answer at 100 tok/s.
# A run that still runs out reports `exhausted` in its meta.
CLOSED_LOOP_REQUESTS_PER_SLOT_S = 4


@dataclass(slots=True)
class RunResult:
    records: list[RequestRecord]
    mode: LoadMode
    load_value: float  # open loop: offered rate (req/s); closed loop: concurrency
    # Window (seconds from t0) over which throughput is computed; warmup excluded.
    t_measure_start_s: float
    t_measure_end_s: float
    timeline: Timeline = "measured"  # see `jobs.Timeline`
    # Open loop: requests that found all `max_inflight` slots busy at their arrival time.
    client_saturated_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OpenLoopPlan:
    arrivals: Sequence[float]  # sorted offsets in seconds, from `loadgen.arrivals`
    duration_s: float
    warmup_s: float
    max_inflight: int
    drain_timeout_s: float
    load_value: float | None = None  # nominal rate label; defaults to the offered rate


@dataclass(frozen=True, slots=True)
class ClosedLoopPlan:
    concurrency: int
    num_requests: int | None = None  # measured requests, after warmup
    duration_s: float | None = None
    warmup_requests: int = 0


LoadPlan = OpenLoopPlan | ClosedLoopPlan


@dataclass(frozen=True, slots=True)
class LoadContext:
    """One `LoadJob`, resolved once; the whole input of `LoadGenerator.run`."""

    job: LoadJob
    profile: WorkloadProfile
    arrival: ArrivalSpec | None  # open loop only
    plan: LoadPlan
    requests: Sequence[PreparedRequest]  # send order, warmup first; `job.extra_body` merged
    tokenizer: Tokenizer


class LoadGenerator(Protocol):
    name: str

    async def run(self, ctx: LoadContext) -> RunResult: ...


def _make_tokenizer(spec: TokenizerSpec) -> Tokenizer:
    if spec.kind == "simple":
        return SimpleTokenizer()
    if spec.repo is None or spec.revision is None:
        raise ValueError("hf tokenizer needs repo and a pinned revision")
    return HFTokenizer(spec.repo, spec.revision, token=os.environ.get("HF_TOKEN"))


def _plan(job: LoadJob, arrival: ArrivalSpec | None) -> tuple[LoadPlan, int]:
    """Load plan and how many requests it needs."""
    if job.mode is LoadMode.OPEN_LOOP:
        if arrival is None or job.duration_s is None:
            raise ValueError("open-loop jobs need arrival and duration_s")
        offsets = arrival.schedule(job.duration_s, job.seed)
        plan = OpenLoopPlan(
            arrivals=offsets,
            duration_s=job.duration_s,
            warmup_s=job.warmup_s,
            max_inflight=job.max_inflight,
            drain_timeout_s=job.drain_timeout_s,
            load_value=job.load_value,
        )
        return plan, len(offsets)
    concurrency = int(job.load_value)
    if job.num_requests is not None:
        n = job.warmup_requests + job.num_requests
    elif job.duration_s is not None:
        n = job.warmup_requests + math.ceil(
            concurrency * job.duration_s * CLOSED_LOOP_REQUESTS_PER_SLOT_S
        )
    else:
        raise ValueError("closed-loop jobs need num_requests or duration_s")
    plan = ClosedLoopPlan(
        concurrency=concurrency,
        num_requests=job.num_requests,
        duration_s=job.duration_s if job.num_requests is None else None,
        warmup_requests=job.warmup_requests,
    )
    return plan, n


def prepare_load(job: LoadJob) -> LoadContext:
    """Parse the job's workload and arrivals, plan the load and build its requests."""
    tokenizer = _make_tokenizer(job.tokenizer)
    profile = parse_profile(job.workload)
    arrival = parse_arrivals(job.arrival) if job.arrival is not None else None
    plan, n = _plan(job, arrival)
    requests = build_requests(profile, tokenizer, n, rng_seed=job.seed)
    if job.extra_body:
        for req in requests:
            req.payload = {**req.payload, **job.extra_body}
    return LoadContext(
        job=job,
        profile=profile,
        arrival=arrival,
        plan=plan,
        requests=requests,
        tokenizer=tokenizer,
    )


def _native() -> LoadGenerator:
    from loom_bench.loadgen.native import NativeLoadGenerator

    return NativeLoadGenerator()


# Factories import lazily so a tool's optional dependencies load only when selected.
LOAD_GENERATORS: dict[str, Callable[[], LoadGenerator]] = {
    "native": _native,
    "vllm_bench": lambda: import_module("loom_bench.loadgen.vllm_bench").generator(),
    "sglang_bench": lambda: import_module("loom_bench.loadgen.sglang_bench").generator(),
}


def get_load_generator(name: str) -> LoadGenerator:
    try:
        factory = LOAD_GENERATORS[name]
    except KeyError:
        known = ", ".join(sorted(LOAD_GENERATORS))
        raise KeyError(f"unknown load generator {name!r} (known: {known})") from None
    return factory()
