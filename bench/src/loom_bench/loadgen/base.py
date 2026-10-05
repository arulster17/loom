"""Load-generator plug-in slot.

The native driver is registered here; wrappers for engine-shipped tools (vLLM
`bench serve`, SGLang `bench_serving`, GuideLLM) add entries to
`LOAD_GENERATORS` and must also emit `RequestRecord`s via `RunResult`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from loom_bench.client.openai_stream import PreparedRequest
from loom_bench.records import LoadMode, RequestRecord


@dataclass(slots=True)
class RunResult:
    records: list[RequestRecord]
    mode: LoadMode
    load_value: float  # open loop: offered rate (req/s); closed loop: concurrency
    # Window (seconds from t0) over which throughput is computed; warmup excluded.
    t_measure_start_s: float
    t_measure_end_s: float
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


class LoadGenerator(Protocol):
    name: str

    async def run(
        self,
        base_url: str,
        requests: Sequence[PreparedRequest],
        plan: LoadPlan,
        *,
        request_timeout_s: float,
        model: str | None = None,
        api_key: str | None = None,
        keep_output: bool = False,
    ) -> RunResult: ...


def _native() -> LoadGenerator:
    from loom_bench.loadgen.native import NativeLoadGenerator

    return NativeLoadGenerator()


# Factories import lazily so a tool's optional dependencies load only when selected.
LOAD_GENERATORS: dict[str, Callable[[], LoadGenerator]] = {"native": _native}


def get_load_generator(name: str) -> LoadGenerator:
    try:
        factory = LOAD_GENERATORS[name]
    except KeyError:
        known = ", ".join(sorted(LOAD_GENERATORS))
        raise KeyError(f"unknown load generator {name!r} (known: {known})") from None
    return factory()
