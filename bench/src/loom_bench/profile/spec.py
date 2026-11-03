"""Experiment `profile:` section and the per-job capture instructions.

A profile window is a short extra load job, run after a cell's measured load points:
the same workload at one of its load values, held long enough to reach steady state
(`settle_s`), then the engine's torch profiler is switched on for at most
`max_iterations` engine steps (and at most `capture_s` of wall time) and off again.
The trace is then collected from the host, stored under
`results/<experiment>/profiles/<run id>/` and broken down by kernel category.

Profile windows are never load runs: they are not recorded in `bench_runs`, have no
summary in the leaderboard and take no part in goodput, cost at SLO or the search.
The profiler slows the steps it records, so their latencies would be wrong; the
window's own client latencies are kept next to the trace only to show that cost.

This module depends on pydantic alone, so `experiment` and `jobs` can import it.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

# vLLM's profile_prefix rule: 1-128 ASCII letters, digits, '.', '_' or '-', starting
# with a letter or digit.
PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_TRACE_DIR = "/var/lib/loom/profiles"
# Time beyond the window to stop the profiler (vLLM flushes the trace in /stop_profile),
# archive the trace, upload it and fetch it back.
COLLECT_ALLOWANCE_S = 90.0
# Providers that collect a window's trace (`collect_profile`). On aws_ec2 the engine runs
# in Docker, which bind-mounts the trace directory at the same path on the host
# (engines.docker_run_argv), where the collection script finds it.
PROFILE_PROVIDERS = frozenset({"aws_ec2", "runpod", "mock"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProfileWindow(_Strict):
    workload: str  # a workload label of the experiment
    # Load values to profile at; each must be one of the workload's `load.values`.
    # Default: every one of them.
    load_values: Annotated[list[PositiveFloat], Field(min_length=1)] | None = None


class ProfileSpec(_Strict):
    windows: Annotated[list[ProfileWindow], Field(min_length=1)]
    settle_s: PositiveFloat = 45.0  # load before the profiler starts (steady state)
    capture_s: PositiveFloat = 20.0  # longest the profiler stays on
    max_iterations: PositiveInt = 128  # engine steps recorded (vLLM max_iterations)
    delay_iterations: NonNegativeInt = 0  # steps skipped after start before recording
    tail_s: PositiveFloat = 5.0  # load kept on after the profiler stops
    drain_timeout_s: float = Field(default=15.0, ge=0)  # in-flight requests are not measured
    trace_dir: str = DEFAULT_TRACE_DIR  # where the engine writes traces on the host
    max_trace_mb: PositiveInt = 256  # a bigger archive fails the window, not the run
    stop_timeout_s: PositiveFloat = 300.0  # /stop_profile flushes the trace: let it finish

    @property
    def job_s(self) -> float:
        """The window's load duration."""
        return self.settle_s + self.capture_s + self.tail_s

    def window_s(self, run_overhead_s: float) -> float:
        """Planned seconds for one window: load, drain, staging, collection."""
        return self.job_s + self.drain_timeout_s + run_overhead_s + COLLECT_ALLOWANCE_S


class ProfileCapture(_Strict):
    """What `bench job run` does to the engine's profiler during one load job."""

    start_after_s: PositiveFloat  # after the job's t0
    stop_after_s: PositiveFloat
    prefix: str = Field(pattern=PREFIX_RE.pattern)
    max_iterations: PositiveInt
    delay_iterations: NonNegativeInt = 0
    stop_timeout_s: PositiveFloat = 300.0


def check_profile(
    profile: ProfileSpec,
    *,
    provider_kind: str,
    workloads: dict[str, Any],
    variant_args: dict[str, dict[str, Any]],
) -> None:
    """Raise ValueError when the section cannot run as written.

    `workloads`: label -> LoadSpec; `variant_args`: variant name -> its `engine.args`
    (an empty dict when the variant sets none)."""
    if provider_kind not in PROFILE_PROVIDERS:
        raise ValueError(
            f"profile windows run on the {', '.join(sorted(PROFILE_PROVIDERS))} providers "
            f"only, not {provider_kind}"
        )
    for w in profile.windows:
        load = workloads.get(w.workload)
        if load is None:
            raise ValueError(f"profile window workload {w.workload!r} is not a workload label")
        if load.values is None:
            raise ValueError(
                f"profile window {w.workload!r}: its workload must use load.values, not a search"
            )
        unknown = sorted(set(w.load_values or ()) - set(load.values))
        if unknown:
            raise ValueError(
                f"profile window {w.workload!r}: load values {unknown} are not in its load.values"
            )
    if provider_kind == "mock":
        return
    for name, args in variant_args.items():
        cfg = args.get("profiler_config")
        if not isinstance(cfg, dict):
            raise ValueError(
                f"variant {name}: a profiled vLLM engine needs engine.args.profiler_config "
                f"(profiler: torch, torch_profiler_dir: {profile.trace_dir})"
            )
        if cfg.get("profiler") != "torch":
            raise ValueError(f"variant {name}: profiler_config.profiler must be torch")
        if cfg.get("torch_profiler_dir") != profile.trace_dir:
            raise ValueError(
                f"variant {name}: profiler_config.torch_profiler_dir must be the profile "
                f"section's trace_dir {profile.trace_dir!r}"
            )
        if cfg.get("torch_profiler_with_stack", True):
            raise ValueError(
                f"variant {name}: set profiler_config.torch_profiler_with_stack: false "
                "(Python stacks multiply the trace's size and the profiler's overhead)"
            )
