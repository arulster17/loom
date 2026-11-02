"""The mock backend's stand-in for vLLM's torch profiler.

Between `/start_profile` and `/stop_profile` the simulated engine reports each step it
runs; on stop this writes a Chrome trace shaped like a torch profiler trace (one GPU
kernel event per simulated kernel, a CPU `execute_model` annotation per step) to the
profile directory, named as vLLM names its files. Each step's real duration is split
by fixed shares (`SHARES`), after a CPU gap, so the analysis has a known answer.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Share of each simulated step: the CPU gap before its kernels, then its kernels.
GAP_SHARE = 0.10
SHARES: tuple[tuple[str, str, float], ...] = (
    # (kernel name, category it must land in, share of the step)
    ("dynamic_per_token_scaled_fp8_quant_kernel", "quant", 0.05),
    ("cutlass_scaled_mm_sm89_fp8_gemm", "gemm", 0.50),
    ("flash_attn_decode_kernel", "attention", 0.20),
    ("rms_norm_kernel", "other", 0.10),
    ("topk_topp_sampler_kernel", "sampling", 0.05),
)


@dataclass
class StepRecorder:
    """One profiling session of the mock engine."""

    prefix: str
    max_iterations: int = 0  # 0: no limit, as vLLM
    delay_iterations: int = 0
    seen: int = 0
    steps: list[tuple[float, float, int, int]] = field(default_factory=list)

    def step(
        self, start_s: float, duration_s: float, decode_seqs: int, prefill_tokens: int
    ) -> None:
        self.seen += 1
        if self.seen <= self.delay_iterations:
            return
        if self.max_iterations and len(self.steps) >= self.max_iterations:
            return
        self.steps.append((start_s, duration_s, decode_seqs, prefill_tokens))

    def events(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for i, (start, dur, seqs, prefill) in enumerate(self.steps):
            t = start * 1e6
            d = dur * 1e6
            out.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "execute_model",
                    "ts": t,
                    "dur": d,
                    "pid": 1,
                    "tid": 1,
                    "args": {"step": i, "decode_seqs": seqs, "prefill_tokens": prefill},
                }
            )
            k = t + GAP_SHARE * d
            for name, _, share in SHARES:
                kd = share * d
                out.append(
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": name,
                        "ts": k,
                        "dur": kd,
                        "pid": 0,
                        "tid": 7,
                        "args": {"device": 0, "stream": 7},
                    }
                )
                k += kd
        return out

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.prefix}_rank0.{time.time_ns()}.pt.trace.json.gz"
        path.write_bytes(gzip.compress(json.dumps({"traceEvents": self.events()}).encode()))
        return path
