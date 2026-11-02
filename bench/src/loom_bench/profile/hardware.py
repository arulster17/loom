"""Peak hardware figures the roofline divides by.

Every figure is a vendor spec-sheet peak (dense, no sparsity) or a link measured on a
Loom pod; each says which. Spec peaks are ceilings no kernel reaches: the roofline
is a floor on time, and "achieved % of roofline" is how close a run came to it.

Memory bandwidth reuses `loom_bench.plan.GPU_MEMORY_BANDWIDTH_GB_S`, the table the
planner already sizes eval time with, so the two never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from loom_bench.plan import GPU_MEMORY_BANDWIDTH_GB_S


@dataclass(frozen=True)
class GpuSpec:
    name: str
    hbm_gb_s: float  # peak memory bandwidth, GB/s (1e9 bytes)
    bf16_tflops: float  # dense BF16 tensor-core peak, TFLOP/s
    fp8_tflops: float | None  # dense FP8 tensor-core peak; None: no FP8 tensor cores
    source: str

    def tflops(self, precision: str) -> float:
        """Dense tensor peak for GEMMs whose inputs are `precision` (fp8 | bf16)."""
        if precision == "fp8" and self.fp8_tflops is not None:
            return self.fp8_tflops
        return self.bf16_tflops


# Dense tensor-core peaks from the vendor datasheets (the "with sparsity" figures are
# 2x these). L40S: 362.05 BF16 / 733 FP8 (NVIDIA L40S datasheet). H100 SXM5: 989.4 /
# 1978.9. H200 SXM: same compute as H100 SXM, more and faster memory.
GPUS: dict[str, GpuSpec] = {
    "L40S": GpuSpec(
        "L40S", GPU_MEMORY_BANDWIDTH_GB_S["L40S"], 362.05, 733.0, "NVIDIA L40S datasheet"
    ),
    "H100": GpuSpec(
        "H100", GPU_MEMORY_BANDWIDTH_GB_S["H100"], 989.4, 1978.9, "NVIDIA H100 SXM5 datasheet"
    ),
    "H200": GpuSpec(
        "H200", GPU_MEMORY_BANDWIDTH_GB_S["H200"], 989.4, 1978.9, "NVIDIA H200 SXM datasheet"
    ),
}


def gpu(name: str) -> GpuSpec:
    try:
        return GPUS[name]
    except KeyError:
        raise ValueError(
            f"no roofline figures for GPU {name!r} (known: {', '.join(sorted(GPUS))})"
        ) from None


@dataclass(frozen=True)
class LinkSpec:
    """The path a tensor-parallel all-reduce takes, as a straight line through two
    points: `latency_us` for a tiny message plus bytes over `bus_gb_s`.

    For a ring all-reduce over n GPUs each GPU moves 2(n-1)/n of the message; the
    measured figures below already include that (they are message size over
    wall time), so `allreduce_ms` applies them to the message size directly.
    """

    name: str
    latency_us: float
    bus_gb_s: float  # message bytes per second once latency is paid
    basis: str  # "measured: ..." or "assumed: ..."

    def allreduce_ms(self, message_bytes: float) -> float:
        return self.latency_us / 1000 + message_bytes / (self.bus_gb_s * 1e9) * 1000


def _through(
    name: str, small: tuple[float, float], large: tuple[float, float], basis: str
) -> LinkSpec:
    """A link fitted through (bytes, seconds) at a small and a large message."""
    (b1, t1), (b2, t2) = small, large
    bus = (b2 - b1) / (t2 - t1)
    latency = t1 - b1 / bus
    return LinkSpec(name, latency * 1e6, bus / 1e9, basis)


LINKS: dict[str, LinkSpec] = {
    # 4x L40S across two CPU sockets with NCCL_P2P_DISABLE=1 (the only setting that did
    # not hang): the 70B tuning and diagnostic pods of 2026-10-08 measured a 16 KB
    # all-reduce (one decode token of 8192 BF16 activations) at 27.6 us and a 32 MB one
    # (a 2048-token prefill) at 2.76 ms (docs/benchmark-lab.md, "70B tuning").
    "pcie-host-p2p-off": _through(
        "pcie-host-p2p-off",
        (16_384, 27.6e-6),
        (2048 * 8192 * 2, 2.76e-3),
        "measured: 70B tuning pods 2026-10-08 (4x L40S, NCCL_P2P_DISABLE=1)",
    ),
    # 2x H100 SXM over NVLink 4 (450 GB/s each way). Not measured in Loom: the latency
    # is an assumption for a small one-shot all-reduce (vLLM's custom all-reduce), to be
    # replaced by the profiled all-reduce time once a 2x H100 trace exists.
    "nvlink4": LinkSpec(
        "nvlink4", 10.0, 450.0, "assumed: 10 us per small all-reduce, 450 GB/s NVLink 4"
    ),
}


def link_for(gpu_name: str, tp: int, env: dict[str, str]) -> LinkSpec | None:
    """The all-reduce path a TP group uses, from the GPU and the launch's NCCL env."""
    if tp <= 1:
        return None
    if env.get("NCCL_P2P_DISABLE") == "1" or gpu_name in {"L40S", "L40"}:
        return LINKS["pcie-host-p2p-off"]
    if gpu_name in {"H100", "H200"}:
        return LINKS["nvlink4"]
    raise ValueError(f"no all-reduce link figures for {tp}x {gpu_name}")
