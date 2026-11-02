"""Theoretical floors on decode-step time and prefill time.

A decode step produces one token for every running sequence. It has to:

1. read every weight once (all sequences share that read),
2. read each sequence's cached keys and values (grows with batch x context),
3. do 2 FLOPs per weight per sequence, plus attention over each context,
4. with tensor parallelism (TP), all-reduce the activations twice per layer.

Reading (1)+(2) at the GPU's peak memory bandwidth, or doing (3) at its peak tensor
throughput, whichever takes longer, plus (4) over the link, is the least time the
step can take: the roofline. A measured step is longer by whatever else the GPU
and the server spend time on (kernel launch gaps, scheduling, sampling, kernels
that don't stream at full bandwidth); `achieved = floor / measured`.

Prefill (time to first token) processes a prompt's tokens in one or a few big
batches, so it is bound by FLOPs, not bytes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from loom_bench.profile.arch import ModelBytes, model_bytes
from loom_bench.profile.hardware import GpuSpec, LinkSpec, gpu, link_for
from loom_bench.registry import ModelSpec

ACTIVATION_BYTES = 2  # BF16 activations cross the TP link in every config here


@dataclass(frozen=True)
class ServingSetup:
    """One served replica: what it reads, on what, split how."""

    label: str
    model: ModelBytes
    gpu: GpuSpec
    tp: int = 1
    link: LinkSpec | None = None

    @classmethod
    def from_spec(
        cls, spec: ModelSpec, *, env: dict[str, str] | None = None, label: str = ""
    ) -> ServingSetup:
        tp = spec.parallelism.tp * spec.parallelism.pp
        g = gpu(spec.hardware.gpu)
        return cls(
            label=label or spec.id,
            model=model_bytes(spec),
            gpu=g,
            tp=tp,
            link=link_for(g.name, tp, env or {}),
        )

    def describe(self) -> dict[str, object]:
        m = self.model
        return {
            "label": self.label,
            "model": m.arch.repo,
            "weights": m.weights,
            "kv_cache": m.kv,
            "gpu": self.gpu.name,
            "tp": self.tp,
            "weight_gb_per_gpu_per_step": round(m.step_weight_bytes / self.tp / 1e9, 3),
            "kv_kib_per_token_per_gpu": round(m.kv_bytes_per_token / self.tp / 1024, 2),
            "hbm_gb_s": self.gpu.hbm_gb_s,
            "link": self.link.name if self.link else None,
        }


@dataclass(frozen=True)
class StepFloor:
    """Floor on one decode step, by component (ms, per GPU: TP shards run in parallel)."""

    batch: float
    kv_tokens: float  # cached tokens summed over the batch
    weights_ms: float
    kv_ms: float
    compute_ms: float
    comm_ms: float

    @property
    def memory_ms(self) -> float:
        return self.weights_ms + self.kv_ms

    @property
    def bound(self) -> str:
        return "memory" if self.memory_ms >= self.compute_ms else "compute"

    @property
    def total_ms(self) -> float:
        return max(self.memory_ms, self.compute_ms) + self.comm_ms

    def as_dict(self) -> dict[str, float | str]:
        return {
            **asdict(self),
            "memory_ms": self.memory_ms,
            "total_ms": self.total_ms,
            "bound": self.bound,
        }


def decode_step(setup: ServingSetup, batch: float, kv_tokens: float) -> StepFloor:
    """Floor for a decode step of `batch` sequences holding `kv_tokens` cached tokens
    in total (batch x mean context)."""
    m, g, tp = setup.model, setup.gpu, setup.tp
    bw = g.hbm_gb_s * 1e9
    weights_ms = m.step_weight_bytes / tp / bw * 1000
    kv_ms = kv_tokens * m.kv_bytes_per_token / tp / bw * 1000
    flops = (batch * m.flops_per_token + m.attention_flops(kv_tokens)) / tp
    compute_ms = flops / (g.tflops(m.weights) * 1e12) * 1000
    comm_ms = 0.0
    if setup.link is not None and batch > 0:
        msg = batch * m.arch.hidden * ACTIVATION_BYTES
        comm_ms = 2 * m.arch.layers * setup.link.allreduce_ms(msg)
    return StepFloor(batch, kv_tokens, weights_ms, kv_ms, compute_ms, comm_ms)


@dataclass(frozen=True)
class PrefillFloor:
    tokens: int
    compute_ms: float  # linear layers at the weight precision's peak, attention at BF16's
    weights_ms: float  # one read of the weights
    comm_ms: float

    @property
    def total_ms(self) -> float:
        return max(self.compute_ms, self.weights_ms) + self.comm_ms

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "total_ms": self.total_ms}


def prefill(setup: ServingSetup, tokens: int) -> PrefillFloor:
    """Floor for prefilling one `tokens`-long prompt with nothing cached: the least
    possible time to first token, before any queueing."""
    m, g, tp = setup.model, setup.gpu, setup.tp
    linear = tokens * m.flops_per_token / tp / (g.tflops(m.weights) * 1e12)
    # causal attention: token i attends to i earlier tokens, so ~tokens^2 / 2 pairs
    attn = m.attention_flops(tokens * tokens / 2) / tp / (g.bf16_tflops * 1e12)
    weights = m.step_weight_bytes / tp / (g.hbm_gb_s * 1e9)
    comm = 0.0
    if setup.link is not None:
        comm = (
            2
            * m.arch.layers
            * setup.link.allreduce_ms(tokens * m.arch.hidden * ACTIVATION_BYTES)
            / 1000
        )
    return PrefillFloor(tokens, (linear + attn) * 1000, weights * 1000, comm * 1000)


def max_batch_at(setup: ServingSetup, step_ms: float, context: float) -> float:
    """Largest batch whose floor stays within `step_ms` when every sequence holds
    `context` tokens (memory-bound regime, no TP link)."""
    lo, hi = 0.0, 1.0
    while decode_step(setup, hi, hi * context).total_ms <= step_ms:
        hi *= 2
        if hi > 1e6:
            return float("inf")
    if decode_step(setup, lo, 0).total_ms > step_ms:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if decode_step(setup, mid, mid * context).total_ms <= step_ms:
            lo = mid
        else:
            hi = mid
    return lo
