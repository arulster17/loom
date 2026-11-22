"""Model shapes and the bytes and FLOPs they imply.

`ARCHITECTURES` copies the fields the roofline needs from each base model's
`config.json` at the revision pinned in `config/models.yaml`; a test checks them
against the Hugging Face cache when it holds that snapshot. A pre-quantized
checkpoint (RedHatAI's FP8-dynamic) has its base model's shapes, so it maps to the
base entry (`BASE_REPO`); its precision comes from the registry row.

What a decode step reads, per the checkpoint layout:

- every linear layer's weights (attention q/k/v/o and the MLP's gate/up/down), at the
  serving precision: 2 bytes for BF16, 1 for FP8 (plus per-channel scales);
- the LM head (vocab x hidden), which llm-compressor leaves in BF16 in FP8 checkpoints;
- not the input embedding table: a step gathers one row per sequence, not the table.

So the bytes a step reads are a bit less than the checkpoint's size: Qwen3-8B BF16 is
16.38 GB on disk and reads 15.14 GB per step (the 1.24 GB embedding table stays put).
"""

from __future__ import annotations

from dataclasses import dataclass

from loom_bench.registry import ModelSpec


@dataclass(frozen=True)
class Architecture:
    repo: str
    revision: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    tie_word_embeddings: bool

    @property
    def attn_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def layer_linear_params(self) -> int:
        """q, k, v, o projections and the gated MLP's three matrices of one layer."""
        qkv_o = self.hidden * (self.attn_dim + 2 * self.kv_dim) + self.attn_dim * self.hidden
        mlp = 3 * self.hidden * self.intermediate
        return qkv_o + mlp

    @property
    def linear_params(self) -> int:
        return self.layers * self.layer_linear_params

    @property
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    @property
    def lm_head_params(self) -> int:
        return 0 if self.tie_word_embeddings else self.vocab * self.hidden

    @property
    def norm_params(self) -> int:
        # input and post-attention RMSNorm per layer, the final norm; Qwen3 adds q/k norms
        return (2 * self.layers + 1) * self.hidden

    @property
    def total_params(self) -> int:
        return self.linear_params + self.embed_params + self.lm_head_params + self.norm_params

    def kv_elements_per_token(self) -> int:
        """K and V, every layer, every KV head: what one cached token holds."""
        return 2 * self.layers * self.kv_dim


ARCHITECTURES: dict[str, Architecture] = {
    "Qwen/Qwen3-8B": Architecture(
        repo="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        layers=36,
        hidden=4096,
        heads=32,
        kv_heads=8,
        head_dim=128,
        intermediate=12288,
        vocab=151936,
        tie_word_embeddings=False,
    ),
    "meta-llama/Llama-3.3-70B-Instruct": Architecture(
        repo="meta-llama/Llama-3.3-70B-Instruct",
        revision="6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        layers=80,
        hidden=8192,
        heads=64,
        kv_heads=8,
        head_dim=128,
        intermediate=28672,
        vocab=128256,
        tie_word_embeddings=False,
    ),
}

# Pre-quantized checkpoints -> the base model whose shapes they keep.
BASE_REPO: dict[str, str] = {
    "RedHatAI/Qwen3-8B-FP8-dynamic": "Qwen/Qwen3-8B",
    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic": "meta-llama/Llama-3.3-70B-Instruct",
}

PRECISION_BYTES = {"bf16": 2.0, "fp8": 1.0}


def architecture(repo: str) -> Architecture:
    key = BASE_REPO.get(repo, repo)
    try:
        return ARCHITECTURES[key]
    except KeyError:
        raise ValueError(f"no architecture for {repo!r}: add its config.json fields") from None


def weight_precision(quantization: str) -> str:
    """The precision linear-layer weights are read at, from the registry field."""
    if quantization == "none":
        return "bf16"
    if quantization in {"fp8", "w8a8"}:
        return "fp8"
    raise ValueError(f"no roofline byte count for quantization {quantization!r}")


def kv_precision(kv_cache_dtype: str, weights: str) -> str:
    """The KV cache's element type. `auto` follows the model's activations (BF16 for
    every checkpoint here, FP8-dynamic ones included)."""
    if kv_cache_dtype == "auto":
        return "bf16"
    if kv_cache_dtype.startswith("fp8"):
        return "fp8"
    if kv_cache_dtype in {"bfloat16", "float16", "half"}:
        return "bf16"
    raise ValueError(f"unknown kv_cache_dtype {kv_cache_dtype!r}")


@dataclass(frozen=True)
class ModelBytes:
    """What one replica of a served model reads and computes, before TP splits it."""

    arch: Architecture
    weights: str  # bf16 | fp8 (linear layers)
    kv: str  # bf16 | fp8

    @property
    def step_weight_bytes(self) -> float:
        """Bytes of weights one decode step reads (linear layers + LM head)."""
        return (
            self.arch.linear_params * PRECISION_BYTES[self.weights]
            + self.arch.lm_head_params * PRECISION_BYTES["bf16"]
        )

    @property
    def checkpoint_bytes(self) -> float:
        """Approximate size on disk (no per-channel scales), to check against the registry."""
        return (
            self.step_weight_bytes
            + (self.arch.embed_params + self.arch.norm_params) * PRECISION_BYTES["bf16"]
        )

    @property
    def kv_bytes_per_token(self) -> float:
        return self.arch.kv_elements_per_token() * PRECISION_BYTES[self.kv]

    @property
    def flops_per_token(self) -> float:
        """Matmul FLOPs per token through the linear layers and LM head (2 per MAC)."""
        return 2.0 * (self.arch.linear_params + self.arch.lm_head_params)

    def attention_flops(self, context: float) -> float:
        """QK^T and AV for one new token attending to `context` cached tokens."""
        return 4.0 * self.arch.layers * self.arch.attn_dim * context


def model_bytes(spec: ModelSpec) -> ModelBytes:
    weights = weight_precision(spec.quantization)
    return ModelBytes(
        arch=architecture(spec.hf.repo),
        weights=weights,
        kv=kv_precision(spec.kv_cache_dtype, weights),
    )
