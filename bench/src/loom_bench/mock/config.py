"""Mock backend configuration.

Defaults roughly match an 8B model on one 48 GB GPU, so unscaled runs give
plausible absolute numbers; benchmarks of the Lab itself only rely on shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

Probability = Annotated[float, Field(ge=0, le=1)]


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[str] = Field(default_factory=lambda: ["mock-model"], min_length=1)
    max_model_len: PositiveInt = 32768

    # Scheduler / KV cache (vLLM V1 semantics: the token budget covers decode tokens too).
    max_num_seqs: PositiveInt = 256
    max_batched_tokens: PositiveInt = 8192
    kv_capacity_tokens: PositiveInt = 196_608
    block_size: PositiveInt = 16
    prefix_caching: bool = True

    # Step cost model, in simulated milliseconds.
    step_base_ms: NonNegativeFloat = 6.0
    decode_ms_per_seq: NonNegativeFloat = 0.15
    prefill_ms_per_token: NonNegativeFloat = 0.06
    # Real seconds per simulated second; applies to every simulated wait (steps, startup).
    time_scale: PositiveFloat = 1.0

    seed: int = 0
    # Free-text answers have an exponentially distributed natural length with this mean.
    mean_output_tokens: PositiveInt = 200

    # Quality knobs: share of prompts answered wrongly / with broken JSON, and logit noise.
    degrade: Probability = 0.0
    logprob_noise: NonNegativeFloat = 0.0
    # Logit noise drawn afresh for every request, like a batch-variant engine whose
    # numerics depend on what shares the batch: scoring the same text twice differs a
    # little, which gives the divergence noise floor something to measure.
    logprob_jitter: NonNegativeFloat = 0.0
    # False reports every completions `text_offset` as -1, as SGLang does.
    text_offsets: bool = True

    # Faults.
    error_rate: Probability = 0.0
    abort_rate: Probability = 0.0
    startup_delay_s: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def _one_sequence_fits(self) -> Self:
        if self.num_kv_blocks < -(-self.max_model_len // self.block_size):
            raise ValueError("kv_capacity_tokens must hold at least one max_model_len sequence")
        return self

    @property
    def num_kv_blocks(self) -> int:
        return self.kv_capacity_tokens // self.block_size

    @classmethod
    def from_yaml(cls, path: str | Path) -> MockConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
