"""Mock backend configuration.

Defaults roughly match an 8B model on one 48 GB GPU, so unscaled runs give
plausible absolute numbers; benchmarks of the Lab itself only rely on shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    models: list[str] = Field(default_factory=lambda: ["mock-model"], min_length=1)
    max_model_len: int = Field(32768, ge=1)

    # Scheduler / KV cache (vLLM V1 semantics: the token budget covers decode tokens too).
    max_num_seqs: int = Field(256, ge=1)
    max_batched_tokens: int = Field(8192, ge=1)
    kv_capacity_tokens: int = Field(196_608, ge=1)
    block_size: int = Field(16, ge=1)
    prefix_caching: bool = True

    # Step cost model, in simulated milliseconds.
    step_base_ms: float = Field(6.0, ge=0)
    decode_ms_per_seq: float = Field(0.15, ge=0)
    prefill_ms_per_token: float = Field(0.06, ge=0)
    # Real seconds per simulated second; applies to every simulated wait (steps, startup).
    time_scale: float = Field(1.0, gt=0)

    seed: int = 0
    # Mean of the exponential natural output length (tokens) when the request has no EOS rule.
    mean_output_tokens: int = Field(200, ge=1)

    # Quality knobs: probability a prompt gets a wrong/broken answer, and logprob perturbation.
    degrade: float = Field(0.0, ge=0, le=1)
    logprob_noise: float = Field(0.0, ge=0)

    # Faults.
    error_rate: float = Field(0.0, ge=0, le=1)
    abort_rate: float = Field(0.0, ge=0, le=1)
    startup_delay_s: float = Field(0.0, ge=0)

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
