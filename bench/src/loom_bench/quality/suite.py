"""Pinned eval suites: `bench/evals/<model>.yaml`, validated at load time.

A suite fixes everything that decides which items are scored and how: task
list and parameters, sample counts, seeds, chat-template kwargs, gate
thresholds and divergence / sanity limits. Baseline and candidate must run the
same suite for the gate to compare them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from loom_bench.quality.gate import GatePolicy, MinSamples, TaskPolicy, Threshold
from loom_bench.quality.sanity import SanityConfig, SanityLimits
from loom_bench.quality.tasks import build_task
from loom_bench.registry import REPO_ROOT, read_yaml

EVALS_DIR = REPO_ROOT / "bench" / "evals"

TaskName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_.-]*$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SuiteTask(_Strict):
    name: TaskName
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    threshold: Threshold | None = None
    min_samples: MinSamples | None = None


class GateSpec(_Strict):
    threshold: Threshold = 0.01
    min_samples: MinSamples = 300
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    n_boot: Annotated[int, Field(ge=100)] = 10_000
    inconclusive_blocks: bool = True


class DivergenceSpec(_Strict):
    prompts: Annotated[int, Field(ge=2)] | None = None  # first N pinned prompts; None = all
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    max_new_tokens: Annotated[int, Field(ge=1)] = 64
    max_kl: Annotated[float, Field(ge=0.0)] = 0.05
    min_top1: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95


class Suite(_Strict):
    suite: TaskName
    model: str  # registry id the suite is pinned for
    seed: int = 0
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    gate: GateSpec = Field(default_factory=GateSpec)
    tasks: Annotated[list[SuiteTask], Field(min_length=1)]
    divergence: DivergenceSpec | None = None
    sanity_limits: SanityLimits = Field(default_factory=SanityLimits)
    sanity_checks: SanityConfig = Field(default_factory=SanityConfig)

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [t.name for t in self.tasks]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate task names: {dupes}")
        for t in self.tasks:
            build_task(t.kind, t.name, t.params)  # validates kind and params
        return self

    def policy(self) -> GatePolicy:
        div = self.divergence
        return GatePolicy(
            threshold=self.gate.threshold,
            min_samples=self.gate.min_samples,
            confidence=self.gate.confidence,
            n_boot=self.gate.n_boot,
            seed=self.seed,
            inconclusive_blocks=self.gate.inconclusive_blocks,
            tasks={
                t.name: TaskPolicy(threshold=t.threshold, min_samples=t.min_samples)
                for t in self.tasks
                if t.threshold is not None or t.min_samples is not None
            },
            max_kl=div.max_kl if div else None,
            min_top1=div.min_top1 if div else None,
            sanity=self.sanity_limits,
        )

    def extra_body(self) -> dict[str, Any]:
        """Per-request extras every eval request carries."""
        return (
            {"chat_template_kwargs": self.chat_template_kwargs} if self.chat_template_kwargs else {}
        )


def load_suite(name_or_path: str | Path) -> Suite:
    """Load `bench/evals/<name>.yaml`, or a suite file by path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = EVALS_DIR / f"{name_or_path}.yaml"
    return Suite.model_validate(read_yaml(path))
