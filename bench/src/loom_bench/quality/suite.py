"""Pinned eval suites: `bench/evals/<model>.yaml`, validated at load time.

A suite fixes everything that decides which items are scored and how: task
list and parameters, sample counts, seeds, chat-template kwargs, gate
thresholds and divergence / sanity limits. Baseline and candidate must run the
same suite for the gate to compare them. Named `subsets` pick some of its
tasks, unchanged, for runs that cannot afford all of them; a subset's results
pair item for item with a full run of the same tasks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    model_validator,
)

from loom_bench.quality.gate import (
    GatePolicy,
    MinSamples,
    Nats,
    NoiseMultiple,
    Share,
    TaskPolicy,
    Threshold,
)
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
    # Items the task scores, for the planner's time estimate; only needed when the task
    # cannot count them itself (lm_eval docs without explicit `samples`).
    items: PositiveInt | None = None

    def planned_items(self) -> int | None:
        """Items a run scores, when known before it runs."""
        if self.items is not None:
            return self.items
        return build_task(self.kind, self.name, self.params).planned_items()


class GateSpec(_Strict):
    threshold: Threshold = 0.01
    min_samples: MinSamples = 300
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.95
    n_boot: Annotated[int, Field(ge=100)] = 10_000
    inconclusive_blocks: bool = True
    review_blocks: bool = False  # REVIEW (divergence beyond the noise floor) blocks too


class DivergenceSpec(_Strict):
    prompts: Annotated[int, Field(ge=2)] | None = None  # first N pinned prompts; None = all
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    max_new_tokens: Annotated[int, Field(ge=1)] = 64
    # Absolute limits; the noise-calibrated limits are never stricter than these.
    max_kl: Nats = 0.05
    min_top1: Share = 0.95
    # Limits widen to this multiple of the baseline's self-divergence (its noise floor).
    noise_multiple: NoiseMultiple = 5.0
    # Hard ceiling: divergence beyond it FAILs whatever the noise floor and task scores.
    ceiling_kl: Nats = 0.5
    ceiling_top1: Share = 0.80
    # Concurrency of the baseline's second scoring pass (the noise floor); it must differ
    # from the eval job's, so requests land in different batches than at capture.
    floor_concurrency: PositiveInt = 1


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
    subsets: dict[TaskName, Annotated[list[TaskName], Field(min_length=1)]] = Field(
        default_factory=dict
    )
    # Set by `limited`: every task capped near this many items. A smoke check that each
    # task loads and runs end to end; its scores measure nothing.
    item_limit: PositiveInt | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [t.name for t in self.tasks]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate task names: {dupes}")
        for t in self.tasks:
            counted = build_task(t.kind, t.name, t.params).planned_items()  # validates params
            if t.items is not None and counted is not None and t.items != counted:
                raise ValueError(f"task {t.name}: items {t.items} but its params give {counted}")
        for subset, members in self.subsets.items():
            unknown = sorted(set(members) - set(names))
            if unknown or len(set(members)) != len(members):
                raise ValueError(f"subset {subset}: unknown or repeated tasks in {members}")
        self.policy()  # validates the divergence limits against the ceiling
        return self

    def select(self, subset: str | None) -> list[SuiteTask]:
        """The tasks a run of `subset` (None: the whole suite) executes, in suite order."""
        if subset is None:
            return list(self.tasks)
        if subset not in self.subsets:
            raise ValueError(f"suite {self.suite} has no subset {subset!r}: {sorted(self.subsets)}")
        return [t for t in self.tasks if t.name in self.subsets[subset]]

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
            review_blocks=self.gate.review_blocks,
            max_kl=div.max_kl if div else None,
            min_top1=div.min_top1 if div else None,
            noise_multiple=div.noise_multiple if div else GatePolicy().noise_multiple,
            ceiling_kl=div.ceiling_kl if div else None,
            ceiling_top1=div.ceiling_top1 if div else None,
            sanity=self.sanity_limits,
        )

    def limited(self, n: int) -> Suite:
        """This suite with every task capped near `n` items and divergence at `n` prompts
        (at least 2), everything else unchanged: same tasks, datasets, parameters and
        harness code paths at smoke scale. Raises for a task kind with no item limit."""
        if n < 1:
            raise ValueError("limit must be at least 1")
        doc = self.model_dump(mode="json")
        doc["tasks"] = [_limit_task(t, n) for t in self.tasks]
        if self.divergence is not None:
            cap = max(2, n)
            prompts = self.divergence.prompts
            doc["divergence"]["prompts"] = cap if prompts is None else min(prompts, cap)
        doc["item_limit"] = n
        return Suite.model_validate(doc)

    def extra_body(self) -> dict[str, Any]:
        """Per-request extras every eval request carries."""
        return (
            {"chat_template_kwargs": self.chat_template_kwargs} if self.chat_template_kwargs else {}
        )


# The parameter that caps each task kind's items (`Suite.limited`).
_ITEM_LIMIT_PARAM = {
    "lm_eval": "limit",  # first N docs of every (sub)task
    "json_schema": "limit",
    "tool_calling": "limit",
    "code_exec": "limit",  # per dataset
    "toy_arithmetic": "n",
}


def _limit_task(task: SuiteTask, n: int) -> dict[str, Any]:
    doc = task.model_dump(mode="json")
    params = dict(task.params)
    if task.kind == "needle":
        params["samples_per_cell"] = 1  # one item per (length, depth) cell
        doc["items"] = None
    elif task.kind == "lm_eval" and params.get("samples") is not None:
        params["samples"] = {k: list(v)[:n] for k, v in params["samples"].items()}
        doc["items"] = None
    elif task.kind in _ITEM_LIMIT_PARAM:
        key = _ITEM_LIMIT_PARAM[task.kind]
        old = params.get(key)
        params[key] = n if old is None else min(int(old), n)
        if task.items is not None and task.kind == "lm_eval":
            # planner estimate only: scale the declared count by the new per-task limit
            scaled = math.ceil(task.items * params[key] / old) if old else min(task.items, n)
            doc["items"] = max(1, scaled)
        else:
            doc["items"] = None
    else:
        raise ValueError(f"task {task.name}: kind {task.kind!r} has no item limit")
    doc["params"] = params
    return doc


def load_suite(name_or_path: str | Path) -> Suite:
    """Load `bench/evals/<name>.yaml`, or a suite file by path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = EVALS_DIR / f"{name_or_path}.yaml"
    return Suite.model_validate(read_yaml(path))
