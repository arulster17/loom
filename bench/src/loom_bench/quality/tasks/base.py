"""The eval-task contract: a task turns an endpoint into per-item scores."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from loom_bench.provenance import config_hash
from loom_bench.quality.client import EvalClient, EvalRequestError


class ItemResult(BaseModel):
    """Score of one eval item, in [0, 1].

    `item_id` pairs baseline and candidate results; `content_hash` (when known)
    fingerprints the item itself so the gate refuses to pair different data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    score: float = Field(ge=0.0, le=1.0)
    content_hash: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class OutputKind(StrEnum):
    TEXT = "text"
    CODE = "code"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class Completion:
    """A model output kept in memory for sanity checks; never persisted or logged."""

    item_id: str
    text: str
    finish_reason: str | None
    kind: OutputKind = OutputKind.TEXT
    length_expected: bool = False


@dataclass(slots=True)
class TaskOutput:
    items: list[ItemResult]
    completions: list[Completion] = field(default_factory=list)
    # Task-specific provenance: dataset source/revision/license, harness versions, ...
    provenance: dict[str, Any] = field(default_factory=dict)
    # Version of what actually ran, when only known at run time (harness task versions,
    # data file version); stored with the eval run instead of the task's own version.
    version: str | None = None


@dataclass(frozen=True, slots=True)
class EvalContext:
    client: EvalClient
    workdir: Path
    # Code-executing tasks refuse to run unless this is set explicitly.
    allow_code_exec: bool = False
    # HF repo -> local snapshot directory to load its tokenizer from instead of the Hub.
    local_tokenizers: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class EvalTask(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str:
        """Bumped whenever items, prompts or scoring change; stored with every eval run."""
        ...

    def planned_items(self) -> int | None:
        """Items a run will score, when known before it runs (the planner's eval estimate)."""
        ...

    async def run(self, ctx: EvalContext) -> TaskOutput: ...


class TaskParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ParamTask[P: TaskParams]:
    """Base for tasks configured by a Pydantic params model (suite YAML `params`)."""

    Params: ClassVar[type[TaskParams]]
    version: ClassVar[str]

    def __init__(self, name: str, params: P) -> None:
        self.name = name
        self.params = params

    @classmethod
    def from_params(cls, name: str, params: dict[str, Any]) -> Self:
        return cls(name, cls.Params.model_validate(params))  # type: ignore[arg-type]


def item_hash(content: Any) -> str:
    return config_hash(content)


async def score_items[T](
    items: Sequence[T],
    score: Callable[[T], Awaitable[tuple[ItemResult, Completion | None]]],
) -> TaskOutput:
    """Score every item concurrently (the client bounds in-flight requests).

    An item whose request fails after retries scores 0 with the error in
    `meta`: a config that cannot answer must never look as good as one that can.
    """
    results = await asyncio.gather(*(score(i) for i in items))
    return TaskOutput(
        items=[r for r, _ in results],
        completions=[c for _, c in results if c is not None],
    )


# Above this share of items rejected with a non-retryable 4xx, a task fails instead of
# scoring: the server refused the request shape (a missing engine flag, an unsupported
# parameter), so its zeros would measure the config, not the model. A few rejections
# can be item-specific (one over-long prompt) and stay scored 0.
MAX_REJECTED_FRACTION = 0.10


class EvalTaskFailed(RuntimeError):
    """A task's requests failed too broadly for its scores to mean anything."""


def _rejected(status: object) -> bool:
    return isinstance(status, int) and 400 <= status < 500 and status not in (408, 429)


def check_request_errors(task: str, items: Sequence[ItemResult]) -> None:
    """Fail a task whose scores would mostly measure request errors, not answers.

    Two engines that both reject every request would otherwise score 0 vs 0 and pass
    the gate. A task fails when every item errored (nothing was measured), or when
    more than `MAX_REJECTED_FRACTION` of items were rejected with a non-retryable 4xx.
    Errors after exhausted retries (5xx, timeouts) below that stay scored 0: a config
    that cannot answer reliably must not look as good as one that can.
    """
    errored = [i for i in items if "error" in i.meta]
    if not errored:
        return
    rejected = [i for i in errored if _rejected(i.meta.get("status"))]
    n = len(items)
    if rejected and len(rejected) > MAX_REJECTED_FRACTION * n:
        first = rejected[0].meta
        raise EvalTaskFailed(
            f"{task}: {len(rejected)} of {n} requests rejected "
            f"(HTTP {first.get('status')}); first: {first.get('error')}"
        )
    if len(errored) == n:
        first = errored[0].meta
        raise EvalTaskFailed(
            f"{task}: all {n} requests failed (HTTP {first.get('status')}); "
            f"first: {first.get('error')}"
        )


def failed_item(item_id: str, content_hash: str, error: EvalRequestError) -> ItemResult:
    return ItemResult(
        item_id=item_id,
        score=0.0,
        content_hash=content_hash,
        meta={"error": str(error), "status": error.status},
    )
