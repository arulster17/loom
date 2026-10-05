"""Eval task registry: suite YAML `kind` -> task class."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loom_bench.quality.tasks.arithmetic import ArithmeticTask
from loom_bench.quality.tasks.base import EvalTask, ItemResult, TaskOutput

TaskFactory = Callable[[str, dict[str, Any]], EvalTask]

TASKS: dict[str, TaskFactory] = {
    "toy_arithmetic": ArithmeticTask.from_params,
}


def build_task(kind: str, name: str, params: dict[str, Any] | None = None) -> EvalTask:
    try:
        factory = TASKS[kind]
    except KeyError:
        raise ValueError(f"unknown eval task kind {kind!r}; known: {sorted(TASKS)}") from None
    return factory(name, params or {})


__all__ = ["TASKS", "EvalTask", "ItemResult", "TaskOutput", "build_task"]
