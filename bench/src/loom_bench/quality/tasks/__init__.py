"""Eval task registry: suite YAML `kind` -> task class."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loom_bench.quality.tasks.arithmetic import ArithmeticTask
from loom_bench.quality.tasks.base import EvalTask, ItemResult, TaskOutput
from loom_bench.quality.tasks.code_exec import CodeExecTask
from loom_bench.quality.tasks.json_schema import JsonSchemaTask
from loom_bench.quality.tasks.lmeval import LmEvalTask
from loom_bench.quality.tasks.needle import NeedleTask
from loom_bench.quality.tasks.tool_calling import ToolCallingStrictTask, ToolCallingTask

TaskFactory = Callable[[str, dict[str, Any]], EvalTask]

TASKS: dict[str, TaskFactory] = {
    "toy_arithmetic": ArithmeticTask.from_params,
    "json_schema": JsonSchemaTask.from_params,
    "tool_calling": ToolCallingTask.from_params,
    "tool_calling_strict": ToolCallingStrictTask.from_params,
    "needle": NeedleTask.from_params,
    "code_exec": CodeExecTask.from_params,
    "lm_eval": LmEvalTask.from_params,
}


def build_task(kind: str, name: str, params: dict[str, Any] | None = None) -> EvalTask:
    try:
        factory = TASKS[kind]
    except KeyError:
        raise ValueError(f"unknown eval task kind {kind!r}; known: {sorted(TASKS)}") from None
    return factory(name, params or {})


__all__ = ["TASKS", "EvalTask", "ItemResult", "TaskOutput", "build_task"]
