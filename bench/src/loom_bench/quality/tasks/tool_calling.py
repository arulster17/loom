"""`tool_calling`: BFCL-style function calling with AST-style argument matching.

Items come from the pinned, self-authored `data/tool_calling.yaml`. The model
gets the item's functions as `tools` (tool_choice "auto") and must answer with
exactly one tool call.

Matching rules (score 1 only if all hold):

- exactly one tool call, to the expected function name (exact match);
- arguments parse as a JSON object;
- no argument outside the function's schema, every schema-required argument
  present;
- every argument the model passed equals one of the expected values; an
  argument it left out must have `null` among its expected values.

Value equality, by the parameter's JSON-schema type:

- string: equal after Unicode case folding, trimming and collapsing runs of
  whitespace ("New  York " == "new york"); no other normalisation;
- integer: an int, or a float with an integral value (3.0 == 3); never bool
  or a numeric string;
- number: an int or float (not bool or string), equal within 1e-9 relative;
- boolean: a real JSON boolean only ("true" is rejected);
- array: same length, elements equal pairwise in order using `items`;
- object: same keys, values equal per `properties`.

Typed JSON is part of the contract: a string "5" for an integer parameter is
wrong, as an engine that loses type fidelity would break real clients.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import Field

from loom_bench.quality.client import EvalRequestError, ToolCall
from loom_bench.quality.tasks.base import (
    Completion,
    EvalContext,
    ItemResult,
    OutputKind,
    ParamTask,
    TaskOutput,
    TaskParams,
    failed_item,
    item_hash,
    score_items,
)

DATA_FILE = "tool_calling.yaml"
Category = Literal["simple", "multiple"]


@dataclass(frozen=True, slots=True)
class ToolItem:
    item_id: str
    category: Category
    query: str
    functions: tuple[str, ...]
    expected_name: str
    expected_args: dict[str, list[Any]]


@dataclass(frozen=True, slots=True)
class ToolData:
    functions: dict[str, dict[str, Any]]  # name -> {description, parameters}
    items: tuple[ToolItem, ...]
    version: int

    def tools_for(self, item: ToolItem) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": name, **self.functions[name]}}
            for name in item.functions
        ]


def _validate(data: ToolData) -> None:
    ids = [i.item_id for i in data.items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{DATA_FILE}: duplicate item ids")
    for item in data.items:
        where = f"{DATA_FILE}: {item.item_id}"
        unknown = [f for f in item.functions if f not in data.functions]
        if unknown:
            raise ValueError(f"{where}: unknown functions {unknown}")
        if item.expected_name not in item.functions:
            raise ValueError(f"{where}: expected function is not offered")
        params = data.functions[item.expected_name]["parameters"]
        props, required = params["properties"], set(params.get("required", []))
        if set(item.expected_args) != set(props):
            raise ValueError(f"{where}: expected arguments must list every parameter")
        for name, values in item.expected_args.items():
            if not values:
                raise ValueError(f"{where}: no acceptable values for {name}")
            if name in required and None in values:
                raise ValueError(f"{where}: required argument {name} cannot be omitted")


@cache
def load_data() -> ToolData:
    path = resources.files("loom_bench.quality") / "data" / DATA_FILE
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = tuple(
        ToolItem(
            item_id=i["id"],
            category=i["category"],
            query=i["query"],
            functions=tuple(i["functions"]),
            expected_name=i["expected"]["name"],
            expected_args={k: list(v) for k, v in i["expected"]["arguments"].items()},
        )
        for i in raw["items"]
    )
    data = ToolData(functions=raw["functions"], items=items, version=int(raw["version"]))
    _validate(data)
    return data


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def values_equal(got: Any, want: Any, schema: dict[str, Any]) -> bool:
    """Whether the model's `got` equals the expected `want` for a parameter `schema`."""
    kind = schema.get("type")
    match kind:
        case "string":
            return isinstance(got, str) and isinstance(want, str) and _norm(got) == _norm(want)
        case "integer":
            if isinstance(got, bool) or not isinstance(got, int | float):
                return False
            return float(got).is_integer() and int(got) == want
        case "number":
            if isinstance(got, bool) or not isinstance(got, int | float):
                return False
            return math.isclose(float(got), float(want), rel_tol=1e-9, abs_tol=1e-12)
        case "boolean":
            return isinstance(got, bool) and got is want
        case "array":
            if not isinstance(got, list) or len(got) != len(want):
                return False
            items = schema.get("items", {})
            return all(values_equal(g, w, items) for g, w in zip(got, want, strict=True))
        case "object":
            if not isinstance(got, dict) or got.keys() != want.keys():
                return False
            props = schema.get("properties", {})
            return all(values_equal(got[k], want[k], props.get(k, {})) for k in want)
        case _:
            raise ValueError(f"unsupported parameter type {kind!r}")


def match_call(
    calls: tuple[ToolCall, ...], item: ToolItem, function: dict[str, Any]
) -> tuple[bool, str]:
    """(matched, reason) for the model's tool calls against the item's expectation."""
    if len(calls) != 1:
        return False, f"expected 1 tool call, got {len(calls)}"
    call = calls[0]
    if call.name != item.expected_name:
        return False, "wrong_function"
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return False, "arguments_not_json"
    if not isinstance(args, dict):
        return False, "arguments_not_object"
    params = function["parameters"]
    props, required = params["properties"], set(params.get("required", []))
    if set(args) - set(props):
        return False, "unexpected_argument"
    if required - set(args):
        return False, "missing_required_argument"
    for name, accepted in item.expected_args.items():
        if name not in args:
            if None not in accepted:
                return False, "missing_argument"
            continue
        if not any(
            values_equal(args[name], want, props[name]) for want in accepted if want is not None
        ):
            return False, "wrong_value"
    return True, "match"


class ToolCallingParams(TaskParams):
    categories: tuple[Category, ...] = ("simple", "multiple")
    max_tokens: Annotated[int, Field(ge=1)] = 512


class ToolCallingTask(ParamTask[ToolCallingParams]):
    Params = ToolCallingParams
    version: ClassVar[str] = "1"

    async def run(self, ctx: EvalContext) -> TaskOutput:
        data = load_data()
        items = [i for i in data.items if i.category in self.params.categories]

        async def score(item: ToolItem) -> tuple[ItemResult, Completion | None]:
            tools = data.tools_for(item)
            content = item_hash(
                {"query": item.query, "tools": tools, "expected": item.expected_args}
            )
            try:
                res = await ctx.client.chat(
                    [{"role": "user", "content": item.query}],
                    max_tokens=self.params.max_tokens,
                    tools=tools,
                    tool_choice="auto",
                )
            except EvalRequestError as e:
                return failed_item(item.item_id, content, e), None
            ok, why = match_call(res.tool_calls, item, data.functions[item.expected_name])
            text = res.text or "".join(c.arguments for c in res.tool_calls)
            return (
                ItemResult(
                    item_id=item.item_id,
                    score=float(ok),
                    content_hash=content,
                    meta={"result": why, "category": item.category},
                ),
                Completion(item.item_id, text, res.finish_reason, kind=OutputKind.JSON),
            )

        out = await score_items(items, score)
        out.version = f"{self.version}+data.{data.version}"
        out.provenance = {
            "dataset": {
                "name": "loom-tool-calling",
                "source": f"loom_bench/quality/data/{DATA_FILE}",
                "revision": str(data.version),
                "license": "Apache-2.0",
            }
        }
        return out
