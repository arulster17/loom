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
wrong, as an engine that loses type fidelity would break real clients. This
matches BFCL's AST checker for Python-style schemas (int accepted for a float,
numeric strings rejected). vLLM's `llama3_json` parser re-serialises the model's
own JSON, so a quoted number in the arguments is the model's output, not the
parser's.

A failed item stores its raw tool calls and text (capped) and the expected call
in `meta`, plus the parameter that failed, so a low score can be explained from
the samples without re-running the model.

`tool_calling_strict` is the same items and scoring with each tool sent in
strict mode: `"strict": true` on the function and `additionalProperties: false`
on every object schema (`strict_parameters`). With tool_choice "auto", vLLM
v0.30 and SGLang v0.5.21 then constrain the call to the schema with an xgrammar
structural tag, but only once the model starts the call with the tag's trigger.
For vLLM `llama3_json` that is exactly `{"name": `, which Llama 3.3 70B does not
write, so its strict calls are unconstrained (docs/quality-gate.md, "Strict tool
calling"); a failed strict item therefore also stores the model's raw reply
(`unconstrained_text`). `tool_calling` stays the headline: most clients send plain tools, and
the strict variant measures what a client that opts in gets. Required lists are
left as they are (OpenAI's strict style would require every property and make
optional ones nullable): neither engine asks for that, and it would change what
the model is asked to do and how an omitted argument is scored.
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
    clip,
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

    def tools_for(self, item: ToolItem, *, strict: bool = False) -> list[dict[str, Any]]:
        """The item's functions as request `tools`; `strict` sends them in strict mode."""
        tools = []
        for name in item.functions:
            function = {"name": name, **self.functions[name]}
            if strict:
                function["parameters"] = strict_parameters(function["parameters"])
                function["strict"] = True
            tools.append({"type": "function", "function": function})
        return tools


def strict_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """`schema` with `additionalProperties: false` on every object, nested ones included.

    Grammar backends differ on whether a schema that is silent about extra keys
    allows them; stating it keeps the constraint the same on every engine.
    """
    out = dict(schema)
    if out.get("type") == "object":
        out["additionalProperties"] = False
    if "properties" in out:
        out["properties"] = {k: strict_parameters(v) for k, v in out["properties"].items()}
    if isinstance(out.get("items"), dict):
        out["items"] = strict_parameters(out["items"])
    return out


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


@dataclass(frozen=True, slots=True)
class Match:
    ok: bool
    reason: str
    argument: str | None = None  # the parameter a value or argument check failed on


def check_call(calls: tuple[ToolCall, ...], item: ToolItem, function: dict[str, Any]) -> Match:
    """The model's tool calls against the item's expectation, naming what failed."""
    if len(calls) != 1:
        return Match(False, f"expected 1 tool call, got {len(calls)}")
    call = calls[0]
    if call.name != item.expected_name:
        return Match(False, "wrong_function")
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError:
        return Match(False, "arguments_not_json")
    if not isinstance(args, dict):
        return Match(False, "arguments_not_object")
    params = function["parameters"]
    props, required = params["properties"], set(params.get("required", []))
    if extra := sorted(set(args) - set(props)):
        return Match(False, "unexpected_argument", extra[0])
    if missing := sorted(required - set(args)):
        return Match(False, "missing_required_argument", missing[0])
    for name, accepted in item.expected_args.items():
        if name not in args:
            if None not in accepted:
                return Match(False, "missing_argument", name)
            continue
        if not any(
            values_equal(args[name], want, props[name]) for want in accepted if want is not None
        ):
            return Match(False, "wrong_value", name)
    return Match(True, "match")


def match_call(
    calls: tuple[ToolCall, ...], item: ToolItem, function: dict[str, Any]
) -> tuple[bool, str]:
    """(matched, reason) for the model's tool calls against the item's expectation."""
    m = check_call(calls, item, function)
    return m.ok, m.reason


MAX_STORED_CALLS = 4


def failure_meta(
    match: Match, calls: tuple[ToolCall, ...], text: str | None, item: ToolItem
) -> dict[str, Any]:
    """What a failed item stores beyond its reason: the raw output and the expectation.

    Raw arguments are kept as the server sent them (a string), so a type mismatch
    such as `"250"` for an integer stays visible; each string is capped by `clip`.
    """
    meta: dict[str, Any] = {
        "output": {
            "tool_calls": [
                {"name": c.name, "arguments": clip(c.arguments)} for c in calls[:MAX_STORED_CALLS]
            ],
            "n_tool_calls": len(calls),
            "text": clip(text or ""),
        },
        "expected": {"name": item.expected_name, "arguments": item.expected_args},
    }
    if match.argument is not None:
        meta["argument"] = match.argument
    return meta


async def unconstrained_text(
    ctx: EvalContext, item: ToolItem, tools: list[dict[str, Any]], max_tokens: int
) -> dict[str, Any]:
    """The model's raw, unparsed reply to a failed strict item, for diagnosis only.

    The tool parser re-serialises a call, so the scored reply cannot show how the model
    began it, and that decides whether strict mode engaged: vLLM's structural tag for
    `llama3_json` only constrains a call that starts with exactly `{"name": ` (see
    docs/quality-gate.md, "Strict tool calling"). This re-asks with tool_choice "none"
    and special tokens kept: no tool parser and no structural tag run, and on vLLM the
    tools stay in the prompt (`--exclude-tools-when-tool-choice-none` is off by default),
    so with greedy decoding this is the reply the model writes when nothing constrains
    it. (SGLang drops the tools from the prompt under "none", so there the text answers
    a different prompt.) Never scored.
    """
    try:
        res = await ctx.client.chat(
            [{"role": "user", "content": item.query}],
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="none",
            skip_special_tokens=False,
        )
    except EvalRequestError as e:
        return {"unconstrained_error": clip(str(e))}
    return {"unconstrained_text": clip(res.text)}


class ToolCallingParams(TaskParams):
    categories: tuple[Category, ...] = ("simple", "multiple")
    limit: Annotated[int, Field(ge=1)] | None = None  # first N items of those categories
    max_tokens: Annotated[int, Field(ge=1)] = 512


class ToolCallingTask(ParamTask[ToolCallingParams]):
    Params = ToolCallingParams
    version: ClassVar[str] = "1"
    strict: ClassVar[bool] = False  # send tools in strict mode (`tool_calling_strict`)

    def planned_items(self) -> int:
        n = sum(i.category in self.params.categories for i in load_data().items)
        return n if self.params.limit is None else min(n, self.params.limit)

    async def run(self, ctx: EvalContext) -> TaskOutput:
        data = load_data()
        items = [i for i in data.items if i.category in self.params.categories]
        if self.params.limit is not None:
            items = items[: self.params.limit]

        async def score(item: ToolItem) -> tuple[ItemResult, Completion | None]:
            tools = data.tools_for(item, strict=self.strict)
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
            match = check_call(res.tool_calls, item, data.functions[item.expected_name])
            text = res.text or "".join(c.arguments for c in res.tool_calls)
            meta: dict[str, Any] = {"result": match.reason, "category": item.category}
            if not match.ok:
                meta |= failure_meta(match, res.tool_calls, res.text, item)
                if self.strict:
                    meta |= await unconstrained_text(ctx, item, tools, self.params.max_tokens)
            return (
                ItemResult(
                    item_id=item.item_id,
                    score=float(match.ok),
                    content_hash=content,
                    meta=meta,
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
        if self.strict:
            out.provenance["tools"] = "strict: true, additionalProperties: false"
        return out


class ToolCallingStrictTask(ToolCallingTask):
    """`tool_calling_strict`: the `tool_calling` items and scoring, tools sent strict.

    Its version ("strict.1+data.N") never equals `tool_calling`'s, and every item's
    content hash covers the strict tools, so the two never pair in a gate.
    """

    version: ClassVar[str] = "strict.1"
    strict: ClassVar[bool] = True
