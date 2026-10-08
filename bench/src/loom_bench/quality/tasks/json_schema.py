"""`json_schema`: share of replies that are valid JSON for the requested schema.

Items come from the pinned, self-authored `data/json_schema.yaml`. Each request
sends the schema as ``response_format: {type: json_schema, strict: true}``, so
on a healthy engine the score is near 1 and the task catches broken structured
output (grammar backend, tokenizer or quantization regressions).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Annotated, Any, ClassVar

import yaml
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from jsonschema.exceptions import best_match  # type: ignore[import-untyped]
from pydantic import Field

from loom_bench.quality.client import EvalRequestError
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

DATA_FILE = "json_schema.yaml"


@dataclass(frozen=True, slots=True)
class SchemaItem:
    item_id: str
    instruction: str
    schema: dict[str, Any]


@cache
def load_items() -> tuple[tuple[SchemaItem, ...], int]:
    """The pinned items and the data file's version. Schemas are checked on load."""
    path = resources.files("loom_bench.quality") / "data" / DATA_FILE
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = tuple(
        SchemaItem(item_id=raw["id"], instruction=raw["instruction"], schema=raw["schema"])
        for raw in data["items"]
    )
    if len({i.item_id for i in items}) != len(items):
        raise ValueError(f"{DATA_FILE}: duplicate item ids")
    for item in items:
        Draft202012Validator.check_schema(item.schema)
    return items, int(data["version"])


def score_reply(text: str, schema: dict[str, Any]) -> tuple[float, str]:
    """(1.0, "valid") when `text` is JSON valid for `schema`, else (0.0, why)."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return 0.0, "invalid_json"
    if not Draft202012Validator(schema).is_valid(value):
        return 0.0, "schema_violation"
    return 1.0, "valid"


def failure_meta(text: str, schema: dict[str, Any], why: str) -> dict[str, Any]:
    """What a failed item stores: its raw reply and, for a violation, the main reason.

    The expectation is the item's pinned schema, found by item id in the data file.
    """
    meta: dict[str, Any] = {"output": clip(text)}
    if why == "schema_violation":
        error = best_match(Draft202012Validator(schema).iter_errors(json.loads(text)))
        if error is not None:
            where = "/".join(str(p) for p in error.absolute_path) or "(root)"
            meta["violation"] = clip(f"{where}: {error.message}", 500)
    return meta


class JsonSchemaParams(TaskParams):
    limit: Annotated[int, Field(ge=1)] | None = None
    max_tokens: Annotated[int, Field(ge=1)] = 1024


class JsonSchemaTask(ParamTask[JsonSchemaParams]):
    Params = JsonSchemaParams
    version: ClassVar[str] = "1"

    def planned_items(self) -> int:
        n = len(load_items()[0])
        return n if self.params.limit is None else min(n, self.params.limit)

    async def run(self, ctx: EvalContext) -> TaskOutput:
        items, data_version = load_items()
        if self.params.limit is not None:
            items = items[: self.params.limit]

        async def score(item: SchemaItem) -> tuple[ItemResult, Completion | None]:
            content = item_hash({"instruction": item.instruction, "schema": item.schema})
            fmt = {
                "type": "json_schema",
                "json_schema": {"name": item.item_id, "schema": item.schema, "strict": True},
            }
            messages = [
                {"role": "system", "content": "Reply with a single JSON object and nothing else."},
                {"role": "user", "content": item.instruction},
            ]
            try:
                res = await ctx.client.chat(
                    messages, max_tokens=self.params.max_tokens, response_format=fmt
                )
            except EvalRequestError as e:
                return failed_item(item.item_id, content, e), None
            value, why = score_reply(res.text, item.schema)
            meta: dict[str, Any] = {"result": why}
            if value < 1.0:
                meta |= failure_meta(res.text, item.schema, why)
            return (
                ItemResult(item_id=item.item_id, score=value, content_hash=content, meta=meta),
                Completion(item.item_id, res.text, res.finish_reason, kind=OutputKind.JSON),
            )

        out = await score_items(items, score)
        out.version = f"{self.version}+data.{data_version}"
        out.provenance = {
            "dataset": {
                "name": "loom-json-schema",
                "source": f"loom_bench/quality/data/{DATA_FILE}",
                "revision": str(data_version),
                "license": "Apache-2.0",
            }
        }
        return out
