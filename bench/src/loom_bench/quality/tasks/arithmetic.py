"""`toy_arithmetic`: generated "What is A <op> B?" questions, scored by parsing
"The answer is N". Cheap and fully deterministic; the gate's acceptance test
runs it against the mock backend."""

from __future__ import annotations

import operator
import random
import re
from dataclasses import dataclass
from typing import Annotated, ClassVar, Literal

from pydantic import Field

from loom_bench.quality.client import EvalRequestError
from loom_bench.quality.tasks.base import (
    Completion,
    EvalContext,
    ItemResult,
    ParamTask,
    TaskOutput,
    TaskParams,
    failed_item,
    item_hash,
    score_items,
)

_ANSWER_RE = re.compile(r"The answer is\s+(-?\d+)", re.IGNORECASE)
_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


@dataclass(frozen=True, slots=True)
class ArithmeticItem:
    item_id: str
    a: int
    op: str
    b: int

    @property
    def question(self) -> str:
        return f"What is {self.a} {self.op} {self.b}?"

    @property
    def expected(self) -> int:
        return _OPS[self.op](self.a, self.b)


def parse_answer(text: str) -> int | None:
    """The last "The answer is N" in `text`, or None."""
    matches = _ANSWER_RE.findall(text)
    return int(matches[-1]) if matches else None


class ArithmeticParams(TaskParams):
    n: Annotated[int, Field(ge=1)] = 400
    seed: int = 0
    ops: tuple[Literal["+", "-", "*"], ...] = ("+", "-", "*")
    max_operand: Annotated[int, Field(ge=1)] = 999
    max_tokens: Annotated[int, Field(ge=1)] = 32


def generate_items(params: ArithmeticParams) -> list[ArithmeticItem]:
    rng = random.Random(params.seed)
    return [
        ArithmeticItem(
            item_id=f"arith-{i:05d}",
            a=rng.randint(0, params.max_operand),
            op=rng.choice(params.ops),
            b=rng.randint(0, params.max_operand),
        )
        for i in range(params.n)
    ]


class ArithmeticTask(ParamTask[ArithmeticParams]):
    Params = ArithmeticParams
    version: ClassVar[str] = "1"

    def planned_items(self) -> int:
        return self.params.n

    async def run(self, ctx: EvalContext) -> TaskOutput:
        p = self.params

        async def score(item: ArithmeticItem) -> tuple[ItemResult, Completion | None]:
            content = item_hash({"q": item.question, "expected": item.expected})
            prompt = f"{item.question} Reply in the form 'The answer is N.'"
            try:
                res = await ctx.client.chat(
                    [{"role": "user", "content": prompt}], max_tokens=p.max_tokens
                )
            except EvalRequestError as e:
                return failed_item(item.item_id, content, e), None
            ok = parse_answer(res.text) == item.expected
            return (
                ItemResult(item_id=item.item_id, score=float(ok), content_hash=content),
                Completion(item.item_id, res.text, res.finish_reason),
            )

        out = await score_items(generate_items(p), score)
        out.provenance = {"dataset": {"name": "toy_arithmetic", "content": "synthetic"}}
        return out
