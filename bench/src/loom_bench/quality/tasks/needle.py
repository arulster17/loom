"""`needle`: retrieve one fact hidden in a long synthetic haystack.

For every (context length, depth) cell, `samples_per_cell` items are built
from a seeded RNG: filler sentences assembled from a small self-authored
grammar, with one needle sentence ("The access code for the <colour> <animal>
vault is <6 digits>.") inserted at the given depth (0 = start, 1 = end). The
question follows the haystack. Score 1 when the six-digit code appears in the
reply.

Lengths are sized in characters at `chars_per_token` characters per token, so
they are approximate. The default 3.5 is below what Llama and Qwen tokenizers
average on English prose (about 4 to 4.5), so prompts stay under the target
token count; pick context lengths at most ~90% of the served context window.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Annotated, ClassVar

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

_SUBJECTS = (
    "The harbour master", "A retired teacher", "The night baker", "Our neighbour",
    "The museum guide", "A travelling botanist", "The ferry captain", "The village choir",
    "A young cartographer", "The orchard keeper", "The station clerk", "A quiet violinist",
)  # fmt: skip
_VERBS = (
    "repaired", "painted", "counted", "described", "carried", "measured", "photographed",
    "sketched", "polished", "catalogued", "rearranged", "inspected",
)  # fmt: skip
_OBJECTS = (
    "the old wooden boats", "a basket of pears", "the brass lanterns", "the winding staircase",
    "a stack of letters", "the garden benches", "the copper kettles", "a row of tulips",
    "the stone bridge", "the clock tower", "a map of the coast", "the library shelves",
)  # fmt: skip
_TIMES = (
    "before sunrise", "after the rain", "on a windy afternoon", "during the festival",
    "late in the evening", "on the first day of spring", "while the market was busy",
    "as the tide came in", "under a grey sky", "just after lunch",
)  # fmt: skip
_COLOURS = ("amber", "cobalt", "crimson", "emerald", "ivory", "violet", "silver", "saffron")
_ANIMALS = ("otter", "falcon", "lynx", "heron", "badger", "walrus", "gecko", "ibis")


@dataclass(frozen=True, slots=True)
class NeedleItem:
    item_id: str
    context_tokens: int
    depth: float
    key: str
    code: str
    prompt: str


def _sentence(rng: random.Random) -> str:
    return (
        f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} {rng.choice(_TIMES)}."
    )


def build_item(
    rng: random.Random, item_id: str, context_tokens: int, depth: float, chars_per_token: float
) -> NeedleItem:
    key = f"{rng.choice(_COLOURS)} {rng.choice(_ANIMALS)}"
    code = f"{rng.randrange(10**6):06d}"
    needle = f"The access code for the {key} vault is {code}."
    question = (
        f"\n\nWhat is the access code for the {key} vault? Reply with the six-digit code only."
    )
    header = "Read the following notes carefully.\n\n"
    budget = int(context_tokens * chars_per_token) - len(header) - len(question) - len(needle)
    sentences: list[str] = []
    used = 0
    while True:
        s = _sentence(rng)
        if used + len(s) + 1 > budget:
            break
        sentences.append(s)
        used += len(s) + 1
    sentences.insert(round(depth * len(sentences)), needle)
    prompt = header + " ".join(sentences) + question
    return NeedleItem(item_id, context_tokens, depth, key, code, prompt)


class NeedleParams(TaskParams):
    context_tokens: tuple[Annotated[int, Field(ge=256)], ...] = (4096, 16384)
    depths: tuple[Annotated[float, Field(ge=0.0, le=1.0)], ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    samples_per_cell: Annotated[int, Field(ge=1)] = 20
    seed: int = 0
    chars_per_token: Annotated[float, Field(gt=0)] = 3.5
    max_tokens: Annotated[int, Field(ge=1)] = 32


def generate_items(p: NeedleParams) -> list[NeedleItem]:
    out = []
    for length in p.context_tokens:
        for depth in p.depths:
            for j in range(p.samples_per_cell):
                rng = random.Random(f"{p.seed}:{length}:{depth}:{j}")
                item_id = f"needle-{length}-{depth:.2f}-{j:03d}"
                out.append(build_item(rng, item_id, length, depth, p.chars_per_token))
    return out


def found(reply: str, code: str) -> bool:
    return re.search(rf"(?<!\d){code}(?!\d)", reply) is not None


class NeedleTask(ParamTask[NeedleParams]):
    Params = NeedleParams
    version: ClassVar[str] = "1"

    async def run(self, ctx: EvalContext) -> TaskOutput:
        async def score(item: NeedleItem) -> tuple[ItemResult, Completion | None]:
            content = item_hash({"prompt": item.prompt, "code": item.code})
            try:
                res = await ctx.client.chat(
                    [{"role": "user", "content": item.prompt}], max_tokens=self.params.max_tokens
                )
            except EvalRequestError as e:
                return failed_item(item.item_id, content, e), None
            return (
                ItemResult(
                    item_id=item.item_id,
                    score=float(found(res.text, item.code)),
                    content_hash=content,
                    meta={"context_tokens": item.context_tokens, "depth": item.depth},
                ),
                Completion(item.item_id, res.text, res.finish_reason),
            )

        out = await score_items(generate_items(self.params), score)
        out.provenance = {"dataset": {"name": "loom-needle", "content": "synthetic"}}
        return out
