"""`code_exec`: HumanEval / MBPP-style functional correctness (pass@1, greedy).

The model writes code through the chat endpoint; each program plus its tests
then runs in `loom_bench.quality.sandbox`. The task refuses to run unless the
eval context sets `allow_code_exec=True`.

Datasets are read from the Hugging Face Hub at pinned revisions, only when the
task runs:

- humaneval: openai/openai_humaneval (MIT), 164 problems;
- mbpp: google-research-datasets/mbpp, config "full", test split (CC-BY-4.0),
  500 problems.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from loom_bench.quality.client import EvalRequestError
from loom_bench.quality.sandbox import CodeExecDisabled, SandboxLimits, run_python
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

Dataset = Literal["humaneval", "mbpp"]


@dataclass(frozen=True, slots=True)
class DatasetSource:
    repo: str
    revision: str
    filename: str
    license: str


SOURCES: dict[str, DatasetSource] = {
    "humaneval": DatasetSource(
        repo="openai/openai_humaneval",
        revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        filename="openai_humaneval/test-00000-of-00001.parquet",
        license="MIT",
    ),
    "mbpp": DatasetSource(
        repo="google-research-datasets/mbpp",
        revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        filename="full/test-00000-of-00001.parquet",
        license="CC-BY-4.0",
    ),
}

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CodeProblem:
    item_id: str
    prompt: str  # what the model is asked
    prefix: str  # code placed before the model's code (HumanEval: the stub with imports)
    tests: str  # code placed after it; must raise on failure


def extract_code(reply: str) -> str:
    """The first fenced code block of `reply`, or the whole reply if there is none."""
    m = _FENCE_RE.search(reply)
    return m.group(1) if m else reply


def build_program(problem: CodeProblem, reply: str) -> str:
    return f"{problem.prefix}\n{extract_code(reply)}\n\n{problem.tests}\n"


def humaneval_problems(rows: Sequence[dict[str, Any]]) -> list[CodeProblem]:
    out = []
    for r in rows:
        prompt = (
            "Complete the following Python function. Reply with the complete function, "
            "including its signature and any imports it needs, in one ```python code block.\n\n"
            f"```python\n{r['prompt']}```"
        )
        # The stub (signature + docstring) is a valid function on its own, so a reply
        # holding either just the body or the whole function both run.
        out.append(
            CodeProblem(
                item_id=str(r["task_id"]),
                prompt=prompt,
                prefix=r["prompt"],
                tests=f"{r['test']}\ncheck({r['entry_point']})",
            )
        )
    return out


def mbpp_problems(rows: Sequence[dict[str, Any]]) -> list[CodeProblem]:
    out = []
    for r in rows:
        tests = "\n".join(r["test_list"])
        prompt = (
            f"{r['text']}\nYour code should pass these tests:\n\n{tests}\n\n"
            "Reply with the Python code in one ```python code block."
        )
        out.append(
            CodeProblem(
                item_id=f"mbpp/{r['task_id']}",
                prompt=prompt,
                prefix=r.get("test_setup_code") or "",
                tests=tests,
            )
        )
    return out


def load_problems(dataset: Dataset) -> list[CodeProblem]:
    """Download (or reuse the HF cache of) the pinned parquet file and parse it."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    src = SOURCES[dataset]
    path = hf_hub_download(src.repo, src.filename, repo_type="dataset", revision=src.revision)
    rows = pq.read_table(path).to_pylist()
    return humaneval_problems(rows) if dataset == "humaneval" else mbpp_problems(rows)


def _provenance(dataset: Dataset) -> dict[str, str]:
    src = SOURCES[dataset]
    return {
        "name": dataset,
        "source": f"hf://datasets/{src.repo}/{src.filename}",
        "revision": src.revision,
        "license": src.license,
    }


class CodeExecParams(TaskParams):
    datasets: Annotated[tuple[Dataset, ...], Field(min_length=1)]
    limit: Annotated[int, Field(ge=1)] | None = None  # per dataset
    max_tokens: Annotated[int, Field(ge=1)] = 1024
    parallelism: Annotated[int, Field(ge=1)] = 4
    sandbox: SandboxLimits = Field(default_factory=SandboxLimits)


class CodeExecTask(ParamTask[CodeExecParams]):
    Params = CodeExecParams
    version: ClassVar[str] = "1"

    def __init__(
        self, name: str, params: CodeExecParams, problems: Sequence[CodeProblem] | None = None
    ) -> None:
        """`problems` replaces the Hub download (tests and local fixtures)."""
        super().__init__(name, params)
        self._problems = list(problems) if problems is not None else None

    async def run(self, ctx: EvalContext) -> TaskOutput:
        if not ctx.allow_code_exec:
            raise CodeExecDisabled(
                f"{self.name}: code_exec runs model-written code; enable it explicitly "
                "with allow_code_exec=True inside an isolated container or VM"
            )
        problems = self._problems
        if problems is None:
            problems = []
            for dataset in self.params.datasets:
                loaded = await asyncio.to_thread(load_problems, dataset)
                problems += loaded[: self.params.limit]
        sem = asyncio.Semaphore(self.params.parallelism)

        async def score(p: CodeProblem) -> tuple[ItemResult, Completion | None]:
            content = item_hash({"prompt": p.prompt, "prefix": p.prefix, "tests": p.tests})
            try:
                res = await ctx.client.chat(
                    [{"role": "user", "content": p.prompt}], max_tokens=self.params.max_tokens
                )
            except EvalRequestError as e:
                return failed_item(p.item_id, content, e), None
            async with sem:
                result = await asyncio.to_thread(
                    run_python,
                    build_program(p, res.text),
                    allow_code_exec=True,
                    limits=self.params.sandbox,
                )
            meta: dict[str, Any] = {"status": result.status}
            if result.error_type:
                meta["error_type"] = result.error_type
            return (
                ItemResult(
                    item_id=p.item_id,
                    score=float(result.status == "passed"),
                    content_hash=content,
                    meta=meta,
                ),
                Completion(p.item_id, res.text, res.finish_reason, kind=OutputKind.CODE),
            )

        out = await score_items(problems, score)
        out.provenance = {"datasets": [_provenance(name) for name in self.params.datasets]}
        return out
