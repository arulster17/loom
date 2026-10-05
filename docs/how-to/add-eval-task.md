# How to add an eval task

An eval task turns an OpenAI-compatible endpoint into per-item scores that the quality
gate can pair between a baseline and a candidate config. A new task is code: a class in
`bench/src/loom_bench/quality/tasks/`, registered by a `kind` name, then listed in the
pinned suites. Adding an existing kind to a suite, or changing its parameters, is a YAML
change in `bench/evals/<model>.yaml`. The gate's method is in
[quality-gate.md](../quality-gate.md).

## The contract

`quality/tasks/base.py`:

- `EvalTask` protocol: `name`, `version`, `async run(ctx: EvalContext) -> TaskOutput`.
- `EvalContext`: `client` (an `EvalClient` bound to one endpoint and served model, with
  retries, a concurrency limit, the suite's seed and `chat_template_kwargs`),
  `workdir`, `allow_code_exec`.
- `TaskOutput`: `items` (one `ItemResult` per item), `completions` (raw outputs for the
  sanity checks, kept in memory only), `provenance` (dataset name, source, revision,
  license), optional `version`.
- `ItemResult`: `item_id` (stable: it pairs baseline and candidate), `score` in [0, 1],
  `content_hash` (the gate refuses to pair items whose content changed), `meta` (short
  metadata only; never the model output).

Helpers: `ParamTask` (parameters validated from the suite YAML by a Pydantic `Params`
model), `score_items` (scores items concurrently), `item_hash`, and `failed_item` (an
item whose request fails after retries scores 0 with the error in `meta`, so a config
that cannot answer never looks as good as one that can).

## 1. Write the task

`bench/src/loom_bench/quality/tasks/repeat_word.py`, a minimal complete example:

```python
"""`repeat_word`: ask for a word back verbatim; an item scores 1 if the reply contains it."""

from __future__ import annotations

import random
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

WORDS = ("amber", "basalt", "cobalt", "delta", "ember", "fjord")


class RepeatWordParams(TaskParams):
    n: Annotated[int, Field(ge=1)] = 300
    seed: int = 0
    max_tokens: Annotated[int, Field(ge=1)] = 16


class RepeatWordTask(ParamTask[RepeatWordParams]):
    Params = RepeatWordParams
    version: ClassVar[str] = "1"  # bump when items, prompt or scoring change

    async def run(self, ctx: EvalContext) -> TaskOutput:
        p = self.params
        rng = random.Random(p.seed)
        items = [(f"repeat-{i:05d}", rng.choice(WORDS)) for i in range(p.n)]

        async def score(item: tuple[str, str]) -> tuple[ItemResult, Completion | None]:
            item_id, word = item
            content = item_hash({"word": word})
            prompt = f"Reply with the word {word} and nothing else."
            try:
                res = await ctx.client.chat(
                    [{"role": "user", "content": prompt}], max_tokens=p.max_tokens
                )
            except EvalRequestError as e:
                return failed_item(item_id, content, e), None
            ok = word in res.text.lower()
            return (
                ItemResult(item_id=item_id, score=float(ok), content_hash=content),
                Completion(item_id, res.text, res.finish_reason),
            )

        out = await score_items(items, score)
        out.provenance = {"dataset": {"name": "repeat_word", "content": "synthetic"}}
        return out
```

Use `ctx.client.complete(...)` for raw completions, and pass `tools=` or
`response_format=` through `chat(...)` as the `tool_calling` and `json_schema` tasks do.
Mark `Completion(kind=OutputKind.CODE or JSON)` for code and JSON outputs (they skip the
n-gram and script sanity checks) and `length_expected=True` when hitting `max_tokens` is
normal for the task.

## 2. Data

- **Fixed items you write**: ship them as `bench/src/loom_bench/quality/data/<task>.yaml`
  with a `version` field and a license note in the header (see `json_schema.yaml`), and
  load them with `importlib.resources.files("loom_bench.quality") / "data" / ...`.
- **Third-party data**: load it from the Hugging Face Hub at a pinned revision, only when
  the task runs (`hf_hub_download(..., revision=<sha>)`, as `code_exec.py` does), and
  record source, revision and license in `TaskOutput.provenance`. Use only data whose
  license allows redistribution of derived scores.
- **Generated items**: seed them from `params`, as above.

## 3. Register the kind

`bench/src/loom_bench/quality/tasks/__init__.py`:

```python
from loom_bench.quality.tasks.repeat_word import RepeatWordTask

TASKS: dict[str, TaskFactory] = {
    ...
    "repeat_word": RepeatWordTask.from_params,
}
```

Suites validate every task's `kind` and `params` when they load, so a typo fails at
`load_suite`, at `bench plan` (for experiments with `quality:`) and in the suite tests.

## 4. Add it to the suites

In `bench/evals/<model>.yaml`:

```yaml
  - name: repeat_word
    kind: repeat_word
    threshold: 0.02         # the margin this sample size can decide
    min_samples: 300
    params: {n: 1000, seed: 1234}
```

Choose `n` and `threshold` from the sample-size table in
[quality-gate.md](../quality-gate.md#sample-sizes): the gate never claims less than ±3/n
around the delta, and fewer than `min_samples` items is INCONCLUSIVE. Changing items,
prompts or scoring means bumping `version` and re-running the baseline: gate decisions
compare runs of the same suite only.

## 5. Model-written code

A task that executes model output must refuse to run unless `ctx.allow_code_exec` is
set, and must run programs through `quality/sandbox.py` (`run_python`), as `code_exec`
does. See [security.md](../security.md#code-execution-sandbox) for what that sandbox
does and does not protect.

## 6. Tests and proof

`bench/tests/quality/test_repeat_word.py`: unit-test the scorer on hand-written cases,
and run the task against a fake endpoint. Tests never touch the network.

```python
import json

import httpx

from loom_bench.quality.client import EvalClient
from loom_bench.quality.tasks import build_task
from loom_bench.quality.tasks.base import EvalContext


async def test_scores_against_fake_server(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        word = json.loads(request.content)["messages"][0]["content"].split()[4]
        reply = word if word != "delta" else "gamma"  # one word always wrong
        message = {"role": "assistant", "content": reply}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )

    task = build_task("repeat_word", "repeat", {"n": 60, "seed": 1})
    client = EvalClient("http://t/v1", "m", transport=httpx.MockTransport(handler), max_retries=0)
    async with client:
        out = await task.run(EvalContext(client=client, workdir=tmp_path))
    assert len(out.items) == 60
    assert 0 < sum(i.score for i in out.items) < 60
    assert all(i.content_hash for i in out.items)
```

```bash
uv run pytest -q bench/tests/quality          # includes the suite validation tests
```

Then against a live endpoint (the mock server, or a real engine), only this task:

```bash
uv run bench quality run qwen3-8b --base-url http://127.0.0.1:8000/v1 --model <served model> \
  --only repeat_word --out quality/samples.json
```

The mock backend answers "What is A op B?" correctly, returns JSON sampled from the
`response_format` schema, always calls a tool when tools are offered, and otherwise produces
random words (`mock/content.py`, `mock/server.py`). It checks plumbing, not accuracy: a
task with a new answer format will mostly score 0 against it.
