"""`lm_eval`: standard benchmarks through EleutherAI lm-evaluation-harness.

Runs ``python -m lm_eval run`` as a subprocess against the endpoint
(`local-chat-completions` or `local-completions`) with ``--log_samples``, then
reads the per-sample JSONL into ItemResults keyed ``<task>/<doc_id>`` with
lm_eval's `doc_hash` as the content hash. Command line and sample format were
checked against lm_eval 0.4.13 (`lm_eval/_cli/run.py`, `evaluator.py`,
`loggers/evaluation_tracker.py`):

- ``--model_args`` / ``--gen_kwargs`` / ``--metadata`` accept a JSON object;
- ``--limit N`` takes the first N docs of every (sub)task; ``--samples`` takes
  a JSON map of task -> doc indices and excludes ``--limit``. The harness maps
  logged doc_ids back through the index list in the order given, so indices
  are passed sorted;
- ``--seed S`` sets the python, numpy, torch and fewshot seeds;
- tasks that execute code (humaneval, mbpp) need ``--confirm_run_unsafe_code``
  and ``HF_ALLOW_CODE_EVAL=1``; both are only passed when the eval context
  allows code execution;
- each sample row holds doc_id, doc, resps, filtered_resps, filter, metrics,
  doc_hash, prompt_hash, target_hash and one key per metric. Tasks with
  several filters (gsm8k: strict-match / flexible-extract) log one row per
  filter, so `filter` must pick one.

lm_eval is the optional `lmeval` extra (``uv sync --all-packages --extra lmeval``). RULER
tasks also need `transformers` for their tokenizer, humaneval needs
`evaluate`. The working directory keeps lm_eval's raw samples (prompts and
outputs) and its log; nothing from them is logged by Loom.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import Field, model_validator

from loom_bench.quality.tasks.base import (
    Completion,
    EvalContext,
    ItemResult,
    OutputKind,
    ParamTask,
    TaskOutput,
    TaskParams,
)

VERIFIED_VERSION = "0.4.13"
_SAMPLES_RE = re.compile(r"^samples_(?P<task>.+)_(?P<date>\d{4}-\d{2}-\d{2}T[\d.\-]+)\.jsonl$")
_MODEL = {"chat": "local-chat-completions", "completions": "local-completions"}
_PATH = {"chat": "/chat/completions", "completions": "/completions"}
_LOG_TAIL_LINES = 20


class LmEvalNotInstalled(RuntimeError):
    pass


class LmEvalError(RuntimeError):
    pass


class LmEvalParams(TaskParams):
    tasks: Annotated[tuple[str, ...], Field(min_length=1)]
    api: Literal["chat", "completions"] = "chat"
    # Pinned subset: the first `limit` docs of every (sub)task, or explicit indices.
    limit: Annotated[int, Field(ge=1)] | None = None
    samples: dict[str, list[Annotated[int, Field(ge=0)]]] | None = None
    num_fewshot: Annotated[int, Field(ge=0)] | None = None
    apply_chat_template: bool = True
    fewshot_as_multiturn: bool | None = None
    # Per-sample score: row[metric], or row[str(doc[metric_from_doc])] (RULER keys its
    # metric by the sample's sequence length).
    metric: str | None = None
    metric_from_doc: str | None = None
    filter: str | None = None
    gen_kwargs: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_args: dict[str, Any] = Field(default_factory=dict)
    num_concurrent: Annotated[int, Field(ge=1)] = 16
    max_retries: Annotated[int, Field(ge=0)] = 3
    unsafe_code: bool = False
    output_kind: OutputKind = OutputKind.TEXT
    timeout_s: Annotated[float, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.metric is None) == (self.metric_from_doc is None):
            raise ValueError("set exactly one of metric and metric_from_doc")
        if self.limit is not None and self.samples is not None:
            raise ValueError("limit and samples are mutually exclusive in lm_eval")
        if self.api == "chat" and not self.apply_chat_template:
            raise ValueError("local-chat-completions needs apply_chat_template")
        reserved = {"model", "base_url", "num_concurrent", "max_retries", "seed"}
        if reserved & self.model_args.keys():
            raise ValueError(f"model_args may not set {sorted(reserved & self.model_args.keys())}")
        return self


def _local_tokenizer(
    args: Mapping[str, Any], local_tokenizers: Mapping[str, str] | None
) -> dict[str, Any]:
    local = local_tokenizers or {}
    tokenizer = args.get("tokenizer")
    if isinstance(tokenizer, str) and tokenizer in local:
        return {**args, "tokenizer": local[tokenizer]}
    return dict(args)


def build_command(
    params: LmEvalParams,
    *,
    base_url: str,
    model: str,
    seed: int,
    output_path: Path,
    allow_code_exec: bool = False,
    extra_body: dict[str, Any] | None = None,
    local_tokenizers: Mapping[str, str] | None = None,
    python: str = sys.executable,
) -> list[str]:
    """The `lm_eval run` argv for one task entry. The API key goes in the env, never here.

    `extra_body` (the client's per-request extras, e.g. chat_template_kwargs) is
    merged under `gen_kwargs` for the chat API, which forwards gen_kwargs into
    every request body. A `tokenizer` in `model_args` or `metadata` (RULER reads
    `model_args | metadata`) naming a repo in `local_tokenizers` is replaced by its
    local snapshot directory, which transformers loads without the Hub.
    """
    if params.unsafe_code and not allow_code_exec:
        raise LmEvalError(
            f"{','.join(params.tasks)} executes model-written code; "
            "it needs allow_code_exec=True (inside an isolated container or VM)"
        )
    model_args = {
        "model": model,
        "base_url": base_url.rstrip("/") + _PATH[params.api],
        "num_concurrent": params.num_concurrent,
        "max_retries": params.max_retries,
        "seed": seed,
        **_local_tokenizer(params.model_args, local_tokenizers),
    }
    argv = [
        python, "-m", "lm_eval", "run",
        "--model", _MODEL[params.api],
        "--model_args", json.dumps(model_args, sort_keys=True),
        "--tasks", ",".join(params.tasks),
        "--output_path", str(output_path),
        "--log_samples",
        "--seed", str(seed),
    ]  # fmt: skip
    if params.apply_chat_template:
        argv.append("--apply_chat_template")
    if params.fewshot_as_multiturn is not None:
        argv += ["--fewshot_as_multiturn", str(params.fewshot_as_multiturn).lower()]
    if params.num_fewshot is not None:
        argv += ["--num_fewshot", str(params.num_fewshot)]
    if params.limit is not None:
        argv += ["--limit", str(params.limit)]
    if params.samples is not None:
        ordered = {task: sorted(set(idx)) for task, idx in sorted(params.samples.items())}
        argv += ["--samples", json.dumps(ordered)]
    gen_kwargs = (
        {**(extra_body or {}), **params.gen_kwargs} if params.api == "chat" else (params.gen_kwargs)
    )
    if gen_kwargs:
        argv += ["--gen_kwargs", json.dumps(gen_kwargs, sort_keys=True)]
    if params.metadata:
        metadata = _local_tokenizer(params.metadata, local_tokenizers)
        argv += ["--metadata", json.dumps(metadata, sort_keys=True)]
    if params.unsafe_code:
        argv.append("--confirm_run_unsafe_code")
    return argv


def _score(value: Any, where: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        score = float(value)
    elif (
        isinstance(value, list) and value and all(isinstance(v, bool | int | float) for v in value)
    ):
        score = sum(float(v) for v in value) / len(value)  # e.g. IFEval instruction-level
    else:
        raise LmEvalError(f"{where}: metric value is not a score: {type(value).__name__}")
    if not 0.0 <= score <= 1.0:
        raise LmEvalError(f"{where}: score {score} outside [0, 1]")
    return score


def _response_text(row: dict[str, Any]) -> str:
    resps = row.get("resps") or []
    first = resps[0] if resps else ""
    while isinstance(first, list):
        first = first[0] if first else ""
    return first if isinstance(first, str) else ""


def parse_samples(
    files: Iterable[Path], params: LmEvalParams
) -> tuple[list[ItemResult], list[Completion]]:
    """ItemResults (and outputs for sanity checks) from lm_eval sample JSONL files."""
    items: dict[str, ItemResult] = {}
    completions: list[Completion] = []
    filters: dict[str, set[str]] = {}
    for path in sorted(files):
        m = _SAMPLES_RE.match(path.name)
        if m is None:
            raise LmEvalError(f"unexpected samples file name {path.name!r}")
        task = m["task"]
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                filters.setdefault(task, set()).add(row.get("filter", "none"))
                if params.filter is not None and row.get("filter") != params.filter:
                    continue
                item_id = f"{task}/{row['doc_id']}"
                where = f"{path.name}:{line_no}"
                key = params.metric
                if params.metric_from_doc is not None:
                    key = str(row["doc"][params.metric_from_doc])
                if key not in row:
                    raise LmEvalError(f"{where}: no metric {key!r} in sample")
                if item_id in items:
                    raise LmEvalError(
                        f"{task}: doc {row['doc_id']} appears twice; "
                        f"set `filter` to one of {sorted(filters[task])}"
                    )
                items[item_id] = ItemResult(
                    item_id=item_id,
                    score=_score(row[key], where),
                    content_hash=row.get("doc_hash"),
                    meta={"task": task, "doc_id": row["doc_id"]},
                )
                completions.append(
                    Completion(item_id, _response_text(row), None, kind=params.output_kind)
                )
    if params.filter is not None:
        missing = [t for t, fs in filters.items() if params.filter not in fs]
        if missing:
            raise LmEvalError(f"filter {params.filter!r} not logged for {missing}")
    return list(items.values()), completions


def parse_results(files: Iterable[Path]) -> dict[str, Any]:
    """lm_eval version, task versions and sample counts from results_*.json files."""
    out: dict[str, Any] = {"task_versions": {}, "n_samples": {}}
    for path in sorted(files):
        data = json.loads(path.read_text(encoding="utf-8"))
        out["lm_eval_version"] = data.get("lm_eval_version")
        out["task_versions"].update(data.get("versions", {}))
        out["n_samples"].update(data.get("n-samples", {}))
        out.setdefault("task_hashes", {}).update(data.get("task_hashes", {}))
    return out


def ensure_installed() -> None:
    if importlib.util.find_spec("lm_eval") is None:
        raise LmEvalNotInstalled(
            "lm-evaluation-harness is not installed; install the optional extra with "
            "`uv sync --all-packages --extra lmeval` (or `pip install 'loom-bench[lmeval]'`), "
            f"verified with lm_eval=={VERIFIED_VERSION}"
        )


class LmEvalTask(ParamTask[LmEvalParams]):
    Params = LmEvalParams
    version: ClassVar[str] = "1"

    def planned_items(self) -> int | None:
        """Known only for explicit `samples`: doc counts live in the harness's datasets."""
        if self.params.samples is None:
            return None
        return sum(len(set(idx)) for idx in self.params.samples.values())

    async def run(self, ctx: EvalContext) -> TaskOutput:
        ensure_installed()
        out_dir = ctx.workdir / f"lm_eval-{self.name}"
        if out_dir.exists() and any(out_dir.iterdir()):
            raise LmEvalError(f"{out_dir} is not empty; use a fresh workdir per run")
        out_dir.mkdir(parents=True, exist_ok=True)
        argv = build_command(
            self.params,
            base_url=ctx.client.base_url,
            model=ctx.client.model,
            seed=ctx.client.seed,
            output_path=out_dir,
            allow_code_exec=ctx.allow_code_exec,
            extra_body=ctx.client.extra_body,
            local_tokenizers=ctx.local_tokenizers,
        )
        env = dict(os.environ)
        if ctx.client.api_key:
            env["OPENAI_API_KEY"] = ctx.client.api_key
        if self.params.unsafe_code:
            env["HF_ALLOW_CODE_EVAL"] = "1"
        log_path = out_dir / "lm_eval.log"
        with log_path.open("wb") as log:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env, cwd=out_dir
            )
            try:
                code = await asyncio.wait_for(proc.wait(), self.params.timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise LmEvalError(f"{self.name}: lm_eval timed out; log at {log_path}") from None
        if code != 0:
            tail = log_path.read_text(errors="replace").splitlines()[-_LOG_TAIL_LINES:]
            raise LmEvalError(f"{self.name}: lm_eval exited {code}:\n" + "\n".join(tail))
        sample_files = list(out_dir.rglob("samples_*.jsonl"))
        if not sample_files:
            raise LmEvalError(f"{self.name}: lm_eval wrote no samples under {out_dir}")
        items, completions = parse_samples(sample_files, self.params)
        info = parse_results(out_dir.rglob("results_*.json"))
        info["command"] = argv[2:]
        versions = info["task_versions"]
        version = ";".join(
            [f"lm_eval={info.get('lm_eval_version')}"]
            + [f"{t}={versions[t]}" for t in sorted(versions)]
        )
        return TaskOutput(
            items=items, completions=completions, provenance={"lm_eval": info}, version=version
        )
