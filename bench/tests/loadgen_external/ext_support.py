"""Shared builders for the wrapped-tool tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loom_bench.client.openai_stream import PreparedRequest
from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.loadgen.arrivals import constant, parse_arrivals
from loom_bench.loadgen.base import ClosedLoopPlan, LoadContext, LoadPlan, OpenLoopPlan
from loom_bench.loadgen.external import ToolRun
from loom_bench.records import LoadMode
from loom_bench.tokenize import SimpleTokenizer
from loom_bench.workloads import build_requests, parse_profile

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
FAKE_TOOL = HERE / "fake_bench_tool.py"
TOK = SimpleTokenizer()
BASE_URL = "http://127.0.0.1:8000/v1"
MODEL = "qwen3-8b"
TOKENIZER = "Qwen/Qwen3-8B"

_COMMON = {"description": "d", "content": "synthetic"}
_PROFILES: dict[str, dict[str, Any]] = {
    "synthetic": {"kind": "synthetic", "input_len": 1024, "output_len": 128},
    "shared_prefix": {
        "kind": "shared_prefix", "input_len": 2048, "output_len": 128,
        "prefix_share": 0.5, "num_prefix_groups": 8,
    },
    "long_context_needle": {"kind": "long_context_needle", "context_len": 256},
    "code_completion": {"kind": "code_completion", "input_len": 64, "output_len": 16},
    "long_generation": {"kind": "long_generation", "input_len": 64, "output_len": 256},
}  # fmt: skip


def profile(kind: str, **overrides: Any):
    return parse_profile(
        {"name": kind.replace("_", "-"), **_COMMON, **_PROFILES[kind], **overrides}
    )


def sharegpt_profile(tmp_path: Path, *, multi_turn: bool = False):
    """A chat_dataset profile over a small ShareGPT file written to `tmp_path`."""
    convs = []
    for i in range(12):
        turns = [{"from": "human", "value": f"question {i} about ships"}]
        if multi_turn:
            turns += [
                {"from": "gpt", "value": f"first answer {i}"},
                {"from": "human", "value": f"follow-up {i}"},
            ]
        turns.append({"from": "gpt", "value": f"reference answer number {i} with words"})
        convs.append({"conversations": turns})
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(convs))
    return parse_profile(
        {"name": "chat", **_COMMON, "content": "realistic", "kind": "chat_dataset",
         "path": str(path)}
    )  # fmt: skip


def trace_profile(tmp_path: Path):
    path = tmp_path / "trace.csv"
    rows = "\n".join(f"{i}.0,Llama,{20 + i},{8 + i},0,Conversation log" for i in range(12))
    path.write_text("Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n" + rows)
    return parse_profile(
        {"name": "trace", **_COMMON, "content": "realistic", "kind": "trace",
         "path": str(path), "format": "burstgpt"}
    )  # fmt: skip


def closed_plan(concurrency: int = 4, num_requests: int = 6, warmup: int = 2) -> ClosedLoopPlan:
    return ClosedLoopPlan(concurrency, num_requests=num_requests, warmup_requests=warmup)


def open_plan(rate: float = 4.0, duration_s: float = 10.0, warmup_s: float = 2.0) -> OpenLoopPlan:
    """Constant arrivals, so counts are exact: rate*warmup_s warmups, the rest measured."""
    return OpenLoopPlan(constant(rate, duration_s), duration_s, warmup_s, 64, 5.0)


def load_context(
    prof,
    plan: LoadPlan,
    *,
    requests: list[PreparedRequest] | None = None,
    arrival: dict[str, Any] | None = None,
    engine: str = "vllm",
    tokenizer: str | None = TOKENIZER,
    extra_body: dict[str, Any] | None = None,
    seed: int = 7,
    base_url: str = BASE_URL,
    keep_output: bool = False,
) -> LoadContext:
    """A LoadContext for `plan`; by default with the requests the plan needs."""
    open_loop = isinstance(plan, OpenLoopPlan)
    if requests is None:
        n = 64 if open_loop else plan.warmup_requests + (plan.num_requests or 0)
        requests = build_requests(prof, TOK, n)
    job = LoadJob(
        run_id="r1",
        base_url=base_url,
        engine=engine,
        served_model=MODEL,
        workload=prof.model_dump(mode="json"),
        tokenizer=(
            TokenizerSpec(kind="hf", repo=tokenizer, revision="0" * 40)
            if tokenizer
            else TokenizerSpec(kind="simple")
        ),
        mode=LoadMode.OPEN_LOOP if open_loop else LoadMode.CLOSED_LOOP,
        load_value=(plan.load_value or 0.0) if open_loop else plan.concurrency,
        arrival=arrival,
        seed=seed,
        keep_output=keep_output,
        extra_body=extra_body or {},
    )
    return LoadContext(
        job=job,
        profile=prof,
        arrival=parse_arrivals(arrival) if arrival else None,
        plan=plan,
        requests=requests,
        tokenizer=TOK,
    )


def tool_run(prof, plan: LoadPlan, tmp_path: Path, **context: Any):
    """(ToolRun, the native requests it was resolved from)."""
    ctx = load_context(prof, plan, **context)
    return ToolRun.resolve(ctx, tmp_path), ctx.requests


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]
