"""Wrapped load generators: engine-shipped benchmark CLIs run as subprocesses.

vLLM's `vllm bench serve` and SGLang's `python -m sglang.benchmark.serving` drive
the same OpenAI-compatible server the native generator does. A wrapper maps a
Loom workload and load plan onto the tool's flags (`ExternalTool.build_argv`),
runs it, and rebuilds `RequestRecord`s from the tool's per-request result arrays
(`ExternalTool.parse`), so metrics and reports treat every generator alike.

What a tool cannot reproduce is recorded in `RunResult.meta`: `prompts_source`
is "loom" when the tool replays our PreparedRequests from a dataset file and
"tool" when it generates its own prompts (random and shared-prefix workloads);
open-loop arrival times are always sampled by the tool (`arrivals_source`).
Warmup requests are sent by the tool and never appear in its result arrays;
their count is in `meta["tool_warmup_requests"]`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

import numpy as np

from loom_bench.client.openai_stream import Endpoint, PreparedRequest
from loom_bench.jobs import LoadJob
from loom_bench.loadgen.arrivals import ArrivalSpec, parse_arrivals
from loom_bench.loadgen.base import LoadPlan, OpenLoopPlan, RunResult
from loom_bench.records import LoadMode, RequestRecord, RequestStatus
from loom_bench.workloads.profiles import (
    NeedleProfile,
    SharedPrefixProfile,
    SyntheticProfile,
    WorkloadProfile,
    parse_profile,
)

PromptsSource = Literal["loom", "tool"]

API_PATHS: dict[Endpoint, str] = {"chat": "/chat/completions", "completions": "/completions"}

_MAX_ERROR_CHARS = 500
_OUTPUT_TAIL_BYTES = 8192
_REQUIRED_ARRAYS = ("ttfts", "itls", "input_lens", "output_lens", "errors")


class ExternalToolError(RuntimeError):
    """The wrapped tool failed or produced no usable result."""


class UnsupportedByTool(ValueError):
    """The workload or load plan has no faithful equivalent in the tool's flags."""


@dataclass(frozen=True, slots=True)
class WorkloadContext:
    """What a wrapped tool needs that `LoadGenerator.run` does not carry."""

    profile: WorkloadProfile
    arrival: ArrivalSpec | None = None  # open loop only
    seed: int = 0
    engine: str = "vllm"  # server under test; selects SGLang's backend name
    tokenizer: str | None = None  # HF repo or local path the tool loads to size prompts
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_job(cls, job: LoadJob) -> WorkloadContext:
        return cls(
            profile=parse_profile(job.workload),
            arrival=parse_arrivals(job.arrival) if job.arrival else None,
            seed=job.seed,
            engine=job.engine,
            tokenizer=job.tokenizer.repo if job.tokenizer.kind == "hf" else None,
            extra_body=dict(job.extra_body),
        )


@dataclass(frozen=True, slots=True)
class ToolRun:
    """One wrapped run resolved from the load plan; the input to `ExternalTool`."""

    base_url: str  # includes the API prefix, e.g. http://127.0.0.1:8000/v1
    model: str
    context: WorkloadContext
    mode: LoadMode
    load_value: float
    num_prompts: int  # measured requests
    warmup_requests: int  # sent by the tool before measuring
    concurrency: int  # closed loop: workers; open loop: in-flight cap (`max_inflight`)
    prompts_source: PromptsSource
    requests: Sequence[PreparedRequest]  # measured Loom requests in send order ("loom" only)
    workdir: Path
    keep_output: bool = False

    @property
    def profile(self) -> WorkloadProfile:
        return self.context.profile

    @property
    def dataset_path(self) -> Path:
        return self.workdir / "dataset.jsonl"

    @classmethod
    def resolve(
        cls,
        *,
        base_url: str,
        model: str,
        requests: Sequence[PreparedRequest],
        plan: LoadPlan,
        context: WorkloadContext,
        workdir: Path,
        keep_output: bool = False,
    ) -> ToolRun:
        """Measured and warmup counts the native driver would use for `plan`.

        Open loop: arrivals before `warmup_s` become the tool's warmup requests and
        those in [warmup_s, duration_s) its measured prompts; the tool samples its
        own arrival times from `context.arrival`. Closed loop needs `num_requests`
        (the tools have no duration bound). Shared-prefix runs are rounded down to
        a multiple of `num_prefix_groups`, since both tools generate whole groups.
        """
        profile = context.profile
        if isinstance(plan, OpenLoopPlan):
            if context.arrival is None:
                raise ValueError("open loop needs the arrival spec in WorkloadContext.arrival")
            offsets = np.asarray(plan.arrivals, dtype=np.float64)
            warmup = int(np.count_nonzero(offsets < plan.warmup_s))
            n = int(np.count_nonzero((offsets >= plan.warmup_s) & (offsets < plan.duration_s)))
            mode = LoadMode.OPEN_LOOP
            load_value = (
                plan.load_value
                if plan.load_value is not None
                else n / (plan.duration_s - plan.warmup_s)
            )
            concurrency = plan.max_inflight
        else:
            if plan.num_requests is None:
                raise UnsupportedByTool(
                    "wrapped tools stop after a request count, not a duration; "
                    "give the closed-loop plan num_requests"
                )
            warmup, n = plan.warmup_requests, plan.num_requests
            mode = LoadMode.CLOSED_LOOP
            load_value = float(plan.concurrency)
            concurrency = plan.concurrency
        if isinstance(profile, SharedPrefixProfile):
            n -= n % profile.num_prefix_groups
        if n < 1:
            raise ValueError("the plan has no measured requests for the tool to send")

        source: PromptsSource = (
            "tool" if isinstance(profile, SyntheticProfile | SharedPrefixProfile) else "loom"
        )
        measured: Sequence[PreparedRequest] = ()
        if source == "loom":
            measured = requests[warmup : warmup + n]
            if len(measured) < n:
                raise ValueError(f"need {warmup + n} prepared requests, got {len(requests)}")
        return cls(
            base_url=base_url,
            model=model,
            context=context,
            mode=mode,
            load_value=load_value,
            num_prompts=n,
            warmup_requests=warmup,
            concurrency=concurrency,
            prompts_source=source,
            requests=measured,
            workdir=workdir,
            keep_output=keep_output,
        )


class ExternalTool(Protocol):
    """Plug-in slot for a wrapped benchmark CLI (vLLM, SGLang, GuideLLM, GenAI-Perf...)."""

    name: str
    result_file: str  # name of the result file inside the run's workdir

    def build_argv(self, run: ToolRun, result_path: Path) -> list[str]:
        """Command line, without any `command_prefix`. Writes `run.dataset_path`
        when the tool replays Loom prompts. Raises `UnsupportedByTool`."""
        ...

    def parse(self, run: ToolRun, result_path: Path) -> RunResult: ...


class ExternalLoadGenerator:
    """`LoadGenerator` over an `ExternalTool`, run with `asyncio.create_subprocess_exec`.

    The registry builds one without a context; `bind` the job's `WorkloadContext`
    before `run`. `command_prefix` runs the tool elsewhere, e.g. in the engine's
    image on the GPU host: ``["docker", "run", "--rm", "--network", "host", "-v",
    f"{workdir}:{workdir}", "-e", "OPENAI_API_KEY", "--entrypoint", "", image]``.
    Run files live in a temporary directory under `workdir`, which must be
    visible at the same path wherever the tool runs.
    """

    def __init__(
        self,
        tool: ExternalTool,
        *,
        context: WorkloadContext | None = None,
        command_prefix: Sequence[str] = (),
        workdir: Path | None = None,
    ) -> None:
        self.tool = tool
        self.name = tool.name
        self.context = context
        self.command_prefix = list(command_prefix)
        self.workdir = workdir

    def bind(self, context: WorkloadContext) -> ExternalLoadGenerator:
        return ExternalLoadGenerator(
            self.tool, context=context, command_prefix=self.command_prefix, workdir=self.workdir
        )

    async def run(
        self,
        base_url: str,
        requests: Sequence[PreparedRequest],
        plan: LoadPlan,
        *,
        request_timeout_s: float,
        model: str | None = None,
        api_key: str | None = None,
        keep_output: bool = False,
    ) -> RunResult:
        """Run the tool once. `request_timeout_s` cannot be passed on: both tools
        use a fixed 6 h client timeout. The API key reaches the tool as
        `OPENAI_API_KEY` in its environment, never on the command line."""
        if self.context is None:
            raise RuntimeError(
                f"{self.name} needs the job's workload context: "
                "call bind(WorkloadContext.from_job(job)) before run()"
            )
        if model is None:
            raise ValueError(f"{self.name} needs the served model name")
        with tempfile.TemporaryDirectory(
            prefix=f"{self.name}-", dir=self.workdir, ignore_cleanup_errors=True
        ) as tmp:
            run = ToolRun.resolve(
                base_url=base_url,
                model=model,
                requests=requests,
                plan=plan,
                context=self.context,
                workdir=Path(tmp),
                keep_output=keep_output,
            )
            result_path = run.workdir / self.tool.result_file
            argv = [*self.command_prefix, *self.tool.build_argv(run, result_path)]
            env = None if api_key is None else {**os.environ, "OPENAI_API_KEY": api_key}
            await run_command(argv, env=env)
            return self.tool.parse(run, result_path)


async def run_command(
    argv: Sequence[str], *, env: Mapping[str, str] | None = None, stop_grace_s: float = 10.0
) -> None:
    """Run `argv` to completion; raise `ExternalToolError` with the output tail on failure.

    stdout and stderr are merged and only the last few KiB are kept. Cancelling
    terminates the process (SIGTERM, SIGKILL after `stop_grace_s`); SIGTERM lets a
    `docker run` prefix stop its container.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=None if env is None else dict(env),
        )
    except OSError as e:
        raise ExternalToolError(f"cannot start {argv[0]!r}: {e}") from e
    assert proc.stdout is not None
    tail = b""
    try:
        while chunk := await proc.stdout.read(65536):
            tail = (tail + chunk)[-_OUTPUT_TAIL_BYTES:]
        code = await proc.wait()
    except asyncio.CancelledError:
        await _stop(proc, stop_grace_s)
        raise
    if code != 0:
        text = tail.decode("utf-8", errors="replace").strip()
        raise ExternalToolError(f"{argv[0]} exited with status {code}; last output:\n{text}")


async def _stop(proc: asyncio.subprocess.Process, grace_s: float) -> None:
    if proc.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        async with asyncio.timeout(grace_s):
            await proc.wait()
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()


# --- helpers shared by tool wrappers ------------------------------------------


def split_base_url(base_url: str) -> tuple[str, str]:
    """``http://host:8000/v1`` -> (``http://host:8000``, ``/v1``): the tools take the
    server root and append their own API paths."""
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"base_url must be absolute, got {base_url!r}")
    return f"{parts.scheme}://{parts.netloc}", parts.path.rstrip("/")


def ignore_eos(profile: WorkloadProfile) -> bool:
    return False if isinstance(profile, NeedleProfile) else profile.ignore_eos


def shared_prefix_lengths(profile: SharedPrefixProfile, tool: str) -> tuple[int, int, int]:
    """(prefix, unique suffix, output) token lengths of a fixed-length shared-prefix profile."""
    if profile.range_ratio:
        raise UnsupportedByTool(
            f"{tool} cannot jitter shared-prefix lengths like range_ratio does; "
            "use range_ratio: 0 or the native generator"
        )
    prefix = profile.prefix_len
    return prefix, profile.input_len - prefix, profile.output_len


def single_user_prompt(req: PreparedRequest, tool: str) -> str:
    """The prompt text of a completions request or a one-user-message chat request."""
    if req.endpoint == "completions":
        return str(req.payload["prompt"])
    messages = req.payload["messages"]
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise UnsupportedByTool(
            f"{tool} replays one user message per request; {req.request_id} has "
            f"{len(messages)} messages (use the native generator)"
        )
    return str(messages[0]["content"])


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_result(path: Path, tool: str, *, jsonl: bool = False) -> dict[str, Any]:
    """The tool's result object; for JSONL, the last line (the file is per-run)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ExternalToolError(f"{tool} exited cleanly but wrote no result at {path}") from None
    if jsonl:
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            raise ExternalToolError(f"{tool} result file {path} is empty")
        text = lines[-1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ExternalToolError(f"{tool} result is not a JSON object")
    return data


def result_from_arrays(
    run: ToolRun, data: Mapping[str, Any], *, tool: str, meta: Mapping[str, Any] | None = None
) -> RunResult:
    """RequestRecords from per-request result arrays (index i = i-th measured request).

    Required: `ttfts`, `itls`, `input_lens`, `output_lens`, `errors` (seconds and
    token counts). Optional: `start_times` (tool clock, seconds), `latencies`,
    `queue_times` (wait for a `--max-concurrency` slot), `generated_texts`.

    t0 is the earliest arrival (start - queue time); without `start_times` every
    request is placed at t=0. A request succeeded when its error is empty and,
    if latencies are given, its latency is positive. Without latencies the end
    of a request is its last content chunk (ttft + sum of ITLs). The measurement
    window is the tool's own `duration`, so throughput uses its denominator.
    """
    missing = [k for k in _REQUIRED_ARRAYS if k not in data]
    if missing:
        raise ExternalToolError(
            f"{tool} result has no per-request {', '.join(missing)} (detailed output disabled?)"
        )
    ttfts, itls, input_lens, output_lens, errors = (list(data[k]) for k in _REQUIRED_ARRAYS)
    starts, latencies, queues, texts = (
        data.get(k) for k in ("start_times", "latencies", "queue_times", "generated_texts")
    )
    n = len(ttfts)
    arrays: dict[str, Sequence[Any] | None] = {
        "itls": itls,
        "input_lens": input_lens,
        "output_lens": output_lens,
        "errors": errors,
        "start_times": starts,
        "latencies": latencies,
        "queue_times": queues,
    }
    for key, arr in arrays.items():
        if arr is not None and len(arr) != n:
            raise ExternalToolError(f"{tool} result arrays disagree: {n} ttfts, {len(arr)} {key}")
    if n == 0:
        raise ExternalToolError(f"{tool} result has no requests")
    if run.prompts_source == "loom" and n != len(run.requests):
        raise ExternalToolError(f"{tool} returned {n} results for {len(run.requests)} requests")

    t0 = 0.0
    if starts is not None:
        t0 = min(s - (queues[i] if queues is not None else 0.0) for i, s in enumerate(starts))
    profile = run.profile
    records: list[RequestRecord] = []
    for i in range(n):
        latency = None if latencies is None else float(latencies[i])
        ok = not errors[i] and (latency is None or latency > 0)
        sent = 0.0 if starts is None else float(starts[i]) - t0
        ttft = float(ttfts[i] or 0.0)
        gaps = [float(g) for g in itls[i] or ()]
        if latency is not None and latency > 0:
            finished: float | None = sent + latency
        elif latency is None and ok:
            finished = sent + ttft + sum(gaps)
        else:
            finished = None
        status, error = (RequestStatus.OK, None) if ok else _failure(str(errors[i] or ""))
        if run.prompts_source == "loom":
            req = run.requests[i]
            rec = RequestRecord(
                request_id=req.request_id,
                status=status,
                sent_at_s=sent,
                expected_prompt_tokens=req.expected_prompt_tokens,
                max_tokens=req.max_tokens,
                meta=dict(req.meta),
            )
        else:
            rid = f"{profile.name}-{tool}-s{run.context.seed}-{i:06d}"
            rec = RequestRecord(request_id=rid, status=status, sent_at_s=sent)
        if run.mode is LoadMode.OPEN_LOOP and starts is not None and queues is not None:
            rec.scheduled_at_s = sent - float(queues[i])
        rec.first_token_at_s = sent + ttft if ttft > 0 else None
        rec.finished_at_s = finished
        rec.itl_s = gaps
        rec.error = error
        if ok:
            rec.prompt_tokens = int(input_lens[i])
            rec.completion_tokens = int(output_lens[i])
        if run.keep_output and texts is not None:
            rec.output_text = texts[i]
        records.append(rec)

    duration = data.get("duration")
    end = float(duration) if duration else max(r.finished_at_s or r.sent_at_s for r in records)
    return RunResult(
        records=records,
        mode=run.mode,
        load_value=run.load_value,
        t_measure_start_s=0.0,
        t_measure_end_s=end,
        meta={
            "tool": tool,
            "prompts_source": run.prompts_source,
            "arrivals_source": "tool" if run.mode is LoadMode.OPEN_LOOP else None,
            "num_prompts": run.num_prompts,
            "tool_warmup_requests": run.warmup_requests,
            "send_times": starts is not None,
            **(meta or {}),
        },
    )


def tool_summary(data: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """The tool's own aggregates, kept in meta for cross-checking ours."""
    return {k: data[k] for k in keys if k in data}


def _failure(error: str) -> tuple[RequestStatus, str]:
    """Status and a one-line message (the tools store full tracebacks)."""
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    message = lines[-1] if lines else "request failed (no error text from tool)"
    status = RequestStatus.TIMEOUT if "Timeout" in message else RequestStatus.ERROR
    return status, message[:_MAX_ERROR_CHARS]
