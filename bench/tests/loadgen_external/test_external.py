import asyncio
import json
import os
import sys

import pytest
from ext_support import (
    BASE_URL,
    MODEL,
    TOK,
    TOKENIZER,
    closed_plan,
    open_plan,
    profile,
    tool_run,
)

from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.loadgen.arrivals import GammaArrivals
from loom_bench.loadgen.base import ClosedLoopPlan
from loom_bench.loadgen.external import (
    ExternalLoadGenerator,
    ExternalToolError,
    ToolRun,
    UnsupportedByTool,
    WorkloadContext,
    result_from_arrays,
    run_command,
    split_base_url,
)
from loom_bench.records import LoadMode, RequestStatus
from loom_bench.workloads import build_requests

# --- plan resolution ----------------------------------------------------------


def test_open_loop_splits_warmup_and_measured_arrivals(tmp_path):
    prof = profile("long_generation")
    run, reqs = tool_run(
        prof, open_plan(4.0, 10.0, 2.0), tmp_path, arrival={"kind": "poisson", "rate": 4}
    )
    assert (run.mode, run.warmup_requests, run.num_prompts) == (LoadMode.OPEN_LOOP, 8, 32)
    assert run.load_value == 4.0  # measured / (duration - warmup) when the plan gives no label
    assert run.concurrency == 64  # max_inflight
    assert run.prompts_source == "loom"
    assert [r.request_id for r in run.requests] == [r.request_id for r in reqs[8:40]]


def test_closed_loop_slices_after_warmup(tmp_path):
    run, reqs = tool_run(profile("code_completion"), closed_plan(3, 5, 2), tmp_path)
    assert (run.mode, run.load_value, run.concurrency) == (LoadMode.CLOSED_LOOP, 3.0, 3)
    assert (run.warmup_requests, run.num_prompts) == (2, 5)
    assert list(run.requests) == reqs[2:7]


@pytest.mark.parametrize("kind", ["synthetic", "shared_prefix"])
def test_tool_generated_prompts_carry_no_loom_requests(tmp_path, kind):
    run, _ = tool_run(profile(kind), closed_plan(num_requests=16), tmp_path)
    assert run.prompts_source == "tool" and run.requests == ()


def test_shared_prefix_rounds_down_to_whole_groups(tmp_path):
    run, _ = tool_run(profile("shared_prefix"), closed_plan(num_requests=30), tmp_path)
    assert run.num_prompts == 24
    with pytest.raises(ValueError, match="no measured requests"):
        tool_run(profile("shared_prefix"), closed_plan(num_requests=5), tmp_path)


def test_resolve_rejects_unrepresentable_plans(tmp_path):
    with pytest.raises(UnsupportedByTool, match="num_requests"):
        tool_run(profile("synthetic"), ClosedLoopPlan(4, duration_s=30.0), tmp_path)
    with pytest.raises(ValueError, match="arrival spec"):
        tool_run(profile("synthetic"), open_plan(), tmp_path)


def test_resolve_needs_enough_prepared_requests(tmp_path):
    prof = profile("code_completion")
    ctx = WorkloadContext(profile=prof)
    with pytest.raises(ValueError, match="need 7 prepared requests, got 3"):
        ToolRun.resolve(
            base_url=BASE_URL, model=MODEL, requests=build_requests(prof, TOK, 3),
            plan=closed_plan(2, 5, 2), context=ctx, workdir=tmp_path,
        )  # fmt: skip


def test_context_from_job():
    job = LoadJob(
        run_id="r1", base_url=BASE_URL, engine="sglang", served_model=MODEL,
        loadgen="sglang_bench", workload=profile("synthetic").model_dump(mode="json"),
        tokenizer=TokenizerSpec(kind="hf", repo=TOKENIZER, revision="abc"),
        mode=LoadMode.OPEN_LOOP, load_value=2.0,
        arrival={"kind": "gamma", "rate": 2.0, "burstiness": 0.5}, seed=3,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )  # fmt: skip
    ctx = WorkloadContext.from_job(job)
    assert ctx.profile == profile("synthetic")
    assert ctx.arrival == GammaArrivals(rate=2.0, burstiness=0.5)
    assert (ctx.seed, ctx.engine, ctx.tokenizer) == (3, "sglang", TOKENIZER)
    assert ctx.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    simple = job.model_copy(update={"tokenizer": TokenizerSpec(kind="simple"), "arrival": None})
    assert WorkloadContext.from_job(simple).tokenizer is None
    assert WorkloadContext.from_job(simple).arrival is None


def test_split_base_url():
    assert split_base_url("http://h:8000/v1/") == ("http://h:8000", "/v1")
    assert split_base_url("https://h/api/v1") == ("https://h", "/api/v1")
    with pytest.raises(ValueError, match="absolute"):
        split_base_url("/v1")


# --- result parsing -----------------------------------------------------------


def _arrays(**over):
    data = {
        "duration": 2.0,
        "ttfts": [0.1, 0.2],
        "itls": [[0.01, 0.02], [0.03]],
        "input_lens": [10, 12],
        "output_lens": [3, 2],
        "errors": ["", ""],
    }
    return {**data, **over}


def test_open_loop_records_scheduled_time_from_queue_time(tmp_path):
    run, _ = tool_run(
        profile("long_generation"), open_plan(), tmp_path, arrival={"kind": "poisson", "rate": 4}
    )
    data = _arrays(
        ttfts=[0.1] * 32, itls=[[0.01]] * 32, input_lens=[10] * 32, output_lens=[2] * 32,
        errors=[""] * 32, latencies=[0.2] * 32,
        start_times=[100.0 + i * 0.25 for i in range(32)],
        queue_times=[0.0] * 31 + [0.05],
    )  # fmt: skip
    res = result_from_arrays(run, data, tool="t")
    recs = res.records
    assert recs[0].sent_at_s == 0.0 and recs[0].scheduled_at_s == 0.0
    assert recs[31].sent_at_s == pytest.approx(7.75)
    assert recs[31].queue_delay_s == pytest.approx(0.05)
    assert res.meta["arrivals_source"] == "tool" and res.meta["send_times"] is True


def test_without_send_times_requests_start_at_zero_and_end_at_last_chunk(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(num_requests=2), tmp_path)
    res = result_from_arrays(run, _arrays(), tool="t")
    a, b = res.records
    assert (a.sent_at_s, a.first_token_at_s, a.finished_at_s) == (0.0, 0.1, pytest.approx(0.13))
    assert b.e2e_s == pytest.approx(0.23)
    assert a.request_id == "synthetic-t-s7-000000" and a.expected_prompt_tokens is None
    assert a.scheduled_at_s is None
    assert (res.t_measure_start_s, res.t_measure_end_s) == (0.0, 2.0)
    assert res.meta["send_times"] is False and res.meta["arrivals_source"] is None


def test_parse_rejects_incomplete_or_inconsistent_results(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(num_requests=2), tmp_path)
    data = _arrays()
    del data["itls"]
    with pytest.raises(ExternalToolError, match="no per-request itls"):
        result_from_arrays(run, data, tool="t")
    with pytest.raises(ExternalToolError, match="2 ttfts, 1 errors"):
        result_from_arrays(run, _arrays(errors=[""]), tool="t")
    loom_run, _ = tool_run(profile("code_completion"), closed_plan(num_requests=3), tmp_path)
    with pytest.raises(ExternalToolError, match="2 results for 3 requests"):
        result_from_arrays(loom_run, _arrays(), tool="t")


# --- subprocess runner --------------------------------------------------------


async def test_run_command_reports_exit_status_and_output_tail():
    script = "import sys; print('x' * 20000); print('boom: bad flag', file=sys.stderr); sys.exit(2)"
    with pytest.raises(ExternalToolError) as err:
        await run_command([sys.executable, "-c", script])
    msg = str(err.value)
    assert "exited with status 2" in msg and msg.endswith("boom: bad flag")
    assert len(msg) < 9000


async def test_run_command_missing_executable():
    with pytest.raises(ExternalToolError, match="cannot start"):
        await run_command(["/nonexistent/loom-bench-tool"])


async def test_cancelling_run_command_terminates_the_tool(tmp_path):
    pid_file = tmp_path / "pid"
    script = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    task = asyncio.create_task(run_command([sys.executable, "-c", script], stop_grace_s=5))
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


# --- generator ----------------------------------------------------------------


class _EchoTool:
    """`python -c` writing a fixed result; records the runs it was given."""

    name = "echo"
    result_file = "out.json"

    def __init__(self):
        self.seen = []

    def build_argv(self, run, result_path):
        self.seen.append((run, result_path))
        data = json.dumps(
            _arrays(ttfts=[0.1], itls=[[]], input_lens=[5], output_lens=[1], errors=[""])
        )
        code = f"import pathlib; pathlib.Path({str(result_path)!r}).write_text({data!r})"
        return ["-c", code]

    def parse(self, run, result_path):
        return result_from_arrays(run, json.loads(result_path.read_text()), tool=self.name)


async def test_generator_runs_tool_under_prefix_in_temporary_workdir(tmp_path):
    tool = _EchoTool()
    gen = ExternalLoadGenerator(tool, command_prefix=[sys.executable], workdir=tmp_path)
    with pytest.raises(RuntimeError, match="bind"):
        await gen.run(BASE_URL, [], closed_plan(), request_timeout_s=60, model=MODEL)
    bound = gen.bind(WorkloadContext(profile=profile("synthetic")))
    assert bound.command_prefix == [sys.executable] and bound.workdir == tmp_path
    with pytest.raises(ValueError, match="served model"):
        await bound.run(BASE_URL, [], closed_plan(), request_timeout_s=60)

    plan = closed_plan(num_requests=1, warmup=0)
    res = await bound.run(BASE_URL, [], plan, request_timeout_s=60, model=MODEL)
    assert [r.status for r in res.records] == [RequestStatus.OK]
    run, result_path = tool.seen[0]
    assert run.model == MODEL and run.workdir.parent == tmp_path
    assert result_path == run.workdir / "out.json"
    assert not run.workdir.exists()  # run files are removed afterwards


def test_failure_status_mapping(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(num_requests=2), tmp_path)
    data = _arrays(errors=["Traceback...\n  line\nasyncio.exceptions.TimeoutError\n", ""])
    data["latencies"] = [0.0, 0.0]  # vLLM: latency 0 means failed even with an empty error
    a, b = result_from_arrays(run, data, tool="t").records
    assert (a.status, a.error) == (RequestStatus.TIMEOUT, "asyncio.exceptions.TimeoutError")
    assert b.status is RequestStatus.ERROR and "no error text" in b.error
    assert a.prompt_tokens is None and a.completion_tokens is None and a.finished_at_s is None
