import json
import shutil
import sys

import pytest
from ext_support import (
    BASE_URL,
    FAKE_TOOL,
    FIXTURES,
    MODEL,
    TOK,
    TOKENIZER,
    closed_plan,
    open_plan,
    profile,
    read_jsonl,
    sharegpt_profile,
    tool_run,
    trace_profile,
)

from loom_bench.loadgen.base import LOAD_GENERATORS, get_load_generator
from loom_bench.loadgen.external import ExternalLoadGenerator, UnsupportedByTool, WorkloadContext
from loom_bench.loadgen.sglang_bench import SglangBench, random_ids_lengths
from loom_bench.records import LoadMode, RequestStatus
from loom_bench.workloads import build_requests

TOOL = SglangBench()
EXE = ["python3", "-m", "sglang.benchmark.serving"]


def argv(run, tmp_path):
    return TOOL.build_argv(run, tmp_path / "result.jsonl")


def test_synthetic_closed_loop_exact_argv_against_sglang(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(4, 100, 2), tmp_path, engine="sglang")
    assert argv(run, tmp_path) == [
        *EXE,
        "--backend", "sglang-oai",
        "--base-url", "http://127.0.0.1:8000",
        "--model", MODEL,
        "--tokenizer", TOKENIZER,
        "--num-prompts", "100",
        "--seed", "7",
        "--warmup-requests", "2",
        "--request-rate", "inf",
        "--max-concurrency", "4",
        "--dataset-name", "random-ids",
        "--random-input-len", "1024",
        "--random-output-len", "128",
        "--random-range-ratio", "1.0",
        "--extra-request-body", '{"temperature": 0.0}',
        "--output-file", str(tmp_path / "result.jsonl"),
        "--output-details",
        "--disable-tqdm",
    ]  # fmt: skip


@pytest.mark.parametrize(
    "arrival", [{"kind": "poisson", "rate": 4}, {"kind": "gamma", "rate": 4, "burstiness": 1}]
)
def test_open_loop_poisson_chat_against_vllm(tmp_path, arrival):
    prof = profile("synthetic", endpoint="chat", ignore_eos=False, temperature=0.5)
    run, _ = tool_run(prof, open_plan(4.0, 10.0, 2.0), tmp_path, arrival=arrival, tokenizer=None)
    assert argv(run, tmp_path) == [
        *EXE,
        "--backend", "vllm-chat",
        "--base-url", "http://127.0.0.1:8000",
        "--model", MODEL,
        "--num-prompts", "32",
        "--seed", "7",
        "--warmup-requests", "8",
        "--request-rate", "4.0",
        "--max-concurrency", "64",
        "--dataset-name", "random-ids",
        "--random-input-len", "1024",
        "--random-output-len", "128",
        "--random-range-ratio", "1.0",
        "--disable-ignore-eos",
        "--extra-request-body", '{"stream_options": {"include_usage": true}, "temperature": 0.5}',
        "--output-file", str(tmp_path / "result.jsonl"),
        "--output-details",
        "--disable-tqdm",
    ]  # fmt: skip


def test_range_ratio_maps_onto_sglang_bounds():
    p = profile("synthetic", input_len=1000, output_len=100, range_ratio=0.33)
    full_in, full_out, ratio = random_ids_lengths(p)
    assert (full_in, full_out) == (1330, 133)
    # SGLang draws from [max(int(len * ratio), 1), len]; Loom from [int(len*0.67), int(len*1.33)].
    assert abs(int(full_in * ratio) - int(1000 * 0.67)) <= 1
    assert abs(int(full_out * ratio) - int(100 * 0.67)) <= 1


def test_shared_prefix_maps_to_generated_shared_prefix(tmp_path):
    run, _ = tool_run(profile("shared_prefix"), closed_plan(8, 30, 0), tmp_path)
    args = argv(run, tmp_path)
    i = args.index("--dataset-name")
    assert args[i : i + 14] == [
        "--dataset-name", "generated-shared-prefix",
        "--gsp-num-groups", "8",
        "--gsp-prompts-per-group", "3",
        "--gsp-system-prompt-len", "1024",
        "--gsp-question-len", "1024",
        "--gsp-output-len", "128",
        "--gsp-range-ratio", "1.0",
    ]  # fmt: skip
    assert args[args.index("--num-prompts") + 1] == "24"
    assert args[args.index("--backend") + 1] == "vllm-chat"


@pytest.mark.parametrize("kind", ["long_context_needle", "long_generation", "chat", "multi_turn"])
def test_chat_kinds_replayed_through_openai_dataset(tmp_path, kind):
    prof = {
        "long_context_needle": lambda: profile("long_context_needle"),
        "long_generation": lambda: profile("long_generation"),
        "chat": lambda: sharegpt_profile(tmp_path),
        "multi_turn": lambda: sharegpt_profile(tmp_path, multi_turn=True),
    }[kind]()
    run, reqs = tool_run(prof, closed_plan(2, 6, 3), tmp_path)
    args = argv(run, tmp_path)
    i = args.index("--dataset-name")
    assert args[i : i + 4] == [
        "--dataset-name", "openai", "--dataset-path", str(tmp_path / "dataset.jsonl")
    ]  # fmt: skip
    needle = kind == "long_context_needle"
    assert ("--disable-ignore-eos" in args) == needle

    rows = read_jsonl(run.dataset_path)
    native = reqs[3:9]
    assert [row["messages"] for row in rows] == [r.payload["messages"] for r in native]
    for row, req in zip(rows, native, strict=True):
        assert row == {**req.payload, "ignore_eos": not needle}
    if kind == "multi_turn":
        assert len(rows[0]["messages"]) == 3


@pytest.mark.parametrize("kind", ["code_completion", "trace", "long_generation_completions"])
def test_completions_kinds_with_loom_prompts_are_unsupported(tmp_path, kind):
    prof = {
        "code_completion": lambda: profile("code_completion"),
        "trace": lambda: trace_profile(tmp_path),
        "long_generation_completions": lambda: profile("long_generation", endpoint="completions"),
    }[kind]()
    run, _ = tool_run(prof, closed_plan(), tmp_path)
    with pytest.raises(UnsupportedByTool, match="only on the chat endpoint"):
        argv(run, tmp_path)


@pytest.mark.parametrize(
    ("arrival", "match"),
    [
        ({"kind": "gamma", "rate": 4, "burstiness": 0.5}, "not gamma with burstiness != 1"),
        ({"kind": "constant", "rate": 4}, "not constant"),
        ({"kind": "ramp", "rate_start": 1, "rate_end": 8}, "not ramp"),
    ],
)
def test_non_poisson_arrivals_raise(tmp_path, arrival, match):
    run, _ = tool_run(profile("synthetic"), open_plan(), tmp_path, arrival=arrival)
    with pytest.raises(UnsupportedByTool, match=match):
        argv(run, tmp_path)


def test_other_unsupported_options(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(), tmp_path, base_url="http://h/api/v1")
    with pytest.raises(UnsupportedByTool, match="must end in /v1"):
        argv(run, tmp_path)
    run, _ = tool_run(profile("shared_prefix", range_ratio=0.1), closed_plan(8, 8, 0), tmp_path)
    with pytest.raises(UnsupportedByTool, match="range_ratio"):
        argv(run, tmp_path)


def test_extra_body_merged_into_request_body(tmp_path):
    run, _ = tool_run(
        profile("long_generation"), closed_plan(), tmp_path,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )  # fmt: skip
    args = argv(run, tmp_path)
    assert json.loads(args[args.index("--extra-request-body") + 1]) == {
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }


def test_parse_last_jsonl_line_into_records(tmp_path):
    prof = profile("long_generation")
    run, reqs = tool_run(prof, closed_plan(2, 3, 1), tmp_path)
    shutil.copy(FIXTURES / "sglang_openai_closed.jsonl", tmp_path / "result.jsonl")
    res = TOOL.parse(run, tmp_path / "result.jsonl")

    assert (res.mode, res.load_value, res.t_measure_end_s) == (LoadMode.CLOSED_LOOP, 2.0, 0.9)
    a, b, failed = res.records
    assert [r.request_id for r in res.records] == [r.request_id for r in reqs[1:4]]
    assert (a.status, b.status, failed.status) == (RequestStatus.OK,) * 2 + (RequestStatus.ERROR,)
    assert (a.sent_at_s, a.ttft_s) == (0.0, 0.07)
    assert a.e2e_s == pytest.approx(0.07 + 0.02 + 0.023 + 0.021)
    assert a.itl_s == [0.02, 0.023, 0.021]
    assert (a.prompt_tokens, a.completion_tokens) == (80, 4)
    assert b.tpot_s == pytest.approx(0.041 / 2)
    assert (
        failed.error.startswith("Internal Server Error:") and "KV cache exhausted" in failed.error
    )
    assert failed.first_token_at_s is None and failed.finished_at_s is None
    assert res.meta["send_times"] is False
    assert res.meta["output_lens_source"] == "server_usage"
    assert res.meta["tool_summary"]["completed"] == 2


def test_completions_output_lengths_are_requested_not_observed(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(2, 3, 0), tmp_path)
    shutil.copy(FIXTURES / "sglang_openai_closed.jsonl", tmp_path / "result.jsonl")
    res = TOOL.parse(run, tmp_path / "result.jsonl")
    assert res.meta["output_lens_source"] == "requested"
    assert res.meta["prompts_source"] == "tool"
    assert res.records[0].request_id == "synthetic-sglang_bench-s7-000000"


async def test_end_to_end_with_fake_tool(tmp_path, monkeypatch):
    log = tmp_path / "log.json"
    monkeypatch.setenv("FAKE_BENCH_FIXTURE", str(FIXTURES / "sglang_openai_closed.jsonl"))
    monkeypatch.setenv("FAKE_BENCH_LOG", str(log))
    prof = profile("long_generation")
    reqs = build_requests(prof, TOK, 4)
    gen = ExternalLoadGenerator(
        SglangBench(), command_prefix=[sys.executable, str(FAKE_TOOL)], workdir=tmp_path
    ).bind(WorkloadContext(profile=prof, engine="sglang", tokenizer=TOKENIZER))

    res = await gen.run(BASE_URL, reqs, closed_plan(2, 3, 1), request_timeout_s=60, model=MODEL)

    seen = json.loads(log.read_text())
    assert seen["argv"][:5] == [*EXE, "--backend", "sglang-oai-chat"]
    assert seen["api_key"] is None
    rows = [json.loads(line) for line in seen["dataset"].splitlines()]
    assert [r["messages"] for r in rows] == [r.payload["messages"] for r in reqs[1:4]]
    assert [r.status for r in res.records] == [RequestStatus.OK] * 2 + [RequestStatus.ERROR]
    assert res.records[0].output_text is None
    assert list(tmp_path.iterdir()) == [log]


def test_registered():
    assert "sglang_bench" in LOAD_GENERATORS
    gen = get_load_generator("sglang_bench")
    assert isinstance(gen, ExternalLoadGenerator) and isinstance(gen.tool, SglangBench)
    assert gen.name == "sglang_bench" and gen.context is None
