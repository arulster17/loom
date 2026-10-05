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
from loom_bench.loadgen.external import (
    ExternalLoadGenerator,
    ExternalToolError,
    UnsupportedByTool,
    WorkloadContext,
)
from loom_bench.loadgen.vllm_bench import VllmBench
from loom_bench.records import LoadMode, RequestStatus
from loom_bench.workloads import build_requests

TOOL = VllmBench()
TAIL = [
    "--percentile-metrics", "ttft,tpot,itl,e2el",
    "--metric-percentiles", "50,90,95,99",
    "--save-result", "--save-detailed",
]  # fmt: skip


def argv(run, tmp_path):
    return TOOL.build_argv(run, tmp_path / "result.json")


def test_synthetic_closed_loop_exact_argv(tmp_path):
    run, _ = tool_run(profile("synthetic"), closed_plan(4, 100, 2), tmp_path)
    assert argv(run, tmp_path) == [
        "vllm", "bench", "serve",
        "--backend", "openai",
        "--base-url", "http://127.0.0.1:8000",
        "--endpoint", "/v1/completions",
        "--model", MODEL,
        "--tokenizer", TOKENIZER,
        "--num-prompts", "100",
        "--seed", "7",
        "--num-warmups", "2",
        "--request-rate", "inf",
        "--max-concurrency", "4",
        "--dataset-name", "random",
        "--random-input-len", "1024",
        "--random-output-len", "128",
        "--random-range-ratio", "0.0",
        "--ignore-eos",
        "--temperature", "0.0",
        *TAIL,
        "--result-filename", str(tmp_path / "result.json"),
        "--disable-tqdm",
    ]  # fmt: skip
    assert not run.dataset_path.exists()


@pytest.mark.parametrize(
    ("arrival", "rate", "burstiness"),
    [
        ({"kind": "poisson", "rate": 4}, "4.0", "1.0"),
        ({"kind": "gamma", "rate": 4, "burstiness": 0.5}, "4.0", "0.5"),
        ({"kind": "constant", "rate": 4}, "4.0", "inf"),
    ],
)
def test_open_loop_arrivals(tmp_path, arrival, rate, burstiness):
    prof = profile("synthetic", endpoint="chat", range_ratio=0.25, temperature=0.7)
    run, _ = tool_run(prof, open_plan(4.0, 10.0, 2.0), tmp_path, arrival=arrival, tokenizer=None)
    assert argv(run, tmp_path) == [
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", "http://127.0.0.1:8000",
        "--endpoint", "/v1/chat/completions",
        "--model", MODEL,
        "--num-prompts", "32",
        "--seed", "7",
        "--num-warmups", "8",
        "--request-rate", rate,
        "--burstiness", burstiness,
        "--max-concurrency", "64",
        "--dataset-name", "random",
        "--random-input-len", "1024",
        "--random-output-len", "128",
        "--random-range-ratio", "0.25",
        "--ignore-eos",
        "--temperature", "0.7",
        *TAIL,
        "--result-filename", str(tmp_path / "result.json"),
        "--disable-tqdm",
    ]  # fmt: skip


def test_shared_prefix_maps_to_prefix_repetition(tmp_path):
    run, _ = tool_run(profile("shared_prefix", ignore_eos=False), closed_plan(8, 30, 0), tmp_path)
    args = argv(run, tmp_path)
    i = args.index("--dataset-name")
    assert args[i : i + 10] == [
        "--dataset-name", "prefix_repetition",
        "--prefix-repetition-prefix-len", "1024",
        "--prefix-repetition-suffix-len", "1024",
        "--prefix-repetition-num-prefixes", "8",
        "--prefix-repetition-output-len", "128",
    ]  # fmt: skip
    assert args[args.index("--num-prompts") + 1] == "24"
    assert args[args.index("--endpoint") + 1] == "/v1/chat/completions"
    assert "--ignore-eos" not in args


def _loom_cases(tmp_path):
    return {
        "long_context_needle": profile("long_context_needle"),
        "code_completion": profile("code_completion"),
        "long_generation": profile("long_generation", endpoint="completions"),
        "chat_dataset": sharegpt_profile(tmp_path),
        "trace": trace_profile(tmp_path),
    }


@pytest.mark.parametrize(
    "kind", ["long_context_needle", "code_completion", "long_generation", "chat_dataset", "trace"]
)
def test_loom_prompts_replayed_through_custom_dataset(tmp_path, kind):
    prof = _loom_cases(tmp_path)[kind]
    run, reqs = tool_run(prof, closed_plan(2, 6, 3), tmp_path)
    args = argv(run, tmp_path)
    i = args.index("--dataset-name")
    assert args[i : i + 9] == [
        "--dataset-name", "custom",
        "--dataset-path", str(tmp_path / "dataset.jsonl"),
        "--custom-output-len", "-1",
        "--skip-chat-template",
        "--disable-shuffle",
        "--no-oversample",
    ]  # fmt: skip
    endpoint = prof.endpoint
    assert (
        args[args.index("--endpoint") + 1]
        == f"/v1/{'chat/' if endpoint == 'chat' else ''}completions"
    )
    assert ("--ignore-eos" in args) == (kind != "long_context_needle")

    rows = read_jsonl(run.dataset_path)
    native = reqs[3:9]  # what the native driver sends after 3 warmup requests
    assert len(rows) == 6
    for row, req in zip(rows, native, strict=True):
        text = (
            req.payload["prompt"]
            if endpoint == "completions"
            else req.payload["messages"][0]["content"]
        )
        assert row == {"prompt": text, "output_tokens": req.payload["max_tokens"]}
    assert run.prompts_source == "loom"


def test_multi_turn_chat_is_unsupported(tmp_path):
    run, _ = tool_run(sharegpt_profile(tmp_path, multi_turn=True), closed_plan(), tmp_path)
    with pytest.raises(UnsupportedByTool, match=r"one user message per request.*3 messages"):
        argv(run, tmp_path)


@pytest.mark.parametrize(
    "arrival",
    [
        {"kind": "ramp", "rate_start": 1, "rate_end": 8},
        {"kind": "diurnal", "mean_rate": 4, "amplitude": 0.5, "period_s": 60},
        {"kind": "onoff_burst", "rate_hi": 8, "rate_lo": 1, "t_on_s": 5, "t_off_s": 5},
        {"kind": "trace", "path": "trace.csv", "format": "azure"},
    ],
)
def test_unrepresentable_arrivals_raise(tmp_path, arrival):
    run, _ = tool_run(profile("synthetic"), open_plan(), tmp_path, arrival=arrival)
    with pytest.raises(UnsupportedByTool, match=f"not {arrival['kind']}; use the native"):
        argv(run, tmp_path)


def test_unsupported_workload_options(tmp_path):
    run, _ = tool_run(profile("synthetic", ignore_eos=False), closed_plan(), tmp_path)
    with pytest.raises(UnsupportedByTool, match="always ignores EOS"):
        argv(run, tmp_path)
    run, _ = tool_run(profile("shared_prefix", range_ratio=0.1), closed_plan(8, 8, 0), tmp_path)
    with pytest.raises(UnsupportedByTool, match="range_ratio"):
        argv(run, tmp_path)


def test_extra_body_and_api_prefix(tmp_path):
    run, _ = tool_run(
        profile("long_context_needle"), closed_plan(), tmp_path,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        base_url="https://gw.example/api/v1/",
    )  # fmt: skip
    args = argv(run, tmp_path)
    assert args[args.index("--base-url") + 1] == "https://gw.example"
    assert args[args.index("--endpoint") + 1] == "/api/v1/chat/completions"
    body = args[args.index("--extra-body") + 1]
    assert json.loads(body) == {"chat_template_kwargs": {"enable_thinking": False}}


def test_parse_fixture_into_records(tmp_path):
    prof = profile("long_context_needle")
    run, reqs = tool_run(prof, closed_plan(2, 5, 1), tmp_path)
    shutil.copy(FIXTURES / "vllm_custom_closed.json", tmp_path / "result.json")
    res = TOOL.parse(run, tmp_path / "result.json")

    assert (res.mode, res.load_value) == (LoadMode.CLOSED_LOOP, 2.0)
    assert (res.t_measure_start_s, res.t_measure_end_s) == (0.0, 0.75)
    recs = res.records
    assert [r.request_id for r in recs] == [r.request_id for r in reqs[1:6]]
    assert [r.status for r in recs] == [RequestStatus.OK] * 3 + [
        RequestStatus.ERROR,
        RequestStatus.TIMEOUT,
    ]
    ok = recs[2]
    assert ok.sent_at_s == pytest.approx(0.3)
    assert ok.ttft_s == pytest.approx(0.045) and ok.e2e_s == pytest.approx(0.104)
    assert ok.itl_s == [0.018, 0.02, 0.021]
    assert (ok.prompt_tokens, ok.completion_tokens) == (71, 4)
    assert ok.expected_prompt_tokens == reqs[3].expected_prompt_tokens
    assert ok.max_tokens == 32 and ok.meta == reqs[3].meta
    assert ok.scheduled_at_s is None and ok.output_text is None
    assert recs[0].tpot_s == pytest.approx((0.11 - 0.05) / 3)

    bad, timed_out = recs[3], recs[4]
    assert bad.error == "Bad Request" and bad.first_token_at_s is None
    assert bad.finished_at_s is None and bad.prompt_tokens is None
    assert timed_out.error == "TimeoutError"
    assert timed_out.first_token_at_s == pytest.approx(0.6 + 0.05)

    assert res.meta["prompts_source"] == "loom" and res.meta["tool"] == "vllm_bench"
    assert res.meta["tool_warmup_requests"] == 1 and res.meta["send_times"] is True
    assert res.meta["tool_summary"]["failed"] == 2


async def test_end_to_end_with_fake_tool(tmp_path, monkeypatch):
    log = tmp_path / "log.json"
    monkeypatch.setenv("FAKE_BENCH_FIXTURE", str(FIXTURES / "vllm_custom_closed.json"))
    monkeypatch.setenv("FAKE_BENCH_LOG", str(log))
    prof = profile("long_context_needle")
    reqs = build_requests(prof, TOK, 6)
    gen = ExternalLoadGenerator(
        VllmBench(), command_prefix=[sys.executable, str(FAKE_TOOL)], workdir=tmp_path
    ).bind(WorkloadContext(profile=prof, seed=7, tokenizer=TOKENIZER))

    res = await gen.run(
        BASE_URL, reqs, closed_plan(2, 5, 1), request_timeout_s=60, model=MODEL,
        api_key="sk-test", keep_output=True,
    )  # fmt: skip

    seen = json.loads(log.read_text())
    assert seen["api_key"] == "sk-test" and "sk-test" not in seen["argv"]
    assert seen["argv"][:3] == ["vllm", "bench", "serve"]
    rows = [json.loads(line) for line in seen["dataset"].splitlines()]
    assert [r["prompt"] for r in rows] == [r.payload["messages"][0]["content"] for r in reqs[1:6]]
    assert [r.request_id for r in res.records] == [r.request_id for r in reqs[1:6]]
    assert res.records[0].output_text == "harbor violet tiger lantern"
    assert list(tmp_path.iterdir()) == [log]  # run directory cleaned up


async def test_tool_failure_surfaces_output_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BENCH_EXIT", "1")
    gen = ExternalLoadGenerator(VllmBench(), command_prefix=[sys.executable, str(FAKE_TOOL)]).bind(
        WorkloadContext(profile=profile("synthetic"))
    )
    with pytest.raises(ExternalToolError, match=r"status 1(.|\n)*Initial test run failed"):
        await gen.run(BASE_URL, [], closed_plan(), request_timeout_s=60, model=MODEL)


def test_registered():
    assert "vllm_bench" in LOAD_GENERATORS
    gen = get_load_generator("vllm_bench")
    assert isinstance(gen, ExternalLoadGenerator) and isinstance(gen.tool, VllmBench)
    assert gen.name == "vllm_bench" and gen.context is None and gen.command_prefix == []
