"""TensorRT-LLM profile windows: the spec rules, its /start_profile body, the mock that
imitates it, and its trace's step labels and kernel names.

TensorRT-LLM 1.4.0rc2 (PyTorch backend) profiles at run time: POST /start_profile with a
StartProfileRequest (`output_dir`, `num_steps`, `start_step`; extra fields are a 422), it
records `num_steps` executor iterations, writes `trtllm-trace-<id>-rank-<r>.json` and
stops by itself; /stop_profile returns once the trace is written. Each iteration is one
bare `step[DECODE bs=N]` / `step[EXTEND bs=N toks=M]` scope; the overlap scheduler's
sampling runs in a second `step[...] sample` scope that is not a step boundary."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from loom_bench.experiment import ExpansionError, Experiment, expand
from loom_bench.mock.config import MockConfig
from loom_bench.profile.capture import capture_for, run_capture, start_body
from loom_bench.profile.mock_trace import SHARES, StepRecorder
from loom_bench.profile.spec import TRTLLM_TRACE_DIR, ProfileCapture, ProfileSpec
from loom_bench.profile.trace import analyze_events, categorize, step_shape
from loom_bench.registry import load_registry
from loom_bench.runner import RunnerContext, run_experiment

from .test_capture import _Server, ctx, mock_profile_doc  # noqa: F401  (ctx: a fixture)

pytestmark = pytest.mark.timeout(240)

REGISTRY = load_registry()
TRT_IMAGE = (
    "nvcr.io/nvidia/tensorrt-llm/release@sha256:"
    "93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a"
)
TRT_ENGINE = {"name": "trtllm", "version": "1.4.0rc2", "image": TRT_IMAGE}


# --- the body -------------------------------------------------------------------------------


def test_the_trtllm_body_is_its_start_profile_request_and_vllms_is_unchanged() -> None:
    spec = ProfileSpec(windows=[{"workload": "w"}], max_iterations=400, delay_iterations=3)
    trt = capture_for(spec, "r1", "trtllm")
    assert start_body(trt) == {"output_dir": TRTLLM_TRACE_DIR, "num_steps": 400, "start_step": 3}
    vllm = capture_for(spec, "r1")
    assert vllm.api == "vllm" and vllm.output_dir is None
    assert start_body(vllm) == {
        "profile_prefix": "loom-r1",
        "max_iterations": 400,
        "delay_iterations": 3,
    }


async def _capture_with_load(srv: _Server, cap: ProfileCapture, until_s: float) -> dict:
    t0 = time.perf_counter()
    base = f"http://127.0.0.1:{srv.port}/v1"
    async with httpx.AsyncClient(base_url=base, timeout=10) as http:

        async def load() -> None:
            while time.perf_counter() - t0 < until_s:
                await http.post(
                    "/completions", json={"model": "mock-model", "prompt": "hi", "max_tokens": 8}
                )

        out, _ = await asyncio.gather(run_capture(base, cap, t0), load())
    return out


async def test_a_trtllm_capture_records_num_steps_and_writes_its_trace(tmp_path: Path) -> None:
    srv = _Server(MockConfig(time_scale=0.01, profile_dir=str(tmp_path), profile_api="trtllm"))
    try:
        cap = ProfileCapture(
            start_after_s=0.05,
            stop_after_s=0.25,
            prefix="loom-abc",
            max_iterations=5,
            api="trtllm",
            output_dir=TRTLLM_TRACE_DIR,
        )
        out = await _capture_with_load(srv, cap, 0.4)
        assert out["start_status"] == 200 and out["stop_status"] == 200, out
        assert out["api"] == "trtllm" and "start_options_refused" not in out
        (trace,) = os.listdir(tmp_path)
        assert trace.startswith("trtllm-trace-") and trace.endswith("-rank-0.json")
        events = json.loads((tmp_path / trace).read_text())["traceEvents"]
        (rank,) = analyze_events(events, source=trace)
        assert rank.steps == 5 and rank.step_source == "gpu annotations"  # num_steps honoured
    finally:
        srv.close()


async def test_the_trtllm_mock_refuses_a_vllm_body_as_trtllm_does(tmp_path: Path) -> None:
    srv = _Server(MockConfig(time_scale=0.01, profile_dir=str(tmp_path), profile_api="trtllm"))
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{srv.port}") as http:
            r = await http.post("/start_profile", json={"profile_prefix": "p", "max_iterations": 1})
            assert r.status_code == 422
            r = await http.post("/start_profile", json={"num_steps": 1})
            assert r.status_code == 400  # Loom always names output_dir
    finally:
        srv.close()


async def test_a_refused_trtllm_start_is_never_retried_with_the_engines_defaults() -> None:
    # vLLM without per-session options is retried bodiless; TensorRT-LLM's defaults would
    # write the trace to /tmp inside its container, which nothing collects.
    srv = _Server(MockConfig(time_scale=0.01))  # no profiler: 404
    try:
        cap = ProfileCapture(
            start_after_s=0.01,
            stop_after_s=0.02,
            prefix="p",
            max_iterations=1,
            api="trtllm",
            output_dir=TRTLLM_TRACE_DIR,
        )
        out = await run_capture(f"http://127.0.0.1:{srv.port}/v1", cap, time.perf_counter())
        assert "404" in out["error"] and "start_options_refused" not in out
    finally:
        srv.close()


# --- the trace ------------------------------------------------------------------------------


def test_trtllm_step_labels_parse_and_the_sample_scope_is_no_step() -> None:
    assert step_shape("step[DECODE bs=37]") == (0, 0, 37)
    assert step_shape("step[EXTEND bs=2 toks=900]") == (2, 900, 0)
    assert step_shape("step[DECODE bs=37] sample") is None
    assert step_shape("step[EXTEND bs=2 toks=900] sample") is None
    assert step_shape("execute_context_1(12)_generation_3(3)") == (1, 12, 3)
    assert step_shape("forward") is None


def _recorded(style: str, steps: list[tuple[int, int]]) -> list[dict]:
    rec = StepRecorder(prefix="x", style=style)
    t = 0.0
    for seqs, prefill in steps:
        rec.step(t, 0.01, seqs, prefill)
        t += 0.01
    return rec.events()


def test_a_trtllm_trace_splits_by_decode_batch_with_extend_steps_mixed() -> None:
    steps = [(40, 0)] * 3 + [(8, 0)] * 2 + [(8, 512)] + [(40, 0)]
    events = _recorded("trtllm", steps)
    assert any(e["name"].endswith(" sample") for e in events)
    (rank,) = analyze_events(events)
    assert rank.steps == len(steps) and rank.step_source == "gpu annotations"
    shapes = {(s["kind"], s["batch"]): s for s in rank.step_shapes}
    # each step runs to the next one's start: the last has no end, so 6 periods
    assert shapes[("decode", 40)]["steps"] == 3
    assert shapes[("decode", 8)]["steps"] == 2
    assert shapes[("mixed", None)]["steps"] == 1
    assert shapes[("mixed", None)]["prefill_tokens_mean"] == 512
    per = rank.per_step_ms()
    assert per is not None
    total = sum(per[c] for _, c, _ in SHARES)
    for _, cat, share in SHARES:
        assert per[cat] / total == pytest.approx(share / 0.9, rel=1e-4)


def test_trtllm_labels_win_over_vllm_like_names_in_the_same_trace() -> None:
    # TensorRT-LLM's PyTorch model engine has methods whose names vLLM's step pattern
    # would match (e.g. `_model_forward`); once its own step labels are there, only they
    # count.
    events = _recorded("trtllm", [(4, 0)] * 4)
    for e in list(events):
        if e["cat"] == "gpu_user_annotation" and not e["name"].endswith("sample"):
            events.append({**e, "name": "PyTorchModelEngine._model_forward"})
    (rank,) = analyze_events(events)
    assert rank.steps == 4


def test_trtllm_decode_attention_kernels_are_attention() -> None:
    for name in (
        "xqa_kernel_dt_bf16_d_128_beam_1_kvt_e4m3_nqpkv_4_m_16_sm_89",
        "masked_multihead_attention_kernel",
        "mmha::masked_multihead_attention_kernel<__nv_bfloat16>",
        "fmha_v2_flash_attention_bf16_64_128_S_q_paged_kv_128_sm89_kernel_nl",
        "applyBiasRopeUpdateKVCacheV2",
    ):
        assert categorize(name) == "attention", name
    assert categorize("sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64") == "gemm"


# --- the rules ------------------------------------------------------------------------------


def _aws_doc(**variant: object) -> dict:
    return {
        **mock_profile_doc(),
        "provider": {"kind": "aws_ec2", "instance_type": "g6e.xlarge", "market": "on_demand"},
        "variants": [{"name": "t", **variant}],
        "profile": {"windows": [{"workload": "fixed-128-128"}]},
    }


def test_a_trtllm_variant_is_profiled_with_no_engine_args() -> None:
    exp = Experiment.model_validate(_aws_doc(engine=TRT_ENGINE))
    (cell,) = expand(exp, REGISTRY)
    assert cell.launch.engine == "trtllm"


@pytest.mark.parametrize(
    ("variant", "profile", "match"),
    [
        (
            {"engine": {**TRT_ENGINE, "args": {"profiler_config": {"profiler": "torch"}}}},
            {},
            "profiler_config is vLLM's",
        ),
        ({"engine": TRT_ENGINE}, {"trace_dir": "/data/traces"}, "writes to /var/lib/loom"),
        (
            {"engine": {"name": "sglang", "version": "0.5.6", "image": "lmsysorg/sglang:x"}},
            {},
            "vllm engines only, not sglang",
        ),
        # a registry row that is TensorRT-LLM, not named: the vLLM rule asks for either
        ({"model": "qwen3-8b-fp8-modelopt"}, {}, "engine.name: trtllm"),
    ],
)
def test_profile_rules_for_other_engines(variant: dict, profile: dict, match: str) -> None:
    doc = _aws_doc(**variant)
    doc["profile"].update(profile)
    with pytest.raises(ValidationError, match=match):
        Experiment.model_validate(doc)


def test_vllms_profiler_config_on_a_trtllm_row_is_refused_at_expansion() -> None:
    cfg = {
        "profiler": "torch",
        "torch_profiler_dir": TRTLLM_TRACE_DIR,
        "torch_profiler_with_stack": False,
    }
    exp = Experiment.model_validate(
        _aws_doc(model="qwen3-8b-fp8-modelopt", engine={"args": {"profiler_config": cfg}})
    )
    with pytest.raises(ExpansionError, match="profiler_config is vLLM's"):
        expand(exp, REGISTRY)


def test_profile_api_stays_out_of_a_default_mock_configs_dump() -> None:
    # Mock cells hash their whole config: one that does not set it keeps its hash.
    assert "profile_api" not in MockConfig().model_dump()
    trt = MockConfig(profile_api="trtllm")
    assert MockConfig.model_validate_json(trt.model_dump_json()) == trt


# --- end to end on the mock -----------------------------------------------------------------


async def test_trtllm_style_windows_on_the_mock_provider(ctx: RunnerContext) -> None:  # noqa: F811
    doc = mock_profile_doc(
        variants=[{"name": "a", "mock": {"profile_api": "trtllm"}}],
        repetitions=1,
        allow_single_run=True,
    )
    exp = Experiment.model_validate(doc)
    outcome = await run_experiment(exp, ctx)
    assert outcome.exit_code == 0, outcome
    (window,) = sorted((ctx.out_dir / str(outcome.experiment_id) / "profiles").iterdir())
    win = json.loads((window / "window.json").read_text())
    assert win["capture"]["api"] == "trtllm" and win["capture"]["start_status"] == 200
    (rank,) = win["breakdown"]["ranks"]
    assert 0 < rank["steps"] <= 12 and rank["step_source"] == "gpu annotations"
    assert rank["source"].startswith("trtllm-trace-")
