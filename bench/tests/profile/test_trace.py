"""Trace breakdown on synthetic torch-profiler traces with a known answer."""

import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.profile.mock_trace import GAP_SHARE, SHARES, StepRecorder
from loom_bench.profile.trace import (
    TraceBreakdown,
    analyze_events,
    analyze_path,
    categorize,
    render_breakdown,
)


def kernel(name: str, ts: float, dur: float, *, device: int = 0, stream: int = 7) -> dict:
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": ts,
        "dur": dur,
        "pid": device,
        "tid": stream,
        "args": {"device": device, "stream": stream},
    }


@pytest.mark.parametrize(
    ("name", "cat"),
    [
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "allreduce"),
        ("void vllm::cross_device_reduce_1stage<__nv_bfloat16, 2>", "allreduce"),
        ("flash_fwd_splitkv_kernel", "attention"),
        ("void vllm::reshape_and_cache_flash_kernel", "attention"),
        ("BatchDecodeWithPagedKVCacheKernel", "attention"),
        ("kernel_unified_attention_2d", "attention"),
        (
            "void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_64x4_tn_align8>",  # noqa: E501
            "gemm",
        ),
        ("sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64", "gemm"),
        ("ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn", "gemm"),
        ("triton_tem_fused_mm_0", "gemm"),
        ("void vllm::dynamic_per_token_scaled_fp8_quant_kernel", "quant"),
        ("void vllm::scaled_fp8_quant_kernel", "quant"),
        ("void at::native::(anonymous namespace)::cunn_SoftMaxForward", "sampling"),
        ("void at::native::sbtopk::gatherTopK<float>", "sampling"),
        ("void vllm::rms_norm_kernel<c10::BFloat16>", "other"),
        ("triton_poi_fused_mul_silu_0", "other"),
        # vLLM v0.30 on L40S (smoke a183a857): Inductor fuses the FP8 activation
        # quantization into the norm / SiLU before each scaled GEMM; the consumer's name
        # (`cutlass_scaled_mm`) must not make it a GEMM.
        (
            "triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_fused_add_rms_norm_max_mul_reciprocal_unsqueeze_1",
            "quant",
        ),
        (
            "triton_red_fused__to_copy_abs_clamp_cutlass_scaled_mm_div_max_mul_reciprocal_silu_slice_unsqueeze_2",
            "quant",
        ),
        ("triton_red_fused_fused_add_rms_norm_2", "other"),
        ("triton_red_fused_3", "other"),
        (
            "void cutlass::Kernel2<enable_sm89_to_sm90<cutlass::gemm::kernel::DefaultGemmWithVisitor<cutlass::float_e4m3_t,",  # noqa: E501
            "gemm",
        ),
        (
            "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, __nv_bfloat16>",
            "gemm",
        ),
        (
            "void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<",
            "attention",
        ),
        ("_gumbel_sample_kernel", "sampling"),
    ],
)
def test_categories(name: str, cat: str) -> None:
    assert categorize(name) == cat


def two_step_trace() -> list[dict]:
    """Two 10 ms steps on GPU 0, each: 1 ms idle, 5 ms GEMM, 2 ms attention, 1 ms of
    sampling, and an all-reduce of 2 ms on its own stream, half of it under the GEMM."""
    ev = []
    for s in range(2):
        t = s * 10_000.0
        ev.append(
            {
                "ph": "X",
                "cat": "gpu_user_annotation",
                "name": "execute_model",
                "ts": t,
                "dur": 10_000,
                "pid": 0,
                "tid": 1,
                "args": {"device": 0},
            }
        )
        ev.append(kernel("gemm_bf16", t + 1000, 5000))
        ev.append(kernel("ncclDevKernel_AllReduce", t + 5000, 2000, stream=9))
        ev.append(kernel("flash_attn_kernel", t + 7000, 2000))
        ev.append(kernel("topk_sampler", t + 9000, 1000))
    ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 5})  # ignored
    return ev


def test_exclusive_split_counts_only_exposed_communication() -> None:
    (r,) = analyze_events(two_step_trace())
    # window: first kernel (1 ms) to last kernel end (20 ms)
    assert r.window_ms == pytest.approx(19.0)
    assert r.category_ms["gemm"] == pytest.approx(10.0)
    assert r.category_ms["allreduce"] == pytest.approx(2.0)  # 1 ms exposed per step
    assert r.category_ms["attention"] == pytest.approx(4.0)
    assert r.category_ms["sampling"] == pytest.approx(2.0)
    assert r.category_ms["idle"] == pytest.approx(1.0)  # the gap between the steps
    assert sum(r.category_ms.values()) == pytest.approx(r.window_ms)
    assert r.steps == 2 and r.step_source == "gpu annotations"
    assert r.per_step_ms()["gemm"] == pytest.approx(5.0)


def test_steps_fall_back_to_the_requested_count() -> None:
    ev = [e for e in two_step_trace() if e["cat"] != "gpu_user_annotation"]
    (r,) = analyze_events(ev, steps=4)
    assert r.steps == 4 and r.step_source == "requested max_iterations"
    (r,) = analyze_events(ev)
    assert r.steps is None and r.per_step_ms() is None


def vllm_step(name: str, ts: float, dur: float) -> list[dict]:
    """vLLM v0.30's step annotation, CPU and GPU side."""
    return [
        {"ph": "X", "cat": "user_annotation", "name": name, "ts": ts - 500, "dur": 300},
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": name,
            "ts": ts,
            "dur": dur,
            "pid": 0,
            "tid": 7,
            "args": {},
        },
    ]


def vllm_v030_trace() -> list[dict]:
    """Four steps as vLLM v0.30 records them: three pure-decode steps of batch 4 and one
    that also prefills 1024 tokens. Each annotated forward (GEMM then attention) is
    followed, outside the annotation, by the LM head GEMM and sampling, then a CPU gap.
    Decode step: 8 ms GEMM + 2 attention + 1 LM head + 0.5 sampling + 0.5 idle = 12 ms.
    The mixed step's forward is 30 ms of GEMM instead of 8."""
    ev: list[dict] = []
    t = 0.0
    for i, name in enumerate(
        [
            "execute_context_0(0)_generation_4(4)",
            "execute_context_1(1024)_generation_3(3)",
            "execute_context_0(0)_generation_4(4)",
            "execute_context_0(0)_generation_4(4)",
            "execute_context_0(0)_generation_4(4)",  # last: closes the previous period
        ]
    ):
        g = 30_000.0 if i == 1 else 8_000.0
        ev += vllm_step(name, t, g + 2000)
        ev.append(kernel("cutlass_80_wmma_tensorop_bf16_s161616gemm", t, g))
        ev.append(kernel("flash_fwd_splitkv_kernel", t + g, 2000))
        ev.append(kernel("cutlass_80_wmma_tensorop_bf16_s161616gemm", t + g + 2000, 1000))
        ev.append(kernel("_gumbel_sample_kernel", t + g + 3000, 500))
        t += g + 4000
    return ev


def test_vllm_v030_steps_counted_and_split_by_shape() -> None:
    (r,) = analyze_events(vllm_v030_trace(), steps=200)  # the requested count is ignored
    assert r.steps == 5 and r.step_source == "gpu annotations"
    decode, mixed = r.step_shapes
    assert decode["kind"] == "decode" and decode["batch"] == 4 and decode["steps"] == 3
    assert decode["step_ms_p50"] == pytest.approx(12.0)
    cats = decode["category_ms"]
    assert cats["gemm"] == pytest.approx(9.0)  # forward + LM head
    assert cats["attention"] == pytest.approx(2.0)
    assert cats["sampling"] == pytest.approx(0.5)
    assert cats["idle"] == pytest.approx(0.5)
    assert sum(cats.values()) == pytest.approx(decode["step_ms_mean"])
    assert mixed["kind"] == "mixed" and mixed["steps"] == 1 and mixed["batch"] is None
    assert mixed["step_ms_mean"] == pytest.approx(34.0)
    assert mixed["category_ms"]["gemm"] == pytest.approx(31.0)
    assert mixed["prefill_tokens_mean"] == 1024
    assert "decode x4" in render_breakdown(TraceBreakdown([r]))


def test_step_shapes_need_vllm_step_names() -> None:
    (r,) = analyze_events(two_step_trace())  # `execute_model`: counted, not split
    assert r.steps == 2 and r.step_shapes == []


def test_one_breakdown_per_gpu() -> None:
    ev = [*two_step_trace(), kernel("gemm_bf16", 0, 100, device=1)]
    ranks = analyze_events(ev)
    assert [r.device for r in ranks] == [0, 1]
    assert ranks[1].category_ms["gemm"] == pytest.approx(0.1)


def test_no_kernels_is_an_error() -> None:
    with pytest.raises(ValueError, match="no GPU kernel"):
        analyze_events([{"ph": "X", "cat": "cpu_op", "name": "x", "ts": 0, "dur": 1}])


def test_files_gzip_and_tarballs(tmp_path: Path) -> None:
    doc = {"traceEvents": two_step_trace()}
    plain = tmp_path / "t.json"
    plain.write_text(json.dumps(doc))
    gz = tmp_path / "rank0.1.pt.trace.json.gz"
    gz.write_bytes(gzip.compress(json.dumps(doc).encode()))
    assert analyze_path(plain).ranks[0].category_ms["gemm"] == pytest.approx(10.0)
    assert analyze_path(gz).ranks[0].category_ms["gemm"] == pytest.approx(10.0)
    tar = tmp_path / "trace.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(gz, arcname="./loom-r_dp0_tp0_rank0.1.pt.trace.json.gz")
        t.add(gz, arcname="./loom-r_dp0_tp1_rank1.1.pt.trace.json.gz")
        info = tarfile.TarInfo("host_123.async_llm.pt.trace.json.gz")  # CPU-only: skipped
        info.size = 3
        t.addfile(info, io.BytesIO(b"xyz"))
    res = analyze_path(tar)
    assert len(res.ranks) == 2
    assert "GEMM" in render_breakdown(res)


def test_mock_recorder_trace_has_its_known_shares(tmp_path: Path) -> None:
    rec = StepRecorder(prefix="loom-x", max_iterations=3, delay_iterations=1)
    for i in range(6):
        rec.step(start_s=i * 0.02, duration_s=0.02, decode_seqs=4, prefill_tokens=0)
    assert len(rec.steps) == 3  # one skipped, then capped
    (r,) = analyze_path(rec.write(tmp_path)).ranks
    assert r.steps == 3 and r.step_source == "gpu annotations"
    per = r.per_step_ms()
    for _, cat, share in SHARES:
        assert per[cat] == pytest.approx(share * 20, rel=1e-6)
    # idle: the gaps before steps 2 and 3 (the first one is before the window)
    assert r.category_ms["idle"] == pytest.approx(2 * GAP_SHARE * 20)
    # step by step: two whole periods (start to next start), each one simulated step
    (decode,) = r.step_shapes
    assert decode["batch"] == 4 and decode["steps"] == 2
    assert decode["step_ms_p50"] == pytest.approx(20.0)
    for _, cat, share in SHARES:
        assert decode["category_ms"][cat] == pytest.approx(share * 20, rel=1e-6)
    assert decode["category_ms"]["idle"] == pytest.approx(GAP_SHARE * 20)


def test_cli_profile_analyze(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"traceEvents": two_step_trace()}))
    out = tmp_path / "b.json"
    result = CliRunner().invoke(app, ["profile", "analyze", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["ranks"][0]["steps"] == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert CliRunner().invoke(app, ["profile", "analyze", str(bad)]).exit_code == 2
