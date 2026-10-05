from typing import Any

import pytest
from pydantic import ValidationError

from loom_bench.engines import docker_run_argv, render_args, render_launch
from loom_bench.registry import ModelSpec, load_registry

VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
)
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
)
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"
LLAMA_REV = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
TO_SGLANG = {"engine": "sglang", "version": "0.5.21", "image": SGLANG_IMAGE}


@pytest.fixture(scope="module")
def qwen() -> ModelSpec:
    return load_registry().get("qwen3-8b")


@pytest.fixture(scope="module")
def llama() -> ModelSpec:
    return load_registry().get("llama-3.3-70b-instruct")


def patched(spec: ModelSpec, **changes: Any) -> ModelSpec:
    doc = spec.model_dump()
    for path, value in changes.items():
        *parents, leaf = path.split("__")
        node = doc
        for p in parents:
            node = node[p]
        node[leaf] = value
    return ModelSpec.model_validate(doc)


def test_vllm_qwen_exact_argv(qwen: ModelSpec) -> None:
    launch = render_launch(qwen)
    assert launch.engine == "vllm"
    assert launch.image == VLLM_IMAGE
    assert launch.served_model == "qwen3-8b"
    assert launch.model_revision == QWEN_REV
    assert launch.gpus == 1
    assert launch.port == 8000
    assert launch.args == [
        "Qwen/Qwen3-8B",
        "--revision",
        QWEN_REV,
        "--tokenizer-revision",
        QWEN_REV,
        "--served-model-name",
        "qwen3-8b",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        "32768",
        "--kv-cache-dtype",
        "auto",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_vllm_llama_tp4(llama: ModelSpec) -> None:
    launch = render_launch(llama)
    assert launch.gpus == 4
    assert launch.args[:2] == ["meta-llama/Llama-3.3-70B-Instruct", "--revision"]
    i = launch.args.index("--tensor-parallel-size")
    assert launch.args[i + 1] == "4"
    assert "--trust-remote-code" not in launch.args


def test_sglang_qwen_exact_argv(qwen: ModelSpec) -> None:
    launch = render_launch(qwen, TO_SGLANG)
    assert launch.engine == "sglang"
    assert launch.image == SGLANG_IMAGE
    assert launch.args == [
        "--model-path",
        "Qwen/Qwen3-8B",
        "--revision",
        QWEN_REV,
        "--served-model-name",
        "qwen3-8b",
        "--tp-size",
        "1",
        "--context-length",
        "32768",
        "--enable-metrics",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]


def test_sglang_llama_tp2_pp2_fp8_kv(llama: ModelSpec) -> None:
    launch = render_launch(
        llama, {**TO_SGLANG, "tp": 2, "pp": 2, "kv_cache_dtype": "fp8", "port": 30000}
    )
    a = launch.args
    assert a[a.index("--tp-size") + 1] == "2"
    assert a[a.index("--pp-size") + 1] == "2"
    assert a[a.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert a[a.index("--port") + 1] == "30000"
    assert launch.port == 30000
    assert launch.gpus == 4


def test_overrides_patch_args_and_quantization(qwen: ModelSpec) -> None:
    spec = patched(qwen, engine__args={"max_num_seqs": 128, "enable_prefix_caching": True})
    launch = render_launch(
        spec,
        {
            "args": {
                "max_num_seqs": 256,
                "enable_prefix_caching": None,
                "gpu_memory_utilization": 0.9,
            },
            "quantization": "fp8",
            "kv_cache_dtype": "fp8",
            "max_context": 8192,
            "ready_timeout_s": 600,
        },
    )
    a = launch.args
    assert a[a.index("--max-model-len") + 1] == "8192"
    assert a[a.index("--quantization") + 1] == "fp8"
    assert a[a.index("--kv-cache-dtype") + 1] == "fp8"
    assert a[-4:] == ["--max-num-seqs", "256", "--gpu-memory-utilization", "0.9"]
    assert "--enable-prefix-caching" not in a
    assert launch.ready_timeout_s == 600


def test_prequantized_formats_emit_no_flag(qwen: ModelSpec) -> None:
    for q in ("awq", "gptq", "w4a16", "w8a8", "nvfp4"):
        assert "--quantization" not in render_launch(qwen, {"quantization": q}).args


def test_explicit_quantization_arg_replaces_derived(qwen: ModelSpec) -> None:
    ct = render_launch(
        qwen, {"quantization": "fp8", "args": {"quantization": "compressed-tensors"}}
    )
    assert ct.args[ct.args.index("--quantization") + 1] == "compressed-tensors"
    assert ct.args.count("--quantization") == 1
    off = render_launch(qwen, {"quantization": "fp8", "args": {"quantization": False}})
    assert "--quantization" not in off.args


def test_render_args_types() -> None:
    argv, quant = render_args(
        "vllm",
        {
            "enforce_eager": True,
            "disable_log_requests": False,
            "max-num-batched-tokens": 8192,
            "cuda_graph_sizes": [1, 2, 4],
            "compilation_config": {"level": 3, "cudagraph_mode": "FULL"},
            "load_format": "safetensors",
        },
    )
    assert quant is None
    assert argv == [
        "--enforce-eager",
        "--max-num-batched-tokens",
        "8192",
        "--cuda-graph-sizes",
        "1",
        "2",
        "4",
        "--compilation-config",
        '{"cudagraph_mode":"FULL","level":3}',
        "--load-format",
        "safetensors",
    ]


@pytest.mark.parametrize(
    "args",
    [
        {"trust_remote_code": True},
        {"max_model_len": 4096},
        {"tensor_parallel_size": 2},
        {"port": 9000},
        {"Bad Flag": 1},
        {"x": None},
        {"x": [True]},
    ],
)
def test_render_args_rejects(args: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        render_args("vllm", args)


def test_trust_remote_code_only_with_review(qwen: ModelSpec) -> None:
    assert "--trust-remote-code" not in render_launch(qwen).args
    with pytest.raises(ValidationError):
        patched(qwen, hf__trust_remote_code=True)
    reviewed = patched(
        qwen,
        hf__trust_remote_code=True,
        hf__trust_remote_code_review={"reviewer": "a", "date": "2026-10-01", "notes": "ok"},
    )
    assert "--trust-remote-code" in render_launch(reviewed).args
    assert "--trust-remote-code" in render_launch(reviewed, TO_SGLANG).args


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"engine": "sglang"}, "requires image and version"),
        ({"tp": 2}, "tp \\* pp"),
        ({"max_context": 65536}, "exceeds the registry cap"),
        ({"bogus": 1}, "Extra inputs"),
        ({"image": "vllm/vllm-openai:latest"}, "String should match"),
        ({"env": {"HF_TOKEN": "x"}}, "secrets never"),
        ({"env": {"lower": "x"}}, "invalid environment"),
    ],
)
def test_overrides_validated(qwen: ModelSpec, overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        render_launch(qwen, overrides)


def test_engine_switch_drops_other_engine_args(qwen: ModelSpec) -> None:
    spec = patched(qwen, engine__args={"max_num_seqs": 64})
    launch = render_launch(spec, {**TO_SGLANG, "args": {"mem_fraction_static": 0.85}})
    assert "--max-num-seqs" not in launch.args
    assert launch.args[-2:] == ["--mem-fraction-static", "0.85"]


def test_docker_run_argv(qwen: ModelSpec) -> None:
    launch = render_launch(qwen, {"env": {"VLLM_LOGGING_LEVEL": "INFO"}})
    argv = docker_run_argv(launch, weights_dir="/opt/dlami/nvme/hf", container_name="loom-engine")
    assert argv[:5] == ["docker", "run", "--detach", "--name", "loom-engine"]
    joined = " ".join(argv)
    assert "--gpus 1" in joined
    assert "--ipc host" in joined
    assert "--publish 127.0.0.1:8000:8000" in joined
    assert "--volume /opt/dlami/nvme/hf:/root/.cache/huggingface" in joined
    assert "--env HF_HUB_OFFLINE=1" in joined
    assert "--env VLLM_LOGGING_LEVEL" in joined
    assert "INFO" not in argv
    i = argv.index("--entrypoint")
    assert argv[i : i + 4] == ["--entrypoint", "vllm", VLLM_IMAGE, "serve"]
    assert argv[i + 4 :] == launch.args


def test_docker_run_argv_sglang_entrypoint(qwen: ModelSpec) -> None:
    launch = render_launch(qwen, TO_SGLANG)
    argv = docker_run_argv(launch, weights_dir="/w", container_name="c")
    i = argv.index("--entrypoint")
    assert argv[i : i + 5] == [
        "--entrypoint",
        "python3",
        SGLANG_IMAGE,
        "-m",
        "sglang.launch_server",
    ]
