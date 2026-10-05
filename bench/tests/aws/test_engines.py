from typing import Any

import pytest
from pydantic import ValidationError

from loom_bench.engines import docker_run_argv, quantization_flag, render_args, render_launch
from loom_bench.registry import CHECKPOINT_PRECISIONS, ModelSpec, load_registry

VLLM_IMAGE = (
    "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
)
SGLANG_IMAGE = (
    "lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
)
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"


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


def on_sglang(spec: ModelSpec, **changes: Any) -> ModelSpec:
    return patched(
        spec,
        engine__name="sglang",
        engine__version="0.5.21",
        engine__image=SGLANG_IMAGE,
        engine__args={},
        **changes,
    )


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
    launch = render_launch(on_sglang(qwen))
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
    spec = on_sglang(llama, parallelism__tp=2, parallelism__pp=2, kv_cache_dtype="fp8")
    launch = render_launch(spec)
    a = launch.args
    assert a[a.index("--tp-size") + 1] == "2"
    assert a[a.index("--pp-size") + 1] == "2"
    assert a[a.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert launch.gpus == 4


def test_engine_args_follow_registry_flags(qwen: ModelSpec) -> None:
    spec = patched(
        qwen,
        engine__args={"max_num_seqs": 256, "gpu_memory_utilization": 0.9},
        kv_cache_dtype="fp8",
        max_context=8192,
    )
    a = render_launch(spec).args
    assert a[a.index("--max-model-len") + 1] == "8192"
    assert a[a.index("--kv-cache-dtype") + 1] == "fp8"
    assert a[-4:] == ["--max-num-seqs", "256", "--gpu-memory-utilization", "0.9"]


# One row per (hf.quant_method, quantization) pair in the `quantization_flag` table.
QUANT_TABLE = [
    (None, "none", []),
    (None, "fp8", ["--quantization", "fp8"]),
    ("fp8", "fp8", []),
    ("compressed-tensors", "fp8", []),
    ("compressed-tensors", "w8a8", []),
    ("compressed-tensors", "w4a16", []),
    ("compressed-tensors", "nvfp4", []),
    ("awq", "awq", []),
    ("gptq", "gptq", []),
    ("modelopt", "fp8", []),
    ("modelopt", "nvfp4", []),
]


def test_quant_table_covers_every_allowed_pair() -> None:
    allowed = {(m, q) for m, qs in CHECKPOINT_PRECISIONS.items() for q in qs}
    assert {(m, q) for m, q, _ in QUANT_TABLE} == allowed


@pytest.mark.parametrize(("method", "precision", "flag"), QUANT_TABLE)
@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_quantization_flag_table(
    qwen: ModelSpec, engine: str, method: str | None, precision: str, flag: list[str]
) -> None:
    spec = patched(qwen, hf__quant_method=method, quantization=precision)
    if engine == "sglang":
        spec = on_sglang(spec)
    assert quantization_flag(spec) == flag
    args = render_launch(spec).args
    if flag:
        i = args.index("--quantization")
        assert args[i : i + 2] == flag and args.count("--quantization") == 1
    else:
        assert "--quantization" not in args


@pytest.mark.parametrize(
    ("method", "precision"),
    [
        (None, "awq"),
        (None, "gptq"),
        (None, "w4a16"),
        (None, "w8a8"),
        (None, "fp4"),
        (None, "nvfp4"),
        ("fp8", "none"),
        ("awq", "none"),
        ("awq", "fp8"),
        ("gptq", "awq"),
        ("compressed-tensors", "none"),
        ("modelopt", "fp4"),
    ],
)
def test_quantization_pairs_outside_the_table_are_rejected(
    qwen: ModelSpec, method: str | None, precision: str
) -> None:
    with pytest.raises(ValidationError, match="cannot be served from"):
        patched(qwen, hf__quant_method=method, quantization=precision)


def test_render_args_types() -> None:
    argv = render_args(
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
        {"quantization": "compressed-tensors"},
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
    assert "--trust-remote-code" in render_launch(on_sglang(reviewed)).args


def test_docker_run_argv(qwen: ModelSpec) -> None:
    launch = render_launch(qwen).model_copy(update={"env": {"VLLM_LOGGING_LEVEL": "INFO"}})
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


@pytest.mark.parametrize(
    ("env", "match"),
    [({"HF_TOKEN": "x"}, "secrets never"), ({"lower": "x"}, "invalid environment")],
)
def test_docker_run_argv_rejects_env(qwen: ModelSpec, env: dict[str, str], match: str) -> None:
    launch = render_launch(qwen).model_copy(update={"env": env})
    with pytest.raises(ValueError, match=match):
        docker_run_argv(launch, weights_dir="/w", container_name="c")


def test_docker_run_argv_sglang_entrypoint(qwen: ModelSpec) -> None:
    launch = render_launch(on_sglang(qwen))
    argv = docker_run_argv(launch, weights_dir="/w", container_name="c")
    i = argv.index("--entrypoint")
    assert argv[i : i + 5] == [
        "--entrypoint",
        "python3",
        SGLANG_IMAGE,
        "-m",
        "sglang.launch_server",
    ]
