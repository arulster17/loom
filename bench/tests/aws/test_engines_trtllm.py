"""TensorRT-LLM (`trtllm`) launch rendering: trtllm-serve 1.2.1 on the PyTorch backend.

Flag and option names are from tensorrt_llm/commands/serve.py and llmapi/llm_args.py at
v1.2.1 (docs/perf/tensorrt-llm.md)."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from loom_bench.engines import (
    CONTAINER_HF_HOME,
    ENTRYPOINTS,
    docker_run_argv,
    engine_process_argv,
    metrics_path,
    render_launch,
    trtllm_options,
)
from loom_bench.experiment import HASHED_PARSER_ENGINES, serving_model_doc
from loom_bench.registry import ModelSpec, load_registry

TRTLLM_IMAGE = (
    "nvcr.io/nvidia/tensorrt-llm/release@sha256:"
    "33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9"
)
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"
SNAPSHOT = f"{CONTAINER_HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/{QWEN_REV}"


@pytest.fixture(scope="module")
def qwen() -> ModelSpec:
    return load_registry().get("qwen3-8b")


def on_trtllm(spec: ModelSpec, args: dict[str, Any] | None = None, **top: Any) -> ModelSpec:
    doc = spec.model_dump()
    doc["engine"].update(name="trtllm", version="1.2.1", image=TRTLLM_IMAGE, args=args or {})
    for path, value in top.items():
        *parents, leaf = path.split("__")
        node = doc
        for p in parents:
            node = node[p]
        node[leaf] = value
    return ModelSpec.model_validate(doc)


def test_trtllm_qwen_exact_argv(qwen: ModelSpec) -> None:
    launch = render_launch(on_trtllm(qwen))
    assert launch.engine == "trtllm" and launch.image == TRTLLM_IMAGE
    assert launch.served_model == "qwen3-8b" and launch.port == 8000 and launch.gpus == 1
    assert launch.args == [
        '{"guided_decoding_backend":"xgrammar","return_perf_metrics":true}',
        SNAPSHOT,
        "--backend",
        "pytorch",
        "--tp_size",
        "1",
        "--pp_size",
        "1",
        "--max_seq_len",
        "32768",
        "--tool_parser",
        "qwen3",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    # The weights the host must hold before the engine starts offline: the pinned model.
    assert (launch.model_repo, launch.model_revision) == ("Qwen/Qwen3-8B", QWEN_REV)


def test_engine_args_are_llm_api_options(qwen: ModelSpec) -> None:
    spec = on_trtllm(
        qwen,
        {"kv_cache_config": {"enable_block_reuse": False}, "enable_chunked_prefill": True},
        kv_cache_dtype="fp8",
    )
    assert trtllm_options(spec) == {
        "enable_chunked_prefill": True,
        "guided_decoding_backend": "xgrammar",
        "kv_cache_config": {"dtype": "fp8", "enable_block_reuse": False},
        "return_perf_metrics": True,
    }
    # Sorted, compact JSON: the same options always hash the same.
    assert render_launch(spec).args[0] == json.dumps(
        trtllm_options(spec), sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"tensor_parallel_size": 2}, "set from the registry"),
        ({"return_perf_metrics": False}, "set from the registry"),
        ({"guided_decoding_backend": "llguidance"}, "set from the registry"),
        ({"tool_parser": "qwen3_coder"}, "set from the registry"),
        ({"kv_cache_config": {"dtype": "fp8"}}, "kv_cache_dtype"),
        ({"kv_cache_config": 1}, "mapping"),
        ({"max-batch-size": 8}, "invalid TensorRT-LLM option name"),
        ({"speculative_config": {"decoding_type": "Eagle"}}, "not wired"),
    ],
)
def test_reserved_and_malformed_options_are_rejected(
    qwen: ModelSpec, args: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        render_launch(on_trtllm(qwen, args))


def test_kv_cache_e5m2_is_rejected(qwen: ModelSpec) -> None:
    with pytest.raises(ValueError, match="E4M3"):
        render_launch(on_trtllm(qwen, kv_cache_dtype="fp8_e5m2"))


def test_quantization_comes_from_the_checkpoint(qwen: ModelSpec) -> None:
    # No on-load quantization of BF16 weights, and no compressed-tensors loader in 1.2.1.
    with pytest.raises(ValueError, match="does not quantize BF16 weights on load"):
        render_launch(on_trtllm(qwen, quantization="fp8"))
    fp8 = load_registry().get("qwen3-8b-fp8")  # RedHatAI compressed-tensors
    with pytest.raises(ValueError, match="cannot load compressed-tensors"):
        render_launch(on_trtllm(fp8))
    modelopt = on_trtllm(qwen, hf__quant_method="modelopt", quantization="fp8")
    assert "--quantization" not in render_launch(modelopt).args


def test_tools_need_a_trtllm_parser(qwen: ModelSpec) -> None:
    with pytest.raises(ValueError, match=r"tool_call_parsers\.trtllm"):
        render_launch(on_trtllm(qwen, tool_call_parsers={"vllm": "hermes"}))
    no_tools = on_trtllm(
        qwen, tool_call_parsers={}, capabilities={**qwen.capabilities.model_dump(), "tools": False}
    )
    assert "--tool_parser" not in render_launch(no_tools).args


def test_runpod_process_moves_the_snapshot_to_the_pod_cache(qwen: ModelSpec) -> None:
    launch = render_launch(on_trtllm(qwen))
    argv = engine_process_argv(launch, host="127.0.0.1", weights_dir="/opt/loom/hf/")
    assert argv[:4] == list(ENTRYPOINTS["trtllm"])
    assert argv[5] == f"/opt/loom/hf/hub/models--Qwen--Qwen3-8B/snapshots/{QWEN_REV}"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert launch.args[1] == SNAPSHOT  # the config hash covers the container path only
    # vLLM's model argument is a hub id: nothing to move.
    vllm = render_launch(qwen)
    assert engine_process_argv(vllm, host="h", weights_dir="/x")[2:] == [
        a if a != "0.0.0.0" else "h" for a in vllm.args
    ]


def test_docker_run_argv_trtllm_entrypoint(qwen: ModelSpec) -> None:
    launch = render_launch(on_trtllm(qwen))
    argv = docker_run_argv(launch, weights_dir="/w", container_name="c")
    assert "--volume /w:/root/.cache/huggingface" in " ".join(argv)
    i = argv.index("--entrypoint")
    assert argv[i : i + 5] == ["--entrypoint", "bash", TRTLLM_IMAGE, "-c", ENTRYPOINTS["trtllm"][2]]
    assert argv[i + 5] == "trtllm-launch" and argv[i + 6 :] == launch.args


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_writes_the_options_and_execs_trtllm_serve(qwen: ModelSpec, tmp_path) -> None:
    # A stand-in trtllm-serve on PATH that prints its argv and the options file.
    fake = tmp_path / "bin" / "trtllm-serve"
    fake.parent.mkdir()
    fake.write_text(
        '#!/bin/bash\nfor a in "$@"; do echo "arg:$a"; done\n'
        'cfg="${@: -1}"; echo "cfg:$(cat "$cfg")"\n'
    )
    fake.chmod(0o755)
    spec = on_trtllm(qwen, {"kv_cache_config": {"free_gpu_memory_fraction": 0.85}})
    launch = render_launch(spec)
    argv = engine_process_argv(launch, host="127.0.0.1", weights_dir="/opt/loom/hf")
    env = {**os.environ, "PATH": f"{fake.parent}:{os.environ['PATH']}"}
    out = subprocess.run(argv, env=env, capture_output=True, text=True, check=True).stdout
    args = [line[4:] for line in out.splitlines() if line.startswith("arg:")]
    assert args[0] == "serve"
    assert args[1:-2] == argv[5:]  # everything after the options document, in order
    assert args[-2] == "--extra_llm_api_options"
    path = Path(args[-1])
    try:
        assert path.name.startswith("loom-trtllm.") and path.stat().st_mode & 0o077 == 0
        cfg = next(line[4:] for line in out.splitlines() if line.startswith("cfg:"))
        assert json.loads(cfg) == trtllm_options(spec)
    finally:
        path.unlink(missing_ok=True)
    # A second start gets its own file (mktemp randomizes trailing X's only on BSD).
    again = subprocess.run(argv, env=env, capture_output=True, text=True, check=True).stdout
    second = Path(next(ln[4:] for ln in reversed(again.splitlines()) if ln.startswith("arg:")))
    second.unlink(missing_ok=True)
    assert second != path


def test_metrics_paths() -> None:
    assert metrics_path("trtllm") == "/prometheus/metrics"
    assert metrics_path("vllm") == metrics_path("sglang") == "/metrics"


def test_the_trtllm_parser_moves_only_trtllm_config_hashes(qwen: ModelSpec) -> None:
    # Adding `trtllm: qwen3` to the registry must not move a stored vLLM or SGLang hash.
    assert "trtllm" in qwen.tool_call_parsers
    assert set(serving_model_doc(qwen)["tool_call_parsers"]) == HASHED_PARSER_ENGINES
    trt = serving_model_doc(on_trtllm(qwen))["tool_call_parsers"]
    assert trt == {"vllm": "hermes", "sglang": "qwen25", "trtllm": "qwen3"}
