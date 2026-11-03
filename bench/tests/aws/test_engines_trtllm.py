"""TensorRT-LLM (`trtllm`) launch rendering: trtllm-serve 1.4.0rc2 on the PyTorch backend,
started through Loom's prompt-logprobs wrapper (loom_bench/trtllm_scorer.py).

Flag and option names are from tensorrt_llm/commands/serve.py and llmapi/llm_args.py at
v1.4.0rc2 (docs/perf/tensorrt-llm.md)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from loom_bench.engines import (
    CONTAINER_HF_HOME,
    ENTRYPOINTS,
    TRTLLM_LAUNCHER,
    TRTLLM_SERVE_SOURCE,
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
    "93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a"
)
QWEN_REV = "b968826d9c46dd6066d109eabc6255188de91218"
SNAPSHOT = f"{CONTAINER_HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/{QWEN_REV}"


@pytest.fixture(scope="module")
def qwen() -> ModelSpec:
    return load_registry().get("qwen3-8b")


def on_trtllm(spec: ModelSpec, args: dict[str, Any] | None = None, **top: Any) -> ModelSpec:
    doc = spec.model_dump()
    doc["engine"].update(name="trtllm", version="1.4.0rc2", image=TRTLLM_IMAGE, args=args or {})
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
        '{"enable_iter_perf_stats":true,"guided_decoding_backend":"xgrammar",'
        '"return_perf_metrics":true}',
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
        "--served_model_name",
        "qwen3-8b",
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
        "enable_iter_perf_stats": True,
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
        ({"enable_iter_perf_stats": False}, "set from the registry"),
        ({"served_model_name": "x"}, "set from the registry"),
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
    # No on-load quantization of BF16 weights, and no compressed-tensors loader.
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
    assert argv[:5] == list(ENTRYPOINTS["trtllm"])
    assert argv[6] == f"/opt/loom/hf/hub/models--Qwen--Qwen3-8B/snapshots/{QWEN_REV}"
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
    assert argv[i : i + 5] == ["--entrypoint", "bash", TRTLLM_IMAGE, "-c", TRTLLM_LAUNCHER]
    assert argv[i + 5 : i + 7] == ["trtllm-launch", TRTLLM_SERVE_SOURCE]
    assert argv[i + 7 :] == launch.args
    # NVIDIA's documented run flags for its release image, before the entrypoint; the
    # host's IPC namespace (and so its /dev/shm) as for every engine.
    head = argv[:i]
    assert "--ulimit memlock=-1 --ulimit stack=67108864" in " ".join(head)
    assert "--ipc host" in " ".join(head)
    # vLLM gets none of them (95cde129's container, unchanged).
    vllm = docker_run_argv(render_launch(qwen), weights_dir="/w", container_name="c")
    assert "--ulimit" not in vllm


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_launcher_writes_the_options_and_runs_the_wrapper(qwen: ModelSpec, tmp_path) -> None:
    # A stand-in python3 on PATH that records its argv and the options file it was given.
    fake = tmp_path / "bin" / "python3"
    fake.parent.mkdir()
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "cfg = sys.argv[-1]\n"
        "json.dump({'argv': sys.argv[1:], 'cfg': open(cfg).read()},"
        " open(os.environ['LOOM_FAKE_OUT'], 'w'))\n"
    )
    fake.chmod(0o755)
    spec = on_trtllm(qwen, {"kv_cache_config": {"free_gpu_memory_fraction": 0.85}})
    launch = render_launch(spec)
    argv = engine_process_argv(launch, host="127.0.0.1", weights_dir="/opt/loom/hf")
    paths = []
    for run in range(2):
        out = tmp_path / f"out{run}.json"
        path_env = f"{fake.parent}:{os.environ['PATH']}"
        env = {**os.environ, "PATH": path_env, "LOOM_FAKE_OUT": str(out)}
        subprocess.run(argv, env=env, capture_output=True, text=True, check=True)
        seen = json.loads(out.read_text())
        args = seen["argv"]
        # python3 -c <the wrapper source> serve <launch args after the options> ...
        assert args[:3] == ["-c", TRTLLM_SERVE_SOURCE, "serve"]
        assert args[3:-2] == argv[6:]
        assert args[-2] == "--extra_llm_api_options"
        path = Path(args[-1])
        paths.append(path)
        try:
            assert path.name.startswith("loom-trtllm.") and path.stat().st_mode & 0o077 == 0
            assert json.loads(seen["cfg"]) == trtllm_options(spec)
        finally:
            path.unlink(missing_ok=True)
    assert paths[0] != paths[1]  # each start gets its own file


def test_the_wrapper_source_compiles_as_one_program() -> None:
    # detokenize.py's source, then trtllm_scorer.py's: valid Python on its own, with the
    # wrapper's entry point last.
    compile(TRTLLM_SERVE_SOURCE, "<trtllm-serve wrapper>", "exec")
    assert TRTLLM_SERVE_SOURCE.rstrip().endswith('if __name__ == "__main__":\n    main()')


def test_pre_release_engine_versions_are_accepted(qwen: ModelSpec) -> None:
    assert on_trtllm(qwen).engine.version == "1.4.0rc2"


def test_the_modelopt_fp8_row_serves_its_checkpoint_with_an_fp8_kv_cache() -> None:
    spec = load_registry().get("qwen3-8b-fp8-modelopt")
    launch = render_launch(spec)
    assert launch.engine == "trtllm" and launch.image == TRTLLM_IMAGE
    assert launch.served_model == "qwen3-8b-fp8-modelopt"
    assert launch.args[1] == (
        f"{CONTAINER_HF_HOME}/hub/models--nvidia--Qwen3-8B-FP8/snapshots/"
        "2cebc4c89e25abc17668c81b01dceaf3d8b914d5"
    )
    assert json.loads(launch.args[0])["kv_cache_config"] == {"dtype": "fp8"}
    assert "--quantization" not in launch.args  # read from the checkpoint's config


def test_metrics_paths() -> None:
    assert metrics_path("trtllm") == "/prometheus/metrics"
    assert metrics_path("vllm") == metrics_path("sglang") == "/metrics"


def test_the_trtllm_parser_moves_only_trtllm_config_hashes(qwen: ModelSpec) -> None:
    # Adding `trtllm: qwen3` to the registry must not move a stored vLLM or SGLang hash.
    assert "trtllm" in qwen.tool_call_parsers
    assert set(serving_model_doc(qwen)["tool_call_parsers"]) == HASHED_PARSER_ENGINES
    trt = serving_model_doc(on_trtllm(qwen))["tool_call_parsers"]
    assert trt == {"vllm": "hermes", "sglang": "qwen25", "trtllm": "qwen3"}


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_bash_reads_the_rendered_engine_command_back_exactly(qwen: ModelSpec) -> None:
    # The pod scripts carry ENGINE_CMD as a bash array; the wrapper source spans lines.
    from loom_bench.providers.aws_ssm import _assign

    argv = engine_process_argv(render_launch(on_trtllm(qwen)), host="127.0.0.1")
    program = _assign("ENGINE_CMD", argv) + '\nprintf "%s\\0" "${ENGINE_CMD[@]}"\n'
    out = subprocess.run(["bash", "-c", program], capture_output=True, check=True).stdout
    assert out.decode().split("\0")[:-1] == argv
