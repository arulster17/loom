"""Render a registry `ModelSpec` into an exact engine invocation.

`render_launch` turns a model into an `EngineLaunch`: the engine's CLI args after
its entrypoint. Experiment variants patch the spec before it gets here
(`experiment.apply_variant`). `docker_run_argv` wraps the launch into the
`docker run` used on a GPU host; `engine_process_argv` is the bare process used
where the host is the engine's own container (RunPod).

TensorRT-LLM (`trtllm`) differs from vLLM and SGLang in three ways, all handled here:
its `engine.args` are LLM-API options (the YAML `trtllm-serve --extra_llm_api_options`
reads, e.g. `{kv_cache_config: {enable_block_reuse: false}}`), not CLI flags, so they go
through a small launcher that writes them to a file; it loads the checkpoint from its HF
cache snapshot directory (the pinned revision is in the path, and nothing resolves a hub
ref offline); and it serves Prometheus metrics at `/prometheus/metrics` (`metrics_path`).
docs/perf/tensorrt-llm.md has the reasons and the verified version facts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loom_bench.providers.base import EngineLaunch
from loom_bench.registry import ModelSpec

# TensorRT-LLM's launcher. $1 is the Python source of `trtllm-serve` with Loom's
# prompt-logprobs route (`TRTLLM_SERVE_SOURCE`); $2 the LLM-API options as JSON (JSON is
# YAML), written to a private temp file for `--extra_llm_api_options`; the rest are
# `trtllm-serve serve` arguments. `trtllm-launch` is only bash's $0 (the process name).
TRTLLM_LAUNCHER = (
    'set -eu; src="$1"; cfg="$(mktemp /tmp/loom-trtllm.XXXXXX)"; printf "%s\\n" "$2" >"$cfg"; '
    'shift 2; exec python3 -c "$src" serve "$@" --extra_llm_api_options "$cfg"'
)
# loom_bench/trtllm_scorer.py after the detokenize.py it renders tokens with, run by the
# image's python3 (loom_bench is not installed there). Code, not config: it changes only
# the answer to echo+logprobs completions requests, which trtllm-serve itself cannot give.
TRTLLM_SERVE_SOURCE = (
    (Path(__file__).parent / "detokenize.py").read_text()
    + "\n\n"
    + (Path(__file__).parent / "trtllm_scorer.py").read_text()
)

ENTRYPOINTS: dict[str, tuple[str, ...]] = {
    "vllm": ("vllm", "serve"),
    "sglang": ("python3", "-m", "sglang.launch_server"),
    "trtllm": ("bash", "-c", TRTLLM_LAUNCHER, "trtllm-launch", TRTLLM_SERVE_SOURCE),
}

CONTAINER_HF_HOME = "/root/.cache/huggingface"
ENGINE_PORT = 8000

# Prometheus path per engine (default `/metrics`). TensorRT-LLM's `/metrics` is JSON
# iteration stats; its Prometheus text is at `/prometheus/metrics`, mounted only with the
# `return_perf_metrics` option, which `trtllm_options` always sets (as SGLang always gets
# `--enable-metrics`); its running/waiting/KV gauges also need `enable_iter_perf_stats`.
# Verified in tensorrt_llm/serve/openai_server.py at v1.2.1 and v1.4.0rc2.
METRICS_PATHS: dict[str, str] = {"trtllm": "/prometheus/metrics"}


def metrics_path(engine: str) -> str:
    """The path of `engine`'s Prometheus text exposition."""
    return METRICS_PATHS.get(engine, "/metrics")


_SGLANG_KV_CACHE_DTYPE = {"fp8": "fp8_e4m3", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2"}

# Flags rendered from structured registry fields; `engine.args` may not set them.
_RESERVED_FLAGS: dict[str, frozenset[str]] = {
    "vllm": frozenset(
        {
            "model",
            "revision",
            "tokenizer",
            "tokenizer-revision",
            "served-model-name",
            "tensor-parallel-size",
            "tp",
            "pipeline-parallel-size",
            "pp",
            "max-model-len",
            "quantization",
            "kv-cache-dtype",
            "host",
            "port",
            "download-dir",
            "trust-remote-code",
            "enable-auto-tool-choice",
            "tool-call-parser",
        }
    ),
    "sglang": frozenset(
        {
            "model-path",
            "model",
            "revision",
            "tokenizer-path",
            "served-model-name",
            "tp-size",
            "tp",
            "pp-size",
            "pp",
            "context-length",
            "quantization",
            "kv-cache-dtype",
            "enable-metrics",
            "host",
            "port",
            "download-dir",
            "trust-remote-code",
            "tool-call-parser",
        }
    ),
    # LLM-API option names (snake_case) and serve flags, set from the registry by
    # `_trtllm_args` / `trtllm_options`.
    "trtllm": frozenset(
        {
            "model",
            "tokenizer",
            "revision",
            "backend",
            "tensor_parallel_size",
            "tp_size",
            "pipeline_parallel_size",
            "pp_size",
            "max_seq_len",
            "trust_remote_code",
            "tool_parser",
            "guided_decoding_backend",
            "return_perf_metrics",
            "enable_iter_perf_stats",
            "served_model_name",
            "host",
            "port",
        }
    ),
}

# Engines whose `engine.args` are an options document rather than CLI flags.
OPTION_ENGINES = frozenset({"trtllm"})

_FLAG_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_ENV_RE = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|(^|_)KEY($|_)")


def _flag(key: str) -> str:
    name = key.replace("_", "-")
    if not _FLAG_RE.match(name):
        raise ValueError(f"invalid engine arg name {key!r}")
    return name


def _render_value(flag: str, value: Any) -> list[str]:
    if isinstance(value, bool):
        return [f"--{flag}"] if value else []
    if isinstance(value, int | float | str):
        return [f"--{flag}", str(value)]
    if isinstance(value, list) and value:
        if not all(isinstance(v, int | float | str) and not isinstance(v, bool) for v in value):
            raise ValueError(f"--{flag}: list values must be scalars")
        return [f"--{flag}", *(str(v) for v in value)]
    if isinstance(value, dict):
        return [f"--{flag}", json.dumps(value, sort_keys=True, separators=(",", ":"))]
    raise ValueError(f"--{flag}: unsupported value {value!r}")


def render_args(engine: str, args: dict[str, Any]) -> list[str]:
    """Render registry `engine.args` as CLI flags (not for `OPTION_ENGINES`).

    `{max_num_seqs: 256, enable_prefix_caching: true, disable_log_requests: false}`
    renders as `--max-num-seqs 256 --enable-prefix-caching`.
    """
    out: list[str] = []
    reserved = _RESERVED_FLAGS[engine]
    for key, value in args.items():
        flag = _flag(key)
        if flag == "trust-remote-code":
            raise ValueError("trust_remote_code comes only from hf.trust_remote_code with a review")
        if flag in reserved:
            raise ValueError(f"engine arg {key!r} is set from the registry; change it there")
        out += _render_value(flag, value)
    return out


def quantization_flag(spec: ModelSpec) -> list[str]:
    """`--quantization` from the checkpoint format and the serving precision.

    Same for vLLM and SGLang:

    | `hf.quant_method`  | `quantization`          | flag                 |
    |--------------------|-------------------------|----------------------|
    | null (BF16/FP16)   | none                    | none                 |
    | null (BF16/FP16)   | fp8                     | `--quantization fp8` |
    | fp8                | fp8                     | none                 |
    | compressed-tensors | fp8, w8a8, w4a16, nvfp4 | none                 |
    | awq                | awq                     | none                 |
    | gptq               | gptq                    | none                 |
    | modelopt           | fp8, nvfp4              | none                 |

    FP8 is the only precision the engines apply on load to unquantized weights. A
    pre-quantized checkpoint names its method in `config.json`, which the engine
    reads; a different `--quantization` makes it refuse to start, so none is
    passed. Other pairs are rejected by the registry (`CHECKPOINT_PRECISIONS`).
    """
    if spec.hf.quant_method is None and spec.quantization == "fp8":
        return ["--quantization", "fp8"]
    return []


def _trust_remote_code(spec: ModelSpec) -> list[str]:
    if spec.hf.trust_remote_code and spec.hf.trust_remote_code_review is not None:
        return ["--trust-remote-code"]
    return []


def _tool_call_parser(spec: ModelSpec) -> str | None:
    """The engine's tool-call parser when the model serves tools, else None."""
    if not spec.capabilities.tools:
        return None
    parser = spec.tool_call_parsers.get(spec.engine.name)
    if parser is None:
        raise ValueError(
            f"{spec.id}: capabilities.tools needs tool_call_parsers.{spec.engine.name}"
        )
    return parser


def _vllm_args(spec: ModelSpec, extra: list[str]) -> list[str]:
    parser = _tool_call_parser(spec)
    tools = ["--enable-auto-tool-choice", "--tool-call-parser", parser] if parser else []
    return [
        spec.hf.repo,
        "--revision",
        spec.hf.revision,
        "--tokenizer-revision",
        spec.hf.revision,
        "--served-model-name",
        spec.id,
        "--tensor-parallel-size",
        str(spec.parallelism.tp),
        "--pipeline-parallel-size",
        str(spec.parallelism.pp),
        "--max-model-len",
        str(spec.max_context),
        *quantization_flag(spec),
        "--kv-cache-dtype",
        spec.kv_cache_dtype,
        *_trust_remote_code(spec),
        *tools,
        "--host",
        "0.0.0.0",
        "--port",
        str(ENGINE_PORT),
        *extra,
    ]


def _sglang_args(spec: ModelSpec, extra: list[str]) -> list[str]:
    parser = _tool_call_parser(spec)
    tools = ["--tool-call-parser", parser] if parser else []
    pp = ["--pp-size", str(spec.parallelism.pp)] if spec.parallelism.pp > 1 else []
    kv = (
        []
        if spec.kv_cache_dtype == "auto"
        else ["--kv-cache-dtype", _SGLANG_KV_CACHE_DTYPE[spec.kv_cache_dtype]]
    )
    return [
        "--model-path",
        spec.hf.repo,
        "--revision",
        spec.hf.revision,
        "--served-model-name",
        spec.id,
        "--tp-size",
        str(spec.parallelism.tp),
        *pp,
        "--context-length",
        str(spec.max_context),
        *quantization_flag(spec),
        *kv,
        *_trust_remote_code(spec),
        *tools,
        "--enable-metrics",
        "--host",
        "0.0.0.0",
        "--port",
        str(ENGINE_PORT),
        *extra,
    ]


_OPTION_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# TensorRT-LLM's FP8 KV cache is E4M3 (`kv_cache_config.dtype: fp8`).
_TRTLLM_KV_CACHE_DTYPE = {"fp8": "fp8", "fp8_e4m3": "fp8"}


def _trtllm_snapshot(spec: ModelSpec) -> str:
    """The checkpoint's HF cache snapshot directory in the engine's container.

    The path names the pinned revision, so loading from it pins the weights, tokenizer
    and chat template to it with no hub lookup (offline, a hub id resolves `main`, which
    a download by commit never records). `engine_process_argv` moves it to the host's
    weights directory where that differs (RunPod).
    """
    folder = "models--" + spec.hf.repo.replace("/", "--")
    return f"{CONTAINER_HF_HOME}/hub/{folder}/snapshots/{spec.hf.revision}"


def trtllm_options(spec: ModelSpec) -> dict[str, Any]:
    """`engine.args` as TensorRT-LLM LLM-API options, plus those set from the registry.

    Set here: `guided_decoding_backend: xgrammar` (`response_format` needs a grammar
    backend; vLLM and SGLang use xgrammar by default, TensorRT-LLM has none by default),
    `return_perf_metrics: true` (mounts the Prometheus endpoint), `enable_iter_perf_stats:
    true` (feeds its running, waiting and KV-cache gauges; v1.4.0rc2 openai_server.py
    starts the collector only with it) and, unless `auto`, `kv_cache_config.dtype`.
    Everything else stays the engine's default unless `engine.args` sets it
    (`kv_cache_config` is merged).
    """
    out: dict[str, Any] = {}
    for key, value in spec.engine.args.items():
        if not _OPTION_RE.match(key):
            raise ValueError(f"invalid TensorRT-LLM option name {key!r}")
        if key in _RESERVED_FLAGS["trtllm"]:
            raise ValueError(f"engine arg {key!r} is set from the registry; change it there")
        if key == "speculative_config":
            raise ValueError(
                "speculative_config on trtllm is not wired: its draft checkpoint would not "
                "be downloaded (engines.draft_weights)"
            )
        out[key] = value
    kv = out.get("kv_cache_config", {})
    if not isinstance(kv, dict):
        raise ValueError("kv_cache_config must be a mapping")
    if "dtype" in kv:
        raise ValueError("kv_cache_config.dtype is set from the registry's kv_cache_dtype")
    if spec.kv_cache_dtype != "auto":
        if spec.kv_cache_dtype not in _TRTLLM_KV_CACHE_DTYPE:
            raise ValueError(f"trtllm has no {spec.kv_cache_dtype} KV cache (its FP8 is E4M3)")
        kv = {**kv, "dtype": _TRTLLM_KV_CACHE_DTYPE[spec.kv_cache_dtype]}
    if kv:
        out["kv_cache_config"] = kv
    out["guided_decoding_backend"] = "xgrammar"
    out["return_perf_metrics"] = True
    out["enable_iter_perf_stats"] = True
    return out


def _trtllm_quantization(spec: ModelSpec) -> None:
    """TensorRT-LLM's PyTorch backend reads a pre-quantized checkpoint's format from its
    files (ModelOpt `hf_quant_config.json`, or `quantization_config` with quant_method
    fp8) and has no flag to quantize BF16 weights on load. It reads no compressed-tensors
    checkpoint (tensorrt_llm/_torch/model_config.py at v1.2.1 and v1.4.0rc2)."""
    method = spec.hf.quant_method
    if method is None and spec.quantization == "fp8":
        raise ValueError(
            f"{spec.id}: trtllm does not quantize BF16 weights on load; serve a "
            "pre-quantized FP8 checkpoint (ModelOpt) instead"
        )
    if method not in (None, "modelopt", "fp8"):
        raise ValueError(f"{spec.id}: trtllm cannot load {method} checkpoints")


def _trtllm_args(spec: ModelSpec, extra: list[str]) -> list[str]:
    """`[options JSON, trtllm-serve serve args...]` for `TRTLLM_LAUNCHER`.

    The PyTorch backend (trtllm-serve's default and, in 1.4.0rc2, its only choice; named
    so the config hash records it) loads the HF checkpoint directly: no TensorRT engine
    build. `--served_model_name` (new in 1.4.0rc2) serves it as `spec.id`, as the other
    engines do.
    """
    if extra:
        raise ValueError("trtllm takes engine.args as options, not flags")
    _trtllm_quantization(spec)
    parser = _tool_call_parser(spec)
    tools = ["--tool_parser", parser] if parser else []
    options = json.dumps(trtllm_options(spec), sort_keys=True, separators=(",", ":"))
    return [
        options,
        _trtllm_snapshot(spec),
        "--backend",
        "pytorch",
        "--tp_size",
        str(spec.parallelism.tp),
        "--pp_size",
        str(spec.parallelism.pp),
        "--max_seq_len",
        str(spec.max_context),
        *(["--trust_remote_code"] if _trust_remote_code(spec) else []),
        *tools,
        "--served_model_name",
        spec.id,
        "--host",
        "0.0.0.0",
        "--port",
        str(ENGINE_PORT),
    ]


def validate_engine_env(env: dict[str, str]) -> dict[str, str]:
    for name in env:
        if not _ENV_NAME_RE.match(name):
            raise ValueError(f"invalid environment variable name {name!r}")
        if _SECRET_ENV_RE.search(name):
            raise ValueError(f"{name}: secrets never go into an engine launch")
    return dict(env)


# Each engine's CLI renderer: (spec, rendered engine.args) -> args after the entrypoint.
ARG_RENDERERS: dict[str, Callable[[ModelSpec, list[str]], list[str]]] = {
    "vllm": _vllm_args,
    "sglang": _sglang_args,
    "trtllm": _trtllm_args,
}


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
# vLLM speculative methods that need no draft checkpoint (n-gram/suffix lookups, MTP heads
# inside the target model).
_VLLM_DRAFTLESS = frozenset({"ngram", "[ngram]", "suffix", "mtp"})


def _flag_value(args: list[str], flag: str) -> str | None:
    at = [i for i, a in enumerate(args) if a == flag]
    if len(at) > 1:
        raise ValueError(f"{flag} is set more than once")
    if not at:
        return None
    if at[0] + 1 >= len(args):
        raise ValueError(f"{flag} has no value")
    return args[at[0] + 1]


def draft_weights(engine: str, args: list[str]) -> list[tuple[str, str]]:
    """The speculative-decoding draft checkpoints a launch loads, as (repo, revision).

    Hosts download them next to the target weights (the engine runs offline), so a
    draft must be a Hugging Face repo pinned to a 40-hex commit. vLLM names it in
    `--speculative-config {"model": ..., "revision": ...}`; SGLang in
    `--speculative-draft-model-path` / `--speculative-draft-model-revision`.
    """
    if engine == "vllm":
        raw = _flag_value(args, "--speculative-config")
        if raw is None:
            return []
        cfg = json.loads(raw)
        if not isinstance(cfg, dict):
            raise ValueError("--speculative-config must be a JSON object")
        method, repo = cfg.get("method"), cfg.get("model")
        if repo is None or method in _VLLM_DRAFTLESS or repo in _VLLM_DRAFTLESS:
            return []
        revision = cfg.get("revision")
    elif engine == "sglang":
        repo = _flag_value(args, "--speculative-draft-model-path")
        if repo is None:
            return []
        revision = _flag_value(args, "--speculative-draft-model-revision")
    else:
        return []
    if not isinstance(repo, str) or repo.startswith(("/", ".")) or repo.count("/") != 1:
        raise ValueError(f"speculative draft {repo!r} must be a Hugging Face repo id (owner/name)")
    if not isinstance(revision, str) or not _COMMIT_RE.match(revision):
        raise ValueError(
            f"speculative draft {repo}: pin its revision to a 40-hex commit (got {revision!r})"
        )
    return [(repo, revision)]


def launch_weights(launch: EngineLaunch) -> list[tuple[str, str]]:
    """Every pinned checkpoint the engine loads, as (repo, revision): the model, then any
    speculative draft. The engine runs offline, so the host must hold all of them."""
    model = (launch.model_repo, launch.model_revision)
    return list(dict.fromkeys([model, *draft_weights(launch.engine, launch.args)]))


def render_launch(spec: ModelSpec) -> EngineLaunch:
    """Render the engine invocation for `spec`; an engine without a renderer is an error."""
    renderer = ARG_RENDERERS.get(spec.engine.name)
    if renderer is None:
        raise ValueError(
            f"no launch renderer for engine {spec.engine.name!r} "
            f"(known: {', '.join(sorted(ARG_RENDERERS))}); see docs/how-to/add-engine.md"
        )
    name = spec.engine.name
    args = renderer(spec, [] if name in OPTION_ENGINES else render_args(name, spec.engine.args))
    draft_weights(spec.engine.name, args)  # a draft must be a pinned HF repo
    return EngineLaunch(
        engine=spec.engine.name,
        image=spec.engine.image,
        model_repo=spec.hf.repo,
        model_revision=spec.hf.revision,
        served_model=spec.id,
        args=args,
        port=ENGINE_PORT,
        gpus=spec.parallelism.tp * spec.parallelism.pp,
    )


def engine_process_argv(
    launch: EngineLaunch, *, host: str, weights_dir: str = CONTAINER_HF_HOME
) -> list[str]:
    """The engine run as a plain process (no container runtime), listening on `host`.

    For providers whose host is itself the engine's container (RunPod): the
    entrypoint plus `launch.args` with the `--host` value replaced, and a checkpoint
    path under the container HF cache (`CONTAINER_HF_HOME`: TensorRT-LLM's model
    argument) moved to `weights_dir`, so the config hash, which covers `launch.args`,
    is unchanged. `launch.env` is validated (no secret names); the caller sets it in
    the process environment.
    """
    if launch.engine not in ENTRYPOINTS:
        raise ValueError(f"no entrypoint for engine {launch.engine!r}")
    validate_engine_env(launch.env)
    args = list(launch.args)
    at = [i for i, a in enumerate(args) if a == "--host"]
    if len(at) != 1 or at[0] + 1 >= len(args):
        raise ValueError("engine args must set --host exactly once")
    args[at[0] + 1] = host
    prefix = CONTAINER_HF_HOME + "/"
    home = weights_dir.rstrip("/") + "/"
    args = [home + a.removeprefix(prefix) if a.startswith(prefix) else a for a in args]
    return [*ENTRYPOINTS[launch.engine], *args]


def docker_run_argv(launch: EngineLaunch, *, weights_dir: str, container_name: str) -> list[str]:
    """`docker run` for the engine container on a GPU host.

    Weights are pre-downloaded into `weights_dir` (an HF cache) before this runs, so
    the engine starts with `HF_HUB_OFFLINE=1` and never sees a Hugging Face token.
    `launch.env` is passed by name only (`-e NAME`); the caller exports the values.
    The port is published on loopback only: load is generated on the host itself.
    """
    if launch.engine not in ENTRYPOINTS:
        raise ValueError(f"no container entrypoint for engine {launch.engine!r}")
    entry = ENTRYPOINTS[launch.engine]
    argv = [
        "docker",
        "run",
        "--detach",
        "--name",
        container_name,
        "--label",
        "loom.managed=true",
        "--gpus",
        str(launch.gpus),
        "--ipc",
        "host",
        "--publish",
        f"127.0.0.1:{launch.port}:{launch.port}",
        "--volume",
        f"{weights_dir}:{CONTAINER_HF_HOME}",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
    ]
    for name in sorted(validate_engine_env(launch.env)):
        argv += ["--env", name]
    argv += ["--entrypoint", entry[0], launch.image, *entry[1:], *launch.args]
    return argv
