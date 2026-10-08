"""Render a registry `ModelSpec` into an exact engine invocation.

`render_launch` turns a model into an `EngineLaunch`: the engine's CLI args after
its entrypoint. Experiment variants patch the spec before it gets here
(`experiment.apply_variant`). `docker_run_argv` wraps the launch into the
`docker run` used on a GPU host; `engine_process_argv` is the bare process used
where the host is the engine's own container (RunPod).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from loom_bench.providers.base import EngineLaunch
from loom_bench.registry import ModelSpec

ENTRYPOINTS: dict[str, tuple[str, ...]] = {
    "vllm": ("vllm", "serve"),
    "sglang": ("python3", "-m", "sglang.launch_server"),
}

CONTAINER_HF_HOME = "/root/.cache/huggingface"
ENGINE_PORT = 8000

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
}

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
    """Render registry `engine.args`.

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


def render_launch(spec: ModelSpec) -> EngineLaunch:
    """Render the engine invocation for `spec`; an engine without a renderer is an error."""
    renderer = ARG_RENDERERS.get(spec.engine.name)
    if renderer is None:
        raise ValueError(
            f"no launch renderer for engine {spec.engine.name!r} "
            f"(known: {', '.join(sorted(ARG_RENDERERS))}); see docs/how-to/add-engine.md"
        )
    args = renderer(spec, render_args(spec.engine.name, spec.engine.args))
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


def engine_process_argv(launch: EngineLaunch, *, host: str) -> list[str]:
    """The engine run as a plain process (no container runtime), listening on `host`.

    For providers whose host is itself the engine's container (RunPod): the
    entrypoint plus `launch.args` with the `--host` value replaced, so the config
    hash, which covers `launch.args`, is unchanged. `launch.env` is validated (no
    secret names); the caller sets it in the process environment.
    """
    if launch.engine not in ENTRYPOINTS:
        raise ValueError(f"no entrypoint for engine {launch.engine!r}")
    validate_engine_env(launch.env)
    args = list(launch.args)
    at = [i for i, a in enumerate(args) if a == "--host"]
    if len(at) != 1 or at[0] + 1 >= len(args):
        raise ValueError("engine args must set --host exactly once")
    args[at[0] + 1] = host
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
