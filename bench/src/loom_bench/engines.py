"""Render a registry `ModelSpec` into an exact engine invocation.

`render_launch` turns a model (plus validated experiment overrides) into an
`EngineLaunch`: the engine's CLI args after its entrypoint. `docker_run_argv`
wraps that into the `docker run` used on a GPU host.

Quantization: registry `quantization` names the precision being served. vLLM and
SGLang read the method of a pre-quantized checkpoint (AWQ, GPTQ, compressed-tensors
W4A16/W8A8/FP8-dynamic, ModelOpt FP4) from its `config.json`, and passing a
different `--quantization` makes them refuse to start, so those formats emit no
flag. Only `fp8` emits `--quantization fp8`, which quantizes a BF16 checkpoint on
load and matches checkpoints saved with `quant_method: fp8`. A checkpoint whose
config names another method (e.g. RedHatAI `*-FP8-dynamic`, compressed-tensors)
sets `engine.args.quantization` to that method, or to `false` to emit nothing;
an explicit `quantization` arg always replaces the derived flag.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from loom_bench.providers.base import EngineLaunch
from loom_bench.registry import KvCacheDtype, ModelSpec, PinnedImage, Quantization

EngineName = Literal["vllm", "sglang"]

ENTRYPOINTS: dict[str, tuple[str, ...]] = {
    "vllm": ("vllm", "serve"),
    "sglang": ("python3", "-m", "sglang.launch_server"),
}

CONTAINER_HF_HOME = "/root/.cache/huggingface"

_ON_LOAD_QUANTIZATION: dict[str, dict[str, str]] = {
    "vllm": {"fp8": "fp8"},
    "sglang": {"fp8": "fp8"},
}
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
            "kv-cache-dtype",
            "host",
            "port",
            "download-dir",
            "trust-remote-code",
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
            "kv-cache-dtype",
            "enable-metrics",
            "host",
            "port",
            "download-dir",
            "trust-remote-code",
        }
    ),
}

_FLAG_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_SECRET_ENV_RE = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|(^|_)KEY($|_)")


class EngineOverrides(BaseModel):
    """Experiment-level patches to a registry model's serving config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: EngineName | None = None
    version: str | None = None
    image: PinnedImage | None = None
    # Merged onto the registry args; a None value removes that key. When the engine
    # changes, registry args (which belong to the other engine) are not carried over.
    args: dict[str, Any] | None = None
    tp: Annotated[int, Field(ge=1)] | None = None
    pp: Annotated[int, Field(ge=1)] | None = None
    quantization: Quantization | None = None
    kv_cache_dtype: KvCacheDtype | None = None
    max_context: Annotated[int, Field(ge=1)] | None = None
    port: Annotated[int, Field(ge=1024, le=65535)] | None = None
    env: dict[str, str] | None = None
    ready_timeout_s: Annotated[float, Field(gt=0)] | None = None


def apply_overrides(spec: ModelSpec, overrides: EngineOverrides) -> ModelSpec:
    """Return `spec` patched by `overrides`, re-validated by the registry rules."""
    doc = spec.model_dump()
    engine = doc["engine"]
    if overrides.engine is not None and overrides.engine != spec.engine.name:
        if overrides.image is None or overrides.version is None:
            raise ValueError(
                f"switching engine to {overrides.engine} requires image and version overrides"
            )
        engine.update(name=overrides.engine, args={})
    if overrides.version is not None:
        engine["version"] = overrides.version
    if overrides.image is not None:
        engine["image"] = overrides.image
    if overrides.args is not None:
        merged = dict(engine["args"])
        for key, value in overrides.args.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        engine["args"] = merged
    if overrides.tp is not None:
        doc["parallelism"]["tp"] = overrides.tp
    if overrides.pp is not None:
        doc["parallelism"]["pp"] = overrides.pp
    if overrides.quantization is not None:
        doc["quantization"] = overrides.quantization
    if overrides.kv_cache_dtype is not None:
        doc["kv_cache_dtype"] = overrides.kv_cache_dtype
    if overrides.max_context is not None:
        if overrides.max_context > spec.max_context:
            raise ValueError(
                f"max_context override {overrides.max_context} exceeds the registry cap "
                f"{spec.max_context} for {spec.id}"
            )
        doc["max_context"] = overrides.max_context
    return ModelSpec.model_validate(doc)


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


def render_args(engine: str, args: dict[str, Any]) -> tuple[list[str], str | bool | None]:
    """Render registry `engine.args`. Returns (argv, quantization arg if given).

    `{max_num_seqs: 256, enable_prefix_caching: true, disable_log_requests: false}`
    renders as `--max-num-seqs 256 --enable-prefix-caching`.
    """
    out: list[str] = []
    quantization: str | bool | None = None
    reserved = _RESERVED_FLAGS[engine]
    for key, value in args.items():
        flag = _flag(key)
        if flag == "trust-remote-code":
            raise ValueError("trust_remote_code comes only from hf.trust_remote_code with a review")
        if flag in reserved:
            raise ValueError(f"engine arg {key!r} is set from the registry; change it there")
        if flag == "quantization":
            if value is True or not isinstance(value, str | bool):
                raise ValueError("engine arg quantization must be a method name or false")
            quantization = value
            continue
        out += _render_value(flag, value)
    return out, quantization


def _quantization(spec: ModelSpec, arg: str | bool | None) -> list[str]:
    if arg is False:
        return []
    if isinstance(arg, str):
        return ["--quantization", arg]
    method = _ON_LOAD_QUANTIZATION[spec.engine.name].get(spec.quantization)
    return ["--quantization", method] if method else []


def _trust_remote_code(spec: ModelSpec) -> list[str]:
    if spec.hf.trust_remote_code and spec.hf.trust_remote_code_review is not None:
        return ["--trust-remote-code"]
    return []


def _vllm_args(spec: ModelSpec, port: int, extra: list[str], quant: list[str]) -> list[str]:
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
        *quant,
        "--kv-cache-dtype",
        spec.kv_cache_dtype,
        *_trust_remote_code(spec),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        *extra,
    ]


def _sglang_args(spec: ModelSpec, port: int, extra: list[str], quant: list[str]) -> list[str]:
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
        *quant,
        *kv,
        *_trust_remote_code(spec),
        "--enable-metrics",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        *extra,
    ]


def _validate_env(env: dict[str, str]) -> dict[str, str]:
    for name in env:
        if not _ENV_NAME_RE.match(name):
            raise ValueError(f"invalid environment variable name {name!r}")
        if _SECRET_ENV_RE.search(name):
            raise ValueError(f"{name}: secrets never go into an engine launch")
    return dict(env)


def render_launch(spec: ModelSpec, overrides: dict[str, Any] | None = None) -> EngineLaunch:
    """Render the engine invocation for `spec`, patched by experiment `overrides`."""
    ov = EngineOverrides.model_validate(overrides or {})
    spec = apply_overrides(spec, ov)
    port = ov.port or 8000
    extra, quant_arg = render_args(spec.engine.name, spec.engine.args)
    quant = _quantization(spec, quant_arg)
    if spec.engine.name == "vllm":
        args = _vllm_args(spec, port, extra, quant)
    else:
        args = _sglang_args(spec, port, extra, quant)
    launch = EngineLaunch(
        engine=spec.engine.name,
        image=spec.engine.image,
        model_repo=spec.hf.repo,
        model_revision=spec.hf.revision,
        served_model=spec.id,
        args=args,
        env=_validate_env(ov.env or {}),
        port=port,
        gpus=spec.parallelism.tp * spec.parallelism.pp,
    )
    if ov.ready_timeout_s is not None:
        launch = launch.model_copy(update={"ready_timeout_s": ov.ready_timeout_s})
    return launch


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
    for name in sorted(_validate_env(launch.env)):
        argv += ["--env", name]
    argv += ["--entrypoint", entry[0], launch.image, *entry[1:], *launch.args]
    return argv
