import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from loom_bench.registry import DEFAULT_MODELS_YAML, Registry, load_registry, read_yaml

DIGEST = "sha256:" + "a" * 64
SHA = "0" * 40


def model(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "m-1",
        "display_name": "M 1",
        "hf": {
            "repo": "org/m-1",
            "revision": SHA,
            "license": "apache-2.0",
            "gated": False,
            "size_bytes": 1,
        },
        "engine": {"name": "vllm", "version": "0.30.0", "image": f"vllm/vllm-openai@{DIGEST}"},
        "hardware": {"gpu": "L40S", "gpus_per_replica": 1, "instance_types": {"aws": "g6e.xlarge"}},
        "parallelism": {"tp": 1},
        "max_context": 4096,
        "quantization": "none",
        "pricing": None,
        "scaling": {"min_replicas": 0, "max_replicas": 1},
        "clouds": ["aws"],
        "capabilities": {"tools": True},
        "routing_tier": 1,
        "status": "preview",
    }
    for dotted, value in overrides.items():
        *parents, leaf = dotted.split("__")
        node = base
        for p in parents:
            node = node[p]
        node[leaf] = value
    return base


def validate(*models: dict[str, Any]) -> Registry:
    return Registry.model_validate({"models": [copy.deepcopy(m) for m in models]})


PRICING = {"input_per_mtok": 100_000, "output_per_mtok": 300_000, "cached_input_per_mtok": 50_000}


def test_real_registry_loads():
    reg = load_registry()
    assert [m.id for m in reg.models] == ["qwen3-8b", "llama-3.3-70b-instruct"]
    assert reg.enabled() == []

    qwen = reg.get("qwen3-8b")
    assert qwen.hf.revision == "b968826d9c46dd6066d109eabc6255188de91218"
    assert qwen.engine.chat_template_kwargs == {"enable_thinking": False}
    assert qwen.hardware.instance_types.aws == "g6e.xlarge"
    assert qwen.capabilities.reasoning

    llama = reg.get("llama-3.3-70b-instruct")
    assert llama.hf.gated == "manual"
    assert (llama.hardware.gpus_per_replica, llama.parallelism.tp) == (4, 4)
    assert llama.max_context == 32768
    assert all(m.pricing is None and m.status == "preview" for m in reg.models)


def test_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [model(id="only-one")]}))
    monkeypatch.setenv("LOOM_MODELS_YAML", str(path))
    assert [m.id for m in load_registry().models] == ["only-one"]
    assert load_registry(DEFAULT_MODELS_YAML).models[0].id == "qwen3-8b"


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="nope"):
        validate(model()).get("nope")


def test_minimal_model_is_valid_and_enabled_with_pricing():
    reg = validate(model(), model(id="m-2", status="enabled", pricing=PRICING))
    assert [m.id for m in reg.enabled()] == ["m-2"]
    assert reg.get("m-1").kv_cache_dtype == "auto"
    assert reg.get("m-1").hf.trust_remote_code is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"engine__image": "vllm/vllm-openai:v0.30.0"}, "image"),
        ({"engine__image": f"vllm/vllm-openai:v0.30.0@{DIGEST}"}, "image"),
        ({"engine__image": "vllm/vllm-openai@sha256:abc"}, "image"),
        ({"hf__revision": "b968826"}, "revision"),
        ({"hf__revision": "main"}, "revision"),
        ({"parallelism": {"tp": 2}}, "must equal tp"),
        ({"hardware__gpus_per_replica": 4, "parallelism": {"tp": 2, "pp": 1}}, "must equal tp"),
        ({"hardware__gpus_per_replica": 4, "parallelism": {"tp": 4, "ep": 3}}, "ep"),
        ({"hardware__nodes_per_replica": 2, "parallelism": {"tp": 2}}, "reserved"),
        ({"hf__trust_remote_code": True}, "trust_remote_code_review"),
        ({"status": "enabled"}, "requires pricing"),
        ({"pricing": {**PRICING, "cached_input_per_mtok": 100_001}}, "cached_input"),
        ({"pricing": {**PRICING, "input_per_mtok": 0.1}}, "input_per_mtok"),
        ({"scaling": {"min_replicas": 2, "max_replicas": 1}}, "max_replicas"),
        ({"clouds": ["gcp"]}, "instance_types missing"),
        ({"quantization": "int3"}, "quantization"),
        ({"engine__name": "tgi"}, "name"),
        ({"capabilities__audio": True}, "audio"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_rejects_bad_model(overrides: dict[str, Any], message: str):
    with pytest.raises(ValidationError, match=message):
        validate(model(**overrides))


def test_trust_remote_code_with_review_is_allowed():
    review = {"reviewer": "arul", "date": "2026-10-04", "notes": "read modeling_*.py at SHA"}
    m = validate(model(hf__trust_remote_code=True, hf__trust_remote_code_review=review))
    assert m.models[0].hf.trust_remote_code_review is not None


def test_ep_dividing_gpus_is_allowed():
    validate(model(hardware__gpus_per_replica=4, parallelism={"tp": 4, "ep": 2}))


def test_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="duplicate model ids"):
        validate(model(), model())


def test_rejects_unknown_top_level_key():
    with pytest.raises(ValidationError, match="extra"):
        Registry.model_validate({"models": [], "defaults": {}})


def test_yaml_duplicate_keys_rejected(tmp_path: Path):
    path = tmp_path / "dup.yaml"
    path.write_text("models: []\nmodels: []\n")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key 'models'"):
        read_yaml(path)
