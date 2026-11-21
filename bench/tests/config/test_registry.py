import copy
import csv
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from loom_bench.engines import render_launch
from loom_bench.registry import (
    DEFAULT_MODELS_YAML,
    ModelSpec,
    Registry,
    load_registry,
    read_yaml,
)

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
        "tool_call_parsers": {"vllm": "hermes"},
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


def test_every_shipped_model_loads_and_can_be_launched():
    """Properties of whatever config/models.yaml ships, so adding a model is config only."""
    reg = load_registry()
    assert reg.models
    for m in reg.models:
        assert reg.get(m.id) is m
        assert m.hardware.gpus_per_replica == m.parallelism.tp * m.parallelism.pp
        launch = render_launch(m)
        assert (launch.model_repo, launch.model_revision) == (m.hf.repo, m.hf.revision)
        assert launch.served_model == m.id and launch.image == m.engine.image
        assert launch.gpus == m.hardware.gpus_per_replica


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
    assert reg.get("m-2").status == "enabled"
    assert reg.get("m-1").kv_cache_dtype == "auto"
    assert reg.get("m-1").hf.trust_remote_code is False
    assert reg.get("m-1").hf.quant_method is None


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
        ({"clouds": ["aws", "runpod"]}, "instance_types missing for clouds \\['runpod'\\]"),
        ({"clouds": ["azure"]}, "clouds"),
        ({"quantization": "int3"}, "quantization"),
        ({"quantization": "awq"}, "cannot be served from unquantized weights"),
        ({"hf__quant_method": "awq"}, "cannot be served from awq weights"),
        ({"hf__quant_method": "bitsandbytes", "quantization": "w4a16"}, "quant_method"),
        ({"engine__name": "tgi"}, "name"),
        ({"capabilities__audio": True}, "audio"),
        ({"tool_call_parsers": {}}, "capabilities.tools needs tool_call_parsers.vllm"),
        ({"tool_call_parsers": {"sglang": "qwen25"}}, "needs tool_call_parsers.vllm"),
        ({"tool_call_parsers": {"tgi": "hermes"}}, "tool_call_parsers"),
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


def quantized(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {"id": "m-1-fp8", "quantization": "fp8", "base_model": "m-1"}
    return model(**(fields | overrides))


def test_base_model_names_the_entry_whose_list_prices_apply():
    reg = validate(model(), quantized())
    assert reg.market_model_id("m-1-fp8") == "m-1"
    assert reg.market_model_id("m-1") == "m-1"


def test_shipped_quantized_entries_compare_with_their_base_models():
    reg = load_registry()
    assert reg.market_model_id("llama-3.3-70b-instruct-fp8") == "llama-3.3-70b-instruct"
    assert reg.market_model_id("qwen3-8b-fp8") == "qwen3-8b"


def test_base_model_is_left_out_of_dumps_so_config_hashes_do_not_move():
    from loom_bench.provenance import config_hash

    with_base = validate(model(), quantized()).get("m-1-fp8")
    without = validate(model(id="m-1-fp8", quantization="fp8")).get("m-1-fp8")
    assert "base_model" not in with_base.model_dump(mode="json")
    assert config_hash(with_base.model_dump(mode="json")) == config_hash(
        without.model_dump(mode="json")
    )


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ([quantized()], "base_model 'm-1' is not another entry"),
        ([model(base_model="m-1")], "is not another entry"),
        ([model(), quantized(quantization="none")], "same precision"),
        (
            [model(), quantized(), model(id="m-1-fp8-b", quantization="fp8", base_model="m-1-fp8")],
            "has a base_model of its own",
        ),
    ],
)
def test_rejects_bad_base_model(models: list[dict[str, Any]], message: str):
    with pytest.raises(ValidationError, match=message):
        validate(*models)


def test_an_enabled_snapshot_without_a_price_loads_but_the_registry_refuses_it():
    # A cell's stored model snapshot records pricing as null (serving_model_doc), so the
    # rule lives on Registry: the snapshot of an enabled row must still load.
    snap = ModelSpec.model_validate(model(status="enabled"))
    assert snap.status == "enabled" and snap.pricing is None
    with pytest.raises(ValidationError, match="m-1: status: enabled requires pricing"):
        validate(model(status="enabled"))


# --- prices (user decision 2026-10-11) ----------------------------------------------

REPORTS = DEFAULT_MODELS_YAML.parents[1] / "reports"
# (model id, report the cost comes from, its workload, its config): the shipped winners.
PRICED = [
    ("qwen3-8b", "8b-config-sweep-v1", "chat-sharegpt", "bf16"),
    ("llama-3.3-70b-instruct-fp8", "70b-h100-v2", "fixed-1k-1k", "vllm-tp2-fp8"),
]


def _cost_ci_high(report: str, model_id: str, workload: str, config: str) -> dict[str, int]:
    with (REPORTS / report / "leaderboard.competitiveness.csv").open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if (r["model_id"], r["workload"], r["best_config"]) == (model_id, workload, config)
        ]
    sides = {s: {int(r[f"cost_{s}_per_mtok_hi_micros"]) for r in rows} for s in ("input", "output")}
    assert rows and all(len(v) == 1 for v in sides.values()), (report, model_id)
    return {s: v.pop() for s, v in sides.items()}


@pytest.mark.parametrize(("model_id", "report", "workload", "config"), PRICED)
def test_shipped_prices_are_cost_plus_20_percent_on_the_ci_high_bound(
    model_id: str, report: str, workload: str, config: str
):
    # docs/how-to/price-a-model.md option A at u = 100%, markup 20%, per side, each
    # rounded up to the next $0.001 per 1M; cached input not discounted.
    price = load_registry().get(model_id).pricing
    assert price is not None
    hi = _cost_ci_high(report, model_id, workload, config)
    for side, value in (("input", price.input_per_mtok), ("output", price.output_per_mtok)):
        assert value == -(-hi[side] * 12 // 10_000) * 1_000, side
        assert value >= hi[side] * 1.2  # rounding never takes the markup under 20%
    assert price.cached_input_per_mtok == price.input_per_mtok


def test_shipped_prices_and_the_rows_left_unpriced():
    reg = load_registry()
    prices = {
        m.id: None if m.pricing is None else (m.pricing.input_per_mtok, m.pricing.output_per_mtok)
        for m in reg.models
    }
    assert prices == {
        "qwen3-8b": (77_000, 368_000),
        "llama-3.3-70b-instruct": None,  # the BF16 reference row, not the shipped winner
        "llama-3.3-70b-instruct-fp8": (571_000, 797_000),
        "qwen3-8b-fp8": None,  # fp8-kv8 not shipped (gate inconclusive)
    }
    assert {m.status for m in reg.models} == {"preview"}  # not yet approved for serving


def test_the_70b_rows_serve_the_measured_h100_config():
    # 9f0853d7 / 9e5f7866 (reports/70b-h100-v2): 2x H100 SXM, TP=2, vLLM v0.30.0 with
    # gpu_memory_utilization 0.95, BF16 and its FP8 row on the same pod.
    reg = load_registry()
    for model_id in ("llama-3.3-70b-instruct", "llama-3.3-70b-instruct-fp8"):
        m = reg.get(model_id)
        assert (m.hardware.gpu, m.hardware.gpus_per_replica, m.parallelism.tp) == ("H100", 2, 2)
        assert m.hardware.instance_types.model_dump() == {
            "aws": None,
            "gcp": None,
            "runpod": "h100-sxm-x2",
        }
        assert m.clouds == ["runpod"]
        assert m.engine.args == {"gpu_memory_utilization": 0.95}
        assert (m.engine.name, m.engine.version) == ("vllm", "0.30.0")
        assert m.max_context == 32768
