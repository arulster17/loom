import uuid

import pytest
from pydantic import ValidationError

from loom_bench.experiment import (
    ExpansionError,
    Experiment,
    LoadSpec,
    Variant,
    derive_seed,
    expand,
    load_experiment,
    sweep_points,
)
from loom_bench.records import Market
from loom_bench.registry import Registry, load_registry, read_yaml
from loom_bench.runner import _host_request, next_search_load

from .conftest import ABORT, LLAMA, QWEN, SMOKE, mock_doc, mock_experiment

REGISTRY = load_registry()


@pytest.mark.parametrize("path", [SMOKE, ABORT, QWEN, LLAMA], ids=lambda p: p.stem)
def test_shipped_experiments_validate_and_expand(path):
    exp = load_experiment(path)
    cells = expand(exp, REGISTRY)
    assert cells and len({c.key for c in cells}) == len(cells)


def test_expansion_is_deterministic_and_hashes_are_stable():
    exp = load_experiment(SMOKE)
    a, b = expand(exp, REGISTRY), expand(load_experiment(SMOKE), REGISTRY)
    assert [c.key for c in a] == ["baseline", "small-batch"]
    assert [c.config_hash for c in a] == [c.config_hash for c in b]
    assert a[0].config_hash != a[1].config_hash
    assert a[0].host_key == a[1].host_key  # one host, warm restart between them


def test_vllm_and_sglang_variants_render_their_own_launch():
    cells = expand(load_experiment(QWEN), REGISTRY)
    vllm, sglang = cells
    assert vllm.launch.engine == "vllm" and sglang.launch.engine == "sglang"
    assert sglang.launch.image.startswith("lmsysorg/sglang@sha256:")
    assert sglang.spec.engine.version == "0.5.21"
    assert vllm.host_key == sglang.host_key


def test_sweep_is_a_cartesian_product_applied_to_each_variant():
    exp = mock_experiment(
        variants=[{"name": "a"}, {"name": "b", "mock": {"step_base_ms": 9.0}}],
        sweep={"mock.max_num_seqs": [8, 16], "kv_cache_dtype": ["auto", "fp8"]},
    )
    cells = expand(exp, REGISTRY)
    assert len(cells) == 8
    assert cells[0].key == "a[kv_cache_dtype=auto,mock.max_num_seqs=8]"
    assert {c.mock.max_num_seqs for c in cells} == {8, 16}
    assert {c.spec.kv_cache_dtype for c in cells} == {"auto", "fp8"}
    assert all(c.mock.step_base_ms == 9.0 for c in cells if c.variant == "b")
    assert len({c.config_hash for c in cells}) == 8


def test_random_sample_is_seeded_and_drawn_from_the_grid():
    grid = {"mock.max_num_seqs": [4, 8, 16, 32], "mock.step_base_ms": [3.0, 6.0, 9.0]}
    exp = mock_experiment(sweep=grid, sample={"random": 4, "seed": 7})
    full = sweep_points(mock_experiment(sweep=grid))
    picked = sweep_points(exp)
    assert len(full) == 12 and len(picked) == 4
    assert all(p in full for p in picked)
    assert picked == sweep_points(mock_experiment(sweep=grid, sample={"random": 4, "seed": 7}))
    assert [c.key for c in expand(exp, REGISTRY)] == [c.key for c in expand(exp, REGISTRY)]


def test_engine_args_merge_and_null_removes():
    doc = Experiment.model_validate(
        {
            **mock_doc(),
            "provider": {"kind": "aws_ec2"},
            "variants": [
                {
                    "name": "a",
                    "engine": {"args": {"max_num_seqs": 64, "enable_chunked_prefill": True}},
                },
            ],
            "sweep": {"engine.args.max_num_seqs": [128, None]},
        }
    )
    cells = expand(doc, REGISTRY)
    assert cells[0].spec.engine.args == {"max_num_seqs": 128, "enable_chunked_prefill": True}
    assert "--max-num-seqs" in cells[0].launch.args
    assert cells[1].spec.engine.args == {"enable_chunked_prefill": True}


@pytest.mark.parametrize(
    ("variant", "match"),
    [
        ({"name": "x", "parallelism": {"tp": 2}}, "tp"),  # GPUs no longer match tp * pp
        ({"name": "x", "engine": {"name": "sglang"}}, "image"),  # engine switch needs image
        ({"name": "x", "max_context": 10**6}, "max_context"),
        ({"name": "x", "quantization": "int3"}, "quantization"),
        ({"name": "x", "hf": {"revision": "main"}}, "revision"),  # must be pinned
        ({"name": "x", "quantization": "awq"}, "cannot be served"),  # needs an AWQ checkpoint
    ],
)
def test_invalid_overrides_are_rejected(variant, match):
    exp = Experiment.model_validate({**mock_doc(), "variants": [variant]})
    with pytest.raises(ExpansionError, match=match):
        expand(exp, REGISTRY)


def test_variant_serves_a_prequantized_checkpoint():
    hf = {"repo": "Qwen/Qwen3-8B-AWQ", "revision": "1" * 40, "quant_method": "awq"}
    exp = Experiment.model_validate(
        {
            **mock_doc(),
            "provider": {"kind": "aws_ec2"},
            "variants": [{"name": "awq", "hf": hf, "quantization": "awq"}],
        }
    )
    (cell,) = expand(exp, REGISTRY)
    assert (cell.spec.hf.quant_method, cell.spec.quantization) == ("awq", "awq")
    assert cell.launch.args[0] == "Qwen/Qwen3-8B-AWQ" and "--quantization" not in cell.launch.args


def test_reserved_engine_arg_is_rejected_by_the_launch_renderer():
    exp = Experiment.model_validate(
        {
            **mock_doc(),
            "provider": {"kind": "aws_ec2"},
            "variants": [{"name": "x", "engine": {"args": {"port": 9000}}}],
        }
    )
    with pytest.raises(ExpansionError, match="port"):
        expand(exp, REGISTRY)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Experiment.model_validate({**mock_doc(), "variants": [{"name": "x", "tp": 2}]})
    with pytest.raises(ValidationError):
        Experiment.model_validate({**mock_doc(), "budgets": {}})


def test_money_parses_to_micros_and_rejects_floats():
    assert mock_experiment(budget={"max_spend": "$1.50", "ttl_minutes": 1}).budget.max_spend == (
        1_500_000
    )
    with pytest.raises(ValidationError, match="string"):
        mock_experiment(budget={"max_spend": 1.5, "ttl_minutes": 1})


def test_single_repetition_needs_explicit_opt_in_and_is_untrusted():
    with pytest.raises(ValidationError, match="allow_single_run"):
        mock_experiment(repetitions=1)
    exp = mock_experiment(repetitions=1, allow_single_run=True)
    assert not exp.trusted


@pytest.mark.parametrize(
    ("load", "match"),
    [
        ({"mode": "open_loop", "values": [1]}, "duration_s"),
        ({"mode": "open_loop", "values": [1], "duration_s": 5, "warmup_s": 5}, "warmup_s"),
        (
            {"mode": "open_loop", "values": [1], "duration_s": 5, "arrival": {"kind": "ramp"}},
            "single rate",
        ),
        (
            {
                "mode": "open_loop",
                "values": [1],
                "duration_s": 5,
                "arrival": {"kind": "poisson", "rate": 3},
            },
            "load value",
        ),
        ({"mode": "closed_loop", "values": [1.5], "num_requests": 4}, "integers"),
        ({"mode": "closed_loop", "values": [1], "num_requests": 4, "duration_s": 3}, "exactly"),
        ({"mode": "closed_loop", "num_requests": 4}, "exactly one of load.values"),
    ],
)
def test_load_spec_validation(load, match):
    entry = {"profile": "fixed-128-128", "load": load}
    with pytest.raises(ValidationError, match=match):
        mock_experiment(workloads=[entry])


def test_search_needs_an_slo():
    load = {"mode": "closed_loop", "search": {"lo": 1, "hi": 8}, "num_requests": 4}
    with pytest.raises(ValidationError, match="slo"):
        mock_experiment(workloads=[{"profile": "fixed-128-128", "load": load}], slo=None)


def test_search_scale_defaults_to_linear_and_accepts_geometric():
    base = {"mode": "open_loop", "duration_s": 10}
    linear = LoadSpec.model_validate({**base, "search": {"lo": 1, "hi": 8}})
    assert linear.search is not None and linear.search.scale == "linear"
    geo = LoadSpec.model_validate(
        {**base, "search": {"lo": 1, "hi": 8, "scale": "geometric", "step": 1.5}}
    )
    assert geo.search is not None and (geo.search.scale, geo.search.step) == ("geometric", 1.5)
    with pytest.raises(ValidationError):
        LoadSpec.model_validate({**base, "search": {"lo": 1, "hi": 8, "step": 1}})
    with pytest.raises(ValidationError):
        LoadSpec.model_validate({**base, "search": {"lo": 1, "hi": 8, "scale": "cubic"}})


def test_next_search_load_follows_the_scale_rounds_closed_loop_and_caps_points():
    geo = LoadSpec.model_validate(
        {
            "mode": "open_loop",
            "duration_s": 10,
            "search": {"lo": 1, "hi": 8, "scale": "geometric", "max_points": 3},
        }
    )
    assert next_search_load(geo, []) == 1
    assert next_search_load(geo, [(1, True)]) == 2
    assert next_search_load(geo, [(1, True), (2, True)]) == 4
    assert next_search_load(geo, [(1, True), (2, True), (4, False)]) is None  # max_points
    closed = LoadSpec.model_validate(
        {
            "mode": "closed_loop",
            "num_requests": 4,
            "search": {"lo": 2, "hi": 16, "scale": "geometric", "rel_tol": 0.01},
        }
    )
    assert next_search_load(closed, [(2, True), (4, False)]) == 3.0  # sqrt(8) rounds to 3
    assert next_search_load(closed, [(2, True), (3, True), (4, False)]) is None  # a repeat


def test_next_search_load_descends_below_a_failing_lo():
    spec = {"mode": "open_loop", "duration_s": 10}
    search = {"lo": 1, "hi": 8, "scale": "geometric"}
    plain = LoadSpec.model_validate({**spec, "search": search})
    assert plain.search is not None and plain.search.descend == 0
    assert next_search_load(plain, [(1, False)]) is None
    down = LoadSpec.model_validate({**spec, "search": {**search, "descend": 2}})
    assert next_search_load(down, [(1, False)]) == 0.5
    assert next_search_load(down, [(1, False), (0.5, False)]) == 0.25
    assert next_search_load(down, [(1, False), (0.5, False), (0.25, False)]) is None
    with pytest.raises(ValidationError):
        LoadSpec.model_validate({**spec, "search": {**search, "descend": -1}})


def test_arrival_template_gets_the_load_value_as_rate():
    exp = mock_experiment(
        workloads=[
            {
                "profile": "fixed-128-128",
                "load": {
                    "mode": "open_loop",
                    "values": [3],
                    "duration_s": 2,
                    "arrival": {"kind": "gamma", "burstiness": 0.5},
                },
            }
        ]
    )
    arrival = exp.workloads[0].load.arrival_for(3.0)
    assert arrival == {"kind": "gamma", "rate": 3.0, "burstiness": 0.5}


def test_local_provider_takes_exactly_one_cell():
    provider = {
        "kind": "local",
        "base_url": "http://127.0.0.1:1/v1",
        "engine": "mock",
        "served_model": "m",
    }
    exp = mock_experiment(provider=provider, variants=[{"name": "a"}, {"name": "b"}])
    with pytest.raises(ExpansionError, match="one running endpoint"):
        expand(exp, REGISTRY)


def test_derived_seeds_are_stable_and_distinct():
    assert derive_seed(0, "w", 1.0, 0) == derive_seed(0, "w", 1.0, 0)
    seeds = {derive_seed(0, "w", v, r) for v in (1.0, 2.0) for r in range(3)}
    assert len(seeds) == 6


def runpod_experiment(path=QWEN, **provider) -> Experiment:
    """A shipped experiment moved onto the runpod provider."""
    doc = read_yaml(path)
    return Experiment.model_validate({**doc, "provider": {"kind": "runpod", **provider}})


def test_runpod_hardware_and_host_key_per_image():
    vllm, sglang = expand(runpod_experiment(), REGISTRY)
    assert vllm.hardware == {
        "provider": "runpod",
        "cloud": "runpod",
        "region": "secure",
        "instance_type": "l40s-x1",
        "market": "on_demand",
        "gpu": "L40S",
        "gpus": 1,
        "gpu_type_id": "NVIDIA L40S",
        "disk_gb": 80,
        "allowed_cuda_versions": ["13.0"],
    }
    # One pod runs one engine image: different images never share a host.
    assert vllm.host_key == "runpod/secure/l40s-x1/80gb/vllm@8a69ffad015f"
    assert sglang.host_key == "runpod/secure/l40s-x1/80gb/sglang@b1259f3ea327"
    assert sglang.hardware == vllm.hardware


def test_runpod_cells_on_one_image_share_a_pod():
    exp = runpod_experiment(container_disk_gb=120)
    exp = exp.model_copy(
        update={"variants": exp.variants[:1], "sweep": {"max_context": [8192, 16384]}}
    )
    a, b = expand(exp, REGISTRY)
    assert a.host_key == b.host_key == "runpod/secure/l40s-x1/120gb/vllm@8a69ffad015f"
    assert a.config_hash != b.config_hash


P2P_OFF = {"engine_env": {"NCCL_P2P_DISABLE": "1"}}


def test_runpod_llama_uses_four_gpus():
    (cell,) = expand(runpod_experiment(LLAMA, **P2P_OFF), REGISTRY)
    assert cell.hardware["instance_type"] == "l40s-x4"
    assert cell.hardware["gpus"] == 4 == cell.gpus


def test_multi_gpu_runpod_pod_must_decide_nccl_p2p():
    # 874110b6: a 4x L40S pod across two sockets hung at NCCL init with P2P on.
    with pytest.raises(ExpansionError, match="4-GPU RunPod pod must set NCCL_P2P_DISABLE"):
        expand(runpod_experiment(LLAMA), REGISTRY)
    for env in ({"NCCL_P2P_DISABLE": "1"}, {"NCCL_P2P_LEVEL": "PIX"}):
        (cell,) = expand(runpod_experiment(LLAMA, engine_env=env), REGISTRY)
        assert cell.launch.env == env
    # One GPU has no peers: no decision needed, and no env is added.
    vllm, _ = expand(runpod_experiment(), REGISTRY)
    assert vllm.launch.env == {}


def test_runpod_engine_env_is_part_of_the_config_hash():
    (off,) = expand(runpod_experiment(LLAMA, **P2P_OFF), REGISTRY)
    (level,) = expand(runpod_experiment(LLAMA, engine_env={"NCCL_P2P_LEVEL": "PIX"}), REGISTRY)
    assert off.config["launch"]["env"] == {"NCCL_P2P_DISABLE": "1"}
    assert off.config_hash != level.config_hash
    assert off.host_key == level.host_key  # the same pod shape either way


def test_runpod_engine_env_rejects_secret_and_bad_names():
    for env in ({"HF_TOKEN": "x"}, {"nccl_debug": "INFO"}):
        with pytest.raises(ExpansionError, match=r"provider\.engine_env"):
            expand(runpod_experiment(engine_env=env), REGISTRY)


def test_runpod_spec_defaults_and_rejects_bad_fields():
    exp = runpod_experiment()
    assert exp.provider.kind == "runpod"
    assert exp.provider.cloud_type == "secure"
    assert exp.provider.data_center_ids is None  # no datacenter pin by default
    for bad in ({"market": "spot"}, {"cloud_type": "community"}, {"allowed_cuda_versions": []}):
        with pytest.raises(ValidationError):
            runpod_experiment(**bad)


def registry_without_runpod_instance() -> Registry:
    doc = REGISTRY.model_dump(mode="json")
    for m in doc["models"]:
        m["hardware"]["instance_types"].pop("runpod")
        m["clouds"] = ["aws"]
    return Registry.model_validate(doc)


def test_runpod_expansion_errors():
    with pytest.raises(ExpansionError, match="no runpod instance type"):
        expand(runpod_experiment(), registry_without_runpod_instance())
    a100 = [Variant.model_validate({"name": "a100", "hardware": {"gpu": "A100"}})]
    with pytest.raises(ExpansionError, match="no RunPod GPU type id for A100"):
        expand(runpod_experiment().model_copy(update={"variants": a100}), REGISTRY)
    exp = runpod_experiment(gpu_type_id="NVIDIA A100 80GB PCIe")
    (cell,) = expand(exp.model_copy(update={"variants": a100}), REGISTRY)
    assert cell.hardware["gpu_type_id"] == "NVIDIA A100 80GB PCIe"


def test_host_request_carries_the_engine_image():
    exp = runpod_experiment()
    _, sglang = expand(exp, REGISTRY)
    req = _host_request(exp, sglang, uuid.uuid4())
    assert req.image == sglang.launch.image
    assert (req.cloud, req.region, req.instance_type, req.disk_gb, req.gpus) == (
        "runpod",
        "secure",
        "l40s-x1",
        80,
        1,
    )
    assert req.market is Market.ON_DEMAND
    mock = mock_experiment()
    assert _host_request(mock, expand(mock, REGISTRY)[0], uuid.uuid4()).image is None
