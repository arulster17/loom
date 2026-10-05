import pytest
from pydantic import ValidationError

from loom_bench.experiment import (
    ExpansionError,
    Experiment,
    derive_seed,
    expand,
    load_experiment,
    sweep_points,
)
from loom_bench.registry import load_registry

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
    ],
)
def test_invalid_overrides_are_rejected(variant, match):
    exp = Experiment.model_validate({**mock_doc(), "variants": [variant]})
    with pytest.raises(ExpansionError, match=match):
        expand(exp, REGISTRY)


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
