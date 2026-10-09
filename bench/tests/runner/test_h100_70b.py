"""The 2x H100 SXM run for Llama 3.3 70B (approved 2026-10-08): BF16 reference and
FP8 row on one pod at TP=2, the FP8 cell gated against the BF16 cell in-run; its smoke;
variants that serve another registry entry (`Variant.model`); and the phase0-strict
subset the FP8 runs use."""

import asyncio

import pytest

from loom_bench.budget import caps_for, load_budget
from loom_bench.engines import render_launch
from loom_bench.experiment import (
    EXPERIMENTS_DIR,
    ExpansionError,
    Experiment,
    apply_variant,
    expand,
    load_experiment,
)
from loom_bench.money import parse_usd
from loom_bench.plan import build_plan
from loom_bench.prices import load_prices
from loom_bench.providers.mock import MockProvider
from loom_bench.quality.suite import load_suite
from loom_bench.registry import load_registry
from loom_bench.runner import run_experiment

from .conftest import (
    LLAMA_FP8_RUNPOD,
    LLAMA_H100_RUNPOD,
    RUNPOD_SMOKE_FP8,
    RUNPOD_SMOKE_H100,
    mock_doc,
    write_yaml,
)
from .test_plan import _eval_paths, _load_shape

REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()
BF16 = REGISTRY.get("llama-3.3-70b-instruct")
FP8 = REGISTRY.get("llama-3.3-70b-instruct-fp8")
BF16_L40S_HASH = "eff0371d25425c61e0881abfee51e36348b9d4066895604cda30481e84c22d36"
SXM = "NVIDIA H100 80GB HBM3"


def _plan(exp):
    return build_plan(
        exp, expand(exp, REGISTRY), prices=PRICES, caps=caps_for(exp.budget.max_spend, BUDGET, 0)
    )


def _swap(args: list[str]) -> list[str]:
    """BF16 argv with the checkpoint, its revision and the served name swapped to FP8's."""
    swap = {BF16.hf.repo: FP8.hf.repo, BF16.hf.revision: FP8.hf.revision, BF16.id: FP8.id}
    return [swap.get(a, a) for a in args]


@pytest.mark.parametrize("path", [LLAMA_H100_RUNPOD, RUNPOD_SMOKE_H100], ids=lambda p: p.stem)
def test_the_h100_specs_carry_the_approval_note(path):
    text = path.read_text()
    assert text.startswith("# APPROVED 2026-10-08")
    header = text.split("\nname:")[0]
    # The caps and the order: the $9 smoke runs first, then the $45 run.
    assert "$45 cap" in header and "$9 cap" in header and "first" in header
    assert "DRAFT" not in text and "not approved" not in text.lower()
    assert "DRAFT" not in load_experiment(path).description


def test_the_l40s_fp8_spec_is_on_hold():
    assert LLAMA_FP8_RUNPOD.read_text().startswith("# ON HOLD (2026-10-08): superseded by")


def test_the_h100_price_is_recorded_with_its_source_and_date():
    it = PRICES.instance("runpod", "secure", "h100-sxm-x2")
    assert (it.gpu, it.gpu_count, it.gpu_memory_gb) == ("H100", 2, 80)
    assert it.on_demand_per_hour == parse_usd("$7.98")
    assert it.sources and str(it.last_checked) == "2026-10-08"


def test_the_h100_run_serves_bf16_and_its_fp8_row_on_one_pod():
    exp = load_experiment(LLAMA_H100_RUNPOD)
    base, cand = expand(exp, REGISTRY)
    assert (base.variant, cand.variant) == ("vllm-tp2", "vllm-tp2-fp8")
    # Each cell is its own registry row: results land on "Llama 3.3 70B FP8" by repo.
    assert (base.spec.id, cand.spec.id) == (BF16.id, FP8.id)
    assert cand.spec.display_name == "Llama 3.3 70B FP8"
    assert (cand.spec.hf.quant_method, cand.spec.quantization) == ("compressed-tensors", "fp8")
    assert base.host_key == cand.host_key  # one pod: a warm restart onto FP8
    for c in (base, cand):
        assert c.hardware["gpu_type_id"] == SXM and c.hardware["instance_type"] == "h100-sxm-x2"
        assert (c.hardware["gpu"], c.hardware["gpus"], c.launch.gpus) == ("H100", 2, 2)
        assert c.launch.env == {"NCCL_P2P_LEVEL": "NVL"}  # P2P over NVLink only
        args = c.launch.args
        assert args[args.index("--tensor-parallel-size") + 1] == "2"
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"
        assert "--quantization" not in args
    # The gate measures precision alone: the argv differs only in the checkpoint.
    assert cand.launch.args == _swap(base.launch.args)
    assert cand.launch.image == base.launch.image
    hashes = {base.config_hash, cand.config_hash, BF16_L40S_HASH}
    (l40s_fp8,) = expand(load_experiment(LLAMA_FP8_RUNPOD), REGISTRY)
    assert len(hashes | {l40s_fp8.config_hash}) == 4


def test_the_h100_run_gates_fp8_against_its_own_bf16_cell():
    exp, l40s = load_experiment(LLAMA_H100_RUNPOD), load_experiment(LLAMA_FP8_RUNPOD)
    q = exp.quality
    assert q is not None and q.baseline is None and q.baseline_variant == "vllm-tp2"
    assert (q.suite, q.subset) == (l40s.quality.suite, "phase0-strict")
    assert "tool_calling_strict" in {t.name for t in q.load().select(q.subset)}
    # Same profiles, durations, repetitions and SLO as the approved L40S FP8 run.
    assert exp.slo == l40s.slo and exp.repetitions == l40s.repetitions
    assert _load_shape(exp) == _load_shape(l40s)
    for a, b in zip(exp.workloads, l40s.workloads, strict=True):
        assert a.profile == b.profile and a.overrides == b.overrides
        assert (a.load.duration_s, a.load.warmup_s) == (b.load.duration_s, b.load.warmup_s)


def test_the_h100_quote_fits_its_cap_to_ttl():
    exp = load_experiment(LLAMA_H100_RUNPOD)
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$45")
    assert plan.total_micros < plan.ttl_worst_micros <= plan.caps.effective
    (host,) = plan.hosts
    kinds = [s.kind for s in host.steps]
    assert kinds.count("cold_start") == 1 and kinds.count("warm_start") == 1
    assert kinds.count("eval") == 2
    # The disk holds both checkpoints.
    assert exp.provider.container_disk_gb * 1e9 > BF16.hf.size_bytes + FP8.hf.size_bytes


def test_the_h100_smoke_runs_the_h100_paths():
    smoke, real = load_experiment(RUNPOD_SMOKE_H100), load_experiment(LLAMA_H100_RUNPOD)
    assert smoke.smoke and not real.smoke
    for field in ("instance_type", "gpu_type_id", "engine_env", "allowed_cuda_versions"):
        assert getattr(smoke.provider, field) == getattr(real.provider, field), field
    assert _load_shape(smoke) == _load_shape(real)
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    assert smoke.quality.subset == real.quality.subset
    assert _eval_paths(real) <= _eval_paths(smoke)
    assert smoke.quality.baseline_variant is not None and smoke.quality.baseline is None
    cells = expand(smoke, REGISTRY)
    real_cells = expand(real, REGISTRY)
    assert len({c.host_key for c in cells}) == 1
    assert {c.launch.image for c in cells} == {c.launch.image for c in real_cells}
    for c in cells:
        assert (c.hardware["gpu"], c.launch.gpus, c.launch.env) == (
            "H100",
            2,
            real_cells[0].launch.env,
        )
        assert c.launch.args[c.launch.args.index("--gpu-memory-utilization") + 1] == "0.95"
    fp8 = cells[1]
    assert (fp8.spec.hf.quant_method, fp8.spec.quantization) == ("compressed-tensors", "fp8")
    # The same FP8 smoke checkpoint as the L40S smoke.
    l40s_fp8 = expand(load_experiment(RUNPOD_SMOKE_FP8), REGISTRY)[1]
    assert fp8.spec.hf == l40s_fp8.spec.hf
    plan = _plan(smoke)
    assert plan.ok, plan.refusals
    assert plan.ttl_worst_micros <= plan.caps.effective == parse_usd("$9")


def test_the_fp8_smoke_and_runs_include_strict_tool_calling():
    for path in (
        RUNPOD_SMOKE_FP8,
        LLAMA_FP8_RUNPOD,
        EXPERIMENTS_DIR / "llama-3.3-70b-fp8-tp4.yaml",
    ):
        q = load_experiment(path).quality
        assert q is not None and q.subset == "phase0-strict", path.stem


@pytest.mark.parametrize("suite", ["qwen3-8b", "llama-3.3-70b-instruct"])
def test_phase0_strict_is_phase0_plus_the_strict_task(suite):
    s = load_suite(suite)
    phase0 = [t.name for t in s.select("phase0")]
    strict = [t.name for t in s.select("phase0-strict")]
    assert set(strict) == {*phase0, "tool_calling_strict"} and len(strict) == len(phase0) + 1


# --- Variant.model -----------------------------------------------------------------


def test_a_variant_patches_the_registry_entry_it_names():
    exp = Experiment.model_validate(
        mock_doc(
            model="llama-3.3-70b-instruct",
            variants=[{"name": "bf16"}, {"name": "fp8", "model": "llama-3.3-70b-instruct-fp8"}],
        )
    )
    base, cand = expand(exp, REGISTRY)
    assert (base.spec.id, cand.spec.id) == (BF16.id, FP8.id)
    assert cand.spec.model_dump() == FP8.model_dump()
    # `model` picks the entry; it is not a field patched into it.
    assert apply_variant(FP8, exp.variants[1]) == FP8.model_dump(mode="json")
    assert render_launch(cand.spec).served_model == FP8.id


def test_an_unknown_variant_model_is_refused():
    exp = Experiment.model_validate(
        mock_doc(variants=[{"name": "a"}, {"name": "b", "model": "no-such-model"}])
    )
    with pytest.raises(ExpansionError, match="no-such-model"):
        expand(exp, REGISTRY)


def test_the_suite_must_cover_every_variants_model():
    doc = mock_doc(
        model="llama-3.3-70b-instruct",
        variants=[{"name": "a"}, {"name": "b", "model": "qwen3-8b"}],
        quality={"suite": "llama-3.3-70b-instruct", "baseline_variant": "a"},
    )
    with pytest.raises(ExpansionError, match=r"is pinned for .* not qwen3-8b"):
        expand(Experiment.model_validate(doc), REGISTRY)


SUITE = {
    "suite": "mixed-rows",
    "model": "llama-3.3-70b-instruct",
    "also_models": ["llama-3.3-70b-instruct-fp8"],
    "seed": 1234,
    "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 1000},
    "tasks": [
        {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
        {"name": "json_schema", "kind": "json_schema"},
    ],
    "divergence": {
        "prompts": 12,
        "top_k": 5,
        "max_new_tokens": 16,
        "noise_multiple": 2.0,
        "ceiling_top1": 0.6,
    },
}


@pytest.mark.timeout(240)
def test_an_in_run_bf16_baseline_gates_the_fp8_row(ctx, tmp_path):
    # The H100 draft's gate wiring on the mock: the BF16 cell captures the reference and
    # the FP8 row's cell (another registry entry) is scored on it and gated against it.
    suite = write_yaml(tmp_path / "suite.yaml", SUITE)
    jitter = {"logprob_jitter": 0.25}
    exp = Experiment.model_validate(
        mock_doc(
            name="mixed-rows",
            model="llama-3.3-70b-instruct",
            variants=[
                {"name": "bf16", "mock": jitter},
                {
                    "name": "fp8",
                    "model": "llama-3.3-70b-instruct-fp8",
                    "mock": {**jitter, "logprob_noise": 0.3},
                },
            ],
            workloads=[],
            quality={"suite": str(suite), "baseline_variant": "bf16"},
        )
    )
    ctx.provider = MockProvider(hourly_micros=1_000_000)
    outcome = asyncio.run(run_experiment(exp, ctx))
    assert outcome.status.value == "completed", outcome.reason
    (gate,) = outcome.gates
    assert (gate.cell, gate.baseline) == ("fp8", "bf16")
    assert gate.decision == "pass" and not gate.blocked
    kinds = [e["kind"] for e in outcome.events]
    assert "reference_captured" in kinds and "gate" in kinds


# --- two checkpoints on one pod: downloads and disk ----------------------------------


def test_the_fp8_warm_start_is_planned_with_its_download():
    # Each pod downloads a checkpoint on the first start that serves it (the engine runs
    # offline): the FP8 cell's warm restart downloads ~73 GB, and the plan prices it.
    from loom_bench.plan import RUNPOD_TIMING as t

    exp = load_experiment(LLAMA_H100_RUNPOD)
    (host,) = _plan(exp).hosts
    (warm,) = [s for s in host.steps if s.kind == "warm_start"]
    assert warm.label == "vllm-tp2-fp8"
    size = FP8.hf.size_bytes
    assert warm.seconds == pytest.approx(
        size / t.download_bytes_per_s + size / t.load_bytes_per_s + t.engine_init_s
    )


@pytest.mark.parametrize("path", [LLAMA_H100_RUNPOD, RUNPOD_SMOKE_H100, RUNPOD_SMOKE_FP8])
def test_a_pod_disk_must_hold_every_checkpoint_its_cells_serve(path):
    import yaml

    from loom_bench.plan import RUNPOD_DISK_HEADROOM_GB

    exp = load_experiment(path)
    cells = expand(exp, REGISTRY)
    need_gb = sum(dict((c.spec.hf.repo, c.spec.hf.size_bytes) for c in cells).values()) / 1e9
    need_gb += RUNPOD_DISK_HEADROOM_GB
    assert len({c.spec.hf.repo for c in cells}) == 2
    assert exp.provider.container_disk_gb >= need_gb
    assert not [r for r in _plan(exp).refusals if "container_disk_gb" in r]
    # One checkpoint's worth of disk is not enough: both stay on the pod.
    largest = max(c.spec.hf.size_bytes for c in cells) / 1e9 + RUNPOD_DISK_HEADROOM_GB
    doc = yaml.safe_load(path.read_text())
    doc["provider"]["container_disk_gb"] = int(largest) + 1
    small = Experiment.model_validate(doc)
    (refusal,) = [r for r in _plan(small).refusals if "container_disk_gb" in r]
    assert "2 checkpoints" in refusal


def test_a_checkpoint_already_on_the_pod_is_not_planned_again():
    from loom_bench.plan import RUNPOD_TIMING as t
    from loom_bench.plan import Estimator

    exp = load_experiment(RUNPOD_SMOKE_H100)
    base, cand = expand(exp, REGISTRY)
    est = Estimator(exp)
    load_only = base.spec.hf.size_bytes / t.load_bytes_per_s + t.engine_init_s
    # Back onto BF16 after FP8: both are on the pod, nothing is downloaded.
    assert est.warm_start_s([base, cand], base) == pytest.approx(load_only)
    assert est.warm_start_s([base], cand) > est.warm_start_s([base, cand], cand)
