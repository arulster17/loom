"""Llama 3.3 70B FP8: its registry row, the twin experiment specs that gate it against
the stored BF16 baseline (cf4d1614), and the smoke that runs its FP8 path first."""

import pytest

from loom_bench.budget import caps_for, load_budget
from loom_bench.engines import draft_weights, quantization_flag, render_launch
from loom_bench.experiment import ExpansionError, Experiment, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.plan import build_plan
from loom_bench.prices import load_prices
from loom_bench.quality.suite import Suite, load_suite
from loom_bench.registry import load_registry
from loom_bench.runner import plan_experiment

from .conftest import (
    LLAMA_FP8,
    LLAMA_FP8_RUNPOD,
    LLAMA_RUNPOD,
    RUNPOD_SMOKE_FP8,
    mock_doc,
)
from .test_plan import _eval_paths, _load_shape

REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()
BF16 = REGISTRY.get("llama-3.3-70b-instruct")
FP8 = REGISTRY.get("llama-3.3-70b-instruct-fp8")
QWEN = REGISTRY.get("qwen3-8b")
QWEN_FP8 = REGISTRY.get("qwen3-8b-fp8")
CAP = parse_usd("$15")  # the approved hard cap for the FP8 run (docs/PLAN.md)
# cf4d1614: llama-3.3-70b-tp4-runpod, BF16 at defaults (variant vllm-tp4).
BF16_RUN = "cf4d1614-2453-4ad4-b345-afd6b769a52d"
BF16_HASH = "eff0371d25425c61e0881abfee51e36348b9d4066895604cda30481e84c22d36"


def _plan(exp):
    return build_plan(
        exp, expand(exp, REGISTRY), prices=PRICES, caps=caps_for(exp.budget.max_spend, BUDGET, 0)
    )


def test_the_fp8_row_carries_its_precision_and_a_pinned_checkpoint():
    assert FP8.display_name == "Llama 3.3 70B FP8"
    hf = FP8.hf
    assert hf.repo == "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    assert hf.revision == "f50dbad2c84590ca17dc51e207c34321b65ff14b"
    assert (hf.license, hf.gated, hf.size_bytes) == ("llama3.3", False, 72_669_954_704)
    assert hf.quant_method == "compressed-tensors" and FP8.quantization == "fp8"
    assert not hf.trust_remote_code
    assert FP8.status == "preview" and FP8.pricing is None
    # BF16 stays the reference row, unchanged.
    assert (BF16.display_name, BF16.quantization, BF16.hf.quant_method) == (
        "Llama 3.3 70B Instruct",
        "none",
        None,
    )


def test_the_fp8_row_is_the_bf16_row_with_another_checkpoint():
    differs = {"id", "display_name", "hf", "quantization"}
    assert FP8.model_dump(exclude=differs) == BF16.model_dump(exclude=differs)
    assert FP8.hf.model_dump(
        exclude={"repo", "revision", "gated", "size_bytes", "quant_method"}
    ) == (BF16.hf.model_dump(exclude={"repo", "revision", "gated", "size_bytes", "quant_method"}))


def test_the_fp8_launch_passes_no_quantization_flag():
    # compressed-tensors names its method in config.json; a --quantization flag would make
    # vLLM refuse to start (registry pair table).
    assert quantization_flag(FP8) == []
    fp8, bf16 = render_launch(FP8).args, render_launch(BF16).args
    assert "--quantization" not in fp8
    swap = {
        BF16.hf.repo: FP8.hf.repo,
        BF16.hf.revision: FP8.hf.revision,
        BF16.id: FP8.id,
    }
    assert fp8 == [swap.get(a, a) for a in bf16]
    assert fp8[:5] == [
        FP8.hf.repo,
        "--revision",
        FP8.hf.revision,
        "--tokenizer-revision",
        FP8.hf.revision,
    ]
    assert fp8[fp8.index("--tensor-parallel-size") + 1] == "4"
    assert fp8[fp8.index("--tool-call-parser") + 1] == "llama3_json"
    assert draft_weights("vllm", fp8) == []


def test_the_fp8_cell_differs_from_its_stored_bf16_baseline_only_in_the_checkpoint():
    exp = load_experiment(LLAMA_FP8_RUNPOD)
    (cell,) = expand(exp, REGISTRY)
    # cf4d1614's cell, rebuilt: today's BF16 RunPod spec at defaults (no EAGLE3).
    bf16_exp = load_experiment(LLAMA_RUNPOD)
    bf16_exp = bf16_exp.model_copy(
        update={
            "variants": [
                bf16_exp.variants[0].model_copy(update={"name": "vllm-tp4", "engine": None})
            ]
        }
    )
    (base,) = expand(bf16_exp, REGISTRY)
    assert base.config_hash == BF16_HASH  # what the FP8 spec pins as its baseline
    assert exp.quality is not None and exp.quality.baseline is not None
    assert str(exp.quality.baseline.experiment) == BF16_RUN
    assert exp.quality.baseline.config_hash == BF16_HASH
    assert cell.config_hash != base.config_hash
    assert cell.config["hardware"] == base.config["hardware"] | {"disk_gb": 150}
    assert cell.launch.env == base.launch.env == {"NCCL_P2P_DISABLE": "1"}
    assert cell.launch.image == base.launch.image
    assert cell.launch.args == [
        {BF16.hf.repo: FP8.hf.repo, BF16.hf.revision: FP8.hf.revision, BF16.id: FP8.id}.get(a, a)
        for a in base.launch.args
    ]
    # The AWS twin's cell is another config (another host), also not BF16's.
    (aws,) = expand(load_experiment(LLAMA_FP8), REGISTRY)
    assert len({aws.config_hash, cell.config_hash, BF16_HASH}) == 3


def test_the_fp8_twins_match_apart_from_provider_and_budget():
    aws, runpod = load_experiment(LLAMA_FP8), load_experiment(LLAMA_FP8_RUNPOD)
    for field in ("model", "variants", "workloads", "repetitions", "slo", "quality"):
        assert getattr(aws, field) == getattr(runpod, field), field
    # Same load shapes, SLO and repetitions as the BF16 run it is compared with.
    bf16 = load_experiment(LLAMA_RUNPOD)
    for field in ("workloads", "repetitions", "slo"):
        assert getattr(runpod, field) == getattr(bf16, field), field
    assert runpod.provider.engine_env == bf16.provider.engine_env
    assert runpod.quality.suite == bf16.quality.suite and runpod.quality.subset == "phase0-strict"
    # FP8 at defaults: no speculation in this run.
    assert [v.name for v in runpod.variants] == ["vllm-tp4-fp8"]
    assert all(v.engine is None and v.hf is None for v in runpod.variants)


def test_the_fp8_runpod_run_fits_the_fifteen_dollar_cap_to_ttl():
    exp = load_experiment(LLAMA_FP8_RUNPOD)
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert exp.budget.max_spend == CAP and plan.caps.effective == CAP
    assert plan.ttl_worst_micros <= CAP  # even if the runner dies and the pod lives to TTL
    assert plan.total_micros < CAP
    (host,) = plan.hosts
    assert host.provider == "runpod" and host.steps[0].kind == "cold_start"
    assert host.seconds < 0.8 * host.ttl_s
    evals = [s for s in host.steps if s.kind == "eval"]
    assert len(evals) == 1 and evals[0].label.endswith("[phase0-strict]")


def test_without_its_stored_baseline_the_fp8_run_is_refused(ctx):
    # The test DB has no cf4d1614: the plan refuses before anything is provisioned.
    _, plan = plan_experiment(load_experiment(LLAMA_FP8_RUNPOD), ctx)
    assert not plan.ok
    assert any(r.startswith("quality.baseline:") and BF16_RUN in r for r in plan.refusals)


def test_the_llama_suite_covers_the_fp8_row_and_nothing_else():
    suite = load_suite("llama-3.3-70b-instruct")
    assert suite.covers("llama-3.3-70b-instruct") and suite.covers("llama-3.3-70b-instruct-fp8")
    assert not suite.covers("qwen3-8b")
    doc = mock_doc(
        model="qwen3-8b", quality={"suite": "llama-3.3-70b-instruct", "baseline_variant": "a"}
    )
    with pytest.raises(ExpansionError, match="is pinned for"):
        expand(Experiment.model_validate(doc), REGISTRY)
    with pytest.raises(ValueError, match="also_models"):
        Suite.model_validate(suite.model_dump() | {"also_models": [suite.model]})


def test_the_fp8_smoke_runs_the_fp8_path_of_the_70b_run():
    smoke, real = load_experiment(RUNPOD_SMOKE_FP8), load_experiment(LLAMA_FP8_RUNPOD)
    assert smoke.smoke and not real.smoke
    assert smoke.provider.kind == real.provider.kind == "runpod"
    # Same load shapes, repetitions, SLO and eval paths as the run it smokes.
    assert _load_shape(smoke) == _load_shape(real)
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    assert smoke.quality is not None and real.quality is not None
    assert smoke.quality.subset == real.quality.subset
    assert smoke.quality.allow_code_exec == real.quality.allow_code_exec
    assert _eval_paths(real) <= _eval_paths(smoke)
    assert smoke.quality.limit is not None and real.quality.limit is None
    # One engine image for both: the FP8 run's, by digest.
    cells = {c.variant: c for c in expand(smoke, REGISTRY)}
    (real_cell,) = expand(real, REGISTRY)
    assert {c.launch.image for c in cells.values()} == {real_cell.launch.image}
    assert len({c.host_key for c in cells.values()}) == 1  # one pod, warm restart
    # A BF16 baseline that captures the reference, and a pre-quantized compressed-tensors
    # FP8 candidate scored on it and gated, loaded with no --quantization flag.
    base, cand = cells[smoke.quality.baseline_variant], cells["vllm-fp8"]
    # Selected as the 70B run selects its FP8 row: a variant naming the FP8 registry
    # entry (Variant.model), served under that entry's id.
    (fp8_variant,) = [v for v in smoke.variants if v.name == "vllm-fp8"]
    assert fp8_variant.model == "qwen3-8b-fp8" and fp8_variant.hf is None
    assert (base.spec.id, cand.spec.id) == ("qwen3-8b", "qwen3-8b-fp8")
    assert cand.spec == QWEN_FP8 and cand.launch.served_model == "qwen3-8b-fp8"
    assert (base.spec.hf.quant_method, base.spec.quantization) == (None, "none")
    assert (cand.spec.hf.quant_method, cand.spec.quantization) == (
        real_cell.spec.hf.quant_method,
        real_cell.spec.quantization,
    )
    assert cand.spec.hf.repo.startswith("RedHatAI/") and cand.spec.hf.repo.endswith("FP8-dynamic")
    assert real_cell.spec.hf.repo.startswith("RedHatAI/") and real_cell.spec.hf.repo.endswith(
        "FP8-dynamic"
    )
    assert (
        "--quantization" not in cand.launch.args and "--quantization" not in real_cell.launch.args
    )
    assert cand.spec.hf.gated is False and real_cell.spec.hf.gated is False
    # The real run adds no engine path beyond the smoke's: no engine args, no draft.
    assert real_cell.spec.engine.args == cand.spec.engine.args == {}


def test_the_fp8_smoke_fits_its_cap_to_ttl():
    plan = _plan(load_experiment(RUNPOD_SMOKE_FP8))
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$1.25")
    assert plan.ttl_worst_micros <= plan.caps.effective
    (host,) = plan.hosts
    assert [s.kind for s in host.steps].count("eval") == 2 == plan.n_cells
    assert host.seconds < 0.8 * host.ttl_s


# --- Qwen3 8B FP8: the smokes' FP8 row ----------------------------------------------


def test_the_qwen_fp8_row_carries_its_precision_and_a_pinned_checkpoint():
    assert QWEN_FP8.display_name == "Qwen3 8B FP8"
    hf = QWEN_FP8.hf
    assert hf.repo == "RedHatAI/Qwen3-8B-FP8-dynamic"
    assert hf.revision == "05233ce1e0565b5fdc9cfa000ab840152ed30c70"
    assert (hf.license, hf.gated, hf.size_bytes) == ("apache-2.0", False, 9_438_581_960)
    assert hf.quant_method == "compressed-tensors" and QWEN_FP8.quantization == "fp8"
    assert not hf.trust_remote_code
    assert QWEN_FP8.status == "preview" and QWEN_FP8.pricing is None
    assert (QWEN.display_name, QWEN.quantization, QWEN.hf.quant_method) == (
        "Qwen3 8B",
        "none",
        None,
    )


def test_the_qwen_fp8_row_is_the_bf16_row_with_another_checkpoint():
    differs = {"id", "display_name", "hf", "quantization"}
    assert QWEN_FP8.model_dump(exclude=differs) == QWEN.model_dump(exclude=differs)
    keep = {"repo", "revision", "size_bytes", "quant_method"}
    assert QWEN_FP8.hf.model_dump(exclude=keep) == QWEN.hf.model_dump(exclude=keep)
    assert quantization_flag(QWEN_FP8) == []
    swap = {QWEN.hf.repo: QWEN_FP8.hf.repo, QWEN.hf.revision: QWEN_FP8.hf.revision}
    swap[QWEN.id] = QWEN_FP8.id
    bf16 = render_launch(QWEN).args
    assert render_launch(QWEN_FP8).args == [swap.get(a, a) for a in bf16]


def test_the_qwen_suite_covers_its_fp8_row_and_nothing_else():
    suite = load_suite("qwen3-8b")
    assert suite.also_models == ["qwen3-8b-fp8"]
    assert suite.covers("qwen3-8b") and suite.covers("qwen3-8b-fp8")
    assert not suite.covers("llama-3.3-70b-instruct-fp8")
