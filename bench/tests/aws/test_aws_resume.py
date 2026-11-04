"""The three follow-ups to 46938d1f (qwen3-8b-trtllm-vs-vllm-aws, 2026-11-04), before any
of them spends anything.

46938d1f completed vllm-bf16 (load and evals) and trtllm-bf16's load sweep, lost
trtllm-bf16's quality suite (TensorRT-LLM died under the divergence check) and never ran
trtllm-fp8 (its host was read as lost). The runner has no resume, so the missing parts are
three specs, each gated against 46938d1f's stored vLLM BF16 evals:

- aws-smoke-trtllm-divergence: both TensorRT-LLM cells, one full-size pass each with the
  full 48-prompt divergence (the path that failed; the earlier smokes sent 4 prompts);
- qwen3-8b-trtllm-bf16-quality-aws: trtllm-bf16's quality suite only;
- qwen3-8b-trtllm-fp8-aws: trtllm-fp8, load and quality, exactly as in 46938d1f's spec.

Each keeps 46938d1f's config hashes, so its results pair with that run's, and each fits
its approved cap even with its host running to TTL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.quality.divergence import load_prompts

from .test_aws_g6e import ON_DEMAND_ACCRUAL, REGISTRY, _plan

X = EXPERIMENTS_DIR
ORIGINAL = X / "qwen3-8b-trtllm-vs-vllm-aws.yaml"
SMOKE = X / "aws-smoke-trtllm-divergence.yaml"
BF16_QUALITY = X / "qwen3-8b-trtllm-bf16-quality-aws.yaml"
FP8 = X / "qwen3-8b-trtllm-fp8-aws.yaml"
EXPERIMENT_46938D1F = "46938d1f-3022-4695-a0b9-29b18c06eaea"
# The cells of 46938d1f, as recorded in the results DB (bench_runs.config_hash).
HASHES = {
    "vllm-bf16": "48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375",
    "trtllm-bf16": "ae3b5705d2fa54e053eeadaf507ef46ed681a1fbcf79631dc7f65f4f5457fb8a",
    "trtllm-fp8": "55b356557da002ca376de348fdcdd39422699bc205e2f9016776877ca4509b80",
}
# Approved 2026-11-04: spec -> (cap, TTL minutes, hosts, cells).
QUOTES = {
    SMOKE: ("$2.25", 71, 1, ["trtllm-bf16", "trtllm-fp8"]),
    BF16_QUALITY: ("$3.25", 100, 1, ["trtllm-bf16"]),
    FP8: ("$9.00", 285, 1, ["trtllm-fp8"]),
}
SPECS = list(QUOTES)


def _doc(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text()))


def _variants(path: Path) -> dict[str, dict[str, Any]]:
    return {v["name"]: v for v in _doc(path)["variants"]}


def _quality(exp: Experiment) -> dict[str, Any]:
    """The quality section without what says where the baseline comes from."""
    assert exp.quality is not None
    return exp.quality.model_dump(exclude={"baseline", "baseline_variant"})


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_each_spec_keeps_46938d1fs_config_hashes(path):
    cells = expand(load_experiment(path), REGISTRY)
    assert [c.key for c in cells] == QUOTES[path][3]
    for cell in cells:
        assert cell.config_hash == HASHES[cell.key]
    # And the original still expands to them (registry rows and engine args unchanged).
    original = {c.key: c.config_hash for c in expand(load_experiment(ORIGINAL), REGISTRY)}
    assert original == HASHES


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_each_spec_is_gated_against_46938d1fs_stored_vllm_bf16(path):
    exp = load_experiment(path)
    assert exp.provider == load_experiment(ORIGINAL).provider
    assert exp.quality is not None and exp.quality.baseline is not None
    assert exp.quality.baseline_variant is None
    assert str(exp.quality.baseline.experiment) == EXPERIMENT_46938D1F
    assert exp.quality.baseline.config_hash == HASHES["vllm-bf16"]
    assert exp.quality.limit is None  # full size: a limit would not pair with the baseline
    original = load_experiment(ORIGINAL)
    assert original.quality is not None and original.quality.baseline_variant == "vllm-bf16"


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.stem)
def test_each_plan_fits_its_cap_with_its_host_to_ttl(path):
    cap, ttl, n_hosts, _ = QUOTES[path]
    exp = load_experiment(path)
    assert exp.name == path.stem
    assert exp.budget.ttl_minutes == ttl
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd(cap)
    assert plan.total_micros < plan.ttl_worst_micros <= plan.caps.effective
    assert len(plan.hosts) == n_hosts
    for host in plan.hosts:
        assert host.market == "on_demand" and host.hourly_micros == ON_DEMAND_ACCRUAL
        assert host.instance_type == "g6e.xlarge"
        assert [s.kind for s in host.steps].count("cold_start") == 1
        assert host.seconds < 0.85 * host.ttl_s


def test_the_three_fit_the_approved_total():
    assert sum(parse_usd(cap) for cap, _, _, _ in QUOTES.values()) == parse_usd("$14.50")


def test_the_fp8_run_is_46938d1fs_trtllm_fp8_cell_exactly():
    fp8, original = load_experiment(FP8), load_experiment(ORIGINAL)
    assert _variants(FP8) == {"trtllm-fp8": _variants(ORIGINAL)["trtllm-fp8"]}
    assert fp8.workloads == original.workloads
    assert (fp8.repetitions, fp8.slo, fp8.seed) == (
        original.repetitions,
        original.slo,
        original.seed,
    )
    assert _quality(fp8) == _quality(original)
    assert not fp8.smoke
    (cell,) = expand(fp8, REGISTRY)
    by_key = {c.key: c for c in expand(original, REGISTRY)}
    assert cell.host_key == by_key["trtllm-fp8"].host_key


def test_the_bf16_quality_run_is_46938d1fs_trtllm_bf16_eval_only():
    bf16, original = load_experiment(BF16_QUALITY), load_experiment(ORIGINAL)
    assert _variants(BF16_QUALITY) == {"trtllm-bf16": _variants(ORIGINAL)["trtllm-bf16"]}
    assert bf16.workloads == [] and bf16.slo is None
    assert _quality(bf16) == _quality(original)
    assert not bf16.smoke
    steps = [s.kind for h in _plan(bf16).hosts for s in h.steps]
    assert steps == ["cold_start", "eval_setup", "eval", "teardown"]


def test_the_smoke_runs_both_trtllm_evals_at_full_size_one_pass_each():
    smoke, original = load_experiment(SMOKE), load_experiment(ORIGINAL)
    assert smoke.smoke and smoke.workloads == []
    # The variants as in the original, on one host (warm start between them).
    variants = _variants(SMOKE)
    for name, v in _variants(ORIGINAL).items():
        if name.startswith("trtllm"):
            assert {k: x for k, x in variants[name].items() if k != "host_group"} == {
                k: x for k, x in v.items() if k != "host_group"
            }
    assert len({c.host_key for c in expand(smoke, REGISTRY)}) == 1
    # Every per-pass quality setting of the follow-ups; one pass instead of three.
    assert smoke.quality is not None and original.quality is not None
    assert smoke.quality.replicates == 1 and original.quality.replicates == 3
    assert _quality(smoke) == {**_quality(original), "replicates": 1}
    # The full divergence: 48 prompts, as 46938d1f's reference holds (its smokes had 4).
    suite = smoke.quality.load()
    assert suite.divergence is not None
    assert len(suite.divergence.select(load_prompts())) == 48
    steps = [s.kind for h in _plan(smoke).hosts for s in h.steps]
    assert steps == ["cold_start", "eval_setup", "eval", "warm_start", "eval", "teardown"]
