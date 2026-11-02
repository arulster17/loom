"""The TensorRT-LLM vs vLLM RunPod run and its smoke: plans, caps, and smoke coverage."""

import pytest

from loom_bench.budget import caps_for, load_budget
from loom_bench.experiment import Experiment, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.plan import ENGINE_START_EXTRA_S, RUNPOD_TIMING, build_plan
from loom_bench.prices import load_prices
from loom_bench.registry import load_registry

from .conftest import EXPERIMENTS_DIR, QWEN_SWEEP_RUNPOD

REAL = EXPERIMENTS_DIR / "qwen3-8b-trtllm-vs-vllm-runpod.yaml"
SMOKE = EXPERIMENTS_DIR / "runpod-smoke-trtllm.yaml"
REGISTRY = load_registry()
# The config sweep's bf16 cell (7a8237d0): vLLM v0.30.0, Qwen3-8B BF16, Secure 1x L40S.
SWEEP_BF16_HASH = "7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a"


def _plan(exp: Experiment):
    return build_plan(
        exp,
        expand(exp, REGISTRY),
        prices=load_prices(),
        caps=caps_for(exp.budget.max_spend, load_budget(), 0),
    )


@pytest.mark.parametrize(("path", "cap"), [(REAL, "$11"), (SMOKE, "$2.75")], ids=["real", "smoke"])
def test_plans_fit_their_caps_even_to_ttl(path, cap):
    plan = _plan(load_experiment(path))
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd(cap)
    assert plan.ttl_worst_micros <= plan.caps.effective
    assert plan.total_micros < 0.8 * plan.caps.effective
    assert len(plan.hosts) == 2  # one pod per engine image
    for host in plan.hosts:
        assert host.provider == "runpod" and host.steps[0].kind == "cold_start"
        assert host.seconds < 0.8 * host.ttl_s


def test_the_real_run_stays_under_ten_dollars_planned():
    assert _plan(load_experiment(REAL)).total_micros < parse_usd("$10")


def test_the_trtllm_pod_is_planned_with_its_slower_start():
    plan = _plan(load_experiment(REAL))
    vllm, trt = (h.steps[0].seconds for h in plan.hosts)
    pull, init = ENGINE_START_EXTRA_S["trtllm"]
    assert trt - vllm == pytest.approx(pull + init)
    assert RUNPOD_TIMING.image_pull_s + pull >= 400  # 20.9 GB image


def test_the_vllm_reference_is_the_config_sweeps_bf16_config():
    # Same config hash: this run's vLLM row is a same-config repeat of 7a8237d0's bf16,
    # and the windows and SLO match the sweep's so the rows compare point for point.
    real, sweep = load_experiment(REAL), load_experiment(QWEN_SWEEP_RUNPOD)
    cells = {c.variant: c for c in expand(real, REGISTRY)}
    assert cells["vllm-bf16"].config_hash == SWEEP_BF16_HASH
    assert cells["trtllm-bf16"].launch.engine == "trtllm"
    assert cells["vllm-bf16"].host_key != cells["trtllm-bf16"].host_key
    assert real.slo == sweep.slo and real.repetitions == sweep.repetitions
    shape = lambda e: {  # noqa: E731
        w.profile: (w.load.duration_s, w.load.warmup_s, w.load.drain_timeout_s) for w in e.workloads
    }
    assert shape(real) == shape(sweep)
    assert real.quality is not None and real.quality.baseline_variant == "vllm-bf16"
    # phase0, not phase0-strict: TRT-LLM 1.2.1 rejects a tool's `strict` field.
    assert real.quality.subset == "phase0" and real.quality.replicates == 3


def _eval_paths(exp: Experiment) -> set[tuple[str, str]]:
    assert exp.quality is not None
    out = set()
    for t in exp.quality.load().select(exp.quality.subset):
        harness = t.params.get("tasks") if t.kind == "lm_eval" else None
        out |= {(t.kind, name) for name in harness or [t.name]}
    return out


def _load_shape(exp: Experiment) -> dict:
    return {
        w.profile: (
            w.load.mode.value,
            w.load.search.scale if w.load.search else None,
            bool(w.load.search and w.load.search.descend),
        )
        for w in exp.workloads
    }


def test_the_smoke_covers_every_code_path_of_the_real_run():
    smoke, real = load_experiment(SMOKE), load_experiment(REAL)
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model and smoke.provider == real.provider
    assert smoke.variants == real.variants  # both engines, same images by digest
    assert _load_shape(smoke) == _load_shape(real)
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    assert smoke.quality is not None and real.quality is not None
    assert smoke.quality.model_copy(update={"limit": None}) == real.quality
    assert _eval_paths(smoke) == _eval_paths(real)
    assert smoke.quality.limit is not None
    # Same configs: the smoke starts every engine exactly as the run will.
    assert [c.config_hash for c in expand(smoke, REGISTRY)] == [
        c.config_hash for c in expand(real, REGISTRY)
    ]
    # The divergence path runs on the smoke's hard prompts, as on the full set.
    div = smoke.quality.load().divergence
    assert div is not None and div.hard_prompts
