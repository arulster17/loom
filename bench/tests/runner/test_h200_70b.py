"""Option B levers 2 and 3 for Llama 3.3 70B (proposed 2026-10-31, not approved): the FP8
row on one RunPod Secure H200 at TP=1, with and without EAGLE3, gated against 9e5f7866's
stored BF16 cell; its smoke; and the H200 price and GPU type id they need."""

import pytest

from loom_bench.budget import caps_for, load_budget
from loom_bench.engines import draft_weights
from loom_bench.experiment import RUNPOD_GPU_TYPE_IDS, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.plan import RUNPOD_DISK_HEADROOM_GB, build_plan, decode_step_ms
from loom_bench.prices import load_prices
from loom_bench.registry import load_registry

from .conftest import (
    LLAMA_H100_FP8_CAPACITY_RUNPOD,
    LLAMA_H100_QUALITY_RUNPOD,
    LLAMA_H100_RUNPOD,
    LLAMA_H200_FP8_RUNPOD,
    LLAMA_RUNPOD,
    RUNPOD_SMOKE_8B_SWEEP,
    RUNPOD_SMOKE_H200,
)
from .test_plan import _eval_paths, _load_shape

REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()
FP8 = REGISTRY.get("llama-3.3-70b-instruct-fp8")
H200_ID = "NVIDIA H200"
# 9e5f7866 (the 70B quality rerun, tool calling data version 2): its BF16 cell.
BF16_9E5F7866 = ("9e5f7866-f2f3-4672-bc05-925fe270271c", "2f7fb3f9372ba095")
# 8bc65cfa (runpod-smoke-8b-sweep): its bf16 cell, evaluated at limit 4.
BF16_SMOKE_8BC65CFA = ("8bc65cfa-126f-4f0e-a4df-a0db2471f3d7", "7a203a3ffaccb182")
LLAMA_EAGLE3 = (
    "RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3",
    "42864f78d503693ae5fd317419f099ab3b7c13e4",
)


def _drafts(cell):
    return draft_weights(cell.launch.engine, cell.launch.args)


def _plan(exp):
    return build_plan(
        exp, expand(exp, REGISTRY), prices=PRICES, caps=caps_for(exp.budget.max_spend, BUDGET, 0)
    )


@pytest.mark.parametrize("path", [LLAMA_H200_FP8_RUNPOD, RUNPOD_SMOKE_H200], ids=lambda p: p.stem)
def test_the_h200_specs_are_marked_proposed_not_approved(path):
    text = path.read_text()
    assert text.startswith("# PROPOSED 2026-10-31, NOT APPROVED")
    assert "APPROVED 20" not in text.split("\nname:")[0].replace("NOT APPROVED", "")


def test_the_h200_price_and_gpu_type_id_are_recorded():
    it = PRICES.instance("runpod", "secure", "h200-x1")
    assert (it.gpu, it.gpu_count, it.gpu_memory_gb) == ("H200", 1, 141)
    assert it.on_demand_per_hour == parse_usd("$5.29")
    assert it.sources and str(it.last_checked) == "2026-10-31"
    # RunPod's SXM part; "NVIDIA H200 NVL" is the PCIe card.
    assert RUNPOD_GPU_TYPE_IDS["H200"] == H200_ID


def test_the_h200_run_serves_the_fp8_row_alone_on_one_gpu():
    exp = load_experiment(LLAMA_H200_FP8_RUNPOD)
    plain, eagle = expand(exp, REGISTRY)
    assert (plain.variant, eagle.variant) == ("vllm-h200-fp8", "vllm-h200-fp8-eagle3")
    assert plain.host_key == eagle.host_key  # one pod: a warm restart onto EAGLE3
    for c in (plain, eagle):
        assert c.spec.id == FP8.id and c.spec.hf == FP8.hf  # the shipped FP8 checkpoint
        assert c.hardware["instance_type"] == "h200-x1" and c.hardware["gpu_type_id"] == H200_ID
        assert (c.hardware["gpu"], c.hardware["gpus"], c.launch.gpus) == ("H200", 1, 1)
        args = c.launch.args
        assert args[args.index("--tensor-parallel-size") + 1] == "1"
        assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"
        assert c.launch.env == {}  # one GPU: no NCCL setting needed
        assert c.launch.image == FP8.engine.image
    # The EAGLE3 cell differs from the plain one only by the draft.
    i = eagle.launch.args.index("--speculative-config")
    assert eagle.launch.args[:i] + eagle.launch.args[i + 2 :] == plain.launch.args
    assert _drafts(eagle) == [LLAMA_EAGLE3] and _drafts(plain) == []
    # 55102ddb's head and draft length (4x L40S, BF16 target).
    (l40s,) = expand(load_experiment(LLAMA_RUNPOD), REGISTRY)
    assert _drafts(l40s) == [LLAMA_EAGLE3]
    spec = eagle.spec.engine.args["speculative_config"]
    assert spec == l40s.spec.engine.args["speculative_config"]
    # New configs: neither is any recorded 70B config.
    recorded = {
        c.config_hash
        for p in (LLAMA_H100_RUNPOD, LLAMA_H100_QUALITY_RUNPOD, LLAMA_H100_FP8_CAPACITY_RUNPOD)
        for c in expand(load_experiment(p), REGISTRY)
    }
    assert not {plain.config_hash, eagle.config_hash} & recorded
    assert plain.config_hash != eagle.config_hash


def test_the_fp8_checkpoint_and_a_32k_request_fit_one_h200():
    # vLLM refuses to start if one max-length request's KV cache does not fit.
    it = PRICES.instance("runpod", "secure", "h200-x1")
    usable = 0.95 * 140.4 * 2**30  # an H200 shows ~140.4 GiB
    activations = 6 * 2**30  # activations, CUDA graphs, the EAGLE3 head (~2 GB)
    kv_bytes_per_token = 80 * 8 * 128 * 2 * 2  # layers x KV heads x head dim x K,V x BF16
    tokens = (usable - FP8.hf.size_bytes - activations) / kv_bytes_per_token
    assert it.gpu_memory_gb * 1e9 > FP8.hf.size_bytes
    assert tokens > 5 * FP8.max_context  # ~195k tokens: ~95 concurrent 1k+1k requests
    # The planner's decode floor: 72.7 GB per step at 4.8 TB/s, vs 10.8 ms on 2x H100.
    (cell, _) = expand(load_experiment(LLAMA_H200_FP8_RUNPOD), REGISTRY)
    assert decode_step_ms(cell.spec) == pytest.approx(15.1, abs=0.1)


def test_the_h200_run_gates_every_cell_against_9e5f7866s_stored_bf16():
    exp, h100 = load_experiment(LLAMA_H200_FP8_RUNPOD), load_experiment(LLAMA_H100_RUNPOD)
    q = exp.quality
    assert q is not None and q.baseline_variant is None and q.baseline is not None
    assert str(q.baseline.experiment) == BF16_9E5F7866[0]
    assert q.baseline.config_hash.startswith(BF16_9E5F7866[1])
    # 9e5f7866's BF16 cell is 9f0853d7's (same config hash), on the same suite and subset.
    bf16 = expand(load_experiment(LLAMA_H100_QUALITY_RUNPOD), REGISTRY)[0]
    assert bf16.config_hash == q.baseline.config_hash
    assert (q.suite, q.subset, q.replicates) == (h100.quality.suite, h100.quality.subset, 1)
    assert q.load().covers(FP8.id)
    assert "tool_calling_strict" in {t.name for t in q.load().select(q.subset)}


def test_the_h200_run_measures_fixed_1k_1k_as_9f0853d7_did():
    exp, h100 = load_experiment(LLAMA_H200_FP8_RUNPOD), load_experiment(LLAMA_H100_RUNPOD)
    (w,) = exp.workloads
    (ref,) = [x for x in h100.workloads if x.profile == "fixed-1k-1k"]
    assert w.profile == ref.profile and w.overrides == ref.overrides
    assert (w.load.duration_s, w.load.warmup_s, w.load.drain_timeout_s) == (
        ref.load.duration_s,
        ref.load.warmup_s,
        ref.load.drain_timeout_s,
    )
    assert exp.slo == h100.slo and exp.repetitions == h100.repetitions
    s = w.load.search
    assert s is not None and s.scale == "geometric" and s.rel_tol <= 0.1 and s.descend == 2
    assert s.max_points == 5 and s.lo / s.step**s.descend == pytest.approx(1.0)


def test_the_h200_quote_fits_its_cap_to_ttl():
    exp = load_experiment(LLAMA_H200_FP8_RUNPOD)
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$21.50")
    assert plan.total_micros < plan.ttl_worst_micros <= plan.caps.effective
    assert plan.n_runs_max == 2 * 5 * 3
    (host,) = plan.hosts
    kinds = [s.kind for s in host.steps]
    assert kinds.count("cold_start") == 1 and kinds.count("warm_start") == 1
    assert kinds.count("eval") == 2
    assert host.seconds < 0.9 * host.ttl_s
    # The disk holds the checkpoint, the EAGLE3 head (~2 GB) and the engine headroom.
    assert exp.provider.container_disk_gb >= FP8.hf.size_bytes / 1e9 + 2.1 + RUNPOD_DISK_HEADROOM_GB


# --- the smoke ---------------------------------------------------------------------


def test_the_h200_smoke_runs_the_h200_paths():
    smoke, real = load_experiment(RUNPOD_SMOKE_H200), load_experiment(LLAMA_H200_FP8_RUNPOD)
    assert smoke.smoke and not real.smoke
    for field in ("instance_type", "gpu_type_id", "engine_env", "allowed_cuda_versions"):
        assert getattr(smoke.provider, field) == getattr(real.provider, field), field
    assert [v.name for v in smoke.variants] == [v.name for v in real.variants]
    assert _load_shape(smoke) == _load_shape(real)
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    (sw,), (rw,) = smoke.workloads, real.workloads
    assert (sw.load.search.step, sw.load.search.scale) == (
        rw.load.search.step,
        rw.load.search.scale,
    )
    assert smoke.quality.subset == real.quality.subset
    assert _eval_paths(real) <= _eval_paths(smoke)
    cells, real_cells = expand(smoke, REGISTRY), expand(real, REGISTRY)
    assert len({c.host_key for c in cells}) == 1
    assert {c.launch.image for c in cells} == {c.launch.image for c in real_cells}
    for c, r in zip(cells, real_cells, strict=True):
        assert (c.hardware["gpu"], c.hardware["gpu_type_id"], c.launch.gpus, c.launch.env) == (
            r.hardware["gpu"],
            r.hardware["gpu_type_id"],
            r.launch.gpus,
            r.launch.env,
        )
        assert c.spec.hf.quant_method == r.spec.hf.quant_method == "compressed-tensors"
        assert c.spec.quantization == r.spec.quantization == "fp8"
        for flag in ("--tensor-parallel-size", "--gpu-memory-utilization"):
            assert (
                c.launch.args[c.launch.args.index(flag) + 1]
                == r.launch.args[r.launch.args.index(flag) + 1]
            )
    # EAGLE3 on an FP8 target, the draft pinned and fetched on the warm restart.
    plain, eagle = cells
    assert _drafts(plain) == []
    ((repo, rev),) = _drafts(eagle)
    assert repo == "RedHatAI/Qwen3-8B-speculator.eagle3" and len(rev) == 40
    s_spec = eagle.spec.engine.args["speculative_config"]
    r_spec = real_cells[1].spec.engine.args["speculative_config"]
    assert {k: s_spec[k] for k in ("method", "num_speculative_tokens")} == {
        k: r_spec[k] for k in ("method", "num_speculative_tokens")
    }
    assert smoke.quality.load().covers(plain.spec.id)


def test_the_h200_smoke_gates_against_a_stored_smoke_baseline_at_its_limit():
    smoke = load_experiment(RUNPOD_SMOKE_H200)
    q = smoke.quality
    assert q is not None and q.baseline is not None and q.baseline_variant is None
    assert str(q.baseline.experiment) == BF16_SMOKE_8BC65CFA[0]
    assert q.baseline.config_hash.startswith(BF16_SMOKE_8BC65CFA[1])
    # 8bc65cfa's bf16 cell, evaluated at the same limit and subset (a smoke pairs only
    # with a smoke's evals at the same item count).
    sweep_smoke = load_experiment(RUNPOD_SMOKE_8B_SWEEP)
    bf16 = expand(sweep_smoke, REGISTRY)[0]
    assert bf16.variant == "bf16" and bf16.config_hash == q.baseline.config_hash
    assert (q.limit, q.subset, q.suite) == (
        sweep_smoke.quality.limit,
        sweep_smoke.quality.subset,
        sweep_smoke.quality.suite,
    )


def test_the_h200_smoke_fits_its_cap_to_ttl():
    plan = _plan(load_experiment(RUNPOD_SMOKE_H200))
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$4.50")
    assert plan.total_micros < plan.ttl_worst_micros <= plan.caps.effective
    (host,) = plan.hosts
    kinds = [s.kind for s in host.steps]
    assert kinds.count("cold_start") == 1 and kinds.count("warm_start") == 1
    assert kinds.count("eval") == 2


def test_option_b_fits_the_overall_cap_left_on_2026_10_31():
    # $149.75 overall, $96.57 billable spend in the results DB on 2026-10-31: $53.18 left.
    # The three caps together fit it, so each can be approved without raising the cap.
    left = parse_usd("$149.75") - parse_usd("$96.57")
    caps = [
        load_experiment(p).budget.max_spend
        for p in (RUNPOD_SMOKE_H200, LLAMA_H200_FP8_RUNPOD, LLAMA_H100_FP8_CAPACITY_RUNPOD)
    ]
    assert sum(caps) == parse_usd("$48.50") <= left
