"""The Qwen3-8B config sweep and what it added: the eval-time calibration, host groups,
pinned datasets on RunPod pods, and the follow-up on L40 and RTX 6000 Ada."""

from typing import Any

import pytest

from loom_bench.budget import caps_for, load_budget
from loom_bench.experiment import (
    RUNPOD_GPU_TYPE_IDS,
    Experiment,
    expand,
    hardware_for,
    load_experiment,
    load_quality_suite,
)
from loom_bench.money import parse_usd
from loom_bench.plan import (
    AWS_TIMING,
    DATASET_JOB_S,
    EVAL_HARNESS_TASK_S,
    EVAL_ITEM_S_PER_STEP_MS,
    GPU_MEMORY_BANDWIDTH_GB_S,
    MIN_DECODE_STEP_MS,
    RUNPOD_TIMING,
    build_plan,
    decode_step_ms,
    eval_task_key,
    runpod_accrual_terms,
)
from loom_bench.prices import load_prices
from loom_bench.quality.suite import SuiteTask
from loom_bench.registry import ModelSpec, load_registry
from loom_bench.runner import plan_experiment
from loom_bench.workloads import load_profile

from .conftest import QWEN, QWEN_SWEEP_RUNPOD, QWEN_WINNER_ADA_RUNPOD, RUNPOD_SMOKE_8B_SWEEP

REGISTRY = load_registry()
PRICES = load_prices()
BUDGET = load_budget()


def _plan(exp: Experiment):
    return build_plan(
        exp,
        expand(exp, REGISTRY),
        prices=PRICES,
        caps=caps_for(exp.budget.max_spend, BUDGET, 0),
    )


def _spec(model: str, gpu: str | None = None, gpus: int | None = None) -> ModelSpec:
    spec = REGISTRY.get(model)
    hw = spec.hardware.model_copy(
        update={
            "gpu": gpu or spec.hardware.gpu,
            "gpus_per_replica": gpus or spec.hardware.gpus_per_replica,
        }
    )
    return spec.model_copy(update={"hardware": hw})


# --- eval-time calibration --------------------------------------------------------------

SUITE_TASKS = {t.name: t for t in load_experiment(QWEN_SWEEP_RUNPOD).quality.load().tasks}  # type: ignore[union-attr]
CONCURRENCY = {"gsm8k": 32, "ifeval": 32}  # num_concurrent in both suites; native tasks 16

# Seconds per pass of each task in stored runs (events.jsonl `quality.seconds`, divided by
# the run's replicates), with the items it scored. json_schema at 60 items predates data
# version 2 (300 items) and is left out: a 60-item task is mostly fixed overhead.
MEASURED: list[tuple[str, ModelSpec, dict[str, tuple[float, int]]]] = [
    (
        "b03b3c52 Qwen3-8B BF16 vLLM, L40S (3 passes)",
        _spec("qwen3-8b"),
        {
            "gsm8k": (188.9, 1319),
            "ifeval": (181.5, 541),
            "json_schema": (72.9, 300),
            "tool_calling": (4.0, 60),
        },
    ),
    (
        "b03b3c52 Qwen3-8B BF16 SGLang, L40S (3 passes)",
        _spec("qwen3-8b"),
        {
            "gsm8k": (211.1, 1319),
            "ifeval": (198.0, 541),
            "json_schema": (80.0, 300),
            "tool_calling": (5.0, 60),
        },
    ),
    (
        "565b8d3f Qwen3-8B BF16 vLLM, L40S",
        _spec("qwen3-8b"),
        {"gsm8k": (193.8, 1319), "ifeval": (184.0, 541), "tool_calling": (4.0, 60)},
    ),
    (
        "cf4d1614 Llama 3.3 70B BF16 TP=4, 4x L40S",
        _spec("llama-3.3-70b-instruct"),
        {
            "gsm8k": (428.5, 1319),
            "ifeval": (459.9, 541),
            "json_schema": (121.9, 300),
            "tool_calling": (12.2, 60),
        },
    ),
    (
        "9f0853d7 Llama 3.3 70B BF16 TP=2, 2x H100 SXM",
        _spec("llama-3.3-70b-instruct", "H100", 2),
        {
            "gsm8k": (172.9, 1319),
            "ifeval": (188.4, 541),
            "json_schema": (53.5, 300),
            "tool_calling": (4.9, 60),
            "tool_calling_strict": (5.2, 60),
        },
    ),
    (
        "9f0853d7 Llama 3.3 70B FP8 TP=2, 2x H100 SXM",
        _spec("llama-3.3-70b-instruct-fp8", "H100", 2),
        {
            "gsm8k": (120.5, 1319),
            "ifeval": (120.8, 541),
            "json_schema": (31.9, 300),
            "tool_calling": (3.2, 60),
            "tool_calling_strict": (3.8, 60),
        },
    ),
]
# EAGLE3 speculative decoding (55102ddb, 70B on 4x L40S) only makes items faster.
EAGLE3 = (
    _spec("llama-3.3-70b-instruct"),
    {"gsm8k": (267.2, 1319), "ifeval": (314.7, 541), "json_schema": (64.1, 300)},
)


def _planned_task_s(name: str, spec: ModelSpec, items: int) -> float:
    task = SUITE_TASKS[name]
    item_s = EVAL_ITEM_S_PER_STEP_MS[eval_task_key(task)] * decode_step_ms(spec)
    harness = EVAL_HARNESS_TASK_S if task.kind == "lm_eval" else 0.0
    return items * item_s / CONCURRENCY.get(name, 16) + harness


@pytest.mark.parametrize(("label", "spec", "tasks"), MEASURED, ids=[m[0][:8] for m in MEASURED])
def test_every_stored_pass_comes_in_under_its_estimate_but_not_far_under(label, spec, tasks):
    # Conservative (never below a measured pass) but honest (at most 2.5x it: the old
    # fixed constants planned ~30 min for 8B passes that took 7-8).
    for name, (seconds, items) in tasks.items():
        planned = _planned_task_s(name, spec, items)
        assert seconds <= planned <= 2.5 * seconds, (label, name, planned, seconds)


def test_speculative_decoding_only_shortens_the_measured_passes():
    spec, tasks = EAGLE3
    for name, (seconds, items) in tasks.items():
        assert seconds <= _planned_task_s(name, spec, items), name


def test_calibrated_keys_name_tasks_the_shipped_suites_run():
    # A renamed harness task would silently fall back to the uncalibrated constant.
    keys = {eval_task_key(t) for t in SUITE_TASKS.values()}
    llama = load_quality_suite("llama-3.3-70b-instruct")
    assert set(EVAL_ITEM_S_PER_STEP_MS) <= keys
    assert {eval_task_key(t) for t in llama.tasks} >= set(EVAL_ITEM_S_PER_STEP_MS)


def test_decode_step_floor_reads_each_gpus_share_of_the_weights():
    assert decode_step_ms(_spec("qwen3-8b")) == pytest.approx(16381516776 / 864e9 * 1000)
    assert decode_step_ms(_spec("qwen3-8b-fp8")) < 0.6 * decode_step_ms(_spec("qwen3-8b"))
    four = decode_step_ms(_spec("llama-3.3-70b-instruct"))
    assert four == pytest.approx(141107497872 / 4 / 864e9 * 1000)  # ~40.8 ms
    # An unlisted GPU is planned at the slowest listed bandwidth; tiny shares hit the floor.
    slow = min(GPU_MEMORY_BANDWIDTH_GB_S.values())
    assert decode_step_ms(_spec("qwen3-8b", "B200")) == pytest.approx(16381516776 / slow / 1e6)
    assert decode_step_ms(_spec("qwen3-8b-fp8", "H200")) == MIN_DECODE_STEP_MS


def test_every_priced_gpu_has_a_memory_bandwidth():
    gpus = {
        it.gpu
        for cloud in PRICES.clouds.values()
        for region in cloud.values()
        for it in region.instances.values()
    }
    assert gpus <= set(GPU_MEMORY_BANDWIDTH_GB_S)


def test_mock_cells_keep_fixed_item_times():
    # The mock's simulated GPU reads no weights: its eval time scales with time_scale only
    # (bench/tests/runner/test_plan.py::test_eval_subset_and_mock_time_scale).
    task = SuiteTask(name="json_schema", kind="json_schema")
    assert eval_task_key(task) == "json_schema"
    gsm = SuiteTask(name="g", kind="lm_eval", params={"tasks": ["gsm8k_cot_llama"]})
    assert eval_task_key(gsm) == "lm_eval:gsm8k_cot_llama"


# --- host groups ------------------------------------------------------------------------


def _grouped(groups: list[str | None]) -> Experiment:
    doc = load_experiment(QWEN_SWEEP_RUNPOD).model_dump(mode="json", exclude_none=True)
    variants = doc["variants"][: len(groups)]
    for v, g in zip(variants, groups, strict=True):
        v.pop("host_group", None)
        if g is not None:
            v["host_group"] = g
    return Experiment.model_validate({**doc, "variants": variants})


def test_host_groups_split_one_image_over_pods_without_changing_configs():
    one = _grouped([None, None, None])
    two = _grouped(["a", "a", "b"])
    cells_one, cells_two = expand(one, REGISTRY), expand(two, REGISTRY)
    assert [c.config_hash for c in cells_one] == [c.config_hash for c in cells_two]
    assert len({c.host_key for c in cells_one}) == 1
    keys = [c.host_key for c in cells_two]
    assert keys[0] == keys[1] != keys[2] and keys[2].endswith("/group=b")
    plan = _plan(two)
    assert [h.cells for h in plan.hosts] == [["bf16", "bf16-kv8"], ["fp8"]]
    for host in plan.hosts:  # each pod pays its own cold start and eval setup
        kinds = [s.kind for s in host.steps]
        assert kinds[:2] == ["cold_start", "eval_setup"] and kinds.count("cold_start") == 1


# --- pinned datasets --------------------------------------------------------------------


def _with_chat(exp: Experiment, **profile_over: Any) -> Experiment:
    doc = exp.model_dump(mode="json", exclude_none=True)
    load = {"mode": "open_loop", "values": [1], "duration_s": 60, "warmup_s": 10}
    chat = {"profile": "chat-sharegpt", "load": load}
    if profile_over:
        chat["overrides"] = profile_over
    return Experiment.model_validate({**doc, "workloads": [chat]})


def test_chat_profile_pins_its_dataset_for_remote_hosts():
    p = load_profile("chat-sharegpt")
    assert p.download is not None and p.dataset is not None  # type: ignore[union-attr]
    assert p.dataset.revision == p.download.revision  # type: ignore[union-attr]
    assert p.download.url.startswith(  # type: ignore[union-attr]
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/"
    )


def test_a_dataset_workload_on_a_pod_is_planned_with_its_download_and_tokenizing():
    exp = _with_chat(load_experiment(QWEN_SWEEP_RUNPOD))
    plan = _plan(exp)
    assert not [r for r in plan.refusals if "chat-sharegpt" in r]
    p = load_profile("chat-sharegpt")
    for host in plan.hosts:
        fetches = [s for s in host.steps if s.kind == "dataset"]
        assert len(fetches) == 1  # once per pod, before its first chat run
        assert fetches[0].seconds == pytest.approx(
            p.download.size_bytes / RUNPOD_TIMING.download_bytes_per_s  # type: ignore[union-attr]
        )
    step = next(s for s in plan.hosts[0].steps if s.kind == "workload")
    assert step.seconds == pytest.approx(
        exp.repetitions * (60 + 60 + RUNPOD_TIMING.run_overhead_s + DATASET_JOB_S)
    )


def test_a_dataset_the_host_cannot_fetch_is_refused_before_anything_starts():
    no_download = _with_chat(load_experiment(QWEN_SWEEP_RUNPOD), download=None)
    (refusal,) = [r for r in _plan(no_download).refusals if "chat-sharegpt" in r]
    assert "runpod host does not have" in refusal and "download" in refusal
    aws_no_download = _with_chat(load_experiment(QWEN), download=None)
    (refusal,) = [r for r in _plan(aws_no_download).refusals if "chat-sharegpt" in r]
    assert "aws_ec2 host does not have" in refusal and "download" in refusal


def test_a_pinned_dataset_on_an_ec2_host_is_planned_with_its_download():
    # Since 2026-10-10 the aws_ec2 host fetches a pinned dataset itself, as a pod does.
    exp = _with_chat(load_experiment(QWEN))
    plan = _plan(exp)
    assert not [r for r in plan.refusals if "chat-sharegpt" in r]
    p = load_profile("chat-sharegpt")
    (host,) = plan.hosts
    fetches = [s for s in host.steps if s.kind == "dataset"]
    assert len(fetches) == 1
    assert fetches[0].seconds == pytest.approx(
        p.download.size_bytes / AWS_TIMING.download_bytes_per_s  # type: ignore[union-attr]
    )


# --- the sweep and its smoke ------------------------------------------------------------


def test_the_sweep_fits_a_25_dollar_cap_even_if_every_pod_lives_to_its_ttl():
    exp = load_experiment(QWEN_SWEEP_RUNPOD)
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd("$25")
    assert plan.ttl_worst_micros <= plan.caps.effective
    assert exp.budget.ttl_s <= runpod_accrual_terms().max_ttl_s
    assert [h.cells for h in plan.hosts] == [
        ["bf16", "bf16-kv8"],
        ["fp8", "fp8-kv8"],
        ["fp8-kv8-mbt1024"],
    ]
    for host in plan.hosts:
        assert host.provider == "runpod" and host.seconds < 0.95 * host.ttl_s


def test_the_sweep_gates_every_candidate_against_bf16_with_replicated_passes():
    exp = load_experiment(QWEN_SWEEP_RUNPOD)
    assert exp.quality is not None
    assert exp.quality.baseline_variant == "bf16" and exp.quality.replicates == 3
    assert exp.quality.subset == "phase0-strict" and exp.quality.limit is None
    cells = {c.key: c for c in expand(exp, REGISTRY)}
    assert cells["bf16"].spec.quantization == "none"
    assert cells["bf16"].spec.kv_cache_dtype == "auto"
    assert {k for k, c in cells.items() if c.spec.id == "qwen3-8b-fp8"} == {
        "fp8",
        "fp8-kv8",
        "fp8-kv8-mbt1024",
    }
    assert {k for k, c in cells.items() if c.spec.kv_cache_dtype == "fp8"} == {
        "bf16-kv8",
        "fp8-kv8",
        "fp8-kv8-mbt1024",
    }
    knob = cells["fp8-kv8-mbt1024"].launch.args
    assert knob[knob.index("--max-num-batched-tokens") + 1] == "1024"
    # The realistic workload leads (reports order chat first too), 1k/1k follows.
    assert [w.profile for w in exp.workloads] == ["chat-sharegpt", "fixed-1k-1k"]


def test_the_sweep_windows_sample_enough_requests_to_be_trusted():
    # Poisson arrivals: N requests in a window give throughput a run-to-run CV of
    # 1/sqrt(N) (output tokens sqrt((1 + cv_len^2) / N), cv_len 0.63 on chat). At the
    # bf16 knees (~3 req/s chat, ~1.2 req/s 1k/1k) the chat window must keep that CV under
    # 5%; 1k/1k, kept for comparability, under 7%. 565b8d3f's windows held 100-200.
    exp = load_experiment(QWEN_SWEEP_RUNPOD)
    knee = {"chat-sharegpt": (3.0, 0.63, 0.05), "fixed-1k-1k": (1.2, 0.0, 0.07)}
    for w in exp.workloads:
        rate, cv_len, limit = knee[w.profile]
        assert w.load.duration_s is not None and w.load.warmup_s is not None
        n = rate * (w.load.duration_s - w.load.warmup_s)
        assert ((1 + cv_len**2) / n) ** 0.5 < limit, w.profile
        # Warmup covers a request's life at the knee (~50 s on 1k/1k, under 45 s for 99% of
        # chat requests), so the measured window starts at steady concurrency.
        assert w.load.warmup_s >= {"chat-sharegpt": 45, "fixed-1k-1k": 60}[w.profile]
        search = w.load.search
        assert search is not None and search.scale == "geometric" and search.rel_tol <= 0.05


def _search_shape(exp: Experiment) -> dict[str, tuple[Any, ...]]:
    """Profile -> (load mode, search scale, step, whether it can descend below lo)."""
    out: dict[str, tuple[Any, ...]] = {}
    for w in exp.workloads:
        s = w.load.search
        assert s is not None
        out[w.profile] = (w.load.mode, s.scale, s.step, bool(s.descend))
    return out


def test_the_smoke_runs_every_path_of_the_sweep():
    smoke, real = load_experiment(RUNPOD_SMOKE_8B_SWEEP), load_experiment(QWEN_SWEEP_RUNPOD)
    assert smoke.smoke and not real.smoke
    assert smoke.model == real.model and smoke.provider == real.provider
    # Same variants: names, host groups (so the same three pods), checkpoints, KV-cache
    # dtypes and engine args, hence the same config hashes and launches.
    assert smoke.variants == real.variants
    assert [c.config_hash for c in expand(smoke, REGISTRY)] == [
        c.config_hash for c in expand(real, REGISTRY)
    ]
    assert _search_shape(smoke) == _search_shape(real)
    # The chat workload keeps its pinned download: the pod fetches the real file.
    chat = next(w for w in smoke.workloads if w.profile == "chat-sharegpt")
    assert chat.resolve().download == load_profile("chat-sharegpt").download  # type: ignore[union-attr]
    assert smoke.repetitions == real.repetitions and smoke.slo == real.slo
    assert smoke.quality is not None and real.quality is not None
    assert smoke.quality.limit is not None and real.quality.limit is None
    assert smoke.quality.model_copy(update={"limit": None}) == real.quality


def test_the_smoke_fits_its_cap_to_ttl():
    plan = _plan(load_experiment(RUNPOD_SMOKE_8B_SWEEP))
    assert plan.ok, plan.refusals
    assert plan.ttl_worst_micros <= plan.caps.effective <= parse_usd("$4.25")
    assert len(plan.hosts) == 3
    for host in plan.hosts:
        assert host.seconds < 0.95 * host.ttl_s


# --- the L40 / RTX 6000 Ada follow-up ---------------------------------------------------

SWEEP_EXPERIMENT = "7a8237d0-9917-47e7-b93e-8cb0230e0059"  # the sweep, 2026-10-09/10
ADA_CARDS = {
    "winner-l40": ("L40", "l40-x1"),
    "winner-rtx6000ada": ("RTX 6000 Ada", "rtx6000ada-x1"),
}


def test_the_follow_up_runs_the_sweeps_winner_unchanged_on_each_ada_card():
    follow, sweep = load_experiment(QWEN_WINNER_ADA_RUNPOD), load_experiment(QWEN_SWEEP_RUNPOD)
    winner = next(c for c in expand(sweep, REGISTRY) if c.key == "fp8-kv8")
    cells = {c.key: c for c in expand(follow, REGISTRY)}
    assert set(cells) == set(ADA_CARDS)
    for key, cell in cells.items():
        gpu, instance = ADA_CARDS[key]
        assert cell.spec.hardware.gpu == gpu
        # Everything but the GPU is the sweep's cell: checkpoint, KV cache, engine launch.
        assert cell.spec.id == winner.spec.id and cell.spec.hf == winner.spec.hf
        assert cell.spec.kv_cache_dtype == winner.spec.kv_cache_dtype
        assert cell.launch.args == winner.launch.args and cell.launch.image == winner.launch.image
        assert cell.config_hash != winner.config_hash  # a new GPU is a new config
        _, hw = hardware_for(follow, cell.spec)
        assert (hw["instance_type"], hw["gpu_type_id"]) == (instance, RUNPOD_GPU_TYPE_IDS[gpu])
        assert PRICES.instance("runpod", "secure", instance).gpu == gpu
    # RunPod's own ids (gpuTypes, 2026-10-09).
    assert RUNPOD_GPU_TYPE_IDS["L40"] == "NVIDIA L40"
    assert RUNPOD_GPU_TYPE_IDS["RTX 6000 Ada"] == "NVIDIA RTX 6000 Ada Generation"
    assert len({c.host_key for c in cells.values()}) == 2  # one pod per card


def test_the_follow_up_measures_chat_as_the_sweep_does():
    follow, sweep = load_experiment(QWEN_WINNER_ADA_RUNPOD), load_experiment(QWEN_SWEEP_RUNPOD)
    (chat,) = follow.workloads
    ref = next(w for w in sweep.workloads if w.profile == "chat-sharegpt")
    assert chat.profile == ref.profile and chat.overrides == ref.overrides
    for k in ("mode", "duration_s", "warmup_s", "drain_timeout_s"):
        assert getattr(chat.load, k) == getattr(ref.load, k), k
    assert chat.load.search is not None and ref.load.search is not None
    for k in ("rel_tol", "scale", "step"):
        assert getattr(chat.load.search, k) == getattr(ref.load.search, k), k
    assert follow.repetitions == sweep.repetitions and follow.slo == sweep.slo
    assert follow.quality is not None and sweep.quality is not None
    assert follow.quality.baseline is not None and follow.quality.baseline_variant is None
    assert (follow.quality.suite, follow.quality.subset, follow.quality.replicates) == (
        sweep.quality.suite,
        sweep.quality.subset,
        sweep.quality.replicates,
    )


def test_the_follow_up_fits_its_cap_to_ttl():
    plan = _plan(load_experiment(QWEN_WINNER_ADA_RUNPOD))
    assert plan.ok, plan.refusals
    assert plan.ttl_worst_micros <= plan.caps.effective == parse_usd("$5")
    assert plan.total_micros < parse_usd("$4")
    assert [h.cells for h in plan.hosts] == [["winner-l40"], ["winner-rtx6000ada"]]
    l40s = PRICES.instance("runpod", "secure", "l40s-x1").on_demand_per_hour
    rates = [h.hourly_micros for h in plan.hosts]
    assert rates == sorted(rates) and rates[-1] < l40s  # both cheaper per hour
    for host in plan.hosts:
        assert host.seconds < 0.85 * host.ttl_s


def test_the_follow_up_is_gated_against_the_sweeps_stored_bf16_baseline():
    follow = load_experiment(QWEN_WINNER_ADA_RUNPOD)
    assert follow.quality is not None and follow.quality.baseline is not None
    assert str(follow.quality.baseline.experiment) == SWEEP_EXPERIMENT
    sweep_cells = expand(load_experiment(QWEN_SWEEP_RUNPOD), REGISTRY)
    bf16 = next(c for c in sweep_cells if c.key == "bf16")
    assert follow.quality.baseline.config_hash == bf16.config_hash


def test_the_follow_up_is_refused_where_the_sweeps_baseline_is_not_stored(ctx):
    # The baseline's evals live in the results database the sweep ran against; anywhere
    # else (this empty test database) planning refuses before any pod exists.
    _, plan = plan_experiment(load_experiment(QWEN_WINNER_ADA_RUNPOD), ctx)
    assert not plan.ok
    assert any(r.startswith("quality.baseline:") for r in plan.refusals)
