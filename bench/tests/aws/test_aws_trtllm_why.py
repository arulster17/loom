"""The follow-up to reports/8b-trtllm-v1's two open "why"s, before it spends anything.

- qwen3-8b-trtllm-bf16-profile-aws: profile windows on 46938d1f's trtllm-bf16 cell (its
  config hash unchanged: TensorRT-LLM profiles at run time).
- qwen3-8b-trtllm-fp8-cudagraph-aws: the CUDA-graph padding variant (its own hash) next
  to stock FP8 55b35655, at 08204560's chat points, gsm8k-only gated.
- aws-smoke-trtllm-why: their variants, windows and quality at smoke scale.

The specs load and expand to the hashes they claim, each plan fits its cap even with its
host to TTL, the smoke carries every variant, window shape and quality setting of the
real runs, and end to end on the aws_ec2 provider (moto, HostSim behind SSM) the
TensorRT-LLM windows send TensorRT-LLM's /start_profile body and come back through the
container's bind mount with TensorRT-LLM's step labels read."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from loom_bench.budget import load_budget
from loom_bench.engines import docker_run_argv, profiler_trace_dir, trtllm_options
from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.money import parse_usd
from loom_bench.profile.spec import TRTLLM_TRACE_DIR

from .hostsim import HostSim, array, var
from .test_aws_g6e import ON_DEMAND_ACCRUAL, REGISTRY, _plan, run_on_host_sim
from .test_aws_perf_track import PINNED_PROVIDER, VLLM_BF16_HASH
from .test_aws_perf_track import at_test_scale as perf_at_test_scale

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("aws-e2e")]

X = EXPERIMENTS_DIR
SMOKE = X / "aws-smoke-trtllm-why.yaml"
BF16 = X / "qwen3-8b-trtllm-bf16-profile-aws.yaml"
FP8 = X / "qwen3-8b-trtllm-fp8-cudagraph-aws.yaml"
# The quotes (bench plan, 2026-11-04): spec -> (cap, TTL minutes).
QUOTES = {
    "aws-smoke-trtllm-why": ("$2.55", 80),
    "qwen3-8b-trtllm-bf16-profile-aws": ("$2.55", 80),
    "qwen3-8b-trtllm-fp8-cudagraph-aws": ("$5.05", 160),
}
# 46938d1f's TensorRT-LLM cells as recorded (and 08204560's trtllm-fp8).
TRT_BF16_HASH = "ae3b5705d2fa54e053eeadaf507ef46ed681a1fbcf79631dc7f65f4f5457fb8a"
TRT_FP8_HASH = "55b356557da002ca376de348fdcdd39422699bc205e2f9016776877ca4509b80"
PADDING = {"cuda_graph_config": {"enable_padding": True, "max_batch_size": 128}}
STORED_BASELINE = ("46938d1f-3022-4695-a0b9-29b18c06eaea", VLLM_BF16_HASH)


def _cells(path: Path) -> dict[str, Any]:
    return {c.variant: c for c in expand(load_experiment(path), REGISTRY)}


# --- the specs ------------------------------------------------------------------------------


@pytest.mark.parametrize("path", [SMOKE, BF16, FP8], ids=lambda p: p.stem)
def test_each_spec_loads_on_one_pinned_g6e_xlarge(path: Path) -> None:
    exp = load_experiment(path)
    assert exp.name == path.stem
    assert yaml.safe_load(path.read_text())["provider"] == PINNED_PROVIDER
    assert exp.profile is not None and exp.profile.trace_dir == TRTLLM_TRACE_DIR
    for cell in expand(exp, REGISTRY):
        assert cell.launch.engine == "trtllm"
        assert cell.hardware["instance_type"] == "g6e.xlarge"


def test_the_stock_cells_keep_their_recorded_hashes() -> None:
    # Profiling changes nothing at launch, and naming the registry's own engine for the
    # profile check is no patch: the stock cells pair with 46938d1f's and 08204560's runs.
    assert _cells(BF16)["trtllm-bf16"].config_hash == TRT_BF16_HASH
    assert _cells(FP8)["trtllm-fp8"].config_hash == TRT_FP8_HASH
    smoke = _cells(SMOKE)
    assert smoke["trtllm-bf16"].config_hash == TRT_BF16_HASH


def test_the_padded_variant_is_stock_fp8_plus_padding_with_its_own_hash() -> None:
    cells = _cells(FP8)
    pad, stock = cells["trtllm-fp8-cudagraph-pad"], cells["trtllm-fp8"]
    assert pad.config_hash != stock.config_hash
    assert not pad.config_hash.startswith("55b35655")
    assert trtllm_options(pad.spec) == {**trtllm_options(stock.spec), **PADDING}
    # Nothing else differs: same checkpoint, image, flags.
    assert pad.spec.model_copy(update={"engine": stock.spec.engine}) == stock.spec
    assert pad.launch.args[1:] == stock.launch.args[1:]
    assert pad.launch.image == stock.launch.image
    assert pad.host_key == stock.host_key  # one host, a warm restart between them
    assert _cells(SMOKE)["trtllm-fp8-cudagraph-pad"].config_hash == pad.config_hash


def test_the_fp8_points_are_08204560s_and_the_bf16_ones_14d959e5s_and_46938d1fs() -> None:
    (chat,) = load_experiment(FP8).workloads
    assert chat.load.values == [2.619489972287021, 3.375]  # stored search points, exactly
    assert load_experiment(FP8).profile.windows[0].load_values == [3.375]  # type: ignore[union-attr]
    bf16 = {w.name: w.load for w in load_experiment(BF16).workloads}
    assert bf16["fixed-1k-1k"].values == [16, 64] and bf16["chat-sharegpt"].values == [2.25]
    vllm = {w.name: w.load for w in load_experiment(X / "qwen3-8b-profile-aws.yaml").workloads}
    k1 = bf16["fixed-1k-1k"]
    assert (k1.mode, k1.duration_s, k1.warmup_requests) == (
        vllm["fixed-1k-1k"].mode,
        vllm["fixed-1k-1k"].duration_s,
        vllm["fixed-1k-1k"].warmup_requests,
    )


def test_quality_is_gsm8k_only_against_the_stored_vllm_bf16_evals() -> None:
    for path in (FP8, SMOKE):
        q = load_experiment(path).quality
        assert q is not None and q.subset == "gsm8k-only" and q.replicates == 1
        assert q.limit is None  # a stored baseline pairs item for item
        assert q.baseline is not None
        assert (str(q.baseline.experiment), q.baseline.config_hash) == STORED_BASELINE
        assert [t.name for t in q.load().select(q.subset)] == ["gsm8k"]
    assert load_experiment(BF16).quality is None


@pytest.mark.parametrize("name", sorted(QUOTES))
def test_each_plan_uses_one_host_and_fits_its_cap_to_ttl(name: str) -> None:
    cap, ttl = QUOTES[name]
    exp = load_experiment(X / f"{name}.yaml")
    assert exp.budget.ttl_minutes == ttl
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd(cap) <= parse_usd(load_budget().per_experiment_cap)
    assert plan.ttl_worst_micros <= plan.caps.effective < plan.ttl_worst_micros + 100_000
    (host,) = plan.hosts
    assert host.market == "on_demand" and host.hourly_micros == ON_DEMAND_ACCRUAL
    assert [s.kind for s in host.steps].count("cold_start") == 1
    assert host.seconds < 0.85 * host.ttl_s
    windows = [s for s in host.steps if s.kind == "profile"]
    assert windows, "every spec profiles"


def test_the_three_together_stay_near_ten_dollars_and_inside_the_overall_cap() -> None:
    budget = load_budget()
    caps = sum(parse_usd(cap) for cap, _ in QUOTES.values())
    assert caps == parse_usd("$10.15")
    assert caps <= parse_usd(budget.overall_cap)


def test_every_trtllm_cell_mounts_the_trace_dir_its_windows_name() -> None:
    for path in (SMOKE, BF16, FP8):
        for cell in expand(load_experiment(path), REGISTRY):
            assert profiler_trace_dir(cell.launch) == TRTLLM_TRACE_DIR
            argv = docker_run_argv(cell.launch, weights_dir="/w", container_name="c")
            mount = f"{TRTLLM_TRACE_DIR}:{TRTLLM_TRACE_DIR}"
            assert argv[argv.index(mount) - 1] == "--volume"


def test_the_smoke_carries_every_variant_window_and_quality_setting_of_the_real_runs() -> None:
    smoke = load_experiment(SMOKE)
    real = [load_experiment(BF16), load_experiment(FP8)]
    assert smoke.smoke and not any(r.smoke for r in real)
    strip = {"host_group"}
    smoke_variants = {v.name: v.model_dump(exclude=strip) for v in smoke.variants}
    for r in real:
        assert r.provider == smoke.provider
        for v in r.variants:
            if v.name == "trtllm-fp8":
                continue  # stock FP8: ran in 08204560; its windows take trtllm-bf16's path
            assert smoke_variants[v.name] == v.model_dump(exclude=strip)
    assert smoke.profile is not None
    for r in real:
        assert r.profile is not None
        assert (r.profile.trace_dir, r.profile.max_trace_mb) == (
            smoke.profile.trace_dir,
            smoke.profile.max_trace_mb,
        )
        # every (workload, load mode) a real window profiles, the smoke profiles too
        modes = {w.name: w.load.mode for w in r.workloads}
        smoke_modes = {w.name: w.load.mode for w in smoke.workloads}
        for w in r.profile.windows:
            assert smoke_modes[w.workload] == modes[w.workload]
            assert w.workload in [sw.workload for sw in smoke.profile.windows]
    fp8_q = real[1].quality
    assert smoke.quality == fp8_q
    # the smoke's cells are the real runs' cells
    real_hashes = {c.config_hash for r in (BF16, FP8) for c in _cells(r).values()}
    assert {c.config_hash for c in _cells(SMOKE).values()} <= real_hashes


# --- end to end on the aws_ec2 provider ---------------------------------------------------


def _without_quality(path: Path, tmp: Path, body: bytes) -> Experiment:
    """The spec at perf-track test scale, quality left out (a stored baseline needs
    46938d1f in the DB; tests/quality covers the stored-baseline gate), with answers
    long enough that a window's closed loop still runs when the profiler starts."""
    doc = yaml.safe_load(path.read_text())
    doc.pop("quality", None)
    flat = tmp / path.name
    flat.write_text(yaml.safe_dump(doc))
    exp = perf_at_test_scale(flat, tmp, body).model_dump(mode="json", exclude_unset=True)
    for w in exp["workloads"]:
        if w["profile"] == "chat-sharegpt":
            w["overrides"].update(max_output_len=256)
        else:
            w["overrides"].update(output_len=256)
    return Experiment.model_validate(exp)


class SlowerHostSim(HostSim):
    """Engines at the speed tests/profile uses (HostSim's default is 100x)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.mock = {"time_scale": 0.3}


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    return run_on_host_sim(
        SMOKE, tmp_path_factory.mktemp("trtllm-why"), build=_without_quality, sim_cls=SlowerHostSim
    )


def test_trtllm_windows_come_back_through_the_mount_with_its_step_labels(smoke_run) -> None:
    outcome, sim, ctx, cells, exp, instances, reaped = smoke_run
    assert outcome.status.value == "completed", outcome.reason
    (inst,) = instances
    assert inst["State"]["Name"] == "terminated" and reaped == []
    starts = sim.scripts_of("start_engine")
    assert len(starts) == 2
    for script in starts:
        cmd = array(script, "ENGINE_CMD")
        assert f"{TRTLLM_TRACE_DIR}:{TRTLLM_TRACE_DIR}" in cmd
    # the padded cell starts with the padding in its LLM-API options, stock BF16 without
    options = [
        json.loads(a)
        for s in starts
        for a in array(s, "ENGINE_CMD")
        if a.startswith("{") and "guided_decoding_backend" in a
    ]
    assert [o.get("cuda_graph_config") for o in options] == [None, PADDING["cuda_graph_config"]]
    collects = sim.scripts_of("collect_profile")
    assert len(collects) == len(cells) * len(exp.workloads)
    assert {var(s, "TRACE_DIR") for s in collects} == {TRTLLM_TRACE_DIR}
    events = [
        json.loads(line)
        for line in (ctx.out_dir / str(outcome.experiment_id) / "events.jsonl")
        .read_text()
        .splitlines()
    ]
    assert not [e for e in events if e["kind"] == "profile_failed"]
    windows = sorted((ctx.out_dir / str(outcome.experiment_id) / "profiles").iterdir())
    assert len(windows) == len(collects)
    for w in windows:
        doc = json.loads((w / "window.json").read_text())
        cap = doc["capture"]
        assert cap["api"] == "trtllm" and "start_options_refused" not in cap
        assert cap["start_status"] == 200 and cap["stop_status"] == 200
        with tarfile.open(w / "trace.tar.gz") as t:
            names = t.getnames()
        assert names and all("trtllm-trace-" in n and n.endswith("-rank-0.json") for n in names)
        if doc["workload"] != "fixed-1k-1k":
            # the test dataset's chat answers are a few tokens: its window's closed loop
            # has run out by the time the profiler starts, so its trace has no steps
            continue
        assert "breakdown" in doc, doc.get("breakdown_error")
        (rank,) = doc["breakdown"]["ranks"]
        # counted from the bare step[...] labels only, never the `sample` scopes
        assert rank["step_source"] == "gpu annotations"
        assert 0 < rank["steps"] <= 12  # num_steps, which TensorRT-LLM honours
        assert rank["step_shapes"], "TensorRT-LLM's step labels split the window"
