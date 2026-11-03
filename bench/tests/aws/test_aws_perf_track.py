"""The three perf-track runs (profiling, speculative decoding, TensorRT-LLM vs vLLM) on
one AWS g6e.xlarge on-demand host at a time, and their smokes, before any of them spends
anything. RunPod Secure 1x L40S was out of stock on 2026-11-02; the account's on-demand
G/VT quota (4 vCPU) fits exactly one g6e.xlarge, the same L40S.

- Each AWS spec is its RunPod original with only the provider block and budget changed
  (and, for TensorRT-LLM, a host group for trtllm-bf16: AWS keys hosts by instance type,
  not image), so the twins compare point for point; each AWS smoke mirrors its AWS run.
- Every plan uses one host at a time, fits its cap even to TTL, and the six together fit
  the overall cap; TensorRT-LLM's start is priced at the EC2 pull rate.
- End to end on the aws_ec2 provider (moto EC2 and S3, HostSim behind SSM), each run at
  test scale: profile traces written through the engine container's bind mount and
  collected over SSM; n-gram and EAGLE3 rows with the EAGLE3 head fetched on its warm
  restart, the CommitPackFT file fetched onto the host and the EAGLE3 rows' output text
  stored; TensorRT-LLM on three hosts in a row, each launched only after the previous one
  is terminated (the quota fits one), with its wrapper, ulimits and Prometheus path."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from sqlalchemy import select

from loom_bench.budget import load_budget
from loom_bench.engines import (
    ENTRYPOINTS,
    TRTLLM_SERVE_SOURCE,
    docker_run_argv,
    profiler_trace_dir,
)
from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand, load_experiment
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.money import parse_usd
from loom_bench.plan import AWS_ENGINE_START_EXTRA_S, AWS_TIMING, _aws_disk_problems
from loom_bench.store.db import session_scope
from loom_bench.store.models import BenchRun
from loom_bench.store.parquet import read_requests
from loom_bench.workloads import load_profile

from .fakes import client_error
from .hostsim import HostSim, array, pairs, var
from .test_aws_g6e import ON_DEMAND_ACCRUAL, PRICES, REGISTRY, _plan, _suite, run_on_host_sim

pytestmark = [pytest.mark.timeout(300), pytest.mark.xdist_group("aws-e2e")]

X = EXPERIMENTS_DIR
# run -> (AWS smoke, AWS real, RunPod smoke, RunPod real)
RUNS = {
    "profile": (
        X / "aws-smoke-profile.yaml",
        X / "qwen3-8b-profile-aws.yaml",
        X / "runpod-smoke-profile.yaml",
        X / "qwen3-8b-profile-runpod.yaml",
    ),
    "specdec": (
        X / "aws-smoke-specdec.yaml",
        X / "qwen3-8b-specdec-aws.yaml",
        X / "runpod-smoke-specdec.yaml",
        X / "qwen3-8b-specdec-runpod.yaml",
    ),
    "trtllm": (
        X / "aws-smoke-trtllm.yaml",
        X / "qwen3-8b-trtllm-vs-vllm-aws.yaml",
        X / "runpod-smoke-trtllm.yaml",
        X / "qwen3-8b-trtllm-vs-vllm-runpod.yaml",
    ),
}
TWINS = [(aws, runpod) for s, r, rs, rr in RUNS.values() for aws, runpod in ((s, rs), (r, rr))]
PAIRS = [(s, r) for s, r, _, _ in RUNS.values()]
# The quotes (bench plan, 2026-11-02): spec -> (cap, TTL minutes, hosts).
QUOTES = {
    "aws-smoke-profile": ("$1.75", 55, 1),
    "qwen3-8b-profile-aws": ("$5.25", 165, 1),
    "aws-smoke-specdec": ("$3.75", 115, 1),
    "qwen3-8b-specdec-aws": ("$15.25", 480, 1),
    "aws-smoke-trtllm": ("$6.25", 65, 3),
    "qwen3-8b-trtllm-vs-vllm-aws": ("$27", 285, 3),
}
# Billable spend in the results DB when the quotes were made (bench plan, 2026-11-02).
SPENT_AT_QUOTE = parse_usd("$96.57")
PINNED_PROVIDER = {
    "kind": "aws_ec2",
    "region": "us-east-1",
    "instance_type": "g6e.xlarge",
    "market": "on_demand",
    "disk_gb": 200,
}
# 95cde129's bf16 cell (vLLM v0.30.0 BF16 on g6e.xlarge on-demand), as recorded.
VLLM_BF16_HASH = "48f3d8d1850a0b1bfc6f454ee35a29902d4dc858ed984252defa8ffe5f621375"
EAGLE3 = ("RedHatAI/Qwen3-8B-speculator.eagle3", "08610ffa01dd9f16731fe8f627b85905b6aa51c4")
FP8_MODELOPT = ("nvidia/Qwen3-8B-FP8", "2cebc4c89e25abc17668c81b01dceaf3d8b914d5")


def _doc(path: Path) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text()))


# --- the specs ------------------------------------------------------------------------------


@pytest.mark.parametrize(("aws", "runpod"), TWINS, ids=lambda p: p.stem)
def test_each_aws_spec_is_its_runpod_original_on_one_g6e_xlarge(aws, runpod):
    a, r = _doc(aws), _doc(runpod)
    assert a["provider"] == PINNED_PROVIDER  # pinned, no fallback
    assert a["name"] == aws.stem
    for key in ("name", "description", "provider", "budget"):
        a.pop(key)
        r.pop(key)
    # TensorRT-LLM: AWS keys a host by instance type, not image, so trtllm-bf16 needs its
    # own host group to keep RunPod's three hosts (one per cell).
    if "trtllm" in aws.stem:
        (trt,) = [v for v in a["variants"] if v["name"] == "trtllm-bf16"]
        assert trt.pop("host_group") == "trtllm-bf16"
    assert a == r  # variants, workloads, windows, search, SLO, quality, profile, seed
    a_cells, r_cells = (
        expand(load_experiment(aws), REGISTRY),
        expand(load_experiment(runpod), REGISTRY),
    )
    assert [c.key for c in a_cells] == [c.key for c in r_cells]
    for ac, rc in zip(a_cells, r_cells, strict=True):
        assert ac.spec == rc.spec  # same registry rows and engine args
        assert (ac.launch.image, ac.launch.args, ac.launch.env) == (
            rc.launch.image,
            rc.launch.args,
            rc.launch.env,
        )
        assert ac.hardware["instance_type"] == "g6e.xlarge" and ac.hardware["gpu"] == "L40S"
        assert ac.config_hash != rc.config_hash  # the hardware is part of the hash


@pytest.mark.parametrize(("smoke", "real"), PAIRS, ids=lambda p: p.stem)
def test_the_aws_smoke_mirrors_the_aws_run_except_scale(smoke, real):
    s, r = load_experiment(smoke), load_experiment(real)
    assert s.smoke and not r.smoke
    assert s.model == r.model and s.provider == r.provider and s.variants == r.variants
    s_cells, r_cells = expand(s, REGISTRY), expand(r, REGISTRY)
    assert [c.config_hash for c in s_cells] == [c.config_hash for c in r_cells]
    assert [c.host_key for c in s_cells] == [c.host_key for c in r_cells]
    assert [(w.profile, w.name, w.load.mode, bool(w.load.search)) for w in s.workloads] == [
        (w.profile, w.name, w.load.mode, bool(w.load.search)) for w in r.workloads
    ]
    assert s.repetitions == r.repetitions and s.slo == r.slo
    if r.quality is not None:
        assert s.quality is not None and s.quality.limit is not None
        assert s.quality.model_copy(update={"limit": None}) == r.quality
    else:
        assert s.quality is None
    if r.profile is not None:
        assert s.profile is not None
        assert [w.workload for w in s.profile.windows] == [w.workload for w in r.profile.windows]
        assert (s.profile.trace_dir, s.profile.max_trace_mb) == (
            r.profile.trace_dir,
            r.profile.max_trace_mb,
        )
    else:
        assert s.profile is None


@pytest.mark.parametrize("name", sorted(QUOTES))
def test_each_plan_uses_one_host_at_a_time_and_fits_its_cap_to_ttl(name):
    cap, ttl, n_hosts = QUOTES[name]
    exp = load_experiment(X / f"{name}.yaml")
    assert exp.budget.ttl_minutes == ttl
    plan = _plan(exp)
    assert plan.ok, plan.refusals
    assert plan.caps.effective == parse_usd(cap) <= parse_usd(load_budget().per_experiment_cap)
    # A cap a little above the worst case, as the RunPod quotes were set.
    assert plan.ttl_worst_micros <= plan.caps.effective < plan.ttl_worst_micros + 250_000
    assert len(plan.hosts) == n_hosts
    for host in plan.hosts:
        assert host.market == "on_demand" and host.hourly_micros == ON_DEMAND_ACCRUAL
        assert host.instance_type == "g6e.xlarge"
        assert host.steps[0].kind == "cold_start" and host.steps[-1].kind == "teardown"
        assert [s.kind for s in host.steps].count("cold_start") == 1
        assert host.seconds < 0.95 * host.ttl_s


def test_the_real_runs_fit_the_per_experiment_cap_and_all_six_the_overall_cap():
    budget = load_budget()
    caps = {name: parse_usd(cap) for name, (cap, _, _) in QUOTES.items()}
    assert all(c <= parse_usd(budget.per_experiment_cap) for c in caps.values())
    assert sum(caps.values()) == parse_usd("$59.25")
    assert sum(caps.values()) <= parse_usd(budget.overall_cap) - SPENT_AT_QUOTE


def test_the_runner_runs_host_groups_one_after_another():
    # The three TensorRT-LLM host groups run in sequence (runner._run_group, each host torn
    # down before the next group): with the teardown wait, never two hosts at once.
    exp = load_experiment(RUNS["trtllm"][1])
    cells = expand(exp, REGISTRY)
    assert [c.variant for c in cells] == ["vllm-bf16", "trtllm-bf16", "trtllm-fp8"]
    assert len({c.host_key for c in cells}) == 3


def test_the_trtllm_runs_vllm_row_is_95cde129s_bf16_cell():
    cells = {c.variant: c for c in expand(load_experiment(RUNS["trtllm"][1]), REGISTRY)}
    assert cells["vllm-bf16"].config_hash == VLLM_BF16_HASH


def test_trtllm_cold_starts_are_priced_at_the_ec2_pull_rate():
    plan = _plan(load_experiment(RUNS["trtllm"][1]))
    vllm, trt, fp8 = (h.steps[0].seconds for h in plan.hosts)
    pull, init = AWS_ENGINE_START_EXTRA_S["trtllm"]
    assert trt - vllm == pytest.approx(pull + init)  # the same BF16 checkpoint
    # 17.3 GB at the ~37 MB/s EC2 hosts pulled SGLang's 15.2 GB in 416 s (fccf9ed3).
    assert AWS_TIMING.image_pull_s + pull >= 17.3e9 / (15.2e9 / 416)
    assert fp8 < trt  # the FP8 checkpoint is half the BF16 one


def test_every_host_fits_its_checkpoints_and_datasets_on_the_instance_store():
    for paths in RUNS.values():
        for path in paths[:2]:
            exp = load_experiment(path)
            cells = expand(exp, REGISTRY)
            profiles = [w.resolve() for w in exp.workloads]
            for key in {c.host_key for c in cells}:
                group = [c for c in cells if c.host_key == key]
                assert _aws_disk_problems(key, group, PRICES, profiles) == []
    # 300 GB of checkpoints on one 250 GB g6e.xlarge store is refused.
    exp = load_experiment(RUNS["specdec"][1])
    (cell, *_) = expand(exp, REGISTRY)
    big = cell.model_copy(
        update={
            "spec": cell.spec.model_copy(
                update={"hf": cell.spec.hf.model_copy(update={"size_bytes": 300 * 10**9})}
            )
        }
    )
    (problem,) = _aws_disk_problems(cell.host_key, [big], PRICES, [])
    assert "250 GB instance store" in problem


def test_every_profiled_aws_cell_mounts_its_trace_dir_into_the_engine_container():
    for path in RUNS["profile"][:2]:
        exp = load_experiment(path)
        assert exp.profile is not None
        for cell in expand(exp, REGISTRY):
            assert profiler_trace_dir(cell.launch) == exp.profile.trace_dir
            argv = docker_run_argv(cell.launch, weights_dir="/w", container_name="c")
            mount = f"{exp.profile.trace_dir}:{exp.profile.trace_dir}"
            assert argv[argv.index(mount) - 1] == "--volume"


# --- end to end on the aws_ec2 provider ---------------------------------------------------


def _edits(n: int = 40) -> bytes:
    """CommitPackFT-format JSON lines: a small Python file, its commit message, the edit."""
    rows = []
    for i in range(n):
        old = "".join(f"def f{j}(x):\n    return x + {j}\n\n" for j in range(3 + i % 4))
        new = old.replace("return x + 1", "return x + 100") + f"# edit {i}\n"
        rows.append(
            {
                "old_file": f"pkg/m{i}.py",
                "message": f"Change f1 in module {i}",
                "old_contents": old,
                "new_contents": new,
                "license": "mit",
            }
        )
    return "".join(json.dumps(r) + "\n" for r in rows).encode()


EDITS = _edits()
EDIT_URL = load_profile("code-edit-commitpackft").download.url  # type: ignore[union-attr]


def _pin(body: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(body).hexdigest(), "size_bytes": len(body)}


def at_test_scale(path: Path, tmp: Path, body: bytes) -> Experiment:
    """The shipped spec's model, provider, variants, workload profiles and labels, profile
    windows and quality baseline and replicates; tiny pinned datasets, short requests, one
    load point each, a tiny suite and short profile windows."""
    doc = _doc(path)
    load = {
        "mode": "closed_loop",
        "values": [2],
        "num_requests": 6,
        "warmup_requests": 1,
        "scrape_interval_s": 0.05,
    }
    workloads = []
    for w in doc["workloads"]:
        if w["profile"] == "chat-sharegpt":
            over = {"download": _pin(body), "max_output_len": 8, "min_output_len": 1}
        elif w["profile"] == "code-edit-commitpackft":
            over = {"download": _pin(EDITS), "max_output_len": 256, "min_output_len": 1}
        else:
            over = {"input_len": 32, "output_len": 8}
        workloads.append(
            {
                **{k: v for k, v in w.items() if k == "label"},
                "profile": w["profile"],
                "overrides": over,
                "load": load,
            }
        )
    doc.update(
        workloads=workloads,
        repetitions=1,
        allow_single_run=True,
        slo={"max_error_rate": 0.01},
        budget={"max_spend": "$45", "ttl_minutes": 240, "accrual_interval_s": 0.2},
    )
    if doc.get("profile"):
        doc["profile"] = {
            **{k: doc["profile"][k] for k in ("windows", "trace_dir", "max_trace_mb")},
            "settle_s": 0.3,
            "capture_s": 0.4,
            "tail_s": 0.2,
            "max_iterations": 12,
            "drain_timeout_s": 2,
        }
    if doc.get("quality"):
        q = doc["quality"]
        also = sorted({v["model"] for v in doc["variants"] if v.get("model")})
        suite = tmp / "suite.yaml"
        suite.write_text(yaml.safe_dump(_suite(also)))
        doc["quality"] = {
            "suite": str(suite),
            "baseline_variant": q["baseline_variant"],
            "replicates": q["replicates"],
        }
    doc.pop("smoke", None)
    return Experiment.model_validate(doc)


def _events(outcome: Any, ctx: Any) -> list[dict[str, Any]]:
    path = ctx.out_dir / str(outcome.experiment_id) / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _runs(ctx: Any) -> list[BenchRun]:
    with session_scope(ctx.db_url) as s:
        return list(s.scalars(select(BenchRun)))


# Profiling ---------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=RUNS["profile"][:2], ids=lambda p: p.stem)
def profile_run(request, tmp_path_factory):
    return run_on_host_sim(
        request.param, tmp_path_factory.mktemp(request.param.stem), build=at_test_scale
    )


def test_profile_windows_collect_traces_through_the_containers_bind_mount(profile_run):
    outcome, sim, ctx, cells, exp, instances, reaped = profile_run
    assert outcome.status.value == "completed", outcome.reason
    (inst,) = instances
    assert inst["State"]["Name"] == "terminated" and reaped == []
    for script in sim.scripts_of("start_engine"):
        cmd = array(script, "ENGINE_CMD")
        assert "/var/lib/loom/profiles:/var/lib/loom/profiles" in cmd
    collects = sim.scripts_of("collect_profile")
    n_windows = len(cells) * len(exp.workloads)  # one load value per workload here
    assert len(collects) == n_windows
    assert {var(s, "TRACE_DIR") for s in collects} == {"/var/lib/loom/profiles"}
    assert {var(s, "MAX_BYTES") for s in collects} == {str(256 * 1024 * 1024)}
    events = _events(outcome, ctx)
    assert not [e for e in events if e["kind"] == "profile_failed"]
    prof = [e for e in events if e["kind"] == "profile"]
    assert sorted((e["cell"], e["workload"]) for e in prof) == sorted(
        (c.key, w.name) for c in cells for w in exp.workloads
    )
    windows = sorted((ctx.out_dir / str(outcome.experiment_id) / "profiles").iterdir())
    assert len(windows) == n_windows
    for w in windows:
        doc = json.loads((w / "window.json").read_text())
        assert doc["capture"]["start_status"] == 200 and doc["capture"]["stop_status"] == 200
        with tarfile.open(w / "trace.tar.gz") as t:
            assert any(n.endswith(".pt.trace.json.gz") for n in t.getnames())
    # Windows are never load runs: one measured run per cell and workload.
    assert len(_runs(ctx)) == len(cells) * len(exp.workloads)


def test_without_the_bind_mount_the_profile_windows_fail_visibly(tmp_path, monkeypatch):
    # The engine's traces would stay inside its container: every window is recorded as
    # failed (the profiler is unreachable from the host), the measured runs still complete.
    import loom_bench.engines as engines

    monkeypatch.setattr(engines, "profiler_trace_dir", lambda launch: None)
    outcome, _, ctx, cells, exp, _, _ = run_on_host_sim(
        RUNS["profile"][0], tmp_path, build=at_test_scale
    )
    events = _events(outcome, ctx)
    failed = [e for e in events if e["kind"] == "profile_failed"]
    assert len(failed) == len(cells) * len(exp.workloads)
    assert not [e for e in events if e["kind"] == "profile"]
    assert len(_runs(ctx)) == len(cells) * len(exp.workloads)


# Speculative decoding ------------------------------------------------------------------------


class SpeculatingHostSim(HostSim):
    """The host's mock engine drafts `num_speculative_tokens` per step when the rendered
    launch has a speculative config, accepting fewer drafts for n-gram than EAGLE3."""

    def start(self, instance: str, script: str) -> tuple[int, str, str]:
        cmd = array(script, "ENGINE_CMD")
        self.mock = {}
        if "--speculative-config" in cmd:
            cfg = json.loads(cmd[cmd.index("--speculative-config") + 1])
            ngram = cfg["method"] == "ngram"
            self.mock = {
                "spec_tokens": cfg["num_speculative_tokens"],
                "spec_acceptance": 0.4 if ngram else 0.7,
                "spec_draft_rate": 0.5 if ngram else 1.0,
            }
        return super().start(instance, script)


@pytest.fixture(scope="module")
def specdec_run(tmp_path_factory):
    # At test scale the smoke and the run are the same experiment (equal provider,
    # variants, workload profiles and quality section), so the smoke runs here.
    path = RUNS["specdec"][0]
    return run_on_host_sim(
        path,
        tmp_path_factory.mktemp(path.stem),
        build=at_test_scale,
        sim_cls=SpeculatingHostSim,
        served={EDIT_URL: EDITS},
    )


def test_three_rows_on_one_host_with_the_eagle3_head_fetched_on_its_restart(specdec_run):
    outcome, sim, _, cells, _, instances, reaped = specdec_run
    assert outcome.status.value == "completed", outcome.reason
    (inst,) = instances
    assert inst["State"]["Name"] == "terminated" and reaped == []
    assert [c.variant for c in cells] == ["bf16", "bf16-ngram", "bf16-eagle3"]
    cold, ngram, eagle = sim.scripts_of("start_engine")
    bf16 = (cells[0].spec.hf.repo, cells[0].spec.hf.revision)
    assert pairs(cold, "FETCH_WEIGHTS") == [bf16]
    assert var(ngram, "WARM") == "1" and pairs(ngram, "FETCH_WEIGHTS") == []
    assert var(eagle, "WARM") == "1" and pairs(eagle, "FETCH_WEIGHTS") == [EAGLE3]
    assert pairs(eagle, "CACHED_WEIGHTS") == [bf16]
    (downloads,) = sim.downloads.values()
    assert downloads == [bf16, EAGLE3]
    # vLLM loads the head offline from the mounted cache, by repo id and revision.
    cmd = array(eagle, "ENGINE_CMD")
    cfg = json.loads(cmd[cmd.index("--speculative-config") + 1])
    assert (cfg["model"], cfg["revision"]) == EAGLE3
    assert "HF_HUB_OFFLINE=1" in cmd


def test_the_commitpackft_file_is_fetched_once_and_read_through_the_mount(specdec_run):
    _, sim, _, _, _, _, _ = specdec_run
    (fetches,) = sim.dataset_fetches.values()
    assert sorted(fetches) == sorted(
        [load_profile("chat-sharegpt").download.url, EDIT_URL]  # type: ignore[union-attr]
    )
    edits = [j for j in sim.jobs if isinstance(j, LoadJob) and j.workload["kind"] == "edit_dataset"]
    assert len(edits) == 3  # one per row
    sha = hashlib.sha256(EDITS).hexdigest()
    assert {j.workload["path"] for j in edits} == {f"/data/{sha}/data.jsonl"}


def test_spec_counters_land_in_every_speculating_run_and_nowhere_else(specdec_run):
    _, _, ctx, cells, _, _, _ = specdec_run
    by_hash = {c.config_hash: c.variant for c in cells}
    runs = _runs(ctx)
    assert runs and {r.status for r in runs} == {"completed"}, [
        (r.cell_key, r.workload, r.summary) for r in runs if r.status != "completed"
    ]
    for run in runs:
        server = run.summary["server"]
        variant = by_hash[run.config_hash]
        assert "spec_decode" not in server["missing"], variant
        if variant == "bf16":
            assert server["spec_drafts"] is None
        else:
            assert server["spec_drafts"] > 0, variant


def test_only_the_eagle3_rows_store_output_text(specdec_run):
    _, sim, ctx, cells, _, _, _ = specdec_run
    by_hash = {c.config_hash: c.variant for c in cells}
    loads = [j for j in sim.jobs if isinstance(j, LoadJob)]
    assert {j.keep_output for j in loads} == {True, False}
    stored: dict[str, list[Any]] = {}
    for run in _runs(ctx):
        recs = [r for r in read_requests(str(run.requests_uri)) if r.ok and not r.warmup]
        stored.setdefault(by_hash[run.config_hash], []).extend(recs)
    assert stored["bf16-eagle3"] and all(r.output_text for r in stored["bf16-eagle3"])
    assert not any(r.output_text for v in ("bf16", "bf16-ngram") for r in stored[v])


def test_both_speculative_rows_are_gated_against_bf16s_in_run_reference(specdec_run):
    outcome, sim, _, (bf16, ngram, eagle), _, _, _ = specdec_run
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert len(evals) == 3 * 3
    first = [j for j in evals if j.divergence is not None]
    assert [j.divergence for j in first] == ["capture_and_floor", "score", "score"]
    for job in first[1:]:
        assert job.reference is not None and job.reference.config_hash == bf16.config_hash
    assert {(g.cell, g.baseline) for g in outcome.gates} == {
        (ngram.key, bf16.key),
        (eagle.key, bf16.key),
    }


# TensorRT-LLM ------------------------------------------------------------------------------


class TrtllmHostSim(HostSim):
    """A TensorRT-LLM engine answers echo+logprobs scoring only through Loom's wrapper."""

    def start(self, instance: str, script: str) -> tuple[int, str, str]:
        cmd = array(script, "ENGINE_CMD")
        stock = "trtllm-launch" in cmd and TRTLLM_SERVE_SOURCE not in cmd
        self.mock = {"completions_logprobs": not stock}
        return super().start(instance, script)


class OneHostQuota:
    """EC2 held to the account's 4-vCPU G/VT quota (one g6e.xlarge): RunInstances fails
    with VcpuLimitExceeded while any instance is not yet terminated, and a terminated
    instance reports shutting-down for its first two describes, as a real one does."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.shutting: dict[str, int] = {}
        self.launched: list[str] = []
        self.refused = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def terminate_instances(self, **kw: Any) -> Any:
        for iid in kw["InstanceIds"]:
            self.shutting.setdefault(iid, 2)
        return self._inner.terminate_instances(**kw)

    def describe_instances(self, **kw: Any) -> Any:
        resp = self._inner.describe_instances(**kw)
        for r in resp["Reservations"]:
            for inst in r["Instances"]:
                left = self.shutting.get(inst["InstanceId"], 0)
                if left and inst["State"]["Name"] == "terminated":
                    inst["State"] = {"Code": 32, "Name": "shutting-down"}
                    self.shutting[inst["InstanceId"]] = left - 1
        return resp

    def run_instances(self, **kw: Any) -> Any:
        resp = self._inner.describe_instances()
        live = [
            i["InstanceId"]
            for r in resp["Reservations"]
            for i in r["Instances"]
            if i["State"]["Name"] != "terminated" or self.shutting.get(i["InstanceId"], 0)
        ]
        if live:
            self.refused += 1
            raise client_error("VcpuLimitExceeded")
        out = self._inner.run_instances(**kw)
        self.launched.append(out["Instances"][0]["InstanceId"])
        return out


@pytest.fixture(scope="module", params=RUNS["trtllm"][:2], ids=lambda p: p.stem)
def trtllm_run(request, tmp_path_factory):
    quota: dict[str, OneHostQuota] = {}

    def wrap(ec2: Any) -> OneHostQuota:
        quota["ec2"] = OneHostQuota(ec2)
        return quota["ec2"]

    out = run_on_host_sim(
        request.param,
        tmp_path_factory.mktemp(request.param.stem),
        build=at_test_scale,
        sim_cls=TrtllmHostSim,
        wrap_ec2=wrap,
    )
    return (*out, quota["ec2"])


def test_three_hosts_in_a_row_each_launched_after_the_last_terminated(trtllm_run):
    outcome, sim, _, cells, _, instances, reaped, quota = trtllm_run
    assert outcome.status.value == "completed", outcome.reason
    assert quota.refused == 0 and len(quota.launched) == 3
    assert {i["InstanceId"] for i in instances} == set(quota.launched)
    assert all(i["State"]["Name"] == "terminated" for i in instances) and reaped == []
    # Each host serves one cell: its instance runs one engine start, in variant order.
    starts = [(i, s) for i, k, s in sim.launched if k == "start_engine"]
    assert [i for i, _ in starts] == quota.launched
    assert [var(s, "SERVED_MODEL") for _, s in starts] == [c.spec.id for c in cells]
    assert all(var(s, "WARM") == "0" for _, s in starts)
    # The chat dataset is fetched once per host.
    assert sorted(sim.dataset_fetches) == sorted(quota.launched)


def test_each_trtllm_host_runs_the_wrapper_with_nvidias_ulimits(trtllm_run):
    _, sim, _, cells, _, _, _, _ = trtllm_run
    vllm_s, bf16_s, fp8_s = sim.scripts_of("start_engine")
    vllm_cmd = array(vllm_s, "ENGINE_CMD")
    assert vllm_cmd[vllm_cmd.index("--entrypoint") + 1] == "vllm" and "--ulimit" not in vllm_cmd
    trt_image = cells[1].launch.image
    for script, cell in ((bf16_s, cells[1]), (fp8_s, cells[2])):
        cmd = array(script, "ENGINE_CMD")
        at = cmd.index(trt_image)
        assert cmd[at - 2 : at] == ["--entrypoint", "bash"]
        assert cmd[at + 1 : at + 5] == list(ENTRYPOINTS["trtllm"][1:])  # launcher + wrapper
        assert cmd[at + 5 :] == cell.launch.args
        assert "memlock=-1" in cmd[:at] and "stack=67108864" in cmd[:at]
        assert var(script, "IMAGE") == trt_image
    rev = cells[1].spec.hf.revision
    bf16_cmd, fp8_cmd = array(bf16_s, "ENGINE_CMD"), array(fp8_s, "ENGINE_CMD")
    assert f"/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/{rev}" in bf16_cmd
    assert pairs(fp8_s, "FETCH_WEIGHTS") == [FP8_MODELOPT]
    assert json.loads(fp8_cmd[fp8_cmd.index(trt_image) + 5])["kv_cache_config"] == {"dtype": "fp8"}


def test_trtllm_load_jobs_scrape_its_prometheus_path(trtllm_run):
    _, sim, _, _, _, _, _, _ = trtllm_run
    urls = {j.engine: j.metrics_url for j in sim.jobs if isinstance(j, LoadJob)}
    assert urls["trtllm"] == "http://127.0.0.1:8000/prometheus/metrics"
    assert urls["vllm"] == "http://127.0.0.1:8000/metrics"


def test_both_trtllm_cells_are_gated_on_vllms_reference_across_hosts(trtllm_run):
    outcome, sim, ctx, cells, _, _, _, _ = trtllm_run
    events = _events(outcome, ctx)
    assert {e["cell"] for e in events if e["kind"] == "quality"} == {c.key for c in cells}
    assert not [e for e in events if e["kind"] in ("quality_failed", "divergence_failed")]
    scored = [j for j in sim.jobs if isinstance(j, EvalJob) and j.divergence == "score"]
    assert len(scored) == 2
    assert {j.reference.config_hash for j in scored if j.reference} == {cells[0].config_hash}
    gates = {g.cell: g.baseline for g in outcome.gates}
    assert gates == {"trtllm-bf16": "vllm-bf16", "trtllm-fp8": "vllm-bf16"}
    for e in (e for e in events if e["kind"] == "gate"):
        assert "KL" in json.dumps(e)


def test_without_the_teardown_wait_the_second_host_hits_the_vcpu_quota(tmp_path, monkeypatch):
    # What the wait prevents: the next host group launched while the last instance is
    # still shutting down is refused (docs/runbook.md, section 7), and the run fails.
    import loom_bench.providers.aws_ec2 as aws_ec2

    monkeypatch.setattr(aws_ec2, "TERMINATE_WAIT_S", -1)
    quota: dict[str, OneHostQuota] = {}

    def wrap(ec2: Any) -> OneHostQuota:
        quota["ec2"] = OneHostQuota(ec2)
        return quota["ec2"]

    outcome, *_ = run_on_host_sim(
        RUNS["trtllm"][0], tmp_path, build=at_test_scale, sim_cls=TrtllmHostSim, wrap_ec2=wrap
    )
    assert outcome.status.value == "failed"
    assert "VcpuLimitExceeded" in str(outcome.reason)
    assert quota["ec2"].refused == 1 and len(quota["ec2"].launched) == 1


def test_trtllm_start_scripts_stay_well_inside_ssms_document_size():
    # The launcher sends the wrapper's Python source in ENGINE_CMD (~21 KB rendered). SSM
    # documents no limit on a command's parameters; its nearest one, a document's size, is
    # 64 KB (docs.aws.amazon.com/general/latest/gr/ssm.html).
    from loom_bench.providers.aws_ec2 import AwsEc2Provider, AwsSettings
    from loom_bench.providers.aws_ssm import ssm_commands
    from loom_bench.providers.weights import WeightsPlan

    settings = AwsSettings(
        bucket="bkt",
        instance_profile_name="p",
        security_group_id="sg-0123",
        subnet_ids=["subnet-0123"],
        hf_token_secret_name="loom/hf-token",
        owner="arul",
    )
    provider = AwsEc2Provider(settings, prices=PRICES, ec2=object(), ssm=object(), s3=object())
    for cell in expand(load_experiment(RUNS["trtllm"][1]), REGISTRY):
        ckpt = ((cell.launch.model_repo, cell.launch.model_revision),)
        script = provider.engine_script(
            cell.launch, warm=False, weights=WeightsPlan(fetch=ckpt, cached=())
        )
        size = len(json.dumps(ssm_commands(script)).encode())
        assert size < 32 * 1024, (cell.key, size)
