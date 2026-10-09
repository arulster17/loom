"""One pod serving two checkpoints, end to end on the runpod provider: the shipped specs
whose cells share a pod but not a checkpoint (runpod-smoke-h100 and runpod-smoke-fp8 by an
`hf` override, the 70B H100 run by `Variant.model`), with their own provider, variants and
model, at test scale (one tiny workload, a small suite, one repetition).

runpod-smoke-h100 (4e50b5a5, $1.18) failed at its second cell: warm restarts downloaded no
weights, so the FP8 cell's offline engine found RedHatAI/Qwen3-8B-FP8-dynamic "not cached".
The earlier e2e tests passed because neither the mock provider nor this file's pod
simulator had a weight cache; PodSim now holds every pod to start_engine.sh's offline rules,
and these runs would fail as the smoke did without the fix."""

import asyncio
from pathlib import Path
from typing import Any

import boto3
import pytest
import yaml
from moto import mock_aws
from sqlalchemy import select

from loom_bench.experiment import EXPERIMENTS_DIR, Experiment, expand
from loom_bench.jobs import EvalJob, LoadJob
from loom_bench.providers import runpod_api
from loom_bench.providers.runpod import RunpodProvider, RunpodSettings
from loom_bench.runner import RunnerContext, run_experiment
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import BenchColdStart, BenchRun

from .fakes import FakeRunpod, script_pairs, script_var
from .test_runpod_e2e import BUCKET, PodSim, s3_key, write_yaml

pytestmark = [pytest.mark.timeout(240), pytest.mark.xdist_group("runpod-e2e")]

SPECS = [
    EXPERIMENTS_DIR / "runpod-smoke-h100.yaml",
    EXPERIMENTS_DIR / "llama-3.3-70b-h100-tp2-runpod.yaml",
    EXPERIMENTS_DIR / "runpod-smoke-fp8.yaml",
]


def suite_for(model: str, also: list[str]) -> dict[str, Any]:
    return {
        "suite": "two-checkpoints",
        "model": model,
        "also_models": also,
        "seed": 1234,
        "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 500},
        "tasks": [
            {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
            {"name": "json_schema", "kind": "json_schema"},
        ],
        "divergence": {
            "prompts": 8,
            "top_k": 5,
            "max_new_tokens": 8,
            "noise_multiple": 2.0,
            "ceiling_top1": 0.6,
        },
    }


def at_test_scale(path: Path, tmp: Path) -> Experiment:
    """The shipped spec's model, provider and variants; a tiny workload and suite."""
    doc = yaml.safe_load(path.read_text())
    models = {v.get("model") for v in doc["variants"]} - {None, doc["model"]}
    suite = write_yaml(tmp / "suite.yaml", suite_for(doc["model"], sorted(models)))
    doc.update(
        workloads=[
            {
                "profile": "fixed-128-128",
                "overrides": {"input_len": 32, "output_len": 8},
                "load": {
                    "mode": "closed_loop",
                    "values": [2],
                    "num_requests": 6,
                    "warmup_requests": 1,
                    "scrape_interval_s": 0.05,
                },
            }
        ],
        repetitions=1,
        allow_single_run=True,
        slo={"max_error_rate": 0.01},
        quality={"suite": str(suite), "baseline_variant": doc["quality"]["baseline_variant"]},
        budget={"max_spend": "$45", "ttl_minutes": 240, "accrual_interval_s": 0.2},
    )
    doc.pop("smoke", None)
    return Experiment.model_validate(doc)


class RecordingPodSim(PodSim):
    """PodSim that keeps each job it ran, as staged."""

    def __init__(self, s3: Any) -> None:
        super().__init__(s3)
        self.jobs: list[LoadJob | EvalJob] = []

    async def job(self, target: Any, script: str) -> Any:
        body = self.s3.get_object(Bucket=BUCKET, Key=s3_key(script_var(script, "JOB_URL")))
        raw = body["Body"].read()
        is_eval = "BENCH_CMD=(quality job)" in script
        self.jobs.append(
            EvalJob.model_validate_json(raw) if is_eval else LoadJob.model_validate_json(raw)
        )
        return await super().job(target, script)


@pytest.fixture(scope="module", params=SPECS, ids=lambda p: p.stem)
def pod_run(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp(request.param.stem)
    mp = pytest.MonkeyPatch()
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        mp.setenv(k, v)
    mp.delenv(runpod_api.KEY_ENV, raising=False)
    with mock_aws():
        from loom_bench.budget import load_budget
        from loom_bench.prices import load_prices
        from loom_bench.registry import load_registry

        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        db = f"sqlite:///{tmp / 'loom.db'}"
        upgrade(db)
        ctx = RunnerContext(
            db_url=db,
            out_dir=tmp / "results",
            registry=load_registry(),
            prices=load_prices(),
            budget=load_budget(),
        )
        exp = at_test_scale(request.param, tmp)
        cells = expand(exp, ctx.registry)
        wheel = tmp / "loom_bench-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        fake = FakeRunpod(gets_before_ssh=1)
        sim = RecordingPodSim(s3)
        ctx.provider = RunpodProvider(
            RunpodSettings(bucket=BUCKET, owner="arul", poll_interval_s=0.01),
            spec=exp.provider,  # type: ignore[arg-type]
            api=fake.api(),
            prices=ctx.prices,
            s3=s3,
            ssh=sim,
            requirements=lambda extra: f"pkg-{extra or 'base'}==1.0 --hash=sha256:{'0' * 64}\n",
            wheel_path=wheel,
            ssh_dir=tmp / "ssh",
        )
        try:
            outcome = asyncio.run(run_experiment(exp, ctx))
        finally:
            sim.close()
            mp.undo()
    return outcome, sim, ctx, cells


def test_both_cells_run_on_one_pod(pod_run):
    outcome, sim, ctx, cells = pod_run
    assert outcome.status.value == "completed", outcome.reason
    assert len(cells) == 2 and cells[0].host_key == cells[1].host_key
    assert len(sim.downloads) == 1  # one pod
    with session_scope(ctx.db_url) as s:
        kinds = sorted(c.kind for c in s.scalars(select(BenchColdStart)))
        runs = list(s.scalars(select(BenchRun)))
    assert kinds == ["cold", "warm"]
    assert {r.status for r in runs} == {"completed"} and len(runs) == 2


def test_each_checkpoint_is_downloaded_once_at_its_pin(pod_run):
    _, sim, _, (base, cand) = pod_run
    (downloads,) = sim.downloads.values()
    bf16 = (base.spec.hf.repo, base.spec.hf.revision)
    fp8 = (cand.spec.hf.repo, cand.spec.hf.revision)
    assert bf16 != fp8
    assert downloads == [bf16, fp8]
    cold, warm = sim.scripts("start_engine")
    assert script_pairs(cold, "FETCH_WEIGHTS") == [bf16]
    assert script_var(warm, "WARM") == "1"
    assert script_pairs(warm, "FETCH_WEIGHTS") == [fp8]
    assert script_pairs(warm, "CACHED_WEIGHTS") == []


def test_the_fp8_cell_serves_and_is_measured_as_its_own_checkpoint(pod_run):
    _, sim, _, (base, cand) = pod_run
    _, warm = sim.scripts("start_engine")
    assert script_var(warm, "SERVED_MODEL") == cand.spec.id
    for job in sim.jobs:
        assert job.served_model in (base.spec.id, cand.spec.id)
    cand_jobs = [j for j in sim.jobs if j.served_model == cand.spec.id]
    if cand.spec.id == base.spec.id:  # an hf override keeps the registry id
        cand_jobs = sim.jobs[len(sim.jobs) // 2 :]
    assert cand_jobs
    for job in cand_jobs:  # the tokenizer is the FP8 checkpoint's, read from the pod
        assert job.tokenizer is not None
        assert (job.tokenizer.repo, job.tokenizer.revision) == (
            cand.spec.hf.repo,
            cand.spec.hf.revision,
        )


def test_the_fp8_cell_is_scored_on_the_bf16_reference_and_gated_against_it(pod_run):
    outcome, sim, _, (base, cand) = pod_run
    evals = [j for j in sim.jobs if isinstance(j, EvalJob)]
    assert [j.divergence for j in evals] == ["capture_and_floor", "score"]
    assert evals[1].reference is not None
    assert evals[1].reference.config_hash == base.config_hash
    (gate,) = outcome.gates
    assert (gate.cell, gate.baseline) == (cand.key, base.key)
    assert gate.decision in ("pass", "review")


def test_runs_record_each_cells_checkpoint_and_precision(pod_run):
    _, _, ctx, (base, cand) = pod_run
    with session_scope(ctx.db_url) as s:
        runs = list(s.scalars(select(BenchRun)))
    by_hash = {r.config_hash: r.provenance["model"] for r in runs}
    assert by_hash[base.config_hash] == {
        "repo": base.spec.hf.repo,
        "revision": base.spec.hf.revision,
        "quantization": "none",
    }
    assert by_hash[cand.config_hash] == {
        "repo": cand.spec.hf.repo,
        "revision": cand.spec.hf.revision,
        "quantization": "fp8",
    }
