"""`bench reproduce` of a stored RunPod run, end to end on the runpod provider against
FakeRunpod and the pod simulator.

The source is a real run: the Qwen3-8B config sweep 7a8237d0 (Secure 1x L40S, vLLM
v0.30.0), cell bf16, chat-sharegpt at 2.449 req/s (its goodput load), repetition 0
(aa7af8ed). The fixture holds that experiment's stored spec and the stored rows of the
point's three repetitions, as the results DB has them. The reproduction is run as the
paid A1 check runs it (docs/phase0-acceptance.md): by run id prefix, with `--spec` the
stored spec with only its budget lowered ($2.50, a 60-minute pod TTL).

What this proves before paying: today's code rebuilds that cell from the provenance
record alone and hashes it to the stored config hash; the pod it asks RunPod for, the
engine command, the dataset pin and the load job are the original's; the new run is
recorded under the same config hash and compared with the original's repetitions on it;
the pod is terminated. The pod simulator's mock engine is not an L40S, so the verdict
itself is not checked here (the metrics come out far apart): only that the comparison
matched by config hash."""

import asyncio
import copy
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pytest
from moto import mock_aws

from loom_bench.budget import load_budget
from loom_bench.experiment import Experiment
from loom_bench.jobexec import execute_load_job
from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.prices import load_prices
from loom_bench.providers import runpod_api
from loom_bench.providers.runpod import RunpodProvider, RunpodSettings
from loom_bench.registry import load_registry
from loom_bench.runner import RunnerContext, reproduce
from loom_bench.store import repo
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import BenchExperiment
from loom_bench.workloads.profiles import load_profile

from .fakes import FakeRunpod, ok, script_array, script_var
from .test_runpod_e2e import BUCKET, PodSim, s3_key

pytestmark = [pytest.mark.timeout(240), pytest.mark.xdist_group("runpod-e2e")]

FIXTURE = Path(__file__).parent / "fixtures" / "reproduce_7a8237d0"
SOURCE_RUN = "aa7af8ed"
CONFIG_HASH = "7a203a3ffaccb1820fd23e6da9fab1d00c4cb51c427ce1572c56b08a8036107a"
# The A1 run's spec override: the stored spec with only the budget lowered.
BUDGET = {"max_spend": 2_500_000, "ttl_minutes": 60.0, "accrual_interval_s": 15.0}


def _sharegpt(n: int = 40) -> bytes:
    convs = [
        {
            "id": f"c{i}",
            "conversations": [
                {"from": "human", "value": f"question {i} " + "word " * (5 + i % 7)},
                {"from": "gpt", "value": "answer " * (6 + i % 5)},
            ],
        }
        for i in range(n)
    ]
    return json.dumps(convs).encode()


def reproduce_spec(stored: dict[str, Any]) -> dict[str, Any]:
    doc = copy.deepcopy(stored)
    doc["budget"] = dict(BUDGET)
    return doc


class ShortLoadPodSim(PodSim):
    """PodSim that keeps each staged load job as staged, then runs it for a few seconds
    on a tiny ShareGPT-format file at the pod's dataset path (the pinned 673 MB file is
    what run_job.sh downloads on a real pod)."""

    def __init__(self, s3: Any, dataset: Path) -> None:
        super().__init__(s3)
        self.dataset = dataset
        self.staged: list[LoadJob] = []

    async def job(self, target: Any, script: str) -> Any:
        raw = self.s3.get_object(Bucket=BUCKET, Key=s3_key(script_var(script, "JOB_URL")))
        staged = LoadJob.model_validate_json(raw["Body"].read())
        self.staged.append(staged)
        root = f"http://127.0.0.1:{self.servers[self._pod(target)].port}"
        short = staged.model_copy(
            update={
                "base_url": f"{root}/v1",
                "metrics_url": f"{root}/metrics",
                "tokenizer": TokenizerSpec(kind="simple"),
                "workload": {**staged.workload, "path": str(self.dataset)},
                "duration_s": 3.0,
                "warmup_s": 0.5,
                "drain_timeout_s": 10.0,
                "scrape_interval_s": 0.1,
            }
        )
        out = (await execute_load_job(short)).model_dump_json()
        self.s3.put_object(Bucket=BUCKET, Key=s3_key(script_var(script, "RESULT_URL")), Body=out)
        if script_var(script, "SAMPLE_GPU") == "1":
            key = s3_key(script_var(script, "GPU_CSV_URL"))
            self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"2026/10/06, 0, 50, 1, 2, 3\n")
        return ok("loom-sys job_isolation ok\nloom-job-done\n")


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)  # SQLite stores UTC, naive


def _seed_db(db: str, exp_doc: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    """The source experiment and its point's repetitions, as stored in the results DB."""
    exp_id = uuid.UUID(exp_doc["id"])
    with session_scope(db) as s:
        s.add(
            BenchExperiment(
                id=exp_id,
                name=exp_doc["name"],
                spec=exp_doc["spec"],
                spec_hash="0" * 64,
                git_sha=exp_doc["git_sha"],
                git_dirty=exp_doc["git_dirty"],
                status="completed",
                budget_micros=exp_doc["spec"]["budget"]["max_spend"],
                spent_micros=0,
                created_at=_utc(exp_doc["created_at"]),
            )
        )
        s.flush()
        for r in runs:
            repo.record_run(
                s,
                run_id=uuid.UUID(r["id"]),
                experiment_id=exp_id,
                config_hash=r["config_hash"],
                provenance=r["provenance"],
                status=r["status"],
                summary=r["summary"],
                cell_key=r["cell_key"],
                workload=r["workload"],
                load_mode=r["load_mode"],
                load_value=r["load_value"],
                repetition=r["repetition"],
                started_at=_utc(r["started_at"]),
                finished_at=_utc(r["finished_at"]),
            )


@pytest.fixture(scope="module")
def repro(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runpod-reproduce")
    exp_doc = json.loads((FIXTURE / "experiment.json").read_text())
    runs = json.loads((FIXTURE / "runs.json").read_text())
    spec_path = tmp / "spec.json"
    spec_path.write_text(json.dumps(reproduce_spec(exp_doc["spec"])))
    dataset = tmp / "sharegpt.json"
    dataset.write_bytes(_sharegpt())
    mp = pytest.MonkeyPatch()
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        mp.setenv(k, v)
    mp.delenv(runpod_api.KEY_ENV, raising=False)
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        db = f"sqlite:///{tmp / 'loom.db'}"
        upgrade(db)
        _seed_db(db, exp_doc, runs)
        ctx = RunnerContext(
            db_url=db,
            out_dir=tmp / "results",
            registry=load_registry(),
            prices=load_prices(),
            budget=load_budget(),
        )
        exp = Experiment.model_validate(reproduce_spec(exp_doc["spec"]))
        wheel = tmp / "loom_bench-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        fake = FakeRunpod(gets_before_ssh=1)
        sim = ShortLoadPodSim(s3, dataset)
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
            result = asyncio.run(reproduce(SOURCE_RUN, ctx, spec_path=spec_path))
        finally:
            sim.close()
            mp.undo()
    return result, fake, sim, ctx, runs


def test_the_fixture_is_the_stored_source_point():
    runs = json.loads((FIXTURE / "runs.json").read_text())
    assert [r["repetition"] for r in runs] == [0, 1, 2]
    assert {r["config_hash"] for r in runs} == {CONFIG_HASH}
    assert runs[0]["id"].startswith(SOURCE_RUN)
    assert {(r["cell_key"], r["workload"], r["status"]) for r in runs} == {
        ("bf16", "chat-sharegpt", "completed")
    }


def test_the_reproduction_completes_and_terminates_its_pod(repro):
    result, fake, _, _, _ = repro
    assert result.outcome.status.value == "completed", result.outcome.reason
    assert result.new_run_id is not None
    assert len(fake.bodies("POST", "/v1/pods")) == 1
    assert fake.pods == {}


def test_the_pod_is_the_originals(repro):
    _, fake, _, _, runs = repro
    (body,) = fake.bodies("POST", "/v1/pods")
    hw = runs[0]["provenance"]["config"]["hardware"]
    launch = runs[0]["provenance"]["config"]["launch"]
    assert body["cloudType"] == "SECURE"
    assert body["gpuTypeIds"] == [hw["gpu_type_id"]] == ["NVIDIA L40S"]
    assert body["gpuCount"] == hw["gpus"] == 1
    assert body["containerDiskInGb"] == hw["disk_gb"] == 80
    assert body["volumeInGb"] == 0
    assert body["allowedCudaVersions"] == hw["allowed_cuda_versions"] == ["13.0"]
    assert body["imageName"] == launch["image"]
    assert body["interruptible"] is False


def test_the_engine_command_is_the_stored_launch(repro):
    _, _, sim, _, runs = repro
    launch = runs[0]["provenance"]["config"]["launch"]
    (start,) = sim.scripts("start_engine")
    cmd = script_array(start, "ENGINE_CMD")
    assert script_var(start, "WARM") == "0"
    assert script_var(start, "SERVED_MODEL") == launch["served_model"]
    # Every stored engine argument, in order, but the bind address (the pod binds the
    # engine to loopback).
    host = launch["args"].index("--host")
    stored = launch["args"][:host] + launch["args"][host + 2 :]
    i = cmd.index("--host")
    assert cmd[cmd.index(stored[0]) : i] + cmd[i + 2 :] == stored


def test_the_load_job_is_the_original_points(repro):
    _, _, sim, _, runs = repro
    prov = runs[0]["provenance"]
    (job,) = sim.staged
    assert job.mode == "open_loop"
    assert job.load_value == prov["load"]["value"]
    assert job.seed == prov["load"]["seed"]
    assert (job.duration_s, job.warmup_s, job.drain_timeout_s) == (255.0, 45.0, 60.0)
    assert job.engine == "vllm" and job.served_model == "qwen3-8b"
    assert job.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    pin = load_profile("chat-sharegpt").download
    assert pin is not None
    assert job.workload["download"]["sha256"] == pin.sha256
    assert job.workload["path"].endswith(f"/{pin.sha256}/{pin.filename}")


def test_the_new_run_has_the_original_config_hash(repro):
    result, _, _, ctx, runs = repro
    with session_scope(ctx.db_url) as s:
        new = repo.get_run(s, result.new_run_id)
        assert new is not None
        assert new.config_hash == CONFIG_HASH
        assert new.provenance["config"] == runs[0]["provenance"]["config"]
        assert new.provenance["load"] == runs[0]["provenance"]["load"]
        assert new.provenance["repetition"] == 0
        assert new.provenance["workload"] == runs[0]["provenance"]["workload"]


def test_the_comparison_matches_the_original_point_by_config_hash(repro):
    result, _, _, _, runs = repro
    assert result.original_run_id == uuid.UUID(runs[0]["id"])
    cmp = result.comparison
    assert cmp is not None
    assert cmp.verdict != "nothing_matched"
    (point,) = cmp.points
    assert point.config_hash_match
    assert point.config_hash_a == point.config_hash_b == CONFIG_HASH
    assert point.load == pytest.approx(runs[0]["load_value"])


def test_the_only_warnings_are_about_todays_git_state(repro):
    # The workload still resolves to the stored profile hash (no "resolves differently"
    # warning) and the original came from a clean tree; today's sha and any uncommitted
    # change in the test's checkout are expected.
    result, *_ = repro
    for w in result.warnings:
        assert w.startswith(("git sha differs", "working tree is dirty")), w
