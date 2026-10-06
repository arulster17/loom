"""The runner on the runpod provider end to end: FakeRunpod for the API, moto S3, and a
pod simulator behind the SSH channel that serves each pod's engine with the mock backend
and executes the staged jobs for real."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import boto3
import pytest
import yaml
from moto import mock_aws
from sqlalchemy import select
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.experiment import Experiment
from loom_bench.jobexec import execute_load_job
from loom_bench.jobs import EvalJob, LoadJob, TokenizerSpec
from loom_bench.mock.config import MockConfig
from loom_bench.providers import runpod_api
from loom_bench.providers.mock import _start_server, _stop_server
from loom_bench.providers.runpod import RunpodProvider, RunpodSettings
from loom_bench.providers.runpod_ssh import SshTarget
from loom_bench.quality.runner import execute_eval_job
from loom_bench.runner import RunnerContext, reap, run_experiment
from loom_bench.store import repo
from loom_bench.store.db import session_scope, upgrade
from loom_bench.store.models import BenchColdStart, BenchResource, BenchRun, BenchSpend

from .fakes import FakePodExec, FakeRunpod, engine_stdout, ok, script_var

pytestmark = [pytest.mark.timeout(240), pytest.mark.xdist_group("runpod-e2e")]

BUCKET = "loom-bench-test"
VLLM = "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
SGLANG = "lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
SUITE = {
    "suite": "runpod-e2e",
    "model": "qwen3-8b",
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


@pytest.fixture
def db(tmp_path: Path) -> str:
    url = f"sqlite:///{tmp_path / 'loom.db'}"
    upgrade(url)
    return url


def write_yaml(path: Path, doc: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(doc))
    return path


def s3_key(url: str) -> str:
    return unquote(urlparse(url).path).lstrip("/").removeprefix(f"{BUCKET}/")


def _started_server(cfg: MockConfig) -> Any:
    srv = _start_server(cfg)
    deadline = time.monotonic() + 10
    while not srv.server.started:
        assert time.monotonic() < deadline, "mock server did not start"
        time.sleep(0.005)
    return srv


class PodSim(FakePodExec):
    """Each pod's engine is a mock server; staged jobs run against it, inputs read from
    and results written to the S3 keys the script's presigned URLs name. The pod's
    loopback address becomes the mock's, and the hf tokenizer (read from the pod's weight
    cache) becomes the simple one."""

    def __init__(self, s3: Any) -> None:
        super().__init__(
            {"start_engine": self.start, "stop_engine": self.stop, "run_job": self.job}
        )
        self.s3 = s3
        self.servers: dict[str, Any] = {}  # pod (by its known_hosts file) -> mock server

    def _pod(self, target: SshTarget) -> str:
        return target.known_hosts.name

    async def start(self, target: SshTarget, script: str) -> Any:
        served = script_var(script, "SERVED_MODEL")
        cfg = MockConfig(time_scale=0.01, models=[served], logprob_jitter=0.25)
        await self.stop(target, script)
        srv = await asyncio.to_thread(_started_server, cfg)
        self.servers[self._pod(target)] = srv
        return ok(engine_stdout())

    async def stop(self, target: SshTarget, script: str) -> Any:
        srv = self.servers.pop(self._pod(target), None)
        if srv is not None:
            await asyncio.to_thread(_stop_server, srv)
        return ok("loom-stopped\n")

    async def job(self, target: SshTarget, script: str) -> Any:
        root = f"http://127.0.0.1:{self.servers[self._pod(target)].port}"
        raw = self.s3.get_object(Bucket=BUCKET, Key=s3_key(script_var(script, "JOB_URL")))
        body = raw["Body"].read()
        simple = TokenizerSpec(kind="simple")
        if "BENCH_CMD=(quality job)" in script:
            eval_job = EvalJob.model_validate_json(body)
            eval_job = eval_job.model_copy(update={"base_url": f"{root}/v1", "tokenizer": simple})
            out = (await execute_eval_job(eval_job)).model_dump_json()
        else:
            job = LoadJob.model_validate_json(body).model_copy(
                update={
                    "base_url": f"{root}/v1",
                    "metrics_url": f"{root}/metrics",
                    "tokenizer": simple,
                }
            )
            out = (await execute_load_job(job)).model_dump_json()
            if script_var(script, "SAMPLE_GPU") == "1":
                key = s3_key(script_var(script, "GPU_CSV_URL"))
                self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"2026/10/06, 0, 50, 1, 2, 3\n")
        self.s3.put_object(Bucket=BUCKET, Key=s3_key(script_var(script, "RESULT_URL")), Body=out)
        return ok("loom-sys job_isolation ok\nloom-job-done\n")

    def close(self) -> None:
        for srv in self.servers.values():
            _stop_server(srv)
        self.servers.clear()


def runpod_experiment(suite: Path) -> Experiment:
    return Experiment.model_validate(
        {
            "name": "runpod-e2e",
            "description": "vLLM vs SGLang on RunPod, faked",
            "model": "qwen3-8b",
            "provider": {"kind": "runpod", "container_disk_gb": 80},
            "variants": [
                {"name": "vllm", "engine": {"name": "vllm", "version": "0.30.0", "image": VLLM}},
                {
                    "name": "sglang",
                    "engine": {"name": "sglang", "version": "0.5.21", "image": SGLANG},
                },
            ],
            "workloads": [
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
            "repetitions": 1,
            "allow_single_run": True,
            "slo": {"max_error_rate": 0.01},
            "quality": {"suite": str(suite), "baseline_variant": "vllm"},
            "budget": {"max_spend": "$5", "ttl_minutes": 60, "accrual_interval_s": 0.2},
        }
    )


@pytest.fixture(scope="module")
def runpod_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("runpod-e2e")
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
        exp = runpod_experiment(write_yaml(tmp / "suite.yaml", SUITE))
        wheel = tmp / "loom_bench-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        fake = FakeRunpod(gets_before_ssh=1)
        sim = PodSim(s3)
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
    return outcome, fake, sim, ctx


def test_two_engine_images_get_two_pods_each_cold_started_and_terminated(runpod_run):
    outcome, fake, _, ctx = runpod_run
    assert outcome.status.value == "completed", outcome.reason
    bodies = fake.bodies("POST", "/v1/pods")
    assert [b["imageName"] for b in bodies] == [VLLM, SGLANG]
    assert all(b["volumeInGb"] == 0 and b["interruptible"] is False for b in bodies)
    assert fake.pods == {}  # both terminated
    assert not any("/stop" in r.url.path for r in fake.requests)
    with session_scope(ctx.db_url) as s:
        starts = list(s.scalars(select(BenchColdStart)))
        resources = list(s.scalars(select(BenchResource)))
    assert sorted(c.kind for c in starts) == ["cold", "cold"]
    assert {r.provider for r in resources} == {"runpod"}
    assert {r.resource_type for r in resources} == {"runpod_pod"}
    assert len(resources) == 2 and all(r.terminated_at for r in resources)


def test_runs_carry_runpod_provenance_and_observed_prices(runpod_run):
    _, _, _, ctx = runpod_run
    with session_scope(ctx.db_url) as s:
        runs = list(s.scalars(select(BenchRun)))
        spend = list(s.scalars(select(BenchSpend)))
    assert len(runs) == 2 and all(r.status == "completed" for r in runs)
    for run in runs:
        prov = run.provenance
        assert prov["cloud"] == "runpod"
        assert prov["region"] == "secure"
        assert prov["hardware"]["instance_type"] == "l40s-x1"
        assert prov["hardware"]["gpu_type"] == "L40S"
        assert prov["hardware"]["gpu_count"] == 1
        assert prov["cuda_version"] == "13.0"
        assert prov["driver_version"] == "580.159.03"
        assert prov["price_basis"]["source"] == "observed_api"
        assert prov["price_basis"]["availability_zone"] == "EUR-IS-2"
        assert prov["hourly_micros"] == 1_100_959
    assert spend and sum(row.amount_micros for row in spend) > 0


def test_sglang_is_gated_against_vllm_across_pods(runpod_run):
    outcome, _, sim, _ = runpod_run
    (gate,) = outcome.gates
    assert gate.cell.startswith("sglang") or "sglang" in gate.cell
    assert gate.decision in ("pass", "review")
    evals = [s for s in sim.scripts("run_job") if "BENCH_CMD=(quality job)" in s]
    loads = [s for s in sim.scripts("run_job") if "BENCH_CMD=(job run)" in s]
    assert len(evals) == 2 and len(loads) == 2
    # Each pod ran its own jobs: the eval on the SGLang pod went to the second pod.
    pods = [t.known_hosts.name for t, name, _ in sim.launched if name == "start_engine"]
    assert len(set(pods)) == 2


def test_engine_started_events_record_job_isolation(runpod_run):
    outcome, _, _, _ = runpod_run
    started = [e for e in outcome.events if e["kind"] == "engine_started"]
    assert len(started) == 2
    for e in started:
        assert e["system"]["job_isolation"] == "ok"
        assert e["system"]["data_center"] == "EUR-IS-2"
        assert {"pod_created", "ssh_online", "first_token"} <= e["stages"].keys()


def test_no_presigned_url_reaches_the_runpod_api(runpod_run):
    _, fake, _, _ = runpod_run
    for r in fake.requests:
        text = r.content.decode(errors="replace")
        assert "Signature" not in text and "amazonaws" not in text


# --- reaping -----------------------------------------------------------------------


def test_reap_terminates_recorded_and_unrecorded_managed_pods(db):
    fake = FakeRunpod()
    now = datetime.now(UTC)
    past, future = now - timedelta(minutes=5), now + timedelta(hours=1)
    epoch = int(past.timestamp())
    recorded = fake.add_pod(
        name=f"loom-bench-e1-{epoch}-aaa111", env={"LOOM_MANAGED": "true", "LOOM_TTL": str(epoch)}
    )
    unrecorded = fake.add_pod(name=f"loom-bench-e2-{epoch}-bbb222", env={"LOOM_MANAGED": "true"})
    foreign = fake.add_pod(name=f"loom-bench-e3-{epoch}-ccc333", env={})
    alive = fake.add_pod(
        name=f"loom-bench-e4-{int(future.timestamp())}-ddd444", env={"LOOM_MANAGED": "true"}
    )
    with session_scope(db) as s:
        for rid, ttl in ((recorded["id"], past), ("gonepod0001", past)):
            repo.record_resource(
                s,
                provider="runpod",
                resource_type="runpod_pod",
                resource_id=rid,
                ttl_at=ttl,
            )

    dry = asyncio.run(reap(db, dry_run=True, runpod=fake.api(), runpod_prefix="loom-bench"))
    assert {r.action for r in dry} == {"would terminate"}
    assert len(fake.pods) == 4

    out = {r.resource_id: r for r in asyncio.run(reap(db, runpod=fake.api()))}
    assert out[recorded["id"]].action == "terminated"
    assert out["gonepod0001"].action == "gone"  # its own TTL watchdog ended it
    assert out[unrecorded["id"]].action == "terminated"
    assert out[unrecorded["id"]].ttl_at is None
    assert set(fake.pods) == {foreign["id"], alive["id"]}
    with session_scope(db) as s:
        rows = {r.resource_id: r for r in s.scalars(select(BenchResource))}
    assert rows[recorded["id"]].terminated_by == "reaper"
    assert rows["gonepod0001"].terminated_by == "self-ttl"


def test_recorded_pods_are_skipped_without_a_runpod_client(db):
    with session_scope(db) as s:
        repo.record_resource(
            s,
            provider="runpod",
            resource_type="runpod_pod",
            resource_id="podx",
            ttl_at=datetime.now(UTC) - timedelta(minutes=1),
        )
    (row,) = asyncio.run(reap(db))
    assert row.action == "skipped: runpod not configured"


def test_cli_reap_without_a_key_makes_no_runpod_calls(db, monkeypatch):
    def refuse(*a: Any, **k: Any) -> None:
        raise AssertionError("RunpodApi must not be built without a key")

    monkeypatch.setattr(runpod_api.RunpodApi, "__init__", refuse)
    result = CliRunner().invoke(app, ["reap", "--db", db])
    assert result.exit_code == 0, result.output
    assert "No RunPod API key" in result.output


def test_events_file_has_the_isolation_marker(runpod_run):
    outcome, _, _, ctx = runpod_run
    run_dir = ctx.out_dir / str(outcome.experiment_id)
    lines = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines()]
    started = [e for e in lines if e["kind"] == "engine_started"]
    assert [e["system"]["job_isolation"] for e in started] == ["ok", "ok"]
