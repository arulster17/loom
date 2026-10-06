"""RunpodProvider against FakeRunpod (REST/GraphQL), moto S3 and FakePodExec (SSH)."""

import asyncio
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import boto3
import httpx
import pytest
from pydantic import ValidationError

from loom_bench.engines import render_launch
from loom_bench.experiment import RunpodProviderSpec
from loom_bench.jobs import (
    EvalJob,
    EvalJobResult,
    EvalTaskResult,
    LoadJob,
    LoadJobResult,
    TokenizerSpec,
)
from loom_bench.prices import load_prices
from loom_bench.providers.base import HostLost, HostRequest
from loom_bench.providers.runpod import (
    MAX_START_CMD_BYTES,
    RunpodProvider,
    RunpodSettings,
    check_url,
    load_runpod_settings,
)
from loom_bench.providers.runpod_api import RunpodApiError
from loom_bench.providers.runpod_ssh import SshTarget
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.records import LoadMode, Market
from loom_bench.registry import load_registry

from .conftest import BUCKET, REGION
from .fakes import FakePodExec, FakeRunpod, engine_stdout, ok, script_var

VLLM = "vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90"
SGLANG = "lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
# 80 GB of container disk at $0.10/GB-month over 730 h = 10_958.9 micros/h.
DISK_80 = Fraction(100_000 * 80, 730)
L40S_X1 = 1_090_000


async def no_sleep(s: float) -> None:
    await asyncio.sleep(0)


def locked(extra: str) -> str:
    return f"pkg-{extra or 'base'}==1.0 --hash=sha256:{'0' * 64}\n"


@pytest.fixture
def s3(moto_aws: None) -> Any:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    return client


@pytest.fixture
def fake() -> FakeRunpod:
    return FakeRunpod(gets_before_ssh=0)


def settings(**kw: Any) -> RunpodSettings:
    return RunpodSettings(bucket=BUCKET, owner="arul", poll_interval_s=0.01, **kw)


def provider(
    fake: FakeRunpod,
    s3: Any,
    tmp_path: Path,
    *,
    ssh: FakePodExec | None = None,
    spec: RunpodProviderSpec | None = None,
    **kw: Any,
) -> RunpodProvider:
    return RunpodProvider(
        kw.pop("settings", None) or settings(),
        spec=spec or RunpodProviderSpec(kind="runpod"),
        api=fake.api(),
        prices=load_prices(),
        s3=s3,
        ssh=ssh or FakePodExec(),
        requirements=locked,
        sleep=no_sleep,
        ssh_dir=tmp_path / "ssh",
        **kw,
    )


def request(**kw: Any) -> HostRequest:
    base: dict[str, Any] = {
        "cloud": "runpod",
        "region": "secure",
        "instance_type": "l40s-x1",
        "market": Market.ON_DEMAND,
        "gpus": 1,
        "disk_gb": 80,
        "image": VLLM,
        "ttl_s": 2400,
        "tags": {"loom:experiment": "0f1e2d3c-aaaa-bbbb-cccc-000000000001", "team": "bench"},
    }
    return HostRequest(**{**base, **kw})


def vllm_launch() -> Any:
    return render_launch(load_registry().get("qwen3-8b"))


# --- provisioning ---------------------------------------------------------------------


async def test_pod_body_is_secure_on_demand_with_safety_rails(fake, s3, tmp_path) -> None:
    before = datetime.now(UTC)
    host = await provider(fake, s3, tmp_path).provision(request())

    (body,) = fake.bodies("POST", "/v1/pods")
    assert body["cloudType"] == "SECURE"
    assert body["interruptible"] is False
    assert body["computeType"] == "GPU"
    assert body["gpuTypeIds"] == ["NVIDIA L40S"]
    assert body["gpuCount"] == 1
    assert body["containerDiskInGb"] == 80
    assert body["volumeInGb"] == 0
    assert body["allowedCudaVersions"] == ["13.0"]
    assert body["ports"] == ["22/tcp"]
    assert body["imageName"] == VLLM
    assert body["dockerEntrypoint"] == ["bash", "-c"]
    assert "dataCenterIds" not in body

    ttl = int(host.ttl_at.timestamp())
    assert host.ttl_at - host.launched_at == timedelta(seconds=2400)
    assert host.launched_at >= before
    assert re.fullmatch(rf"loom-bench-0f1e2d3c-{ttl}-[0-9a-f]{{6}}", body["name"])
    (start,) = body["dockerStartCmd"]
    assert f"TTL_EPOCH={ttl}" in start
    assert len(start.encode()) <= MAX_START_CMD_BYTES
    assert host.info["tags"]["loom:pod-name"] == body["name"]
    assert host.info["tags"]["team"] == "bench"


async def test_pod_env_holds_only_the_secret_reference_loom_values_and_a_public_key(
    fake, s3, tmp_path
) -> None:
    p = provider(fake, s3, tmp_path)
    host = await p.provision(request())
    (body,) = fake.bodies("POST", "/v1/pods")
    env = body["env"]
    assert env["HF_TOKEN"] == "{{ RUNPOD_SECRET_hf_token }}"
    assert env["LOOM_MANAGED"] == "true"
    assert env["LOOM_TTL"] == str(int(host.ttl_at.timestamp()))
    assert env["LOOM_SSH_PUBKEY"].startswith("ssh-ed25519 ")
    assert env["LOOM_SSH_PUBKEY"] == p._keypair()[1]
    assert set(env) - {"HF_TOKEN"} <= {k for k in env if k.startswith("LOOM_")}
    assert "aws" not in json.dumps(env).lower()
    assert "RUNPOD_SECRET_aws" not in json.dumps(body)
    for value in env.values():
        assert "http" not in value
    start = body["dockerStartCmd"][0]
    assert "X-Amz" not in start and "RUNPOD_SECRET" not in start


async def test_ssh_key_stays_in_the_ssh_dir(fake, s3, tmp_path) -> None:
    p = provider(fake, s3, tmp_path)
    await p.provision(request())
    key, pub = p._keypair()
    assert key.parent.parent == tmp_path / "ssh"
    assert oct((tmp_path / "ssh").stat().st_mode & 0o777) == "0o700"
    assert pub in (key.with_suffix(".pub")).read_text()


async def test_accrual_is_the_api_price_and_disk_and_as_run_is_observed(fake, s3, tmp_path) -> None:
    host = await provider(fake, s3, tmp_path).provision(request())
    assert host.hourly_micros == L40S_X1 + 10_959  # disk rounded up
    assert host.as_run_micros == round(L40S_X1 + DISK_80)  # 1_100_958.9 -> half up
    basis = host.price_basis
    assert basis is not None
    assert basis.source == "observed_api"
    assert basis.market is Market.ON_DEMAND
    assert basis.availability_zone == "EUR-IS-2"
    assert basis.storage_gb == 80
    assert basis.observed_at == host.launched_at
    assert host.info["accrual_basis"]["api_micros_per_hour"] == L40S_X1


@pytest.mark.parametrize(
    ("cost", "accrued", "as_run"),
    [(1.2, 1_200_000, 1_200_000), (0.99, L40S_X1, 990_000)],
)
async def test_accrual_uses_the_higher_of_api_and_prices_yaml(
    s3, tmp_path, cost, accrued, as_run
) -> None:
    fake = FakeRunpod(cost_per_hr=cost, gets_before_ssh=0)
    host = await provider(fake, s3, tmp_path).provision(request())
    assert host.hourly_micros == accrued + 10_959
    assert host.as_run_micros == round(as_run + DISK_80)


async def test_missing_api_price_falls_back_to_prices_yaml(s3, tmp_path) -> None:
    fake = FakeRunpod(cost_per_hr=0, gets_before_ssh=0)
    host = await provider(fake, s3, tmp_path).provision(request())
    assert host.hourly_micros == L40S_X1 + 10_959
    assert host.price_basis is not None and host.price_basis.source == "prices_yaml"


async def test_price_far_above_prices_yaml_terminates_the_pod(s3, tmp_path) -> None:
    fake = FakeRunpod(cost_per_hr=1.5, gets_before_ssh=0)  # > 1.09 x 1.25
    with pytest.raises(RuntimeError, match=r"above prices\.yaml"):
        await provider(fake, s3, tmp_path).provision(request())
    assert fake.pods == {}


async def test_a_pod_created_by_a_failed_request_is_terminated(fake, s3, tmp_path) -> None:
    handle = fake.handle

    def drop_after_create(req: httpx.Request) -> httpx.Response:
        resp = handle(req)
        if req.method == "POST" and req.url.path == "/v1/pods":
            raise httpx.ReadTimeout("connection lost", request=req)
        return resp

    fake.handle = drop_after_create  # type: ignore[method-assign]
    p = provider(fake, s3, tmp_path)
    with pytest.raises(RunpodApiError):
        await p.provision(request())
    assert fake.pods == {}


async def test_a_refused_create_is_not_retried(fake, s3, tmp_path) -> None:
    fake.fail[("POST", "/v1/pods")] = [400]
    with pytest.raises(RunpodApiError, match="400"):
        await provider(fake, s3, tmp_path).provision(request())
    assert len(fake.bodies("POST", "/v1/pods")) == 1
    assert fake.bodies("GET", "/v1/pods") == []


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"cloud": "aws"}, "cloud 'runpod'"),
        ({"region": "community"}, "cloud type"),
        ({"market": Market.SPOT}, "on_demand"),
        ({"image": None}, "pinned"),
        ({"image": "vllm/vllm-openai:v0.30.0"}, "pinned"),
        ({"disk_gb": 0}, "container disk"),
        ({"ttl_s": 9 * 3600}, "ttl_s"),
        ({"gpus": 4}, "priced for 1 GPUs"),
        ({"tags": {}}, "loom:experiment"),
        ({"tags": {"loom:experiment": "e1", "loom:pod-name": "x"}}, "set by the provider"),
    ],
)
async def test_provision_rejects_bad_requests(fake, s3, tmp_path, kw, match) -> None:
    with pytest.raises(ValueError, match=match):
        await provider(fake, s3, tmp_path).provision(request(**kw))
    assert fake.pods == {}


async def test_spec_fields_reach_the_pod_body(fake, s3, tmp_path) -> None:
    spec = RunpodProviderSpec(
        kind="runpod",
        gpu_type_id="NVIDIA L40",
        allowed_cuda_versions=["13.0", "12.8"],
        data_center_ids=["EUR-IS-2"],
    )
    await provider(fake, s3, tmp_path, spec=spec).provision(request())
    (body,) = fake.bodies("POST", "/v1/pods")
    assert body["gpuTypeIds"] == ["NVIDIA L40"]
    assert body["allowedCudaVersions"] == ["13.0", "12.8"]
    assert body["dataCenterIds"] == ["EUR-IS-2"]


# --- host state and teardown -----------------------------------------------------------


@pytest.mark.parametrize("status", ["EXITED", "TERMINATED", None])
async def test_check_host_raises_host_lost(fake, s3, tmp_path, status) -> None:
    p = provider(fake, s3, tmp_path)
    host = await p.provision(request())
    if status is None:
        fake.pods.clear()
    else:
        fake.pods[host.host_id]["desiredStatus"] = status
    with pytest.raises(HostLost) as e:
        await p.check_host(host)
    assert e.value.host_id == host.host_id
    assert e.value.state == (status or "not-found").lower()


async def test_teardown_terminates_and_is_idempotent(fake, s3, tmp_path) -> None:
    p = provider(fake, s3, tmp_path)
    host = await p.provision(request())
    await p.teardown(host)
    await p.teardown(host)
    assert fake.pods == {}
    assert not any("/stop" in r.url.path for r in fake.requests)


async def test_teardown_falls_back_to_graphql_when_rest_refuses(fake, s3, tmp_path) -> None:
    p = provider(fake, s3, tmp_path)
    host = await p.provision(request())
    fake.rest_delete_status = 403
    await p.teardown(host)
    assert fake.pods == {}


async def test_reap_terminates_expired_managed_pods_only(fake, s3, tmp_path) -> None:
    past = int((datetime.now(UTC) - timedelta(minutes=5)).timestamp())
    expired = fake.add_pod(name=f"loom-bench-e-{past}-abc123", env={"LOOM_MANAGED": "true"})
    foreign = fake.add_pod(name=f"loom-bench-e-{past}-def456", env={})
    p = provider(fake, s3, tmp_path)
    assert await p.reap(datetime.now(UTC)) == [expired["id"]]
    assert set(fake.pods) == {foreign["id"]}


# --- engine ------------------------------------------------------------------------


async def test_cold_start_waits_for_ssh_and_reports_stages_and_system(fake, s3, tmp_path) -> None:
    fake.gets_before_ssh = 2
    ssh = FakePodExec(offline=2)
    p = provider(fake, s3, tmp_path, ssh=ssh)
    host = await p.provision(request())
    ep = await p.start_engine(host, vllm_launch(), warm=False)

    assert ssh.probes == 3
    assert ep.base_url == "http://127.0.0.1:8000/v1"
    assert ep.metrics_url == "http://127.0.0.1:8000/metrics"
    assert not ep.warm
    stages = ep.start_stages
    for name in (
        "pod_created",
        "pod_running",
        "ssh_online",
        "image_pulled",
        "sshd_ready",
        "weights_ready",
        "engine_started",
        "engine_healthy",
        "first_token",
    ):
        assert name in stages, name
    assert list(stages)[-1] == "first_token"
    assert list(stages.values()) == sorted(stages.values())
    assert ep.system["gpus"] == ["NVIDIA L40S"]
    assert ep.system["gpu_count"] == 1
    assert ep.system["driver_version"] == "580.159.03"
    assert ep.system["cuda_version"] == "13.0"
    assert ep.system["job_isolation"] == "ok"
    assert ep.system["pod_id"] == host.host_id
    assert ep.system["data_center"] == "EUR-IS-2"
    assert ep.system["instance_type"] == "l40s-x1"
    assert "gpu_name" not in ep.system

    target, _, script = ssh.launched[0]
    assert (target.host, target.port) == ("203.0.113.1", 28801)
    assert target.key_path == p._keypair()[0]
    assert script_var(script, "WARM") == "0"
    assert "--host 127.0.0.1" in script
    assert "0.0.0.0" not in script
    assert "RUNPOD_SECRET" not in script


async def test_engine_start_without_the_isolation_marker_fails(fake, s3, tmp_path) -> None:
    ssh = FakePodExec({"start_engine": lambda t, s: ok(engine_stdout(isolation=False))})
    p = provider(fake, s3, tmp_path, ssh=ssh)
    host = await p.provision(request())
    with pytest.raises(RuntimeError, match="job isolation"):
        await p.start_engine(host, vllm_launch(), warm=False)


async def test_failed_engine_start_raises_with_its_stderr(fake, s3, tmp_path) -> None:
    ssh = FakePodExec({"start_engine": lambda t, s: (1, "", "loom-error engine process exited")})
    p = provider(fake, s3, tmp_path, ssh=ssh)
    host = await p.provision(request())
    with pytest.raises(RuntimeError, match="engine process exited"):
        await p.start_engine(host, vllm_launch(), warm=False)


async def test_a_pod_serves_only_its_own_image(fake, s3, tmp_path) -> None:
    p = provider(fake, s3, tmp_path)
    host = await p.provision(request(image=SGLANG))
    with pytest.raises(ValueError, match="one engine image"):
        await p.start_engine(host, vllm_launch(), warm=False)


async def test_warm_start_and_stop(fake, s3, tmp_path) -> None:
    ssh = FakePodExec()
    p = provider(fake, s3, tmp_path, ssh=ssh)
    host = await p.provision(request())
    await p.start_engine(host, vllm_launch(), warm=False)
    probes = ssh.probes
    await p.stop_engine(host)
    ep = await p.start_engine(host, vllm_launch(), warm=True)
    assert ep.warm and ssh.probes == probes
    assert [name for _, name, _ in ssh.launched] == ["start_engine", "stop_engine", "start_engine"]
    assert script_var(ssh.scripts("start_engine")[1], "WARM") == "1"
    assert "pod_created" not in ep.start_stages


async def test_lost_pod_during_a_command_raises_host_lost(fake, s3, tmp_path) -> None:
    async def vanish(target: SshTarget, script: str) -> Any:
        fake.pods.clear()
        return (1, "", "connection closed")

    p = provider(fake, s3, tmp_path, ssh=FakePodExec({"stop_engine": vanish}))
    host = await p.provision(request())
    fake.pods[host.host_id].update(publicIp="203.0.113.9", portMappings={"22": 2222})
    with pytest.raises(HostLost):
        await p.stop_engine(host)


# --- jobs --------------------------------------------------------------------------


def load_job(**kw: Any) -> LoadJob:
    base: dict[str, Any] = {
        "run_id": "run-7",
        "base_url": "http://127.0.0.1:8000/v1",
        "metrics_url": "http://127.0.0.1:8000/metrics",
        "engine": "vllm",
        "served_model": "qwen3-8b",
        "workload": {"kind": "fixed", "input_len": 128, "output_len": 128},
        "tokenizer": TokenizerSpec(kind="simple"),
        "mode": LoadMode.CLOSED_LOOP,
        "load_value": 8,
        "duration_s": 60,
        "sample_gpu": True,
    }
    return LoadJob(**{**base, **kw})


def s3_key(url: str) -> str:
    return unquote(urlparse(url).path).lstrip("/").removeprefix(f"{BUCKET}/")


def pod_writes(s3: Any, result_json: str, *, gpu_csv: bool = True) -> Any:
    """A run_job handler that uploads results where the script's PUT URLs point."""

    def handler(target: SshTarget, script: str) -> Any:
        s3.put_object(Bucket=BUCKET, Key=s3_key(script_var(script, "RESULT_URL")), Body=result_json)
        if gpu_csv:
            key = s3_key(script_var(script, "GPU_CSV_URL"))
            s3.put_object(Bucket=BUCKET, Key=key, Body=b"2026/10/06, 0, 97, 1, 2, 300\n")
        return ok("loom-sys job_isolation ok\nloom-job-done\n")

    return handler


def load_result() -> LoadJobResult:
    return LoadJobResult(
        run_id="run-7",
        mode=LoadMode.CLOSED_LOOP,
        load_value=8,
        t_measure_start_s=0,
        t_measure_end_s=60,
        records=[],
        started_at="2026-10-06T00:00:00Z",
        finished_at="2026-10-06T00:01:00Z",
    )


async def test_run_job_round_trip_through_s3(fake, s3, tmp_path) -> None:
    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel-bytes")
    ssh = FakePodExec({"run_job": pod_writes(s3, load_result().model_dump_json())})
    p = provider(fake, s3, tmp_path, ssh=ssh, wheel_path=wheel)
    host = await p.provision(request())
    job = load_job()
    got = await p.run_job(host, job)

    assert got.run_id == "run-7"
    assert got.nvidia_smi_csv == "2026/10/06, 0, 97, 1, 2, 300\n"
    prefix = f"runs/{host.request.tags['loom:experiment']}/run-7/"
    stored = s3.get_object(Bucket=BUCKET, Key=prefix + "job.json")["Body"].read()
    assert LoadJob.model_validate_json(stored) == job
    assert s3.get_object(Bucket=BUCKET, Key=prefix + wheel.name)["Body"].read() == b"wheel-bytes"

    (script,) = ssh.scripts("run_job")
    for var in ("JOB_URL", "WHEEL_URL", "REQS_URL", "RESULT_URL", "GPU_CSV_URL"):
        url = script_var(script, var)
        assert url.startswith("https://") and "Signature" in url, var
    assert script_var(script, "SAMPLE_GPU") == "1"
    assert script_var(script, "REQS_SHA256") == hashlib.sha256(locked("").encode()).hexdigest()
    assert script_var(script, "WHEEL_SHA256") == hashlib.sha256(b"wheel-bytes").hexdigest()
    assert script_var(script, "PYTHON_SHA256") == p.settings.client_python_sha256
    assert script_var(script, "JOB_UID") == "10001"
    # The URLs never reach the pod's API-visible definition.
    pod = json.dumps(fake.pods[host.host_id])
    assert "Signature" not in pod and "amazonaws" not in pod


async def test_run_job_reads_the_tokenizer_from_the_pod_snapshot(fake, s3, tmp_path) -> None:
    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"w")
    spec = load_registry().get("qwen3-8b")
    ssh = FakePodExec({"run_job": pod_writes(s3, load_result().model_dump_json())})
    p = provider(fake, s3, tmp_path, ssh=ssh, wheel_path=wheel)
    host = await p.provision(request())
    tok = TokenizerSpec(kind="hf", repo=spec.hf.repo, revision=spec.hf.revision)
    await p.run_job(host, load_job(tokenizer=tok))
    prefix = f"runs/{host.request.tags['loom:experiment']}/run-7/"
    stored = LoadJob.model_validate_json(
        s3.get_object(Bucket=BUCKET, Key=prefix + "job.json")["Body"].read()
    )
    cache = "/opt/loom/hf/hub/models--Qwen--Qwen3-8B"
    assert stored.tokenizer is not None
    assert stored.tokenizer.local_dir == f"{cache}/snapshots/{spec.hf.revision}"
    (script,) = ssh.scripts("run_job")
    assert script_var(script, "MODEL_CACHE_DIR") == cache
    assert script_var(script, "MODEL_REVISION") == spec.hf.revision


async def test_job_without_the_isolation_marker_fails(fake, s3, tmp_path) -> None:
    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"w")
    ssh = FakePodExec({"run_job": lambda t, s: ok("loom-job-done\n")})
    p = provider(fake, s3, tmp_path, ssh=ssh, wheel_path=wheel)
    host = await p.provision(request())
    with pytest.raises(RuntimeError, match="isolation"):
        await p.run_job(host, load_job(sample_gpu=False))


async def test_run_job_needs_a_wheel_and_a_safe_run_id(fake, s3, tmp_path) -> None:
    p = provider(fake, s3, tmp_path)
    host = await p.provision(request())
    with pytest.raises(ValueError, match="no bench wheel"):
        await p.run_job(host, load_job())
    p.wheel_path = tmp_path / "w.whl"
    p.wheel_path.write_bytes(b"w")
    with pytest.raises(ValueError, match="run_id"):
        await p.run_job(host, load_job(run_id="../x"))


async def test_unsafe_presigned_urls_are_rejected(fake, s3, tmp_path) -> None:
    class QuotingS3:
        def __getattr__(self, name: str) -> Any:
            return getattr(s3, name)

        def generate_presigned_url(self, *a: Any, **k: Any) -> str:
            return 'https://b.s3.amazonaws.com/k?X-Amz-Signature=1"; rm -rf /'

    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"w")
    p = provider(fake, s3, tmp_path, wheel_path=wheel)
    p.s3 = QuotingS3()
    host = await p.provision(request())
    with pytest.raises(ValueError, match="quote"):
        await p.run_job(host, load_job())


@pytest.mark.parametrize(
    "url",
    [
        "http://b.s3.amazonaws.com/k",
        'https://b/k"',
        "https://b/k\\x",
        "https://b/k\nurl = evil",
    ],
)
def test_check_url_rejects_what_would_break_a_curl_config(url: str) -> None:
    with pytest.raises(ValueError):
        check_url(url)


async def test_run_eval_round_trip_through_s3(fake, s3, tmp_path) -> None:
    wheel = tmp_path / "loom_bench-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"w")
    suite = Suite.model_validate(
        {
            "suite": "s",
            "model": "qwen3-8b",
            "tasks": [{"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 3}}],
        }
    )
    job = EvalJob(
        run_id="eval-1",
        base_url="http://127.0.0.1:8000/v1",
        served_model="qwen3-8b",
        suite=suite,
        tokenizer=TokenizerSpec(kind="simple"),
    )
    result = EvalJobResult(
        run_id="eval-1",
        suite="s",
        model="qwen3-8b",
        tasks={
            "arithmetic": EvalTaskResult(
                kind="toy_arithmetic",
                version="1",
                items=[ItemResult(item_id="a", score=1.0, content_hash="h")],
                seconds=1.0,
            )
        },
        sanity=SanityResult(n=1, counts={"empty": 0}),
        started_at="2026-10-06T00:00:00+00:00",
        finished_at="2026-10-06T00:01:00+00:00",
    )
    ssh = FakePodExec({"run_job": pod_writes(s3, result.model_dump_json(), gpu_csv=False)})
    p = provider(fake, s3, tmp_path, ssh=ssh, wheel_path=wheel)
    host = await p.provision(request())
    got = await p.run_eval(host, job)
    assert got == result
    (script,) = ssh.scripts("run_job")
    assert "BENCH_CMD=(quality job)" in script
    assert script_var(script, "SAMPLE_GPU") == "0"


# --- settings ------------------------------------------------------------------------


def test_settings_from_yaml_then_env(tmp_path: Path) -> None:
    path = tmp_path / "runpod.yaml"
    path.write_text("bucket: loom-bench-x\nowner: arul\nmax_ttl_s: 3600\n")
    s = load_runpod_settings(path, env={"LOOM_RUNPOD_MAX_TTL_S": "7200"})
    assert s.bucket == "loom-bench-x" and s.owner == "arul"
    assert s.max_ttl_s == 7200
    assert s.hf_secret_name == "hf_token"
    s2 = load_runpod_settings(env={"LOOM_RUNPOD_CONFIG": str(path)})
    assert s2.max_ttl_s == 3600


@pytest.mark.parametrize(
    "kw",
    [
        {"hf_secret_name": "aws_access_key_id"},
        {"hf_secret_name": "AWS_SECRET"},
        {"hf_secret_name": "hf token"},
        {"graphql_url": "http://api.runpod.io/graphql"},
        {"name_prefix": "Loom Bench"},
        {"max_price_ratio": "0.9"},
    ],
)
def test_settings_reject_unsafe_values(kw: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        settings(**kw)


def test_start_command_is_rendered_from_settings(fake, s3, tmp_path) -> None:
    p = provider(fake, s3, tmp_path, settings=settings(authorize_account_key=False))
    script = p.start_command(datetime(2026, 10, 6, tzinfo=UTC))
    assert script_var(script, "AUTHORIZE_ACCOUNT_KEY") == "0"
    assert script_var(script, "GRAPHQL_URL") == "https://api.runpod.io/graphql"
    assert script_var(script, "TTL_EPOCH") == str(
        int(datetime(2026, 10, 6, tzinfo=UTC).timestamp())
    )
