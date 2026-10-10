"""An EC2 GPU host behind FakeSsm, for running whole experiments on the aws_ec2 provider
offline: moto for EC2 and S3, and each instance's engine served by the mock backend.

Like the RunPod pod simulator (tests/runpod/test_runpod_e2e.PodSim), each host is held to
its scripts' rules, which their own tests (test_ssm_scripts.py) check on the scripts
themselves:

- start_engine.sh: CACHED_WEIGHTS must be in the host's Hugging Face cache, FETCH_WEIGHTS
  are added to it, and the engine, which runs offline, starts only if its checkpoint is in
  it. A cold start reports user-data's stages and the TTL shutdown systemd armed, read
  from the instance's own user data (TTL_EPOCH).
- run_job.sh: a job's tokenizer snapshot (MODEL_CACHE_DIR, MODEL_REVISION) must be in the
  cache; a pinned dataset (DATA_URL) is fetched once per host from `served`, checked
  against DATA_SHA256 and kept under DATA_DIR, and the job reads it where the client
  container mounts DATA_DIR (DATA_MOUNT). Inputs come from, and results go to, the S3 keys
  the script's presigned URLs name.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import shlex
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from loom_bench.jobexec import execute_load_job
from loom_bench.jobs import EvalJob, LoadJob, TokenizerSpec
from loom_bench.mock.config import MockConfig
from loom_bench.providers.mock import _start_server, _stop_server
from loom_bench.quality.runner import execute_eval_job
from loom_bench.tokenize import hf_cache_folder

from .fakes import FakeSsm

BUCKET = "loom-bench-test"


def var(script: str, name: str) -> str:
    """A scalar variable as rendered into a host script (`NAME='value'`)."""
    m = re.search(rf"^{name}=(.*)$", script, re.MULTILINE)
    assert m, name
    (value,) = shlex.split(m.group(1)) or [""]
    return value


def array(script: str, name: str) -> list[str]:
    m = re.search(rf"^{name}=\((.*)\)$", script, re.MULTILINE)
    assert m, name
    return shlex.split(m.group(1))


def pairs(script: str, name: str) -> list[tuple[str, str]]:
    flat = array(script, name)
    return list(zip(flat[::2], flat[1::2], strict=True))


def s3_key(url: str) -> str:
    return unquote(urlparse(url).path).lstrip("/").removeprefix(f"{BUCKET}/")


def kind_of(script: str) -> str:
    if re.search(r"^ENGINE_CMD=", script, re.MULTILINE):
        return "start_engine"
    if re.search(r"^BENCH_CMD=", script, re.MULTILINE):
        return "run_job"
    return "stop_engine"


def engine_checkpoint(script: str) -> tuple[str, str]:
    cmd = array(script, "ENGINE_CMD")
    i = cmd.index("serve")
    return cmd[i + 1], cmd[cmd.index("--revision") + 1]


def _started_server(cfg: MockConfig) -> Any:
    srv = _start_server(cfg)
    deadline = time.monotonic() + 10
    while not srv.server.started:
        assert time.monotonic() < deadline, "mock server did not start"
        time.sleep(0.005)
    return srv


class HostSim(FakeSsm):
    def __init__(
        self,
        ec2: Any,
        s3: Any,
        root: Path,
        *,
        served: dict[str, bytes] | None = None,
        mock: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.ec2, self.s3, self.root = ec2, s3, root
        self.served = served or {}  # dataset URL -> bytes
        self.mock = mock or {}
        self.servers: dict[str, Any] = {}  # instance -> mock server
        self.cache: dict[str, set[tuple[str, str]]] = {}  # instance -> its HF cache
        self.downloads: dict[str, list[tuple[str, str]]] = {}  # instance -> weights fetched
        self.dataset_fetches: dict[str, list[str]] = {}  # instance -> dataset URLs fetched
        self.launched: list[tuple[str, str, str]] = []  # (instance, kind, script)
        self.jobs: list[LoadJob | EvalJob] = []
        self._lock = threading.Lock()

    def scripts_of(self, kind: str) -> list[str]:
        return [s for _, k, s in self.launched if k == kind]

    def send_command(self, **kwargs: Any) -> dict[str, Any]:
        (instance,) = kwargs["InstanceIds"]
        script = kwargs["Parameters"]["commands"][1].split("\n", 1)[1]
        script = script.rsplit("\nLOOM_SCRIPT_EOF", 1)[0]
        kind = kind_of(script)
        with self._lock:
            self.launched.append((instance, kind, script))
        handler = {"start_engine": self.start, "stop_engine": self.stop, "run_job": self.job}
        rc, out, err = handler[kind](instance, script)
        inv = {
            "Status": "Success" if rc == 0 else "Failed",
            "StandardOutputContent": out,
            "StandardErrorContent": err,
        }
        self.on_command = lambda _script: [inv]
        return super().send_command(**kwargs)

    # -- start / stop ---------------------------------------------------------------

    def _ttl_epoch(self, instance: str) -> int:
        raw = self.ec2.describe_instance_attribute(InstanceId=instance, Attribute="userData")
        user_data = base64.b64decode(raw["UserData"]["Value"]).decode()
        m = re.search(r"^TTL_EPOCH=(\d+)$", user_data, re.MULTILINE)
        assert m, "user data arms no TTL"
        return int(m.group(1))

    def start(self, instance: str, script: str) -> tuple[int, str, str]:
        cache = self.cache.setdefault(instance, set())
        absent = [c for c in pairs(script, "CACHED_WEIGHTS") if c not in cache]
        if absent:
            return 1, "", f"loom-error weights for {absent} are not cached"
        fetch = pairs(script, "FETCH_WEIGHTS")
        cache.update(fetch)
        self.downloads.setdefault(instance, []).extend(fetch)
        if engine_checkpoint(script) not in cache:
            return 1, "", f"OfflineModeIsEnabled: the engine's {engine_checkpoint(script)}"
        cfg = MockConfig(
            **{
                "time_scale": 0.01,
                "models": [var(script, "SERVED_MODEL")],
                "logprob_jitter": 0.25,
                "byte_level": True,
                **self.mock,
            }
        )
        self.stop(instance, script)
        self.servers[instance] = _started_server(cfg)
        now = time.time()
        lines = []
        if var(script, "WARM") == "0":
            lines += [
                f"loom-stage user_data_started {now - 30}",
                f"loom-stage user_data_done {now - 29}",
                f"loom-sys ttl_shutdown_usec {self._ttl_epoch(instance) * 1_000_000}",
            ]
        lines += [
            f"loom-stage image_pulled {now + 1}",
            f"loom-stage weights_ready {now + 2}",
            f"loom-stage engine_started {now + 3}",
            f"loom-stage engine_healthy {now + 4}",
            f"loom-stage first_token {now + 5}",
            "loom-sys gpus NVIDIA L40S",
            "loom-sys driver_version 595.91.07",
            "loom-sys cuda_version 13.2",
            f"loom-sys image_digest {var(script, 'IMAGE')}",
        ]
        return 0, "\n".join(lines) + "\n", ""

    def stop(self, instance: str, script: str) -> tuple[int, str, str]:
        srv = self.servers.pop(instance, None)
        if srv is not None:
            _stop_server(srv)
        return 0, "loom-stopped loom-engine\n", ""

    # -- jobs -----------------------------------------------------------------------

    def _dataset(self, instance: str, script: str, workload: dict[str, Any]) -> dict[str, Any]:
        url = var(script, "DATA_URL")
        if not url:
            return workload
        host_path, data_dir = var(script, "DATA_PATH"), var(script, "DATA_DIR")
        mount = var(script, "DATA_MOUNT")
        assert host_path.startswith(data_dir + "/"), (host_path, data_dir)
        local = self.root / instance / host_path.lstrip("/")
        if not local.exists():
            body = self.served[url]
            assert hashlib.sha256(body).hexdigest() == var(script, "DATA_SHA256")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(body)
            self.dataset_fetches.setdefault(instance, []).append(url)
        # The container sees DATA_DIR at DATA_MOUNT (read-only).
        path = workload["path"]
        assert path.startswith(mount + "/"), (path, mount)
        rel = path.removeprefix(mount + "/")
        assert host_path == f"{data_dir}/{rel}"
        return {**workload, "path": str(local)}

    def job(self, instance: str, script: str) -> tuple[int, str, str]:
        model_dir = var(script, "MODEL_CACHE_DIR")
        if model_dir:
            revision = var(script, "MODEL_REVISION")
            held = {(hf_cache_folder(r), rev) for r, rev in self.cache.get(instance, set())}
            if (Path(model_dir).name, revision) not in held:
                return 1, "", f"loom-error no tokenizer.json in snapshot {revision}"
        root = f"http://127.0.0.1:{self.servers[instance].port}"
        body = self.s3.get_object(Bucket=BUCKET, Key=s3_key(var(script, "JOB_URL")))["Body"]
        raw = body.read()
        simple = TokenizerSpec(kind="simple")
        if array(script, "BENCH_CMD") == ["quality", "job"]:
            ej = EvalJob.model_validate_json(raw)
            self.jobs.append(ej)
            ej = ej.model_copy(update={"base_url": f"{root}/v1", "tokenizer": simple})
            out = asyncio.run(execute_eval_job(ej)).model_dump_json()
        else:
            lj = LoadJob.model_validate_json(raw)
            self.jobs.append(lj)
            lj = lj.model_copy(
                update={
                    "base_url": f"{root}/v1",
                    "metrics_url": f"{root}/metrics",
                    "tokenizer": simple,
                    "workload": self._dataset(instance, script, dict(lj.workload)),
                }
            )
            out = asyncio.run(execute_load_job(lj)).model_dump_json()
            if var(script, "SAMPLE_GPU") == "1":
                key = s3_key(var(script, "GPU_CSV_URL"))
                self.s3.put_object(Bucket=BUCKET, Key=key, Body=b"2026/10/10, 0, 50, 1, 2, 3\n")
        self.s3.put_object(Bucket=BUCKET, Key=s3_key(var(script, "RESULT_URL")), Body=out)
        return 0, "loom-job-done\n", ""

    def close(self) -> None:
        for srv in self.servers.values():
            _stop_server(srv)
        self.servers.clear()
