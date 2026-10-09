"""The RunPod pod scripts: rendering, static safety properties, and behaviour run
locally with stub binaries on PATH (setpriv, setsid, curl, nvidia-smi, ...).

The stubs stand in for Linux-only tools so the control flow is tested on any
machine; the real kernel enforcement (the job uid denied /proc/1/environ) is
asserted again inside the pod by the smoke test (`loom-sys job_isolation ok`).
"""

import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from loom_bench.engines import engine_process_argv, render_launch
from loom_bench.providers.aws_ssm import render_script, required_vars
from loom_bench.registry import load_registry

BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not installed")
DIR = "runpod_scripts"
TOKEN = "hf_FAKEtokenFAKEtokenFAKEtoken"
POD_KEY = "rpa_PODSCOPEDFAKEKEYFAKEKEY"

POD_START_VARS: dict[str, Any] = {
    "TTL_EPOCH": 1_790_000_000,
    "STAGE_FILE": "/var/lib/loom/stages",
    "LOOM_ROOT": "/var/lib/loom",
    "SSH_DIR": "/root/.ssh",
    "SSHD_CONFIG": "/var/lib/loom/sshd_config",
    "CTL_ROOT": "/var/lib/loom/ctl",
    "JOB_UID": 10001,
    "JOB_USER": "loom",
    "JOB_HOME": "/var/lib/loom/home",
    "AUTHORIZE_ACCOUNT_KEY": 1,
    "GRAPHQL_URL": "https://api.runpod.io/graphql",
}
START_VARS: dict[str, Any] = {
    "WARM": 0,
    "FETCH_WEIGHTS": ["Qwen/Qwen3-8B", "b" * 40],
    "CACHED_WEIGHTS": [],
    "WEIGHTS_DIR": "/opt/loom/hf",
    "PORT": 8000,
    "SERVED_MODEL": "qwen3-8b",
    "READY_TIMEOUT_S": 1800,
    "ENGINE_STALL_S": 600,
    "LOG_DIR": "/var/log/loom",
    "STAGE_FILE": "/var/lib/loom/stages",
    "ENGINE_ENV": ["VLLM_LOGGING_LEVEL=INFO"],
    "ENGINE_CMD": ["vllm", "serve", "Qwen/Qwen3-8B", "--host", "127.0.0.1"],
    "ENGINE_PIDFILE": "/var/lib/loom/engine.pid",
    "PROC1_ENVIRON": "/proc/1/environ",
    "JOB_UID": 10001,
    "JOB_HOME": "/var/lib/loom/home",
    "SCAN_DIRS": ["/etc", "/opt/loom", "/var/lib/loom", "/tmp", "/root"],
}
JOB_VARS: dict[str, Any] = {
    "WORK_DIR": "/var/lib/loom/jobs/r1",
    "ENV_ROOT": "/opt/loom/clientenv",
    "PYTHON_DIR": "/opt/loom/python",
    "PYTHON_URL": "https://example.test/cpython.tar.gz",
    "PYTHON_SHA256": "e" * 64,
    "JOB_URL": "https://b.s3.amazonaws.com/runs/e/r1/job.json?X-Amz-Signature=1&b=2",
    "WHEEL_URL": "https://b.s3.amazonaws.com/w.whl?X-Amz-Signature=3",
    "WHEEL_NAME": "loom_bench-0.1.0-py3-none-any.whl",
    "WHEEL_SHA256": "c" * 64,
    "REQS_URL": "https://b.s3.amazonaws.com/requirements.txt?X-Amz-Signature=4",
    "REQS_SHA256": "d" * 64,
    "BENCH_CMD": ["job", "run"],
    "RESULT_URL": "https://b.s3.amazonaws.com/result.json?X-Amz-Signature=2",
    "GPU_CSV_URL": "",
    "SAMPLE_GPU": 0,
    "MODEL_CACHE_DIR": "",
    "MODEL_REVISION": "",
    "PROC1_ENVIRON": "/proc/1/environ",
    "JOB_UID": 10001,
    "JOB_HOME": "/var/lib/loom/home",
    "PROC_ROOT": "/proc",
}
RENDERED = {
    "pod_start": POD_START_VARS,
    "start_engine": START_VARS,
    "stop_engine": {"ENGINE_PIDFILE": "/var/lib/loom/engine.pid", "STOP_TIMEOUT_S": 60},
    "run_job": JOB_VARS,
}


def render(name: str, **over: Any) -> str:
    return render_script(name, template_dir=DIR, **{**RENDERED[name], **over})


def template(name: str) -> str:
    from importlib.resources import files

    return (files("loom_bench.providers") / DIR / f"{name}.sh").read_text()


def stub(bin_dir: Path, name: str, body: str) -> Path:
    path = bin_dir / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
    return path


def run_bash(
    script: str, *, env: dict[str, str], timeout: float = 60
) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, "-s"], input=script, env=env, capture_output=True, text=True, timeout=timeout
    )


def proc1_file(path: Path, env: dict[str, str]) -> Path:
    path.write_bytes(b"".join(f"{k}={v}".encode() + b"\0" for k, v in env.items()))
    return path


def setpriv_stub(bin_dir: Path, log: Path, proc1: Path) -> None:
    """Drops setpriv's options and runs the rest as the current user. It denies the
    job user PID 1's environ unless `$STUB_JOB_READS_PROC1` is 1 (simulating a
    kernel or setup that leaks it)."""
    stub(
        bin_dir,
        "setpriv",
        f'echo "setpriv $*" >>{shlex.quote(str(log))}\n'
        'opts=(); while [ "${1#--}" != "$1" ]; do opts+=("$1"); shift; done\n'
        f'if [ "${{STUB_JOB_READS_PROC1:-0}}" != 1 ]; then\n'
        f'  for a in "$@"; do [ "$a" = {shlex.quote(str(proc1))} ] && exit 1; done\n'
        "fi\n"
        'exec "$@"\n',
    )


# ---------------------------------------------------------------------------- rendering


def test_required_vars_match() -> None:
    for name, values in RENDERED.items():
        assert required_vars(name, template_dir=DIR) == set(values), name


def test_aws_templates_still_render_by_default() -> None:
    assert required_vars("stop_engine") == {"CONTAINER"}
    assert "docker rm -f" in render_script("stop_engine", CONTAINER="c")


@pytest.mark.parametrize("name", sorted(RENDERED))
def test_rendered_scripts_are_valid_bash(name: str) -> None:
    script = render(name)
    assert BASH is not None
    assert not re.search(r"^\s*set -[a-z]*x|xtrace", script, re.MULTILINE)
    subprocess.run([BASH, "-n"], input=script, text=True, check=True)
    assert "loom-stage" in script or name == "stop_engine"


def test_values_are_quoted_not_interpreted() -> None:
    hostile = 'x\'; echo pwned; $(id) `id` "q"'
    script = render("start_engine", SERVED_MODEL=hostile, ENGINE_CMD=[hostile, "b c"])
    header = script.split("\n\n", 1)[0]
    out = run_bash(
        header + '\nprintf "%s|" "$SERVED_MODEL" "${ENGINE_CMD[@]}"', env=dict(os.environ)
    ).stdout
    assert out == f"{hostile}|{hostile}|b c|"


def test_no_script_holds_aws_credentials_or_secret_placeholders() -> None:
    for name in RENDERED:
        script = render(name)
        assert "RUNPOD_SECRET" not in script
        assert "AWS_ACCESS_KEY" not in script and "AWS_SECRET" not in script
        assert "aws " not in script  # no AWS CLI anywhere in the pod
        assert not re.search(r"\$\{?RUNPOD_API_KEY", script), name  # never expanded in shell


# ---------------------------------------------------------------------------- pod_start


def test_pod_start_arms_the_watchdog_before_anything_can_fail() -> None:
    script = render("pod_start")
    body = script.split("\n\n", 1)[1]
    watchdog = body.index("loom_terminate ttl")
    assert watchdog < body.index("apt-get")
    assert watchdog < body.index("loom_setup()")
    assert "TTL_EPOCH=1790000000" in script
    # Absolute: a container restart re-runs this with the same epoch.
    assert "remaining=$((TTL_EPOCH - $(date +%s)))" in script
    assert "set +euo pipefail" in body


def test_pod_start_never_exits_and_never_stops_the_pod() -> None:
    body = template("pod_start")
    assert not re.search(r"^\s*exit\b", body, re.MULTILINE)
    assert re.search(r"while :; do\n  sleep infinity\ndone\n$", body)
    assert "podTerminate" in body
    assert "podStop" not in body and "/stop" not in body
    # Setup failure terminates the pod instead of leaving it billing.
    assert "loom_terminate setup_failed &" in body


def test_pod_start_sshd_is_key_only_and_env_free() -> None:
    body = template("pod_start")
    assert "env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin /usr/sbin/sshd -f" in body
    for line in [
        "PermitRootLogin prohibit-password",
        "AllowUsers root",
        "PasswordAuthentication no",
        "KbdInteractiveAuthentication no",
        "AllowTcpForwarding no",
        "AllowAgentForwarding no",
        "PermitUserEnvironment no",
    ]:
        assert f"\n{line}\n" in body
    assert 'useradd --uid "$JOB_UID"' in body
    assert "--shell /usr/sbin/nologin" in body
    assert 'chmod 700 "$SSH_DIR" "$CTL_ROOT"' in body


def test_pod_start_fits_a_start_command() -> None:
    # The whole script is the pod's dockerStartCmd; keep it small.
    assert len(render("pod_start").encode()) < 8192


def test_authorized_keys_filter(tmp_path: Path) -> None:
    body = template("pod_start")
    m = re.search(r"(  \{\n    printf.*?\n  \} \| grep -E '[^']+')", body, re.DOTALL)
    assert m is not None
    snippet = m.group(1).strip()
    good = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFAKE loom-bench"
    env = {
        **os.environ,
        "LOOM_SSH_PUBKEY": good,
        "PUBLIC_KEY": "garbage; rm -rf /\nssh-rsa AAAAB3Nza account",
        "AUTHORIZE_ACCOUNT_KEY": "1",
    }
    out = run_bash(snippet + "\n", env=env).stdout.splitlines()
    assert out == [good, "ssh-rsa AAAAB3Nza account"]
    env["AUTHORIZE_ACCOUNT_KEY"] = "0"
    assert run_bash(snippet + "\n", env=env).stdout.splitlines() == [good]


class _GraphqlHandler(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.seen.append(
            {"auth": self.headers["Authorization"], "body": json.loads(self.rfile.read(length))}
        )
        payload = json.dumps({"data": {"podTerminate": None}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a: Any) -> None:
        return None


def test_in_pod_terminate_uses_the_pod_key_from_its_environment() -> None:
    body = template("pod_start")
    m = re.search(r"^TERMINATE_PY='(.*?)'$", body, re.MULTILINE | re.DOTALL)
    assert m is not None
    code = m.group(1)
    server = HTTPServer(("127.0.0.1", 0), _GraphqlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/graphql"
        env = {**os.environ, "RUNPOD_API_KEY": POD_KEY, "RUNPOD_POD_ID": "fakepod000001"}
        proc = subprocess.run(
            ["python3", "-c", code, url], env=env, capture_output=True, text=True, timeout=30
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "terminate requested"
        (req,) = _GraphqlHandler.seen
        assert req["auth"] == f"Bearer {POD_KEY}"
        assert "podTerminate" in req["body"]["query"]
        assert req["body"]["variables"] == {"podId": "fakepod000001"}
        missing = subprocess.run(
            ["python3", "-c", code, url],
            env={k: v for k, v in env.items() if k != "RUNPOD_API_KEY"},
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert missing.returncode != 0 and "missing" in missing.stderr
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------- start_engine


def engine_env_world(tmp_path: Path, *, token: str = TOKEN) -> dict[str, Any]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    scan = tmp_path / "scan"
    (scan / "hf").mkdir(parents=True)
    (scan / "hf" / "weights.bin").write_text(f"weights may embed anything {token}")
    (scan / "etc.conf").write_text("nothing secret\n")
    stages = tmp_path / "stages"
    stages.write_text("image_pulled 100.0\nsshd_ready 101.5\n")
    proc1 = proc1_file(
        tmp_path / "proc1_environ",
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
            "CUDA_HOME": "/usr/local/cuda",
            "HF_TOKEN": token,
            "RUNPOD_API_KEY": POD_KEY,
            "RUNPOD_POD_ID": "fakepod000001",
            "RUNPOD_DC_ID": "EUR-IS-2",
            "PUBLIC_KEY": "ssh-ed25519 AAAA account",
            "LOOM_SSH_PUBKEY": "ssh-ed25519 AAAA runner",
            "MY_SERVICE_PASSWORD": "hunter2hunter2",
            "WANDB_API_KEY": "wandbwandbwandb",
        },
    )
    setpriv_stub(bin_dir, logs / "setpriv", proc1)
    stub(bin_dir, "setsid", 'exec "$@"\n')
    stub(
        bin_dir,
        "python3",
        f"env >{shlex.quote(str(logs / 'download_env'))}\n"
        f'echo "$*" >{shlex.quote(str(logs / "download_args"))}\n',
    )
    stub(bin_dir, "curl", "exit 0\n")
    stub(
        bin_dir,
        "nvidia-smi",
        'case "$*" in\n'
        "  *--query-gpu=name*)\n"
        '    echo "NVIDIA L40S"; [ -z "$STUB_TWO_GPUS" ] || echo "NVIDIA L40S" ;;\n'
        '  *--query-gpu=driver_version*) echo "580.159.03" ;;\n'
        '  "topo -m")\n'
        '    if [ -n "$STUB_TWO_GPUS" ]; then\n'
        "      printf '\\tGPU0\\tGPU1\\tNIC0\\tCPU Affinity\\n'\n"
        "      printf 'GPU0\\t X \\tSYS\\tPXB\\t0-31\\nGPU1\\tSYS\\t X \\tNODE\\t32-63\\n'\n"
        "    else\n"
        "      printf '\\tGPU0\\tNIC0\\tCPU Affinity\\nGPU0\\t X \\tPXB\\t0-31\\n'\n"
        "    fi ;;\n"
        '  "") echo "| NVIDIA-SMI 580.159.03  Driver Version: 580.159.03  CUDA Version: 13.0" ;;\n'
        "esac\n",
    )
    engine = stub(
        bin_dir,
        "fake-engine",
        f"env >{shlex.quote(str(logs / 'engine_env'))}\n"
        f'echo "$*" >{shlex.quote(str(logs / "engine_args"))}\n'
        "exec sleep 30\n",
    )
    launch = render_launch(load_registry().get("qwen3-8b"))
    argv = engine_process_argv(launch, host="127.0.0.1")
    values = {
        **START_VARS,
        "WEIGHTS_DIR": str(scan / "hf"),
        "LOG_DIR": str(tmp_path / "log"),
        "STAGE_FILE": str(stages),
        "ENGINE_CMD": [str(engine), *argv[1:]],
        "ENGINE_PIDFILE": str(tmp_path / "engine.pid"),
        "PROC1_ENVIRON": str(proc1),
        "JOB_HOME": str(tmp_path / "home"),
        "SCAN_DIRS": [str(scan), str(tmp_path / "does-not-exist")],
    }
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
    return {
        "values": values,
        "env": env,
        "logs": logs,
        "scan": scan,
        "pidfile": values["ENGINE_PIDFILE"],
    }


def kill_engine(pidfile: str) -> None:
    with contextlib.suppress(FileNotFoundError, ProcessLookupError, ValueError):
        os.kill(int(Path(pidfile).read_text()), signal.SIGKILL)


def read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        k, _, v = line.partition("=")
        out[k] = v
    return out


def test_start_engine_keeps_secrets_from_the_engine_and_proves_isolation(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path)
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"]
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 0, proc.stderr
    for line in [
        "loom-stage image_pulled 100.0",
        "loom-stage sshd_ready 101.5",
        "loom-sys job_isolation ok",
        "loom-sys gpus NVIDIA L40S",
        "loom-sys gpu_count 1",
        "loom-sys driver_version 580.159.03",
        "loom-sys cuda_version 13.0",
        "loom-sys data_center EUR-IS-2",
        "loom-sys gpu_topology GPU0:X",
    ]:
        assert line in proc.stdout.splitlines()
    for name in ["weights_ready", "engine_started", "engine_healthy", "first_token"]:
        assert re.search(rf"^loom-stage {name} ", proc.stdout, re.MULTILINE)
    # Only the download child got the token, and nobody printed it.
    download = read_env(w["logs"] / "download_env")
    assert download["HF_TOKEN"] == TOKEN
    assert download["HF_HOME"] == w["values"]["WEIGHTS_DIR"]
    assert TOKEN not in proc.stdout + proc.stderr and POD_KEY not in proc.stdout + proc.stderr
    engine = read_env(w["logs"] / "engine_env")
    for secret in [
        "HF_TOKEN",
        "RUNPOD_API_KEY",
        "RUNPOD_POD_ID",
        "PUBLIC_KEY",
        "LOOM_SSH_PUBKEY",
        "MY_SERVICE_PASSWORD",
        "WANDB_API_KEY",
    ]:
        assert secret not in engine
    assert engine["LD_LIBRARY_PATH"] == "/usr/local/cuda/lib64"
    assert engine["CUDA_HOME"] == "/usr/local/cuda"
    assert engine["HF_HUB_OFFLINE"] == "1" and engine["TRANSFORMERS_OFFLINE"] == "1"
    assert engine["VLLM_LOGGING_LEVEL"] == "INFO"
    args = (w["logs"] / "engine_args").read_text()
    assert "--host 127.0.0.1 --port 8000" in args
    assert "0.0.0.0" not in args


def test_start_engine_records_the_gpu_topology_before_the_engine(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path)
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]),
            env={**w["env"], "STUB_TWO_GPUS": "1"},
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    # The NICs and CPU affinity columns are dropped; SYS marks a cross-socket pair.
    assert "loom-sys gpu_topology GPU0:X,SYS;GPU1:SYS,X" in lines
    assert "loom-sys gpu_count 2" in lines
    started = next(i for i, x in enumerate(lines) if x.startswith("loom-stage engine_started"))
    assert lines.index("loom-sys gpu_topology GPU0:X,SYS;GPU1:SYS,X") < started


def stalled_world(tmp_path: Path, engine_body: str, **values: Any) -> dict[str, Any]:
    w = engine_env_world(tmp_path)
    bin_dir = tmp_path / "bin"
    stub(bin_dir, "curl", "exit 7\n")  # never healthy
    engine = stub(bin_dir, "fake-engine", engine_body)
    w["values"] = {**w["values"], "ENGINE_CMD": [str(engine)], **values}
    return w


def test_start_engine_fails_fast_when_the_engine_goes_silent(tmp_path: Path) -> None:
    # 874110b6: vLLM printed its NCCL line, then nothing until the 1800 s timeout.
    w = stalled_world(tmp_path, 'echo "vLLM is using nccl"\nexec sleep 60\n', ENGINE_STALL_S=3)
    t0 = time.monotonic()
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"]
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 1
    assert "loom-error engine stalled: no output for 3s before it was healthy" in proc.stderr
    assert "vLLM is using nccl" in proc.stderr  # the log tail comes with the error
    assert time.monotonic() - t0 < 30  # not the 1800 s ready timeout
    assert "loom-sys gpu_topology GPU0:X" in proc.stdout.splitlines()


def test_start_engine_output_resets_the_stall_clock(tmp_path: Path) -> None:
    # An engine that keeps logging is slow, not stalled: only the deadline stops it.
    w = stalled_world(
        tmp_path,
        "while :; do echo loading; sleep 1; done\n",
        ENGINE_STALL_S=3,
        READY_TIMEOUT_S=8,
    )
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"]
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 1
    assert "loom-error engine not healthy after 8s" in proc.stderr
    assert "stalled" not in proc.stderr


def test_start_engine_fails_when_the_job_user_can_read_proc1(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path)
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]),
            env={**w["env"], "STUB_JOB_READS_PROC1": "1"},
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 1
    assert "loom-error job user can read" in proc.stderr
    assert "job_isolation ok" not in proc.stdout


def test_start_engine_fails_on_a_stored_secret(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path)
    (w["scan"] / "leak.env").write_text(f"export HF_TOKEN={TOKEN}\n")
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"]
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 1
    assert "loom-error a secret value is stored under" in proc.stderr
    assert TOKEN not in proc.stderr


def test_start_engine_refuses_an_unsubstituted_secret(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path, token="{{ RUNPOD_SECRET_hf_token }}")
    proc = run_bash(render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"])
    assert proc.returncode == 1
    assert "was not substituted" in proc.stderr
    assert not (w["logs"] / "engine_args").exists()


def test_warm_start_on_held_weights_downloads_nothing_and_needs_no_token(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path)
    values = {
        **w["values"],
        "WARM": 1,
        "FETCH_WEIGHTS": [],
        "CACHED_WEIGHTS": ["Qwen/Qwen3-8B", "b" * 40],
    }
    try:
        proc = run_bash(render_script("start_engine", template_dir=DIR, **values), env=w["env"])
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 0, proc.stderr
    download = read_env(w["logs"] / "download_env")
    assert "HF_TOKEN" not in download
    assert download["HF_HUB_OFFLINE"] == "1"
    assert "loom-stage sshd_ready" not in proc.stdout


def hf_cache_python(bin_dir: Path, log: Path) -> None:
    """A python3 that runs DOWNLOAD_PY against a fake HF cache under $HF_HOME, as
    huggingface_hub would: online (a token in its environment) it creates each pinned
    snapshot; with HF_HUB_OFFLINE=1 it fails on a snapshot that is not there. Each call
    appends `<online|offline> <repo> <revision> ...` to `log`."""
    stub(
        bin_dir,
        "python3",
        "shift 2  # -c DOWNLOAD_PY\n"
        'mode=online; [ "${HF_HUB_OFFLINE:-0}" = 1 ] && mode=offline\n'
        f'echo "$mode $*" >>{shlex.quote(str(log))}\n'
        "while [ $# -gt 0 ]; do\n"
        '  snap="$HF_HOME/hub/models--${1/\\//--}/snapshots/$2"\n'
        '  if [ "$mode" = offline ]; then\n'
        '    [ -d "$snap" ] || { echo "OfflineModeIsEnabled: $1@$2" >&2; exit 1; }\n'
        "  else\n"
        '    [ -n "${HF_TOKEN:-}" ] || { echo "no token" >&2; exit 1; }\n'
        '    mkdir -p "$snap"\n'
        "  fi\n"
        "  shift 2\n"
        "done\n",
    )


BF16 = ["Qwen/Qwen3-8B", "b" * 40]
FP8 = ["RedHatAI/Qwen3-8B-FP8-dynamic", "0" * 40]


def test_a_cold_then_warm_start_onto_another_checkpoint_downloads_it(tmp_path: Path) -> None:
    # runpod-smoke-h100 (4e50b5a5): the warm restart onto the FP8 cell checked its
    # checkpoint offline only and failed "weights for RedHatAI/Qwen3-8B-FP8-dynamic ...
    # are not cached". On one pod's cache: BF16 cold, FP8 warm (downloaded with the
    # token), then BF16 again (held: offline only).
    w = engine_env_world(tmp_path)
    log = tmp_path / "hf_calls"
    hf_cache_python(tmp_path / "bin", log)
    starts = [
        {"WARM": 0, "FETCH_WEIGHTS": BF16, "CACHED_WEIGHTS": []},
        {"WARM": 1, "FETCH_WEIGHTS": FP8, "CACHED_WEIGHTS": []},
        {"WARM": 1, "FETCH_WEIGHTS": [], "CACHED_WEIGHTS": BF16},
    ]
    for over in starts:
        try:
            proc = run_bash(
                render_script("start_engine", template_dir=DIR, **{**w["values"], **over}),
                env=w["env"],
            )
        finally:
            kill_engine(w["pidfile"])
        assert proc.returncode == 0, proc.stderr
        assert re.search(r"^loom-stage weights_ready ", proc.stdout, re.MULTILINE)
    assert log.read_text().splitlines() == [
        "online " + " ".join(BF16),
        "online " + " ".join(FP8),
        "offline " + " ".join(BF16),
    ]


def test_a_warm_start_whose_weights_are_not_held_fails_before_the_engine(tmp_path: Path) -> None:
    # What the smoke hit: the FP8 checkpoint offered as cached when no start downloaded it.
    w = engine_env_world(tmp_path)
    hf_cache_python(tmp_path / "bin", tmp_path / "hf_calls")
    values = {**w["values"], "WARM": 1, "FETCH_WEIGHTS": [], "CACHED_WEIGHTS": FP8}
    try:
        proc = run_bash(render_script("start_engine", template_dir=DIR, **values), env=w["env"])
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 1
    assert f"weights for {' '.join(FP8)} are not cached" in proc.stderr
    assert not (w["logs"] / "engine_args").exists()


def with_proc1_path(w: dict[str, Any], path: str) -> None:
    """Rewrite PATH in the world's fake /proc/1/environ."""
    proc1 = Path(w["values"]["PROC1_ENVIRON"])
    entries = [kv for kv in proc1.read_bytes().split(b"\0") if kv]
    env = dict(kv.decode().split("=", 1) for kv in entries)
    proc1_file(proc1, {**env, "PATH": path})


def test_start_engine_downloads_with_the_images_python(tmp_path: Path) -> None:
    # SGLang's Python is a venv on the image's PATH (/opt/sglang/bin). The SSH session's
    # default PATH misses it and finds a system python3 without huggingface_hub, which
    # is how the 2026-10-06 smoke test failed.
    w = engine_env_world(tmp_path)
    bin_dir = tmp_path / "bin"
    venv_bin = tmp_path / "opt-sglang-bin"
    venv_bin.mkdir()
    (bin_dir / "python3").rename(venv_bin / "python3")
    stub(
        bin_dir,
        "python3",
        "echo \"ModuleNotFoundError: No module named 'huggingface_hub'\" >&2\nexit 1\n",
    )
    with_proc1_path(w, f"{venv_bin}:{bin_dir}:/usr/bin:/bin")
    try:
        proc = run_bash(
            render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"]
        )
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 0, proc.stderr
    assert read_env(w["logs"] / "download_env")["HF_TOKEN"] == TOKEN
    assert re.search(r"^loom-stage weights_ready ", proc.stdout, re.MULTILINE)


def test_start_engine_downloads_a_speculative_draft_with_the_model(tmp_path: Path) -> None:
    # The engine runs offline (HF_HUB_OFFLINE=1), so a draft named in
    # --speculative-config must be in the cache before it starts: 2026-10-08's tuning
    # pod failed with "Cannot reach .../model.safetensors: offline mode is enabled".
    w = engine_env_world(tmp_path)
    draft = ["RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3", "c" * 40]
    values = {**w["values"], "FETCH_WEIGHTS": [*w["values"]["FETCH_WEIGHTS"], *draft]}
    try:
        proc = run_bash(render_script("start_engine", template_dir=DIR, **values), env=w["env"])
    finally:
        kill_engine(w["pidfile"])
    assert proc.returncode == 0, proc.stderr
    args = (w["logs"] / "download_args").read_text().split()
    assert args[-4:] == ["Qwen/Qwen3-8B", "b" * 40, *draft]


def test_start_engine_fails_when_the_image_has_no_python(tmp_path: Path) -> None:
    w = engine_env_world(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with_proc1_path(w, str(empty))
    proc = run_bash(render_script("start_engine", template_dir=DIR, **w["values"]), env=w["env"])
    assert proc.returncode == 1
    assert "python3 is not on the image's PATH" in proc.stderr
    assert not (w["logs"] / "download_env").exists()


# ---------------------------------------------------------------------------- stop_engine


def test_stop_engine_kills_the_process_group(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub(bin_dir, "nvidia-smi", "exit 0\n")
    victim = subprocess.Popen(["sleep", "60"], start_new_session=True)
    pidfile = tmp_path / "engine.pid"
    pidfile.write_text(str(victim.pid))
    proc = run_bash(
        render_script(
            "stop_engine", template_dir=DIR, ENGINE_PIDFILE=str(pidfile), STOP_TIMEOUT_S=10
        ),
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    assert victim.wait(timeout=10) != 0
    assert not pidfile.exists()
    assert "loom-stopped" in proc.stdout


# ---------------------------------------------------------------------------- run_job


def fake_python_tarball(logs: Path) -> bytes:
    """A python-build-standalone-shaped tarball whose `python3 -m venv DIR` makes a
    venv with stub `pip` and `bench` that log what they see."""
    bench = (
        "#!/bin/bash\n"
        f"env >{shlex.quote(str(logs / 'bench_env'))}\n"
        f'echo "$*" >{shlex.quote(str(logs / "bench_args"))}\n'
        'while [ $# -gt 0 ]; do [ "$1" = --out ] && echo \'{"ok": true}\' >"$2"; shift; done\n'
    )
    python = (
        "#!/bin/bash\n"
        '[ "$1 $2" = "-m venv" ] || exit 2\n'
        'mkdir -p "$3/bin"\n'
        f"cat >\"$3/bin/pip\" <<'EOF'\n#!/bin/bash\n"
        f'echo "pip $*" >>{shlex.quote(str(logs / "pip"))}\nEOF\n'
        f"cat >\"$3/bin/bench\" <<'EOF'\n{bench}EOF\n"
        'chmod 755 "$3/bin/pip" "$3/bin/bench"\n'
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = python.encode()
        info = tarfile.TarInfo("python/bin/python3")
        info.size, info.mode = len(data), 0o755
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


SHA256_CHECK_PY = """
import hashlib, sys
for line in sys.stdin:
    want, path = line.split(None, 1)
    if hashlib.sha256(open(path.strip(), "rb").read()).hexdigest() != want:
        sys.exit(1)
"""


def job_world(tmp_path: Path) -> dict[str, Any]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    logs = tmp_path / "logs"
    logs.mkdir()
    served = tmp_path / "served"
    served.mkdir()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    proc1 = proc1_file(tmp_path / "proc1_environ", {"HF_TOKEN": TOKEN, "RUNPOD_API_KEY": POD_KEY})
    setpriv_stub(bin_dir, logs / "setpriv", proc1)
    stub(bin_dir, "chown", "exit 0\n")
    # GNU `sha256sum --check --quiet` as in the pod's Ubuntu (macOS's lacks --quiet).
    stub(bin_dir, "sha256sum", f"exec python3 -c {shlex.quote(SHA256_CHECK_PY)}\n")
    stub(
        bin_dir,
        "install",
        'mode=""\n'
        "while [ $# -gt 1 ]; do\n"
        '  case "$1" in -m) mode="$2"; shift ;; -o|-g) shift ;; esac\n'
        "  shift\n"
        "done\n"
        'mkdir -p "$1"; [ -z "$mode" ] || chmod "$mode" "$1"\n',
    )
    stub(
        bin_dir,
        "curl",
        f'echo "$*" >>{shlex.quote(str(logs / "curl_argv"))}\n'
        'cfg=""; case " $* " in *" -K - "*) cfg="$(cat)" ;; esac\n'
        'url="$(printf \'%s\\n\' "$cfg" | sed -n \'s/^url = "\\(.*\\)"$/\\1/p\')"\n'
        'up="$(printf \'%s\\n\' "$cfg" | sed -n \'s/^upload-file = "\\(.*\\)"$/\\1/p\')"\n'
        'out=""; while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
        'key="${url%%\\?*}"; key="${key##*/}"\n'
        f'if [ -n "$up" ]; then cp "$up" {shlex.quote(str(uploads))}/"$key"; exit 0; fi\n'
        f'[ -f {shlex.quote(str(served))}/"$key" ] || exit 22\n'
        f'cp {shlex.quote(str(served))}/"$key" "$out"\n',
    )
    stub(
        bin_dir,
        "nvidia-smi",
        'echo "2026/10/06 07:05:00.000, 0, 97, 40000, 46068, 300.5"\nexec sleep 30\n',
    )
    wheel = b"fake wheel"
    reqs = b"pkg==1.0 --hash=sha256:00\n"
    tarball = fake_python_tarball(logs)
    (served / "job.json").write_text('{"run_id": "r1"}')
    (served / JOB_VARS["WHEEL_NAME"]).write_bytes(wheel)
    (served / "w.whl").write_bytes(wheel)
    (served / "requirements.txt").write_bytes(reqs)
    (served / "cpython.tar.gz").write_bytes(tarball)
    lingering = subprocess.Popen(["sleep", "60"])
    fake_proc = tmp_path / "proc"
    (fake_proc / str(lingering.pid)).mkdir(parents=True)
    values = {
        **JOB_VARS,
        "WORK_DIR": str(tmp_path / "work" / "r1"),
        "ENV_ROOT": str(tmp_path / "clientenv"),
        "PYTHON_DIR": str(tmp_path / "python"),
        "PYTHON_SHA256": hashlib.sha256(tarball).hexdigest(),
        "WHEEL_SHA256": hashlib.sha256(wheel).hexdigest(),
        "REQS_SHA256": hashlib.sha256(reqs).hexdigest(),
        "PROC1_ENVIRON": str(proc1),
        "JOB_UID": os.getuid(),
        "JOB_HOME": str(tmp_path / "home"),
        "PROC_ROOT": str(fake_proc),
    }
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin:/sbin",
        "HOME": str(tmp_path),
        # The SSH session's own environment may hold anything; none of it reaches the job.
        "HF_TOKEN": TOKEN,
        "RUNPOD_API_KEY": POD_KEY,
        "AWS_SECRET_ACCESS_KEY": "awsawsawsaws",
    }
    return {
        "values": values,
        "env": env,
        "logs": logs,
        "uploads": uploads,
        "lingering": lingering,
    }


def test_run_job_isolates_the_job_and_round_trips_results(tmp_path: Path) -> None:
    w = job_world(tmp_path)
    values = {
        **w["values"],
        "SAMPLE_GPU": 1,
        "GPU_CSV_URL": "https://b.s3.amazonaws.com/gpu.csv?X-Amz-Signature=5",
    }
    try:
        proc = run_bash(render_script("run_job", template_dir=DIR, **values), env=w["env"])
    finally:
        w["lingering"].kill()
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[0] == "loom-sys job_isolation ok"
    assert proc.stdout.splitlines()[-1] == "loom-job-done"
    bench_env = read_env(w["logs"] / "bench_env")
    assert set(bench_env) - {"PWD", "SHLVL", "_", "OLDPWD"} == {"HOME", "PATH", "LANG"}
    assert bench_env["HOME"] == values["JOB_HOME"]
    assert bench_env["PATH"] == "/usr/local/bin:/usr/bin:/bin"
    assert (w["logs"] / "bench_args").read_text().split() == [
        "job",
        "run",
        "--in",
        f"{values['WORK_DIR']}/job.json",
        "--out",
        f"{values['WORK_DIR']}/result.json",
    ]
    setpriv = (w["logs"] / "setpriv").read_text().splitlines()
    uid = os.getuid()
    expected = f"setpriv --reuid={uid} --regid={uid} --clear-groups --no-new-privs env -i "
    assert setpriv and all(line.startswith(expected) for line in setpriv)
    pip = (w["logs"] / "pip").read_text()
    assert f"--require-hashes --no-deps -r {values['WORK_DIR']}/requirements.txt" in pip
    assert "pip check" in pip
    assert json.loads((w["uploads"] / "result.json").read_text()) == {"ok": True}
    assert (w["uploads"] / "gpu.csv").read_text().startswith("2026/10/06")
    # URLs went to curl on stdin only.
    curl_argv = (w["logs"] / "curl_argv").read_text()
    assert "Signature" not in curl_argv and "https://" not in curl_argv
    # Downloads follow HTTPS-only redirects: the client Python's GitHub release URL
    # answers 302, and without -L curl saved the empty redirect body (smoke test).
    fetches = [line for line in curl_argv.splitlines() if " -o /dev/null" not in line]
    assert fetches and all("-L --proto =https --proto-redir =https" in line for line in fetches)
    # A process the job user left behind was killed.
    assert w["lingering"].wait(timeout=5) == -signal.SIGKILL
    env_dir = Path(values["ENV_ROOT"]) / f"{values['WHEEL_SHA256']}-{values['REQS_SHA256']}"
    assert not os.stat(env_dir / "bin" / "bench").st_mode & 0o022


def test_run_job_reuses_the_env_and_python(tmp_path: Path) -> None:
    w = job_world(tmp_path)
    try:
        for _ in range(2):
            proc = run_bash(render_script("run_job", template_dir=DIR, **w["values"]), env=w["env"])
            assert proc.returncode == 0, proc.stderr
    finally:
        w["lingering"].kill()
    curl_argv = (w["logs"] / "curl_argv").read_text().splitlines()
    assert len(curl_argv) == 3 + 1 + 1 + 3 + 1  # inputs, python once, result, inputs, result
    assert (w["logs"] / "pip").read_text().count("pip check") == 1


def test_run_job_fails_before_the_job_when_proc1_is_readable(tmp_path: Path) -> None:
    w = job_world(tmp_path)
    try:
        proc = run_bash(
            render_script("run_job", template_dir=DIR, **w["values"]),
            env={**w["env"], "STUB_JOB_READS_PROC1": "1"},
        )
    finally:
        w["lingering"].kill()
    assert proc.returncode == 1
    assert "loom-error job user can read" in proc.stderr
    assert not (w["logs"] / "bench_args").exists()
    assert not (w["logs"] / "curl_argv").exists()


def test_run_job_checks_checksums(tmp_path: Path) -> None:
    w = job_world(tmp_path)
    try:
        proc = run_bash(
            render_script(
                "run_job", template_dir=DIR, **{**w["values"], "PYTHON_SHA256": "0" * 64}
            ),
            env=w["env"],
        )
    finally:
        w["lingering"].kill()
    assert proc.returncode == 1
    assert "client Python checksum mismatch" in proc.stderr
    assert not (w["logs"] / "bench_args").exists()


def test_run_job_needs_the_tokenizer_snapshot(tmp_path: Path) -> None:
    w = job_world(tmp_path)
    values = {
        **w["values"],
        "MODEL_CACHE_DIR": str(tmp_path / "hub" / "models--Qwen--Qwen3-8B"),
        "MODEL_REVISION": "b" * 40,
    }
    try:
        proc = run_bash(render_script("run_job", template_dir=DIR, **values), env=w["env"])
        assert proc.returncode == 1 and "no tokenizer.json in snapshot" in proc.stderr
        snap = Path(values["MODEL_CACHE_DIR"]) / "snapshots" / ("b" * 40)
        snap.mkdir(parents=True)
        (snap / "tokenizer.json").write_text("{}")
        proc = run_bash(render_script("run_job", template_dir=DIR, **values), env=w["env"])
        assert proc.returncode == 0, proc.stderr
    finally:
        w["lingering"].kill()


def test_engine_process_argv_rewrites_only_the_host() -> None:
    launch = render_launch(load_registry().get("qwen3-8b"))
    argv = engine_process_argv(launch, host="127.0.0.1")
    assert argv[:2] == ["vllm", "serve"]
    assert argv[2:] == [("127.0.0.1" if a == "0.0.0.0" else a) for a in launch.args]
    assert launch.args.count("0.0.0.0") == 1  # the launch (and its config hash) is unchanged
    bad = launch.model_copy(
        update={"args": [a for a in launch.args if a not in ("--host", "0.0.0.0")]}
    )
    with pytest.raises(ValueError, match="--host"):
        engine_process_argv(bad, host="127.0.0.1")
    with pytest.raises(ValueError, match="secrets"):
        engine_process_argv(launch.model_copy(update={"env": {"HF_TOKEN": "x"}}), host="127.0.0.1")
    with pytest.raises(ValueError, match="entrypoint"):
        engine_process_argv(launch.model_copy(update={"engine": "mock"}), host="127.0.0.1")


def test_layout_renders_every_script() -> None:
    from loom_bench.providers import runpod_layout as L

    render_script(
        "pod_start",
        template_dir=DIR,
        TTL_EPOCH=1_790_000_000,
        STAGE_FILE=L.STAGE_FILE,
        LOOM_ROOT=L.LOOM_ROOT,
        SSH_DIR=L.SSH_DIR,
        SSHD_CONFIG=L.SSHD_CONFIG,
        CTL_ROOT=L.CTL_ROOT,
        JOB_UID=L.JOB_UID,
        JOB_USER=L.JOB_USER,
        JOB_HOME=L.JOB_HOME,
        AUTHORIZE_ACCOUNT_KEY=1,
        GRAPHQL_URL="https://api.runpod.io/graphql",
    )
    assert L.WEIGHTS_DIR.startswith("/opt/loom/")  # inside a scanned dir, excluded by name
    assert any(L.WEIGHTS_DIR.startswith(d + "/") for d in L.SECRET_SCAN_DIRS)
    assert L.CTL_ROOT.startswith(L.LOOM_ROOT + "/")
    assert re.fullmatch(r"[0-9a-f]{64}", L.CLIENT_PYTHON_SHA256)
    assert L.CLIENT_PYTHON_URL.startswith(
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
    )
    assert "x86_64-unknown-linux-gnu-install_only" in L.CLIENT_PYTHON_URL
    assert '"' not in L.CLIENT_PYTHON_URL and "\\" not in L.CLIENT_PYTHON_URL
