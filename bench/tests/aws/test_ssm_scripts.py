import hashlib
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from loom_bench.providers.aws_ec2 import CLIENT_IMAGE
from loom_bench.providers.aws_ssm import (
    SsmCommandError,
    parse_markers,
    render_script,
    required_vars,
    run_script,
    ssm_commands,
    stage_offsets,
)

from .fakes import FakeSsm, no_sleep

BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not installed")

START_VARS: dict[str, Any] = {
    "WARM": 0,
    "REGION": "us-east-1",
    "HF_SECRET_ID": "loom/hf-token",
    "IMAGE": "vllm/vllm-openai@sha256:" + "a" * 64,
    "FETCH_WEIGHTS": ["Qwen/Qwen3-8B", "b" * 40],
    "CACHED_WEIGHTS": [],
    "WEIGHTS_DIR": "/opt/dlami/nvme/loom-hf",
    "CONTAINER": "loom-engine",
    "PORT": 8000,
    "SERVED_MODEL": "qwen3-8b",
    "READY_TIMEOUT_S": 1800,
    "LOG_DIR": "/var/log/loom",
    "STAGE_FILE": "/var/lib/loom/stages",
    "ENGINE_ENV": ["VLLM_LOGGING_LEVEL=INFO"],
    "ENGINE_CMD": ["docker", "run", "--name", "loom-engine", "img", "--x", "a b"],
}
JOB_VARS: dict[str, Any] = {
    "WORK_DIR": "/var/lib/loom/jobs/r1",
    "ENV_ROOT": "/var/lib/loom/clientenv",
    "CLIENT_IMAGE": CLIENT_IMAGE,
    "CLIENT_UID": 10001,
    "JOB_URL": "https://b.s3.amazonaws.com/runs/e/r1/job.json?X-Amz-Signature=1&b=2",
    "WHEEL_URL": "https://b.s3.amazonaws.com/w.whl?sig=1",
    "WHEEL_NAME": "loom_bench-0.1.0-py3-none-any.whl",
    "WHEEL_SHA256": "c" * 64,
    "REQS_URL": "https://b.s3.amazonaws.com/requirements.txt?sig=3",
    "REQS_SHA256": "d" * 64,
    "BENCH_CMD": ["job", "run"],
    "RESULT_URL": "https://b.s3.amazonaws.com/result.json?sig=2",
    "GPU_CSV_URL": "",
    "SAMPLE_GPU": 0,
    "MODEL_CACHE_DIR": "",
    "MODEL_CACHE_MOUNT": "",
    "MODEL_FOLDER": "",
    "MODEL_REVISION": "",
    "DATA_URL": "",
    "DATA_SHA256": "",
    "DATA_PATH": "",
    "DATA_DIR": "",
    "DATA_MOUNT": "",
}
RENDERED = {
    "start_engine": START_VARS,
    "run_job": JOB_VARS,
    "stop_engine": {"CONTAINER": "loom-engine"},
    "user_data": {
        "TTL_EPOCH": 1_790_000_000,
        "STAGE_FILE": "/var/lib/loom/stages",
        "CLIENT_UID": 1,
    },
}


def test_required_vars_match_templates() -> None:
    assert required_vars("stop_engine") == {"CONTAINER"}
    assert required_vars("start_engine") == set(START_VARS)


def test_render_rejects_missing_and_extra() -> None:
    with pytest.raises(ValueError, match="missing \\['CONTAINER'\\]"):
        render_script("stop_engine")
    with pytest.raises(ValueError, match="unexpected \\['EXTRA'\\]"):
        render_script("stop_engine", CONTAINER="c", EXTRA="x")
    with pytest.raises(TypeError):
        render_script("stop_engine", CONTAINER=True)


@needs_bash
@pytest.mark.parametrize("name", sorted(RENDERED))
def test_rendered_scripts_are_valid_bash(name: str) -> None:
    script = render_script(name, **RENDERED[name])
    assert script.startswith("#!/bin/bash\nset -euo pipefail\n")
    assert not re.search(r"^\s*set -[a-z]*x|xtrace", script, re.MULTILINE)
    subprocess.run([BASH, "-n"], input=script, text=True, check=True)


@needs_bash
def test_values_are_quoted_not_interpreted() -> None:
    hostile = 'x\'; echo pwned; $(id) `id` "q"'
    script = render_script(
        "start_engine", **{**START_VARS, "CONTAINER": hostile, "ENGINE_CMD": [hostile, "b c"]}
    )
    header = script.split("\n\n", 1)[0]
    out = subprocess.run(
        [BASH, "-c", header + '\nprintf "%s|" "$CONTAINER" "${ENGINE_CMD[@]}"'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert out == f"{hostile}|{hostile}|b c|"
    assert f"CONTAINER={shlex.quote(hostile)}" in script


def test_start_script_never_holds_a_token_value() -> None:
    script = render_script("start_engine", **START_VARS)
    assert "HF_SECRET_ID=loom/hf-token" in script
    assert 'HF_TOKEN="$(aws secretsmanager get-secret-value' in script
    assert "--env HF_TOKEN " in script
    assert "unset HF_TOKEN" in script


def test_job_script_gpu_sampling_and_isolation() -> None:
    script = render_script("run_job", **{**JOB_VARS, "SAMPLE_GPU": 1, "GPU_CSV_URL": "https://g"})
    assert (
        "nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,"
        "power.draw \\\n    --format=csv,noheader,nounits -lms 1000"
    ) in script
    assert '--network host --user "$CLIENT_UID:$CLIENT_UID"' in script
    assert '/env/bin/bench "${BENCH_CMD[@]}" --in /work/job.json --out /work/result.json' in script
    assert "BENCH_CMD=(job run)" in script
    assert '--volume "$ENV_DIR:/env:ro"' in script
    assert "JOB_URL='https://b.s3.amazonaws.com/runs/e/r1/job.json?X-Amz-Signature=1&b=2'" in script


@needs_bash
def test_job_script_installs_only_locked_hash_pinned_dependencies(tmp_path: Any) -> None:
    script = render_script("run_job", **{**JOB_VARS, "BENCH_CMD": ["quality", "job"]})
    assert "BENCH_CMD=(quality job)" in script
    header = script.split("\n\n", 1)[0]
    probe = 'printf "%s|%s" "$ENV_ROOT/$WHEEL_SHA256-$REQS_SHA256" "${BENCH_CMD[*]}"'
    out = subprocess.run(
        [BASH, "-c", header + "\n" + probe], capture_output=True, text=True, check=True
    ).stdout
    # one virtualenv per wheel and requirements file (an extra changes the file)
    assert out == f"/var/lib/loom/clientenv/{'c' * 64}-{'d' * 64}|quality job"
    assert 'echo "$REQS_SHA256  $WORK_DIR/requirements.txt" | sha256sum --check' in script
    assert "--require-hashes --no-deps -r /work/requirements.txt" in script
    assert '--no-cache-dir --no-deps "/work/$1"' in script
    assert "/env/bin/pip check" in script
    assert "PIP_EXTRAS" not in script


@needs_bash
def test_job_installs_the_env_in_the_pinned_image(tmp_path: Any) -> None:
    proc, calls = run_job_on_stubbed_host(tmp_path)
    assert proc.returncode == 0, proc.stderr
    install, _ = calls
    assert f"|{CLIENT_IMAGE}|sh|-c|python -m venv /env && " in install
    assert "--require-hashes --no-deps -r /work/requirements.txt && " in install
    assert install.endswith(f" && /env/bin/pip check|sh|{JOB_VARS['WHEEL_NAME']}|")


STUBS = {
    "curl": 'out=""\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
    '[ -z "$out" ] || : >"$out"',
    "sha256sum": "cat >/dev/null",
    "install": 'for a; do last="$a"; done\nmkdir -p "$last"',
    "chown": "true",
    "docker": 'printf \'%s|\' "$@" >>"$DOCKER_LOG"\necho >>"$DOCKER_LOG"',
}


def run_job_on_stubbed_host(
    tmp_path: Any, stubs: dict[str, str] | None = None, **over: Any
) -> tuple[Any, list[str]]:
    """Run the rendered run_job.sh with curl, docker and friends stubbed; return the
    process and one `|`-joined argv per docker call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (stubs or STUBS).items():
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "docker.log"
    variables = {
        **JOB_VARS,
        "WORK_DIR": str(tmp_path / "jobs" / "r1"),
        "ENV_ROOT": str(tmp_path / "env"),
        **over,
    }
    proc = subprocess.run(
        [BASH],
        input=render_script("run_job", **variables),
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "DOCKER_LOG": str(log)},
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return proc, calls


QWEN = "models--Qwen--Qwen3-8B"


def hf_hub_cache(root: Path, rev: str, *, folder: str = QWEN) -> Path:
    """An HF hub cache as the vLLM image's huggingface_hub lays it out on the GPU host
    (seen on the first AWS smoke, 2026-10-10): small files are blobs in the model's
    folder, and large ones (tokenizer.json, safetensors) link on into a hub-level store,
    hub/blobs/<xx>/<sha256>. Returns the hub directory."""
    hub = root / "hf" / "hub"
    snap = hub / folder / "snapshots" / rev
    snap.mkdir(parents=True)
    blobs = hub / folder / "blobs"
    blobs.mkdir()
    (blobs / "c0ffee").write_text("{}")
    (snap / "config.json").symlink_to("../../blobs/c0ffee")
    sha = "6aec" + "0" * 60
    (hub / "blobs" / "6a").mkdir(parents=True)
    (hub / "blobs" / "6a" / sha).write_text("{}")
    (blobs / "aeb1").symlink_to(f"../../blobs/6a/{sha}")
    (snap / "tokenizer.json").symlink_to("../../blobs/aeb1")
    return hub


@needs_bash
def test_job_mounts_the_hub_cache_read_only(tmp_path: Path) -> None:
    rev = "b" * 40
    hub = hf_hub_cache(tmp_path, rev)
    proc, calls = run_job_on_stubbed_host(
        tmp_path,
        MODEL_CACHE_DIR=str(hub),
        MODEL_CACHE_MOUNT="/models",
        MODEL_FOLDER=QWEN,
        MODEL_REVISION=rev,
    )
    assert proc.returncode == 0, proc.stderr
    bench = calls[-1]
    # The whole hub cache: tokenizer.json's content lives in hub/blobs/, outside the
    # model's folder, so mounting only the folder left it dangling in the container.
    assert f"|--volume|{hub}:/models:ro|{CLIENT_IMAGE}|/env/bin/bench|job|run|" in bench
    assert "HF_TOKEN" not in bench


@needs_bash
@pytest.mark.parametrize("bad", ["outside", "dangling"])
def test_job_fails_before_running_when_a_snapshot_link_leaves_the_mount(
    tmp_path: Path, bad: str
) -> None:
    rev = "b" * 40
    hub = hf_hub_cache(tmp_path, rev)
    snap = hub / QWEN / "snapshots" / rev
    if bad == "outside":
        (tmp_path / "elsewhere.json").write_text("{}")
        (snap / "vocab.json").symlink_to(tmp_path / "elsewhere.json")
        match = "resolves outside"
    else:
        (snap / "vocab.json").symlink_to("../../blobs/missing")
        match = "dangling link"
    proc, calls = run_job_on_stubbed_host(
        tmp_path,
        MODEL_CACHE_DIR=str(hub),
        MODEL_CACHE_MOUNT="/models",
        MODEL_FOLDER=QWEN,
        MODEL_REVISION=rev,
    )
    assert proc.returncode == 1
    assert "loom-error snapshot file" in proc.stderr and match in proc.stderr, proc.stderr
    assert calls == []


@needs_bash
def test_a_model_folder_mount_fails_here_not_in_the_container(tmp_path: Path) -> None:
    # The first AWS smoke's bug: the mount was the model's folder, whose tokenizer.json
    # resolves into hub/blobs/. The check refuses such a mount before any container.
    rev = "b" * 40
    hub = hf_hub_cache(tmp_path, rev)
    proc, calls = run_job_on_stubbed_host(
        tmp_path,
        MODEL_CACHE_DIR=str(hub / QWEN),
        MODEL_CACHE_MOUNT="/models/" + QWEN,
        MODEL_FOLDER=".",
        MODEL_REVISION=rev,
    )
    assert proc.returncode == 1
    assert "tokenizer.json resolves outside" in proc.stderr, proc.stderr
    assert calls == []


@needs_bash
def test_job_fails_before_running_without_the_pinned_snapshot(tmp_path: Path) -> None:
    hub = hf_hub_cache(tmp_path, "a" * 40)  # another revision only
    proc, calls = run_job_on_stubbed_host(
        tmp_path,
        MODEL_CACHE_DIR=str(hub),
        MODEL_CACHE_MOUNT="/models",
        MODEL_FOLDER=QWEN,
        MODEL_REVISION="b" * 40,
    )
    assert proc.returncode == 1
    assert f"loom-error no tokenizer.json in snapshot {'b' * 40}" in proc.stderr
    assert calls == []


@needs_bash
def test_job_without_an_hf_tokenizer_mounts_no_model(tmp_path: Any) -> None:
    proc, calls = run_job_on_stubbed_host(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert f"|--volume|{tmp_path}/jobs/r1:/work|{CLIENT_IMAGE}|/env/bin/bench|" in calls[-1]
    assert f":ro|{CLIENT_IMAGE}" not in calls[-1]


# GNU `sha256sum --check --quiet`, as on the DLAMI's Ubuntu (macOS's lacks --quiet), for
# the dataset; the other inputs are empty stand-ins, so their checks pass as in STUBS.
SHA256_CHECK_PY = """
import hashlib, sys
for line in sys.stdin:
    want, path = line.split(None, 1)
    if "convs.json" not in path:
        continue
    if hashlib.sha256(open(path.strip(), "rb").read()).hexdigest() != want:
        sys.exit(1)
"""


def _dataset_host(tmp_path: Any, body: bytes, **over: Any) -> tuple[Any, list[str], dict[str, Any]]:
    """run_job.sh with a pinned dataset: curl serves `body` for the dataset URL (and
    logs every fetch), sha256sum really checks. Returns (process, docker calls, vars)."""
    served = tmp_path / "served"
    served.mkdir(exist_ok=True)
    (served / "convs.json").write_bytes(body)
    fetches = tmp_path / "fetches.log"
    stubs = {
        **STUBS,
        "curl": f'echo "$*" >>{shlex.quote(str(fetches))}\n'
        'out=""; url=""\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in -o) out="$2"; shift ;; https://*) url="$1" ;; esac; shift\n'
        "done\n"
        '[ -n "$out" ] || exit 0\n'
        'case "$url" in */convs.json) cp '
        + shlex.quote(str(served))
        + '/convs.json "$out" ;; *) : >"$out" ;; esac\n',
        "sha256sum": f"exec python3 -c {shlex.quote(SHA256_CHECK_PY)}",
    }
    data_dir = tmp_path / "nvme" / "loom-data"
    sha = hashlib.sha256(b"conversations").hexdigest()
    values = {
        "DATA_URL": f"https://huggingface.co/datasets/o/d/resolve/{'a' * 40}/convs.json",
        "DATA_SHA256": sha,
        "DATA_PATH": str(data_dir / sha / "convs.json"),
        "DATA_DIR": str(data_dir),
        "DATA_MOUNT": "/data",
        **over,
    }
    proc, calls = run_job_on_stubbed_host(tmp_path, stubs=stubs, **values)
    return proc, calls, values


@needs_bash
def test_job_fetches_a_pinned_dataset_once_and_mounts_it_read_only(tmp_path: Any) -> None:
    for _ in range(2):
        proc, calls, values = _dataset_host(tmp_path, b"conversations")
        assert proc.returncode == 0, proc.stderr
        assert (
            f"|--volume|{values['DATA_DIR']}:/data:ro|{CLIENT_IMAGE}|/env/bin/bench|" in calls[-1]
        )
    data = Path(values["DATA_PATH"])
    assert data.read_bytes() == b"conversations"
    # Root-owned on the host and readable by the client uid, which only reads it.
    assert data.stat().st_mode & 0o777 == 0o644
    assert data.parent.stat().st_mode & 0o777 == 0o755
    assert Path(values["DATA_DIR"]).stat().st_mode & 0o777 == 0o755
    assert not data.with_name(data.name + ".part").exists()
    fetched = (tmp_path / "fetches.log").read_text().splitlines()
    dataset = [line for line in fetched if "convs.json" in line]
    assert len(dataset) == 1  # the second job reuses it
    assert "--proto =https --proto-redir =https" in dataset[0] and " -L " in dataset[0]


@needs_bash
def test_job_refuses_a_dataset_with_the_wrong_checksum(tmp_path: Any) -> None:
    proc, calls, values = _dataset_host(tmp_path, b"something else")
    assert proc.returncode == 1
    assert "workload dataset checksum mismatch" in proc.stderr
    data = Path(values["DATA_PATH"])
    assert not data.exists() and not data.with_name(data.name + ".part").exists()
    assert not [c for c in calls if "/env/bin/bench" in c]


@needs_bash
def test_job_refuses_a_dataset_path_outside_the_data_dir(tmp_path: Any) -> None:
    proc, calls, _ = _dataset_host(tmp_path, b"conversations", DATA_PATH="/etc/passwd")
    assert proc.returncode == 1
    assert "is not under" in proc.stderr
    assert calls == []


def test_user_data_arms_ttl_shutdown() -> None:
    script = render_script("user_data", **RENDERED["user_data"])
    assert "TTL_EPOCH=1790000000" in script
    assert 'shutdown -h "+$remaining_min"' in script
    assert "iptables -I OUTPUT -d 169.254.169.254 -m owner --uid-owner" in script


@needs_bash
def test_ssm_commands_run_under_bash_and_propagate_exit(tmp_path: Any) -> None:
    script = "#!/bin/bash\nset -euo pipefail\narr=(a 'b c')\necho \"${arr[1]}\"\nexit 3\n"
    wrapper = "\n".join(ssm_commands(script))
    proc = subprocess.run(["/bin/sh", "-c", wrapper], capture_output=True, text=True)
    assert proc.returncode == 3
    assert proc.stdout == "b c\n"


def test_parse_markers_and_offsets() -> None:
    stdout = "\n".join(
        [
            "loom-stage user_data_started 100.5",
            "noise line",
            "loom-stage image_pulled 160.25",
            "loom-stage bad notanumber",
            "loom-sys gpus NVIDIA L40S,NVIDIA L40S",
            "loom-sys driver 595.91.07",
            "loom-stage image_pulled 161.0",
        ]
    )
    stages, system = parse_markers(stdout)
    assert stages == {"user_data_started": 100.5, "image_pulled": 161.0}
    assert system == {"gpus": "NVIDIA L40S,NVIDIA L40S", "driver": "595.91.07"}
    assert stage_offsets(stages, 100.0) == {"user_data_started": 0.5, "image_pulled": 61.0}


async def _ok() -> None:
    return None


async def test_run_script_polls_until_success() -> None:
    ssm = FakeSsm(
        lambda s: [
            {"Status": "Pending"},
            {"Status": "InProgress"},
            {"Status": "Success", "StandardOutputContent": "loom-stage x 1.0\n"},
        ]
    )
    checks = 0

    async def check() -> None:
        nonlocal checks
        checks += 1

    out = await run_script(
        ssm,
        "i-1",
        "#!/bin/bash\necho hi\n",
        timeout_s=60,
        comment="c" * 200,
        poll_s=0,
        check_host=check,
        output_bucket="bkt",
        output_prefix="ssm/e/i-1",
        sleep=no_sleep,
    )
    assert out == "loom-stage x 1.0\n"
    sent = ssm.sent[0]
    assert sent["DocumentName"] == "AWS-RunShellScript"
    assert sent["Parameters"]["executionTimeout"] == ["60"]
    assert len(sent["Comment"]) == 100
    assert sent["OutputS3BucketName"] == "bkt"
    assert checks == 3


async def test_run_script_failure_raises_with_stderr() -> None:
    ssm = FakeSsm(lambda s: [{"Status": "Failed", "StandardErrorContent": "loom-error boom"}])
    with pytest.raises(SsmCommandError, match="boom") as e:
        await run_script(
            ssm, "i-1", "x", timeout_s=5, comment="c", poll_s=0, check_host=_ok, sleep=no_sleep
        )
    assert e.value.status == "Failed"


async def test_run_script_host_check_wins_over_command_failure() -> None:
    ssm = FakeSsm(lambda s: [{"Status": "Failed"}])

    async def gone() -> None:
        raise RuntimeError("host gone")

    with pytest.raises(RuntimeError, match="host gone"):
        await run_script(
            ssm, "i-1", "x", timeout_s=5, comment="c", poll_s=0, check_host=gone, sleep=no_sleep
        )
