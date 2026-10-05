import re
import shlex
import shutil
import subprocess
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
    "MODEL_REPO": "Qwen/Qwen3-8B",
    "MODEL_REVISION": "b" * 40,
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
    "MODEL_REVISION": "",
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


def run_job_on_stubbed_host(tmp_path: Any, **over: Any) -> tuple[Any, list[str]]:
    """Run the rendered run_job.sh with curl, docker and friends stubbed; return the
    process and one `|`-joined argv per docker call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in STUBS.items():
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


@needs_bash
def test_job_mounts_the_model_snapshot_read_only(tmp_path: Any) -> None:
    rev = "b" * 40
    cache = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3-8B"
    (cache / "snapshots" / rev).mkdir(parents=True)
    (cache / "snapshots" / rev / "tokenizer.json").write_text("{}")
    mount = "/models/models--Qwen--Qwen3-8B"
    proc, calls = run_job_on_stubbed_host(
        tmp_path, MODEL_CACHE_DIR=str(cache), MODEL_CACHE_MOUNT=mount, MODEL_REVISION=rev
    )
    assert proc.returncode == 0, proc.stderr
    bench = calls[-1]
    assert f"|--volume|{cache}:{mount}:ro|{CLIENT_IMAGE}|/env/bin/bench|job|run|" in bench
    assert "HF_TOKEN" not in bench


@needs_bash
def test_job_fails_before_running_without_the_pinned_snapshot(tmp_path: Any) -> None:
    cache = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3-8B"
    (cache / "snapshots" / ("a" * 40)).mkdir(parents=True)  # another revision only
    (cache / "snapshots" / ("a" * 40) / "tokenizer.json").write_text("{}")
    proc, calls = run_job_on_stubbed_host(
        tmp_path,
        MODEL_CACHE_DIR=str(cache),
        MODEL_CACHE_MOUNT="/models/models--Qwen--Qwen3-8B",
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
