"""RunPod trace collection: the provider's S3 round trip and collect_profile.sh itself."""

import io
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import boto3
import pytest

from loom_bench.providers.aws_ssm import render_script, required_vars

from .conftest import BUCKET, REGION
from .fakes import FakePodExec, FakeRunpod, ok
from .test_runpod_provider import provider, request, s3_key, script_var


@pytest.fixture
def s3(moto_aws: None) -> Any:
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    return client


@pytest.fixture
def fake() -> FakeRunpod:
    return FakeRunpod(gets_before_ssh=0)


BASH = shutil.which("bash")
TRACE = b"trace-archive-bytes"


def pod_uploads(s3: Any, body: bytes = TRACE, *, done: bool = True) -> Any:
    def handler(target: Any, script: str) -> Any:
        s3.put_object(Bucket=BUCKET, Key=s3_key(script_var(script, "UPLOAD_URL")), Body=body)
        return ok("loom-sys profile_bytes 19\n" + ("loom-profile-done\n" if done else ""))

    return handler


async def test_collect_profile_round_trip_through_s3(fake, s3, tmp_path) -> None:
    ssh = FakePodExec({"collect_profile": pod_uploads(s3)})
    p = provider(fake, s3, tmp_path, ssh=ssh)
    host = await p.provision(request())
    got = await p.collect_profile(host, "run-9", "/var/lib/loom/profiles", 1 << 20)
    assert got == TRACE
    (script,) = ssh.scripts("collect_profile")
    assert script_var(script, "TRACE_DIR") == "/var/lib/loom/profiles"
    assert script_var(script, "MAX_BYTES") == str(1 << 20)
    url = script_var(script, "UPLOAD_URL")
    assert url.startswith("https://") and "Signature" in url
    assert s3_key(url) == f"runs/{host.request.tags['loom:experiment']}/run-9/trace.tar.gz"


async def test_collect_profile_refuses_odd_paths_and_unfinished_scripts(fake, s3, tmp_path) -> None:
    p = provider(
        fake, s3, tmp_path, ssh=FakePodExec({"collect_profile": pod_uploads(s3, done=False)})
    )
    host = await p.provision(request())
    for bad in ("relative/dir", "/var/lib/../etc", "/tmp/$(id)"):
        with pytest.raises(ValueError, match="trace_dir"):
            await p.collect_profile(host, "run-9", bad, 1 << 20)
    with pytest.raises(RuntimeError, match="did not finish"):
        await p.collect_profile(host, "run-9", "/var/lib/loom/profiles", 1 << 20)


# ---------------------------------------------------------------- the script itself

needs_bash = pytest.mark.skipif(BASH is None, reason="bash not installed")


def _world(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bins = tmp_path / "bin"
    bins.mkdir()
    dest = tmp_path / "uploaded.tar.gz"
    # curl reads its config (url, upload-file) on stdin: copy the upload where S3 would be.
    (bins / "curl").write_text(
        "#!/bin/bash\n"
        'cfg="$(cat)"\n'
        'src="$(printf \'%s\\n\' "$cfg" | sed -n \'s/^upload-file = "\\(.*\\)"$/\\1/p\')"\n'
        f'cp "$src" {dest}\n'
    )
    (bins / "curl").chmod(0o755)
    env = {"PATH": f"{bins}:/usr/bin:/bin", "HOME": str(tmp_path)}
    traces = tmp_path / "profiles"
    return env, traces, dest


def _run(
    trace_dir: Path, env: dict[str, str], max_bytes: int = 1 << 20
) -> subprocess.CompletedProcess[str]:
    script = render_script(
        "collect_profile",
        template_dir="runpod_scripts",
        TRACE_DIR=str(trace_dir),
        MAX_BYTES=max_bytes,
        UPLOAD_URL="https://b.s3.amazonaws.com/runs/e/r/trace.tar.gz?X-Amz-Signature=1",
    )
    assert BASH is not None
    return subprocess.run(
        [BASH, "-s"], input=script, env=env, capture_output=True, text=True, timeout=60
    )


def test_required_vars() -> None:
    assert required_vars("collect_profile", template_dir="runpod_scripts") == {
        "TRACE_DIR",
        "MAX_BYTES",
        "UPLOAD_URL",
    }


@needs_bash
def test_script_archives_uploads_and_clears_the_trace_dir(tmp_path: Path) -> None:
    env, traces, dest = _world(tmp_path)
    (traces / "sub").mkdir(parents=True)
    (traces / "loom-r_rank0.1.pt.trace.json.gz").write_bytes(b"a" * 100)
    (traces / "sub" / "loom-r_rank1.1.pt.trace.json.gz").write_bytes(b"b" * 100)
    res = _run(traces, env)
    assert res.returncode == 0, res.stderr
    assert "loom-profile-done" in res.stdout and "loom-sys profile_files 2" in res.stdout
    with tarfile.open(dest) as t:
        names = sorted(n.lstrip("./") for n in t.getnames() if t.getmember(n).isfile())
    assert names == ["loom-r_rank0.1.pt.trace.json.gz", "sub/loom-r_rank1.1.pt.trace.json.gz"]
    assert not [p for p in traces.rglob("*") if p.is_file()]  # cleared for the next window


@needs_bash
def test_script_fails_on_an_empty_or_missing_dir_and_over_the_cap(tmp_path: Path) -> None:
    env, traces, dest = _world(tmp_path)
    res = _run(traces, env)
    assert res.returncode != 0 and "no trace directory" in res.stderr
    traces.mkdir()
    res = _run(traces, env)
    assert res.returncode != 0 and "wrote no files" in res.stderr
    big = io.BytesIO()
    import os

    big.write(os.urandom(50_000))
    (traces / "t.json.gz").write_bytes(big.getvalue())
    res = _run(traces, env, max_bytes=1000)
    assert res.returncode != 0 and "over the 1000-byte cap" in res.stderr
    assert not dest.exists()
    assert not list(traces.iterdir())  # an oversized trace is not left for the next window
