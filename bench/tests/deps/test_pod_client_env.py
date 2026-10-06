"""The GPU hosts' client environment, rebuilt offline: dependency bugs cost nothing here.

Builds exactly what a RunPod or AWS host installs for a job: a clean CPython 3.12 venv,
the hash-locked `uv export` requirements with the `lmeval` extra through `pip install
--require-hashes --no-deps`, the bench wheel with `--no-deps`, then `pip check`. Then it
runs every task of every shipped eval suite at 2 items per task (`Suite.limited`), plus
divergence capture and its noise floor, through `bench quality job` (the hosts' own entry
point) in that venv, against the local mock backend.

The first RunPod 8B sweep (058128e9) lost $3.39 to an lm-eval extra missing from exactly
this set (IFEval's `langdetect`); this check fails on that lock.

It downloads wheels, the lm-eval datasets and a public tokenizer, and takes minutes, so it
is deselected by default (`-m "not network"` in pytest's addopts). Run it with
`uv run pytest -m network` (CI: the `pod-client-env` job). LOOM_POD_REQUIREMENTS=<file>
checks another requirements set, e.g. one exported from an older uv.lock.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

from loom_bench.jobs import EvalJob, EvalJobResult
from loom_bench.providers import build_wheel, export_requirements
from loom_bench.providers.runpod import LMEVAL_EXTRA
from loom_bench.quality.suite import EVALS_DIR, Suite, load_suite

pytestmark = [pytest.mark.network, pytest.mark.timeout(3600)]

ITEMS = 2
CLIENT_PYTHON = "3.12"  # the pinned python-build-standalone the pods use (runpod_layout)
MOCK_MODEL = "mock-model"
# Tokenizers behind a license gate need an HF token with the license accepted.
GATED_PREFIXES = ("meta-llama/",)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run(argv: list[str | Path], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(a) for a in argv], check=True, capture_output=True, text=True, **kw)


def build_client_env(root: Path, requirements: str) -> Path:
    """The hosts' install, step for step (runpod_scripts/run_job.sh); returns the venv."""
    uv = shutil.which("uv")
    assert uv, "needs uv to fetch a CPython 3.12 like the pods' pinned one"
    env = root / "env"
    _run([uv, "venv", "--quiet", "--seed", "--python", CLIENT_PYTHON, env])
    reqs = root / "requirements.txt"
    reqs.write_text(requirements)
    wheel = build_wheel(root / "wheel")
    pip = [env / "bin" / "python", "-m", "pip", "install", "--quiet", "--no-cache-dir"]
    _run([*pip, "--require-hashes", "--no-deps", "-r", reqs])
    _run([*pip, "--no-deps", wheel])
    _run([env / "bin" / "python", "-m", "pip", "check"])
    return env


def _gated(task_params: dict) -> bool:
    names = [
        task_params.get("metadata", {}).get("tokenizer"),
        task_params.get("model_args", {}).get("tokenizer"),
    ]
    return any(isinstance(n, str) and n.startswith(GATED_PREFIXES) for n in names)


def runnable(suite: Suite) -> tuple[Suite, list[str]]:
    """`suite` at smoke scale, minus tasks whose tokenizer is gated when no HF token is
    set (their harness task still runs in a suite with a public tokenizer)."""
    small = suite.limited(ITEMS)
    if os.environ.get("HF_TOKEN"):
        return small, []
    skipped = [t.name for t in small.tasks if t.kind == "lm_eval" and _gated(t.params)]
    keep = [t for t in small.tasks if t.name not in skipped]
    doc = small.model_dump(mode="json")
    doc["tasks"] = [t.model_dump(mode="json") for t in keep]
    doc["subsets"] = {
        k: [n for n in v if n not in skipped]
        for k, v in doc["subsets"].items()
        if any(n not in skipped for n in v)
    }
    return Suite.model_validate(doc), skipped


def harness_tasks(suite: Suite) -> set[tuple[str, str]]:
    return {
        (t.kind, name)
        for t in suite.tasks
        for name in (t.params.get("tasks") if t.kind == "lm_eval" else [t.name])
    }


@pytest.fixture(scope="module")
def mock_backend():
    port = _free_port()
    serve = "from loom_bench.cli import app; app()"
    proc = subprocess.Popen(
        [sys.executable, "-c", serve, "mock-server", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/v1"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{base}/models", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.2)
        else:
            raise RuntimeError("mock backend did not start")
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def client_env(tmp_path_factory) -> Path:
    override = os.environ.get("LOOM_POD_REQUIREMENTS")
    requirements = Path(override).read_text() if override else export_requirements(LMEVAL_EXTRA)
    return build_client_env(tmp_path_factory.mktemp("pod"), requirements)


SUITES = sorted(p.stem for p in EVALS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", SUITES)
def test_every_suite_task_runs_in_the_hosts_client_env(name, client_env, mock_backend, tmp_path):
    suite, skipped = runnable(load_suite(name))
    if skipped:  # a gated tokenizer without HF_TOKEN: the same harness task must run elsewhere
        covered = set().union(*(harness_tasks(load_suite(n)) for n in SUITES if n != name))
        gated = {k for k in harness_tasks(load_suite(name)) if k not in harness_tasks(suite)}
        assert gated <= covered, (name, skipped)
    job = EvalJob(
        run_id=str(uuid.uuid4()),
        suite=suite,
        base_url=mock_backend,
        served_model=MOCK_MODEL,
        extra_body=suite.extra_body(),
        allow_code_exec=True,  # the mock's output, in the sandbox
        seed=suite.seed,
        concurrency=4,
        divergence="capture_and_floor" if suite.divergence is not None else None,
    )
    job_path, out = tmp_path / "job.json", tmp_path / "out" / "result.json"
    job_path.write_text(job.model_dump_json())
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "UV_"))}
    env["PATH"] = f"{client_env / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    proc = subprocess.run(
        [client_env / "bin" / "bench", "quality", "job", "--in", job_path, "--out", out],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    result = EvalJobResult.model_validate_json(out.read_text())
    assert set(result.tasks) == {t.name for t in suite.tasks}
    for task, res in result.tasks.items():
        assert res.items, f"{name}/{task} scored no items"
    if suite.divergence is not None:
        assert result.reference is not None and result.reference.self_divergence is not None
    print(json.dumps({"suite": name, "tasks": sorted(result.tasks), "skipped": skipped}))
