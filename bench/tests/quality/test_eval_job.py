"""EvalJobs against the mock backend: in-process and through `bench quality job`."""

import asyncio
from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from loom_bench.cli import app
from loom_bench.jobs import EvalJob, EvalJobResult, TokenizerSpec
from loom_bench.providers.base import HostRequest
from loom_bench.providers.mock import MockProvider
from loom_bench.quality import runner as quality_runner
from loom_bench.quality.gate import Verdict
from loom_bench.quality.runner import (
    execute_eval_job,
    gate_against_baseline,
    run_suite,
    suite_result_of,
)
from loom_bench.quality.suite import Suite
from loom_bench.tokenize import SnapshotMismatch

from .mock_serve import MODEL, serve

SUITE = Suite.model_validate(
    {
        "suite": "eval-job",
        "model": MODEL,
        "seed": 1234,
        "gate": {"threshold": 0.06, "min_samples": 50, "n_boot": 1000},
        "tasks": [
            {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 60, "seed": 7}},
            {"name": "json_schema", "kind": "json_schema"},
        ],
        "divergence": {"prompts": 12, "top_k": 5, "max_new_tokens": 16},
    }
)


@pytest.fixture(scope="module")
def clean_url() -> Iterator[str]:
    with serve(time_scale=0.001) as url:
        yield url


@pytest.fixture(scope="module")
def broken_url() -> Iterator[str]:
    with serve(time_scale=0.001, degrade=0.3, logprob_noise=3.0) as url:
        yield url


def job(url: str, **kw) -> EvalJob:
    return EvalJob(
        run_id=kw.pop("run_id", "job-1"),
        suite=kw.pop("suite", SUITE),
        base_url=url,
        served_model=MODEL,
        seed=SUITE.seed,
        **kw,
    )


async def test_capture_on_the_baseline_then_score_the_candidate(clean_url, broken_url):
    base = await execute_eval_job(job(clean_url, divergence="capture_and_floor"))
    assert base.reference is not None and base.divergence is None
    assert len(base.reference.prompts) == 12
    # The deterministic mock scores its own capture identically at any concurrency.
    floor = base.reference.self_divergence
    assert floor is not None and floor.n_prompts == 12
    assert floor.kl.point == pytest.approx(0.0, abs=1e-9) and floor.top1.point == 1.0
    assert base.tasks["arithmetic"].kind == "toy_arithmetic"
    assert all(i.content_hash for t in base.tasks.values() for i in t.items)
    assert all(t.seconds > 0 for t in base.tasks.values())

    shipped = EvalJob.model_validate_json(
        job(broken_url, divergence="score", reference=base.reference).model_dump_json()
    )
    cand = EvalJobResult.model_validate_json((await execute_eval_job(shipped)).model_dump_json())
    assert cand.reference is None and cand.divergence is not None
    assert cand.divergence.kl.point > SUITE.divergence.max_kl

    decision = gate_against_baseline(
        suite_result_of(base),
        suite_result_of(cand),
        SUITE,
        cand.divergence,
        self_divergence=base.reference.self_divergence,
    )
    assert decision.divergence.verdict is Verdict.FAIL
    assert "hard ceiling" in decision.divergence.reason
    assert decision.blocked


async def test_plain_capture_measures_no_floor(clean_url):
    base = await execute_eval_job(job(clean_url, divergence="capture", tasks=["json_schema"]))
    assert base.reference is not None and base.reference.self_divergence is None


async def test_task_subset_and_no_divergence(clean_url):
    res = await execute_eval_job(job(clean_url, tasks=["json_schema"]))
    assert list(res.tasks) == ["json_schema"]
    assert res.reference is None and res.divergence is None
    assert suite_result_of(res).tasks["json_schema"].estimate.n == len(
        res.tasks["json_schema"].items
    )


async def test_gate_refuses_results_from_different_task_versions(clean_url):
    """json_schema data version 2 (300 items) never pairs with a version-1 (60 item)
    baseline: the gate names the task and asks for a baseline rerun."""
    res = await execute_eval_job(job(clean_url, tasks=["json_schema"]))
    base, cand = suite_result_of(res), suite_result_of(res)
    assert base.tasks["json_schema"].version.endswith("+data.2")
    base.tasks["json_schema"].version = "1+data.1"
    with pytest.raises(ValueError, match=r"json_schema: baseline ran version 1\+data\.1"):
        gate_against_baseline(base, cand, SUITE)


def test_bench_quality_job_cli(clean_url, tmp_path):
    path = tmp_path / "job.json"
    path.write_text(job(clean_url, run_id="cli-1", divergence="capture").model_dump_json())
    out = tmp_path / "out" / "result.json"
    result = CliRunner().invoke(app, ["quality", "job", "--in", str(path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    got = EvalJobResult.model_validate_json(out.read_text())
    assert got.run_id == "cli-1" and set(got.tasks) == {"arithmetic", "json_schema"}
    assert got.reference is not None


def test_code_exec_stays_off_unless_the_job_allows_it(clean_url, tmp_path):
    suite = Suite.model_validate(
        {
            "suite": "code-only",
            "model": MODEL,
            "tasks": [{"name": "code", "kind": "code_exec", "params": {"datasets": ["mbpp"]}}],
        }
    )
    path = tmp_path / "job.json"
    path.write_text(job(clean_url, suite=suite).model_dump_json())
    result = CliRunner().invoke(
        app, ["quality", "job", "--in", str(path), "--out", str(tmp_path / "r.json")]
    )
    assert result.exit_code == 1
    assert "CodeExecDisabled" in result.output
    assert not (tmp_path / "r.json").exists()


def test_in_process_providers_run_eval_jobs(clean_url):
    async def go() -> EvalJobResult:
        p = MockProvider()
        host = await p.provision(HostRequest(ttl_s=60))
        try:
            return await p.run_eval(host, job(clean_url, tasks=["arithmetic"]))
        finally:
            await p.teardown(host)

    assert asyncio.run(go()).tasks["arithmetic"].items


async def test_local_tokenizer_snapshot_reaches_the_tasks(clean_url, tmp_path, monkeypatch):
    repo, rev = "meta-llama/Llama-3.3-70B-Instruct", "6" * 40
    local = tmp_path / "models--meta-llama--Llama-3.3-70B-Instruct" / "snapshots" / rev
    local.mkdir(parents=True)
    seen: dict[str, str] = {}

    async def spy(*args, **kwargs):
        seen.update(kwargs["local_tokenizers"])
        return await run_suite(*args, **kwargs)

    monkeypatch.setattr(quality_runner, "run_suite", spy)
    tok = TokenizerSpec(kind="hf", repo=repo, revision=rev, local_dir=str(local))
    await execute_eval_job(job(clean_url, tasks=["json_schema"], tokenizer=tok))
    assert seen == {repo: str(local)}

    stale = tok.model_copy(update={"revision": "7" * 40})
    with pytest.raises(SnapshotMismatch, match="is not the snapshot of"):
        await execute_eval_job(job(clean_url, tasks=["json_schema"], tokenizer=stale))


async def test_a_failed_divergence_keeps_the_task_scores(clean_url):
    base = await execute_eval_job(job(clean_url, divergence="capture"))
    assert base.reference is not None
    # A reference the candidate cannot line up with: its first prompt tokenized otherwise.
    first = base.reference.prompts[0]
    retokenized = first.model_copy(
        update={"positions": [p.model_copy(update={"token": "#"}) for p in first.positions]}
    )
    reference = base.reference.model_copy(
        update={"prompts": [retokenized, *base.reference.prompts[1:]]}
    )
    cand = await execute_eval_job(job(clean_url, divergence="score", reference=reference))
    assert cand.divergence is None
    assert (cand.divergence_error or "").startswith(
        "ValueError: reference and candidate tokenized the same text differently"
    )
    assert set(cand.tasks) == {"arithmetic", "json_schema"}
    assert all(t.items for t in cand.tasks.values())
    decision = gate_against_baseline(
        suite_result_of(base),
        suite_result_of(cand),
        SUITE,
        cand.divergence,
        divergence_error=cand.divergence_error,
    )
    assert decision.divergence.verdict is Verdict.INCONCLUSIVE
    assert "tokenized the same text differently" in decision.divergence.reason
    assert len(decision.tasks) == 2 and decision.blocked
