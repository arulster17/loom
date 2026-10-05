from datetime import UTC, datetime, timedelta

import pytest

from loom_bench.jobs import (
    EvalJob,
    EvalJobResult,
    EvalTaskResult,
    LoadJob,
    LoadJobResult,
    TokenizerSpec,
)
from loom_bench.providers.base import Host, HostRequest
from loom_bench.quality.divergence import ReferenceLogprobs, ReferencePosition, ReferencePrompt
from loom_bench.quality.sanity import SanityResult
from loom_bench.quality.suite import Suite
from loom_bench.quality.tasks.base import ItemResult
from loom_bench.records import LoadMode, Market, RequestRecord, RequestStatus


def test_load_job_json_round_trip():
    job = LoadJob(
        run_id="r1",
        base_url="http://127.0.0.1:8000",
        engine="mock",
        served_model="mock-model",
        workload={"kind": "synthetic", "input_len": 128},
        tokenizer=TokenizerSpec(kind="simple"),
        mode=LoadMode.OPEN_LOOP,
        load_value=2.0,
        duration_s=10,
    )
    assert LoadJob.model_validate_json(job.model_dump_json()) == job


def test_load_job_result_rebuilds_records():
    rec = RequestRecord(request_id="a", status=RequestStatus.OK, sent_at_s=0.0, itl_s=[0.1])
    res = LoadJobResult(
        run_id="r1",
        mode=LoadMode.CLOSED_LOOP,
        load_value=4,
        t_measure_start_s=0,
        t_measure_end_s=1,
        records=[rec.to_row()],
        scrapes=[(0.5, "vllm:num_requests_running 1\n")],
        started_at="2026-10-04T00:00:00Z",
        finished_at="2026-10-04T00:00:01Z",
    )
    again = LoadJobResult.model_validate_json(res.model_dump_json())
    assert again.request_records() == [rec]
    assert again.scrapes == [(0.5, "vllm:num_requests_running 1\n")]


def test_host_round_trip():
    now = datetime(2026, 10, 4, tzinfo=UTC)
    host = Host(
        provider="mock",
        host_id="h1",
        request=HostRequest(ttl_s=60, market=Market.SPOT),
        hourly_micros=1_000_000,
        launched_at=now,
        ttl_at=now + timedelta(seconds=60),
    )
    assert Host.model_validate_json(host.model_dump_json()) == host


SUITE = {
    "suite": "unit",
    "model": "mock-model",
    "seed": 7,
    "chat_template_kwargs": {"enable_thinking": False},
    "tasks": [
        {"name": "arithmetic", "kind": "toy_arithmetic", "params": {"n": 10, "seed": 1}},
        {"name": "json_schema", "kind": "json_schema", "threshold": 0.06},
    ],
    "divergence": {"top_k": 3, "max_new_tokens": 8},
}
REFERENCE = ReferenceLogprobs(
    model="mock-model",
    top_k=3,
    max_new_tokens=8,
    prompts=[
        ReferencePrompt(
            prompt="Hi",
            continuation=" there",
            positions=[ReferencePosition(token=" there", top={" there": -0.25, " all": -1.5})],
        ),
        ReferencePrompt(prompt="Empty", continuation="", positions=[]),
    ],
    config_hash="c" * 64,
    provenance={"engine": {"name": "vllm"}},
)


def eval_job(**kw) -> EvalJob:
    base = {
        "run_id": "e1",
        "suite": Suite.model_validate(SUITE),
        "tasks": ["arithmetic"],
        "base_url": "http://127.0.0.1:8000/v1",
        "served_model": "mock-model",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "allow_code_exec": False,
        "seed": 7,
    }
    return EvalJob(**{**base, **kw})


def test_eval_job_json_round_trip():
    for job in (eval_job(divergence="capture"), eval_job(divergence="score", reference=REFERENCE)):
        again = EvalJob.model_validate_json(job.model_dump_json())
        assert again == job
        assert again.suite.policy() == job.suite.policy()
    assert EvalJob.model_validate_json(eval_job(tasks=None).model_dump_json()).tasks is None


def test_eval_job_validates_tasks_and_divergence():
    with pytest.raises(ValueError, match="not in suite"):
        eval_job(tasks=["mmlu"])
    with pytest.raises(ValueError, match="reference"):
        eval_job(divergence="score")
    with pytest.raises(ValueError, match="reference"):
        eval_job(divergence="capture", reference=REFERENCE)
    no_div = Suite.model_validate({k: v for k, v in SUITE.items() if k != "divergence"})
    with pytest.raises(ValueError, match="no divergence"):
        eval_job(suite=no_div, divergence="capture")


def test_eval_job_result_json_round_trip():
    res = EvalJobResult(
        run_id="e1",
        suite="unit",
        model="mock-model",
        tasks={
            "arithmetic": EvalTaskResult(
                kind="toy_arithmetic",
                version="1",
                items=[ItemResult(item_id="q1", score=1.0, content_hash="h1")],
                provenance={"source": "generated"},
                seconds=1.5,
            )
        },
        sanity=SanityResult(n=1, counts={"empty": 0}),
        reference=REFERENCE,
        divergence=None,
        started_at="2026-10-04T00:00:00+00:00",
        finished_at="2026-10-04T00:00:02+00:00",
    )
    assert EvalJobResult.model_validate_json(res.model_dump_json()) == res
