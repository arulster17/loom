import pytest

from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.loadgen.arrivals import GammaArrivals
from loom_bench.loadgen.base import (
    CLOSED_LOOP_REQUESTS_PER_SLOT_S,
    ClosedLoopPlan,
    OpenLoopPlan,
    prepare_load,
)
from loom_bench.records import LoadMode
from loom_bench.tokenize import SimpleTokenizer
from loom_bench.workloads import parse_profile

PROFILE = {
    "name": "chat-short",
    "description": "d",
    "content": "synthetic",
    "kind": "synthetic",
    "endpoint": "chat",
    "input_len": 16,
    "output_len": 4,
}
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}}


def job(**over) -> LoadJob:
    fields = {
        "run_id": "r1",
        "base_url": "http://127.0.0.1:8000/v1",
        "engine": "vllm",
        "served_model": "m",
        "workload": PROFILE,
        "tokenizer": TokenizerSpec(kind="simple"),
        "mode": LoadMode.CLOSED_LOOP,
        "load_value": 3,
        "num_requests": 5,
        "warmup_requests": 2,
        "seed": 4,
        **over,
    }
    return LoadJob(**fields)


def test_open_loop_context_has_one_request_per_arrival():
    ctx = prepare_load(
        job(
            mode=LoadMode.OPEN_LOOP,
            load_value=20,
            arrival={"kind": "gamma", "rate": 20, "burstiness": 0.5},
            duration_s=2.0,
            warmup_s=0.5,
            num_requests=None,
            extra_body=EXTRA,
        )
    )
    assert ctx.profile == parse_profile(PROFILE)
    assert ctx.arrival == GammaArrivals(rate=20, burstiness=0.5)
    assert isinstance(ctx.plan, OpenLoopPlan)
    assert (ctx.plan.duration_s, ctx.plan.warmup_s, ctx.plan.load_value) == (2.0, 0.5, 20)
    assert list(ctx.plan.arrivals) == list(ctx.arrival.schedule(2.0, 4))
    assert len(ctx.requests) == len(ctx.plan.arrivals)
    assert all(
        r.payload["chat_template_kwargs"] == {"enable_thinking": False} for r in ctx.requests
    )
    assert ctx.requests[0].request_id == "chat-short-s4-000000"
    assert isinstance(ctx.tokenizer, SimpleTokenizer)


def test_closed_loop_context():
    ctx = prepare_load(job())
    assert ctx.plan == ClosedLoopPlan(concurrency=3, num_requests=5, warmup_requests=2)
    assert ctx.arrival is None and len(ctx.requests) == 7
    assert "chat_template_kwargs" not in ctx.requests[0].payload

    timed = prepare_load(job(num_requests=None, duration_s=2.0))
    assert timed.plan == ClosedLoopPlan(concurrency=3, duration_s=2.0, warmup_requests=2)
    assert len(timed.requests) == 2 + 3 * 2 * CLOSED_LOOP_REQUESTS_PER_SLOT_S


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"mode": LoadMode.OPEN_LOOP, "duration_s": 1.0}, "arrival and duration_s"),
        ({"num_requests": None}, "num_requests or duration_s"),
        ({"tokenizer": TokenizerSpec(kind="hf", repo="org/m")}, "pinned revision"),
    ],
)
def test_prepare_rejects_incomplete_jobs(over, match):
    with pytest.raises(ValueError, match=match):
        prepare_load(job(**over))
