import time

import numpy as np
import pytest
from fake_server import BASE_URL, FakeServer

from loom_bench.client.openai_stream import PreparedRequest
from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.loadgen.arrivals import constant
from loom_bench.loadgen.base import (
    LOAD_GENERATORS,
    ClosedLoopPlan,
    LoadContext,
    OpenLoopPlan,
    get_load_generator,
)
from loom_bench.loadgen.native import NativeLoadGenerator, run_closed_loop, run_open_loop
from loom_bench.records import LoadMode, RequestStatus
from loom_bench.tokenize import SimpleTokenizer
from loom_bench.workloads import parse_profile


def make_requests(n):
    return [
        PreparedRequest(f"r{i}", "chat", {"messages": [], "max_tokens": 4}, 10, 4, {"i": i})
        for i in range(n)
    ]


def load_context(requests, plan):
    profile = parse_profile(
        {
            "name": "p",
            "description": "d",
            "content": "synthetic",
            "kind": "synthetic",
            "input_len": 8,
            "output_len": 4,
        }
    )
    open_loop = isinstance(plan, OpenLoopPlan)
    job = LoadJob(
        run_id="r",
        base_url=BASE_URL,
        engine="mock",
        served_model="m",
        workload=profile.model_dump(mode="json"),
        tokenizer=TokenizerSpec(kind="simple"),
        mode=LoadMode.OPEN_LOOP if open_loop else LoadMode.CLOSED_LOOP,
        load_value=plan.load_value if open_loop else plan.concurrency,
        request_timeout_s=5,
    )
    return LoadContext(job, profile, None, plan, requests, SimpleTokenizer())


async def open_loop(server, arrivals, **kw):
    kw = {
        "duration_s": 0.4,
        "warmup_s": 0.1,
        "max_inflight": 100,
        "drain_timeout_s": 1.0,
        "request_timeout_s": 5.0,
        **kw,
    }
    return await run_open_loop(
        BASE_URL, make_requests(len(arrivals)), arrivals, transport=server.transport(), **kw
    )


async def test_open_loop_follows_schedule_and_flags_warmup():
    server = FakeServer(ttft_s=0.01, itl_s=0.005, tokens=3)
    arrivals = constant(25.0, 0.4)  # every 40 ms
    res = await open_loop(server, arrivals, model="m")
    assert res.mode is LoadMode.OPEN_LOOP
    assert len(res.records) == 10 and all(r.ok for r in res.records)
    assert [r.scheduled_at_s for r in res.records] == pytest.approx(list(arrivals))
    delays = np.array([r.queue_delay_s for r in res.records])
    assert delays.min() >= 0 and delays.max() < 0.05
    assert [r.warmup for r in res.records] == [True] * 3 + [False] * 7
    assert res.load_value == pytest.approx(7 / 0.3)
    assert (res.t_measure_start_s, res.t_measure_end_s) == (0.1, 0.4)
    assert res.client_saturated_count == 0 and res.meta["unsent"] == 0
    gaps = np.diff(server.arrivals)
    assert gaps.mean() == pytest.approx(0.04, abs=0.01)
    assert all(b["model"] == "m" and b["stream"] for b in server.bodies)
    assert res.records[4].meta == {"i": 4} and res.records[4].completion_tokens == 3


async def test_open_loop_records_client_saturation():
    server = FakeServer(ttft_s=0.12, tokens=1)
    arrivals = constant(50.0, 0.3)  # 15 arrivals, ~3 slots' worth of service capacity
    res = await open_loop(server, arrivals, duration_s=0.3, warmup_s=0.0, max_inflight=2)
    assert server.max_inflight <= 2
    assert res.client_saturated_count > 0
    assert res.meta["unsent"] > 0
    assert res.meta["unsent"] + len(res.records) == 15
    assert max(r.queue_delay_s for r in res.records) > 0.05
    assert all(r.ok for r in res.records)


async def test_open_loop_cancels_stragglers_after_drain_timeout():
    server = FakeServer(stall=True)
    start = time.perf_counter()
    res = await open_loop(
        server, np.array([0.0, 0.05]), duration_s=0.1, warmup_s=0.0, drain_timeout_s=0.1
    )
    assert time.perf_counter() - start < 0.5
    assert [r.status for r in res.records] == [RequestStatus.ABORTED] * 2
    assert all(r.finished_at_s is not None and r.first_token_at_s is None for r in res.records)
    assert server.inflight == 0


async def test_open_loop_validates_inputs():
    with pytest.raises(ValueError, match="only 1 requests"):
        await run_open_loop(
            BASE_URL, make_requests(1), [0.0, 0.1], duration_s=1, warmup_s=0,
            max_inflight=1, drain_timeout_s=1, request_timeout_s=1,
        )  # fmt: skip
    with pytest.raises(ValueError, match="warmup_s"):
        await run_open_loop(
            BASE_URL, make_requests(1), [0.0], duration_s=1, warmup_s=1,
            max_inflight=1, drain_timeout_s=1, request_timeout_s=1,
        )  # fmt: skip


async def test_closed_loop_keeps_concurrency_and_flags_warmup():
    server = FakeServer(ttft_s=0.01, itl_s=0.002, tokens=3)
    res = await run_closed_loop(
        BASE_URL,
        make_requests(40),
        concurrency=3,
        num_requests=12,
        warmup_requests=3,
        request_timeout_s=5.0,
        transport=server.transport(),
    )
    assert res.mode is LoadMode.CLOSED_LOOP and res.load_value == 3.0
    assert server.max_inflight == 3
    assert len(res.records) == 15 and all(r.ok for r in res.records)
    assert sum(r.warmup for r in res.records) == 3
    assert {r.request_id for r in res.records} == {f"r{i}" for i in range(15)}
    assert all(r.scheduled_at_s is None for r in res.records)
    measured = [r for r in res.records if not r.warmup]
    assert res.t_measure_start_s == min(r.sent_at_s for r in measured)
    assert res.t_measure_end_s == max(r.finished_at_s for r in measured)


async def test_closed_loop_duration_mode():
    server = FakeServer(ttft_s=0.04, tokens=1)
    res = await run_closed_loop(
        BASE_URL,
        make_requests(1000),
        concurrency=2,
        duration_s=0.2,
        request_timeout_s=5.0,
        transport=server.transport(),
    )
    assert server.max_inflight == 2
    assert 6 <= len(res.records) <= 12
    assert all(r.sent_at_s < 0.2 for r in res.records)
    assert res.t_measure_end_s == 0.2 and res.meta["exhausted"] is False

    res = await run_closed_loop(
        BASE_URL,
        make_requests(3),
        concurrency=2,
        duration_s=5.0,
        request_timeout_s=5.0,
        transport=FakeServer(tokens=1).transport(),
    )
    assert len(res.records) == 3 and res.meta["exhausted"] is True


async def test_closed_loop_validates_inputs():
    with pytest.raises(ValueError, match="exactly one"):
        await run_closed_loop(BASE_URL, make_requests(1), concurrency=1, request_timeout_s=1)
    with pytest.raises(ValueError, match="need 5"):
        await run_closed_loop(
            BASE_URL, make_requests(4), concurrency=1, num_requests=3, warmup_requests=2,
            request_timeout_s=1,
        )  # fmt: skip


async def test_registry_and_protocol_dispatch():
    assert "native" in LOAD_GENERATORS
    assert get_load_generator("native").name == "native"
    with pytest.raises(KeyError, match="known: native"):
        get_load_generator("nope")

    server = FakeServer(tokens=2)
    gen = NativeLoadGenerator(transport=server.transport())
    plan = OpenLoopPlan(constant(50.0, 0.1), 0.1, 0.0, 10, 1.0, load_value=50.0)
    res = await gen.run(load_context(make_requests(5), plan))
    assert res.mode is LoadMode.OPEN_LOOP and res.load_value == 50.0 and len(res.records) == 5
    assert all(b["model"] == "m" for b in server.bodies)
    res = await gen.run(
        load_context(make_requests(4), ClosedLoopPlan(concurrency=2, num_requests=4))
    )
    assert res.mode is LoadMode.CLOSED_LOOP and len(res.records) == 4
