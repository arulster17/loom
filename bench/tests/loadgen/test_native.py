import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fake_server import BASE_URL, FakeServer

from loom_bench.client import openai_stream
from loom_bench.client.openai_stream import PreparedRequest
from loom_bench.jobs import LoadJob, TokenizerSpec
from loom_bench.loadgen import native
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


class LoopLag:
    """Worst timer lateness of the running event loop while a test's body runs.

    On a loaded machine the whole process can be descheduled for 100+ ms; every
    timer due in that window fires late, the load generator's sends included. That
    lateness is the machine's, not the generator's, so wall-clock bounds in tests
    add it as measured here rather than assume a quiet machine.
    """

    TICK_S = 0.005

    def __init__(self) -> None:
        self.worst_s = 0.0

    async def _probe(self) -> None:
        while True:
            start = time.perf_counter()
            await asyncio.sleep(self.TICK_S)
            self.worst_s = max(self.worst_s, time.perf_counter() - start - self.TICK_S)

    async def __aenter__(self) -> "LoopLag":
        self._task = asyncio.create_task(self._probe())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._task.cancel()

    @property
    def slack_s(self) -> float:
        # A stall can begin up to one tick before the probe's next timer is due.
        return self.worst_s + self.TICK_S


async def test_open_loop_follows_schedule_and_flags_warmup():
    server = FakeServer(ttft_s=0.01, itl_s=0.005, tokens=3)
    arrivals = constant(25.0, 0.4)  # every 40 ms
    async with LoopLag() as lag:
        res = await open_loop(server, arrivals, model="m")
    assert res.mode is LoadMode.OPEN_LOOP
    assert len(res.records) == 10 and all(r.ok for r in res.records)
    assert [r.scheduled_at_s for r in res.records] == pytest.approx(list(arrivals))
    delays = np.array([r.queue_delay_s for r in res.records])
    # Unsaturated, a send's only lag is its timer firing late: 50 ms of slack plus
    # however long the machine stalled the loop. Lag the generator adds itself
    # (drift, a wrong sleep, waiting on responses) does not stall the probe, so
    # it still fails here.
    assert delays.min() >= 0 and delays.max() < 0.05 + lag.slack_s
    assert [r.warmup for r in res.records] == [True] * 3 + [False] * 7
    assert res.load_value == pytest.approx(7 / 0.3)
    assert (res.t_measure_start_s, res.t_measure_end_s) == (0.1, 0.4)
    assert res.client_saturated_count == 0 and res.meta["unsent"] == 0
    gaps = np.diff(server.arrivals)
    # The mean gap spans first to last arrival; a stall can shift either by the loop lag.
    assert gaps.mean() == pytest.approx(0.04, abs=0.01 + lag.slack_s / gaps.size)
    assert all(b["model"] == "m" and b["stream"] for b in server.bodies)
    assert res.records[4].meta == {"i": 4} and res.records[4].completion_tokens == 3


async def test_open_loop_records_client_saturation():
    # The server holds every request until the test releases it, so which requests
    # get a slot is decided by events, not by service times racing the deadline:
    # requests 0 and 1 fill both slots; request 0 is released 0.1 s after it arrives,
    # so request 2 (scheduled at 0.04 s) can only be sent after waiting >= 0.06 s for
    # its slot; everything else stays held past the dispatch deadline, so arrivals
    # 3..14 never get a slot and go unsent. Machine load can only delay the release,
    # which lengthens request 2's wait; the 0.9 s left before the deadline is the margin.
    duration_s = 1.0
    loop = asyncio.get_running_loop()
    released = [asyncio.Event() for _ in range(15)]

    def release_all() -> None:
        for event in released:
            event.set()

    async def gate(index: int) -> None:
        if index == 0:
            loop.call_later(0.1, released[0].set)
            loop.call_later(duration_s + 0.1, release_all)  # past the deadline (>= t0)
        await released[index].wait()

    server = FakeServer(tokens=1, gate=gate)
    arrivals = constant(50.0, 0.3)  # 15 arrivals, all due well before the deadline
    res = await open_loop(
        server, arrivals, duration_s=duration_s, warmup_s=0.0, max_inflight=2, drain_timeout_s=5.0
    )
    assert server.max_inflight <= 2
    assert res.client_saturated_count > 0
    assert len(res.records) == 3 and res.meta["unsent"] == 12
    # A lower bound only: load can delay any send, but request 2 cannot go before
    # request 0's release frees a slot.
    assert res.records[2].queue_delay_s >= 0.06
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


class StepClock:
    """A `perf_counter` that moves only when the fake server starts answering a request.

    Duration mode stops on the clock, so on the real clock how many requests fit in
    the window depends on how long the machine stalls the process (a 0.2 s window of
    40 ms requests held 2, not 10, on a loaded machine). On this clock every request
    answered costs `step_s` and nothing else moves time, and with no timers the event
    loop runs the same steps on any machine, so the count is the generator's alone.
    """

    def __init__(self, step_s: float) -> None:
        self.now = 0.0
        self.step_s = step_s

    def perf_counter(self) -> float:
        return self.now

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The driver and the client take every timestamp from time.perf_counter().
        clock = SimpleNamespace(perf_counter=self.perf_counter)
        monkeypatch.setattr(native, "time", clock)
        monkeypatch.setattr(openai_stream, "time", clock)

    def server(self) -> FakeServer:
        both_in_flight = asyncio.Event()

        async def gate(index: int) -> None:
            if index == 0:  # hold the first until the second worker's is in flight
                await both_in_flight.wait()
            elif index == 1:
                both_in_flight.set()
            self.now += self.step_s

        return FakeServer(tokens=1, gate=gate)


async def test_closed_loop_duration_mode(monkeypatch):
    clock = StepClock(step_s=0.125)  # a power of two: 8 requests reach 1.0 s exactly
    clock.install(monkeypatch)
    server = clock.server()
    res = await run_closed_loop(
        BASE_URL,
        make_requests(1000),
        concurrency=2,
        duration_s=1.0,
        request_timeout_s=5.0,
        transport=server.transport(),
    )
    assert server.max_inflight == 2
    # Each worker sends its next request while the clock is short of the deadline, so
    # the workers keep going until the 8th request's answer reaches it, and then stop:
    # nothing is sent at or after the deadline.
    assert len(res.records) == 8 and all(r.ok for r in res.records)
    assert {r.request_id for r in res.records} == {f"r{i}" for i in range(8)}
    assert all(r.sent_at_s < 1.0 for r in res.records)
    assert max(r.finished_at_s for r in res.records) == 1.0
    assert res.t_measure_end_s == 1.0 and res.meta["exhausted"] is False

    clock = StepClock(step_s=0.125)
    clock.install(monkeypatch)
    res = await run_closed_loop(
        BASE_URL,
        make_requests(3),
        concurrency=2,
        duration_s=1.0,
        request_timeout_s=5.0,
        transport=clock.server().transport(),
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
