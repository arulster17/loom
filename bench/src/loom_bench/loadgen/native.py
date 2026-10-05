"""Native open- and closed-loop drivers over the streaming client.

Open loop dispatches on the arrival schedule whatever the server does, for
latency SLOs; closed loop keeps a fixed number of requests in flight, for
saturation. Both use one t0 (`time.perf_counter()`) for every timestamp.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import httpx
import numpy as np

from loom_bench.client.openai_stream import PreparedRequest, new_record, send
from loom_bench.loadgen.base import LoadPlan, OpenLoopPlan, RunResult
from loom_bench.records import LoadMode, RequestRecord


def _client(
    max_connections: int, request_timeout_s: float, transport: httpx.AsyncBaseTransport | None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=transport,
        limits=httpx.Limits(
            max_connections=max_connections, max_keepalive_connections=max_connections
        ),
        timeout=httpx.Timeout(request_timeout_s),
        # Env proxies would put an extra hop inside every latency measurement.
        trust_env=False,
    )


async def run_open_loop(
    base_url: str,
    requests: Sequence[PreparedRequest],
    arrivals: Sequence[float] | np.ndarray,
    *,
    duration_s: float,
    warmup_s: float,
    max_inflight: int,
    drain_timeout_s: float,
    request_timeout_s: float,
    api_key: str | None = None,
    model: str | None = None,
    keep_output: bool = False,
    load_value: float | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RunResult:
    """Send `requests[i]` at `arrivals[i]` seconds after t0.

    Arrivals before `warmup_s` are flagged warmup. When `max_inflight` requests
    are outstanding, dispatch waits for a slot; the lag shows up as
    `sent_at_s - scheduled_at_s` and in `client_saturated_count`. Nothing is
    dispatched after `duration_s`; in-flight requests get `drain_timeout_s`
    more, then are cancelled and recorded ABORTED.
    """
    offsets = np.asarray(arrivals, dtype=np.float64)
    if len(requests) < offsets.size:
        raise ValueError(f"{offsets.size} arrivals but only {len(requests)} requests")
    if not 0 <= warmup_s < duration_s:
        raise ValueError("need 0 <= warmup_s < duration_s")
    if max_inflight < 1:
        raise ValueError("max_inflight must be >= 1")
    measured = int(np.count_nonzero((offsets >= warmup_s) & (offsets < duration_s)))
    if load_value is None:
        load_value = measured / (duration_s - warmup_s)

    slots = asyncio.Semaphore(max_inflight)
    records: list[RequestRecord] = []
    tasks: list[asyncio.Task[RequestRecord]] = []
    saturated = 0
    async with (
        _client(max_inflight, request_timeout_s, transport) as http,
        asyncio.TaskGroup() as tg,
    ):
        t0 = time.perf_counter()
        deadline = t0 + duration_s
        for i, offset in enumerate(offsets):
            delay = t0 + offset - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if slots.locked():
                saturated += 1
            try:
                async with asyncio.timeout(max(deadline - time.perf_counter(), 0.0)):
                    await slots.acquire()
            except TimeoutError:
                break
            rec = new_record(requests[i], t0, scheduled_at_s=float(offset))
            rec.warmup = bool(offset < warmup_s)
            records.append(rec)
            task = tg.create_task(
                send(
                    http,
                    base_url,
                    requests[i],
                    t0,
                    timeout_s=request_timeout_s,
                    api_key=api_key,
                    keep_output=keep_output,
                    model=model,
                    record=rec,
                )
            )
            task.add_done_callback(lambda _: slots.release())
            tasks.append(task)

        pending = [t for t in tasks if not t.done()]
        if pending:
            remaining = deadline + drain_timeout_s - time.perf_counter()
            _, stragglers = await asyncio.wait(pending, timeout=max(remaining, 0.0))
            for t in stragglers:
                t.cancel()

    return RunResult(
        records=records,
        mode=LoadMode.OPEN_LOOP,
        load_value=load_value,
        t_measure_start_s=warmup_s,
        t_measure_end_s=duration_s,
        client_saturated_count=saturated,
        meta={
            "scheduled": int(offsets.size),
            "unsent": int(offsets.size) - len(records),
            "max_inflight": max_inflight,
        },
    )


async def run_closed_loop(
    base_url: str,
    requests: Sequence[PreparedRequest],
    *,
    concurrency: int,
    num_requests: int | None = None,
    duration_s: float | None = None,
    warmup_requests: int = 0,
    request_timeout_s: float,
    api_key: str | None = None,
    model: str | None = None,
    keep_output: bool = False,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RunResult:
    """`concurrency` workers, each sending its next request when the previous ends.

    Stops after `warmup_requests + num_requests` requests, or once `duration_s`
    has elapsed (in-flight requests finish; none start after). The first
    `warmup_requests` dispatched are flagged warmup.
    """
    if (num_requests is None) == (duration_s is None):
        raise ValueError("give exactly one of num_requests or duration_s")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    total = len(requests) if num_requests is None else warmup_requests + num_requests
    if len(requests) < total:
        raise ValueError(f"need {total} requests, got {len(requests)}")

    queue = iter(enumerate(requests[:total]))
    records: list[RequestRecord] = []

    async def worker(http: httpx.AsyncClient, t0: float) -> None:
        for i, req in queue:  # shared iterator: each worker takes the next request
            if duration_s is not None and time.perf_counter() - t0 >= duration_s:
                return
            rec = new_record(req, t0)
            rec.warmup = i < warmup_requests
            records.append(rec)
            await send(
                http,
                base_url,
                req,
                t0,
                timeout_s=request_timeout_s,
                api_key=api_key,
                keep_output=keep_output,
                model=model,
                record=rec,
            )

    async with _client(concurrency, request_timeout_s, transport) as http:
        t0 = time.perf_counter()
        async with asyncio.TaskGroup() as tg:
            for _ in range(concurrency):
                tg.create_task(worker(http, t0))
        ended_s = time.perf_counter() - t0

    measured = [r for r in records if not r.warmup]
    start = min((r.sent_at_s for r in measured), default=0.0)
    if duration_s is not None:
        end = duration_s
    else:
        end = max((r.finished_at_s or r.sent_at_s for r in measured), default=start)
    return RunResult(
        records=records,
        mode=LoadMode.CLOSED_LOOP,
        load_value=float(concurrency),
        t_measure_start_s=start,
        t_measure_end_s=end,
        meta={
            "concurrency": concurrency,
            # Duration runs that end early ran out of prepared requests.
            "exhausted": duration_s is not None and ended_s < duration_s,
        },
    )


class NativeLoadGenerator:
    name = "native"

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def run(
        self,
        base_url: str,
        requests: Sequence[PreparedRequest],
        plan: LoadPlan,
        *,
        request_timeout_s: float,
        model: str | None = None,
        api_key: str | None = None,
        keep_output: bool = False,
    ) -> RunResult:
        if isinstance(plan, OpenLoopPlan):
            return await run_open_loop(
                base_url,
                requests,
                plan.arrivals,
                duration_s=plan.duration_s,
                warmup_s=plan.warmup_s,
                max_inflight=plan.max_inflight,
                drain_timeout_s=plan.drain_timeout_s,
                request_timeout_s=request_timeout_s,
                api_key=api_key,
                model=model,
                keep_output=keep_output,
                load_value=plan.load_value,
                transport=self._transport,
            )
        return await run_closed_loop(
            base_url,
            requests,
            concurrency=plan.concurrency,
            num_requests=plan.num_requests,
            duration_s=plan.duration_s,
            warmup_requests=plan.warmup_requests,
            request_timeout_s=request_timeout_s,
            api_key=api_key,
            model=model,
            keep_output=keep_output,
            transport=self._transport,
        )
