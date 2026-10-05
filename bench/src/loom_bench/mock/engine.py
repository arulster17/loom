"""Real-time asyncio driver for the simulator."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

from loom_bench.mock.config import MockConfig
from loom_bench.mock.sim import Clock, SimRequest, SimSequence, Simulator


class AsyncEngine:
    def __init__(self, config: MockConfig, clock: Clock = time.monotonic) -> None:
        self.sim = Simulator(config, clock)
        self._wake = asyncio.Event()
        self._queues: dict[int, asyncio.Queue[int]] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def add(self, request: SimRequest) -> SimSequence:
        seq = self.sim.add(request)
        self._queues[seq.seq_id] = asyncio.Queue()
        self._wake.set()
        return seq

    async def tokens(self, seq: SimSequence) -> AsyncIterator[int]:
        """Yields the output token count each time `seq` samples a token."""
        queue = self._queues[seq.seq_id]
        while True:
            n = await queue.get()
            yield n
            if n == seq.request.output_tokens:
                self._queues.pop(seq.seq_id, None)
                return

    def abort(self, seq: SimSequence) -> None:
        """Release a request the caller stopped consuming; no-op once finished."""
        self._queues.pop(seq.seq_id, None)
        self.sim.abort(seq)

    async def _run(self) -> None:
        while True:
            plan = self.sim.schedule()
            if plan is None:
                await self._wake.wait()
                self._wake.clear()
                continue
            await asyncio.sleep(plan.duration_s)
            for seq in self.sim.finish_step(plan):
                queue = self._queues.get(seq.seq_id)
                if queue is not None:
                    queue.put_nowait(seq.num_output)
