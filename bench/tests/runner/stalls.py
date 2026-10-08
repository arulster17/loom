"""Measure how long this process was stalled while a test body ran.

On a loaded machine the whole process can be descheduled, or its timers deferred,
for far longer than any bound an end-to-end test sets on wall time. Those stalls are
the machine's, not the code's, so tests that bound elapsed time (or spend accrued
over it) add the stalls measured inside the window they bound rather than assume a
quiet machine.
"""

import threading
import time
from datetime import UTC, datetime


class StallProbe:
    """Records every tick of a 5 ms sleep loop that woke up late, with its wall time.

    The probe runs in its own thread so it keeps measuring while the code under test
    holds the event loop (or the CLI runs its own loop): a descheduled process,
    deferred timers and a long-held GIL all make the probe's sleeps overrun.
    """

    TICK_S = 0.005

    def __init__(self) -> None:
        self._late: list[tuple[float, float]] = []  # (woke at, epoch s; seconds late)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stall-probe", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            start = time.perf_counter()
            time.sleep(self.TICK_S)
            late = time.perf_counter() - start - self.TICK_S
            if late > self.TICK_S:  # ignore ordinary scheduling jitter
                self._late.append((time.time(), late))

    def __enter__(self) -> "StallProbe":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()

    def stalled_s(self, start: datetime, end: datetime) -> float:
        """Seconds of stall overlapping [start, end] (aware or naive-UTC datetimes)."""
        lo, hi = _epoch(start), _epoch(end)
        total = 0.0
        for woke, late in self._late:
            # The overrun is the tail of the tick that ended at `woke`, plus one tick of
            # slack for a stall that began just before the probe's timer was due.
            s0, s1 = woke - late - self.TICK_S, woke
            total += max(0.0, min(s1, hi) - max(s0, lo))
        return total


def _epoch(dt: datetime) -> float:
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()
