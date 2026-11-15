"""Durable fallback for settles Redis did not accept (docs/phase1-contracts.md §6 "Settle
spool").

A settle that still fails after the proxy's quick retries (Redis down, failing over,
frozen past the socket timeout) is appended to a local append-only file, one JSON
`UsageEvent` per line, flushed and fsynced before the request's handling moves on. A
background task replays the file into `settle.lua` with backoff (0.5 s doubling to 30 s)
until Redis accepts each entry, then drops the replayed lines (rewrite to a temporary
file, fsync, rename).

Replay is idempotent: `settle.lua` refuses a request id it has settled before (the
`done:<request id>` marker, kept `Limiter.settled_ttl_s`), so a settle that did reach
Redis although the client saw an error (reply lost) is not charged twice. An entry older
than the marker's lifetime is first looked up in Postgres (`usage_events`, the billing
worker's copy of the stream): found means it was settled, consumed and billed, so it is
dropped; Postgres stores an event at most once anyway (`ON CONFLICT (id, created_at)`).

One process per spool file (an exclusive `flock`); a gateway restarted on the same
volume replays what its predecessor left.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import fcntl
import logging
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import IO

from pydantic import ValidationError
from redis.exceptions import RedisError

from loom_gateway.metrics import GatewayMetrics
from loom_platform.limiter import Limiter
from loom_platform.usage import UsageEvent

log = logging.getLogger("loom_gateway.spool")

MIN_BACKOFF_S = 0.5
MAX_BACKOFF_S = 30.0

# Is this event already in Postgres? (consulted for entries older than the done marker)
SettledCheck = Callable[[UsageEvent], Awaitable[bool]]


def default_spool_path() -> Path:
    """Per-process file in the temp directory: survives a Redis outage, not a restart.
    Production sets LOOM_SETTLE_SPOOL_PATH on a volume that outlives the process."""
    run = f"loom-gateway-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return Path(tempfile.gettempdir()) / run / "settle-spool.jsonl"


class SettleSpool:
    def __init__(
        self,
        path: Path,
        limiter: Limiter,
        metrics: GatewayMetrics,
        *,
        settled_ttl_s: int,
        settled_check: SettledCheck | None = None,
    ) -> None:
        self.path = path
        self.limiter = limiter
        self.metrics = metrics
        self.settled_ttl_s = settled_ttl_s
        self.settled_check = settled_check
        self.depth = 0
        self._lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._lockfile: IO[bytes] | None = None
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ lifecycle

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lockfile = open(self.path.with_suffix(".lock"), "ab")  # noqa: SIM115
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError(f"settle spool {self.path} is in use by another process") from e
        self.path.touch()
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            # A write torn by a crash: it was never acknowledged, drop the partial line.
            keep = data[: data.rfind(b"\n") + 1]
            log.error("settle spool: dropping a torn last line", extra={"fields": {}})
            self._rewrite(keep.decode().splitlines(keepends=True))
            data = keep
        self.depth = data.count(b"\n")
        self._gauge()
        if self.depth:
            log.warning(
                "settle spool: replaying entries left by a previous run",
                extra={"fields": {"depth": self.depth, "path": str(self.path)}},
            )
            self._wake.set()

    def start(self) -> None:
        self._task = asyncio.create_task(self.run())

    async def stop(self, final_attempt_s: float = 2.0) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.depth:
            with contextlib.suppress(Exception, TimeoutError):
                async with asyncio.timeout(final_attempt_s):
                    await self.replay_once()
        if self.depth:
            log.error(
                "settle spool: entries left at shutdown, replayed at the next start",
                extra={"fields": {"depth": self.depth, "path": str(self.path)}},
            )
        if self._lockfile is not None:
            self._lockfile.close()

    # ------------------------------------------------------------ writing

    async def append(self, event: UsageEvent) -> None:
        """Durably record a settle Redis did not accept. Raises OSError if the disk does."""
        line = event.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_sync, line)
            self.depth += 1
        self.metrics.settle_spooled.inc()
        self._gauge()
        self._wake.set()

    def _append_sync(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def _rewrite(self, lines: list[str]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        dir_fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    def _gauge(self) -> None:
        self.metrics.settle_spool_depth.set(self.depth)

    # ------------------------------------------------------------ replay

    async def replay_once(self) -> bool:
        """Settle spooled entries in order until Redis refuses one. True when empty."""
        async with self._lock:
            lines = (await asyncio.to_thread(self.path.read_text, "utf-8")).splitlines(
                keepends=True
            )
        done = 0
        for line in lines:
            try:
                event = UsageEvent.model_validate_json(line)
            except ValidationError:
                self.metrics.settle_replayed.labels("corrupt").inc()
                log.error("settle spool: unreadable entry dropped", extra={"fields": {}})
                done += 1
                continue
            try:
                result = await self._replay(event)
            except (RedisError, OSError) as e:
                log.warning(
                    "settle spool: Redis still refusing",
                    extra={"fields": {"depth": self.depth - done, "error": type(e).__name__}},
                )
                break
            self.metrics.settle_replayed.labels(result).inc()
            log.info(
                "settle spool: replayed",
                extra={"fields": {"request_id": str(event.id), "result": result}},
            )
            done += 1
        if done:
            async with self._lock:
                current = (await asyncio.to_thread(self.path.read_text, "utf-8")).splitlines(
                    keepends=True
                )
                # Appends only add at the end, so the first `done` lines are the replayed.
                await asyncio.to_thread(self._rewrite, current[done:])
                self.depth = len(current) - done
            self._gauge()
        return done == len(lines)

    async def _replay(self, event: UsageEvent) -> str:
        age_s = (dt.datetime.now(dt.UTC) - event.created_at).total_seconds()
        # The done marker may have expired: then Postgres says whether it was settled.
        old = age_s > self.settled_ttl_s - 60
        if old and self.settled_check is not None and await self.settled_check(event):
            return "already_billed"
        result = await self.limiter.settle(event)
        return "settled" if result.settled else "duplicate"

    async def run(self) -> None:
        backoff = MIN_BACKOFF_S
        while True:
            if self.depth == 0:
                self._wake.clear()
                await self._wake.wait()
            try:
                empty = await self.replay_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("settle spool: replay failed")
                empty = False
            if empty:
                backoff = MIN_BACKOFF_S
                continue
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)
