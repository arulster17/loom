"""Shared record types. Every load generator emits RequestRecord rows; metrics,
storage and reports consume them. Keep this module dependency-free."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RequestStatus(StrEnum):
    OK = "ok"
    ERROR = "error"  # non-2xx or malformed stream
    TIMEOUT = "timeout"
    ABORTED = "aborted"  # client cancelled mid-stream (e.g. run deadline)


class Market(StrEnum):
    """How a GPU host is billed. `local` = not billed by Loom (your own box, mock)."""

    ON_DEMAND = "on_demand"
    SPOT = "spot"
    COMMITTED_1Y = "committed_1y"
    LOCAL = "local"


class LoadMode(StrEnum):
    OPEN_LOOP = "open_loop"  # fixed arrival schedule, for latency SLOs
    CLOSED_LOOP = "closed_loop"  # fixed concurrency, for saturation


@dataclass(slots=True)
class RequestRecord:
    """One request as observed by the client.

    Times are seconds relative to the run's t0 (monotonic clock).
    Token counts come from the server's `usage` block when present; the
    `expected_*` fields are what the workload asked for.
    """

    request_id: str
    status: RequestStatus
    sent_at_s: float
    scheduled_at_s: float | None = None  # open loop: planned arrival; None for closed loop
    first_token_at_s: float | None = None
    finished_at_s: float | None = None
    # Gaps between consecutive content chunks after the first (seconds).
    itl_s: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    expected_prompt_tokens: int | None = None
    max_tokens: int | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None
    warmup: bool = False
    # Optional, only kept when the experiment asks (sanity checks). Never for user traffic.
    output_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is RequestStatus.OK

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_at_s is None:
            return None
        return self.first_token_at_s - self.sent_at_s

    @property
    def e2e_s(self) -> float | None:
        if self.finished_at_s is None:
            return None
        return self.finished_at_s - self.sent_at_s

    @property
    def queue_delay_s(self) -> float | None:
        """Client-side lag between planned and actual send (open loop only)."""
        if self.scheduled_at_s is None:
            return None
        return self.sent_at_s - self.scheduled_at_s

    @property
    def tpot_s(self) -> float | None:
        """Mean time per output token after the first."""
        ttft, e2e, n = self.ttft_s, self.e2e_s, self.completion_tokens
        if ttft is None or e2e is None or n is None or n < 2:
            return None
        return (e2e - ttft) / (n - 1)

    def to_row(self) -> dict[str, Any]:
        """Flat dict for Parquet/CSV; derived latencies included."""
        row = asdict(self)
        row["status"] = self.status.value
        row["ttft_s"] = self.ttft_s
        row["e2e_s"] = self.e2e_s
        row["tpot_s"] = self.tpot_s
        row["queue_delay_s"] = self.queue_delay_s
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> RequestRecord:
        derived = {"ttft_s", "e2e_s", "tpot_s", "queue_delay_s"}
        kwargs = {k: v for k, v in row.items() if k not in derived}
        kwargs["status"] = RequestStatus(kwargs["status"])
        kwargs["itl_s"] = list(kwargs.get("itl_s") or [])
        kwargs["meta"] = dict(kwargs.get("meta") or {})
        return cls(**kwargs)
