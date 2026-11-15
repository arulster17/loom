"""Usage events: what the gateway's settle call appends to the `loom:usage` stream and
the billing worker inserts into `usage_events` (docs/phase1-contracts.md, "Usage stream").

Stream entries are flat string fields. `v` is the message format version; consumers
must refuse a version they do not know rather than guess.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal

import psycopg
from pydantic import BaseModel, ConfigDict, Field, model_validator

from loom_platform.pricing import Price, cost_micros

EVENT_VERSION = "1"
UsageStatus = Literal["ok", "aborted", "error"]
NonNeg = Annotated[int, Field(ge=0)]


class UsageEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: uuid.UUID  # request id (X-Loom-Request-Id), the idempotency key
    org_id: uuid.UUID
    api_key_id: uuid.UUID
    model_id: str
    price_version: str
    cloud: str  # 'local' for the M1 mock, 'runpod', 'aws'
    region: str | None = None
    prompt_tokens: NonNeg
    completion_tokens: NonNeg
    cached_prompt_tokens: NonNeg = 0
    latency_ms: NonNeg
    ttft_ms: NonNeg | None = None
    status: UsageStatus
    cost_micros: NonNeg
    created_at: dt.datetime  # when the gateway received the request; set once
    # Added by settle.lua: the org's cumulative settled usage including this event.
    org_usage_total_micros: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> UsageEvent:
        if self.cached_prompt_tokens > self.prompt_tokens:
            raise ValueError("cached_prompt_tokens > prompt_tokens")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        expected = cost_micros(
            Price.from_version(self.price_version),
            self.prompt_tokens,
            self.completion_tokens,
            self.cached_prompt_tokens,
        )
        if self.cost_micros != expected:
            raise ValueError(
                f"cost_micros {self.cost_micros} does not match price_version "
                f"{self.price_version} for these tokens ({expected})"
            )
        return self

    def to_stream_fields(self) -> list[str]:
        """Name/value pairs for settle.lua (it appends org_usage_total_micros itself)."""
        fields: dict[str, str] = {
            "v": EVENT_VERSION,
            "id": str(self.id),
            "org_id": str(self.org_id),
            "api_key_id": str(self.api_key_id),
            "model_id": self.model_id,
            "price_version": self.price_version,
            "cloud": self.cloud,
            "region": self.region or "",
            "prompt_tokens": str(self.prompt_tokens),
            "completion_tokens": str(self.completion_tokens),
            "cached_prompt_tokens": str(self.cached_prompt_tokens),
            "latency_ms": str(self.latency_ms),
            "ttft_ms": "" if self.ttft_ms is None else str(self.ttft_ms),
            "status": self.status,
            "cost_micros": str(self.cost_micros),
            "created_at": self.created_at.astimezone(dt.UTC).isoformat(timespec="microseconds"),
        }
        return [x for kv in fields.items() for x in kv]

    @classmethod
    def from_stream_fields(cls, fields: Mapping[bytes | str, bytes | str]) -> UsageEvent:
        f = {_s(k): _s(v) for k, v in fields.items()}
        version = f.pop("v", None)
        if version != EVENT_VERSION:
            raise ValueError(f"unknown usage event version {version!r}")
        return cls.model_validate(
            {
                **f,
                "region": f.get("region") or None,
                "ttft_ms": f.get("ttft_ms") or None,
                "org_usage_total_micros": f.get("org_usage_total_micros") or None,
            }
        )


def _s(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


_INSERT = """
INSERT INTO usage_events (
    id, org_id, api_key_id, model_id, price_version, cloud, region, prompt_tokens,
    completion_tokens, cached_prompt_tokens, latency_ms, ttft_ms, status, cost_micros,
    created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id, created_at) DO NOTHING
RETURNING id
"""


async def insert_usage_events(conn: psycopg.AsyncConnection, events: Sequence[UsageEvent]) -> int:
    """Insert events, skipping ones already stored. Returns how many were new.
    Runs in the caller's transaction."""
    inserted = 0
    async with conn.cursor() as cur:
        for e in events:
            await cur.execute(
                _INSERT,
                (
                    e.id, e.org_id, e.api_key_id, e.model_id, e.price_version, e.cloud,
                    e.region, e.prompt_tokens, e.completion_tokens, e.cached_prompt_tokens,
                    e.latency_ms, e.ttft_ms, e.status, e.cost_micros, e.created_at,
                ),
            )  # fmt: skip
            if await cur.fetchone() is not None:
                inserted += 1
    return inserted
