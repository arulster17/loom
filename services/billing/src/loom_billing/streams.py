"""Redis stream helpers: ids, group state, decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError

StreamId = tuple[int, int]


def s(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else v


def parse_id(v: bytes | str) -> StreamId:
    ms, _, seq = s(v).partition("-")
    return int(ms), int(seq or 0)


def format_id(i: StreamId) -> str:
    return f"{i[0]}-{i[1]}"


def next_id(i: StreamId) -> StreamId:
    return (i[0], i[1] + 1)


async def ensure_group(redis: Redis, stream: str, group: str) -> None:
    """Create the consumer group at the start of the stream (so entries appended before
    billing first ran are still consumed). Idempotent."""
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


@dataclass(frozen=True)
class GroupState:
    last_delivered: StreamId
    pending: int
    oldest_pending: StreamId | None
    lag: int | None  # undelivered entries (Redis 7+), None when unknown


async def group_state(redis: Redis, stream: str, group: str) -> GroupState | None:
    """The group's position, read in this order: last-delivered first, then pending.
    Anything delivered after the first read has a larger id, so the pair never claims
    an unacked entry is consumed."""
    try:
        groups = await redis.xinfo_groups(stream)
    except ResponseError:
        return None  # no stream
    info: Mapping[Any, Any] | None = None
    for g in groups:
        if s(g["name"]) == group:
            info = g
            break
    if info is None:
        return None
    last = parse_id(info["last-delivered-id"])
    lag_raw = info.get("lag")
    summary = await redis.xpending(stream, group)
    pending = int(summary["pending"])
    oldest = parse_id(summary["min"]) if pending and summary["min"] else None
    return GroupState(last, pending, oldest, None if lag_raw is None else int(lag_raw))


async def consumed_through(redis: Redis, stream: str, group: str, upto: StreamId | None) -> bool:
    """True when every entry with id <= `upto` was delivered to the group and acked."""
    if upto is None:
        return True
    st = await group_state(redis, stream, group)
    if st is None or st.last_delivered < upto:
        return False
    return st.oldest_pending is None or st.oldest_pending > upto
