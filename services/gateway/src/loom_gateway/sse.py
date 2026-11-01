"""Server-sent events from the engine: splitting, usage tracking, client rewriting.

The engine sends usage on every chunk (the gateway forces `continuous_usage_stats`) and
a final usage-only chunk (`choices: []`). The gateway keeps the latest usage of what it
forwarded, so a stream cut at any point is billed the engine's own counts, and shows the
client only what it asked for (contracts §9): no usage at all, the final usage chunk
(`include_usage`, with `cost_micros` added, §12), or per-chunk usage as well.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

DONE = b"[DONE]"
DONE_EVENT = b"data: [DONE]\n\n"


async def sse_payloads(chunks: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """`data:` payloads of an SSE byte stream, one per event (comments and other fields
    dropped; multi-line data joined with newlines)."""
    buf = b""
    async for chunk in chunks:
        if b"\r" in chunk:
            chunk = chunk.replace(b"\r\n", b"\n")
        buf += chunk
        while True:
            end = buf.find(b"\n\n")
            if end < 0:
                break
            event, buf = buf[:end], buf[end + 2 :]
            payload = _data(event)
            if payload is not None:
                yield payload
    if buf.strip():
        payload = _data(buf)
        if payload is not None:
            yield payload


def _data(event: bytes) -> bytes | None:
    lines = [ln for ln in event.split(b"\n") if ln.startswith(b"data:")]
    if not lines:
        return None
    parts = [ln[5:][1:] if ln[5:6] == b" " else ln[5:] for ln in lines]
    return b"\n".join(parts)


def event_bytes(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


@dataclass
class Processed:
    out: bytes | None  # the event to send to the client, None to drop it
    usage: dict[str, Any] | None  # engine usage carried by this chunk
    done: bool = False  # the [DONE] marker
    content: bool = False  # a chunk with choices (a token, role or finish)
    error: bool = False  # an in-stream engine error


class ChunkRewriter:
    def __init__(
        self,
        *,
        include_usage: bool,
        continuous_usage: bool,
        cost_of: Callable[[dict[str, Any]], int],
    ) -> None:
        self.include_usage = include_usage
        self.continuous_usage = continuous_usage
        self.cost_of = cost_of

    def process(self, payload: bytes) -> Processed:
        if payload.strip() == DONE:
            return Processed(DONE_EVENT, None, done=True)
        try:
            obj = json.loads(payload)
        except ValueError:
            # Not JSON: pass through untouched; it carries no usage.
            return Processed(b"data: " + payload + b"\n\n", None)
        if not isinstance(obj, dict):
            return Processed(b"data: " + payload + b"\n\n", None)
        usage = obj.get("usage")
        usage = usage if isinstance(usage, dict) else None
        choices = obj.get("choices")
        if "error" in obj and not choices:
            return Processed(event_bytes(obj), usage, error=True)
        if not choices:
            # The final usage chunk (or an empty keep-alive chunk).
            if usage is None:
                return Processed(b"data: " + payload + b"\n\n", None)
            if not self.include_usage:
                return Processed(None, usage)
            obj["usage"] = client_usage(usage, self.cost_of(usage))
            return Processed(event_bytes(obj), usage)
        if "usage" not in obj or self.continuous_usage:
            return Processed(b"data: " + payload + b"\n\n", usage, content=True)
        if self.include_usage:
            obj["usage"] = None  # OpenAI: usage is null on every chunk but the last
        else:
            del obj["usage"]
        return Processed(event_bytes(obj), usage, content=True)


def token_counts(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    """(prompt, completion, cached prompt) from an engine usage object; zeros if absent."""
    if not usage:
        return 0, 0, 0

    def as_int(v: Any) -> int:
        return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0

    prompt = as_int(usage.get("prompt_tokens"))
    completion = as_int(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details")
    cached = as_int(details.get("cached_tokens")) if isinstance(details, dict) else 0
    return prompt, completion, min(cached, prompt)


def client_usage(usage: dict[str, Any], cost: int) -> dict[str, Any]:
    """The usage object the client sees: OpenAI's fields, cached tokens always present,
    and the charge in micro-dollars (contracts §12)."""
    prompt, completion, cached = token_counts(usage)
    out = dict(usage)
    out["prompt_tokens"] = prompt
    out["completion_tokens"] = completion
    out["total_tokens"] = prompt + completion
    details = out.get("prompt_tokens_details")
    out["prompt_tokens_details"] = {
        **(details if isinstance(details, dict) else {}),
        "cached_tokens": cached,
    }
    out["cost_micros"] = cost
    return out
