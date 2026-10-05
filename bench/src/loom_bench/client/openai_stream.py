"""Streaming client for OpenAI-compatible servers (vLLM, SGLang, the mock backend).

SSE is parsed by hand so each chunk is timestamped the moment its line arrives.
Prompt and completion text never go into logs or error strings.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from loom_bench.records import RequestRecord, RequestStatus

Endpoint = Literal["chat", "completions"]

_PATHS: dict[str, str] = {"chat": "/chat/completions", "completions": "/completions"}
_MAX_ERROR_CHARS = 500


@dataclass(slots=True)
class PreparedRequest:
    """A request body ready to send; `model` and streaming flags are added by `send`."""

    request_id: str
    endpoint: Endpoint
    payload: dict[str, Any]
    expected_prompt_tokens: int | None = None
    max_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def new_record(
    req: PreparedRequest, t0: float, scheduled_at_s: float | None = None
) -> RequestRecord:
    """A record for `req`, ABORTED until `send` completes it.

    Callers that may cancel the sending task create the record up front and pass
    it to `send`, so a cancelled request still yields its partial timings.
    """
    return RequestRecord(
        request_id=req.request_id,
        status=RequestStatus.ABORTED,
        sent_at_s=time.perf_counter() - t0,
        scheduled_at_s=scheduled_at_s,
        expected_prompt_tokens=req.expected_prompt_tokens,
        max_tokens=req.max_tokens,
        meta=dict(req.meta),
    )


async def send(
    http: httpx.AsyncClient,
    base_url: str,
    req: PreparedRequest,
    t0: float,
    *,
    scheduled_at_s: float | None = None,
    timeout_s: float,
    api_key: str | None = None,
    keep_output: bool = False,
    model: str | None = None,
    record: RequestRecord | None = None,
) -> RequestRecord:
    """Stream one request and return what the client observed.

    `base_url` includes the API prefix (e.g. ``http://host:8000/v1``). Times are
    `time.perf_counter()` seconds relative to `t0`. Failures are recorded, not
    raised; only cancellation propagates, after marking the record ABORTED.
    A `record` from `new_record` is filled in place (its `scheduled_at_s` wins).
    """
    rec = record if record is not None else new_record(req, t0, scheduled_at_s)
    payload = dict(req.payload)
    payload["stream"] = True
    payload["stream_options"] = {**payload.get("stream_options", {}), "include_usage": True}
    if model is not None:
        payload["model"] = model
    # identity encoding: a compressed body would be buffered by the decoder.
    headers = {"Accept": "text/event-stream", "Accept-Encoding": "identity"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + _PATHS[req.endpoint]

    rec.sent_at_s = time.perf_counter() - t0
    try:
        async with (
            asyncio.timeout(timeout_s),
            http.stream("POST", url, json=payload, headers=headers) as resp,
        ):
            rec.http_status = resp.status_code
            if resp.is_success:
                await _consume(resp, rec, t0, keep_output)
            else:
                rec.status = RequestStatus.ERROR
                rec.error = _error_from_body(await resp.aread())
    except TimeoutError:
        rec.status = RequestStatus.TIMEOUT
        rec.error = f"no complete response within {timeout_s:g}s"
    except httpx.TimeoutException as e:
        rec.status = RequestStatus.TIMEOUT
        rec.error = type(e).__name__
    except httpx.HTTPError as e:
        rec.status = RequestStatus.ERROR
        rec.error = f"{type(e).__name__}: {e}"[:_MAX_ERROR_CHARS]
    except asyncio.CancelledError:
        if rec.status is not RequestStatus.OK:  # a cancel while closing a finished stream
            rec.status = RequestStatus.ABORTED
            rec.error = "cancelled"
        raise
    finally:
        if rec.finished_at_s is None:
            rec.finished_at_s = time.perf_counter() - t0
    return rec


async def _consume(resp: httpx.Response, rec: RequestRecord, t0: float, keep_output: bool) -> None:
    parts: list[str] = []
    last_content_at: float | None = None
    done = False
    async for line in resp.aiter_lines():
        now = time.perf_counter() - t0
        if not line.startswith("data:"):
            continue  # blank separators, ": keepalive" comments, event:/id: fields
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            rec.finished_at_s = now
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            rec.status = RequestStatus.ERROR
            rec.error = "malformed stream chunk"
            return
        if not isinstance(chunk, dict):
            rec.status = RequestStatus.ERROR
            rec.error = "malformed stream chunk"
            return
        if "error" in chunk:
            # vLLM reports failures after the 200 header as an error chunk.
            rec.status = RequestStatus.ERROR
            rec.error = _error_message(chunk)
            return
        if usage := chunk.get("usage"):
            rec.prompt_tokens = usage.get("prompt_tokens")
            rec.completion_tokens = usage.get("completion_tokens")
            details = usage.get("prompt_tokens_details") or {}
            rec.cached_prompt_tokens = details.get("cached_tokens")
        has_content = False
        for choice in chunk.get("choices") or ():
            text, other = _delta(choice)
            if text or other:
                has_content = True
            if keep_output and text:
                parts.append(text)
            if choice.get("finish_reason"):
                rec.finish_reason = choice["finish_reason"]
        if has_content:
            if last_content_at is None:
                rec.first_token_at_s = now
            else:
                rec.itl_s.append(now - last_content_at)
            last_content_at = now

    if not done:
        rec.finished_at_s = time.perf_counter() - t0
        # Some servers omit [DONE]; a finish_reason still marks a complete answer.
        if rec.finish_reason is None:
            rec.status = RequestStatus.ERROR
            rec.error = "stream ended before completion"
            return
    rec.status = RequestStatus.OK
    if keep_output:
        rec.output_text = "".join(parts)


def _delta(choice: dict[str, Any]) -> tuple[str, bool]:
    """(content text, whether the chunk carried other generated tokens)."""
    if "text" in choice:  # completions endpoint
        return choice.get("text") or "", False
    delta = choice.get("delta") or {}
    other = bool(
        delta.get("tool_calls") or delta.get("reasoning_content") or delta.get("reasoning")
    )
    return delta.get("content") or "", other


def _error_from_body(body: bytes) -> str:
    try:
        obj = json.loads(body)
    except ValueError:
        return body.decode("utf-8", errors="replace")[:_MAX_ERROR_CHARS]
    return _error_message(obj)


def _error_message(obj: Any) -> str:
    """Message from OpenAI (`{"error": {"message"}}`), older vLLM or FastAPI error bodies."""
    msg: Any = obj
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict) and err.get("message"):
            msg = err["message"]
        elif isinstance(err, str) and err:
            msg = err
        elif obj.get("message"):
            msg = obj["message"]
        elif obj.get("detail"):
            msg = obj["detail"]
    return str(msg)[:_MAX_ERROR_CHARS]
