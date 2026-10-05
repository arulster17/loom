"""Small async OpenAI-compatible client for evals (non-streaming).

Every request is deterministic by default (temperature 0, fixed seed). Server
errors and timeouts are retried with exponential backoff; other 4xx errors are
raised at once. Prompts and outputs never go into logs or exception messages.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import httpx

_RETRY_STATUSES = frozenset({408, 429})
_MAX_ERROR_CHARS = 300
_MAX_BACKOFF_S = 30.0


class EvalRequestError(RuntimeError):
    """A request failed for good (non-retryable status, or retries exhausted)."""

    def __init__(self, message: str, *, status: int | None = None, attempts: int = 1) -> None:
        super().__init__(message)
        self.status = status
        self.attempts = attempts


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: str  # raw JSON text as returned by the server


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    logprobs: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompletionResult:
    text: str
    finish_reason: str | None
    logprobs: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)


def _retryable(status: int) -> bool:
    return status >= 500 or status in _RETRY_STATUSES


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        message = body["error"]["message"] if isinstance(body.get("error"), dict) else body
    except (ValueError, KeyError, TypeError, AttributeError):
        message = response.text
    return str(message)[:_MAX_ERROR_CHARS]


class EvalClient:
    """Client for one endpoint and served model.

    `base_url` includes the API prefix (``http://host:8000/v1``). `extra_body`
    is merged into every request body (e.g. ``{"chat_template_kwargs":
    {"enable_thinking": False}}``); per-call parameters override it.
    `concurrency` bounds in-flight requests across all callers of this client.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        concurrency: int = 16,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        backoff_s: float = 1.0,
        seed: int = 0,
        extra_body: Mapping[str, Any] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.seed = seed
        self.extra_body = dict(extra_body or {})
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sem = asyncio.Semaphore(concurrency)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
            limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    def _body(self, params: Mapping[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "temperature": 0.0, "seed": self.seed}
        body.update(self.extra_body)
        body.update({k: v for k, v in params.items() if v is not None})
        body["stream"] = False
        return body

    async def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        """POST `body` to `path`, retrying 5xx / 408 / 429 / timeouts / connection errors."""
        attempt = 0
        while True:
            attempt += 1
            status: int | None = None
            try:
                async with self._sem:
                    response = await self._http.post(path, json=body)
                status = response.status_code
                if status < 400:
                    data = response.json()
                    if not isinstance(data, dict):
                        raise EvalRequestError(f"{path}: response is not a JSON object")
                    return data
                reason = f"HTTP {status}: {_error_message(response)}"
                if not _retryable(status):
                    raise EvalRequestError(f"{path}: {reason}", status=status, attempts=attempt)
            except httpx.TimeoutException:
                reason = "timeout"
            except httpx.TransportError as e:
                reason = f"transport error ({type(e).__name__})"
            except ValueError as e:
                raise EvalRequestError(
                    f"{path}: invalid JSON response", status=status, attempts=attempt
                ) from e
            if attempt > self.max_retries:
                raise EvalRequestError(
                    f"{path}: giving up after {attempt} attempts, last: {reason}",
                    status=status,
                    attempts=attempt,
                )
            await asyncio.sleep(min(_MAX_BACKOFF_S, self.backoff_s * 2 ** (attempt - 1)))

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        **params: Any,
    ) -> ChatResult:
        body = self._body({"messages": list(messages), "max_tokens": max_tokens, **params})
        data = await self.post("/chat/completions", body)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            calls = tuple(
                ToolCall(name=c["function"]["name"], arguments=c["function"].get("arguments") or "")
                for c in message.get("tool_calls") or ()
            )
            return ChatResult(
                text=message.get("content") or "",
                finish_reason=choice.get("finish_reason"),
                tool_calls=calls,
                logprobs=choice.get("logprobs"),
                usage=data.get("usage") or {},
            )
        except (KeyError, IndexError, TypeError) as e:
            raise EvalRequestError("/chat/completions: malformed response") from e

    async def complete(self, prompt: str, *, max_tokens: int, **params: Any) -> CompletionResult:
        body = self._body({"prompt": prompt, "max_tokens": max_tokens, **params})
        data = await self.post("/completions", body)
        try:
            choice = data["choices"][0]
            return CompletionResult(
                text=choice.get("text") or "",
                finish_reason=choice.get("finish_reason"),
                logprobs=choice.get("logprobs"),
                usage=data.get("usage") or {},
            )
        except (KeyError, IndexError, TypeError) as e:
            raise EvalRequestError("/completions: malformed response") from e
