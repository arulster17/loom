"""OpenAI-shaped API errors (docs/phase1-contracts.md, "Errors").

Body: {"error": {"message": str, "type": str, "param": str | null, "code": str}}.
The OpenAI SDKs pick the exception class from the HTTP status, so the status codes are
the contract; `code` is for programs that need the reason.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from loom_platform.limiter import ReserveResult


@dataclass
class ApiError(Exception):
    status: int
    type: str
    code: str
    message: str
    param: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def body(self) -> dict[str, dict[str, str | None]]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }

    def sse_chunk(self) -> bytes:
        """The error as a final SSE event mid-stream (followed by `data: [DONE]`)."""
        return b"data: " + json.dumps(self.body()).encode() + b"\n\n"


def invalid_request(
    message: str, param: str | None = None, code: str = "invalid_request"
) -> ApiError:
    return ApiError(400, "invalid_request_error", code, message, param)


def invalid_api_key() -> ApiError:
    return ApiError(401, "invalid_request_error", "invalid_api_key", "Invalid or revoked API key.")


def org_not_active(status: str) -> ApiError:
    if status == "pending_approval":
        return ApiError(
            403,
            "permission_error",
            "org_pending_approval",
            "This organization is awaiting approval by a Loom admin.",
        )
    return ApiError(403, "permission_error", "org_suspended", f"This organization is {status}.")


def insufficient_balance(available_micros: int, needed_micros: int) -> ApiError:
    return ApiError(
        402,
        "insufficient_quota",
        "insufficient_balance",
        f"Insufficient credit: this request may cost up to {needed_micros} micro-dollars and "
        f"{max(available_micros, 0)} are available. Add credit or lower max_tokens.",
    )


def key_spend_limit() -> ApiError:
    return ApiError(
        402, "insufficient_quota", "key_spend_limit", "This API key has reached its spend limit."
    )


def model_not_found(model: str) -> ApiError:
    return ApiError(
        404,
        "invalid_request_error",
        "model_not_found",
        f"The model {model!r} does not exist.",
        "model",
    )


def request_too_large(limit_bytes: int) -> ApiError:
    return ApiError(
        413, "invalid_request_error", "request_too_large", f"Request body over {limit_bytes} bytes."
    )


def rate_limited(reason: str, retry_after_ms: int) -> ApiError:
    seconds = max(1, math.ceil(retry_after_ms / 1000))
    code = "concurrency_limit_exceeded" if reason.endswith("concurrency") else "rate_limit_exceeded"
    scope = "API key" if reason.startswith("key") else "organization"
    what = {
        "rpm": "requests per minute",
        "tpm": "tokens per minute",
        "concurrency": "concurrent requests",
    }[reason.split("_", 1)[1]]
    return ApiError(
        429,
        "rate_limit_error",
        code,
        f"Rate limit reached for {what} on this {scope}. Retry in {seconds} s.",
        headers={"Retry-After": str(seconds)},
    )


def model_unavailable(model: str, retry_after_s: int, starting: bool) -> ApiError:
    why = "it is starting up" if starting else "no GPU capacity is available for it right now"
    return ApiError(
        503,
        "server_error",
        "model_unavailable",
        f"The model {model!r} is temporarily unavailable: {why}. Retry in about {retry_after_s} s.",
        headers={"Retry-After": str(retry_after_s)},
    )


def upstream_lost() -> ApiError:
    return ApiError(502, "server_error", "upstream_lost", "The model server stopped responding.")


def service_unavailable() -> ApiError:
    """Redis (balances, limits, auth cache) unreachable or frozen past its timeout."""
    return ApiError(
        503,
        "server_error",
        "service_unavailable",
        "The service is temporarily unavailable. Retry shortly.",
        headers={"Retry-After": "1"},
    )


def server_shutdown() -> ApiError:
    """A stream cut because this gateway is shutting down (deploy or scale-in); the
    tokens already streamed are billed. Retrying reaches another gateway."""
    return ApiError(
        503, "server_error", "server_shutdown", "The server is restarting. Retry the request."
    )


def from_reserve(result: ReserveResult, needed_micros: int) -> ApiError:
    """The error for a refused reservation. 'noinit' is not an error: load and retry."""
    r = result.reason
    if r == "insufficient_balance":
        return insufficient_balance(result.available_micros, needed_micros)
    if r == "key_spend_limit":
        return key_spend_limit()
    if r == "duplicate_request":
        return invalid_request("Duplicate request id.", code="duplicate_request")
    if r in ("key_rpm", "org_rpm", "key_tpm", "org_tpm", "key_concurrency", "org_concurrency"):
        return rate_limited(r, result.retry_after_ms)
    raise ValueError(f"no API error for reserve reason {r!r}")
