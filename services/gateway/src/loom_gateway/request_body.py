"""Request validation and rewriting for the engine (docs/phase1-contracts.md §6 step 2,
§8 step 2, §9 "Request rewriting").

Everything the gateway does not need to understand (tools, response_format, sampling
parameters, logprobs) passes through unchanged. What it does change:

- `max_tokens` is always explicit, at most LOOM_MAX_TOKENS_CAP, so the engine cannot
  generate more than was reserved;
- streams always ask for usage on every chunk (`include_usage`, `continuous_usage_stats`)
  so a stream cut at any point is billed the engine's own token counts;
- `stream_options` is dropped from non-streaming requests (vLLM refuses it there).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from loom_gateway.settings import COMPLETIONS_DEFAULT_MAX_TOKENS
from loom_platform import errors
from loom_platform.pricing import Price, max_cost_micros, prompt_tokens_upper_bound

Endpoint = Literal["chat", "completions"]
MAX_SEQUENCES = 16
# Parameters that make the engine generate more than it returns (cost not covered by the
# reservation's `n * max_tokens`), refused rather than passed through.
_REFUSED = ("best_of", "use_beam_search")


@dataclass(frozen=True)
class Prepared:
    endpoint: Endpoint
    model_id: str
    stream: bool
    client_include_usage: bool
    client_continuous_usage: bool
    max_tokens: int
    sequences: int  # choices the engine generates (n * prompts)
    prompt_upper: int  # per prompt
    tpm_estimate: int
    max_cost_micros: int
    upstream_body: bytes


def parse_json_object(body: bytes) -> dict[str, Any]:
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise errors.invalid_request("The request body is not valid JSON.") from None
    if not isinstance(data, dict):
        raise errors.invalid_request("The request body must be a JSON object.")
    return data


def model_field(data: dict[str, Any]) -> str:
    model = data.get("model")
    if not isinstance(model, str) or not model:
        raise errors.invalid_request("'model' is required and must be a string.", "model")
    return model


def _int_field(data: dict[str, Any], name: str, *, minimum: int = 1) -> int | None:
    v = data.get(name)
    if v is None:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        raise errors.invalid_request(f"'{name}' must be an integer >= {minimum}.", name)
    return v


def _bool_field(data: dict[str, Any], name: str) -> bool:
    v = data.get(name, False)
    if v is None:
        return False
    if not isinstance(v, bool):
        raise errors.invalid_request(f"'{name}' must be a boolean.", name)
    return v


def _prompt_count(prompt: Any) -> int:
    """Prompts in a /v1/completions request: a string, a list of strings, a token list,
    or a list of token lists."""
    if isinstance(prompt, str):
        return 1
    if isinstance(prompt, list) and prompt:
        if all(isinstance(p, str) for p in prompt):
            return len(prompt)
        if all(isinstance(p, int) and not isinstance(p, bool) for p in prompt):
            return 1
        if all(isinstance(p, list) and p for p in prompt):
            return len(prompt)
    raise errors.invalid_request(
        "'prompt' must be a string, a non-empty list of strings, or token ids.", "prompt"
    )


def prepare(
    endpoint: Endpoint,
    data: dict[str, Any],
    *,
    body_bytes: int,
    model_id: str,
    price: Price,
    max_context: int,
    max_tokens_cap: int,
) -> Prepared:
    """Validate a parsed body (400s) and build the engine request and its reservation."""
    for name in _REFUSED:
        if data.get(name) not in (None, False, 1):
            raise errors.invalid_request(f"'{name}' is not supported.", name)
    stream = _bool_field(data, "stream")
    n = _int_field(data, "n") or 1
    if endpoint == "chat":
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise errors.invalid_request("'messages' must be a non-empty list.", "messages")
        if not all(isinstance(m, dict) for m in messages):
            raise errors.invalid_request("Each message must be an object.", "messages")
        prompts = 1
        requested = [
            v
            for v in (_int_field(data, "max_completion_tokens"), _int_field(data, "max_tokens"))
            if v is not None
        ]
        max_tokens = min(requested) if requested else max_tokens_cap
    else:
        prompts = _prompt_count(data.get("prompt"))
        given = _int_field(data, "max_tokens")
        max_tokens = given if given is not None else COMPLETIONS_DEFAULT_MAX_TOKENS
        requested = [] if given is None else [given]
        if stream and prompts > 1:
            raise errors.invalid_request("Streaming supports a single prompt.", "prompt")
    for v in requested:
        if v > max_tokens_cap:
            raise errors.invalid_request(
                f"max_tokens must be at most {max_tokens_cap}.", "max_tokens"
            )
    sequences = n * prompts
    if sequences > MAX_SEQUENCES:
        raise errors.invalid_request(
            f"At most {MAX_SEQUENCES} choices per request (n times prompts).", "n"
        )

    opts = data.get("stream_options")
    if opts is not None and not isinstance(opts, dict):
        raise errors.invalid_request("'stream_options' must be an object.", "stream_options")
    opts = opts or {}
    client_include_usage = stream and _bool_field(opts, "include_usage")
    client_continuous = client_include_usage and _bool_field(opts, "continuous_usage_stats")

    out = dict(data)
    if endpoint == "chat":
        if "max_completion_tokens" in out and out["max_completion_tokens"] is not None:
            out["max_completion_tokens"] = max_tokens
        out["max_tokens"] = max_tokens
    else:
        out["max_tokens"] = max_tokens
    if stream:
        out["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
    else:
        out.pop("stream_options", None)

    prompt_upper = prompt_tokens_upper_bound(body_bytes, max_context)
    max_cost = sequences * max_cost_micros(price, prompt_upper, max_tokens, max_context)
    return Prepared(
        endpoint=endpoint,
        model_id=model_id,
        stream=stream,
        client_include_usage=client_include_usage,
        client_continuous_usage=client_continuous,
        max_tokens=max_tokens,
        sequences=sequences,
        prompt_upper=prompt_upper,
        tpm_estimate=prompt_upper * prompts,
        max_cost_micros=max_cost,
        upstream_body=json.dumps(out, ensure_ascii=False, separators=(",", ":")).encode(),
    )
