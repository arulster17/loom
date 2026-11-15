"""Request pricing from the model registry, in integer micro-dollars.

Prices are `config/models.yaml` `pricing`: micro-dollars per 1M input, cached-input and
output tokens. The gateway, the billing worker and the Lab all read them through
`loom_bench.registry`, so they cannot drift.

Rounding rule: a request's cost is the exact sum of its three parts, rounded **up** to
a whole micro-dollar, **once per request**:

    cost = ceil(((prompt - cached) * input + cached * cached_input + completion * output) / 1e6)

Why: per-request amounts are often fractions of a micro-dollar (one prompt token of
Qwen3 8B is 0.077 micro-dollars). Rounding each part separately could charge up to 3
extra micro-dollars; rounding half-up would make tiny requests free. Rounding the sum up
never charges less than the list price and never more than one micro-dollar
($0.000001) above it per request. `max_cost_micros` uses the same function on an upper
bound of the tokens, and the function is monotone in every count, so a request's
reservation always covers its final cost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loom_bench.registry import ModelSpec, Registry

TOKENS_PER_MTOK = 1_000_000

# Upper bound on the tokens a chat template adds around the request's own text (role
# headers, special tokens, the tool-use preamble a template renders when `tools` is
# set). Generous on purpose: at the 70B's input price it is $0.0006 of reservation.
TEMPLATE_OVERHEAD_TOKENS = 1024

_VERSION_RE = re.compile(r"^(\d+):(\d+):(\d+)$")


class UnpricedModel(LookupError):
    """The registry row has no price, so it cannot be served (maps to 404)."""


@dataclass(frozen=True)
class Price:
    """Micro-dollars per 1M tokens."""

    input_per_mtok: int
    cached_input_per_mtok: int
    output_per_mtok: int

    def __post_init__(self) -> None:
        for v in (self.input_per_mtok, self.cached_input_per_mtok, self.output_per_mtok):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"prices are non-negative integer micros, got {v!r}")
        if self.cached_input_per_mtok > self.input_per_mtok:
            raise ValueError("cached input price must not exceed the input price")

    @property
    def version(self) -> str:
        """`usage_events.price_version`: self-describing, so any event can be re-priced."""
        return f"{self.input_per_mtok}:{self.cached_input_per_mtok}:{self.output_per_mtok}"

    @classmethod
    def from_version(cls, version: str) -> Price:
        m = _VERSION_RE.match(version)
        if m is None:
            raise ValueError(f"not a price version: {version!r}")
        return cls(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    @classmethod
    def of(cls, model: ModelSpec) -> Price:
        if model.pricing is None:
            raise UnpricedModel(model.id)
        p = model.pricing
        return cls(p.input_per_mtok, p.cached_input_per_mtok, p.output_per_mtok)


def price_for(registry: Registry, model_id: str) -> Price:
    """The price of a servable model. KeyError if unknown, UnpricedModel if unpriced."""
    return Price.of(registry.get(model_id))


def servable(registry: Registry) -> list[ModelSpec]:
    """Models the public API offers: priced and not disabled."""
    return [m for m in registry.models if m.pricing is not None and m.status != "disabled"]


def _ceil_div(n: int, d: int) -> int:
    return -(-n // d)


def cost_micros(
    price: Price, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0
) -> int:
    """A request's charge. `cached_prompt_tokens` is the part of `prompt_tokens` served
    from the prefix cache (OpenAI's `prompt_tokens_details.cached_tokens`)."""
    for v in (prompt_tokens, completion_tokens, cached_prompt_tokens):
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise ValueError(f"token counts are non-negative ints, got {v!r}")
    if cached_prompt_tokens > prompt_tokens:
        raise ValueError("cached_prompt_tokens cannot exceed prompt_tokens")
    numerator = (
        (prompt_tokens - cached_prompt_tokens) * price.input_per_mtok
        + cached_prompt_tokens * price.cached_input_per_mtok
        + completion_tokens * price.output_per_mtok
    )
    return _ceil_div(numerator, TOKENS_PER_MTOK)


def prompt_tokens_upper_bound(
    body_bytes: int, max_context: int, *, template_json: bool = False
) -> int:
    """Upper bound on a request's prompt tokens, without tokenizing.

    Byte-level BPE tokenizers (Qwen3, Llama 3) never produce more tokens than the
    UTF-8 bytes of the text, and the JSON request body holds every byte of text the
    template will see verbatim (JSON escaping only adds bytes). The template's own tokens
    are bounded by TEMPLATE_OVERHEAD_TOKENS. The engine refuses prompts over the context.
    Not valid for image inputs (no vision model is served).

    `template_json`: the request carries values the chat template re-serialises rather
    than copies (`tools` and similar, rendered with `tojson`). Those can come out far
    longer than they went in (`9e15` becomes `9000000000000000.0`, one token per digit
    on Qwen3; `tojson(indent=4)` adds indentation per nesting level), so no factor of the
    body size bounds them: the bound is then the context, which the engine enforces.
    """
    if body_bytes < 0 or max_context < 1:
        raise ValueError("body_bytes >= 0 and max_context >= 1 required")
    if template_json:
        return max_context
    return min(body_bytes + TEMPLATE_OVERHEAD_TOKENS, max_context)


def max_cost_micros(
    price: Price, prompt_tokens_upper: int, max_completion_tokens: int, max_context: int
) -> int:
    """The most a request can cost: what the gateway reserves before dispatch.

    Maximises cost over every outcome the engine allows: prompt p <= prompt_tokens_upper,
    completion c <= max_completion_tokens, p + c <= max_context, nothing cached. The cost
    is linear in (p, c), so the maximum is at a vertex: fill the dearer side first.
    """
    if min(prompt_tokens_upper, max_completion_tokens) < 0 or max_context < 1:
        raise ValueError("token bounds must be >= 0 and max_context >= 1")
    p_cap = min(prompt_tokens_upper, max_context)
    c_cap = min(max_completion_tokens, max_context)
    if price.output_per_mtok >= price.input_per_mtok:
        c = c_cap
        p = min(p_cap, max_context - c)
    else:
        p = p_cap
        c = min(c_cap, max_context - p)
    return cost_micros(price, p, c)
