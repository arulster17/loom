"""Turn a workload profile into concrete requests.

Generation is a pure function of (profile, tokenizer, n, seed), and the first k
requests do not depend on n, so a longer run extends a shorter one.
`expected_prompt_tokens` counts message content only; the server's usage adds
chat-template tokens.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from loom_bench.client.openai_stream import Endpoint, PreparedRequest
from loom_bench.loadgen.arrivals import read_trace
from loom_bench.tokenize import Tokenizer
from loom_bench.workloads.profiles import (
    ChatDatasetProfile,
    CodeCompletionProfile,
    LongGenerationProfile,
    NeedleProfile,
    SharedPrefixProfile,
    SyntheticProfile,
    TraceProfile,
    WorkloadProfile,
)
from loom_bench.workloads.sharegpt import Message, read_conversations, to_chat


def build_requests(
    profile: WorkloadProfile, tokenizer: Tokenizer, n: int, rng_seed: int | None = None
) -> list[PreparedRequest]:
    """`n` requests for `profile`; `rng_seed` defaults to the profile's `seed`."""
    if n < 0:
        raise ValueError("n must be >= 0")
    seed = profile.seed if rng_seed is None else rng_seed
    rng = random.Random(seed)
    builder = _BUILDERS[type(profile)]
    reqs = builder(profile, tokenizer, n, rng)
    for i, req in enumerate(reqs):
        req.request_id = f"{profile.name}-s{seed}-{i:06d}"
    return reqs


def _request(
    profile: WorkloadProfile,
    endpoint: Endpoint,
    prompt: str | list[Message],
    prompt_tokens: int,
    max_tokens: int,
    ignore_eos: bool,
    meta: dict[str, Any] | None = None,
) -> PreparedRequest:
    """A request without its id; `build_requests` numbers them."""
    payload: dict[str, Any] = {"max_tokens": max_tokens, "temperature": profile.temperature}
    if isinstance(prompt, list):
        payload["messages"] = prompt
    elif endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    if ignore_eos:
        payload["ignore_eos"] = True
    return PreparedRequest(
        request_id="",
        endpoint=endpoint,
        payload=payload,
        expected_prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        meta=meta or {},
    )


def _jitter(rng: random.Random, base: int, ratio: float) -> int:
    lo = max(1, int(base * (1 - ratio)))
    hi = max(lo, int(base * (1 + ratio)))
    return rng.randint(lo, hi)


def _fit(build: Callable[[int], str], target: int, tok: Tokenizer) -> str:
    """Text from `build(filler_tokens)` whose total count hits `target` (exact for
    additive tokenizers, best effort for BPE where joins can merge tokens)."""
    h = max(target - tok.count(build(0)), 0)
    text = build(h)
    for _ in range(4):
        diff = tok.count(text) - target
        if diff == 0 or h - diff < 0:
            break
        h -= diff
        text = build(h)
    return text


def _synthetic(
    p: SyntheticProfile, tok: Tokenizer, n: int, rng: random.Random
) -> list[PreparedRequest]:
    out = []
    for _ in range(n):
        in_len = _jitter(rng, p.input_len, p.range_ratio)
        out_len = _jitter(rng, p.output_len, p.range_ratio)
        text = tok.random_text(in_len, rng)
        out.append(_request(p, p.endpoint, text, in_len, out_len, p.ignore_eos))
    return out


def _shared_prefix(
    p: SharedPrefixProfile, tok: Tokenizer, n: int, rng: random.Random
) -> list[PreparedRequest]:
    prefix_len = p.prefix_len
    prefixes = [tok.random_text(prefix_len, rng) for _ in range(p.num_prefix_groups)]
    out = []
    for _ in range(n):
        group = rng.randrange(p.num_prefix_groups)
        in_len = _jitter(rng, p.input_len, p.range_ratio)
        out_len = _jitter(rng, p.output_len, p.range_ratio)
        suffix = tok.random_text(in_len - prefix_len, rng)
        # The system message is always present so template overhead is equal at every share.
        messages = [
            {"role": "system", "content": prefixes[group]},
            {"role": "user", "content": suffix},
        ]
        meta = {"prefix_group": group, "prefix_tokens": prefix_len}
        out.append(_request(p, "chat", messages, in_len, out_len, p.ignore_eos, meta))
    return out


def _chat_dataset(
    p: ChatDatasetProfile, tok: Tokenizer, n: int, rng: random.Random
) -> list[PreparedRequest]:
    if n == 0:
        return []
    convs = read_conversations(p.path)
    order = list(range(len(convs)))
    rng.shuffle(order)
    out = []
    for idx in order:
        chat = to_chat(convs[idx])
        if chat is None:
            continue
        messages, reference = chat
        in_len = sum(tok.count(m["content"]) for m in messages)
        ref_len = tok.count(reference)
        if (p.max_input_len and in_len > p.max_input_len) or ref_len < p.min_output_len:
            continue
        out_len = min(ref_len, p.max_output_len)
        meta = {"conversation_index": idx, "turns": len(messages)}
        out.append(_request(p, "chat", messages, in_len, out_len, p.ignore_eos, meta))
        if len(out) == n:
            return out
    raise ValueError(f"{p.name}: only {len(out)} usable conversations in {p.path}, need {n}")


_NEEDLE_ADJ = ("amber", "cobalt", "crimson", "golden", "silver", "violet", "emerald", "scarlet")
_NEEDLE_NOUN = ("falcon", "harbor", "lantern", "meadow", "orchid", "summit", "tiger", "willow")


def _needle(p: NeedleProfile, tok: Tokenizer, n: int, rng: random.Random) -> list[PreparedRequest]:
    out = []
    for i in range(n):
        depth = p.depths[i % len(p.depths)]
        key = f"{rng.choice(_NEEDLE_ADJ)} {rng.choice(_NEEDLE_NOUN)}"
        answer = str(rng.randrange(1_000_000, 10_000_000))
        needle = f"The secret code for the {key} is {answer}."
        question = f"What is the secret code for the {key}? Answer with the code only."
        filler_seed = rng.getrandbits(64)

        def build(h: int, depth=depth, needle=needle, question=question, seed=filler_seed) -> str:
            r = random.Random(seed)
            before = tok.random_text(round(depth * h), r)
            after = tok.random_text(h - round(depth * h), r)
            haystack = " ".join(s for s in (before, needle, after) if s)
            return f"{haystack}\n\n{question}"

        text = _fit(build, p.context_len, tok)
        meta = {"depth": depth, "expected_answer": answer}
        out.append(_request(p, p.endpoint, text, tok.count(text), p.output_len, False, meta))
    return out


_IDENTS = (
    "value", "count", "items", "result", "config", "index", "buffer", "total",
    "offset", "request", "cache", "token", "batch", "layer", "state", "queue",
)  # fmt: skip
_CODE_LINES = (
    "def {a}_{b}({c}, {d}):\n",
    '    """Return the {a} for {b}."""\n',
    "    {a} = {b} + {n}\n",
    "    {a} = {b}.get({c}, {n})\n",
    "    if {a} > {n}:\n        return {b}\n",
    "    for {a} in range({n}):\n        {b}.append({a} * {n})\n",
    "    while {a} < {n}:\n        {a} += 1\n",
    "    return {a}\n\n\n",
    "class {A}{B}:\n    def __init__(self, {a}):\n        self.{a} = {a}\n\n",
    "    # {a} {b} must stay below {n}\n",
)


def _code_text(n_tokens: int, tok: Tokenizer, rng: random.Random) -> str:
    def fill(template: str) -> str:
        a, b, c, d = (rng.choice(_IDENTS) for _ in range(4))
        return template.format(
            a=a, b=b, c=c, d=d, A=a.capitalize(), B=b.capitalize(), n=rng.randrange(1, 512)
        )

    text = fill("# {a}_{b}.py\nimport os\nimport sys\n\n\n")
    # Per-line counts overestimate (whitespace merges across line joins), so recount.
    while (have := tok.count(text)) < n_tokens:
        while have < n_tokens + 16:
            line = fill(rng.choice(_CODE_LINES))
            text += line
            have += tok.count(line)
    return tok.truncate(text, n_tokens)


def _code_completion(
    p: CodeCompletionProfile, tok: Tokenizer, n: int, rng: random.Random
) -> list[PreparedRequest]:
    out = []
    for _ in range(n):
        in_len = _jitter(rng, p.input_len, p.range_ratio)
        out_len = _jitter(rng, p.output_len, p.range_ratio)
        text = _code_text(in_len, tok, rng)
        out.append(_request(p, "completions", text, tok.count(text), out_len, p.ignore_eos))
    return out


_ESSAY = (
    "Write a long, detailed essay that covers each of the following topics in order, "
    "with several paragraphs per topic:\n"
)


def _long_generation(
    p: LongGenerationProfile, tok: Tokenizer, n: int, rng: random.Random
) -> list[PreparedRequest]:
    out = []
    for _ in range(n):
        in_len = _jitter(rng, p.input_len, p.range_ratio)
        out_len = _jitter(rng, p.output_len, p.range_ratio)
        seed = rng.getrandbits(64)

        def build(h: int, seed=seed) -> str:
            return _ESSAY + tok.random_text(h, random.Random(seed))

        text = _fit(build, in_len, tok)
        out.append(_request(p, p.endpoint, text, tok.count(text), out_len, p.ignore_eos))
    return out


def _trace(p: TraceProfile, tok: Tokenizer, n: int, rng: random.Random) -> list[PreparedRequest]:
    if n == 0:
        return []
    rows = read_trace(p.path, p.format, p.max_rows)
    if n > len(rows):
        raise ValueError(f"{p.name}: trace has {len(rows)} rows, need {n}")
    out = []
    for i, row in enumerate(rows[:n]):
        in_len = max(1, min(row.input_tokens, p.max_input_len or row.input_tokens))
        out_len = max(1, min(row.output_tokens, p.max_output_len or row.output_tokens))
        text = tok.random_text(in_len, rng)
        meta = {"trace_row": i, "trace_offset_s": row.timestamp_s}
        out.append(_request(p, p.endpoint, text, in_len, out_len, p.ignore_eos, meta))
    return out


_BUILDERS: dict[type, Callable[..., list[PreparedRequest]]] = {
    SyntheticProfile: _synthetic,
    SharedPrefixProfile: _shared_prefix,
    ChatDatasetProfile: _chat_dataset,
    NeedleProfile: _needle,
    CodeCompletionProfile: _code_completion,
    LongGenerationProfile: _long_generation,
    TraceProfile: _trace,
}
