"""What the mock "model" says. Every output is a pure function of (seed, prompt, params).

All token counting goes through `SimpleTokenizer`, so clients that count with it
agree exactly with the server's `usage`.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Any

from loom_bench.tokenize import SimpleTokenizer

TOKENIZER = SimpleTokenizer()

_ARITHMETIC_RE = re.compile(r"What is (-?\d+) ([-+*]) (-?\d+)\?")


def seeded_rng(seed: int, purpose: str, text: str) -> random.Random:
    digest = hashlib.blake2b(f"{seed}\x00{purpose}\x00{text}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def is_degraded(seed: int, prompt: str, degrade: float) -> bool:
    """Fixed per-prompt draw, so the set of broken prompts only grows with `degrade`."""
    return seeded_rng(seed, "degrade", prompt).random() < degrade


def render_chat_prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """Flatten a chat request into the prompt the mock "sees" and counts.

    Template: an optional `<|tools|>\\n{tools JSON}\\n` header, then per message
    `<|{role}|>\\n{content}\\n` (assistant tool calls appended as JSON), then the
    generation prompt `<|assistant|>\\n`. Content given as a list of parts uses
    the concatenated `text` parts.
    """
    parts = []
    if tools:
        parts.append(f"<|tools|>\n{json.dumps(tools, sort_keys=True)}\n")
    for message in messages:
        body = message_text(message)
        if message.get("tool_calls"):
            body += json.dumps(message["tool_calls"], sort_keys=True)
        parts.append(f"<|{message['role']}|>\n{body}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def arithmetic_answer(question: str, *, seed: int, prompt: str, degrade: float) -> str | None:
    """'The answer is N.' for the last 'What is A <op> B?' in `question`, else None."""
    matches = _ARITHMETIC_RE.findall(question)
    if not matches:
        return None
    a_text, op, b_text = matches[-1]
    a, b = int(a_text), int(b_text)
    value = a + b if op == "+" else a - b if op == "-" else a * b
    if is_degraded(seed, prompt, degrade):
        rng = seeded_rng(seed, "wrong", prompt)
        value += rng.choice((-1, 1)) * rng.randint(1, 9)
    return f"The answer is {value}."


def free_text(*, seed: int, prompt: str, mean_tokens: int) -> str:
    rng = seeded_rng(seed, "text", prompt)
    n = max(1, round(rng.expovariate(1 / mean_tokens)))
    return TOKENIZER.random_text(n, rng)


def json_text(
    schema: dict[str, Any] | None, *, seed: int, prompt: str, degrade: float, purpose: str
) -> str:
    """Compact JSON valid for `schema` (any object when None); truncated when degraded."""
    rng = seeded_rng(seed, purpose, prompt)
    value = sample_json(schema, rng) if schema else {"answer": TOKENIZER.random_text(3, rng)}
    text = json.dumps(value)
    if is_degraded(seed, prompt, degrade):
        return text[:-1]
    return text


def sample_json(schema: dict[str, Any], rng: random.Random) -> Any:
    """A value valid for a simple JSON schema (type/properties/required/items/enum/const)."""
    if "const" in schema:
        return schema["const"]
    if "enum" in schema:
        return rng.choice(schema["enum"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            return sample_json(schema[key][0], rng)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "null")
    if kind is None:
        kind = "object" if "properties" in schema else "string"
    match kind:
        case "object":
            props = dict(schema.get("properties", {}))
            for name in schema.get("required", []):
                props.setdefault(name, {"type": "string"})
            return {name: sample_json(sub, rng) for name, sub in props.items()}
        case "array":
            lo = schema.get("minItems", 1)
            hi = max(lo, schema.get("maxItems", 3))
            items = schema.get("items", {"type": "string"})
            return [sample_json(items, rng) for _ in range(rng.randint(lo, min(hi, lo + 2)))]
        case "integer":
            lo = int(schema.get("minimum", 0))
            return rng.randint(lo, int(schema.get("maximum", lo + 100)))
        case "number":
            lo = float(schema.get("minimum", 0))
            return round(rng.uniform(lo, float(schema.get("maximum", lo + 100))), 2)
        case "boolean":
            return rng.random() < 0.5
        case "null":
            return None
        case _:
            return _sample_string(schema, rng)


def _sample_string(schema: dict[str, Any], rng: random.Random) -> str:
    text = TOKENIZER.random_text(rng.randint(1, 3), rng)
    if "maxLength" in schema:
        text = text[: schema["maxLength"]]
    return text.ljust(schema.get("minLength", 0), "x")


@dataclass(frozen=True, slots=True)
class Output:
    pieces: tuple[str, ...]
    hit_length: bool

    @property
    def text(self) -> str:
        return "".join(self.pieces)


def fit_output(
    text: str,
    *,
    seed: int,
    prompt: str,
    max_tokens: int,
    min_tokens: int = 0,
    ignore_eos: bool = False,
) -> Output:
    """Apply EOS rules: pad with filler to min_tokens (or max_tokens when ignoring
    EOS, as an engine keeps sampling past EOS), then cut at max_tokens."""
    pieces = TOKENIZER.pieces(text)
    target = max_tokens if ignore_eos else max(len(pieces), min_tokens)
    if target > len(pieces):
        # " w1 w2 ..." adds exactly one token per word: each word starts a new piece,
        # and a leading space either starts " w1" or extends a trailing whitespace run.
        filler = TOKENIZER.random_text(target - len(pieces), seeded_rng(seed, "filler", prompt))
        pieces = TOKENIZER.pieces(text + " " + filler)
    return Output(tuple(pieces[:max_tokens]), hit_length=len(pieces) >= max_tokens)
