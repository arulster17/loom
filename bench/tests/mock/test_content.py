import json
import math
import random

import jsonschema
import pytest

from loom_bench.mock.content import (
    arithmetic_answer,
    fit_output,
    json_text,
    render_chat_prompt,
    sample_json,
)
from loom_bench.mock.logprobs import MAX_LOGPROBS, token_logprobs
from loom_bench.tokenize import SimpleTokenizer

TOK = SimpleTokenizer()


@pytest.mark.parametrize(
    "text", ["The answer is 85.", '{"a": [1, 2.5, "x y"]}', "ends with space ", "line\n"]
)
def test_fit_output_counts_are_exact(text):
    natural = TOK.count(text)
    for max_tokens in (1, 3, natural, natural + 5, 100):
        for min_tokens in (0, natural + 2):
            for ignore_eos in (False, True):
                if min_tokens > max_tokens:
                    continue
                out = fit_output(
                    text,
                    seed=0,
                    prompt="p",
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    ignore_eos=ignore_eos,
                )
                assert TOK.pieces(out.text) == list(out.pieces)
                expected = max_tokens if ignore_eos else min(max(natural, min_tokens), max_tokens)
                assert len(out.pieces) == expected
                assert out.hit_length == (ignore_eos or max(natural, min_tokens) >= max_tokens)
                assert out.text.startswith(TOK.truncate(text, max_tokens))


def test_render_chat_prompt_template():
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image"}]},
    ]
    assert render_chat_prompt(messages, None) == (
        "<|system|>\nBe brief.\n<|user|>\nHi\n<|assistant|>\n"
    )
    tools = [{"type": "function", "function": {"name": "f"}}]
    with_tools = render_chat_prompt(messages, tools)
    assert with_tools.startswith("<|tools|>\n")
    assert TOK.count(with_tools) > TOK.count(render_chat_prompt(messages, None))


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is 37 + 48?", 85),
        ("What is 5 - 12?", -7),
        ("What is -3 * 7?", -21),
        ("Q: What is 1 + 1?\nA: 2\nQ: What is 6 * 7?\nA:", 42),
    ],
)
def test_arithmetic_answer(question, expected):
    answer = arithmetic_answer(question, seed=0, prompt=question, degrade=0)
    assert answer == f"The answer is {expected}."


def test_arithmetic_ignores_other_prompts():
    assert arithmetic_answer("Tell me a story.", seed=0, prompt="x", degrade=1) is None


def test_degrade_is_monotonic_per_prompt():
    questions = [f"What is {a} + {a + 3}?" for a in range(200)]

    def wrong(degrade: float) -> set[str]:
        return {
            q
            for q in questions
            if arithmetic_answer(q, seed=0, prompt=q, degrade=degrade)
            != arithmetic_answer(q, seed=0, prompt=q, degrade=0)
        }

    low, high = wrong(0.2), wrong(0.6)
    assert low < high
    assert 0.1 < len(low) / len(questions) < 0.3


SCHEMAS = [
    {"type": "string", "minLength": 5, "maxLength": 8},
    {"type": "integer", "minimum": 10, "maximum": 12},
    {"type": ["null", "number"], "minimum": -1, "maximum": 1},
    {"type": "array", "items": {"enum": [1, "a", None]}, "minItems": 2, "maxItems": 4},
    {"anyOf": [{"type": "boolean"}, {"type": "string"}]},
    {"const": "fixed"},
    {"type": "object", "required": ["x"]},
    {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}, "ok": {"type": "boolean"}},
                    "required": ["id", "ok"],
                },
            }
        },
        "required": ["items"],
    },
]


@pytest.mark.parametrize("schema", SCHEMAS)
def test_sample_json_is_valid(schema):
    for seed in range(20):
        jsonschema.validate(sample_json(schema, random.Random(seed)), schema)


def test_json_text_degrade_breaks_json():
    schema = SCHEMAS[-1]
    ok = json_text(schema, seed=0, prompt="p", degrade=0, purpose="json")
    jsonschema.validate(json.loads(ok), schema)
    broken = json_text(schema, seed=0, prompt="p", degrade=1, purpose="json")
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(math.exp(lp) * (lp - q.get(t, -30.0)) for t, lp in p.items())


def test_logprobs_deterministic_and_normalised():
    tokens = TOK.pieces("The quick brown fox jumps over the lazy dog")
    a = token_logprobs(tokens, seed=1, start=1, k=MAX_LOGPROBS, noise=0)
    b = token_logprobs(tokens, seed=1, start=1, k=MAX_LOGPROBS, noise=0)
    assert a == b
    assert len(a) == len(tokens) - 1
    for entry, token in zip(a, tokens[1:], strict=True):
        assert entry.token == token
        assert entry.top[0] == (token, entry.logprob)  # noise-free: sampled token is the mode
        assert sum(math.exp(lp) for _, lp in entry.top) < 1
    # Same context, same distribution, regardless of what follows.
    prefix = token_logprobs(tokens[:4], seed=1, start=1, k=5, noise=0)
    assert [e.top for e in prefix] == [e.top[:5] for e in a[:3]]


def test_logprob_kl_grows_with_noise():
    tokens = TOK.pieces(TOK.random_text(200, random.Random(0)))

    def mean_kl(noise: float) -> float:
        ref = token_logprobs(tokens, seed=0, start=0, k=MAX_LOGPROBS, noise=0)
        got = token_logprobs(tokens, seed=0, start=0, k=MAX_LOGPROBS, noise=noise)
        return sum(_kl(dict(r.top), dict(g.top)) for r, g in zip(ref, got, strict=True)) / len(ref)

    kls = [mean_kl(n) for n in (0.0, 0.1, 0.5, 2.0)]
    assert kls[0] == 0
    assert kls == sorted(kls) and kls[1] > 0
