"""Divergence on Llama 3.3's tokenizer: its hard prompts, byte splits and BOS.

Llama 3's byte-level BPE (128k vocabulary) keeps the characters that split under Qwen3 (" √",
CJK) whole, but splits others: " ∛", " ∑", " ∫", " ÷" and a word-initial " ß" or " ñ". No
real Llama 3.3 continuation existed when this was written (the 70B run had not happened);
under Llama's tokenizer the real Qwen3-8B BF16 continuations of b1b904dc split nothing.
The suite therefore keeps prompts 20 (a √2 proof, whose continuation reaches for math
symbols) and 40 (German that turned into Chinese) as its hard prompts, and these tests feed
them continuations that do split.

vLLM v0.30.0 echoes a Llama prompt with its BOS: the first prompt token has no logprobs, so
`_create_completion_logprobs` (vllm/entrypoints/openai/completion/serving.py) decodes it on
its own, "<|begin_of_text|>", and its 17 characters count into every later text_offset while
the echoed text holds only the prompt. The tests build exactly that response from the real
tokenizer and check the reference capture reads the continuation back.

They need the gated tokenizer: run with `-m network` and HF_TOKEN set.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import pytest

from loom_bench.detokenize import incremental_text, per_token_text
from loom_bench.quality.divergence import compare_positions, load_prompts, scored_positions
from loom_bench.quality.suite import load_suite
from loom_bench.registry import load_registry

BOS = "<|begin_of_text|>"
# Continuations for the hard prompts that contain characters Llama splits across tokens.
CONTINUATIONS = {
    20: " that √2 = p/q in lowest terms. Then 2 = p²/q², so p² = 2q² is even; note ∛8 = 2, "
    "∑ 1/n² converges, ∫ x dx = x²/2 and 6 ÷ 3 = 2.",
    40: " über die Hauptstraße. ß ist ein Buchstabe, ñ nicht; auf Chinesisch: 中文很难。",
}


def test_the_llama_suite_scores_its_hard_prompts_first_at_smoke_scale():
    div = load_suite("llama-3.3-70b-instruct").divergence
    assert div is not None and div.hard_prompts == sorted(CONTINUATIONS)
    assert div.limited(4).prompt_ids[:2] == [20, 40]


@cache
def _byte_decoder() -> dict[str, int]:
    # GPT-2's bytes_to_unicode, inverted: byte-level BPE vocab pieces -> raw bytes.
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


@cache
def _llama() -> Any:
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("Llama 3.3's tokenizer is gated: set HF_TOKEN")
    tokenizers = pytest.importorskip("tokenizers")
    hub = pytest.importorskip("huggingface_hub")
    hf = load_registry().get("llama-3.3-70b-instruct").hf
    path = hub.hf_hub_download(hf.repo, "tokenizer.json", revision=hf.revision)
    return tokenizers.Tokenizer.from_file(path)


def _ids(text: str) -> list[int]:
    # vLLM tokenizes a completions prompt with the tokenizer's defaults: special tokens on.
    return list(_llama().encode(text, add_special_tokens=True).ids)


def _raw(i: int) -> bytes:
    dec = _byte_decoder()
    return bytes(dec[c] for c in _llama().id_to_token(i))


def _echo(text: str, render: str) -> dict[str, Any]:
    """A vLLM v0.30.0 (or per-token, SGLang-style) echo of `text` plus one generated token."""
    tok = _llama()
    ids = [*_ids(text), _ids(" the")[-1]]
    assert tok.id_to_token(ids[0]) == BOS and _ids("")[0] == ids[0]
    raws = [b""] + [_raw(i) for i in ids[1:]]
    tokens, tops = [BOS], [None]
    for i in range(1, len(ids)):
        vllm = render == "vllm"
        t = incremental_text(raws[i], raws[1:i]) if vllm else per_token_text(raws[i])
        tokens.append(t)
        tops.append({t: -0.1, "▁x": -5.0})
    offsets, pos = [], 0
    for t in tokens:
        offsets.append(pos)
        pos += len(t)
    return {
        "tokens": tokens,
        "token_logprobs": [None] + [-0.1] * (len(ids) - 1),
        "top_logprobs": tops,
        "text_offset": offsets if render == "vllm" else [-1] * len(ids),
    }


@pytest.mark.network
@pytest.mark.parametrize("prompt_id", sorted(CONTINUATIONS))
def test_llama_splits_characters_in_the_hard_continuations(prompt_id):
    prompt = load_prompts()[prompt_id]
    text = prompt + CONTINUATIONS[prompt_id]
    tokens = _echo(text, "vllm")["tokens"]
    assert "" in tokens  # vLLM's rendering of a token that splits a character
    # the characters Qwen3 split stay whole here
    assert len(_ids(" √")) == 2 and len(_ids("中文")) <= 3  # BOS + the piece(s)


@pytest.mark.network
@pytest.mark.parametrize("prompt_id", sorted(CONTINUATIONS))
def test_vllm_echo_with_bos_scores_the_continuation(prompt_id):
    prompt = load_prompts()[prompt_id]
    text = prompt + CONTINUATIONS[prompt_id]
    resp = _echo(text, "vllm")
    # BOS's 17 characters shift every offset past the echoed text's own positions.
    assert resp["text_offset"][1] == len(BOS)
    pos = scored_positions(resp, len(prompt), len(text), text=text)
    assert "".join(p.token for p in pos) == CONTINUATIONS[prompt_id]
    assert "" in [p.token for p in pos]
    d = compare_positions(pos, pos)
    assert d.top1_rate == 1.0


@pytest.mark.network
@pytest.mark.parametrize("prompt_id", sorted(CONTINUATIONS))
def test_a_per_token_engine_scores_like_vllm_on_llama(prompt_id):
    prompt = load_prompts()[prompt_id]
    text = prompt + CONTINUATIONS[prompt_id]
    want = scored_positions(_echo(text, "vllm"), len(prompt), len(text), text=text)
    got = scored_positions(_echo(text, "per_token"), len(prompt), len(text), text=text)
    assert [p.token for p in got] == [p.token for p in want]
    d = compare_positions(want, got)
    assert d.top1_rate == 1.0
    assert max(d.kl) == pytest.approx(0.0, abs=1e-9)
