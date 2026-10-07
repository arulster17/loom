"""Divergence across engines that render byte-split characters differently.

The second RunPod 8B sweep (565b8d3f) lost SGLang's whole eval to divergence prompts 20
(" √") and 40 (Chinese): Qwen3's byte-level BPE splits those characters across tokens,
vLLM renders the first token "" and the next the whole character, and SGLang renders each
token on its own, a fragment as its raw bytes in latin-1 (`loom_bench.detokenize`). The
fixture holds vLLM's real reference capture of those two prompts from that sweep.

The network tests rebuild exactly what SGLang returns for the same text from the real
Qwen3 tokenizer (token ids -> raw bytes -> SGLang's rendering), check that vLLM's rule
reproduces the captured strings, and score the SGLang rendering against the capture. They
download the tokenizer once (cached), so they run with `-m network`.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from loom_bench.detokenize import incremental_text, per_token_text
from loom_bench.quality.divergence import (
    Position,
    ReferenceLogprobs,
    ReferencePrompt,
    compare_positions,
    incremental_logprobs,
    load_prompts,
    scored_positions,
)
from loom_bench.quality.suite import load_suite

FIXTURE = Path(__file__).parent / "fixtures" / "divergence_qwen3_byte_split.json"
QWEN3 = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")
SWEEP_ERROR = "server returned no text_offset and its echoed tokens do not spell the scored text"


def fixture() -> tuple[list[int], ReferenceLogprobs]:
    doc = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ref = ReferenceLogprobs(
        model=doc["model"],
        top_k=doc["top_k"],
        max_new_tokens=doc["max_new_tokens"],
        prompts=[ReferencePrompt.model_validate(p) for p in doc["prompts"]],
    )
    return doc["prompt_ids"], ref


def test_the_fixture_is_the_pinned_prompts_and_splits_characters():
    ids, ref = fixture()
    pinned = load_prompts()
    assert [p.prompt for p in ref.prompts] == [pinned[i] for i in ids]
    # the Qwen3 suite's hard prompts, which a limited suite (the smoke) always scores
    suite = load_suite("qwen3-8b").divergence
    assert suite is not None and suite.hard_prompts == ids
    for p in ref.prompts:
        tokens = [x.token for x in p.positions]
        assert "".join(tokens) == p.continuation
        assert "" in tokens  # vLLM's rendering of a token that splits a character
        assert any(ord(c) > 127 for c in p.continuation)


# --- offline: hand-built splits --------------------------------------------------------

ROOT = " that √2 is"  # " √" is b" \xe2\x88" + b"\x9a", as Qwen3 splits it
SPLIT = [b"x", b" that", b" \xe2\x88", b"\x9a", b"2", b" is", b"."]  # "x" prompt, "." generated


def _response(render: str) -> dict[str, Any]:
    tokens, tops = [], []
    for i, raw in enumerate(SPLIT):
        if render == "vllm":
            text = incremental_text(raw, SPLIT[:i])
            alt = incremental_text(b"\x9b", SPLIT[:i])  # " ∛" after the first half of " √"
        else:
            text, alt = per_token_text(raw), per_token_text(b"\x9b")
        tokens.append(text)
        tops.append(None if i == 0 else {text: -0.1, alt: -3.0, " the": -4.0})
    return {
        "tokens": tokens,
        "token_logprobs": [None] + [-0.1] * (len(SPLIT) - 1),
        "top_logprobs": tops,
        "text_offset": [-1] * len(SPLIT) if render == "sglang" else _offsets(tokens),
    }


def _offsets(tokens: list[str]) -> list[int]:
    out, pos = [], 0
    for t in tokens:
        out.append(pos)
        pos += len(t)
    return out


def test_the_two_renderings_differ_where_a_character_is_split():
    vllm, sglang = _response("vllm"), _response("sglang")
    assert vllm["tokens"][2:4] == ["", " √"]
    assert sglang["tokens"][2:4] == [" \xe2\x88", "\x9a"]
    # what the previous rebuild checked, and why it raised in 565b8d3f
    assert not "".join(sglang["tokens"]).startswith("x" + ROOT)


def test_a_per_token_response_scores_like_vllm_on_split_characters():
    text = "x" + ROOT
    want = scored_positions(_response("vllm"), 1, len(text), text=text)
    got = scored_positions(_response("sglang"), 1, len(text), text=text)
    assert [p.token for p in got] == [" that", "", " √", "2", " is"]
    assert got == want
    d = compare_positions(want, got)
    assert d.top1_rate == 1.0 and max(d.kl) == pytest.approx(0.0, abs=1e-12)


def test_a_per_token_response_is_still_rejected_when_it_does_not_spell_the_text():
    with pytest.raises(ValueError, match="do not spell"):
        scored_positions(_response("sglang"), 1, 12, text="x that √3 is")


def test_a_different_tokenization_of_the_same_text_still_fails():
    # Same text, but the candidate splits " √" as b" " + b"\xe2\x88\x9a": not the same
    # tokenizer, so the positions must not be compared.
    text = "x" + ROOT
    other = [b"x", b" that", b" ", b"\xe2\x88\x9a", b"2", b" is", b"."]
    tokens = [per_token_text(b) for b in other]
    resp = {
        "tokens": tokens,
        "token_logprobs": [None] + [-0.1] * 6,
        "top_logprobs": [None] + [{t: -0.1} for t in tokens[1:]],
        "text_offset": [-1] * 7,
    }
    want = scored_positions(_response("vllm"), 1, len(text), text=text)
    got = scored_positions(resp, 1, len(text), text=text)
    with pytest.raises(ValueError, match="tokenized the same text differently"):
        compare_positions(want, got)


def test_lossy_fragments_are_matched_against_the_text_bytes():
    # A per-token server that renders a fragment as U+FFFD (no lossless bytes).
    text = "x" + ROOT
    tokens = [b.decode("utf-8", "replace") for b in SPLIT]
    resp = {
        "tokens": tokens,
        "token_logprobs": [None] + [-0.1] * 6,
        "top_logprobs": [None] + [{t: -0.1} for t in tokens[1:]],
        "text_offset": [-1] * 7,
    }
    got = scored_positions(resp, 1, len(text), text=text)
    assert [p.token for p in got] == [" that", "", " √", "2", " is"]


def test_ambiguous_fragment_keys():
    # "é" is a token's text and also how SGLang renders the lone byte 0xE9. After Latin
    # text it reads as the letter; a key with a C1 control or an unfinished two-byte
    # character reads as a fragment and joins vLLM's "" bucket (the last value stays).
    tokens, tops, _ = incremental_logprobs(
        ["x", " a", "."],
        [None, {" a": -0.1, "é": -2.0, "ä\xb8": -3.0, "\x9a": -4.0}, {}],
        "x a",
    )
    assert tokens == ["x", " a", "."]
    assert tops[1] == {" a": -0.1, "é": -2.0, "": -4.0}


def test_a_lone_lead_byte_after_cjk_text_is_a_fragment():
    # 0xE9 starts 需 (e9 9c 80); after Chinese text the key "é" is that fragment.
    tokens, tops, _ = incremental_logprobs(
        ["这", "是", "."], [None, {"是": -0.1, "é": -2.0}, {}], "这是"
    )
    assert tokens == ["这", "是", "."]
    assert tops[1] == {"是": -0.1, "": -2.0}


def test_a_key_finishing_the_unfinished_character_reads_as_its_bytes():
    # After b" \xe2\x88" (the first half of " √"), the key "\x9b" is the byte finishing " ∛".
    _, tops, _ = incremental_logprobs(
        ["x", " \xe2\x88", "\x9a", "."],
        [None, {" \xe2\x88": -0.1}, {"\x9a": -0.1, "\x9b": -3.0, " the": -5.0}, {}],
        "x √",
    )
    assert tops[2] == {" √": -0.1, " ∛": -3.0, " the": -5.0}


def test_a_leading_special_token_outside_the_text_is_ignored():
    # vLLM counts BOS's text into its offsets although the echoed text does not hold it.
    resp = {
        "tokens": ["<|begin_of_text|>", "Hi", " there", " you", " go"],
        "token_logprobs": [None, -1.0, -0.5, -0.2, -0.1],
        "top_logprobs": [None, {"Hi": -1.0}, {" there": -0.5}, {" you": -0.2}, {" go": -0.1}],
        "text_offset": [0, 17, 19, 25, 29],
    }
    pos = scored_positions(resp, 2, 12, text="Hi there you")
    assert [p.token for p in pos] == [" there", " you"]
    per_token = {**resp, "text_offset": [-1] * 5}
    assert scored_positions(per_token, 2, 12, text="Hi there you") == pos


# --- network: the real tokenizer on the sweep's prompts ------------------------------------


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
def _qwen3() -> Any:
    tokenizers = pytest.importorskip("tokenizers")
    hub = pytest.importorskip("huggingface_hub")
    repo, revision = QWEN3
    path = hub.hf_hub_download(repo, "tokenizer.json", revision=revision)
    return tokenizers.Tokenizer.from_file(path)


def _token_bytes(text: str) -> list[bytes]:
    tok = _qwen3()
    ids = tok.encode(text, add_special_tokens=False).ids
    dec = _byte_decoder()
    return [bytes(dec[c] for c in tok.id_to_token(i)) for i in ids]


def _sglang_response(p: ReferencePrompt) -> dict[str, Any]:
    """What SGLang v0.5.21 returns for an echo of prompt + continuation, its logprobs
    taken from vLLM's capture so that a correct comparison finds no divergence."""
    text = p.prompt + p.continuation
    raws = [*_token_bytes(text), b" the"]  # plus one generated token
    first = len(raws) - 1 - len(p.positions)  # the continuation's first token
    tokens, tops = [], []
    for i, raw in enumerate(raws):
        tokens.append(per_token_text(raw))
        if i == 0:
            tops.append(None)
        elif first <= i < len(raws) - 1:
            pos = p.positions[i - first]
            tops.append(_sglang_keys(pos.top, pos.token, raw, raws[:i]))
        else:
            tops.append({per_token_text(raw): -0.1})
    return {
        "tokens": tokens,
        "token_logprobs": [None] + [-0.1] * (len(raws) - 1),
        "top_logprobs": tops,
        "text_offset": [-1] * len(raws),
    }


def _sglang_keys(
    top: dict[str, float], token: str, raw: bytes, context: list[bytes]
) -> dict[str, float]:
    # vLLM's keys back to the candidate tokens' bytes, then rendered as SGLang does.
    pending = b""
    for c in reversed(context[-4:]):
        if c.decode("utf-8", "replace").endswith("�"):
            pending = c + pending
        else:
            break
    out = {}
    for key, lp in top.items():
        if key == token:
            cand = raw
        elif key == "":
            cand = b"\xe4\xb8"  # some fragment vLLM collapsed into its "" bucket
        else:
            data = key.encode()
            cand = data[len(pending) :] if pending and data.startswith(pending) else data
        out[per_token_text(cand)] = lp
    return out


@pytest.mark.network
def test_vllm_rule_reproduces_the_captured_strings():
    _, ref = fixture()
    for p in ref.prompts:
        raws = _token_bytes(p.prompt + p.continuation)
        first = len(raws) - len(p.positions)
        rendered = [incremental_text(raws[i], raws[:i]) for i in range(first, len(raws))]
        assert rendered == [x.token for x in p.positions]


@pytest.mark.network
def test_sglang_against_the_real_vllm_reference():
    _, ref = fixture()
    for p in ref.prompts:
        text = p.prompt + p.continuation
        resp = _sglang_response(p)
        # the sweep's failure: the old cumulative-length rebuild had nothing to stand on
        assert not "".join(resp["tokens"]).startswith(text)
        cand = scored_positions(resp, len(p.prompt), len(text), text=text)
        want = [Position(x.token, dict(x.top)) for x in p.positions]
        assert [c.token for c in cand] == [w.token for w in want]
        d = compare_positions(want, cand)
        assert d.top1_rate == 1.0
        assert max(d.kl) == pytest.approx(0.0, abs=1e-9)
