"""A byte-level BPE stand-in for the mock backend (`MockConfig.byte_level`).

Real byte-level tokenizers split some characters across tokens (Qwen3: " √" is
b" \\xe2\\x88" + b"\\x9a"), and engines render such tokens differently in completions
logprobs (`loom_bench.detokenize`). With `byte_level` on, the mock's free text mixes in
words with multi-byte characters, every piece holding one is split into two byte tokens
(all its bytes but the last, then the last), a few raw-byte fragments join the candidate
alternatives, and completions logprobs render the tokens the way the engine it imitates
does: vLLM's `incremental_text` with `text_offsets` on, SGLang's `per_token_text` (and
offsets of -1) with it off.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from loom_bench.detokenize import incremental_text, per_token_text
from loom_bench.mock.logprobs import TokenLogprob

MULTIBYTE_WORDS = (" √2", " 这句话", " straße", " café", " 予定", " ∞")
# Raw-byte fragments as candidate alternatives, written as latin-1 (one char per byte): a
# CJK lead pair and a lone continuation byte. (A lone lead byte such as 0xE9 renders like
# the letter "é" in SGLang's form and is only told apart by context; the mock's
# alternatives ignore context, so it is left to the divergence unit tests.)
FRAGMENT_ALTERNATIVES = ("\xe4\xb8", "\x9b")
# One word in this many becomes a multi-byte one.
EVERY = 5


def with_multibyte(text: str, rng: random.Random) -> str:
    """`text` with about one word in EVERY swapped for a word with multi-byte characters."""
    words = text.split(" ")
    for i in range(1, len(words)):
        if rng.randrange(EVERY) == 0:
            words[i] = rng.choice(MULTIBYTE_WORDS).lstrip(" ")
    return " ".join(words)


def split_pieces(pieces: Sequence[str]) -> list[bytes]:
    """Each piece's bytes as one token, or two when it holds a multi-byte character."""
    out: list[bytes] = []
    for piece in pieces:
        data = piece.encode("utf-8")
        if len(data) == len(piece):
            out.append(data)
        else:
            out += [data[:-1], data[-1:]]
    return out


def identity(token: bytes) -> str:
    """A token's identity for the mock's logprob hashing: its bytes as latin-1."""
    return token.decode("latin-1")


def completion_logprobs(
    units: Sequence[bytes], entries: Sequence[TokenLogprob | None], *, incremental: bool
) -> dict[str, Any]:
    """Completions `logprobs` for byte tokens `units`, rendered vLLM-style (`incremental`)
    or SGLang-style. `entries[i]` holds identities (`identity`) of the candidates."""

    def render(token: bytes, i: int) -> str:
        return incremental_text(token, units[:i]) if incremental else per_token_text(token)

    tokens = [render(u, i) for i, u in enumerate(units)]
    tops: list[dict[str, float] | None] = []
    for i, entry in enumerate(entries):
        if entry is None:
            tops.append(None)
            continue
        # vLLM puts the token itself first, then the top-k by rank; SGLang lists the top-k
        # by rank. Keys that render alike collide, and the last one written stays.
        top = {tokens[i]: entry.logprob} if incremental else {}
        for t, lp in entry.top:
            top[render(t.encode("latin-1"), i)] = lp
        tops.append(top)
    offsets, pos = [], 0
    for t in tokens:
        offsets.append(pos if incremental else -1)
        pos += len(t)
    return {
        "tokens": tokens,
        "token_logprobs": [e.logprob if e else None for e in entries],
        "top_logprobs": tops,
        "text_offset": offsets,
    }
