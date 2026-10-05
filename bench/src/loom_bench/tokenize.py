"""Token counting used to build prompts of exact lengths.

`SimpleTokenizer` is a deterministic regex tokenizer shared by the mock backend
and tests, so client-side and "server"-side counts agree exactly. Real runs use
`HFTokenizer` loaded from the model's pinned revision; the server's `usage`
block stays the source of truth for recorded token counts.
"""

from __future__ import annotations

import random
import re
from typing import Protocol, runtime_checkable

# Every character falls into exactly one alternative, so "".join(pieces) == text.
_PIECE_RE = re.compile(r"\s?[A-Za-z]+|\s?\d|\s?[^\sA-Za-z\d]|\s+")

# Small fixed vocabulary for synthetic text. Plain English words keep the
# simple tokenizer at one token per word (with its leading space).
_WORDS = (
    "the of and to in is was for on that with as by at from his it an were are which "
    "this be or have had not but one their they all there been has more when who will "
    "would no if can out so said what up its about into than them only other new some "
    "time could these two may then do first any my now such like our over man me even "
    "most made after also did many before must through back years where much your way "
    "well down should because each just those people how too little state good very make "
    "world still own see men work long get here between both life being under never day "
    "same another know while last might us great old year off come since against go came "
    "right used take three system model engine token cache batch latency price cost "
    "request queue memory kernel tensor layer prompt output input server cluster region"
).split()


@runtime_checkable
class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, n: int) -> str:
        """Prefix of `text` containing at most `n` tokens."""
        ...

    def random_text(self, n: int, rng: random.Random) -> str:
        """Text that tokenizes to exactly `n` tokens (best effort for BPE tokenizers)."""
        ...


class SimpleTokenizer:
    name = "loom-simple-v1"

    def pieces(self, text: str) -> list[str]:
        return _PIECE_RE.findall(text)

    def count(self, text: str) -> int:
        return len(self.pieces(text))

    def truncate(self, text: str, n: int) -> str:
        return "".join(self.pieces(text)[: max(n, 0)])

    def random_text(self, n: int, rng: random.Random) -> str:
        if n <= 0:
            return ""
        words = [rng.choice(_WORDS) for _ in range(n)]
        return words[0] + "".join(" " + w for w in words[1:])


class HFTokenizer:
    """Wraps a `tokenizers.Tokenizer` loaded from a pinned HF revision."""

    def __init__(self, repo: str, revision: str, token: str | None = None) -> None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer as _Tok

        path = hf_hub_download(repo, "tokenizer.json", revision=revision, token=token)
        self._tok = _Tok.from_file(path)
        self.name = f"{repo}@{revision}"
        vocab = self._tok.get_vocab()
        # Ordinary word-like tokens only; avoids special/control tokens in random prompts.
        self._sample_ids = sorted(
            i for t, i in vocab.items() if t.isascii() and t.strip("Ġ▁ ").isalpha()
        )

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, n: int) -> str:
        ids = self._tok.encode(text, add_special_tokens=False).ids[: max(n, 0)]
        return self._tok.decode(ids)

    def random_text(self, n: int, rng: random.Random) -> str:
        if n <= 0:
            return ""
        ids = [rng.choice(self._sample_ids) for _ in range(n)]
        text = self._tok.decode(ids)
        # Decode/encode is not always a bijection; trim or pad to hit n exactly.
        for _ in range(8):
            got = self.count(text)
            if got == n:
                break
            if got > n:
                text = self.truncate(text, n)
            else:
                extra = [rng.choice(self._sample_ids) for _ in range(n - got)]
                text += self._tok.decode(extra)
        return text
