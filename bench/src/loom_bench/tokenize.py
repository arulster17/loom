"""Token counting used to build prompts of exact lengths.

`SimpleTokenizer` is a deterministic regex tokenizer shared by the mock backend
and tests, so client-side and "server"-side counts agree exactly. Real runs use
`HFTokenizer` loaded from the model's pinned revision; the server's `usage`
block stays the source of truth for recorded token counts.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
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


class SnapshotMismatch(ValueError):
    """A local tokenizer directory is not the expected repo's snapshot at the pinned revision."""


def hf_cache_folder(repo: str) -> str:
    """The Hugging Face cache folder of a model repo: `org/name` -> `models--org--name`."""
    return "models--" + repo.replace("/", "--")


def verify_snapshot(local_dir: str | Path, repo: str, revision: str) -> Path:
    """`local_dir` if it is the HF cache snapshot of `repo` at `revision`.

    The cache names each snapshot directory after the commit it was downloaded at
    (`<cache>/models--org--name/snapshots/<commit>`), so the path itself says which
    repo and revision the files belong to.
    """
    path = Path(local_dir)
    expected = f"{hf_cache_folder(repo)}/snapshots/{revision}"
    if path.parts[-3:] != tuple(expected.split("/")):
        raise SnapshotMismatch(f"{path} is not the snapshot of {repo}@{revision} (…/{expected})")
    if not path.is_dir():
        raise SnapshotMismatch(f"snapshot of {repo}@{revision} not found at {path}")
    return path


class HFTokenizer:
    """Wraps a `tokenizers.Tokenizer` loaded from a pinned HF revision: from the Hub,
    or from `local_dir`, a cache snapshot of that revision (checked, never the Hub)."""

    def __init__(
        self, repo: str, revision: str, token: str | None = None, local_dir: str | None = None
    ) -> None:
        from tokenizers import Tokenizer as _Tok

        if local_dir is not None:
            path = str(verify_snapshot(local_dir, repo, revision) / "tokenizer.json")
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo, "tokenizer.json", revision=revision, token=token)
        self._tok = _Tok.from_file(path)
        # Counting must see every token. A tokenizer.json can carry a truncation (or
        # padding) setting left over from whoever saved it: RedHatAI's Llama 3.3 FP8
        # checkpoint keeps `truncation: {max_length: 2048}` from its calibration, which
        # would cap every count at 2048 and grow long prompts without bound. transformers
        # (the engines' tokenizer) drops both unless asked, so drop them here too.
        self._tok.no_truncation()
        self._tok.no_padding()
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
