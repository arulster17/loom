"""Per-output sanity checks and their aggregate rates.

Four detectors, all dependency-free:

- empty: nothing but whitespace.
- truncated: the server stopped at the token limit (`finish_reason == "length"`)
  where the task did not expect it to.
- repetition: either the output ends in a loop (its tail is periodic with a
  period of at most `max_loop_period` characters, repeated at least
  `min_loop_repeats` times and covering at least `min_loop_chars`), or, for
  prose, more than `max_dup_ngram_ratio` of its word n-grams are repeats.
- language drift (prose only): more than `max_foreign_ratio` of its letters
  belong to scripts outside `target_scripts` (Unicode character names, so
  "CJK", "CYRILLIC", "ARABIC", ...).

Code and JSON outputs skip the n-gram and script checks: repeated tokens and
identifiers are normal there.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from loom_bench.quality.tasks.base import Completion, OutputKind

Rate = Annotated[float, Field(ge=0.0, le=1.0)]

_LOOP_WINDOW_CHARS = 4000
CHECKS = ("empty", "truncated", "repetition", "language_drift")


class SanityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ngram: Annotated[int, Field(ge=1)] = 4
    min_ngrams: Annotated[int, Field(ge=1)] = 16
    max_dup_ngram_ratio: Rate = 0.5
    max_loop_period: Annotated[int, Field(ge=1)] = 200
    min_loop_repeats: Annotated[int, Field(ge=2)] = 4
    min_loop_chars: Annotated[int, Field(ge=1)] = 64
    target_scripts: tuple[str, ...] = ("LATIN",)
    min_letters: Annotated[int, Field(ge=1)] = 20
    max_foreign_ratio: Rate = 0.2


class SanityLimits(BaseModel):
    """Highest acceptable share of candidate outputs flagged by each check."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_empty: Rate = 0.01
    max_truncated: Rate = 0.02
    max_repetition: Rate = 0.02
    max_language_drift: Rate = 0.02


class SanityFlags(BaseModel):
    model_config = ConfigDict(frozen=True)

    empty: bool = False
    truncated: bool = False
    repetition: bool = False
    language_drift: bool = False


class SanityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    counts: dict[str, int]

    @property
    def rates(self) -> dict[str, float]:
        return {k: (v / self.n if self.n else 0.0) for k, v in self.counts.items()}

    def violations(self, limits: SanityLimits) -> list[str]:
        """Human-readable reasons, one per check whose rate exceeds its limit."""
        out = []
        for check in CHECKS:
            rate, limit = self.rates[check], getattr(limits, f"max_{check}")
            if rate > limit:
                out.append(
                    f"{check} rate {rate:.2%} ({self.counts[check]}/{self.n}) exceeds {limit:.2%}"
                )
        return out


def dup_ngram_ratio(text: str, n: int) -> tuple[float, int]:
    """Share of word n-grams that repeat an earlier one, and the n-gram count."""
    words = text.split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0, 0
    return 1.0 - len(set(grams)) / len(grams), len(grams)


def periodic_tail(text: str, max_period: int) -> tuple[int, int] | None:
    """(period, covered chars) of the longest periodic suffix, or None.

    A suffix is periodic with period p when s[i] == s[i + p] throughout it; the
    covered length counts the first copy too. Only the last few thousand
    characters are examined.
    """
    s = text[-_LOOP_WINDOW_CHARS:]
    best: tuple[int, int] | None = None
    for p in range(1, min(max_period, len(s) // 2) + 1):
        i = len(s) - 1 - p
        while i >= 0 and s[i] == s[i + p]:
            i -= 1
        covered = len(s) - 1 - i
        if covered >= 2 * p and (best is None or covered > best[1]):
            best = (p, covered)
    return best


def is_loop(text: str, cfg: SanityConfig) -> bool:
    tail = periodic_tail(text, cfg.max_loop_period)
    if tail is None:
        return False
    period, covered = tail
    return covered >= cfg.min_loop_chars and covered >= cfg.min_loop_repeats * period


def _script(ch: str) -> str:
    words = unicodedata.name(ch, "UNKNOWN").split()
    return "LATIN" if "LATIN" in words else words[0]


def foreign_script_ratio(text: str, targets: Iterable[str]) -> tuple[float, int]:
    """Share of letters outside the target scripts, and the letter count."""
    allowed = set(targets)
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0, 0
    foreign = sum(1 for ch in letters if _script(ch) not in allowed)
    return foreign / len(letters), len(letters)


def check_output(
    text: str,
    finish_reason: str | None = None,
    *,
    kind: OutputKind = OutputKind.TEXT,
    length_expected: bool = False,
    config: SanityConfig | None = None,
) -> SanityFlags:
    cfg = config or SanityConfig()
    prose = kind is OutputKind.TEXT
    repetition = is_loop(text, cfg)
    if prose and not repetition:
        ratio, count = dup_ngram_ratio(text, cfg.ngram)
        repetition = count >= cfg.min_ngrams and ratio > cfg.max_dup_ngram_ratio
    drift = False
    if prose:
        ratio, letters = foreign_script_ratio(text, cfg.target_scripts)
        drift = letters >= cfg.min_letters and ratio > cfg.max_foreign_ratio
    return SanityFlags(
        empty=not text.strip(),
        truncated=finish_reason == "length" and not length_expected,
        repetition=repetition,
        language_drift=drift,
    )


def check_completions(
    completions: Iterable[Completion], config: SanityConfig | None = None
) -> SanityResult:
    counts = dict.fromkeys(CHECKS, 0)
    n = 0
    for c in completions:
        flags = check_output(
            c.text,
            c.finish_reason,
            kind=c.kind,
            length_expected=c.length_expected,
            config=config,
        )
        n += 1
        for check in CHECKS:
            counts[check] += int(getattr(flags, check))
    return SanityResult(n=n, counts=counts)
