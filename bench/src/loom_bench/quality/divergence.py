"""Teacher-forced logprob divergence of a candidate endpoint from a reference.

Method:

1. The reference (normally the BF16 config) greedily continues each pinned
   prompt once, `max_new_tokens` tokens.
2. Both endpoints score the same text, prompt + continuation, through
   `/v1/completions` with ``echo: true, logprobs: k, max_tokens: 1``. Echo
   returns, for every prompt token, the top-k next-token distribution at that
   position, so both servers are conditioned on identical text (teacher
   forcing) and differences come from the weights / kernels alone.
3. For every continuation position we compare the two top-k lists:
   top-1 agreement (same argmax token) and an approximate KL(ref || cand).
4. Per prompt we average over its positions; the reported numbers are the
   mean over prompts with a percentile-bootstrap CI that resamples prompts
   (positions within a prompt are correlated, prompts are not).

KL approximation (servers only return the top-k): over U = union of both
top-k token sets plus one "other" bucket. For each distribution, tokens of U
missing from its own top-k get an equal share of its unlisted mass
(1 - sum of listed probabilities), capped at its smallest listed probability
since they ranked below it; what is left of the unlisted mass goes to "other".
Both vectors are floored at 1e-10 and renormalised before computing
sum P log(P / Q). With identical top-k lists this is exact on the coarsened
distribution (a lower bound of the true KL by the data-processing
inequality); when the lists differ the imputed masses make it an estimate,
which grows sharply when one side's confident token is missing from the
other's list - the case that matters.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from loom_bench.quality.client import EvalClient
from loom_bench.stats import Interval, bootstrap_ci

EPS = 1e-10


def _filled(dist: Mapping[str, float], union: Sequence[str]) -> list[float]:
    probs = {t: math.exp(lp) for t, lp in dist.items()}
    listed = sum(probs.values())
    missing = [t for t in union if t not in probs]
    if missing:
        unlisted = max(0.0, 1.0 - listed)
        share = min(unlisted / (len(missing) + 1), min(probs.values(), default=unlisted))
        for t in missing:
            probs[t] = share
    other = max(0.0, 1.0 - sum(probs.values()))
    vec = np.array([probs[t] for t in union] + [other])
    vec = np.maximum(vec, EPS)
    return list(vec / vec.sum())


def approx_kl(ref: Mapping[str, float], cand: Mapping[str, float]) -> float:
    """Approximate KL(ref || cand) in nats from two top-k {token: logprob} maps."""
    if not ref or not cand:
        raise ValueError("both distributions need at least one listed token")
    union = sorted(set(ref) | set(cand))
    p = np.array(_filled(ref, union))
    q = np.array(_filled(cand, union))
    return max(0.0, float(np.sum(p * np.log(p / q))))


def top1(dist: Mapping[str, float]) -> str:
    """Highest-logprob token; ties go to the lexicographically smallest token."""
    return min(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]


@dataclass(frozen=True, slots=True)
class Position:
    token: str
    top: dict[str, float]


def scored_positions(logprobs: Mapping[str, Any], start: int, end: int) -> list[Position]:
    """Echoed positions whose token starts in text[start:end].

    `logprobs` is a completions `choices[0].logprobs` object (tokens,
    top_logprobs, text_offset). The final generated token sits at offset
    `end` and is excluded.
    """
    tokens, tops, offsets = logprobs["tokens"], logprobs["top_logprobs"], logprobs["text_offset"]
    if not (len(tokens) == len(tops) == len(offsets)):
        raise ValueError("logprobs arrays differ in length")
    out = []
    for token, top, offset in zip(tokens, tops, offsets, strict=True):
        if start <= offset < end:
            if not top:
                raise ValueError("position without top_logprobs")
            out.append(Position(token, dict(top)))
    return out


@dataclass(frozen=True, slots=True)
class PromptDivergence:
    kl: list[float]
    agree: list[bool]

    @property
    def mean_kl(self) -> float:
        return float(np.mean(self.kl))

    @property
    def top1_rate(self) -> float:
        return float(np.mean(self.agree))


def compare_positions(ref: Sequence[Position], cand: Sequence[Position]) -> PromptDivergence:
    if [p.token for p in ref] != [p.token for p in cand]:
        raise ValueError(
            "reference and candidate tokenized the same text differently; "
            "divergence needs the same tokenizer on both sides"
        )
    if not ref:
        raise ValueError("no positions to compare")
    return PromptDivergence(
        kl=[approx_kl(r.top, c.top) for r, c in zip(ref, cand, strict=True)],
        agree=[top1(r.top) == top1(c.top) for r, c in zip(ref, cand, strict=True)],
    )


class DivergenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_prompts: int
    n_positions: int
    n_skipped: int  # prompts where the reference produced no continuation
    top_k: int
    kl: Interval  # mean over prompts of per-prompt mean KL(ref || cand), nats
    top1: Interval  # mean over prompts of per-prompt top-1 agreement rate


def aggregate(
    prompts: Sequence[PromptDivergence],
    *,
    top_k: int,
    n_skipped: int = 0,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> DivergenceResult:
    if not prompts:
        raise ValueError("no prompts to aggregate")
    return DivergenceResult(
        n_prompts=len(prompts),
        n_positions=sum(len(p.kl) for p in prompts),
        n_skipped=n_skipped,
        top_k=top_k,
        kl=bootstrap_ci(
            [p.mean_kl for p in prompts], confidence=confidence, n_boot=n_boot, seed=seed
        ),
        top1=bootstrap_ci(
            [p.top1_rate for p in prompts], confidence=confidence, n_boot=n_boot, seed=seed
        ),
    )


def load_prompts() -> list[str]:
    """The pinned divergence prompt set shipped with the package."""
    text = resources.files("loom_bench.quality") / "data" / "divergence_prompts.yaml"
    data = yaml.safe_load(text.read_text(encoding="utf-8"))
    return [str(p) for p in data["prompts"]]


async def _score(client: EvalClient, text: str, top_k: int) -> Mapping[str, Any]:
    res = await client.complete(text, max_tokens=1, echo=True, logprobs=top_k)
    if res.logprobs is None:
        raise ValueError("server returned no logprobs for an echo request")
    return res.logprobs


async def measure_divergence(
    reference: EvalClient,
    candidate: EvalClient,
    prompts: Sequence[str],
    *,
    top_k: int = 5,
    max_new_tokens: int = 64,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> DivergenceResult:
    async def one(prompt: str) -> PromptDivergence | None:
        cont = await reference.complete(prompt, max_tokens=max_new_tokens)
        if not cont.text:
            return None
        text = prompt + cont.text
        ref_lp, cand_lp = await asyncio.gather(
            _score(reference, text, top_k), _score(candidate, text, top_k)
        )
        span = (len(prompt), len(text))
        return compare_positions(scored_positions(ref_lp, *span), scored_positions(cand_lp, *span))

    results = await asyncio.gather(*(one(p) for p in prompts))
    kept = [r for r in results if r is not None]
    return aggregate(
        kept,
        top_k=top_k,
        n_skipped=len(results) - len(kept),
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
    )
