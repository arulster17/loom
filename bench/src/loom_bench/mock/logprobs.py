"""Deterministic pseudo next-token distributions.

Position i's distribution depends only on (seed, tokens[:i]) and the sampled
token tokens[i], which always has the highest base logit. `noise` adds Gaussian
logit noise from a separate stream, so servers that differ only in noise share
the base distribution and their KL divergence grows with the noise level.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

MAX_LOGPROBS = 20

_VOCAB = tuple(
    [str(d) for d in range(10)]
    + list(".,;:!?()")
    + [
        " " + w
        for w in (
            "the of and to in is was for on that with as by at from it an are this be or "
            "have not but one all they there more when will would can out so what up about "
            "into than only other new time two first like over made after also many before"
        ).split()
    ]
)
# Unnormalised log-mass of every token outside the candidate set.
_TAIL_LOGIT = -2.0


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token: str
    logprob: float
    top: tuple[tuple[str, float], ...]  # descending by logprob


def token_logprobs(
    tokens: Sequence[str], *, seed: int, start: int, k: int, noise: float
) -> list[TokenLogprob]:
    """Logprobs of tokens[start:], each conditioned on the tokens before it."""
    ctx = hashlib.blake2b(str(seed).encode(), digest_size=16)
    out: list[TokenLogprob] = []
    for i, token in enumerate(tokens):
        if i >= start:
            out.append(_distribution(ctx.digest(), token, k, noise))
        data = token.encode()
        ctx.update(len(data).to_bytes(4, "big"))
        ctx.update(data)
    return out


def _distribution(ctx: bytes, token: str, k: int, noise: float) -> TokenLogprob:
    rng = random.Random(ctx)
    alternatives = [t for t in rng.sample(_VOCAB, MAX_LOGPROBS) if t != token]
    candidates = [token, *alternatives[: MAX_LOGPROBS - 1]]
    logits = [0.0] + [-rng.uniform(0.5, 8.0) for _ in candidates[1:]]
    if noise:
        noise_rng = random.Random(ctx + b"noise")
        logits = [x + noise * noise_rng.gauss(0.0, 1.0) for x in logits]
    norm = math.log(sum(math.exp(x) for x in logits) + math.exp(_TAIL_LOGIT))
    ranked = sorted(zip(candidates, (x - norm for x in logits), strict=True), key=lambda c: -c[1])
    return TokenLogprob(token=token, logprob=logits[0] - norm, top=tuple(ranked[:k]))
