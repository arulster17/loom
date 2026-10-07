"""Deterministic pseudo next-token distributions.

Position i's distribution depends only on (seed, tokens[:i]) and the sampled
token tokens[i], which always has the highest base logit. `noise` adds Gaussian
logit noise from a separate stream, so servers that differ only in noise share
the base distribution and their KL divergence grows with the noise level.

`jitter` adds Gaussian logit noise drawn per request (`draw`, a counter the
server advances), standing in for batch-variant kernels: the same text scored
twice gets slightly different logprobs.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

MAX_LOGPROBS = 20

VOCAB = tuple(
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
    tokens: Sequence[str],
    *,
    seed: int,
    start: int,
    k: int,
    noise: float,
    jitter: float = 0.0,
    draw: int = 0,
    vocab: Sequence[str] = (),
) -> list[TokenLogprob]:
    """Logprobs of tokens[start:], each conditioned on the tokens before it. `vocab`
    (default: the mock's word vocabulary) is where alternatives are drawn from."""
    ctx = hashlib.blake2b(str(seed).encode(), digest_size=16)
    out: list[TokenLogprob] = []
    for i, token in enumerate(tokens):
        if i >= start:
            out.append(_distribution(ctx.digest(), token, k, noise, jitter, draw, vocab or VOCAB))
        data = token.encode()
        ctx.update(len(data).to_bytes(4, "big"))
        ctx.update(data)
    return out


def _distribution(
    ctx: bytes, token: str, k: int, noise: float, jitter: float, draw: int, vocab: Sequence[str]
) -> TokenLogprob:
    rng = random.Random(ctx)
    alternatives = [t for t in rng.sample(vocab, MAX_LOGPROBS) if t != token]
    candidates = [token, *alternatives[: MAX_LOGPROBS - 1]]
    logits = [0.0] + [-rng.uniform(0.5, 8.0) for _ in candidates[1:]]
    if noise:
        noise_rng = random.Random(ctx + b"noise")
        logits = [x + noise * noise_rng.gauss(0.0, 1.0) for x in logits]
    if jitter:
        jitter_rng = random.Random(ctx + b"jitter" + draw.to_bytes(8, "big"))
        logits = [x + jitter * jitter_rng.gauss(0.0, 1.0) for x in logits]
    norm = math.log(sum(math.exp(x) for x in logits) + math.exp(_TAIL_LOGIT))
    ranked = sorted(zip(candidates, (x - norm for x in logits), strict=True), key=lambda c: -c[1])
    return TokenLogprob(token=token, logprob=logits[0] - norm, top=tuple(ranked[:k]))
