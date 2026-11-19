"""Stripe webhook signatures (https://docs.stripe.com/webhooks#verify-manually).

Header: `Stripe-Signature: t=<unix seconds>,v1=<hex>[,v1=<hex>...][,v0=...]`.
`v1` = HMAC-SHA256 keyed with the endpoint secret (the whole `whsec_...` string) over
`"<t>.<raw request body>"`. A request is genuine if any `v1` matches (Stripe sends two
while a secret is being rolled) and `t` is within the tolerance of now (replay window,
default 300 s). Only `v1` counts; `v0` is ignored, as Stripe's libraries do.
"""

from __future__ import annotations

import hashlib
import hmac
import time

DEFAULT_TOLERANCE_S = 300
MAX_TIMESTAMP = 2**53  # a `t` beyond this is no unix time: a malformed header


class SignatureError(Exception):
    pass


class StaleTimestamp(SignatureError):
    """The HMAC is valid but `t` is outside the tolerance: a replay, or the box's clock
    is off (a genuine retry carries a fresh signature)."""


class SignatureMismatch(SignatureError):
    """A well-formed header (`t` and at least one `v1`) with `t` within the tolerance, but
    no `v1` matches: the secret on the box is not the endpoint's (or a forgery)."""


def compute(payload: bytes, secret: str, timestamp: int) -> str:
    signed = str(timestamp).encode() + b"." + payload
    return hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def sign(payload: bytes, secret: str, timestamp: int | None = None) -> str:
    """A `Stripe-Signature` header value for `payload` (used by the stub and tests)."""
    t = int(time.time()) if timestamp is None else timestamp
    return f"t={t},v1={compute(payload, secret, t)}"


def verify(
    payload: bytes,
    header: str | None,
    secret: str,
    *,
    tolerance_s: int = DEFAULT_TOLERANCE_S,
    now: float | None = None,
) -> int:
    """Raise SignatureError unless `header` signs `payload` with `secret` within the
    tolerance. Returns the signed timestamp. The subclasses say why: `StaleTimestamp`
    (valid HMAC, `t` outside the tolerance) and `SignatureMismatch` (no `v1` matches while
    `t` is current); anything else (a missing or malformed header, or no match with a
    stale `t`) is a plain `SignatureError`."""
    if not header:
        raise SignatureError("missing Stripe-Signature header")
    timestamp: int | None = None
    candidates: list[str] = []
    for part in header.split(","):
        k, sep, v = part.strip().partition("=")
        if not sep:
            continue
        if k == "t":
            try:
                timestamp = int(v)
            except ValueError:
                raise SignatureError("bad timestamp") from None
        elif k == "v1":
            candidates.append(v)
    if timestamp is None:
        raise SignatureError("no timestamp in Stripe-Signature")
    if not candidates:
        raise SignatureError("no v1 signature in Stripe-Signature")
    if abs(timestamp) > MAX_TIMESTAMP:
        raise SignatureError("bad timestamp")
    expected = compute(payload, secret, timestamp)
    matched = False
    malformed = False
    for c in candidates:
        try:
            matched = hmac.compare_digest(expected, c) or matched
        except TypeError:  # a non-ASCII v1: compare_digest takes ASCII str only
            malformed = True
    if malformed and not matched:
        raise SignatureError("bad v1 signature")
    current = time.time() if now is None else now
    fresh = abs(current - timestamp) <= tolerance_s
    if not matched:
        if fresh:
            raise SignatureMismatch("signature does not match")
        raise SignatureError("signature does not match")
    if not fresh:
        raise StaleTimestamp("timestamp outside the tolerance window")
    return timestamp
