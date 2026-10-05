"""Money is integer micro-dollars (1 USD = 1_000_000). Never floats.

Cost math goes through Fraction and rounds once, half-up, at the end.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

MICROS_PER_USD = 1_000_000
SECONDS_PER_HOUR = 3600
TOKENS_PER_MTOK = 1_000_000

Micros = int

_USD_RE = re.compile(r"^\s*\$?\s*(-?\d+(?:\.\d+)?)\s*$")


def round_half_up(value: Fraction) -> int:
    """Round a Fraction to the nearest int, ties away from zero."""
    sign = -1 if value < 0 else 1
    mag = abs(value)
    return sign * int(mag + Fraction(1, 2))


def parse_usd(text: str | int) -> Micros:
    """Parse "$1.23" / "1.23" into micros. Ints are taken as micros already.

    Rejects sub-micro precision instead of silently rounding.
    """
    if isinstance(text, bool):
        raise TypeError("bool is not money")
    if isinstance(text, int):
        return text
    m = _USD_RE.match(text)
    if not m:
        raise ValueError(f"not a USD amount: {text!r}")
    try:
        micros = Decimal(m.group(1)) * MICROS_PER_USD
    except InvalidOperation as e:  # pragma: no cover - regex already guards
        raise ValueError(f"not a USD amount: {text!r}") from e
    if micros != micros.to_integral_value():
        raise ValueError(f"more precision than one micro-dollar: {text!r}")
    return int(micros)


def format_usd(micros: Micros, places: int = 4) -> str:
    """Human display only. Do not parse the result back for math."""
    q = Decimal(micros) / MICROS_PER_USD
    return f"${q:.{places}f}" if micros >= 0 else f"-${-q:.{places}f}"


def cost_for_seconds(hourly_micros: Micros, seconds: float | Fraction) -> Micros:
    """Spend for running a resource billed per hour for `seconds`."""
    secs = Fraction(seconds)
    return round_half_up(Fraction(hourly_micros) * secs / SECONDS_PER_HOUR)


def cost_per_mtok(hourly_micros: Micros, tokens_per_second: float | Fraction) -> Micros | None:
    """Micro-dollars per 1M tokens for a resource producing `tokens_per_second`.

    Returns None when throughput is zero (cost is unbounded).
    """
    tps = Fraction(tokens_per_second)
    if tps <= 0:
        return None
    tokens_per_hour = tps * SECONDS_PER_HOUR
    return round_half_up(Fraction(hourly_micros) * TOKENS_PER_MTOK / tokens_per_hour)


def cost_for_tokens(per_mtok_micros: Micros, tokens: int) -> Micros:
    """Charge for `tokens` at a per-1M-token price."""
    return round_half_up(Fraction(per_mtok_micros) * tokens / TOKENS_PER_MTOK)
