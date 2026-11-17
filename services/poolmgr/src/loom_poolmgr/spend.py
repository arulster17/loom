"""Serving spend: month to date, the create guard, alerts (docs/phase1-m2-spec.md §5
"Create").

- **Month to date** sums `cost_per_hr_micros x billed hours` over the rows that ran this
  UTC month: from `created_at` (or the month's start) to `terminated_at` (or now).
- **Guard**, checked before a row is written: `month_to_date + listed price x
  lease_ttl_s <= monthly cap`. Refused: no pod, pool `unavailable` (`spend_cap`), alert.
  The guard never terminates a running pod.
- **Alerts** fire once per (month, fraction of the cap) through `serving_spend_alerts`,
  so a restart does not repeat them.

Amounts are integer micro-dollars; times are aware UTC datetimes from the injected clock.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from decimal import ROUND_CEILING, Decimal

import psycopg

from loom_poolmgr import store
from loom_poolmgr.store import PodRow

MICROS_PER_USD = 1_000_000


def usd_micros(usd: float) -> int:
    return int((Decimal(str(usd)) * MICROS_PER_USD).to_integral_value(rounding=ROUND_CEILING))


def month_start(now: dt.datetime) -> dt.datetime:
    """Midnight UTC on the first of `now`'s UTC month."""
    utc = now.astimezone(dt.UTC)
    return utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def billed_micros(row: PodRow, *, since: dt.datetime, now: dt.datetime) -> int:
    """What `row` cost between `since` and `now` (rounded up to a micro-dollar)."""
    if row.cost_per_hr_micros is None:
        return 0
    start = max(row.created_at, since)
    end = min(row.terminated_at or now, now)
    if end <= start:
        return 0
    seconds = Decimal((end - start).total_seconds())
    return int((row.cost_per_hr_micros * seconds / 3600).to_integral_value(rounding=ROUND_CEILING))


def month_to_date(rows: Iterable[PodRow], now: dt.datetime) -> int:
    """Month-to-date spend of `rows` at `now`, in micros."""
    since = month_start(now)
    return sum(billed_micros(r, since=since, now=now) for r in rows)


def guard_allows(
    month_to_date_micros: int, listed_per_hr_micros: int, lease_ttl_s: int, cap_micros: int
) -> bool:
    """Whether one more pod, run for a whole lease, stays within the cap."""
    worst = Decimal(listed_per_hr_micros) * lease_ttl_s / 3600
    return month_to_date_micros + worst <= cap_micros


def crossed_fractions(
    month_to_date_micros: int, cap_micros: int, fractions: Sequence[float]
) -> list[float]:
    """The alert fractions month-to-date spend has reached, ascending."""
    return sorted(f for f in fractions if month_to_date_micros >= Decimal(str(f)) * cap_micros)


async def month_to_date_micros(conn: psycopg.AsyncConnection, now: dt.datetime) -> int:
    """`month_to_date` over this month's `serving_pods` rows."""
    since = month_start(now)
    rows = await store.rows_since(conn, since)
    return month_to_date(rows, now)


async def record_alerts(
    conn: psycopg.AsyncConnection, now: dt.datetime, fractions: Sequence[float]
) -> list[float]:
    """Insert (month, fraction) into `serving_spend_alerts` for each of `fractions`; return
    the ones not sent before (the alerts to send now). Commits."""
    month = month_start(now).date()
    sent: list[float] = []
    async with conn.transaction():
        for fraction in sorted(fractions):
            cur = await conn.execute(
                "INSERT INTO serving_spend_alerts (month, fraction, sent_at) VALUES (%s, %s, %s)"
                " ON CONFLICT (month, fraction) DO NOTHING RETURNING fraction",
                (month, Decimal(str(fraction)), now),
            )
            if await cur.fetchone() is not None:
                sent.append(fraction)
    if not conn.autocommit:
        await conn.commit()
    return sent
