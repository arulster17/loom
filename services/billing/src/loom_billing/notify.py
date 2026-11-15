"""Notifications outbox and the auto-recharge hook.

The outbox is a Redis stream (`LOOM_NOTIFY_STREAM`, default `loom:notify`) of flat
string entries: `type`, `org_id`, `balance_micros`, `threshold_micros`, `at` (and
`amount_micros` for `auto_recharge_due`). The web app's email layer reads it with its
own consumer group and sends the email; billing only decides *when*. In local dev
nothing consumes it; the entries and the `notification.emitted` log line are the record.

Types:
- `low_balance`: the org's balance (credit - usage) fell below the threshold. Emitted
  once per crossing: a marker key `loom:org:<org>:lowbal` is set in the same Lua call
  that appends the entry, and cleared when the balance is seen at or above the
  threshold again.
- `auto_recharge_due`: auto-recharge is on for the org and its balance is below its
  `auto_recharge.threshold_micros`. Charging a saved card off-session needs a Stripe
  customer and payment method, which M1 does not have; this event is the hook point.
  Same once-per-crossing marker (`loom:org:<org>:recharge`).
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from psycopg import AsyncConnection
from redis.asyncio import Redis

from loom_billing import metrics
from loom_billing.logs import fields
from loom_platform.limiter import org_key

log = logging.getLogger("loom_billing.notify")

MARKER_TTL_S = 30 * 86_400

# KEYS: marker, stream. ARGV: ttl, then the entry's name/value pairs.
# Appends only if the marker was absent, atomically with setting it.
_EMIT_ONCE = """
if redis.call('SET', KEYS[1], '1', 'NX', 'EX', ARGV[1]) then
  local entry = {}
  for i = 2, #ARGV do entry[#entry + 1] = ARGV[i] end
  return redis.call('XADD', KEYS[2], '*', unpack(entry))
end
return false
"""


class Notifier:
    def __init__(self, redis: Redis, *, stream: str, low_balance_micros: int) -> None:
        self.redis = redis
        self.stream = stream
        self.low_balance_micros = low_balance_micros
        self._emit = redis.register_script(_EMIT_ONCE)

    async def _emit_once(
        self,
        marker: str,
        kind: str,
        org_id: uuid.UUID,
        balance: int,
        threshold: int,
        **extra: str,
    ) -> bool:
        entry = [
            "type", kind,
            "org_id", str(org_id),
            "balance_micros", str(balance),
            "threshold_micros", str(threshold),
            "at", dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        ]  # fmt: skip
        for k, v in extra.items():
            entry += [k, v]
        res = await self._emit(keys=[marker, self.stream], args=[MARKER_TTL_S, *entry])
        if not res:
            return False
        metrics.NOTIFICATIONS.labels(kind).inc()
        log.info(
            "notification.emitted",
            extra=fields(type=kind, org_id=str(org_id), balance_micros=balance,
                         threshold_micros=threshold),
        )  # fmt: skip
        return True

    async def check_low_balance(self, org_id: uuid.UUID, balance: int) -> bool:
        """Emit `low_balance` on a downward crossing. Returns True if emitted."""
        marker = org_key(org_id, "lowbal")
        if balance >= self.low_balance_micros:
            await self.redis.delete(marker)
            return False
        return await self._emit_once(
            marker, "low_balance", org_id, balance, self.low_balance_micros
        )


class AutoRecharge:
    """Hook point for auto-recharge, behind LOOM_AUTO_RECHARGE_ENABLED (default off)."""

    def __init__(self, notifier: Notifier, *, enabled: bool) -> None:
        self.notifier = notifier
        self.enabled = enabled

    async def check(self, conn: AsyncConnection, org_id: uuid.UUID, balance: int) -> bool:
        if not self.enabled:
            return False
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT enabled, threshold_micros, amount_micros FROM auto_recharge "
                "WHERE org_id = %s",
                (org_id,),
            )
            row = await cur.fetchone()
        marker = org_key(org_id, "recharge")
        if row is None or not row[0] or balance >= row[1]:
            await self.notifier.redis.delete(marker)
            return False
        # Real Stripe: create an off-session PaymentIntent for row[2] on the org's saved
        # payment method; its webhook then credits through the normal purchase path.
        return await self.notifier._emit_once(
            marker, "auto_recharge_due", org_id, balance, int(row[1]), amount_micros=str(row[2])
        )
