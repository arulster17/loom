"""API key authentication (docs/phase1-contracts.md §5, §6 "Revocation within 5 s").

Lookup order: an in-process copy (at most `auth_local_ttl_s`, 1 s, and never past the
Redis entry's own expiry), then `loom:apikey:<sha256 hex>` in Redis (TTL
`LOOM_AUTH_CACHE_TTL_S`, 4 s), then Postgres, whose answer is written back to Redis. A
revoked key or a suspended org is therefore refused within the Redis TTL with no other
mechanism; the `loom_auth_changed` listener deletes cached records as soon as Postgres
commits the change, so in practice it takes milliseconds.

Unknown keys are cached the same way (`{"found":false}` in Redis for the TTL, and the
in-process copy), so repeating one invalid key costs one Postgres lookup per TTL. A
flood of distinct random keys still misses every cache; Postgres lookups are capped at
`PG_LOOKUPS` at a time so it cannot take the pool from balance loads (a per-IP limit or
WAF in front of the gateway is the M3 answer, contracts §8).

`api_keys.last_used_at` is written off the request path, at most once a minute per key
per gateway (and the UPDATE itself skips a row touched in the last minute).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

import psycopg
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from loom_platform import errors
from loom_platform.apikeys import InvalidApiKey, bearer_token, hash_api_key
from loom_platform.auth import AuthRecord, auth_cache_key, load_auth_record, put_cached
from loom_platform.db import libpq_url

log = logging.getLogger("loom_gateway.auth")

AUTH_CHANNEL = "loom_auth_changed"
_LOCAL_MAX = 100_000
PG_LOOKUPS = 4  # concurrent Postgres key lookups per gateway (cache misses only)
LAST_USED_INTERVAL_S = 60.0
_TOUCH = (
    "UPDATE api_keys SET last_used_at = now() WHERE id = %s "
    "AND (last_used_at IS NULL OR last_used_at < now() - interval '1 minute')"
)


class Authenticator:
    def __init__(
        self,
        redis: Redis,
        pg_pool: AsyncConnectionPool,
        *,
        redis_ttl_s: int,
        local_ttl_s: float,
    ) -> None:
        self.redis = redis
        self.pg_pool = pg_pool
        self.redis_ttl_s = redis_ttl_s
        self.local_ttl_s = local_ttl_s
        # key hash -> (monotonic expiry, record or None for an unknown key)
        self._local: dict[bytes, tuple[float, AuthRecord | None]] = {}
        # Bumped by every invalidation: a lookup that overlapped one does not keep a
        # local copy (it may have read the record just before it was dropped).
        self._epoch = 0
        self.listening = asyncio.Event()
        self._pg_lookups = asyncio.Semaphore(PG_LOOKUPS)
        # key id -> monotonic time of the last last_used_at write from this process
        self._touched: dict[uuid.UUID, float] = {}
        self._touch_tasks: set[asyncio.Task[None]] = set()

    async def authenticate(self, authorization: str | None) -> AuthRecord:
        """The key's record, or ApiError 401 (malformed, unknown, revoked) / 403 (org not
        active). A malformed token is refused before any lookup."""
        try:
            key_hash = hash_api_key(bearer_token(authorization))
        except InvalidApiKey:
            raise errors.invalid_api_key() from None
        record = await self.lookup(key_hash)
        if record is None or record.revoked:
            raise errors.invalid_api_key()
        if record.org_status != "active":
            raise errors.org_not_active(record.org_status)
        self._touch(record.key_id)
        return record

    def _touch(self, key_id: uuid.UUID) -> None:
        """Schedule a `last_used_at` write, at most once per LAST_USED_INTERVAL_S."""
        now = time.monotonic()
        last = self._touched.get(key_id)
        if last is not None and now - last < LAST_USED_INTERVAL_S:
            return
        if len(self._touched) >= _LOCAL_MAX:
            self._touched.clear()
        self._touched[key_id] = now
        task = asyncio.create_task(self._write_last_used(key_id))
        self._touch_tasks.add(task)
        task.add_done_callback(self._touch_tasks.discard)

    async def _write_last_used(self, key_id: uuid.UUID) -> None:
        try:
            async with self.pg_pool.connection() as conn:
                await conn.execute(_TOUCH, (key_id,))
        except Exception as e:  # best effort: never affects the request
            log.debug("last_used_at not written", extra={"fields": {"error": type(e).__name__}})

    async def lookup(self, key_hash: bytes) -> AuthRecord | None:
        now = time.monotonic()
        hit = self._local.get(key_hash)
        if hit is not None and hit[0] > now:
            return hit[1]
        epoch = self._epoch
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.get(auth_cache_key(key_hash))
            pipe.pttl(auth_cache_key(key_hash))
            raw, pttl = await pipe.execute()
        if raw is not None:
            record = AuthRecord.from_json(raw)
            # Never keep the copy past the Redis entry: that is what bounds revocation.
            ttl = min(self.local_ttl_s, max(int(pttl), 0) / 1000)
        else:
            async with self._pg_lookups, self.pg_pool.connection() as conn:
                record = await load_auth_record(conn, key_hash)
            await put_cached(self.redis, key_hash, record, self.redis_ttl_s)
            ttl = self.local_ttl_s
        if len(self._local) >= _LOCAL_MAX:
            self._local.clear()
        if ttl > 0 and epoch == self._epoch:
            self._local[key_hash] = (now + ttl, record)
        return record

    # ------------------------------------------------------------ invalidation

    def _drop_local(self, org_id: uuid.UUID | None = None, key_hash: bytes | None = None) -> None:
        self._epoch += 1
        if key_hash is not None:
            self._local.pop(key_hash, None)
        if org_id is not None:
            for kh, (_, rec) in list(self._local.items()):
                if rec is not None and rec.org_id == org_id:
                    self._local.pop(kh, None)

    async def drop_key(self, key_hash: bytes) -> None:
        self._drop_local(key_hash=key_hash)
        await self.redis.delete(auth_cache_key(key_hash))
        # Again: a lookup may have copied the Redis entry while it was being deleted.
        self._drop_local(key_hash=key_hash)

    async def drop_org(self, org_id: uuid.UUID) -> None:
        self._drop_local(org_id=org_id)
        async with self.pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT key_hash FROM api_keys WHERE org_id = %s", (org_id,))
            hashes = [bytes(r[0]) for r in await cur.fetchall()]
        if hashes:
            await self.redis.delete(*(auth_cache_key(h) for h in hashes))
        self._drop_local(org_id=org_id)

    async def on_notify(self, payload: str) -> None:
        try:
            msg: dict[str, Any] = json.loads(payload)
            if msg.get("kind") == "api_key":
                await self.drop_key(bytes.fromhex(msg["key_hash"]))
            elif msg.get("kind") == "org":
                await self.drop_org(uuid.UUID(msg["org_id"]))
        except Exception:
            # The TTL still bounds a missed invalidation.
            log.exception("auth invalidation failed", extra={"fields": {"channel": AUTH_CHANNEL}})

    async def listen_forever(self, database_url: str) -> None:
        """LISTEN loom_auth_changed and drop cached records; reconnect on failure. A
        reconnect needs no resync: the TTL covers anything missed meanwhile."""
        backoff = 0.5
        while True:
            try:
                conn = await psycopg.AsyncConnection.connect(
                    libpq_url(database_url), autocommit=True
                )
                async with conn:
                    await conn.execute(f"LISTEN {AUTH_CHANNEL}")
                    self.listening.set()
                    backoff = 0.5
                    async for note in conn.notifies():
                        await self.on_notify(note.payload)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.listening.clear()
                log.warning(
                    "auth listener disconnected",
                    extra={"fields": {"error": type(e).__name__, "retry_in_s": backoff}},
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
