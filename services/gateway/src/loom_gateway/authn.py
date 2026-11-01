"""API key authentication (docs/phase1-contracts.md §5, §6 "Revocation within 5 s").

Lookup order: an in-process copy (at most `auth_local_ttl_s`, 1 s, and never past the
Redis entry's own expiry), then `loom:apikey:<sha256 hex>` in Redis (TTL
`LOOM_AUTH_CACHE_TTL_S`, 4 s), then Postgres, whose answer is written back to Redis. A
revoked key or a suspended org is therefore refused within the Redis TTL with no other
mechanism; the `loom_auth_changed` listener deletes cached records as soon as Postgres
commits the change, so in practice it takes milliseconds.
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
        self.listening = asyncio.Event()

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
        return record

    async def lookup(self, key_hash: bytes) -> AuthRecord | None:
        now = time.monotonic()
        hit = self._local.get(key_hash)
        if hit is not None and hit[0] > now:
            return hit[1]
        async with self.redis.pipeline(transaction=False) as pipe:
            pipe.get(auth_cache_key(key_hash))
            pipe.pttl(auth_cache_key(key_hash))
            raw, pttl = await pipe.execute()
        if raw is not None:
            record = AuthRecord.from_json(raw)
            # Never keep the copy past the Redis entry: that is what bounds revocation.
            ttl = min(self.local_ttl_s, max(int(pttl), 0) / 1000)
        else:
            async with self.pg_pool.connection() as conn:
                record = await load_auth_record(conn, key_hash)
            await put_cached(self.redis, key_hash, record, self.redis_ttl_s)
            ttl = self.local_ttl_s
        if len(self._local) >= _LOCAL_MAX:
            self._local.clear()
        if ttl > 0:
            self._local[key_hash] = (now + ttl, record)
        return record

    # ------------------------------------------------------------ invalidation

    async def drop_key(self, key_hash: bytes) -> None:
        self._local.pop(key_hash, None)
        await self.redis.delete(auth_cache_key(key_hash))

    async def drop_org(self, org_id: uuid.UUID) -> None:
        for kh, (_, rec) in list(self._local.items()):
            if rec is not None and rec.org_id == org_id:
                self._local.pop(kh, None)
        async with self.pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SELECT key_hash FROM api_keys WHERE org_id = %s", (org_id,))
            hashes = [bytes(r[0]) for r in await cur.fetchall()]
        if hashes:
            await self.redis.delete(*(auth_cache_key(h) for h in hashes))

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
