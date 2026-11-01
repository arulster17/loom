"""API-key auth records: loaded from Postgres, cached in Redis under
`loom:apikey:<sha256 hex>` for LOOM_AUTH_CACHE_TTL_S seconds (docs/phase1-contracts.md).

The short TTL is what guarantees that a revoked key or a suspended org is refused
within 5 s on every gateway, even if no invalidation message arrives; the
`loom_auth_changed` NOTIFY (migration 0002) only makes it faster.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import psycopg
from redis.asyncio import Redis

from loom_platform.limiter import Limits
from loom_platform.settings import PlatformSettings

_NOT_FOUND = '{"found":false}'


def auth_cache_key(key_hash: bytes) -> str:
    return f"loom:apikey:{key_hash.hex()}"


@dataclass(frozen=True)
class AuthRecord:
    key_id: uuid.UUID
    org_id: uuid.UUID
    org_status: str  # pending_approval | active | suspended | rejected
    revoked: bool
    scopes: tuple[str, ...]
    spend_limit_micros: int | None
    # Overrides from api_keys / orgs; None = the platform default.
    key_rpm: int | None
    key_tpm: int | None
    key_concurrency: int | None
    org_rpm: int | None
    org_tpm: int | None
    org_concurrency: int | None

    @property
    def usable(self) -> bool:
        return not self.revoked and self.org_status == "active"

    def limits(self, s: PlatformSettings) -> Limits:
        return Limits(
            key_rpm=self.key_rpm or s.default_key_rpm,
            key_tpm=self.key_tpm or s.default_key_tpm,
            org_rpm=self.org_rpm or s.default_org_rpm,
            org_tpm=self.org_tpm or s.default_org_tpm,
            key_concurrency=self.key_concurrency or s.default_key_concurrency,
            org_concurrency=self.org_concurrency or s.default_org_concurrency,
            key_spend_limit_micros=self.spend_limit_micros,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "found": True,
                "key_id": str(self.key_id),
                "org_id": str(self.org_id),
                "org_status": self.org_status,
                "revoked": self.revoked,
                "scopes": list(self.scopes),
                "spend_limit_micros": self.spend_limit_micros,
                "key_rpm": self.key_rpm,
                "key_tpm": self.key_tpm,
                "key_concurrency": self.key_concurrency,
                "org_rpm": self.org_rpm,
                "org_tpm": self.org_tpm,
                "org_concurrency": self.org_concurrency,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> AuthRecord | None:
        d = json.loads(text)
        if not d.get("found"):
            return None
        return cls(
            key_id=uuid.UUID(d["key_id"]),
            org_id=uuid.UUID(d["org_id"]),
            org_status=d["org_status"],
            revoked=d["revoked"],
            scopes=tuple(d["scopes"]),
            spend_limit_micros=d["spend_limit_micros"],
            key_rpm=d["key_rpm"],
            key_tpm=d["key_tpm"],
            key_concurrency=d["key_concurrency"],
            org_rpm=d["org_rpm"],
            org_tpm=d["org_tpm"],
            org_concurrency=d["org_concurrency"],
        )


async def load_auth_record(conn: psycopg.AsyncConnection, key_hash: bytes) -> AuthRecord | None:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT k.id, k.org_id, o.status, k.revoked_at IS NOT NULL, k.scopes,
                   k.spend_limit_micros, k.rpm_limit, k.tpm_limit, k.concurrency_limit,
                   o.rpm_limit, o.tpm_limit, o.concurrency_limit
            FROM api_keys k JOIN orgs o ON o.id = k.org_id
            WHERE k.key_hash = %s
            """,
            (key_hash,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return AuthRecord(
        key_id=row[0],
        org_id=row[1],
        org_status=row[2],
        revoked=row[3],
        scopes=tuple(row[4]),
        spend_limit_micros=row[5],
        key_rpm=row[6],
        key_tpm=row[7],
        key_concurrency=row[8],
        org_rpm=row[9],
        org_tpm=row[10],
        org_concurrency=row[11],
    )


class _Miss:
    pass


MISS = _Miss()


async def get_cached(redis: Redis, key_hash: bytes) -> AuthRecord | _Miss | None:
    """The cached record, None for a cached 'no such key', or MISS."""
    raw = await redis.get(auth_cache_key(key_hash))
    if raw is None:
        return MISS
    return AuthRecord.from_json(raw)


async def put_cached(redis: Redis, key_hash: bytes, record: AuthRecord | None, ttl_s: int) -> None:
    """Cache a record, or a negative result (unknown key) for the same TTL."""
    value = _NOT_FOUND if record is None else record.to_json()
    await redis.set(auth_cache_key(key_hash), value, ex=ttl_s)


async def drop_cached(redis: Redis, key_hash: bytes) -> None:
    await redis.delete(auth_cache_key(key_hash))
