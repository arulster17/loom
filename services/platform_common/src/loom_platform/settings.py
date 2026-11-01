"""Platform settings, read from LOOM_* environment variables (docs/phase1-contracts.md).

Services subclass `PlatformSettings` for their own fields and keep the `LOOM_` prefix.
Unset fields take the defaults below, which are the local docker-compose values.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOOM_", extra="ignore", frozen=True)

    # libpq URL; a SQLAlchemy-style 'postgresql+psycopg://' is accepted too.
    database_url: str = "postgresql://loom:loom@localhost:5432/loom_platform"
    redis_url: str = "redis://localhost:6379/0"
    # Path to config/models.yaml; None means loom_bench.registry's default lookup
    # ($LOOM_MODELS_YAML, else the repo's config/models.yaml).
    models_yaml: str | None = None

    usage_stream: str = "loom:usage"

    # Gateway auth cache: Redis TTL of a cached key record. Revocation and suspension
    # reach every gateway within this bound even if no invalidation is delivered.
    auth_cache_ttl_s: PositiveInt = 4
    # Hard upper bound on one request's duration; reservations live a minute longer.
    request_timeout_s: PositiveInt = 600
    reservation_grace_s: PositiveInt = 60

    # Free credit posted when an admin approves a signup ($5).
    signup_promo_micros: Annotated[int, Field(ge=0)] = 5_000_000

    # Request caps.
    max_body_bytes: PositiveInt = 1_048_576
    max_tokens_cap: PositiveInt = 16_384

    # Defaults when api_keys / orgs leave a limit NULL.
    default_key_rpm: PositiveInt = 60
    default_key_tpm: PositiveInt = 200_000
    default_key_concurrency: PositiveInt = 8
    default_org_rpm: PositiveInt = 120
    default_org_tpm: PositiveInt = 400_000
    default_org_concurrency: PositiveInt = 16

    @property
    def reservation_ttl_ms(self) -> int:
        return (self.request_timeout_s + self.reservation_grace_s) * 1000


@lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    return PlatformSettings()
