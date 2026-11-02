"""Platform settings, read from LOOM_* environment variables (docs/phase1-contracts.md).

Services subclass `PlatformSettings` for their own fields and keep the `LOOM_` prefix.
Unset fields take the defaults below, which are the local docker-compose values.

`LOOM_ENV` is `dev` or `prod`; unset means `prod`. docker-compose sets `dev`. Outside dev
a service refuses to start (`enforce_environment`) while any setting is still a dev
default or a secret it needs is missing (`production_problems`, extended per service).
"""

from __future__ import annotations

import sys
import urllib.parse
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PositiveInt = Annotated[int, Field(gt=0)]
LoomEnv = Literal["dev", "prod"]
DEV_DATABASE_PASSWORD = "loom"  # docker-compose's postgres password


class PlatformSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOOM_", extra="ignore", frozen=True)

    # dev | prod (unset: prod). Only dev may run on the dev defaults below.
    env: LoomEnv = "prod"
    # libpq URL; a SQLAlchemy-style 'postgresql+psycopg://' is accepted too.
    database_url: str = "postgresql://loom:loom@localhost:5432/loom_platform"
    redis_url: str = "redis://localhost:6379/0"
    # Path to config/models.yaml; None means loom_bench.registry's default lookup
    # ($LOOM_MODELS_YAML, else the repo's config/models.yaml).
    models_yaml: str | None = None

    usage_stream: str = "loom:usage"
    # Redis client timeouts, so a frozen or unreachable Redis fails a call quickly (the
    # gateway answers 503 and /ready fails) instead of hanging it. Billing raises the
    # read timeout above its blocking XREADGROUP wait.
    redis_socket_timeout_s: Annotated[float, Field(gt=0)] = 2.0
    redis_connect_timeout_s: Annotated[float, Field(gt=0)] = 1.0

    # Gateway auth cache: Redis TTL of a cached key record. Revocation and suspension
    # reach every gateway within this bound even if no invalidation is delivered.
    auth_cache_ttl_s: PositiveInt = 4
    # Hard upper bound on one request's duration; reservations live a minute longer.
    request_timeout_s: PositiveInt = 600
    reservation_grace_s: PositiveInt = 60
    # How long a spooled settle may wait for Redis and still be refused as a repeat by
    # its `done:` marker (the marker lives the reservation TTL plus this).
    settle_replay_window_s: PositiveInt = 3600

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

    @property
    def settled_ttl_s(self) -> int:
        """Lifetime of a `done:<request id>` marker: the reservation TTL plus the settle
        replay window (contracts §6)."""
        return self.request_timeout_s + self.reservation_grace_s + self.settle_replay_window_s

    def production_problems(self) -> list[str]:
        """Why this process must not start outside LOOM_ENV=dev (empty in dev). Services
        extend it with their own secrets and dev defaults."""
        if self.env == "dev":
            return []
        problems = []
        url = urllib.parse.urlsplit(self.database_url)
        if url.password is None or url.password == DEV_DATABASE_PASSWORD:
            problems.append("LOOM_DATABASE_URL has no password or the docker-compose one")
        return problems


def enforce_environment(
    settings: PlatformSettings, service: str, problems: list[str] | None = None
) -> None:
    """Exit (status 2) listing every problem when `settings` may not run in this LOOM_ENV
    (`problems` defaults to `settings.production_problems()`)."""
    problems = settings.production_problems() if problems is None else problems
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        print(
            f"{service}: refusing to start with LOOM_ENV={settings.env}:\n{lines}\n"
            "Set LOOM_ENV=dev only for local development.",
            file=sys.stderr,
        )
        raise SystemExit(2)


@lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    return PlatformSettings()
