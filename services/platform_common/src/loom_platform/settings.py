"""Platform settings, read from LOOM_* environment variables (docs/phase1-contracts.md).

Services subclass `PlatformSettings` for their own fields and keep the `LOOM_` prefix.
Unset fields take the defaults below, which are the local docker-compose values.

`LOOM_ENV` is `dev` or `prod`; unset means `prod`. docker-compose sets `dev`. Outside dev
a service refuses to start (`enforce_environment`) while any setting is still a dev
default or a secret it needs is missing (`production_problems`, extended per service),
or while its database role could switch off the ledger's guards
(`database_role_problems`: a superuser, or the owner of tables in the schema).
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
DEV_DATABASE_PASSWORD = "loom"  # docker-compose's postgres (owner) password
# docker-compose's password for the runtime login (loom_app, LOOM_APP_DATABASE_PASSWORD).
DEV_APP_DATABASE_PASSWORD = "loom-app-dev"
DEV_DATABASE_PASSWORDS = (DEV_DATABASE_PASSWORD, DEV_APP_DATABASE_PASSWORD)


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
        if url.password is None or urllib.parse.unquote(url.password) in DEV_DATABASE_PASSWORDS:
            problems.append("LOOM_DATABASE_URL has no password or the docker-compose one")
        rurl = urllib.parse.urlsplit(self.redis_url)
        if not rurl.password:
            problems.append("LOOM_REDIS_URL has no password (Redis AUTH)")
        if rurl.scheme != "rediss":
            problems.append("LOOM_REDIS_URL is not TLS (rediss://)")
        return problems


def database_role_problems(database_url: str, *, timeout_s: int = 5) -> list[str]:
    """Outside dev, at start, over the service's own connection: its role must not be a
    superuser and must own no table in the schema (directly or through a role it is a
    member of). Either could disable, drop or replace the ledger's append-only and
    zero-sum triggers (contracts §3); services run as the runtime role (`loom_app`).
    A database that cannot be reached is a problem too: the role is unchecked."""
    import psycopg

    from loom_platform.db import libpq_url

    try:
        with psycopg.connect(libpq_url(database_url), connect_timeout=timeout_s) as conn:
            row = conn.execute(
                "SELECT current_user, r.rolsuper, ARRAY(SELECT c.relname::text "
                "  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "  WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') "
                "  AND pg_has_role(c.relowner, 'MEMBER') ORDER BY 1) "
                "FROM pg_roles r WHERE r.rolname = current_user"
            ).fetchone()
    except (psycopg.Error, ValueError) as e:
        first = str(e).strip().splitlines()[0] if str(e).strip() else ""
        return [f"cannot check LOOM_DATABASE_URL's database role: {type(e).__name__}: {first}"]
    if row is None:  # current_user always has a pg_roles row
        return ["cannot check LOOM_DATABASE_URL's database role: not in pg_roles"]
    user, superuser, owned = row
    problems = []
    if superuser:
        problems.append(
            f"LOOM_DATABASE_URL's database role {user} is a superuser: services run as the "
            "runtime role (loom_app), which cannot switch off the ledger's guards"
        )
    if owned:
        names = ", ".join(owned[:5]) + (", ..." if len(owned) > 5 else "")
        tables = f"{len(owned)} table{'' if len(owned) == 1 else 's'}"
        problems.append(
            f"LOOM_DATABASE_URL's database role {user} owns {tables} in schema public "
            f"({names}): services run as the runtime role, not the migration owner"
        )
    return problems


def enforce_environment(
    settings: PlatformSettings,
    service: str,
    problems: list[str] | None = None,
    *,
    check_database_role: bool = True,
) -> None:
    """Exit (status 2) listing every problem when `settings` may not run in this LOOM_ENV
    (`problems` defaults to `settings.production_problems()`). Outside dev the
    database role is checked too (`database_role_problems`, connecting with
    `settings.database_url`), even when other problems already refuse the start, so the
    message lists them all; `check_database_role=False` for a process that never
    connects to Postgres."""
    problems = list(settings.production_problems() if problems is None else problems)
    if settings.env != "dev" and check_database_role:
        problems += database_role_problems(settings.database_url)
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        print(
            f"{service}: refusing to start with LOOM_ENV={settings.env}:\n{lines}\n"
            "Set LOOM_ENV=dev only for local development.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def redis_policy_problems(redis_url: str, *, timeout_s: float = 3.0) -> list[str]:
    """Outside dev, at start: Redis must never evict (a balance, reservation or stream
    key evicted is wrong, not slow). A problem when `maxmemory-policy` is anything but
    `noeviction`; only a warning on stderr when CONFIG GET is not permitted (managed
    Redis often disables it: check the parameter group instead) or Redis is
    unreachable (the service's readiness check covers that)."""
    import redis

    client = redis.Redis.from_url(
        redis_url, socket_timeout=timeout_s, socket_connect_timeout=timeout_s
    )
    try:
        policy = client.config_get("maxmemory-policy").get("maxmemory-policy")
    except redis.exceptions.ResponseError as e:
        print(
            f"warning: cannot read Redis maxmemory-policy ({e}); make sure it is noeviction",
            file=sys.stderr,
        )
        return []
    except redis.exceptions.RedisError as e:
        print(
            f"warning: Redis unreachable at start ({type(e).__name__}); "
            "maxmemory-policy not checked",
            file=sys.stderr,
        )
        return []
    finally:
        client.close()
    if policy != "noeviction":
        return [f"Redis maxmemory-policy is {policy!r}, not 'noeviction'"]
    return []


@lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    return PlatformSettings()
