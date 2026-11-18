"""Platform settings, read from LOOM_* environment variables (docs/phase1-contracts.md).

Services subclass `PlatformSettings` for their own fields and keep the `LOOM_` prefix.
Unset fields take the defaults below, which are the local docker-compose values.

`LOOM_ENV` is `dev` or `prod`; unset means `prod`. docker-compose sets `dev`. Outside dev
a service refuses to start (`enforce_environment`) while any setting is still a dev
default or a secret it needs is missing (`production_problems`, extended per service),
or while its database role could switch off the ledger's guards
(`database_role_problems`: a superuser, or the owner of tables in the schema). When
Postgres cannot be reached at start (any connection error but an explicit
misconfiguration: `connection_outage`), the service starts with a warning and the role
is checked on its first successful connection, before any work
(`loom_platform.db.role_gate`).
"""

from __future__ import annotations

import sys
import urllib.parse
from functools import lru_cache
from typing import Annotated, Literal, NoReturn

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


class DatabaseUnreachable(Exception):
    """Postgres could not be connected to (an outage, `connection_outage`), so the
    database role is not checked yet."""


# Only an outage counts as unreachable, not a misconfiguration (contracts §3 "Roles").
# A connection error naming one of these exits 2, at start and in the deferred check
# (`loom_platform.db.role_gate`): retrying cannot fix it. Every other connection error
# is an outage and is retried (refused or timed out, a host name that does not resolve,
# network unreachable, a missing socket, any "the database system is ..." state, too
# many clients, a connection closed unexpectedly, and anything unforeseen).
MISCONFIGURATION_MARKERS = (
    "authentication failed",  # any method: password, peer, ident, cert, ldap, ...
    "does not exist",  # a role or a database
    "no pg_hba.conf entry",
    "pg_hba.conf rejects",
)


def connection_outage(e: BaseException) -> bool:
    """Whether a failed Postgres connection is an outage, to be retried: every
    connection error except an explicit misconfiguration (`MISCONFIGURATION_MARKERS`,
    found anywhere in the message: libpq lists every address it tried), which exits
    2."""
    text = str(e).lower()
    return not any(m in text for m in MISCONFIGURATION_MARKERS)


def connection_error_summary(e: BaseException) -> str:
    """`error_summary`, but naming the misconfiguration when libpq's message has it on a
    later line (one line per address tried)."""
    for line in str(e).splitlines():
        if any(m in line.lower() for m in MISCONFIGURATION_MARKERS):
            return f"{type(e).__name__}: {line.strip()}"
    return error_summary(e)


def connection_problem(e: BaseException) -> str:
    """The exit-2 problem for a connection error that is not an outage."""
    return f"cannot connect with LOOM_DATABASE_URL: {connection_error_summary(e)}"


# The role check (contracts §3 "Roles"): system catalogs and built-in functions only, so
# it names nothing of the schema and does no work as a role that may turn out wrong.
# Both the login (session_user) and the role it acts as (current_user) count: either
# can be switched to over the connection.
ROLE_CHECK_SQL = (
    "SELECT session_user::text, current_user::text, "
    "  ARRAY(SELECT r.rolname::text FROM pg_roles r "
    "   WHERE r.rolsuper AND r.rolname IN (session_user, current_user) ORDER BY 1), "
    "  ARRAY(SELECT c.relname::text FROM pg_class c "
    "   JOIN pg_namespace n ON n.oid = c.relnamespace "
    "   WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') "
    "   AND (pg_has_role(session_user, c.relowner, 'MEMBER') "
    "     OR pg_has_role(current_user, c.relowner, 'MEMBER')) ORDER BY 1)"
)


def error_summary(e: BaseException) -> str:
    """An error's type and the first line of its message."""
    text = str(e).strip()
    return f"{type(e).__name__}: {text.splitlines()[0]}" if text else type(e).__name__


def role_problems(row: tuple[object, ...] | None) -> list[str]:
    """The problems in a `ROLE_CHECK_SQL` row."""
    if row is None:  # the query always returns one row
        return ["cannot check LOOM_DATABASE_URL's database role: no answer"]
    login, acting, superusers, owned = row
    user = login if acting == login else f"{login} (as {acting})"
    problems = []
    for name in superusers if isinstance(superusers, list) else []:
        problems.append(
            f"LOOM_DATABASE_URL's database role {name} is a superuser: services run as the "
            "runtime role (loom_app), which cannot switch off the ledger's guards"
        )
    tables_owned = [str(t) for t in owned] if isinstance(owned, list) else []
    if tables_owned:
        names = ", ".join(tables_owned[:5]) + (", ..." if len(tables_owned) > 5 else "")
        tables = f"{len(tables_owned)} table{'' if len(tables_owned) == 1 else 's'}"
        problems.append(
            f"LOOM_DATABASE_URL's database role {user} owns {tables} in schema public "
            f"({names}): services run as the runtime role, not the migration owner"
        )
    return problems


def database_role_problems(database_url: str, *, timeout_s: int = 5) -> list[str]:
    """Outside dev, at start, over the service's own connection: its role must not be a
    superuser and must own no table in the schema (directly or through a role it is a
    member of). Either could disable, drop or replace the ledger's append-only and
    zero-sum triggers (contracts §3); services run as the runtime role (`loom_app`).
    Raises `DatabaseUnreachable` when Postgres cannot be connected to because of an
    outage (any connection error but a misconfiguration); a misconfiguration (failed
    authentication, missing role or database, pg_hba rejection) is a problem like the others
    (`connection_outage`)."""
    import psycopg

    from loom_platform.db import libpq_url

    try:
        url = libpq_url(database_url)
    except ValueError as e:
        return [f"cannot check LOOM_DATABASE_URL's database role: {error_summary(e)}"]
    try:
        conn = psycopg.connect(url, connect_timeout=timeout_s, autocommit=True)
    except psycopg.OperationalError as e:
        if connection_outage(e):
            raise DatabaseUnreachable(error_summary(e)) from e
        return [connection_problem(e)]
    except psycopg.Error as e:
        return [f"cannot check LOOM_DATABASE_URL's database role: {error_summary(e)}"]
    try:
        with conn:
            row = conn.execute(ROLE_CHECK_SQL).fetchone()
    except psycopg.Error as e:
        return [f"cannot check LOOM_DATABASE_URL's database role: {error_summary(e)}"]
    return role_problems(row)


def print_refusal(service: str, env: str, problems: list[str], *, started: bool = False) -> None:
    """Print every problem (to stderr) in the one exit-2 message."""
    lines = "\n".join(f"  - {p}" for p in problems)
    what = "stopping" if started else "refusing to start"
    print(
        f"{service}: {what} with LOOM_ENV={env}:\n{lines}\n"
        "Set LOOM_ENV=dev only for local development.",
        file=sys.stderr,
        flush=True,
    )


def refuse(service: str, env: str, problems: list[str]) -> NoReturn:
    """Print every problem and exit with status 2."""
    print_refusal(service, env, problems)
    raise SystemExit(2)


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
    connects to Postgres.

    When Postgres cannot be reached (an outage, which is any connection error but a
    misconfiguration: `connection_outage`), that alone does not refuse the start: one
    warning line, and the check is left to `loom_platform.db.role_gate`, which runs it
    on the first successful connection and holds back all of the process's Postgres
    work (its pools, its listeners) until it has passed. A misconfiguration (failed
    authentication, missing role or database, pg_hba rejection) refuses the start like any
    other problem."""
    problems = list(settings.production_problems() if problems is None else problems)
    if settings.env != "dev" and check_database_role:
        try:
            problems += database_role_problems(settings.database_url)
        except DatabaseUnreachable as e:
            then = (
                "the database role is not checked"
                if problems
                else "starting anyway, the database role is checked on the first connection"
            )
            print(
                f"warning: {service}: Postgres unreachable at start ({e}); {then}",
                file=sys.stderr,
                flush=True,
            )
            if not problems:
                from loom_platform.db import role_gate

                role_gate.defer(settings.database_url, service, settings.env)
    if problems:
        refuse(service, settings.env, problems)


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
