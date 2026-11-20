"""Entrypoints: `loom-billing <command>` or `python -m loom_billing <command>`.

| Command | What | Ports |
|---|---|---|
| `all` (default) | worker + webhook/internal API in one process | 8090, metrics 9101 |
| `worker` | consumer, batches, reconcile, partitions, credit listener | metrics 9101 |
| `api` | webhook + internal API only | 8090, metrics 9101 |
| `stripe-stub` | local Stripe stand-in (never with real keys) | 12111 |
| `batch-once` / `reconcile-once` / `partitions-once` | one run of that job (ops) | |
| `stripe-event list|resolve` | stuck Stripe events (ops, spec §8.6) | |
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys

import psycopg
import uvicorn
from prometheus_client import start_http_server
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from loom_billing.api import create_app
from loom_billing.logs import configure_logging
from loom_billing.resolve import ResolveRefused, list_events, resolve_event
from loom_billing.settings import BillingSettings, billing_conninfo
from loom_billing.stripe_stub import create_stub_app
from loom_billing.worker import Worker, redis_from_settings, run_worker
from loom_platform.db import RoleCheckFailed, open_pool, role_gate
from loom_platform.settings import enforce_environment, redis_policy_problems

log = logging.getLogger("loom_billing")

BUSY_EXIT = 75  # `stripe-event`: lock/statement/pool timeout or no database; retry (EX_TEMPFAIL)

COMMANDS = (
    "all",
    "worker",
    "api",
    "stripe-stub",
    "batch-once",
    "reconcile-once",
    "partitions-once",
)


def _server(app: object, host: str, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False)  # type: ignore[arg-type]
    return uvicorn.Server(config)


async def _all(settings: BillingSettings) -> None:
    pool = await open_pool(billing_conninfo(settings.database_url), max_size=10)
    redis = redis_from_settings(settings)
    try:
        app = create_app(settings, pool=pool, redis=redis)
        server = _server(app, settings.billing_host, settings.billing_port)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(Worker(settings, pool, redis).run())
            tg.create_task(server.serve())
    finally:
        await pool.close()
        await redis.aclose()


async def _once(settings: BillingSettings, job: str) -> None:
    pool = await open_pool(billing_conninfo(settings.database_url), max_size=2)
    redis = redis_from_settings(settings)
    try:
        w = Worker(settings, pool, redis)
        await {
            "batch-once": w.batch,
            "reconcile-once": w.reconcile,
            "partitions-once": w.partitions,
        }[job]()
    finally:
        await pool.close()
        await redis.aclose()


def _stripe_event_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-billing stripe-event",
        description="Stuck Stripe events (docs/phase1-m3-spec.md §8.6): list them, or resolve one "
        "by hand (audited; posts nothing).",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    ls = sub.add_parser("list", help="stored events, newest first")
    ls.add_argument("--unapplied", action="store_true", help="only events not applied")
    rs = sub.add_parser("resolve", help="close a stuck event (posts nothing)")
    rs.add_argument("event_id")
    rs.add_argument("--admin-email", required=True, help="an existing user with role admin")
    rs.add_argument("--reason", required=True, help="written to the audit log")
    return parser


async def _list_events(pool: AsyncConnectionPool, args: argparse.Namespace) -> int:
    for e in await list_events(pool, unapplied=args.unapplied):
        stale = f" stale {e['stale_reported_at']:%Y-%m-%d %H:%M}" if e["stale_reported_at"] else ""
        err = f" errors={e['error_count']} {e['last_error']}" if e["last_error"] else ""
        when = f"{e['received_at']:%Y-%m-%d %H:%M:%S}"
        print(f"{e['id']}  {e['type']}  {when}  {e['result']}{stale}{err}")
    return 0


async def _stripe_event(settings: BillingSettings, args: argparse.Namespace) -> int:
    resolving = args.action == "resolve"
    what = f"resolve {args.event_id}" if resolving else "list events"
    pool: AsyncConnectionPool | None = None
    try:
        try:
            # The pool wait is bounded like the webhook's (spec §6.13): an unreachable
            # database fails fast instead of after psycopg's 30 s.
            pool = await asyncio.wait_for(
                open_pool(billing_conninfo(settings.database_url), max_size=2),
                timeout=settings.stripe_webhook_db_timeout_s,
            )
            if not resolving:
                return await _list_events(pool, args)
            await resolve_event(
                pool,
                args.event_id,
                admin_email=args.admin_email,
                reason=args.reason,
                settings=settings,
            )
        except ResolveRefused as e:
            print(f"refused: {e.reason}", file=sys.stderr)
            return 2
        except (TimeoutError, PoolTimeout) as e:
            # No connection in time (nothing was sent), or the pool wait timed out.
            print(
                f"busy, retry: could not {what} ({type(e).__name__}): the database did not "
                f"answer in time; nothing changed",
                file=sys.stderr,
            )
            return BUSY_EXIT
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as e:
            # The row lock or the statement timed out: the transaction rolled back.
            print(
                f"busy, retry: could not {what} ({type(e).__name__}); nothing changed",
                file=sys.stderr,
            )
            return BUSY_EXIT
        except psycopg.OperationalError as e:
            # The connection broke: a COMMIT may or may not have reached the server.
            print(
                f"database connection failed during {what} ({type(e).__name__}); "
                + (
                    f"{args.event_id} may or may not have been resolved: check with "
                    "`stripe-event list`, then retry if it is still unresolved"
                    if resolving
                    else "retry"
                ),
                file=sys.stderr,
            )
            return BUSY_EXIT
        except psycopg.Error as e:
            print(f"database error during {what} ({type(e).__name__}): {e}", file=sys.stderr)
            return 1
        print(f"resolved {args.event_id}: nothing was posted")
        return 0
    finally:
        if pool is not None:
            await pool.close()


def _stripe_event_main(argv: list[str]) -> int:
    args = _stripe_event_parser().parse_args(argv)
    configure_logging(os.environ.get("LOOM_LOG_LEVEL", "INFO"))
    settings = BillingSettings()
    enforce_environment(
        settings, "loom-billing stripe-event", settings.production_problems(serves_api=False)
    )
    try:
        code = asyncio.run(_stripe_event(settings, args))
    except RoleCheckFailed:
        return 2
    return 2 if role_gate.failed else code


def main(argv: list[str] | None = None) -> int:
    if (argv if argv is not None else sys.argv[1:])[:1] == ["stripe-event"]:
        return _stripe_event_main(list((argv if argv is not None else sys.argv[1:])[1:]))
    parser = argparse.ArgumentParser(prog="loom-billing", description=__doc__)
    parser.add_argument("command", nargs="?", default="all", choices=COMMANDS)
    args = parser.parse_args(argv)
    configure_logging(os.environ.get("LOOM_LOG_LEVEL", "INFO"))
    settings = BillingSettings()
    problems = settings.production_problems(serves_api=args.command in ("all", "api"))
    if settings.env != "dev" and args.command != "stripe-stub":
        problems += redis_policy_problems(settings.redis_url)
    # The stub never connects to Postgres: no database role to check.
    enforce_environment(
        settings,
        f"loom-billing {args.command}",
        problems,
        check_database_role=args.command != "stripe-stub",
    )

    if args.command == "stripe-stub":
        if settings.stripe_mode != "stub":
            print("the stub runs only with LOOM_STRIPE_MODE=stub (local dev)", file=sys.stderr)
            return 2
        app = create_stub_app(
            public_url=settings.stripe_stub_public_url,
            webhook_url=settings.stripe_stub_webhook_url,
            webhook_secret=settings.webhook_secret(),
        )
        asyncio.run(_server(app, settings.stripe_stub_host, settings.stripe_stub_port).serve())
        return 0

    if args.command in ("all", "api"):
        try:
            settings.stripe_api()
            settings.webhook_secret()
        except ValueError as e:
            print(f"loom-billing {args.command}: {e}", file=sys.stderr)
            return 2

    if args.command in ("all", "worker", "api") and settings.billing_metrics_port:
        start_http_server(settings.billing_metrics_port)
    try:
        with contextlib.suppress(KeyboardInterrupt):
            if args.command == "all":
                asyncio.run(_all(settings))
            elif args.command == "worker":
                asyncio.run(run_worker(settings))
            elif args.command == "api":
                app = create_app(settings)
                asyncio.run(_server(app, settings.billing_host, settings.billing_port).serve())
            else:
                asyncio.run(_once(settings, args.command))
    except BaseException as e:
        # A failed deferred role check (printed) exits 2 whatever ended the run:
        # `RoleCheckFailed` from `open_pool`, before any work (the unwinding closed what
        # was open); `api`: uvicorn's exit 3 for the lifespan's failed `open_pool`; or
        # any other error on the way out.
        if not role_gate.failed:
            raise
        if not isinstance(e, RoleCheckFailed | SystemExit):
            log.error("stopped with an error after the failed database role check", exc_info=e)
        return 2
    return 2 if role_gate.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
