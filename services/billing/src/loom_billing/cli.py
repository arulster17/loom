"""Entrypoints: `loom-billing <command>` or `python -m loom_billing <command>`.

| Command | What | Ports |
|---|---|---|
| `all` (default) | worker + webhook/internal API in one process | 8090, metrics 9101 |
| `worker` | consumer, batches, reconcile, partitions, credit listener | metrics 9101 |
| `api` | webhook + internal API only | 8090, metrics 9101 |
| `stripe-stub` | local Stripe stand-in (never with real keys) | 12111 |
| `batch-once` / `reconcile-once` / `partitions-once` | one run of that job (ops) | |
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys

import uvicorn
from prometheus_client import start_http_server

from loom_billing.api import create_app
from loom_billing.logs import configure_logging
from loom_billing.settings import BillingSettings
from loom_billing.stripe_stub import create_stub_app
from loom_billing.worker import Worker, redis_from_settings, run_worker
from loom_platform.db import open_pool

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
    pool = await open_pool(settings.database_url, max_size=10)
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
    pool = await open_pool(settings.database_url, max_size=2)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loom-billing", description=__doc__)
    parser.add_argument("command", nargs="?", default="all", choices=COMMANDS)
    args = parser.parse_args(argv)
    configure_logging(os.environ.get("LOOM_LOG_LEVEL", "INFO"))
    settings = BillingSettings()

    if args.command == "stripe-stub":
        if settings.stripe_mode == "stripe":
            print("real Stripe keys are set; the stub is for local dev only", file=sys.stderr)
            return 2
        app = create_stub_app(
            public_url=settings.stripe_stub_public_url,
            webhook_url=settings.stripe_stub_webhook_url,
            webhook_secret=settings.webhook_secret(),
        )
        asyncio.run(_server(app, settings.stripe_stub_host, settings.stripe_stub_port).serve())
        return 0

    if args.command in ("all", "worker", "api") and settings.billing_metrics_port:
        start_http_server(settings.billing_metrics_port)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
