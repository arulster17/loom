"""loom-poolmgr: the pool manager service and its admin commands.

The admin commands act on the running manager's Postgres and Redis
(docs/phase1-m2-spec.md §11).

    loom-poolmgr run                   # the service
    loom-poolmgr status                # pools, pods, month-to-date spend against the cap
    loom-poolmgr warm <model>          # request_wake (pre-warm)
    loom-poolmgr drain <pod id>        # drain and terminate one pod
    loom-poolmgr stop <model>          # drain and terminate a model's pods (stays cold)
    loom-poolmgr reset <model>         # clear crash_loop / spend_cap / price_too_high holds
    loom-poolmgr issue-ca --out DIR    # dev only: CA plus manager and gateway client certs

`drain`, `stop` and `reset` queue a command the running manager picks up on its next
tick (`loom_poolmgr.manager.push_admin`).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from redis.asyncio import Redis

from loom_platform.pool import request_wake
from loom_poolmgr.manager import push_admin
from loom_poolmgr.pki import write_dev_ca
from loom_poolmgr.settings import PoolmgrSettings


def build_parser() -> argparse.ArgumentParser:
    doc = __doc__ or ""
    p = argparse.ArgumentParser(prog="loom-poolmgr", description=doc.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sub.add_parser("run", help="run the pool manager (single writer: one per database)")
    sub.add_parser("status", help="per model: pool state, reason, pods; spend against the cap")
    warm = sub.add_parser("warm", help="ask the manager to start a model (request_wake)")
    warm.add_argument("model")
    drain = sub.add_parser("drain", help="drain and terminate one pod")
    drain.add_argument("pod_id", metavar="POD_ID")
    stop = sub.add_parser("stop", help="drain and terminate all of a model's pods")
    stop.add_argument("model")
    reset = sub.add_parser("reset", help="clear a model's crash_loop / spend_cap hold")
    reset.add_argument("model")
    ca = sub.add_parser(
        "issue-ca", help="dev only: create a CA plus manager and gateway client certificates"
    )
    ca.add_argument("--out", type=Path, required=True, help="directory to write into")
    return p


async def _run(settings: PoolmgrSettings) -> int:
    raise NotImplementedError("phase B")


async def _status(settings: PoolmgrSettings) -> int:
    raise NotImplementedError("phase B")


async def _admin(settings: PoolmgrSettings, args: argparse.Namespace) -> int:
    redis = Redis.from_url(settings.redis_url)
    try:
        if args.command == "warm":
            made = await request_wake(redis, args.model)
            print(f"{args.model}: wake {'requested' if made else 'already pending'}")
        elif args.command == "drain":
            await push_admin(redis, {"op": "drain", "pod_id": args.pod_id})
            print(f"{args.pod_id}: drain queued")
        else:
            await push_admin(redis, {"op": args.command, "model": args.model})
            print(f"{args.model}: {args.command} queued")
    finally:
        await redis.aclose()
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = PoolmgrSettings()
    if args.command == "issue-ca":
        if settings.env != "dev":
            print("loom-poolmgr issue-ca: dev only (LOOM_ENV=dev)", file=sys.stderr)
            raise SystemExit(2)
        written = write_dev_ca(args.out)
        print(f"{args.out}: {', '.join(written)} written")
        return
    if args.command == "run":
        code = asyncio.run(_run(settings))
    elif args.command == "status":
        code = asyncio.run(_status(settings))
    else:
        code = asyncio.run(_admin(settings, args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
