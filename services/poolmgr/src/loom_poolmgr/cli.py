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
    loom-poolmgr local-capacity off|on # dev only, provider local: creates fail for capacity

`drain`, `stop`, `reset` and `local-capacity` queue a command the running manager picks
up on its next tick (`loom_poolmgr.manager.push_admin`).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import hashlib
import logging
import signal
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import psycopg
from redis.asyncio import Redis

from loom_bench.prices import load_prices
from loom_bench.providers.runpod_api import RunpodApi
from loom_bench.providers.runpod_ssh import OpenSshExec, SshExec
from loom_bench.registry import REPO_ROOT, load_registry
from loom_platform.db import RoleCheckFailed, libpq_url, role_gate
from loom_platform.pool import get_pool, request_wake
from loom_platform.pricing import servable
from loom_platform.serving import LOCAL_PROVIDERS, load_serving
from loom_platform.settings import enforce_environment
from loom_podproxy.bundle import LOCK_DEFAULT, read_lock  # type: ignore[import-untyped]
from loom_poolmgr import spend, store
from loom_poolmgr.bootstrap import ProxyBundle, start_command
from loom_poolmgr.leases import LeaseRecords, MemoryLeaseRecords, S3LeaseRecords
from loom_poolmgr.local import LocalServing
from loom_poolmgr.manager import (
    CaExpiring,
    ManagerLocked,
    PoolManager,
    SshIdentity,
    build_deployments,
    push_admin,
)
from loom_poolmgr.metrics import PoolmgrMetrics
from loom_poolmgr.metrics import serve as serve_metrics
from loom_poolmgr.pki import Pki, write_dev_ca
from loom_poolmgr.provider import ServingProvider
from loom_poolmgr.runpod import USER_AGENT, RunpodServing
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
    cap = sub.add_parser(
        "local-capacity",
        help="dev only, provider local: off makes every create fail for capacity, on undoes it",
    )
    cap.add_argument("capacity", choices=("on", "off"))
    return p


def _bundle(settings: PoolmgrSettings, provider: str) -> ProxyBundle:
    """The pinned proxy bundle, checked against config/serving-artifacts.lock. Only the
    local and fake providers (which install nothing) may run without one: a placeholder.
    `provider: runpod` always needs the real one, whatever LOOM_ENV is (spec §11)."""
    if settings.poolmgr_proxy_bundle is None:
        if provider not in LOCAL_PROVIDERS:
            raise ValueError(f"provider {provider} needs LOOM_POOLMGR_PROXY_BUNDLE")
        data = b"loom-poolmgr dev: no proxy bundle"
        return ProxyBundle(data, hashlib.sha256(data).hexdigest())
    lock = read_lock(REPO_ROOT / LOCK_DEFAULT)
    return ProxyBundle.load(settings.poolmgr_proxy_bundle, str(lock["sha256"]))


async def _run(settings: PoolmgrSettings) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # httpx logs every health check and probe at INFO (one line per pod every few seconds).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_serving(
        settings.poolmgr_serving_yaml, registry=load_registry(settings.models_yaml)
    )
    if config.provider == "fake":
        print("loom-poolmgr: provider fake is for tests only", file=sys.stderr)
        return 2
    enforce_environment(settings, "loom-poolmgr", settings.production_problems(config))
    if config.provider not in LOCAL_PROVIDERS and not settings.poolmgr_lease_bucket:
        print(f"loom-poolmgr: provider {config.provider} needs a lease bucket", file=sys.stderr)
        return 2
    try:
        bundle = _bundle(settings, config.provider)
    except (OSError, ValueError) as e:
        print(f"loom-poolmgr: proxy bundle: {e}", file=sys.stderr)
        return 2
    if settings.allow_cidrs and config.provider != "local":
        print("loom-poolmgr: LOOM_POOLMGR_ALLOW_CIDRS is for provider local", file=sys.stderr)
        return 2
    if settings.poolmgr_idle_stop_s is not None:
        idle = settings.poolmgr_idle_stop_s
        models = {m: s.model_copy(update={"idle_stop_s": idle}) for m, s in config.models.items()}
        config = config.model_copy(update={"models": models})
        logging.getLogger("loom_poolmgr").warning(
            "LOOM_POOLMGR_IDLE_STOP_S: every model's idle_stop_s is %d s (dev)", idle
        )
    registry = load_registry(settings.models_yaml)
    deployments = build_deployments(config, registry, load_prices())
    state = settings.poolmgr_state_dir or Path(tempfile.mkdtemp(prefix="loom-poolmgr-"))
    state.mkdir(parents=True, exist_ok=True)
    if settings.poolmgr_ca_cert and settings.poolmgr_ca_key:
        if not (settings.poolmgr_client_cert and settings.poolmgr_client_key):
            print("loom-poolmgr: the CA needs LOOM_POOLMGR_CLIENT_CERT/KEY", file=sys.stderr)
            return 2
        pki = Pki.load(
            settings.poolmgr_ca_cert,
            settings.poolmgr_ca_key,
            settings.poolmgr_client_cert,
            settings.poolmgr_client_key,
        )
    else:
        pki = Pki.generate(state / "pki")  # dev only (production_problems requires the CA)
    if settings.poolmgr_ssh_key is not None:
        ssh = SshIdentity.load(settings.poolmgr_ssh_key, state / "known_hosts")
        ssh.known_hosts_dir.mkdir(mode=0o700, exist_ok=True)
    else:
        ssh = SshIdentity.generate(state / "ssh")
    leases: LeaseRecords = (
        S3LeaseRecords(settings.poolmgr_lease_bucket)
        if settings.poolmgr_lease_bucket
        else MemoryLeaseRecords()
    )
    local: LocalServing | None = None
    provider: ServingProvider
    pod_exec: SshExec
    if config.provider == "runpod":
        if settings.runpod_api_key is None:
            print("loom-poolmgr: RUNPOD_API_KEY is not set", file=sys.stderr)
            return 2
        provider = RunpodServing(
            RunpodApi(settings.runpod_api_key, user_agent=USER_AGENT),
            placement=config.placement,
            network=config.network,
            start_command=start_command(),
        )
        pod_exec = OpenSshExec()
    else:
        host = settings.poolmgr_local_host
        local = LocalServing(
            cold_start_s=settings.poolmgr_local_cold_start_s,
            root=state / "pods",
            host=host,
            bind="127.0.0.1" if host in ("127.0.0.1", "localhost") else "0.0.0.0",
        )
        provider, pod_exec = local, local.exec
    metrics = PoolmgrMetrics()
    redis = Redis.from_url(settings.redis_url)
    manager = PoolManager(
        config,
        deployments=deployments,
        provider=provider,
        pod_exec=pod_exec,
        redis=redis,
        database_url=settings.database_url,
        leases=leases,
        pki=pki,
        ssh=ssh,
        bundle=bundle,
        servable=tuple(m.id for m in servable(registry)),
        metrics=metrics,
        extra_allow=settings.allow_cidrs,
    )
    try:
        # Postgres unreachable at start (outside dev): nothing touches it until the
        # role check has passed on the first connection. If it fails (printed), stop
        # through the normal shutdown below and exit 2.
        try:
            await role_gate.wait()
        except RoleCheckFailed:
            return 2
        try:
            await manager.start()
        except ManagerLocked as e:
            print(f"loom-poolmgr: {e}", file=sys.stderr)
            return 1
        except CaExpiring as e:
            print(f"loom-poolmgr: {e}", file=sys.stderr)
            return 2
        # Served only once the lock is held and adoption is done: the image's
        # HEALTHCHECK reads it, so `docker compose up --wait` means a running manager.
        if settings.poolmgr_metrics_port:
            serve_metrics(metrics, settings.poolmgr_metrics_port)
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop.set)
        logging.getLogger("loom_poolmgr").info(
            "loom-poolmgr running: provider %s, models %s",
            config.provider,
            ", ".join(config.models),
        )
        try:
            await manager.run(stop)
        except ManagerLocked as e:
            print(f"loom-poolmgr: {e}", file=sys.stderr)
            return 1
    finally:
        await manager.stop()
        if local is not None:
            await local.aclose()
        await redis.aclose()
    return 0


def _age(now: dt.datetime, then: dt.datetime | None) -> str:
    if then is None:
        return "-"
    s = int((now - then).total_seconds())
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


async def _status(settings: PoolmgrSettings) -> int:
    config = load_serving(settings.poolmgr_serving_yaml, check_registry=False)
    now = dt.datetime.now(dt.UTC)
    redis = Redis.from_url(settings.redis_url)
    try:
        async with await psycopg.AsyncConnection.connect(
            libpq_url(settings.database_url), autocommit=True
        ) as conn:
            rows = await store.live_rows(conn)
            month = await spend.month_to_date_micros(conn, now)
        for model in config.models:
            pool = await get_pool(redis, model)
            reason = f" ({pool.reason})" if pool.reason else ""
            print(
                f"{model}: {pool.state}{reason}, retry after {pool.retry_after_s} s, "
                f"{len(pool.replicas)} replica(s)"
            )
            for row in (r for r in rows if r.model_id == model):
                price = (
                    "-"
                    if row.cost_per_hr_micros is None
                    else f"${row.cost_per_hr_micros / 1e6:.2f}/h"
                )
                lease = "-" if row.lease_until is None else row.lease_until.isoformat()
                print(
                    f"  {row.pod_id or row.pod_name}  {row.state}  age {_age(now, row.created_at)}"
                    f"  {price}  lease until {lease}"
                )
        cap = spend.usd_micros(config.spend.monthly_cap_usd)
        print(
            f"spend this month: ${month / 1e6:.2f} of the ${cap / 1e6:.2f} cap "
            f"({100 * month / cap:.0f}%)"
        )
    finally:
        await redis.aclose()
    return 0


async def _admin(settings: PoolmgrSettings, args: argparse.Namespace) -> int:
    redis = Redis.from_url(settings.redis_url)
    try:
        if args.command == "warm":
            made = await request_wake(redis, args.model)
            print(f"{args.model}: wake {'requested' if made else 'already pending'}")
        elif args.command == "drain":
            await push_admin(redis, {"op": "drain", "pod_id": args.pod_id})
            print(f"{args.pod_id}: drain queued")
        elif args.command == "local-capacity":
            await push_admin(redis, {"op": "local_capacity", "capacity": args.capacity})
            print(f"local capacity {args.capacity} queued")
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
    if args.command == "local-capacity":
        provider = load_serving(settings.poolmgr_serving_yaml, check_registry=False).provider
        if settings.env != "dev" or provider != "local":
            print(
                "loom-poolmgr local-capacity: dev only (LOOM_ENV=dev), provider local only "
                f"(this serving file: {provider})",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if args.command == "run":
        code = asyncio.run(_run(settings))
    elif args.command == "status":
        code = asyncio.run(_status(settings))
    else:
        code = asyncio.run(_admin(settings, args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
