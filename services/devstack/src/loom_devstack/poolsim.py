"""Pool simulator: on-demand model pools in Redis, for local dev (contracts §10).

Writes `loom:pool:<model>` (`loom_platform.pool.PoolState`) as the M2 pool manager will,
so the gateway's routing and its "model temporarily unavailable" path run for real:

    ready --(idle for --idle-stop-s, or `set cold`)--> cold
    cold --(gateway: request_wake)--> warming (retry_after_s counts down) --> ready
    any --(`set unavailable`)--> unavailable (wakes ignored: no capacity)

Every servable model is seeded `ready` with its mock pod replica at start; a servable
model without a configured replica is seeded `unavailable`. Wakes arrive on
`PUBLISH loom:pool:wake`; the simulator also polls the wake keys every tick, so a wake
published before it subscribed is not lost. Idle detection reads the usage stream
(`loom:usage`) without a consumer group, so it never interferes with billing.

    loom-poolsim run                         # the compose service
    loom-poolsim set qwen3-8b cold           # next request: 503, wake, ready in ~20 s
    loom-poolsim set qwen3-8b unavailable    # 503 "no GPU capacity", Retry-After 300
    loom-poolsim set qwen3-8b ready
    loom-poolsim status
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import os
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from redis.asyncio import Redis

from loom_platform.pool import (
    WAKE_CHANNEL,
    PoolState,
    PoolStateName,
    Replica,
    get_pool,
    set_pool,
    wake_key,
)

log = logging.getLogger("loom_poolsim")

DEFAULT_COLD_START_S = 20
DEFAULT_UNAVAILABLE_RETRY_S = 300


@dataclass
class SimConfig:
    replicas: dict[str, tuple[Replica, ...]]
    cold_start_s: int = DEFAULT_COLD_START_S
    unavailable_retry_s: int = DEFAULT_UNAVAILABLE_RETRY_S
    idle_stop_s: int = 0  # 0: never stop an idle model
    tick_s: float = 1.0
    usage_stream: str = "loom:usage"


@dataclass
class PoolSim:
    redis: Redis
    models: Sequence[str]
    config: SimConfig
    clock: Callable[[], float] = time.monotonic
    # model -> monotonic time it becomes ready
    _warming: dict[str, float] = field(default_factory=dict)
    _last_used: dict[str, float] = field(default_factory=dict)
    _usage_cursor: str | None = None

    async def seed(self) -> None:
        for model in self.models:
            if self.config.replicas.get(model):
                await self._set(model, "ready")
            else:
                log.warning("%s: no replica configured; seeding unavailable", model)
                await self._set(model, "unavailable")
        self._warming.clear()

    async def wake(self, model: str) -> bool:
        """Start a cold model. True if it moved to warming."""
        if model not in self.models:
            return False
        pool = await get_pool(self.redis, model)
        if pool.state != "cold":
            return False
        if not self.config.replicas.get(model):
            log.warning("%s: wake requested but no replica configured", model)
            return False
        self._warming[model] = self.clock() + self.config.cold_start_s
        await self._set(model, "warming", retry_after_s=self.config.cold_start_s)
        log.info("%s: cold -> warming (%d s)", model, self.config.cold_start_s)
        return True

    async def tick(self) -> None:
        """One reconcile pass: pending wakes, warm-up countdowns, idle stops."""
        now = self.clock()
        await self._read_usage(now)
        for model in self.models:
            pool = await get_pool(self.redis, model)
            if pool.state == "cold":
                if await self.redis.exists(wake_key(model)):
                    await self.wake(model)
            elif pool.state == "warming":
                ready_at = self._warming.get(model)
                if ready_at is None:  # adopted after a restart, or set by hand
                    ready_at = now + pool.retry_after_s
                    self._warming[model] = ready_at
                remaining = ready_at - now
                if remaining <= 0:
                    del self._warming[model]
                    await self._set(model, "ready")
                    await self.redis.delete(wake_key(model))
                    self._last_used[model] = now
                    log.info("%s: warming -> ready", model)
                elif math.ceil(remaining) != pool.retry_after_s:
                    await self._set(model, "warming", retry_after_s=math.ceil(remaining))
            elif pool.state == "ready" and self.config.idle_stop_s > 0:
                last = self._last_used.setdefault(model, now)
                if now - last >= self.config.idle_stop_s:
                    await self.set_state(model, "cold")
                    log.info("%s: idle %d s, ready -> cold", model, self.config.idle_stop_s)

    async def set_state(self, model: str, state: PoolStateName, retry_after_s: int = 0) -> None:
        """Administrative override (the `set` command and tests)."""
        self._warming.pop(model, None)
        if state in ("cold", "ready"):
            await self.redis.delete(wake_key(model))
        if state == "ready":
            self._last_used[model] = self.clock()
        await self._set(model, state, retry_after_s=retry_after_s)

    async def _set(self, model: str, state: PoolStateName, retry_after_s: int = 0) -> None:
        if not retry_after_s:
            retry_after_s = {
                "ready": 0,
                "warming": self.config.cold_start_s,
                "cold": self.config.cold_start_s,
                "unavailable": self.config.unavailable_retry_s,
            }[state]
        replicas = self.config.replicas.get(model, ()) if state == "ready" else ()
        if state == "ready" and not replicas:
            raise ValueError(f"{model}: cannot be ready without a replica")
        await set_pool(self.redis, PoolState(model, state, retry_after_s, replicas))

    async def _read_usage(self, now: float) -> None:
        if self.config.idle_stop_s <= 0:
            return
        if self._usage_cursor is None:
            last = await self.redis.xrevrange(self.config.usage_stream, count=1)
            self._usage_cursor = _s(last[0][0]) if last else "0-0"
        while True:
            batch = await self.redis.xread({self.config.usage_stream: self._usage_cursor}, 1000)
            if not batch:
                return
            entries = batch[0][1]
            for entry_id, fields in entries:
                self._usage_cursor = _s(entry_id)
                model = _s(fields.get(b"model_id") or fields.get("model_id") or "")
                if model:
                    self._last_used[model] = now
            if len(entries) < 1000:
                return

    async def run(self, stop: asyncio.Event) -> None:
        await self.seed()
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(WAKE_CHANNEL)
        log.info(
            "pools: %s (cold start %d s, idle stop %s)",
            ", ".join(self.models),
            self.config.cold_start_s,
            f"{self.config.idle_stop_s} s" if self.config.idle_stop_s else "off",
        )
        try:
            while not stop.is_set():
                deadline = self.clock() + self.config.tick_s
                while (left := deadline - self.clock()) > 0 and not stop.is_set():
                    msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=left)
                    if msg is not None and msg.get("type") == "message":
                        await self.wake(_s(msg["data"]))
                await self.tick()
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(WAKE_CHANNEL)
                await pubsub.aclose()


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


# ---------------------------------------------------------------- configuration


def parse_replicas(specs: Sequence[str], token_dir: Path) -> dict[str, tuple[Replica, ...]]:
    """`MODEL=POD@URL` (repeatable or comma-separated). The pod id is the replica id and
    the certificate name; its token is read from `<token_dir>/<pod>.token`."""
    out: dict[str, list[Replica]] = {}
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            model, sep, rest = part.partition("=")
            pod, sep2, url = rest.partition("@")
            if not (sep and sep2 and model and pod and url):
                raise ValueError(f"replica spec must be MODEL=POD@URL, got {part!r}")
            token = (token_dir / f"{pod}.token").read_text().strip()
            out.setdefault(model, []).append(Replica(pod, url, pod, token))
    return {m: tuple(r) for m, r in out.items()}


def servable_models(models_yaml: str | None = None) -> list[str]:
    from loom_bench.registry import load_registry
    from loom_platform.pricing import servable

    return [m.id for m in servable(load_registry(models_yaml))]


def _parser() -> argparse.ArgumentParser:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="loom-poolsim", description=__doc__.splitlines()[0])
    p.add_argument("--redis-url", default=env("LOOM_REDIS_URL", "redis://localhost:6379/0"))
    p.add_argument(
        "--replica",
        action="append",
        default=[v for v in [env("LOOM_POOLSIM_REPLICAS")] if v],
        metavar="MODEL=POD@URL",
        help="a model's mock pod (repeatable or comma-separated)",
    )
    p.add_argument(
        "--token-dir",
        type=Path,
        default=Path(env("LOOM_POOLSIM_TOKEN_DIR", ".dev/certs")),
        help="directory with <pod>.token files (loom-devcerts output)",
    )
    p.add_argument(
        "--cold-start-s",
        type=int,
        default=int(env("LOOM_POOLSIM_COLD_START_S", str(DEFAULT_COLD_START_S))),
    )
    p.add_argument(
        "--idle-stop-s",
        type=int,
        default=int(env("LOOM_POOLSIM_IDLE_STOP_S", "0")),
        help="stop (cold) a ready model after this long without usage; 0 = never",
    )
    p.add_argument(
        "--unavailable-retry-s",
        type=int,
        default=int(env("LOOM_POOLSIM_UNAVAILABLE_RETRY_S", str(DEFAULT_UNAVAILABLE_RETRY_S))),
    )
    p.add_argument("--models-yaml", default=env("LOOM_MODELS_YAML"))
    p.add_argument("--log-level", default=env("LOOM_POOLSIM_LOG_LEVEL", "info"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="seed the pools, then simulate (the compose service)")
    s = sub.add_parser("set", help="set one model's pool state")
    s.add_argument("model")
    s.add_argument("state", choices=["ready", "warming", "cold", "unavailable"])
    s.add_argument("--retry-after-s", type=int, default=0)
    sub.add_parser("status", help="print every servable model's pool state")
    return p


async def _main(args: argparse.Namespace) -> None:
    models = servable_models(args.models_yaml)
    needs_replicas = args.command == "run" or (args.command == "set" and args.state == "ready")
    replicas = parse_replicas(args.replica, args.token_dir) if needs_replicas else {}
    config = SimConfig(
        replicas=replicas,
        cold_start_s=args.cold_start_s,
        unavailable_retry_s=args.unavailable_retry_s,
        idle_stop_s=args.idle_stop_s,
    )
    redis = Redis.from_url(args.redis_url)
    try:
        sim = PoolSim(redis, models, config)
        if args.command == "run":
            unknown = set(replicas) - set(models)
            if unknown:
                log.warning("replicas for models that are not servable: %s", sorted(unknown))
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                with contextlib.suppress(NotImplementedError):
                    loop.add_signal_handler(sig, stop.set)
            await sim.run(stop)
        elif args.command == "set":
            if args.model not in models:
                raise SystemExit(f"{args.model} is not a servable model ({', '.join(models)})")
            await sim.set_state(args.model, args.state, args.retry_after_s)
            print(_describe(await get_pool(redis, args.model)))
        else:
            for model in models:
                print(_describe(await get_pool(redis, model)))
    finally:
        await redis.aclose()


def _describe(pool: PoolState) -> str:
    replicas = ", ".join(f"{r.id} {r.url}" for r in pool.replicas) or "-"
    retry = f" retry_after={pool.retry_after_s}s" if pool.state != "ready" else ""
    return f"{pool.model_id}: {pool.state}{retry} replicas: {replicas}"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
