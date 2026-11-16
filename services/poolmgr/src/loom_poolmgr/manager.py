"""The pool manager (docs/phase1-m2-spec.md §3, §5, §7): one process that creates,
bootstraps, health-checks, drains and terminates serving pods and writes
`loom:pool:<model>` as the gateway reads it.

Use:

    manager = PoolManager(config, deployments=..., provider=..., pod_exec=..., redis=...,
                          database_url=..., leases=..., pki=..., ssh=..., bundle=...)
    await manager.start()          # single-writer lock, then adoption (§5 table)
    await manager.reconcile_once() # one tick; `run()` calls it every reconcile_every_s
    await manager.wait_idle()      # tests: until the creates/bootstraps it started end
    await manager.stop()           # cancels its tasks, releases the lock; pods keep running

**Time.** Every timing rule reads `clock()` (epoch seconds) and every wait goes through
`sleep`, so tests drive the manager with a fake clock: advance it, call
`reconcile_once()`, then `wait_idle()`. A periodic check is due when `clock() - last >=
period`. A tick runs, in order: the admin commands; the pod listing
(`pod_state_every_s`); for each ready pod the proxy health and engine progress checks
(`proxy_health_every_s`), lease renewal (`lease_renew_every_s`) and certificate renewal;
each draining pod's drain step; retries of failed pods' terminations; the usage stream;
per model demand, creates and idle drains; spend alerts; the pool hashes. A create (and
the bootstrap after it) runs as a task the tick does not wait for, so bootstraps run in
parallel across models. A drain is advanced by the ticks (it needs no task): the pod
leaves the pool hash at once, and is terminated on the first tick at least
`2 x POOL_CACHE_TTL_S` later that sees `vllm:num_requests_running` = 0 (a failed
`/metrics` read is "try again next tick"), or `drain_max_s` after the drain started.

**Failures** (§11 "Failure and restart rules"). A pod is lost only after two consecutive
listings requested after its id was known leave it out, or a GET confirms it gone (or a
listing shows it EXITED/TERMINATED); a lost pod is terminated before its row is marked.
A terminate RunPod refuses leaves the row `failed`, counted against `max_replicas` and
spend, and is retried every tick. A row whose pod id is unknown (an ambiguous create, a
create task that crashed or was stopped mid-way) stays `failed` and counted until a later
listing shows no pod with its name, or the pod found by name is terminated. Any
unexpected exception in a create task fails the row; a `creating`/`bootstrapping` row
with no live task is failed by the next tick.

**Demand** (§5 step 1): a live wake key `loom:pool:<model>:wake`, or a ready model with a
usage event (`loom:usage`, read with `XREAD` from where the last tick stopped, no consumer
group) or an in-flight request within `idle_stop_s` (0: never idle). Idle time runs from
the last demand a tick observed. Desired replicas: `max(min_replicas, 1 if demand else
0)`, capped at `max_replicas`.

**Holds.** `crash_loop` (more than `max_replacements_per_hour` pods failed or lost in the
last hour), `spend_cap` and `price_too_high` keep a model `unavailable` and ignore its
wakes until an admin `reset`; `not_configured` (a servable model missing from
serving.yaml) is permanent. After a restart `crash_loop` and `price_too_high` are
recomputed from the last hour's rows, counting only failures after the model's last
`reset` (kept in Redis, `loom:poolmgr:reset:<model>`), by the same rule as live; the spend
guard trips again by itself. Adopted pods count as demand at adoption, and adoption takes
every pod out of the pool hashes before re-keying them.

**No capacity.** A create refused for capacity (or failing without making a pod) is
retried while demand lasts, `capacity.backoff_s[n]` after the n-th consecutive refusal
(the last step repeats); the pool says `unavailable` / `no_capacity` with that step as
`retry_after_s` while nothing is in flight.

**Admin commands** (`loom-poolmgr drain|stop|reset`, another process) are pushed as JSON
onto the Redis list `admin_queue` (default `ADMIN_QUEUE`; `{"op": "drain", "pod_id":
...}`, `{"op": "stop", "model": ...}`, `{"op": "reset", "model": ...}`, and the dev-only
`{"op": "local_capacity", "capacity": "on"|"off"}`, which only `provider: local` obeys:
`off` makes every create fail for capacity, the acceptance's switch); each tick pops
them all first. `warm` is `request_wake`, as the gateway's.

**Single writer.** `start()` takes `pg_try_advisory_lock(hashtext('loom-poolmgr'))` on a
connection it keeps for its lifetime and raises `ManagerLocked` if another manager holds
it. Adoption considers every `serving_pods` row not terminated, so tests sharing a
database clear the table first.

**Secrets.** Pod tokens live only in memory and in the pool hash; certificates and keys
only on SSH stdin. Nothing here logs a `Replica` or a token.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import math
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, TypeVar

import httpx
import psycopg
from psycopg_pool import AsyncConnectionPool
from redis.asyncio import Redis

from loom_bench.experiment import RUNPOD_GPU_TYPE_IDS
from loom_bench.prices import PriceBook
from loom_bench.providers.aws_ssm import stage_offsets
from loom_bench.providers.runpod_ssh import RemoteCommandError, SshExec, SshTarget
from loom_bench.records import Market
from loom_bench.registry import Registry
from loom_platform.db import libpq_url
from loom_platform.pool import WAKE_CHANNEL, PoolState, Replica, set_pool, wake_key
from loom_platform.serving import ServingConfig
from loom_poolmgr import bootstrap, health, spend, store
from loom_poolmgr.bootstrap import LOOPBACK_ALLOW, PodSecrets, ProxyBundle
from loom_poolmgr.leases import LeaseRecord, LeaseRecords, renew_on_pod
from loom_poolmgr.local import LocalServing
from loom_poolmgr.metrics import PoolmgrMetrics
from loom_poolmgr.pki import Pki
from loom_poolmgr.provider import (
    CreateFailed,
    Deployment,
    Endpoints,
    NoCapacity,
    PodInfo,
    ProviderError,
    ServingProvider,
    pod_name,
)
from loom_poolmgr.store import PodRow, TerminatedBy

LOCK_NAME = "loom-poolmgr"
ADMIN_QUEUE = "loom:poolmgr:admin"
USAGE_STREAM = "loom:usage"
# The gateway caches a pool hash for up to this long; a drain waits twice it after
# removing the pod from the hash (§5 "Idle stop and drain").
POOL_CACHE_TTL_S = 1.0
# RunPod's price region for bench/prices.yaml (Secure Cloud on demand).
RUNPOD_PRICE_REGION = "secure"
# Bootstrap: how often to look for the port mappings and sshd, and to poll serve_engine.
BOOT_POLL_S = 5.0
ENGINE_POLL_S = 5.0
# HTTP timeouts (real seconds) for the proxy calls.
HEALTH_TIMEOUT_S = 5.0
PROBE_TIMEOUT_S = 60.0
# Warming's Retry-After floor (§7).
WARMING_FLOOR_S = 15
CRASH_WINDOW_S = 3600.0
# serving_pods.failure values that are not a pod failing (they never count toward the
# crash loop).
NOT_REPLACEMENTS = (
    "no_capacity",
    "price_too_high",
    "create_failed",
    "spend_cap",
    "manager restarted",
    "stopped",
)
# Where `reset` records its time, so it survives a restart (spec §11).
RESET_KEY = "loom:poolmgr:reset:{model}"
RESET_TTL_S = 2 * int(CRASH_WINDOW_S)

log = logging.getLogger(__name__)

T = TypeVar("T")


class ManagerLocked(RuntimeError):
    """Another manager holds the single-writer lock."""


class PodLost(RuntimeError):
    """RunPod no longer lists the pod, or lists it EXITED/TERMINATED."""


class BootstrapTimeout(RuntimeError):
    """No first token within `bootstrap_timeout_s` of the create."""


@dataclass(frozen=True)
class SshIdentity:
    """The long-lived SSH key serving pods authorize (`LOOM_POOLMGR_SSH_KEY`)."""

    key_path: Path
    public_key: str  # the authorized_keys line given to pods as LOOM_SSH_PUBKEY
    known_hosts_dir: Path  # one known_hosts file per pod id

    @classmethod
    def load(cls, key_path: Path, known_hosts_dir: Path) -> SshIdentity:
        """`key_path` and its `<key_path>.pub`."""
        public = key_path.with_name(key_path.name + ".pub").read_text().strip()
        return cls(key_path, public, known_hosts_dir)

    @classmethod
    def generate(cls, directory: Path) -> SshIdentity:
        """A fresh ed25519 key in `directory` (dev and the local backend)."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519

        directory.mkdir(parents=True, exist_ok=True)
        key = ed25519.Ed25519PrivateKey.generate()
        key_path = directory / "id_ed25519"
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.OpenSSH,
                serialization.NoEncryption(),
            )
        )
        key_path.chmod(0o600)
        public = key.public_key().public_bytes(
            serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
        )
        line = public.decode() + " loom-poolmgr"
        key_path.with_name("id_ed25519.pub").write_text(line + "\n")
        known = directory / "known_hosts"
        known.mkdir(exist_ok=True)
        return cls(key_path, line, known)

    def known_hosts(self, pod_id: str) -> Path:
        return self.known_hosts_dir / f"{pod_id}.known_hosts"


def build_deployments(
    config: ServingConfig, registry: Registry, prices: PriceBook
) -> dict[str, Deployment]:
    """One `Deployment` per model in `config.models` (`config.check_registry` passed)."""
    out: dict[str, Deployment] = {}
    for model_id, served in config.models.items():
        spec = registry.get(model_id)
        instance_type = spec.hardware.instance_types.runpod
        listed = (
            None
            if instance_type is None
            else prices.instance_price(
                "runpod",
                RUNPOD_PRICE_REGION,
                instance_type,
                Market.ON_DEMAND,
                allow_unverified=True,
            ).per_hour
        )
        disk_gb = config.placement.container_disk_gb[model_id]
        try:
            disk = math.ceil(
                prices.storage_per_hour(
                    "runpod", RUNPOD_PRICE_REGION, disk_gb, allow_unverified=True
                )
            )
        except KeyError:
            disk = 0
        out[model_id] = Deployment(
            model_id=model_id,
            served=served,
            spec=spec,
            gpu_type_id=RUNPOD_GPU_TYPE_IDS[spec.hardware.gpu],
            gpu_count=spec.hardware.gpus_per_replica,
            container_disk_gb=disk_gb,
            listed_price_per_hr_micros=None if listed is None else math.ceil(listed),
            disk_per_hr_micros=disk,
        )
    return out


async def push_admin(redis: Redis, command: Mapping[str, str], *, queue: str = ADMIN_QUEUE) -> None:
    """Queue an admin command for the running manager (see the module docstring)."""
    op = command.get("op")
    if op not in ("drain", "stop", "reset", "local_capacity"):
        raise ValueError(f"unknown admin command {command!r}")
    if op == "local_capacity" and command.get("capacity") not in ("on", "off"):
        raise ValueError(f"local_capacity takes capacity on|off: {command!r}")
    await redis.rpush(queue, json.dumps(dict(command)))


def _dt(epoch: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(epoch, dt.UTC)


def _describe(e: BaseException) -> str:
    text = str(e).strip()
    out = f"{type(e).__name__}: {text}" if text else type(e).__name__
    return out[:500]


@dataclass(eq=False)
class _Pod:
    """A live `serving_pods` row and what the manager knows about the pod in memory."""

    row: PodRow
    task: asyncio.Task[None] | None = None  # the create/bootstrap, while it runs
    endpoints: Endpoints | None = None
    token: str | None = field(default=None, repr=False)  # never logged or stored
    routable: bool = False  # in the pool hash
    health_failures: int = 0
    out_since: float | None = None  # out of routing for failed health checks since
    last_health: float = -math.inf
    last_tokens: float | None = None
    last_progress: float = 0.0
    last_lease: float = -math.inf  # last renewal written on the pod and in its record
    lease_until: float | None = None
    cert_not_after: float | None = None
    lost: bool = False  # confirmed gone while its bootstrap runs
    drain_by: TerminatedBy = "manager"
    # The pod id as soon as the create answers (the row may not have it yet).
    known_id: str | None = None
    # Lost-pod rule (spec §11): only listings requested after `watch_seq` count, and a
    # pod is lost after two consecutive misses or a GET that confirms it gone.
    watch_seq: int = 0
    misses: int = 0
    # No pod id, and a pod may exist under the row's name (ambiguous create, a create
    # cancelled or crashed mid-way): the row stays `failed` and counted until a later
    # listing shows no pod with that name, or the one found is terminated.
    pending_name: bool = False
    # Known that no pod was ever made (no capacity, a clear refusal).
    no_pod: bool = False
    terminate_by: TerminatedBy = "manager"
    counted: bool = False  # its failure was counted toward the crash loop
    # The clock when the row was created and when its drain started, as floats: a
    # timestamptz round trip may move an instant by a microsecond, and every timing rule
    # compares at `>=`.
    created: float = field(default=math.nan)
    drain_started: float | None = None

    def __post_init__(self) -> None:
        if math.isnan(self.created):
            self.created = self.row.created_at.timestamp()
        if self.drain_started is None and self.row.drain_started_at is not None:
            self.drain_started = self.row.drain_started_at.timestamp()

    @property
    def model(self) -> str:
        return self.row.model_id

    @property
    def state(self) -> str:
        return self.row.state

    @property
    def pod_id(self) -> str | None:
        return self.row.pod_id or self.known_id

    @property
    def busy(self) -> bool:
        """Its create/bootstrap task owns it."""
        return self.task is not None and not self.task.done()

    def replica(self) -> Replica | None:
        if self.endpoints is None or self.pod_id is None or self.token is None:
            return None
        return Replica(
            id=self.pod_id,
            url=f"https://{self.endpoints.ip}:{self.endpoints.proxy_port}",
            server_name=self.pod_id,
            token=self.token,
        )


@dataclass
class _Capacity:
    """Consecutive create attempts that made no pod (no capacity or an error)."""

    failures: int
    last_attempt: float
    no_capacity: bool


class _Budget:
    """Month-to-date spend for one tick, read once and grown by each create it allows."""

    def __init__(self, manager: PoolManager, now: float) -> None:
        self.manager = manager
        self.now = now
        self.micros: int | None = None

    async def get(self) -> int:
        if self.micros is None:
            self.micros = await self.manager._db_call(spend.month_to_date_micros, _dt(self.now))
        return self.micros


class PoolManager:
    def __init__(
        self,
        config: ServingConfig,
        *,
        deployments: Mapping[str, Deployment],
        provider: ServingProvider,
        pod_exec: SshExec,
        redis: Redis,
        database_url: str,
        leases: LeaseRecords,
        pki: Pki,
        ssh: SshIdentity,
        bundle: ProxyBundle,
        servable: tuple[str, ...] = (),
        http: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        admin_queue: str = ADMIN_QUEUE,
        usage_stream: str = USAGE_STREAM,
        metrics: PoolmgrMetrics | None = None,
        extra_allow: tuple[str, ...] = (),
    ) -> None:
        """
        - `config`: the validated serving.yaml (`loom_platform.serving.load_serving`).
        - `deployments`: per configured model (`build_deployments`).
        - `provider`: where pods come from (`RunpodServing`, `LocalServing`).
        - `pod_exec`: how scripts reach a pod (`OpenSshExec`; the local backend's own;
          tests: `FakeServingExec`; the protocol is in `loom_poolmgr.bootstrap`).
        - `redis`: pool hashes, wake keys, the usage stream, the admin queue.
        - `database_url`: `serving_pods` and the lock; `start()` connects, `stop()` closes.
        - `leases`: lease records (`S3LeaseRecords`; tests: `MemoryLeaseRecords`).
        - `pki`: the CA for pod server certificates, and the manager's client certificate.
        - `ssh`: the key pods authorize.
        - `bundle`: the pinned proxy bundle serve_bootstrap installs.
        - `servable`: every servable model id in the registry; those missing from
          `config.models` are written `unavailable` / `not_configured`.
        - `http`: the client for pod proxies (default: one built on
          `pki.client_ssl_context()`, closed by `stop()`).
        - `clock`, `sleep`: injected time.
        - `admin_queue`: the Redis list admin commands arrive on (`push_admin(queue=)`).
        - `usage_stream`: the gateway's usage stream (`LOOM_USAGE_STREAM`).
        - `metrics`: Prometheus metrics (default: a registry of its own).
        - `extra_allow`: CIDRs added to every pod proxy's allowlist (dev only, the local
          backend's `LOOM_POOLMGR_ALLOW_CIDRS`).
        """
        self.config = config
        self.deployments = dict(deployments)
        self.provider = provider
        self.pod_exec = pod_exec
        self.redis = redis
        self.database_url = database_url
        self.leases = leases
        self.pki = pki
        self.ssh = ssh
        self.bundle = bundle
        self.servable = servable
        self.http = http
        self.clock = clock
        self.sleep = sleep
        self.admin_queue = admin_queue
        self.usage_stream = usage_stream
        self.metrics = metrics or PoolmgrMetrics()
        self.extra_allow = tuple(extra_allow)
        self._own_http = http is None
        self._pods: dict[uuid.UUID, _Pod] = {}
        self._holds: dict[str, str] = {}
        self._failures: dict[str, list[float]] = {}
        self._capacity: dict[str, _Capacity] = {}
        self._last_demand: dict[str, float] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._last_listing = -math.inf
        self._listing_seq = 0
        self._reset_at: dict[str, float] = {}
        self._usage_cursor: str | None = None
        self._db: AsyncConnectionPool | None = None
        self._lock_conn: psycopg.AsyncConnection[Any] | None = None
        self._in_tick = False
        self._stopped = False

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        """Take the single-writer lock (raise `ManagerLocked` if held), connect, then
        adopt (§5 "Restart and adoption"): re-key and probe each `loom-serve-*` pod with
        a live row (failure: `failed`, terminate), terminate orphans (`terminated_by =
        manager`), mark rows whose pod is gone `terminated` (`terminated_by = runpod`),
        and rewrite every pool hash from scratch. Returns once adoption has finished."""
        conn = await psycopg.AsyncConnection.connect(libpq_url(self.database_url), autocommit=True)
        try:
            cur = await conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_NAME,))
            got = await cur.fetchone()
        except BaseException:
            await conn.close()
            raise
        if got is None or not got[0]:
            await conn.close()
            raise ManagerLocked("another loom-poolmgr holds the single-writer lock")
        self._lock_conn = conn
        self._stopped = False
        try:
            self._db = AsyncConnectionPool(
                libpq_url(self.database_url),
                min_size=1,
                max_size=4,
                kwargs={"autocommit": True},
                open=False,
            )
            await self._db.open(wait=True)
            if self.http is None:
                self.http = httpx.AsyncClient(
                    verify=self.pki.client_ssl_context(), timeout=HEALTH_TIMEOUT_S, trust_env=False
                )
            last: Any = await self.redis.xrevrange(self.usage_stream, count=1)
            self._usage_cursor = _str(last[0][0]) if last else "0-0"
            self.metrics.spend_cap.set(spend.usd_micros(self.config.spend.monthly_cap_usd))
            await self._adopt_all()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        """Cancel the manager's tasks and release the lock (close its connections).
        Pods keep running: a restarted manager adopts them. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        tasks = [t for t in self._tasks if not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=10)
        if self._own_http and self.http is not None:
            with contextlib.suppress(Exception):
                await self.http.aclose()
            self.http = None
        if self._db is not None:
            with contextlib.suppress(Exception):
                await self._db.close()
            self._db = None
        if self._lock_conn is not None:
            with contextlib.suppress(Exception):
                await self._lock_conn.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_NAME,)
                )
            with contextlib.suppress(Exception):
                await self._lock_conn.close()
            self._lock_conn = None

    async def wait_idle(self) -> None:
        """Wait until every task this manager started (creates and bootstraps) has
        finished. For tests; `run()` never calls it."""
        while True:
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                return
            await asyncio.wait(pending)

    async def run(self, stop: asyncio.Event) -> None:
        """`reconcile_once()` every `reconcile_every_s` until `stop` is set (a wake
        published on `loom:pool:wake` starts a tick early). Call `start()` first and
        `stop()` after."""
        woken = asyncio.Event()
        listener = asyncio.ensure_future(self._wake_listener(woken))
        try:
            while not stop.is_set():
                await self._check_lock()
                started = self.clock()
                woken.clear()
                try:
                    await self.reconcile_once()
                except Exception:
                    log.exception("reconcile tick failed")
                delay = max(0.0, started + self.config.lifecycle.reconcile_every_s - self.clock())
                waits = [
                    asyncio.ensure_future(self.sleep(delay)),
                    asyncio.ensure_future(stop.wait()),
                    asyncio.ensure_future(woken.wait()),
                ]
                try:
                    await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for w in waits:
                        w.cancel()
        finally:
            listener.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await listener

    async def _check_lock(self) -> None:
        """The single-writer lock lives as long as its connection: once that is gone,
        another manager may hold it, so this one must stop."""
        conn = self._lock_conn
        try:
            if conn is None or conn.closed:
                raise psycopg.OperationalError("the lock connection is closed")
            await conn.execute("SELECT 1")
        except psycopg.Error as e:
            raise ManagerLocked(f"lost the single-writer lock ({e}); stopping") from None

    async def _wake_listener(self, woken: asyncio.Event) -> None:
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(WAKE_CHANNEL)
            try:
                async for msg in pubsub.listen():
                    if msg.get("type") == "message" and _str(msg.get("data") or "") in (
                        self.config.models
                    ):
                        woken.set()
            finally:
                await pubsub.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("wake listener stopped (%s); ticking on the schedule only", e)

    # ------------------------------------------------------------------ the tick

    async def reconcile_once(self) -> None:
        """One tick (see the module docstring). Raises only on a bug; provider, SSH and
        pod failures are handled and logged."""
        if self._in_tick:
            raise RuntimeError("reconcile_once() is not re-entrant")
        if self._db is None:
            raise RuntimeError("start() the manager first")
        self._in_tick = True
        t0 = time.perf_counter()
        try:
            now = self.clock()
            await self._admin_commands(now)
            if now - self._last_listing >= self.config.health.pod_state_every_s:
                self._listing_seq += 1
                seq = self._listing_seq
                try:
                    listing = await self.provider.list()
                except ProviderError as e:
                    log.warning("pod listing failed: %s", e)
                else:
                    self._last_listing = now
                    await self._observe(listing, seq, now)
            await self._check_pods(now)
            await self._read_usage(now)
            budget = _Budget(self, now)
            for model in self.config.models:
                await self._reconcile_model(model, now, budget)
            await self._spend_alerts(now, budget)
            await self._publish_all(now)
        finally:
            self._in_tick = False
            self.metrics.reconcile.observe(time.perf_counter() - t0)

    async def _admin_commands(self, now: float) -> None:
        while True:
            raw: Any = await self.redis.lpop(self.admin_queue)
            if raw is None:
                return
            try:
                cmd = json.loads(raw)
                op = cmd["op"]
            except (ValueError, KeyError, TypeError):
                log.error("ignoring a malformed admin command")
                continue
            if op == "reset":
                model = str(cmd.get("model"))
                hold = self._holds.pop(model, None)
                self._failures.pop(model, None)
                self._reset_at[model] = now
                try:
                    await self.redis.set(RESET_KEY.format(model=model), repr(now), ex=RESET_TTL_S)
                except Exception as e:
                    log.error("recording the reset of %s failed: %s", model, _describe(e))
                log.warning("admin reset of %s (hold was %s)", model, hold)
            elif op == "stop":
                model = str(cmd.get("model"))
                log.warning("admin stop of %s: draining its pods", model)
                await self.redis.delete(wake_key(model))
                self._last_demand.pop(model, None)
                for pod in self._model_pods(model):
                    await self._admin_remove(pod, now)
                await self._publish(model, now)
            elif op == "drain":
                pod_id = str(cmd.get("pod_id"))
                found = [p for p in self._pods.values() if p.pod_id == pod_id]
                if not found:
                    log.warning("admin drain of %s: no such live pod", pod_id)
                for pod in found:
                    log.warning("admin drain of pod %s (%s)", pod_id, pod.model)
                    await self._admin_remove(pod, now)
                    await self._publish(pod.model, now)
            elif op == "local_capacity":
                capacity = cmd.get("capacity")
                if self.config.provider != "local" or not isinstance(self.provider, LocalServing):
                    log.error("ignoring local_capacity: provider %s", self.config.provider)
                elif capacity not in ("on", "off"):
                    log.error("ignoring local_capacity %r", capacity)
                else:
                    self.provider.no_capacity = capacity == "off"
                    log.warning("local backend: capacity %s (admin, dev)", capacity)
            else:
                log.error("ignoring unknown admin command %r", op)

    async def _admin_remove(self, pod: _Pod, now: float) -> None:
        if pod.state == "ready":
            await self._start_drain(pod, now, by="admin")
        elif pod.state in ("creating", "bootstrapping"):
            if pod.task is not None and not pod.task.done():
                pod.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await pod.task
            # Without a pod id the create may still finish on RunPod's side: the row stays
            # counted until a later listing settles it by name.
            await self._terminate(pod, by="admin", failure="stopped by admin")

    async def _observe(self, listing: list[PodInfo], seq: int, now: float) -> None:
        """The pod-state rule (§5 "Health rules", first row; §11 "Failure and restart
        rules"), rows waiting on a name, and orphans. Only a listing requested after a
        pod's id was known (`seq > watch_seq`) says anything about it."""
        by_id = {p.pod_id: p for p in listing}
        by_name = {p.name: p for p in listing}
        for pod in list(self._pods.values()):
            if pod.state == "terminated" or seq <= pod.watch_seq:
                continue
            if pod.pod_id is None:
                if pod.pending_name and not pod.busy:
                    await self._settle_by_name(pod, by_name.get(pod.row.pod_name), now)
                continue
            if pod.state == "failed":
                continue  # its termination is retried each tick
            info = by_id.get(pod.pod_id)
            if info is not None and not info.gone:
                pod.misses = 0
                continue
            if info is None:
                pod.misses += 1
                if pod.misses < 2:
                    try:
                        info = await self.provider.get(pod.pod_id)
                    except ProviderError as e:
                        log.warning("GET of pod %s failed: %s", pod.pod_id, e)
                        continue
                    if info is not None and not info.gone:
                        continue  # the listing lags: the pod is there
            if pod.busy:
                pod.lost = True  # its bootstrap notices on its next check
                continue
            await self._lost(pod, info, now)
        tracked_ids = {p.pod_id for p in self._pods.values()}
        tracked_names = {p.row.pod_name for p in self._pods.values()}
        for info in listing:
            if not info.managed or info.pod_id in tracked_ids or info.name in tracked_names:
                continue
            row = await self._db_call(store.by_pod_id, info.pod_id)
            if row is None:
                await self._terminate_orphan(info)
            elif not row.live:
                # Its row says terminated but RunPod still lists it: terminate again.
                log.info("pod %s is still listed after its termination: terminating", info.pod_id)
                with contextlib.suppress(ProviderError):
                    await self.provider.terminate(info.pod_id)

    async def _lost(self, pod: _Pod, info: PodInfo | None, now: float) -> None:
        """A pod RunPod lost (gone, EXITED, TERMINATED): out of routing, terminated
        (idempotent; an EXITED pod still bills for its disk), then its row is marked. A
        failed terminate leaves the row `failed` and counted, retried each tick."""
        state = "gone" if info is None else info.status
        log.warning("pod %s (%s) is %s: out of routing", pod.pod_id, pod.model, state)
        pod.routable = False
        failure = "pod_lost" if pod.state == "ready" else None
        if failure is not None:
            self._count_failure(pod, failure, now)
        await self._publish(pod.model, now)
        await self._terminate(pod, by="runpod", failure=failure)
        await self._publish(pod.model, now)

    async def _settle_by_name(self, pod: _Pod, info: PodInfo | None, now: float) -> None:
        """A row without a pod id, against a listing requested after it went `failed`:
        no pod by its name means none exists; one found is terminated."""
        if info is None:
            log.info("no pod named %s exists: its row is closed", pod.row.pod_name)
            pod.no_pod = True
        else:
            log.warning("terminating pod %s, found by the name %s", info.pod_id, pod.row.pod_name)
            pod.known_id = info.pod_id
        pod.pending_name = False
        await self._terminate(pod, by=pod.terminate_by)
        await self._publish(pod.model, now)

    async def _terminate_orphan(self, info: PodInfo) -> None:
        log.error(
            "orphan serving pod %s (%s) has no live serving_pods row: terminating it",
            info.pod_id,
            info.name,
        )
        try:
            await self.provider.terminate(info.pod_id)
        except ProviderError as e:
            log.error("terminating orphan %s failed: %s", info.pod_id, e)
            return
        self.metrics.orphans.inc()

    async def _check_pods(self, now: float) -> None:
        """Health, progress, leases and certificates of ready pods; drains; failed pods'
        terminations. Pods are checked concurrently."""
        jobs: list[Coroutine[Any, Any, None]] = []
        for pod in list(self._pods.values()):
            if pod.busy:
                continue
            if pod.state in ("creating", "bootstrapping"):
                # Its task is gone (crashed, cancelled, or a failure it could not record).
                jobs.append(self._fail(pod, "internal: the create task ended unfinished", now=now))
            elif pod.state == "ready":
                jobs.append(self._check_ready(pod, now))
            elif pod.state == "draining":
                jobs.append(self._check_draining(pod, now))
            elif pod.state == "failed" and pod.pod_id is not None:
                jobs.append(self._retry_terminate(pod))
            elif pod.state == "failed" and not pod.pending_name:
                pod.pending_name = True  # settled by name on a later listing
                pod.watch_seq = self._listing_seq
        if jobs:
            results = await asyncio.gather(*jobs, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError):
                    log.error("pod check failed: %s", _describe(r), exc_info=r)

    async def _check_ready(self, pod: _Pod, now: float) -> None:
        h = self.config.health
        replica = pod.replica()
        if replica is None:
            await self._fail(pod, "internal: ready pod without endpoints or token", now=now)
            return
        if now - pod.last_health >= h.proxy_health_every_s:
            pod.last_health = now
            assert self.http is not None
            ok = await health.proxy_health(self.http, replica, timeout_s=HEALTH_TIMEOUT_S)
            if ok:
                if not pod.routable:
                    log.warning(
                        "pod %s (%s) is healthy again: back in routing", pod.pod_id, pod.model
                    )
                pod.health_failures = 0
                pod.out_since = None
                pod.routable = True
            else:
                pod.health_failures += 1
                if pod.routable and pod.health_failures >= h.proxy_health_failures:
                    log.warning(
                        "pod %s (%s) failed %d health checks: out of routing",
                        pod.pod_id,
                        pod.model,
                        pod.health_failures,
                    )
                    pod.routable = False
                    pod.out_since = now
                    await self._publish(pod.model, now)
                if pod.out_since is not None and now - pod.out_since >= h.unhealthy_grace_s:
                    await self._fail(pod, "unhealthy", now=now)
                    return
            try:
                m = await health.engine_metrics(self.http, replica, timeout_s=HEALTH_TIMEOUT_S)
            except httpx.HTTPError:
                m = None
            if m is not None:
                if m.requests_running > 0:
                    self._last_demand[pod.model] = now
                stalled = (
                    m.requests_running > 0
                    and pod.last_tokens is not None
                    and m.generation_tokens_total == pod.last_tokens
                )
                if not stalled:
                    pod.last_tokens = m.generation_tokens_total
                    pod.last_progress = now
                elif now - pod.last_progress >= h.progress_stall_s:
                    await self._fail(pod, "progress_stall", now=now)
                    return
        await self._maintain(pod, now)

    async def _maintain(self, pod: _Pod, now: float) -> None:
        """Lease renewal and certificate renewal, when due."""
        lc = self.config.lifecycle
        if now - pod.last_lease >= lc.lease_renew_every_s:
            await self._renew_lease(pod, now)
            if pod.state not in ("ready", "draining"):
                return
        expires = pod.cert_not_after
        if (
            pod.state == "ready"
            and expires is not None
            and expires - now <= lc.server_cert_renew_before_s
        ):
            try:
                await self._rekey(pod, new_token=False)
                log.info("renewed the server certificate of pod %s", pod.pod_id)
            except Exception as e:
                log.warning("certificate renewal of pod %s failed: %s", pod.pod_id, _describe(e))
                if expires <= now:
                    await self._fail(pod, "certificate_expired", now=now)

    async def _check_draining(self, pod: _Pod, now: float) -> None:
        await self._maintain(pod, now)
        if pod.state != "draining":
            return
        elapsed = now - (pod.drain_started if pod.drain_started is not None else now)
        if elapsed < 2 * POOL_CACHE_TTL_S:
            return
        if elapsed < self.config.lifecycle.drain_max_s:
            replica = pod.replica()
            if replica is not None:
                assert self.http is not None
                try:
                    m = await health.engine_metrics(self.http, replica, timeout_s=HEALTH_TIMEOUT_S)
                except httpx.HTTPError as e:
                    # Unknown, not zero: try again next tick, until drain_max_s.
                    log.info("draining pod %s: /metrics failed (%s)", pod.pod_id, _describe(e))
                    return
                if m.requests_running > 0:
                    return
        log.info("pod %s (%s) drained after %.0f s: terminating", pod.pod_id, pod.model, elapsed)
        await self._terminate(pod, by=pod.drain_by, keep_state=True)
        await self._publish(pod.model, now)

    async def _retry_terminate(self, pod: _Pod) -> None:
        if await self._terminate(pod, by=pod.terminate_by):
            await self._publish(pod.model, self.clock())

    async def _read_usage(self, now: float) -> None:
        """Usage events since the last tick (no consumer group: billing loses nothing)."""
        cursor = self._usage_cursor or "0-0"
        try:
            while True:
                batch: Any = await self.redis.xread({self.usage_stream: cursor}, count=1000)
                if not batch:
                    break
                entries: list[tuple[Any, dict[Any, Any]]] = batch[0][1]
                for entry_id, fields in entries:
                    cursor = _str(entry_id)
                    model = _str(fields.get(b"model_id") or fields.get("model_id") or "")
                    if model in self.config.models:
                        self._last_demand[model] = now
                if len(entries) < 1000:
                    break
        finally:
            self._usage_cursor = cursor

    async def _reconcile_model(self, model: str, now: float, budget: _Budget) -> None:
        served = self.config.models[model]
        pods = self._model_pods(model)
        wake = bool(await self.redis.exists(wake_key(model)))
        if wake:
            self._last_demand[model] = now
        ready = sorted(
            (p for p in pods if p.state == "ready"),
            key=lambda p: (p.row.created_at, p.row.pod_name),
        )
        inflight = [p for p in pods if p.state in ("creating", "bootstrapping")]
        idle_for = now - self._last_demand.get(model, -math.inf)
        demand = wake or (
            bool(ready) and (served.idle_stop_s == 0 or idle_for < served.idle_stop_s)
        )
        desired = min(max(served.min_replicas, 1 if demand else 0), served.max_replicas)
        active = len(ready) + len(inflight)
        if desired > active:
            dep = self.deployments.get(model)
            if dep is None:
                return
            for _ in range(desired - active):
                if not await self._try_create(dep, now, budget):
                    break
        elif not demand and served.idle_stop_s > 0 and len(ready) > desired:
            for pod in ready[desired:]:
                log.info("pod %s (%s) idle for %.0f s: draining", pod.pod_id, model, idle_for)
                await self._start_drain(pod, now, by="manager")

    async def _try_create(self, dep: Deployment, now: float, budget: _Budget) -> bool:
        model = dep.model_id
        served = dep.served
        if model in self._holds:
            return False
        cap = self._capacity.get(model)
        if cap is not None and now < cap.last_attempt + self._backoff(cap.failures):
            return False
        if len(self._model_pods(model)) >= served.max_replicas:
            return False
        lc = self.config.lifecycle
        listed = dep.listed_price_per_hr_micros or 0
        cap_micros = spend.usd_micros(self.config.spend.monthly_cap_usd)
        month = await budget.get()
        if not spend.guard_allows(month, listed, lc.lease_ttl_s, cap_micros):
            self._holds[model] = "spend_cap"
            self.metrics.creates.labels(model, "spend_cap").inc()
            log.error(
                "spend cap: refusing a %s pod (month to date $%.2f + $%.2f/h x %d s lease "
                "> cap $%.2f); `loom-poolmgr reset %s` once there is room",
                model,
                month / 1e6,
                listed / 1e6,
                lc.lease_ttl_s,
                cap_micros / 1e6,
                model,
            )
            return False
        row = await self._db_call(
            store.insert_creating,
            model_id=model,
            pod_name=pod_name(model, now),
            created_at=_dt(now),
            max_replicas=served.max_replicas,
        )
        if row is None:
            return False
        budget.micros = month + math.ceil(Fraction(listed * lc.lease_ttl_s, 3600))
        pod = _Pod(row, created=now)
        self._pods[row.id] = pod
        pod.task = self._spawn(self._create_and_bootstrap(pod, dep), f"create {row.pod_name}")
        log.info("creating %s for %s", row.pod_name, model)
        return True

    def _backoff(self, failures: int) -> float:
        steps = self.config.capacity.backoff_s
        return float(steps[min(max(failures, 1), len(steps)) - 1])

    async def _start_drain(self, pod: _Pod, now: float, *, by: TerminatedBy) -> None:
        pod.routable = False
        pod.drain_by = by
        pod.drain_started = now
        await self._update(pod, state="draining", drain_started_at=_dt(now))
        await self._publish(pod.model, now)

    async def _spend_alerts(self, now: float, budget: _Budget) -> None:
        cap_micros = spend.usd_micros(self.config.spend.monthly_cap_usd)
        month = await self._db_call(spend.month_to_date_micros, _dt(now))
        self.metrics.spend_month.set(month)
        crossed = spend.crossed_fractions(month, cap_micros, self.config.spend.alert_fractions)
        if not crossed:
            return
        for fraction in await self._db_call(spend.record_alerts, _dt(now), crossed):
            log.error(
                "serving spend alert: month to date $%.2f is %.0f%% of the $%.2f cap",
                month / 1e6,
                fraction * 100,
                cap_micros / 1e6,
            )

    # ------------------------------------------------------------------ pools

    def _model_pods(self, model: str) -> list[_Pod]:
        return [p for p in self._pods.values() if p.model == model]

    def _pool_state(self, model: str, now: float) -> PoolState:
        unavailable_s = self.config.capacity.unavailable_retry_after_s
        served = self.config.models.get(model)
        if served is None:
            return PoolState(model, "unavailable", unavailable_s, reason="not_configured")
        pods = self._model_pods(model)
        ready = sorted(
            (p for p in pods if p.state == "ready" and p.routable),
            key=lambda p: (p.row.created_at, p.row.pod_name),
        )
        replicas = tuple(r for r in (p.replica() for p in ready) if r is not None)
        if replicas:
            return PoolState(model, "ready", 0, replicas)
        hold = self._holds.get(model)
        if hold is not None:
            return PoolState(model, "unavailable", unavailable_s, reason=hold)
        inflight = [p for p in pods if p.state in ("creating", "bootstrapping")]
        cap = self._capacity.get(model)
        if cap is not None and cap.no_capacity and not inflight:
            return PoolState(
                model, "unavailable", int(self._backoff(cap.failures)), reason="no_capacity"
            )
        if inflight:
            oldest = min(p.created for p in inflight)
            left = math.ceil(served.cold_start_estimate_s - (now - oldest))
            return PoolState(model, "warming", max(WARMING_FLOOR_S, left))
        return PoolState(model, "cold", served.cold_start_estimate_s)

    async def _publish(self, model: str, now: float | None = None) -> None:
        state = self._pool_state(model, self.clock() if now is None else now)
        try:
            await set_pool(self.redis, state)
        except Exception as e:
            log.error("writing the pool hash of %s failed: %s", model, _describe(e))

    def _published_models(self) -> list[str]:
        models = list(self.config.models)
        models += [m for m in self.servable if m not in self.config.models]
        return models

    async def _publish_all(self, now: float) -> None:
        counts: dict[str, dict[str, int]] = {m: {} for m in self.config.models}
        for pod in self._pods.values():
            by_state = counts.setdefault(pod.model, {})
            by_state[pod.state] = by_state.get(pod.state, 0) + 1
        self.metrics.set_pods(counts)
        for model in self._published_models():
            await self._publish(model, now)

    # ------------------------------------------------------------------ create and bootstrap

    def _spawn(self, coro: Coroutine[Any, Any, None], name: str) -> asyncio.Task[None]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)

        def done(t: asyncio.Task[None]) -> None:
            self._tasks.discard(t)
            exc = None if t.cancelled() else t.exception()
            if exc is not None:
                log.error("%s failed: %s", name, _describe(exc), exc_info=exc)

        task.add_done_callback(done)
        return task

    def _capacity_failure(self, model: str, at: float, *, no_capacity: bool) -> None:
        cap = self._capacity.get(model)
        failures = 1 if cap is None else cap.failures + 1
        self._capacity[model] = _Capacity(failures, at, no_capacity)

    async def _create_and_bootstrap(self, pod: _Pod, dep: Deployment) -> None:
        """The create task. Whatever goes wrong, the row ends `failed` and its pod is
        terminated (by name when its id is unknown); a failure it cannot even record is
        picked up by the next tick (a creating/bootstrapping row with no live task)."""
        try:
            await self._create_and_bootstrap_inner(pod, dep)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error(
                "create task of %s failed unexpectedly: %s",
                pod.row.pod_name,
                _describe(e),
                exc_info=e,
            )
            try:
                await self._fail(pod, f"internal: {_describe(e)}", now=self.clock())
            except Exception as again:
                log.error(
                    "failing %s failed too (the next tick retries): %s",
                    pod.row.pod_name,
                    _describe(again),
                )
            await self._publish(dep.model_id)

    async def _create_and_bootstrap_inner(self, pod: _Pod, dep: Deployment) -> None:
        model = dep.model_id
        created = pod.created
        lc = self.config.lifecycle
        env = {
            "LOOM_MANAGED": "true",
            "LOOM_ROLE": "serving",
            "LOOM_MODEL": model,
            "LOOM_LEASE_EPOCH": str(int(created) + lc.lease_ttl_s),
            "LOOM_SSH_PUBKEY": self.ssh.public_key,
        }
        try:
            info = await self.provider.create(dep, pod.row.pod_name, env)
        except NoCapacity as e:
            log.warning("no capacity for %s: %s", model, e)
            self.metrics.creates.labels(model, "no_capacity").inc()
            self._capacity_failure(model, created, no_capacity=True)
            pod.no_pod = True
            await self._terminated(pod, by="manager", failure="no_capacity")
            await self._publish(model)
            return
        except ProviderError as e:
            log.error("create of %s failed: %s", pod.row.pod_name, e)
            self.metrics.creates.labels(model, "error").inc()
            self._capacity_failure(model, created, no_capacity=False)
            failure = f"create_failed: {e}"[:500]
            if isinstance(e, CreateFailed) and e.ambiguous:
                if e.cleaned:
                    # The pod the create made anyway was found by name and terminated.
                    pod.known_id = e.cleaned[0]
                    await self._terminated(pod, by="manager", failure=failure)
                else:
                    # A pod may still appear under this name: keep the row counted until
                    # a later listing settles it.
                    await self._fail(pod, failure, now=self.clock())
            else:
                pod.no_pod = True
                await self._terminated(pod, by="manager", failure=failure)
            await self._publish(model)
            return
        pod.known_id = info.pod_id
        pod.watch_seq = self._listing_seq
        self._capacity.pop(model, None)
        now = self.clock()
        api_cost = info.cost_per_hr_micros
        cost = (api_cost if api_cost is not None else dep.listed_price_per_hr_micros or 0) + (
            dep.disk_per_hr_micros
        )
        await self._update(
            pod,
            pod_id=info.pod_id,
            cost_per_hr_micros=cost,
            gpu_type_id=dep.gpu_type_id,
            gpu_count=dep.gpu_count,
            data_center=info.data_center,
            stages={"pod_created": round(now - created, 3)},
        )
        listed = dep.listed_price_per_hr_micros
        ratio = Decimal(str(self.config.spend.max_price_ratio))
        if listed is not None and api_cost is not None and Decimal(api_cost) > listed * ratio:
            log.error(
                "pod %s (%s) costs $%.4f/h, above bench/prices.yaml $%.4f x %s: terminating; "
                "update the price list or the ratio, then `loom-poolmgr reset %s`",
                info.pod_id,
                model,
                api_cost / 1e6,
                listed / 1e6,
                ratio,
                model,
            )
            self.metrics.creates.labels(model, "price_too_high").inc()
            self._holds[model] = "price_too_high"
            await self._fail(pod, "price_too_high", now=now)
            await self._publish(model)
            return
        self.metrics.creates.labels(model, "ok").inc()
        try:
            await self._bootstrap(pod, dep, info)
        except asyncio.CancelledError:
            raise
        except PodLost as e:
            now = self.clock()
            log.warning("pod %s (%s) lost during bootstrap: %s", pod.pod_id, model, e)
            failure = "pod_lost_during_bootstrap"
            self._count_failure(pod, failure, now)
            await self._terminate(pod, by="runpod", failure=failure)
        except Exception as e:
            log.error("bootstrap of pod %s (%s) failed: %s", pod.pod_id, model, _describe(e))
            await self._fail(pod, f"bootstrap: {_describe(e)}", now=self.clock())
        await self._publish(model)

    async def _bootstrap(self, pod: _Pod, dep: Deployment, info: PodInfo) -> None:
        h = self.config.health
        created = pod.created
        # §11: the timeout runs from the row's created_at (as stored, to the microsecond).
        deadline = pod.row.created_at.timestamp() + h.bootstrap_timeout_s
        assert pod.pod_id is not None
        pod_id = pod.pod_id

        async def check() -> None:
            if pod.lost:
                raise PodLost(f"{pod_id} is gone")
            if self.clock() >= deadline:
                raise BootstrapTimeout(f"no first token within {h.bootstrap_timeout_s} s")

        stages: dict[str, float] = dict(pod.row.stages)
        endpoints = None if info.gone else self.provider.endpoints(info)
        while endpoints is None:
            if info.gone:
                raise PodLost(f"{pod_id} is {info.status}")
            await check()
            await self.sleep(BOOT_POLL_S)
            await check()
            try:
                found = await self.provider.get(pod_id)
            except ProviderError as e:
                log.warning("GET of pod %s failed: %s", pod_id, e)
                continue
            if found is None:
                raise PodLost(f"{pod_id} is gone (GET answers 404)")
            info = found
            endpoints = None if info.gone else self.provider.endpoints(info)
        pod.endpoints = endpoints
        stages["pod_running"] = round(self.clock() - created, 3)
        await self._update(
            pod,
            state="bootstrapping",
            public_ip=endpoints.ip,
            proxy_port=endpoints.proxy_port,
            ssh_port=endpoints.ssh_port,
            data_center=info.data_center or pod.row.data_center,
            stages=stages,
        )
        await self._publish(pod.model)
        target = self._target(pod)
        await bootstrap.wait_for_ssh(
            self.pod_exec,
            target,
            timeout_s=max(0.0, deadline - self.clock()),
            poll_s=BOOT_POLL_S,
            clock=self.clock,
            sleep=self.sleep,
            check=check,
        )
        stages["ssh_online"] = round(self.clock() - created, 3)
        await self._renew_lease(pod, self.clock(), strict=True)
        await check()
        await self._rekey(pod, new_token=True)
        await check()
        script = bootstrap.engine_script(
            dep, stall_s=h.bootstrap_stall_s, ready_timeout_s=h.bootstrap_timeout_s
        )
        pod_stages = await bootstrap.start_engine(
            self.pod_exec,
            target,
            script,
            timeout_s=float(h.bootstrap_timeout_s),
            poll_s=ENGINE_POLL_S,
            check_host=check,
            sleep=self.sleep,
        )
        stages.update(stage_offsets(pod_stages, created))
        replica = pod.replica()
        assert replica is not None and self.http is not None
        while not await health.proxy_health(self.http, replica, timeout_s=HEALTH_TIMEOUT_S):
            await check()
            await self.sleep(BOOT_POLL_S)
            await check()
        served_model = bootstrap.engine_launch(dep).served_model
        await health.probe(self.http, replica, served_model, timeout_s=PROBE_TIMEOUT_S)
        await check()
        now = self.clock()
        stages = dict(sorted(stages.items(), key=lambda kv: kv[1]))
        await self._update(pod, state="ready", ready_at=_dt(now), stages=stages)
        pod.routable = True
        pod.last_health = now
        pod.last_progress = now
        for stage, offset in stages.items():
            self.metrics.cold_start.labels(pod.model, stage).observe(offset)
        log.info("pod %s (%s) ready %.0f s after its create", pod_id, pod.model, now - created)

    def _target(self, pod: _Pod) -> SshTarget:
        assert pod.endpoints is not None and pod.pod_id is not None
        return bootstrap.ssh_target(
            pod.endpoints.ip,
            pod.endpoints.ssh_port,
            self.ssh.key_path,
            self.ssh.known_hosts(pod.pod_id),
        )

    def _allow(self) -> tuple[str, ...]:
        base = tuple(self.config.network.gateway_egress_cidrs) or LOOPBACK_ALLOW
        return tuple(dict.fromkeys(base + self.extra_allow))

    async def _rekey(self, pod: _Pod, *, new_token: bool) -> None:
        """Deliver a new server certificate (and, with `new_token`, a new token) and have
        the proxy start or reload with it."""
        assert pod.pod_id is not None
        issued = self.pki.ca.issue_server(
            pod.pod_id, days=self.config.lifecycle.server_cert_days, now=self.clock()
        )
        token = pod.token if pod.token is not None and not new_token else secrets.token_urlsafe(32)
        delivered = PodSecrets(
            server_cert=issued.cert_pem(),
            server_key=issued.key_pem(),
            ca_cert=self.pki.ca.cert_pem(),
            token=token,
            allow=self._allow(),
        )
        await bootstrap.deliver(
            self.pod_exec,
            self._target(pod),
            delivered,
            self.bundle,
            proxy_port=self.config.network.proxy_port,
        )
        pod.token = token
        pod.cert_not_after = issued.not_after.timestamp()
        await self._update(pod, cert_not_after=issued.not_after)

    async def _renew_lease(self, pod: _Pod, now: float, *, strict: bool = False) -> bool:
        """serve_lease over SSH, then the lease record. `strict` (bootstrap) raises when
        the pod refuses; otherwise a failure is retried next tick, and the pod fails only
        once its lease ends within 2 renewal periods."""
        lc = self.config.lifecycle
        assert pod.pod_id is not None
        epoch = int(now) + lc.lease_ttl_s
        try:
            await renew_on_pod(
                self.pod_exec, self._target(pod), epoch, timeout_s=bootstrap.LEASE_CALL_TIMEOUT_S
            )
        except (RemoteCommandError, OSError) as e:
            self.metrics.lease_renewals.labels("error").inc()
            if strict:
                raise
            log.warning("lease renewal of pod %s failed: %s", pod.pod_id, _describe(e))
            if pod.lease_until is not None and pod.lease_until - now <= 2 * lc.lease_renew_every_s:
                await self._fail(pod, "lease_renewal_failed", now=now)
            return False
        pod.lease_until = float(epoch)
        await self._update(pod, lease_until=_dt(epoch))
        try:
            await self.leases.put(
                LeaseRecord(
                    pod_id=pod.pod_id,
                    pod_name=pod.row.pod_name,
                    model=pod.model,
                    lease_until=_dt(epoch),
                    written_at=_dt(now),
                )
            )
        except Exception as e:
            self.metrics.lease_renewals.labels("error").inc()
            log.warning("lease record of pod %s not written: %s", pod.pod_id, _describe(e))
            return False
        pod.last_lease = now
        self.metrics.lease_renewals.labels("ok").inc()
        return True

    # ------------------------------------------------------------------ failure and termination

    def _count_failure(self, pod: _Pod, failure: str, now: float) -> None:
        """Count `failure` toward the crash loop, once per pod, when it is one (the same
        rule as `_load_holds` applies to rows after a restart)."""
        if pod.counted or not _counts(failure):
            return
        pod.counted = True
        model = pod.model
        self.metrics.replacements.labels(model, failure.split(":")[0]).inc()
        recent = [t for t in self._failures.get(model, []) if now - t < CRASH_WINDOW_S]
        recent.append(now)
        self._failures[model] = recent
        limit = self.config.lifecycle.max_replacements_per_hour
        if len(recent) > limit and self._holds.get(model) != "crash_loop":
            self._holds[model] = "crash_loop"
            log.error(
                "crash loop: %d %s pods failed or were lost in the last hour (limit %d); no "
                "new pods until `loom-poolmgr reset %s`",
                len(recent),
                model,
                limit,
                model,
            )

    async def _fail(self, pod: _Pod, failure: str, *, now: float) -> None:
        """Out of routing, `failed`, then terminate (retried each tick until it works)."""
        if pod.state in ("failed", "terminated"):
            return
        log.error("pod %s (%s) failed: %s", pod.pod_id or pod.row.pod_name, pod.model, failure)
        pod.routable = False
        self._count_failure(pod, failure, now)
        await self._update(pod, state="failed", failure=failure[:500])
        await self._publish(pod.model, now)
        await self._terminate(pod, by="manager")

    async def _terminate(
        self,
        pod: _Pod,
        *,
        by: TerminatedBy,
        failure: str | None = None,
        keep_state: bool = False,
    ) -> bool:
        """Terminate the pod (never stop) and mark its row `terminated`. When RunPod
        refuses, or the pod id is unknown and a pod may exist under the row's name, the
        row stays live (and counted against `max_replicas` and spend) as `failed` (or as
        it is, with `keep_state`) and False is returned; ticks retry."""
        if pod.state == "terminated":
            return True
        pod.routable = False
        pod.terminate_by = by
        fields: dict[str, Any] = {}
        if failure is not None and pod.row.failure is None:
            fields["failure"] = failure[:500]
        pod_id = pod.pod_id
        if pod_id is None and not pod.no_pod:
            pod.pending_name = True
            pod.watch_seq = self._listing_seq
            if pod.state != "failed":
                await self._update(pod, state="failed", **fields)
            return False
        if pod_id is not None:
            try:
                await self.provider.terminate(pod_id)
            except ProviderError as e:
                log.error("terminating pod %s failed (retried next tick): %s", pod_id, e)
                if not keep_state and pod.state != "failed":
                    await self._update(pod, state="failed", **fields)
                return False
        await self._terminated(pod, by=by, failure=failure)
        return True

    async def _terminated(self, pod: _Pod, *, by: TerminatedBy, failure: str | None = None) -> None:
        fields: dict[str, Any] = {
            "state": "terminated",
            "terminated_by": by,
            "terminated_at": _dt(self.clock()),
        }
        if failure is not None and pod.row.failure is None:
            fields["failure"] = failure[:500]
        if pod.row.pod_id is None and pod.known_id is not None:
            fields["pod_id"] = pod.known_id
        pod.routable = False
        await self._update(pod, **fields)
        self._pods.pop(pod.row.id, None)
        if pod.pod_id is not None:
            try:
                await self.leases.delete(pod.pod_id)
            except Exception as e:
                log.warning("deleting the lease record of %s failed: %s", pod.pod_id, _describe(e))

    # ------------------------------------------------------------------ adoption

    async def _adopt_all(self) -> None:
        now = self.clock()
        await self._load_holds(now)
        rows = await self._db_call(store.live_rows)
        self._listing_seq += 1
        listing = await self.provider.list()
        by_id = {p.pod_id: p for p in listing}
        by_name = {p.name: p for p in listing}
        matched: set[str] = set()
        adopt: list[tuple[_Pod, PodInfo]] = []
        for row in rows:
            pod = _Pod(row, watch_seq=self._listing_seq)
            self._pods[row.id] = pod
            info = by_id.get(row.pod_id) if row.pod_id else by_name.get(row.pod_name)
            if info is None and row.pod_id is not None:
                info = await self.provider.get(row.pod_id)  # the listing may lag
            if info is not None:
                matched.add(info.pod_id)
                pod.known_id = info.pod_id
            if info is None or info.gone:
                log.warning(
                    "pod %s (%s) went away while no manager ran",
                    row.pod_id or row.pod_name,
                    row.model_id,
                )
                if row.pod_id is None and info is None:
                    pod.no_pod = True  # the listing shows no pod with its name
                await self._terminate(pod, by="runpod")
                continue
            if row.pod_id is None:
                await self._update(pod, pod_id=info.pod_id)
            if row.state == "ready" and row.model_id in self.config.models:
                adopt.append((pod, info))
            elif row.state == "draining":
                await self._terminate(pod, by="manager")
            else:
                reason = (
                    f"manager restarted while {row.state}"
                    if row.state in ("creating", "bootstrapping")
                    else row.failure or "not configured"
                )
                await self._fail(pod, reason, now=now)
                if pod.state == "failed":
                    await self._terminate(pod, by="manager")
        for info in listing:
            if info.managed and info.pod_id not in matched:
                await self._terminate_orphan(info)
        # Nothing is routable until re-keyed: the pool hashes lose every old replica (and
        # token) before a re-key starts.
        await self._publish_all(now)
        if adopt:
            await asyncio.gather(*(self._adopt(pod, info) for pod, info in adopt))
        await self._publish_all(self.clock())

    async def _adopt(self, pod: _Pod, info: PodInfo) -> None:
        try:
            endpoints = self.provider.endpoints(info)
            if endpoints is None:
                raise PodLost(f"{info.pod_id} has no endpoints")
            pod.endpoints = endpoints
            await self._rekey(pod, new_token=True)
            await self._renew_lease(pod, self.clock(), strict=True)
            replica = pod.replica()
            assert replica is not None and self.http is not None
            if not await health.proxy_health(self.http, replica, timeout_s=HEALTH_TIMEOUT_S):
                raise RuntimeError("proxy health check failed")
            dep = self.deployments.get(pod.model)
            served_model = pod.model if dep is None else bootstrap.engine_launch(dep).served_model
            await health.probe(self.http, replica, served_model, timeout_s=PROBE_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("adopting pod %s (%s) failed: %s", pod.pod_id, pod.model, _describe(e))
            await self._fail(pod, f"adoption: {_describe(e)}", now=self.clock())
            return
        now = self.clock()
        await self._update(
            pod,
            public_ip=endpoints.ip,
            proxy_port=endpoints.proxy_port,
            ssh_port=endpoints.ssh_port,
        )
        pod.routable = True
        pod.last_health = now
        pod.last_progress = now
        # §11: an adopted pod counts as demand now (no idle drain on the first tick).
        self._last_demand[pod.model] = now
        log.info("adopted pod %s (%s)", pod.pod_id, pod.model)

    async def _load_holds(self, now: float) -> None:
        """`crash_loop` and `price_too_high` from the last hour's rows, counting only
        failures after the model's last `reset` (spec §11), by the rule `_count_failure`
        applies live."""
        for model in self.config.models:
            raw: Any = await self.redis.get(RESET_KEY.format(model=model))
            if raw is not None:
                with contextlib.suppress(ValueError):
                    self._reset_at[model] = float(_str(raw))
        window = now - CRASH_WINDOW_S
        rows = await self._db_call(store.rows_since, _dt(window))
        price: set[str] = set()
        for row in rows:
            if row.failure is None or row.model_id not in self.config.models:
                continue
            at = row.terminated_at.timestamp() if row.terminated_at is not None else now
            if at < window or at <= self._reset_at.get(row.model_id, -math.inf):
                continue
            if row.failure == "price_too_high":
                price.add(row.model_id)
            elif _counts(row.failure):
                self._failures.setdefault(row.model_id, []).append(at)
        limit = self.config.lifecycle.max_replacements_per_hour
        for model, times in self._failures.items():
            if len(times) > limit:
                self._holds[model] = "crash_loop"
                log.error("crash loop: %s had %d failed pods in the last hour", model, len(times))
        for model in price:
            self._holds.setdefault(model, "price_too_high")
            log.error("%s: a pod was refused as priced too high in the last hour", model)

    # ------------------------------------------------------------------ database

    async def _db_call(self, fn: Callable[..., Awaitable[T]], *args: Any, **kw: Any) -> T:
        if self._db is None:
            raise RuntimeError("the manager is not started")
        async with self._db.connection() as conn:
            return await fn(conn, *args, **kw)

    async def _update(self, pod: _Pod, **fields: Any) -> None:
        pod.row = await self._db_call(store.update, pod.row.id, **fields)


def _counts(failure: str) -> bool:
    """Whether a `serving_pods.failure` is a pod failing (counted toward the crash
    loop), not a refusal or a restart."""
    return not failure.startswith(NOT_REPLACEMENTS)


def _str(v: bytes | str) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
