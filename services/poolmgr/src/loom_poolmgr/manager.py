"""The pool manager (docs/phase1-m2-spec.md §3, §5, §7): one process that creates,
bootstraps, health-checks, drains and terminates serving pods and writes
`loom:pool:<model>` as the gateway reads it.

Use:

    manager = PoolManager(config, deployments=..., provider=..., pod_exec=..., redis=...,
                          database_url=..., leases=..., pki=..., ssh=..., bundle=...)
    await manager.start()          # single-writer lock, then adoption (§5 table)
    await manager.reconcile_once() # one tick; `run()` calls it every reconcile_every_s
    await manager.wait_idle()      # tests: until the creates/bootstraps/drains it started end
    await manager.stop()           # cancels its tasks, releases the lock; pods keep running

**Time.** Every timing rule reads `clock()` (epoch seconds) and every wait goes through
`sleep`, so tests drive the manager with a fake clock: advance it, call
`reconcile_once()`, then `wait_idle()`. A tick runs each periodic check that is due at
`clock()`: the pod listing (`pod_state_every_s`), proxy health and engine progress
(`proxy_health_every_s`), lease renewal (`lease_renew_every_s`), certificate renewal;
then demand, creates/drains/terminations, and the pool hashes. A create (and the
bootstrap after it) and a drain run as tasks the tick does not wait for.

**Demand** (§5 step 1): a live wake key `loom:pool:<model>:wake`, or a ready model with a
usage event (`loom:usage`, read with `XREAD` from where the last tick stopped, no consumer
group) or an in-flight request within `idle_stop_s`. Desired replicas:
`max(min_replicas, 1 if demand else 0)`, capped at `max_replicas`.

**Holds.** `crash_loop` (more than `max_replacements_per_hour` replacements in the last
hour), `spend_cap` and `price_too_high` keep a model `unavailable` and ignore its wakes
until an admin `reset`; `not_configured` (a servable model missing from serving.yaml) is
permanent.

**Admin commands** (`loom-poolmgr drain|stop|reset`, another process) are pushed as JSON
onto the Redis list `ADMIN_QUEUE` (`{"op": "drain", "pod_id": ...}`, `{"op": "stop",
"model": ...}`, `{"op": "reset", "model": ...}`); each tick pops them all first.
`warm` is `request_wake`, as the gateway's.

**Single writer.** `start()` takes `pg_try_advisory_lock(hashtext('loom-poolmgr'))` on a
connection it keeps for its lifetime and raises `ManagerLocked` if another manager holds
it. Adoption considers every `serving_pods` row not terminated, so tests sharing a
database clear the table first.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
from redis.asyncio import Redis

from loom_bench.experiment import RUNPOD_GPU_TYPE_IDS
from loom_bench.prices import PriceBook
from loom_bench.providers.runpod_ssh import SshExec
from loom_bench.records import Market
from loom_bench.registry import Registry
from loom_platform.serving import ServingConfig
from loom_poolmgr.bootstrap import ProxyBundle
from loom_poolmgr.leases import LeaseRecords
from loom_poolmgr.pki import Pki
from loom_poolmgr.provider import Deployment, ServingProvider

LOCK_NAME = "loom-poolmgr"
ADMIN_QUEUE = "loom:poolmgr:admin"
# The gateway caches a pool hash for up to this long; a drain waits twice it after
# removing the pod from the hash (§5 "Idle stop and drain").
POOL_CACHE_TTL_S = 1.0
# RunPod's price region for bench/prices.yaml (Secure Cloud on demand).
RUNPOD_PRICE_REGION = "secure"


class ManagerLocked(RuntimeError):
    """Another manager holds the single-writer lock."""


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
        out[model_id] = Deployment(
            model_id=model_id,
            served=served,
            spec=spec,
            gpu_type_id=RUNPOD_GPU_TYPE_IDS[spec.hardware.gpu],
            gpu_count=spec.hardware.gpus_per_replica,
            container_disk_gb=config.placement.container_disk_gb[model_id],
            listed_price_per_hr_micros=listed,
        )
    return out


async def push_admin(redis: Redis, command: Mapping[str, str]) -> None:
    """Queue an admin command for the running manager (see the module docstring)."""
    if command.get("op") not in ("drain", "stop", "reset"):
        raise ValueError(f"unknown admin command {command!r}")
    await redis.rpush(ADMIN_QUEUE, json.dumps(dict(command)))


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
    ) -> None:
        """
        - `config`: the validated serving.yaml (`loom_platform.serving.load_serving`).
        - `deployments`: per configured model (`build_deployments`).
        - `provider`: where pods come from (`RunpodServing`, `LocalServing`).
        - `pod_exec`: how scripts reach a pod (`OpenSshExec`; tests: `FakeServingExec`;
          the protocol is in `loom_poolmgr.bootstrap`).
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

    async def start(self) -> None:
        """Take the single-writer lock (raise `ManagerLocked` if held), connect, then
        adopt (§5 "Restart and adoption"): re-key and probe each `loom-serve-*` pod with
        a live row (failure: `failed`, terminate), terminate orphans (`terminated_by =
        manager`), mark rows whose pod is gone `terminated` (`terminated_by = runpod`),
        and rewrite every pool hash from scratch. Returns once adoption has finished."""
        raise NotImplementedError("phase B")

    async def reconcile_once(self) -> None:
        """One tick (see the module docstring). Raises only on a bug; provider, SSH and
        pod failures are handled and logged."""
        raise NotImplementedError("phase B")

    async def wait_idle(self) -> None:
        """Wait until every task this manager started (creates, bootstraps, drains,
        re-keys) has finished. For tests; `run()` never calls it."""
        raise NotImplementedError("phase B")

    async def run(self, stop: asyncio.Event) -> None:
        """`reconcile_once()` every `reconcile_every_s` until `stop` is set. Call
        `start()` first and `stop()` after."""
        raise NotImplementedError("phase B")

    async def stop(self) -> None:
        """Cancel the manager's tasks and release the lock (close its connections).
        Pods keep running: a restarted manager adopts them. Idempotent."""
        raise NotImplementedError("phase B")
