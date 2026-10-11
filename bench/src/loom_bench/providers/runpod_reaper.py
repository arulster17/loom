"""TTL reaper for Loom-managed RunPod pods.

A pod is Loom's only when it is BOTH named `{prefix}-…` AND carries env
`LOOM_MANAGED=true`, so a shared account's other pods are never touched even if
their names collide. A managed pod is terminated when its TTL has passed, or when
it is `EXITED` (an exited pod still bills for its disk). The TTL comes from env
`LOOM_TTL` (epoch seconds), falling back to the epoch encoded in the name
(`{prefix}-{experiment}-{ttl_epoch}-{nonce}`); with neither, a pod older than
`max_age` is reaped, and one with no readable creation time is reaped outright.

Runs from `bench reap`. The scheduled backstop is the `runpod_reaper_lambda` Lambda,
whose rules are stricter (exact runner name format and an expired TTL only).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

MANAGED_ENV = "LOOM_MANAGED"
TTL_ENV = "LOOM_TTL"
DEFAULT_MAX_AGE = timedelta(hours=24)
GONE_STATES = frozenset({"TERMINATED"})
_NAME_TTL_RE = re.compile(r"-(\d{10})-[a-z0-9]{6}$")

log = logging.getLogger(__name__)


class PodApi(Protocol):
    def list_pods(self) -> list[dict[str, Any]]: ...

    def terminate_pod(self, pod_id: str) -> None: ...


def _env(pod: Mapping[str, Any]) -> Mapping[str, Any]:
    env = pod.get("env")
    return env if isinstance(env, Mapping) else {}


def is_managed(pod: Mapping[str, Any], prefix: str) -> bool:
    name = str(pod.get("name") or "")
    return name.startswith(f"{prefix}-") and str(_env(pod).get(MANAGED_ENV)) == "true"


def _epoch(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(int(str(value)), UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def pod_ttl(pod: Mapping[str, Any]) -> datetime | None:
    ttl = _epoch(_env(pod).get(TTL_ENV))
    if ttl is not None:
        return ttl
    m = _NAME_TTL_RE.search(str(pod.get("name") or ""))
    return _epoch(m.group(1)) if m else None


def created_at(pod: Mapping[str, Any]) -> datetime | None:
    """`createdAt` as RunPod writes it, e.g. `2026-10-06 07:03:35.426 +0000 UTC`."""
    raw = str(pod.get("createdAt") or "").removesuffix(" UTC").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(raw, fmt).astimezone(UTC)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else None


def is_expired(pod: Mapping[str, Any], now: datetime, max_age: timedelta) -> bool:
    if pod.get("desiredStatus") == "EXITED":
        return True
    ttl = pod_ttl(pod)
    if ttl is not None:
        return ttl < now
    created = created_at(pod)
    return created is None or created < now - max_age


def reap(
    api: PodApi,
    now: datetime,
    *,
    prefix: str,
    dry_run: bool = False,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> list[str]:
    """Terminate expired managed pods; returns the ids acted on (or that would be).

    One API call per pod, so a single failure cannot block the rest; failures are
    logged and the next run retries them.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not prefix:
        raise ValueError("an empty prefix would match every pod")
    reaped: list[str] = []
    for pod in api.list_pods():
        if pod.get("desiredStatus") in GONE_STATES or not is_managed(pod, prefix):
            continue
        if not is_expired(pod, now, max_age):
            continue
        pod_id = str(pod.get("id") or "")
        if not pod_id:
            continue
        if not dry_run:
            try:
                api.terminate_pod(pod_id)
            except Exception:
                log.exception("terminate pod %s failed", pod_id)
                continue
        log.info("reaped pod %s (dry_run=%s)", pod_id, dry_run)
        reaped.append(pod_id)
    return reaped
