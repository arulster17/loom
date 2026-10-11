"""Scheduled TTL reaper for Loom's RunPod pods and network volumes (AWS Lambda).

The RunPod account is shared with work that is not Loom's, so this reaper is stricter
than `bench reap` (`runpod_reaper`) and acts on a resource only when all of these hold:

- Pod: its name is exactly the runner's `{prefix}-{experiment[:8]}-{ttl_epoch}-{nonce}`
  (`RunpodProvider.pod_name`), its env has `LOOM_MANAGED=true`, its env `LOOM_TTL`, when
  present, equals the epoch in the name, and that TTL has passed. There is no fallback: no
  age rule, no `EXITED` rule. A Loom-looking pod without a readable TTL is kept and logged.
- Network volume: its name is exactly `{prefix}-vol-{experiment[:8]}-{ttl_epoch}-{nonce}`
  (`volume_name`), its TTL has passed, and no listed pod references it. Loom creates no
  volumes today; any it creates later must be named by `volume_name` to be reaped.

It fails closed: when a listing fails or returns anything but the expected shape, or a
Loom-named resource lacks the fields the rules need, or more than `max_per_run` resources
qualify, nothing is reaped and the run raises (visible as a Lambda error). Pod env values
are never logged.

The key is a RunPod API key read from Secrets Manager on each run. This file imports only
the standard library (and boto3 in the handler) so Terraform can zip it alone.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

REST_URL = "https://rest.runpod.io/v1"
MANAGED_ENV = "LOOM_MANAGED"
TTL_ENV = "LOOM_TTL"
DEFAULT_PREFIX = "loom-bench"
DEFAULT_MAX_PER_RUN = 10
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_ID_RE = re.compile(r"^[a-z0-9]{6,32}$")
# The tail every Loom resource name ends with: `-{experiment[:8]}-{ttl_epoch}-{nonce}`.
_TAIL = r"-(?P<experiment>[A-Za-z0-9][A-Za-z0-9._-]{0,7})-(?P<ttl>\d{10})-(?P<nonce>[0-9a-f]{6})$"

log = logging.getLogger(__name__)

# (method, url, headers, timeout_s) -> (HTTP status, body)
Transport = Callable[[str, str, Mapping[str, str], float], tuple[int, bytes]]


class UnsafeToReap(RuntimeError):
    """The listings could not be trusted, so nothing was reaped."""


class RunpodError(RuntimeError):
    pass


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: it would carry the Authorization header elsewhere."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], timeout_s: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers=dict(headers))
    try:
        with _OPENER.open(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read(MAX_RESPONSE_BYTES)
    except urllib.error.HTTPError as e:
        return int(e.code), e.read(MAX_RESPONSE_BYTES)


class RunpodRest:
    """The four REST calls the reaper makes. Not retried: the next run retries."""

    def __init__(
        self,
        api_key: str,
        *,
        rest_url: str = REST_URL,
        transport: Transport = urllib_transport,
        timeout_s: float = 20.0,
    ) -> None:
        if not api_key:
            raise ValueError("empty RunPod API key")
        self._key = api_key
        self.rest_url = rest_url.rstrip("/")
        self._transport = transport
        self._timeout_s = timeout_s

    def __repr__(self) -> str:
        return f"RunpodRest(rest_url={self.rest_url!r})"

    def _call(self, method: str, path: str, *, ok_404: bool = False) -> Any:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Accept": "application/json",
            "User-Agent": "loom-runpod-reaper",
        }
        try:
            status, body = self._transport(method, self.rest_url + path, headers, self._timeout_s)
        except Exception as e:
            raise RunpodError(f"{method} {path}: {self._redact(repr(e))}") from None
        if status == 404 and ok_404:
            return None
        if not 200 <= status < 300:
            detail = self._redact(body[:300].decode("utf-8", "replace"))
            raise RunpodError(f"{method} {path} failed ({status}): {detail}")
        if not body:
            return None
        try:
            return json.loads(body)
        except ValueError:
            raise RunpodError(f"{method} {path}: response is not JSON") from None

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***")

    def list_pods(self) -> Any:
        return self._call("GET", "/pods")

    def list_network_volumes(self) -> Any:
        return self._call("GET", "/networkvolumes")

    def delete_pod(self, pod_id: str) -> None:
        self._call("DELETE", f"/pods/{_checked_id(pod_id)}", ok_404=True)

    def delete_network_volume(self, volume_id: str) -> None:
        self._call("DELETE", f"/networkvolumes/{_checked_id(volume_id)}", ok_404=True)


def _checked_id(value: str) -> str:
    if not _ID_RE.match(value):
        raise ValueError(f"not a RunPod id: {value!r}")
    return value


def _check_prefix(prefix: str) -> str:
    if not _PREFIX_RE.match(prefix):
        raise ValueError(f"unsafe name prefix {prefix!r}")
    return prefix


def pod_name_re(prefix: str) -> re.Pattern[str]:
    return re.compile("^" + re.escape(_check_prefix(prefix)) + _TAIL)


def volume_name_re(prefix: str) -> re.Pattern[str]:
    return re.compile("^" + re.escape(_check_prefix(prefix)) + "-vol" + _TAIL)


def volume_name(prefix: str, experiment: str, ttl_at: datetime, nonce: str) -> str:
    """The name a Loom network volume must carry for this reaper to delete it."""
    name = f"{prefix}-vol-{experiment[:8]}-{int(ttl_at.timestamp())}-{nonce}"
    if not volume_name_re(prefix).match(name):
        raise ValueError(f"not a valid Loom volume name: {name!r}")
    return name


@dataclass(frozen=True)
class Target:
    kind: str  # "pod" or "volume"
    id: str
    name: str
    ttl: datetime

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id, "name": self.name, "ttl": self.ttl.isoformat()}


def _records(listing: Any, what: str) -> list[Mapping[str, Any]]:
    """Every element a dict with a string `id` and `name`, or UnsafeToReap."""
    if not isinstance(listing, list):
        raise UnsafeToReap(f"{what} listing is {type(listing).__name__}, not a list")
    for i, item in enumerate(listing):
        if not isinstance(item, Mapping):
            raise UnsafeToReap(f"{what}[{i}] is {type(item).__name__}, not an object")
        if not isinstance(item.get("id"), str) or not isinstance(item.get("name"), str):
            raise UnsafeToReap(f"{what}[{i}] has no string id and name")
    return listing


def _ttl(match: re.Match[str]) -> datetime:
    return datetime.fromtimestamp(int(match.group("ttl")), UTC)


def _expired_pods(
    pods: list[Mapping[str, Any]], now: datetime, prefix: str
) -> tuple[list[Target], int]:
    name_re = pod_name_re(prefix)
    targets: list[Target] = []
    kept = 0
    for pod in pods:
        m = name_re.match(pod["name"])
        if m is None or pod.get("desiredStatus") == "TERMINATED":
            continue
        env = pod.get("env")
        if not isinstance(env, Mapping):
            raise UnsafeToReap(f"pod {pod['name']} has no env object")
        if not _ID_RE.match(pod["id"]):
            raise UnsafeToReap(f"pod {pod['name']} has an unexpected id")
        if env.get(MANAGED_ENV) != "true":
            log.warning(
                "kept %s (%s): Loom-style name without %s=true", pod["id"], pod["name"], MANAGED_ENV
            )
            kept += 1
            continue
        ttl = _ttl(m)
        env_ttl = env.get(TTL_ENV)
        if env_ttl is not None and env_ttl != m.group("ttl"):
            log.warning("kept %s (%s): %s disagrees with the name", pod["id"], pod["name"], TTL_ENV)
            kept += 1
            continue
        if ttl >= now:
            kept += 1
            continue
        targets.append(Target("pod", pod["id"], pod["name"], ttl))
    return targets, kept


def _attached_volume_ids(pods: list[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for pod in pods:
        if isinstance(pod.get("networkVolumeId"), str):
            ids.add(pod["networkVolumeId"])
        nested = pod.get("networkVolume")
        if isinstance(nested, Mapping) and isinstance(nested.get("id"), str):
            ids.add(nested["id"])
    return ids


def _expired_volumes(
    volumes: list[Mapping[str, Any]],
    pods: list[Mapping[str, Any]],
    now: datetime,
    prefix: str,
) -> tuple[list[Target], int]:
    name_re = volume_name_re(prefix)
    attached = _attached_volume_ids(pods)
    targets: list[Target] = []
    kept = 0
    for vol in volumes:
        m = name_re.match(vol["name"])
        if m is None:
            continue
        if not _ID_RE.match(vol["id"]):
            raise UnsafeToReap(f"volume {vol['name']} has an unexpected id")
        ttl = _ttl(m)
        if ttl >= now:
            kept += 1
            continue
        if vol["id"] in attached:
            log.warning("kept volume %s (%s): attached to a pod", vol["id"], vol["name"])
            kept += 1
            continue
        targets.append(Target("volume", vol["id"], vol["name"], ttl))
    return targets, kept


def select(
    pods_listing: Any,
    volumes_listing: Any,
    now: datetime,
    *,
    prefix: str,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
) -> tuple[list[Target], int]:
    """(resources to reap, Loom resources kept). Raises UnsafeToReap on any surprise."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    pods = _records(pods_listing, "pods")
    volumes = _records(volumes_listing, "network volumes")
    pod_targets, pods_kept = _expired_pods(pods, now, prefix)
    vol_targets, vols_kept = _expired_volumes(volumes, pods, now, prefix)
    targets = pod_targets + vol_targets
    if len(targets) > max_per_run:
        raise UnsafeToReap(f"{len(targets)} resources qualify, above max_per_run={max_per_run}")
    return targets, pods_kept + vols_kept


def sweep(
    api: RunpodRest,
    now: datetime,
    *,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
    max_per_run: int = DEFAULT_MAX_PER_RUN,
) -> dict[str, Any]:
    """List, decide, then delete one resource per call so one failure cannot block the
    rest. Both listings are read before anything is deleted."""
    pods = api.list_pods()
    volumes = api.list_network_volumes()
    targets, kept = select(pods, volumes, now, prefix=prefix, max_per_run=max_per_run)
    reaped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for t in targets:
        if not dry_run:
            try:
                if t.kind == "pod":
                    api.delete_pod(t.id)
                else:
                    api.delete_network_volume(t.id)
            except Exception as e:
                log.error("reaping %s %s (%s) failed: %s", t.kind, t.id, t.name, e)
                failed.append(t.as_dict())
                continue
        log.info("reaped %s %s (%s, ttl %s) dry_run=%s", t.kind, t.id, t.name, t.ttl, dry_run)
        reaped.append(t.as_dict())
    log.info(
        "%d pods and %d volumes listed; %d Loom-named resources kept", len(pods), len(volumes), kept
    )
    return {"dry_run": dry_run, "reaped": reaped, "failed": failed, "kept": kept}


def _read_key(secret_id: str) -> str:
    import boto3  # type: ignore[import-untyped]

    value = boto3.client("secretsmanager").get_secret_value(SecretId=secret_id)
    key = str(value.get("SecretString") or "").strip()
    if not key:
        raise RuntimeError(f"secret {secret_id} holds no RunPod API key")
    return key


def lambda_handler(
    event: dict[str, Any] | None,
    context: Any,
    *,
    transport: Transport = urllib_transport,
    read_key: Callable[[str], str] = _read_key,
) -> dict[str, Any]:
    """An invoke can ask for a dry run (`{"dry_run": true}`), never force a live one."""
    logging.getLogger().setLevel(logging.INFO)
    event = event or {}
    dry_run = os.environ.get("LOOM_RUNPOD_REAPER_DRY_RUN", "true") != "false" or event.get(
        "dry_run"
    ) in (True, "true")
    prefix = _check_prefix(os.environ.get("LOOM_RUNPOD_REAPER_PREFIX", DEFAULT_PREFIX))
    max_per_run = int(os.environ.get("LOOM_RUNPOD_REAPER_MAX_PER_RUN", DEFAULT_MAX_PER_RUN))
    api = RunpodRest(read_key(os.environ["LOOM_RUNPOD_KEY_SECRET_ID"]), transport=transport)
    result = sweep(api, datetime.now(UTC), prefix=prefix, dry_run=dry_run, max_per_run=max_per_run)
    if result["failed"]:
        raise RuntimeError(f"failed to reap {len(result['failed'])}: {result['failed']}")
    return result
