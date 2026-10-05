"""TTL reaper for Loom-managed EC2 resources.

Terminates instances and deletes unattached volumes tagged `loom:managed=true`
whose `loom:ttl` (ISO-8601 UTC) has passed. A managed resource with a missing or
unparseable TTL is reaped once it is older than `max_age`.

Runs from `bench reap` and as the scheduled Lambda (`lambda_handler`). This file
imports only the standard library and boto3 so Terraform can zip it alone.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

MANAGED_TAG = "loom:managed"
TTL_TAG = "loom:ttl"
DEFAULT_MAX_AGE = timedelta(hours=24)
LIVE_STATES = ["pending", "running", "shutting-down", "stopping", "stopped"]

log = logging.getLogger(__name__)


def parse_ttl(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ttl = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ttl.tzinfo is None:
        return None
    return ttl.astimezone(UTC)


def _tags(resource: dict[str, Any]) -> dict[str, str]:
    return {t["Key"]: t["Value"] for t in resource.get("Tags", [])}


def is_expired(tags: dict[str, str], created: datetime, now: datetime, max_age: timedelta) -> bool:
    ttl = parse_ttl(tags.get(TTL_TAG))
    if ttl is not None:
        return ttl < now
    return created < now - max_age


def _managed_instances(ec2: Any) -> Iterator[dict[str, Any]]:
    pages = ec2.get_paginator("describe_instances").paginate(
        Filters=[
            {"Name": f"tag:{MANAGED_TAG}", "Values": ["true"]},
            {"Name": "instance-state-name", "Values": LIVE_STATES},
        ]
    )
    for page in pages:
        for reservation in page["Reservations"]:
            yield from reservation["Instances"]


def _managed_unattached_volumes(ec2: Any) -> Iterator[dict[str, Any]]:
    pages = ec2.get_paginator("describe_volumes").paginate(
        Filters=[
            {"Name": f"tag:{MANAGED_TAG}", "Values": ["true"]},
            {"Name": "status", "Values": ["available"]},
        ]
    )
    for page in pages:
        yield from page["Volumes"]


def reap(
    ec2_client: Any,
    now: datetime,
    *,
    dry_run: bool = False,
    max_age: timedelta = DEFAULT_MAX_AGE,
) -> list[str]:
    """Terminate expired managed instances and delete expired unattached volumes.

    Returns the ids acted on (or that would be, with `dry_run`). One API call per
    resource so a single failure cannot block the rest; failures are logged and
    the next run retries them.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    reaped: list[str] = []
    for inst in _managed_instances(ec2_client):
        if not is_expired(_tags(inst), inst["LaunchTime"], now, max_age):
            continue
        iid = inst["InstanceId"]
        if not dry_run:
            try:
                ec2_client.terminate_instances(InstanceIds=[iid])
            except Exception:
                log.exception("terminate %s failed", iid)
                continue
        log.info("reaped instance %s (dry_run=%s)", iid, dry_run)
        reaped.append(iid)
    for vol in _managed_unattached_volumes(ec2_client):
        if not is_expired(_tags(vol), vol["CreateTime"], now, max_age):
            continue
        vid = vol["VolumeId"]
        if not dry_run:
            try:
                ec2_client.delete_volume(VolumeId=vid)
            except Exception:
                log.exception("delete %s failed", vid)
                continue
        log.info("reaped volume %s (dry_run=%s)", vid, dry_run)
        reaped.append(vid)
    return reaped


def lambda_handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    import boto3  # type: ignore[import-untyped]

    logging.getLogger().setLevel(logging.INFO)
    event = event or {}
    dry_run = bool(event.get("dry_run", os.environ.get("LOOM_REAPER_DRY_RUN") == "true"))
    max_age = timedelta(hours=float(os.environ.get("LOOM_REAPER_MAX_AGE_HOURS", "24")))
    ec2 = boto3.client("ec2")
    ids = reap(ec2, datetime.now(UTC), dry_run=dry_run, max_age=max_age)
    return {"reaped": ids, "dry_run": dry_run}
