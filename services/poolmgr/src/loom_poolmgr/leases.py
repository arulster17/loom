"""Pod leases (docs/phase1-m2-spec.md §5 "Leases and the watchdog", §6).

Every `lease_renew_every_s` the manager renews each live pod's lease in two places:

1. on the pod, over SSH: `serve_lease` rendered with `LEASE_FILE` and the new
   `LEASE_EPOCH` (now + `lease_ttl_s`), which the pod's watchdog reads;
2. in a lease record, `s3://<LOOM_POOLMGR_LEASE_BUCKET>/serving/leases/<pod id>`, JSON
   `{pod_id, pod_name, model, lease_until, written_at}` with both times ISO 8601 with a
   UTC offset (`datetime.isoformat()` of an aware UTC datetime). The serving reaper rule
   judges staleness on `written_at` only.

A failed SSH renewal is retried each tick; the pod stays in routing, and the manager
marks it `failed` only when `lease_until` is within 2 renewal periods.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Protocol

from loom_bench.providers.runpod_ssh import RemoteCommandError, SshExec, SshTarget
from loom_poolmgr.bootstrap import lease_script

LEASE_PREFIX = "serving/leases/"


def lease_key(pod_id: str) -> str:
    """The S3 key of a pod's lease record."""
    if not pod_id or "/" in pod_id:
        raise ValueError(f"bad pod id {pod_id!r}")
    return f"{LEASE_PREFIX}{pod_id}"


def _aware(value: dt.datetime, name: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(dt.UTC)


@dataclass(frozen=True)
class LeaseRecord:
    pod_id: str
    pod_name: str
    model: str
    lease_until: dt.datetime  # aware, UTC
    written_at: dt.datetime  # aware, UTC

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_until", _aware(self.lease_until, "lease_until"))
        object.__setattr__(self, "written_at", _aware(self.written_at, "written_at"))

    def to_json(self) -> str:
        return json.dumps(
            {
                "pod_id": self.pod_id,
                "pod_name": self.pod_name,
                "model": self.model,
                "lease_until": self.lease_until.isoformat(),
                "written_at": self.written_at.isoformat(),
            }
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> LeaseRecord:
        """Raises ValueError (or KeyError/TypeError) for a record that cannot be parsed."""
        d: Any = json.loads(text)
        return cls(
            pod_id=str(d["pod_id"]),
            pod_name=str(d["pod_name"]),
            model=str(d["model"]),
            lease_until=dt.datetime.fromisoformat(d["lease_until"]),
            written_at=dt.datetime.fromisoformat(d["written_at"]),
        )


class LeaseRecords(Protocol):
    """Where lease records live (S3 in production, memory in tests)."""

    async def put(self, record: LeaseRecord) -> None:
        """Write (replace) the pod's record. Raises on failure."""
        ...

    async def get(self, pod_id: str) -> LeaseRecord | None:
        """The pod's record, or None when there is none."""
        ...

    async def delete(self, pod_id: str) -> None:
        """Remove the pod's record (after it is terminated); a missing record is fine."""
        ...


class MemoryLeaseRecords:
    """`LeaseRecords` in a dict, for tests: `records` maps pod id to the JSON written,
    `fail` makes every call raise."""

    def __init__(self) -> None:
        self.records: dict[str, str] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("lease records unavailable (MemoryLeaseRecords.fail)")

    async def put(self, record: LeaseRecord) -> None:
        self._check()
        self.records[record.pod_id] = record.to_json()

    async def get(self, pod_id: str) -> LeaseRecord | None:
        self._check()
        text = self.records.get(pod_id)
        return None if text is None else LeaseRecord.from_json(text)

    async def delete(self, pod_id: str) -> None:
        self._check()
        self.records.pop(pod_id, None)


class S3LeaseRecords:
    """`LeaseRecords` in S3 (`LOOM_POOLMGR_LEASE_BUCKET`). `client` is a boto3 S3 client
    (default: `boto3.client("s3")`); calls run in a thread."""

    def __init__(self, bucket: str, *, client: Any = None) -> None:
        if not bucket:
            raise ValueError("empty lease bucket")
        if client is None:
            import boto3  # type: ignore[import-untyped]

            client = boto3.client("s3")
        self.bucket = bucket
        self.client = client

    async def put(self, record: LeaseRecord) -> None:
        await asyncio.to_thread(
            self.client.put_object,
            Bucket=self.bucket,
            Key=lease_key(record.pod_id),
            Body=record.to_json().encode(),
            ContentType="application/json",
        )

    async def get(self, pod_id: str) -> LeaseRecord | None:
        def read() -> bytes | None:
            try:
                out = self.client.get_object(Bucket=self.bucket, Key=lease_key(pod_id))
            except self.client.exceptions.NoSuchKey:
                return None
            body: bytes = out["Body"].read()
            return body

        body = await asyncio.to_thread(read)
        return None if body is None else LeaseRecord.from_json(body)

    async def delete(self, pod_id: str) -> None:
        await asyncio.to_thread(
            self.client.delete_object, Bucket=self.bucket, Key=lease_key(pod_id)
        )


async def renew_on_pod(
    pod_exec: SshExec, target: SshTarget, lease_epoch: int, *, timeout_s: float
) -> None:
    """Run serve_lease (LEASE_FILE, LEASE_EPOCH = `lease_epoch`) over SSH. Raises
    `RemoteCommandError` on a non-zero exit or an unreachable pod."""
    res = await pod_exec.run(target, lease_script(lease_epoch), timeout_s=timeout_s)
    if res.returncode != 0:
        raise RemoteCommandError("serve_lease", res.returncode, res.stdout, res.stderr)
    if f"loom-lease {int(lease_epoch)}" not in res.stdout.split("\n"):
        raise RemoteCommandError("serve_lease: no loom-lease line", 0, res.stdout, res.stderr)
