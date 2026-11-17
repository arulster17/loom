"""Lease renewal, the manager's half (docs/phase1-m2-spec.md §13 item 17; §5 "Leases and
the watchdog"): every lease_renew_every_s the manager runs serve_lease over SSH (LEASE_FILE,
LEASE_EPOCH = now + lease_ttl_s), then writes the S3 record
`s3://<bucket>/serving/leases/<pod id>` (moto). A failed SSH renewal is retried each tick,
the pod stays in routing, and it is marked failed only when lease_until is within 2 renewal
periods. (serve_lease itself, run under bash, is test_poolmgr_scripts_lease.py.)

Timings: lease_ttl_s 600, lease_renew_every_s 60 (due at `>=`, §11), so "near lease end"
is the last 120 s. The wake key is gone after bring-up, so no replacement is made.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from typing import Any

import boto3
import pytest
from moto import mock_aws
from poolmgr_fakes import (
    FakePod,
    FakeServing,
    bring_up,
    env_fixture,  # noqa: F401
    serving_config,
    tick,
    unique_model_id,
)

from loom_platform.pool import wake_key
from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.bootstrap import LEASE_FILE
from loom_poolmgr.leases import S3LeaseRecords

BUCKET = "loom-serving-leases-test"


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for name, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(name, value)
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def record(s3: Any, pod_id: str) -> dict[str, Any]:
    body = s3.get_object(Bucket=BUCKET, Key=f"serving/leases/{pod_id}")["Body"].read()
    data: dict[str, Any] = json.loads(body)
    return data


def utc(text: str) -> dt.datetime:
    value = dt.datetime.fromisoformat(text)
    assert value.utcoffset() == dt.timedelta(0), text
    return value


async def row_of(env: FakeServing, pod: FakePod) -> dict[str, Any]:
    (row,) = [r for r in await env.rows(pod.model) if r["pod_name"] == pod.name]
    return row


async def test_renewal_over_ssh_and_s3_every_period(env: FakeServing, s3: Any) -> None:
    model = unique_model_id()
    config = serving_config([model], lifecycle={"lease_ttl_s": 600, "lease_renew_every_s": 60})
    manager = env.manager(config, leases=S3LeaseRecords(BUCKET, client=s3))
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.redis.delete(wake_key(model))

    # The renewal made during bootstrap.
    assert pod.leases, "no lease renewal over SSH"
    at, epoch = pod.leases[-1]
    assert abs(epoch - (at + 600)) <= 1
    for script in env.pods.scripts("serve_lease", pod.pod_id):
        assert f"LEASE_FILE={LEASE_FILE}" in script
    rec = record(s3, pod.pod_id)
    assert set(rec) == {"pod_id", "pod_name", "model", "lease_until", "written_at"}
    assert (rec["pod_id"], rec["pod_name"], rec["model"]) == (pod.pod_id, pod.name, model)
    assert rec["lease_until"].endswith("+00:00") and rec["written_at"].endswith("+00:00")
    assert abs(utc(rec["lease_until"]).timestamp() - epoch) <= 1
    assert abs(utc(rec["written_at"]).timestamp() - at) <= 1

    # Nothing before the period is up; one renewal (SSH, then S3) once it is.
    for _ in range(2):
        n = len(pod.leases)
        last_at = pod.leases[-1][0]
        if env.clock() < last_at + 50:
            env.clock.set(last_at + 50)
            await tick(manager, env.clock)
            assert len(pod.leases) == n, "renewed before lease_renew_every_s"
        env.clock.set(max(env.clock(), last_at + 60))
        await tick(manager, env.clock)
        assert len(pod.leases) == n + 1, "not renewed after lease_renew_every_s"
        at, epoch = pod.leases[-1]
        assert abs(at - env.clock()) <= 1 and abs(epoch - (env.clock() + 600)) <= 1
        rec = record(s3, pod.pod_id)
        assert abs(utc(rec["lease_until"]).timestamp() - epoch) <= 1
        assert abs(utc(rec["written_at"]).timestamp() - at) <= 1
        row = await row_of(env, pod)
        assert abs(row["lease_until"].timestamp() - epoch) <= 1
        assert row["state"] == "ready"


async def test_failing_renewal_fails_pod_only_near_lease_end(env: FakeServing) -> None:
    model = unique_model_id()
    config = serving_config([model], lifecycle={"lease_ttl_s": 600, "lease_renew_every_s": 60})
    manager = env.manager(config)
    await manager.start()
    pod = await bring_up(env, manager, model)
    await env.redis.delete(wake_key(model))
    last_ok, lease_until = pod.leases[-1]
    accepted = json.loads(env.leases.records[pod.pod_id])  # the record last written
    pod.behaviour.fail_lease = True

    # Retried every tick once due; the pod stays in routing while > 2 periods remain.
    for offset in range(60, 451, 30):
        attempts = len(pod.lease_attempts)
        env.clock.set(max(env.clock(), last_ok + offset))
        await tick(manager, env.clock)
        assert len(pod.lease_attempts) > attempts, f"no retry at +{offset}"
        pool = await env.pool(model)
        assert [r.id for r in pool.replicas] == [pod.pod_id], f"out of routing at +{offset}"
        row = await row_of(env, pod)
        assert row["state"] == "ready", f"{row['state']} at +{offset} ({600 - offset} s left)"
    assert env.runpod.has(pod.pod_id)
    # The record never claims a lease the pod did not accept.
    current = json.loads(env.leases.records[pod.pod_id])
    assert current["lease_until"] == accepted["lease_until"]

    # 60 s left, less than 2 renewal periods: failed (and terminated).
    env.clock.set(lease_until - 60)
    await tick(manager, env.clock)
    row = await row_of(env, pod)
    assert row["state"] in ("failed", "terminated"), row["state"]
    assert pod.pod_id not in [r.id for r in (await env.pool(model)).replicas]
