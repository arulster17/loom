import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from loom_bench.providers import aws_reaper
from loom_bench.providers.aws_reaper import is_expired, lambda_handler, parse_ttl, reap

from .conftest import launch, state

NOW = datetime.now(UTC)
PAST = (NOW - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
FUTURE = (NOW + timedelta(hours=2)).isoformat().replace("+00:00", "Z")


def managed(ttl: str | None) -> dict[str, str]:
    tags = {"loom:managed": "true", "loom:experiment": "e1"}
    if ttl is not None:
        tags["loom:ttl"] = ttl
    return tags


def volume(ec2: Any, tags: dict[str, str]) -> str:
    resp = ec2.create_volume(
        AvailabilityZone="us-east-1a",
        Size=1,
        TagSpecifications=[
            {"ResourceType": "volume", "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}
        ],
    )
    return str(resp["VolumeId"])


def volume_ids(ec2: Any) -> set[str]:
    return {v["VolumeId"] for v in ec2.describe_volumes()["Volumes"]}


def test_parse_ttl() -> None:
    assert parse_ttl("2026-10-04T12:00:00Z") == datetime(2026, 10, 4, 12, tzinfo=UTC)
    assert parse_ttl("2026-10-04T14:00:00+02:00") == datetime(2026, 10, 4, 12, tzinfo=UTC)
    assert parse_ttl("2026-10-04T12:00:00") is None
    assert parse_ttl("soon") is None
    assert parse_ttl(None) is None


def test_is_expired_falls_back_to_age() -> None:
    created = NOW - timedelta(hours=3)
    assert is_expired({"loom:ttl": PAST}, NOW, NOW, timedelta(hours=24))
    assert not is_expired({"loom:ttl": FUTURE}, created, NOW, timedelta(hours=1))
    assert not is_expired({"loom:ttl": "garbage"}, created, NOW, timedelta(hours=24))
    assert is_expired({"loom:ttl": "garbage"}, created, NOW, timedelta(hours=1))
    assert is_expired({}, created, NOW, timedelta(hours=1))


def test_reap_terminates_only_expired_managed(ec2: Any) -> None:
    expired = launch(ec2, managed(PAST))
    unexpired = launch(ec2, managed(FUTURE))
    unmanaged = launch(ec2, {"loom:ttl": PAST})
    no_ttl = launch(ec2, managed(None))
    garbled = launch(ec2, managed("tomorrow"))

    assert reap(ec2, NOW) == [expired]
    assert state(ec2, expired) == "terminated"
    for iid in (unexpired, unmanaged, no_ttl, garbled):
        assert state(ec2, iid) == "running"

    later = NOW + timedelta(hours=25)
    assert set(reap(ec2, later)) == {unexpired, no_ttl, garbled}
    assert state(ec2, unmanaged) == "running"


def test_reap_dry_run_changes_nothing(ec2: Any) -> None:
    iid = launch(ec2, managed(PAST))
    vid = volume(ec2, managed(PAST))
    assert reap(ec2, NOW, dry_run=True) == [iid, vid]
    assert state(ec2, iid) == "running"
    assert vid in volume_ids(ec2)


def test_reap_deletes_expired_unattached_volumes(ec2: Any) -> None:
    expired = volume(ec2, managed(PAST))
    fresh = volume(ec2, managed(FUTURE))
    unmanaged = volume(ec2, {"loom:ttl": PAST})
    assert reap(ec2, NOW) == [expired]
    assert volume_ids(ec2) >= {fresh, unmanaged}
    assert expired not in volume_ids(ec2)


def test_reap_requires_aware_now(ec2: Any) -> None:
    with pytest.raises(ValueError):
        reap(ec2, datetime(2026, 10, 4))


def test_reap_continues_after_a_failure(ec2: Any) -> None:
    first = launch(ec2, managed(PAST))
    second = launch(ec2, managed(PAST))

    class Flaky:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def terminate_instances(self, InstanceIds: list[str]) -> Any:
            if InstanceIds == [first]:
                raise RuntimeError("denied")
            return self.inner.terminate_instances(InstanceIds=InstanceIds)

    assert reap(Flaky(ec2), NOW) == [second]
    assert state(ec2, first) == "running"


def test_lambda_handler(ec2: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    iid = launch(ec2, managed(PAST))
    monkeypatch.setenv("LOOM_REAPER_DRY_RUN", "true")
    assert lambda_handler({}, None) == {"reaped": [iid], "dry_run": True}
    assert state(ec2, iid) == "running"
    assert lambda_handler({"dry_run": False}, None) == {"reaped": [iid], "dry_run": False}
    assert state(ec2, iid) == "terminated"


def test_reaper_module_imports_only_stdlib_and_boto3() -> None:
    tree = ast.parse(Path(aws_reaper.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= {"__future__", "logging", "os", "collections", "datetime", "typing", "boto3"}
