from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from loom_bench.providers.runpod_reaper import created_at, is_managed, pod_ttl, reap

from .fakes import FakeRunpod

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
PAST = int((NOW - timedelta(minutes=5)).timestamp())
FUTURE = int((NOW + timedelta(hours=2)).timestamp())
PREFIX = "loom-bench"


def managed_env(ttl: int | None = None) -> dict[str, str]:
    env = {"LOOM_MANAGED": "true"}
    if ttl is not None:
        env["LOOM_TTL"] = str(ttl)
    return env


def name(ttl: int = FUTURE, nonce: str = "a1b2c3") -> str:
    return f"{PREFIX}-0123abcd-{ttl}-{nonce}"


def test_managed_needs_both_prefix_and_env() -> None:
    assert is_managed({"name": name(), "env": managed_env()}, PREFIX)
    assert not is_managed({"name": name(), "env": {}}, PREFIX)
    assert not is_managed({"name": name(), "env": {"LOOM_MANAGED": "false"}}, PREFIX)
    assert not is_managed({"name": "someone-else", "env": managed_env()}, PREFIX)
    assert not is_managed({"name": "loom-benchmark-x", "env": managed_env()}, PREFIX)
    assert not is_managed({"name": name(), "env": None}, PREFIX)


def test_ttl_from_env_then_name() -> None:
    assert pod_ttl({"name": name(FUTURE), "env": managed_env(PAST)}) == datetime.fromtimestamp(
        PAST, UTC
    )
    assert pod_ttl({"name": name(FUTURE), "env": managed_env()}) == datetime.fromtimestamp(
        FUTURE, UTC
    )
    assert pod_ttl({"name": "loom-bench-x", "env": {"LOOM_TTL": "soon"}}) is None


def test_created_at_parses_runpod_format() -> None:
    assert created_at({"createdAt": "2026-10-06 07:03:35.426 +0000 UTC"}) == datetime(
        2026, 10, 6, 7, 3, 35, 426000, tzinfo=UTC
    )
    assert created_at({"createdAt": "2026-10-06T07:03:35+00:00"}) == datetime(
        2026, 10, 6, 7, 3, 35, tzinfo=UTC
    )
    assert created_at({"createdAt": "yesterday"}) is None
    assert created_at({}) is None


def fleet(rp: FakeRunpod) -> dict[str, Any]:
    old = "2026-10-04 07:00:00.000 +0000 UTC"
    return {
        "expired_env": rp.add_pod(name=name(FUTURE, "aaaaaa"), env=managed_env(PAST)),
        "expired_name": rp.add_pod(name=name(PAST, "bbbbbb"), env=managed_env()),
        "live": rp.add_pod(name=name(FUTURE, "cccccc"), env=managed_env(FUTURE)),
        "exited": rp.add_pod(
            name=name(FUTURE, "dddddd"), env=managed_env(FUTURE), desiredStatus="EXITED"
        ),
        "old_no_ttl": rp.add_pod(name=f"{PREFIX}-x", env=managed_env(), createdAt=old),
        "young_no_ttl": rp.add_pod(
            name=f"{PREFIX}-y", env=managed_env(), createdAt="2026-10-06 11:00:00.000 +0000 UTC"
        ),
        "unknown_age": rp.add_pod(name=f"{PREFIX}-z", env=managed_env(), createdAt="?"),
        "foreign_same_prefix": rp.add_pod(name=name(PAST, "eeeeee"), env={}, createdAt=old),
        "foreign_env": rp.add_pod(name="other-1", env=managed_env(PAST), createdAt=old),
        "terminated": rp.add_pod(
            name=name(PAST, "ffffff"), env=managed_env(PAST), desiredStatus="TERMINATED"
        ),
    }


def test_reap_terminates_only_expired_managed_pods() -> None:
    rp = FakeRunpod()
    pods = fleet(rp)
    reaped = reap(rp.api(), NOW, prefix=PREFIX)
    expected = ["expired_env", "expired_name", "exited", "old_no_ttl", "unknown_age"]
    assert sorted(reaped) == sorted(pods[k]["id"] for k in expected)
    left = {k for k, p in pods.items() if p["id"] in rp.pods}
    assert left == {"live", "young_no_ttl", "foreign_same_prefix", "foreign_env", "terminated"}


def test_dry_run_changes_nothing() -> None:
    rp = FakeRunpod()
    pods = fleet(rp)
    reaped = reap(rp.api(), NOW, prefix=PREFIX, dry_run=True)
    assert len(reaped) == 5
    assert len(rp.pods) == len(pods)
    assert all(r.method == "GET" for r in rp.requests)


def test_max_age_applies_without_ttl() -> None:
    rp = FakeRunpod()
    pod = rp.add_pod(
        name=f"{PREFIX}-y", env=managed_env(), createdAt="2026-10-06 11:00:00.000 +0000 UTC"
    )
    assert reap(rp.api(), NOW, prefix=PREFIX, max_age=timedelta(minutes=30)) == [pod["id"]]


def test_one_failure_does_not_stop_the_rest() -> None:
    rp = FakeRunpod()
    first = rp.add_pod(name=name(PAST, "aaaaaa"), env=managed_env())
    second = rp.add_pod(name=name(PAST, "bbbbbb"), env=managed_env())
    rp.fail[("DELETE", f"/v1/pods/{first['id']}")] = [400]
    assert reap(rp.api(), NOW, prefix=PREFIX) == [second["id"]]
    assert first["id"] in rp.pods


def test_reap_refuses_unsafe_arguments() -> None:
    rp = FakeRunpod()
    with pytest.raises(ValueError, match="timezone"):
        reap(rp.api(), datetime(2026, 10, 6), prefix=PREFIX)
    with pytest.raises(ValueError, match="prefix"):
        reap(rp.api(), NOW, prefix="")
