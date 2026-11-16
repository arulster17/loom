"""Reaper hardening from the wave 2b review (docs/phase1-m2-spec.md §6).

- A serving pod's age comes from its name's epoch and RunPod's `createdAt`. A
  `createdAt` that is present but cannot be parsed never leads to a reap (the pod is kept
  or the run raises); a missing `createdAt` falls back to the name's epoch.
- The volume TTL rule, like the pod TTL rule, never matches a `loom-serve-` name, whatever
  `LOOM_RUNPOD_REAPER_PREFIX` is (refusing such a prefix also passes).

Driven through `lambda_handler` as in test_runpod_reaper_serving.py, whose helpers it uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import boto3
import pytest

from .conftest import REGION
from .fakes import FakeRunpod
from .test_runpod_reaper_serving import (
    LEASE_BUCKET,
    MODEL,
    OLD,
    YOUNG,
    _clear_reaper_env,
    _enable_live_serving_rule,
    now,
    reaped_ids,
    run,
    serving_pod,
    survives,
)


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """moto S3 with an empty lease bucket; the serving rule enabled, live."""
    from moto import mock_aws

    _clear_reaper_env(monkeypatch)
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=LEASE_BUCKET)
        _enable_live_serving_rule(monkeypatch)
        yield client


def deleted(rp: FakeRunpod, kind: str) -> list[str]:
    return [
        r.url.path
        for r in rp.requests
        if r.method == "DELETE" and r.url.path.startswith(f"/v1/{kind}/")
    ]


# ---------------------------------------------------------------- createdAt


@pytest.mark.parametrize(
    "created_at",
    ["yesterday", "2026-13-45 99:99:99.000 +0000 UTC", "not a date", ["2026-01-01"], {"a": 1}],
    ids=["word", "impossible-date", "text", "list", "object"],
)
def test_an_unparseable_created_at_is_never_reaped(s3: Any, created_at: Any) -> None:
    """The name's epoch says 3 hours old and there is no lease record, but RunPod's own
    age cannot be read: never a delete."""
    rp = FakeRunpod()
    pod = serving_pod(rp, OLD, "aaaaaa", createdAt=created_at)

    survives(lambda: run(rp))

    assert pod["id"] in rp.pods
    assert deleted(rp, "pods") == []


def test_a_missing_created_at_falls_back_to_the_name_epoch(s3: Any) -> None:
    rp = FakeRunpod()
    old = serving_pod(rp, OLD, "aaaaaa")
    young = serving_pod(rp, YOUNG, "bbbbbb")
    for pod in (old, young):
        del pod["createdAt"]  # the listing carries no createdAt at all

    result = run(rp)

    assert reaped_ids(result) == {old["id"]}
    assert young["id"] in rp.pods


# ---------------------------------------------------------------- volumes


@pytest.mark.parametrize("prefix", ["loom-bench", "loom", "loom-serve", "loom-serve-x"])
def test_volume_ttl_rule_never_reaps_serving_named_volumes(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    _clear_reaper_env(monkeypatch)
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "false")
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_PREFIX", prefix)
    past = int((now() - timedelta(days=2)).timestamp())
    names = {
        f"loom-serve-vol-0123abcd-{past}-aaaaaa",
        f"loom-serve-x-vol-0123abcd-{past}-bbbbbb",
        f"loom-serve-{MODEL}-vol-0123abcd-{past}-cccccc",
        f"loom-serve-vol-x-{past}-dddddd",
    }
    own = f"{prefix}-vol-0123abcd-{past}-eeeeee"  # this prefix's own volume name
    if own.startswith("loom-serve-"):
        names.add(own)
    rp = FakeRunpod()
    serving = [rp.add_volume(name=n) for n in sorted(names)]
    control = None if own in names else rp.add_volume(name=own)

    result = survives(lambda: run(rp))

    assert all(v["id"] in rp.volumes for v in serving)
    if control is not None:  # the rule still works for this prefix's own volumes
        assert result is not None and reaped_ids(result) == {control["id"]}
        assert deleted(rp, "networkvolumes") == [f"/v1/networkvolumes/{control['id']}"]
    else:
        assert deleted(rp, "networkvolumes") == []
