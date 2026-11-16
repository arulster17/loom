"""The reaper Lambda's serving rule (docs/phase1-m2-spec.md §6, §13 item 19).

Driven through `lambda_handler` against FakeRunpod (the Lambda's stdlib transport) and
moto's S3 for the lease records. The handler reads the real clock, so pod ages and lease
timestamps here are relative to `datetime.now(UTC)`, with wide margins around 2 hours.

Fixed by §5 and §6: lease record times are ISO 8601 with a UTC offset (as
`datetime.isoformat()` writes them); staleness is judged on `written_at` only; S3 errors
other than a missing key (a missing bucket included) and an enabled rule without a bucket
fail closed (raise, delete nothing, bench pods included); an unparseable record never
leads to a reap. Because the existing fail-closed rules apply to the union of both rules,
a serving-named pod with an odd id or without an env object makes the run raise, as a
bench-named one does.

Left open by the spec: whether a pod's age comes from its name's epoch or RunPod's
`createdAt`. Here the two always agree, so either source passes.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import socket
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
import pytest

from loom_bench.providers import runpod_reaper_lambda as rl
from loom_bench.providers.runpod_reaper_lambda import RunpodError, UnsafeToReap, lambda_handler

from .conftest import REGION
from .fakes import FAKE_KEY, REST, FakeRunpod
from .test_runpod_reaper_lambda import Override, deletes, raw_json, reaped_ids, transport

LEASE_BUCKET = "loom-serving-leases-test"
SERVING_NAME_RE = r"^loom-serve-[a-z0-9.-]+-\d{10}-[0-9a-f]{6}$"  # spec §6, verbatim
MODEL = "llama-3.1-8b"
SERVING_ENV = {
    "LOOM_MANAGED": "true",
    "LOOM_ROLE": "serving",
    "LOOM_MODEL": MODEL,
    "LOOM_LEASE_EPOCH": "1700000000",
    "LOOM_SSH_PUBKEY": "ssh-ed25519 AAAA test",
}
OLD = timedelta(hours=3)  # comfortably past the 2 hour rule
YOUNG = timedelta(hours=1)
STALE = timedelta(hours=3)
FRESH = timedelta(minutes=10)


def now() -> datetime:
    return datetime.now(UTC)


def runpod_time(at: datetime) -> str:
    """`createdAt` as RunPod writes it, e.g. `2026-10-06 07:03:35.426 +0000 UTC`."""
    return at.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S.000 +0000 UTC")


def serve_name(age: timedelta, nonce: str = "a1b2c3", model: str = MODEL) -> str:
    return f"loom-serve-{model}-{int((now() - age).timestamp())}-{nonce}"


def serving_pod(
    rp: FakeRunpod,
    age: timedelta,
    nonce: str,
    *,
    model: str = MODEL,
    name: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """A pod as the pool manager creates it: name epoch and `createdAt` both `age` ago."""
    record: dict[str, Any] = {
        "name": name if name is not None else serve_name(age, nonce, model),
        "env": dict(SERVING_ENV),
        "createdAt": runpod_time(now() - age),
        "desiredStatus": "RUNNING",
    }
    return rp.add_pod(**{**record, **fields})


def bench_pod(rp: FakeRunpod, ttl: datetime, nonce: str, **fields: Any) -> dict[str, Any]:
    """A bench pod (`loom-bench-<experiment>-<ttl>-<hex>`), created 3 hours ago."""
    record: dict[str, Any] = {
        "name": f"loom-bench-0123abcd-{int(ttl.timestamp())}-{nonce}",
        "env": {"LOOM_MANAGED": "true", "LOOM_EXPERIMENT": "0123abcd-x"},
        "createdAt": runpod_time(now() - OLD),
    }
    return rp.add_pod(**{**record, **fields})


def put_lease(
    s3: Any,
    pod: Mapping[str, Any],
    written_ago: timedelta,
    lease_until: datetime | None = None,
) -> None:
    written = now() - written_ago
    record = {
        "pod_id": pod["id"],
        "pod_name": pod["name"],
        "model": MODEL,
        "lease_until": (lease_until or written + timedelta(minutes=30)).isoformat(),
        "written_at": written.isoformat(),
    }
    s3.put_object(
        Bucket=LEASE_BUCKET, Key=f"serving/leases/{pod['id']}", Body=json.dumps(record).encode()
    )


def run(
    rp: FakeRunpod, event: dict[str, Any] | None = None, override: Override | None = None
) -> dict[str, Any]:
    send = transport(rp, override)

    def to_fake(method: str, url: str, headers: Mapping[str, str], timeout_s: float):
        return send(method, url.replace(rl.REST_URL, REST), headers, timeout_s)

    return lambda_handler(
        event if event is not None else {}, None, transport=to_fake, read_key=lambda _: FAKE_KEY
    )


def survives(call: Callable[[], Any]) -> dict[str, Any] | None:
    """Run the handler; a raise is an acceptable way to keep everything (fail closed)."""
    try:
        result: dict[str, Any] = call()
    except Exception:
        return None
    return result


def _clear_reaper_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LOOM_RUNPOD_REAPER_DRY_RUN",
        "LOOM_RUNPOD_REAPER_PREFIX",
        "LOOM_RUNPOD_REAPER_MAX_PER_RUN",
        "LOOM_RUNPOD_REAPER_SERVING",
        "LOOM_RUNPOD_REAPER_LEASE_BUCKET",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOOM_RUNPOD_KEY_SECRET_ID", "unused-read-key-is-injected")


def _enable_live_serving_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_SERVING", "true")
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_LEASE_BUCKET", LEASE_BUCKET)
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "false")


@pytest.fixture
def s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """moto S3 with the lease bucket; the serving rule enabled, live (not dry run)."""
    from moto import mock_aws

    _clear_reaper_env(monkeypatch)
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=LEASE_BUCKET)
        _enable_live_serving_rule(monkeypatch)
        yield client


# --- which serving pods ------------------------------------------------------------------


def test_reaps_old_serving_pods_with_a_missing_or_stale_lease_record(s3: Any) -> None:
    rp = FakeRunpod()
    missing = serving_pod(rp, OLD, "aaaaaa")
    stale = serving_pod(rp, OLD, "bbbbbb")
    put_lease(s3, stale, STALE)
    exited = serving_pod(rp, OLD, "cccccc", desiredStatus="EXITED")
    other_record = serving_pod(rp, OLD, "dddddd")
    # A fresh record under another pod's id protects only that pod.
    put_lease(s3, {"id": "someotherpod1", "name": serve_name(OLD, "eeeeee")}, FRESH)
    dotted = serving_pod(rp, OLD, "ffffff", model="qwen3-8b.v2")

    result = run(rp)

    reaped = {missing["id"], stale["id"], exited["id"], other_record["id"], dotted["id"]}
    assert reaped_ids(result) == reaped
    assert result["dry_run"] is False and result["failed"] == []
    assert not reaped & set(rp.pods)
    assert sorted(deletes(rp)) == sorted(f"/v1/pods/{i}" for i in reaped)


def test_keeps_a_fresh_lease_and_a_young_pod(s3: Any) -> None:
    rp = FakeRunpod()
    reapable = serving_pod(rp, OLD, "aaaaaa")
    fresh = serving_pod(rp, OLD, "bbbbbb")
    put_lease(s3, fresh, FRESH)
    young_without_record = serving_pod(rp, YOUNG, "cccccc")
    young_stale = serving_pod(rp, YOUNG, "dddddd")
    put_lease(s3, young_stale, STALE)

    result = run(rp)

    assert reaped_ids(result) == {reapable["id"]}
    assert {fresh["id"], young_without_record["id"], young_stale["id"]} <= set(rp.pods)
    assert deletes(rp) == [f"/v1/pods/{reapable['id']}"]


def test_staleness_is_judged_on_written_at_only(s3: Any) -> None:
    """§6: `written_at`, never `lease_until`, decides whether a record is stale."""
    rp = FakeRunpod()
    stale_far_lease = serving_pod(rp, OLD, "aaaaaa")
    put_lease(s3, stale_far_lease, STALE, lease_until=now() + timedelta(days=30))
    fresh_past_lease = serving_pod(rp, OLD, "bbbbbb")
    put_lease(s3, fresh_past_lease, FRESH, lease_until=now() - timedelta(hours=5))

    result = run(rp)

    assert reaped_ids(result) == {stale_far_lease["id"]}
    assert fresh_past_lease["id"] in rp.pods


JUST_UNDER = timedelta(hours=1, minutes=50)
JUST_OVER = timedelta(hours=2, minutes=10)


def test_the_two_hour_thresholds(s3: Any) -> None:
    rp = FakeRunpod()
    pod_under = serving_pod(rp, JUST_UNDER, "aaaaaa")  # no record
    pod_over = serving_pod(rp, JUST_OVER, "bbbbbb")  # no record
    record_under = serving_pod(rp, OLD, "cccccc")
    put_lease(s3, record_under, JUST_UNDER)
    record_over = serving_pod(rp, OLD, "dddddd")
    put_lease(s3, record_over, JUST_OVER)

    result = run(rp)

    assert reaped_ids(result) == {pod_over["id"], record_over["id"]}
    assert {pod_under["id"], record_under["id"]} <= set(rp.pods)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"LOOM_ROLE": "serving"},
        {"LOOM_MANAGED": "true"},
        {"LOOM_MANAGED": "false", "LOOM_ROLE": "serving"},
        {"LOOM_MANAGED": "TRUE", "LOOM_ROLE": "serving"},
        {"LOOM_MANAGED": "true", "LOOM_ROLE": "bench"},
        {"LOOM_MANAGED": "true", "LOOM_ROLE": "Serving"},
        {"LOOM_MANAGED": "true", "LOOM_ROLE": ""},
    ],
    ids=[
        "empty",
        "no-managed",
        "no-role",
        "managed-false",
        "managed-upper",
        "role-bench",
        "role-capitalised",
        "role-empty",
    ],
)
def test_keeps_a_serving_name_without_the_managed_serving_env(s3: Any, env: dict[str, str]) -> None:
    rp = FakeRunpod()
    reapable = serving_pod(rp, OLD, "aaaaaa")
    unmarked = serving_pod(rp, OLD, "bbbbbb", env=env)

    result = run(rp)

    assert reaped_ids(result) == {reapable["id"]}
    assert unmarked["id"] in rp.pods


def _not_serving_names() -> list[str]:
    epoch = int((now() - OLD).timestamp())
    good = f"loom-serve-{MODEL}-{epoch}-a1b2c3"
    names = [
        f"x-{good}",
        f"{good}-x",
        f"{good}\n",  # `$` alone would let a trailing newline through
        good.upper(),
        f"loom-serve-Llama-3.1-8b-{epoch}-a1b2c3",
        f"loom-serve-llama_3.1_8b-{epoch}-a1b2c3",
        f"loom-serve-llama/8b-{epoch}-a1b2c3",
        f"loom-serve--{epoch}-a1b2c3",
        f"loom-serve-{MODEL}-{epoch}-A1B2C3",
        f"loom-serve-{MODEL}-{epoch}-a1b2c",
        f"loom-serve-{MODEL}-{epoch}-a1b2c3d",
        f"loom-serve-{MODEL}-{str(epoch)[:9]}-a1b2c3",
        f"loom-serve-{MODEL}-{epoch}0-a1b2c3",
        f"loom-serve-{MODEL}-a1b2c3",
        "loom-serve",
        "loom-serve-",
        f"loom-serving-{MODEL}-{epoch}-a1b2c3",
        f"loom-bench-0123abcd-{epoch}-a1b2c3",  # a bench name: never the serving rule's
        "my-other-project",
    ]
    for n in names:  # the cases really are outside the spec's pattern
        assert re.fullmatch(SERVING_NAME_RE.removesuffix("$"), n) is None, n
    return names


def test_serving_rule_matches_its_exact_name_only(s3: Any) -> None:
    rp = FakeRunpod()
    reapable = serving_pod(rp, OLD, "aaaaaa")
    # Every one old, serving env, no lease record. The bench-named one carries LOOM_TTL in
    # the future so the bench TTL rule keeps it too.
    others = [
        serving_pod(rp, OLD, "000000", name=n, env={**SERVING_ENV, "LOOM_TTL": "9999999999"})
        for n in _not_serving_names()
    ]

    result = run(rp)

    assert reaped_ids(result) == {reapable["id"]}
    assert all(p["id"] in rp.pods for p in others)


def test_serving_rule_never_reaps_bench_pods(s3: Any) -> None:
    """Bench names with the serving env, old and without a lease record: kept unless the
    bench TTL rule itself says so."""
    rp = FakeRunpod()
    future = now() + timedelta(hours=2)
    benches = [
        bench_pod(rp, future, "aaaaaa", env={**SERVING_ENV}),
        bench_pod(rp, future, "bbbbbb"),
        bench_pod(rp, future, "cccccc", env={"LOOM_MANAGED": "true", "LOOM_ROLE": "serving"}),
    ]
    expired_bench = bench_pod(rp, now() - timedelta(minutes=5), "dddddd")

    result = run(rp)

    assert reaped_ids(result) == {expired_bench["id"]}  # the TTL rule still runs
    assert all(p["id"] in rp.pods for p in benches)


@pytest.mark.parametrize("prefix", ["loom-bench", "loom", "loom-serve"])
def test_bench_ttl_rule_never_reaps_serving_pods(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """With the serving rule off, serving pods whose name epoch is long past (short model
    names fit the bench name's 8-character experiment slot) are never the TTL rule's,
    whatever the prefix. Refusing the prefix outright is also fine."""
    _clear_reaper_env(monkeypatch)
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "false")
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_PREFIX", prefix)
    rp = FakeRunpod()
    pods = [
        serving_pod(rp, timedelta(days=3), "aaaaaa", model="x"),
        serving_pod(rp, timedelta(days=3), "bbbbbb", model="qwen3"),
        serving_pod(rp, timedelta(days=3), "cccccc", model="q-8b"),
        serving_pod(rp, timedelta(days=3), "dddddd", model=MODEL),
    ]

    survives(lambda: run(rp))

    assert deletes(rp) == []
    assert all(p["id"] in rp.pods for p in pods)


# --- enabling the rule ------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "false", ""], ids=["unset", "false", "empty"])
def test_rule_is_off_unless_serving_is_true(
    s3: Any, monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("LOOM_RUNPOD_REAPER_SERVING")
    else:
        monkeypatch.setenv("LOOM_RUNPOD_REAPER_SERVING", value)
    rp = FakeRunpod()
    pod = serving_pod(rp, OLD, "aaaaaa")

    result = run(rp)

    assert result["reaped"] == []
    assert pod["id"] in rp.pods


def test_rule_enabled_without_a_bucket_reaps_nothing(
    s3: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6: enabled without a bucket, the run fails closed (raises, deletes nothing)."""
    monkeypatch.delenv("LOOM_RUNPOD_REAPER_LEASE_BUCKET")
    rp = FakeRunpod()
    serving = serving_pod(rp, OLD, "aaaaaa")
    bench = bench_pod(rp, now() - timedelta(minutes=5), "bbbbbb")

    with pytest.raises(Exception):  # noqa: B017 - any refusal will do; nothing may be deleted
        run(rp)

    assert deletes(rp) == []
    assert {serving["id"], bench["id"]} <= set(rp.pods)


# --- fail closed: S3 ------------------------------------------------------------------------


class _S3Error(http.server.BaseHTTPRequestHandler):
    err_status = 403
    err_code = "AccessDenied"

    def _answer(self) -> None:
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            f"<Error><Code>{self.err_code}</Code><Message>injected</Message></Error>"
        ).encode()
        self.send_response(self.err_status)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = do_HEAD = do_PUT = do_POST = _answer

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def broken_s3(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Iterator[None]:
    """An S3 endpoint answering every request with `request.param` (status, error code)."""
    status, code = request.param
    handler = type("Handler", (_S3Error,), {"err_status": status, "err_code": code})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _clear_reaper_env(monkeypatch)
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    monkeypatch.setenv("AWS_RETRY_MODE", "standard")
    _enable_live_serving_rule(monkeypatch)
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    "broken_s3",
    [(403, "AccessDenied"), (500, "InternalError"), (503, "SlowDown")],
    ids=["access-denied", "internal-error", "slow-down"],
    indirect=True,
)
def test_an_s3_error_reaps_nothing_that_run(broken_s3: None) -> None:
    rp = FakeRunpod()
    serving = serving_pod(rp, OLD, "aaaaaa")
    bench = bench_pod(rp, now() - timedelta(minutes=5), "bbbbbb")

    with pytest.raises(Exception):  # noqa: B017 - a visible Lambda error, of any type
        run(rp)

    assert deletes(rp) == []
    assert {serving["id"], bench["id"]} <= set(rp.pods)


def test_an_unreachable_s3_reaps_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    _clear_reaper_env(monkeypatch)
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", f"http://127.0.0.1:{dead}")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    _enable_live_serving_rule(monkeypatch)
    rp = FakeRunpod()
    serving = serving_pod(rp, OLD, "aaaaaa")

    with pytest.raises(Exception):  # noqa: B017
        run(rp)

    assert deletes(rp) == []
    assert serving["id"] in rp.pods


def test_a_missing_bucket_is_not_a_missing_record(s3: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """NoSuchBucket (a misnamed bucket) must not read as "no lease record" for every pod."""
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_LEASE_BUCKET", "loom-no-such-bucket")
    rp = FakeRunpod()
    serving = serving_pod(rp, OLD, "aaaaaa")

    with pytest.raises(Exception):  # noqa: B017
        run(rp)

    assert deletes(rp) == []
    assert serving["id"] in rp.pods


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b"{}",
        json.dumps({"pod_id": "x", "written_at": "yesterday"}).encode(),
        json.dumps({"pod_id": "x", "written_at": None}).encode(),
    ],
    ids=["not-json", "list", "empty-object", "unparseable-written-at", "null-written-at"],
)
def test_an_unreadable_lease_record_never_reaps_its_pod(s3: Any, body: bytes) -> None:
    rp = FakeRunpod()
    pod = serving_pod(rp, OLD, "aaaaaa")
    s3.put_object(Bucket=LEASE_BUCKET, Key=f"serving/leases/{pod['id']}", Body=body)

    survives(lambda: run(rp))

    assert pod["id"] in rp.pods
    assert deletes(rp) == []


# --- fail closed: the listing ---------------------------------------------------------------


def _bad_listings() -> dict[str, Callable[[list[dict[str, Any]]], tuple[int, bytes]]]:
    name = serve_name(OLD, "999999")
    created = runpod_time(now() - OLD)
    return {
        "object instead of list": lambda real: raw_json({"pods": real}),
        "not json": lambda real: (200, b"<html>maintenance</html>"),
        "server error": lambda real: (500, b'{"error":"down"}'),
        "element without name": lambda real: raw_json([*real, {"id": "fakepod000099"}]),
        "serving pod without env": lambda real: raw_json(
            [*real, {"id": "fakepod000099", "name": name, "createdAt": created}]
        ),
        "serving pod with env list": lambda real: raw_json(
            [
                *real,
                {
                    "id": "fakepod000099",
                    "name": name,
                    "env": ["LOOM_MANAGED=true", "LOOM_ROLE=serving"],
                    "createdAt": created,
                },
            ]
        ),
        "serving pod with odd id": lambda real: raw_json(
            [
                *real,
                {"id": "../pods", "name": name, "env": dict(SERVING_ENV), "createdAt": created},
            ]
        ),
    }


@pytest.mark.parametrize("case", list(_bad_listings()))
def test_a_bad_listing_reaps_nothing(s3: Any, case: str) -> None:
    rp = FakeRunpod()
    serving = serving_pod(rp, OLD, "aaaaaa")
    bench = bench_pod(rp, now() - timedelta(minutes=5), "bbbbbb")
    real = list(rp.pods.values())
    make = _bad_listings()[case]

    def override(method: str, path: str) -> tuple[int, bytes] | None:
        return make(real) if (method, path) == ("GET", "/v1/pods") else None

    with pytest.raises((UnsafeToReap, RunpodError)):
        run(rp, override=override)

    assert deletes(rp) == []
    assert {serving["id"], bench["id"]} <= set(rp.pods)


# --- max_per_run and dry run over both rules --------------------------------------------------


def test_max_per_run_counts_both_rules_together(s3: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rp = FakeRunpod()
    expired = now() - timedelta(minutes=5)
    pods = [
        bench_pod(rp, expired, "aaaaaa"),
        bench_pod(rp, expired, "bbbbbb"),
        serving_pod(rp, OLD, "cccccc"),
        serving_pod(rp, OLD, "dddddd"),
    ]

    monkeypatch.setenv("LOOM_RUNPOD_REAPER_MAX_PER_RUN", "3")
    with pytest.raises(UnsafeToReap, match="max_per_run"):
        run(rp)
    assert deletes(rp) == []
    assert all(p["id"] in rp.pods for p in pods)

    monkeypatch.setenv("LOOM_RUNPOD_REAPER_MAX_PER_RUN", "4")
    result = run(rp)
    assert reaped_ids(result) == {p["id"] for p in pods}
    assert rp.pods == {}


def test_serving_rule_obeys_dry_run(s3: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rp = FakeRunpod()
    pod = serving_pod(rp, OLD, "aaaaaa")

    monkeypatch.delenv("LOOM_RUNPOD_REAPER_DRY_RUN")  # the default
    result = run(rp)
    assert result["dry_run"] is True and reaped_ids(result) == {pod["id"]}

    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "false")
    result = run(rp, {"dry_run": True})
    assert result["dry_run"] is True and reaped_ids(result) == {pod["id"]}

    assert pod["id"] in rp.pods
    assert {r.method for r in rp.requests} == {"GET"}


# --- boto3 only with the rule on --------------------------------------------------------------

_PROBE = r"""
import importlib.util, json, sys

path, pods = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("runpod_reaper_lambda", path)
rl = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rl  # as Lambda's own import would (dataclasses look it up)
spec.loader.exec_module(rl)
loaded_on_import = "boto3" in sys.modules or "botocore" in sys.modules

def transport(method, url, headers, timeout_s):
    if method == "GET" and url.endswith("/pods"):
        return 200, pods.encode()
    if method == "GET" and url.endswith("/networkvolumes"):
        return 200, b"[]"
    return 500, b"unexpected"

error = None
try:
    rl.lambda_handler({}, None, transport=transport, read_key=lambda _: "rpa_test")
except Exception as e:
    error = type(e).__name__
print(json.dumps({
    "on_import": loaded_on_import,
    "boto3": "boto3" in sys.modules,
    "botocore": "botocore" in sys.modules,
    "error": error,
}))
"""


def _probe(env: dict[str, str]) -> dict[str, Any]:
    """Load the Lambda file alone (as Terraform zips it) in a fresh interpreter, run the
    handler once over one old serving pod, and report what got imported."""
    pod = {
        "id": "fakepod000001",
        "name": serve_name(OLD),
        "env": dict(SERVING_ENV),
        "createdAt": runpod_time(now() - OLD),
        "desiredStatus": "RUNNING",
    }
    base = {k: v for k, v in os.environ.items() if not k.startswith(("LOOM_", "AWS_"))}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, rl.__file__, json.dumps([pod])],
        env={
            **base,
            "LOOM_RUNPOD_KEY_SECRET_ID": "unused",
            "LOOM_RUNPOD_REAPER_LEASE_BUCKET": LEASE_BUCKET,
            "AWS_ACCESS_KEY_ID": "testing",
            "AWS_SECRET_ACCESS_KEY": "testing",
            "AWS_DEFAULT_REGION": REGION,
            "AWS_MAX_ATTEMPTS": "1",
            "AWS_ENDPOINT_URL_S3": f"http://127.0.0.1:{dead}",  # closed: never AWS
            **env,
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


def test_with_the_rule_disabled_the_lambda_imports_no_boto3() -> None:
    off = _probe({})
    assert off == {"on_import": False, "boto3": False, "botocore": False, "error": None}
    assert _probe({"LOOM_RUNPOD_REAPER_SERVING": "false"})["boto3"] is False


def test_with_the_rule_enabled_the_lambda_uses_boto3() -> None:
    """The probe's control: the same run with the rule on needs S3, so boto3 is loaded
    (and the unreachable endpoint fails the run closed). A module-level import guarded by
    the setting is allowed, so when it loads is not checked."""
    on = _probe({"LOOM_RUNPOD_REAPER_SERVING": "true"})
    assert on["boto3"] is True
    assert on["error"] is not None
