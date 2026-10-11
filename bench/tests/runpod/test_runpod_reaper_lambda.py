"""The scheduled RunPod reaper Lambda against FakeRunpod (through its stdlib transport)."""

import ast
import http.server
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest

from loom_bench.providers import runpod_reaper_lambda as rl
from loom_bench.providers.runpod_reaper_lambda import (
    RunpodError,
    RunpodRest,
    UnsafeToReap,
    lambda_handler,
    pod_name_re,
    select,
    sweep,
    urllib_transport,
    volume_name,
)

from .conftest import BUCKET, REGION
from .fakes import FAKE_KEY, REST, FakeRunpod

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
PAST = int((NOW - timedelta(minutes=5)).timestamp())
FUTURE = int((NOW + timedelta(hours=2)).timestamp())
PREFIX = "loom-bench"
ANCIENT = "2026-01-01 00:00:00.000 +0000 UTC"

Override = Callable[[str, str], tuple[int, bytes] | None]


def transport(rp: FakeRunpod, override: Override | None = None) -> rl.Transport:
    """The Lambda's transport, served by FakeRunpod; `override(method, path)` may answer
    first with a raw (status, body)."""

    def send(method: str, url: str, headers: Mapping[str, str], timeout_s: float):
        assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
        request = httpx.Request(method, url, headers=dict(headers))
        if override is not None:
            raw = override(method, request.url.path)
            if raw is not None:
                rp.requests.append(request)
                return raw
        resp = rp.handle(request)
        return resp.status_code, resp.content

    return send


def api(rp: FakeRunpod, override: Override | None = None) -> RunpodRest:
    return RunpodRest(FAKE_KEY, rest_url=REST, transport=transport(rp, override))


def name(ttl: int, nonce: str = "a1b2c3", experiment: str = "0123abcd") -> str:
    return f"{PREFIX}-{experiment}-{ttl}-{nonce}"


def loom_env(ttl: int | None) -> dict[str, str]:
    env = {"LOOM_MANAGED": "true", "LOOM_EXPERIMENT": "0123abcd-x", "LOOM_OWNER": "arul"}
    if ttl is not None:
        env["LOOM_TTL"] = str(ttl)
    return env


def loom_pod(rp: FakeRunpod, ttl: int, nonce: str, **kw: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {"name": name(ttl, nonce), "env": loom_env(ttl)}
    return rp.add_pod(**{**fields, **kw})


def deletes(rp: FakeRunpod) -> list[str]:
    return [r.url.path for r in rp.requests if r.method == "DELETE"]


def reaped_ids(result: dict[str, Any]) -> set[str]:
    return {r["id"] for r in result["reaped"]}


# --- which pods ----------------------------------------------------------------------


def test_expired_loom_pod_is_reaped_and_unexpired_kept() -> None:
    rp = FakeRunpod()
    expired = loom_pod(rp, PAST, "aaaaaa")
    no_env_ttl = loom_pod(rp, PAST, "bbbbbb", env=loom_env(None))  # TTL from the name
    exited = loom_pod(rp, PAST, "cccccc", desiredStatus="EXITED")
    live = loom_pod(rp, FUTURE, "dddddd")
    live_exited = loom_pod(rp, FUTURE, "eeeeee", desiredStatus="EXITED")
    gone = loom_pod(rp, PAST, "ffffff", desiredStatus="TERMINATED")

    result = sweep(api(rp), NOW, prefix=PREFIX)

    assert reaped_ids(result) == {expired["id"], no_env_ttl["id"], exited["id"]}
    assert result["failed"] == [] and result["dry_run"] is False
    assert result["kept"] == 2
    assert set(rp.pods) == {live["id"], live_exited["id"], gone["id"]}
    assert {r["name"] for r in result["reaped"]} == {
        expired["name"],
        no_env_ttl["name"],
        exited["name"],
    }
    assert {r["ttl"] for r in result["reaped"]} == {datetime.fromtimestamp(PAST, UTC).isoformat()}


def test_pods_that_are_not_loom_managed_are_never_touched() -> None:
    rp = FakeRunpod()
    reapable = loom_pod(rp, PAST, "aaaaaa")
    foreign = [
        # The user's other project, however old and whatever its env says.
        rp.add_pod(name="my-other-project", env={}, createdAt=ANCIENT),
        rp.add_pod(name="other-1", env=loom_env(PAST), createdAt=ANCIENT),
        rp.add_pod(name=f"x-{name(PAST, 'bbbbbb')}", env=loom_env(PAST)),
        # Loom's prefix but not the runner's exact name format.
        rp.add_pod(name=f"{PREFIX}-x", env=loom_env(PAST), createdAt=ANCIENT),
        rp.add_pod(name=f"{PREFIX}mark-0123abcd-{PAST}-cccccc", env=loom_env(PAST)),
        rp.add_pod(name=f"{PREFIX}-0123abcd-{PAST}-dddddd-x", env=loom_env(PAST)),
        rp.add_pod(name=f"{PREFIX}-0123abcd-{PAST}-DDDDDD", env=loom_env(PAST)),
        rp.add_pod(name=name(PAST, "eeeeee").upper(), env=loom_env(PAST)),
        # The exact format, but not marked managed.
        rp.add_pod(name=name(PAST, "ffffff"), env={}),
        rp.add_pod(name=name(PAST, "111111"), env={"LOOM_MANAGED": "false"}),
        rp.add_pod(name=name(PAST, "222222"), env={"LOOM_MANAGED": "TRUE"}),
        # Managed, but the env TTL disagrees with the name: ambiguous, kept.
        rp.add_pod(name=name(PAST, "333333"), env=loom_env(FUTURE)),
        rp.add_pod(name=name(PAST, "444444"), env={**loom_env(None), "LOOM_TTL": PAST}),
        # A volume-style name on a pod.
        rp.add_pod(name=f"{PREFIX}-vol-0123abcd-{PAST}-555555", env=loom_env(PAST)),
    ]

    result = sweep(api(rp), NOW, prefix=PREFIX)

    assert reaped_ids(result) == {reapable["id"]}
    assert deletes(rp) == [f"/v1/pods/{reapable['id']}"]
    assert all(p["id"] in rp.pods for p in foreign)


def test_a_custom_prefix_matches_only_its_own_pods() -> None:
    rp = FakeRunpod()
    default = loom_pod(rp, PAST, "aaaaaa")
    custom = rp.add_pod(name=f"loom-ci-0123abcd-{PAST}-bbbbbb", env=loom_env(PAST))
    assert reaped_ids(sweep(api(rp), NOW, prefix="loom-ci")) == {custom["id"]}
    assert default["id"] in rp.pods


@pytest.mark.parametrize("prefix", ["", "-", "Loom", "a/b", "loom.*", "x" * 65])
def test_unsafe_prefixes_are_refused(prefix: str) -> None:
    rp = FakeRunpod()
    loom_pod(rp, PAST, "aaaaaa")
    with pytest.raises(ValueError, match="prefix"):
        sweep(api(rp), NOW, prefix=prefix)
    assert deletes(rp) == []


async def test_pods_the_provider_creates_match_and_expire_at_their_ttl(tmp_path: Path) -> None:
    """Contract with RunpodProvider: the name and env it sets are what the reaper reads."""
    from moto import mock_aws

    from .test_runpod_provider import provider, request

    rp = FakeRunpod(gets_before_ssh=0)
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET)
        host = await provider(rp, s3, tmp_path).provision(request())
    (pod,) = rp.pods.values()
    assert pod_name_re(PREFIX).match(pod["name"])

    before = host.ttl_at - timedelta(seconds=1)
    after = host.ttl_at + timedelta(seconds=1)
    assert select(list(rp.pods.values()), [], before, prefix=PREFIX) == ([], 1)
    targets, _ = select(list(rp.pods.values()), [], after, prefix=PREFIX)
    assert [t.id for t in targets] == [host.host_id]


# --- network volumes -------------------------------------------------------------------


def test_expired_loom_volumes_are_deleted_and_nothing_else() -> None:
    rp = FakeRunpod()
    ttl_past = datetime.fromtimestamp(PAST, UTC)
    expired = rp.add_volume(name=volume_name(PREFIX, "0123abcd-e", ttl_past, "aaaaaa"))
    fresh = rp.add_volume(
        name=volume_name(PREFIX, "0123abcd", datetime.fromtimestamp(FUTURE, UTC), "bbbbbb")
    )
    attached = rp.add_volume(name=volume_name(PREFIX, "0123abcd", ttl_past, "cccccc"))
    nested = rp.add_volume(name=volume_name(PREFIX, "0123abcd", ttl_past, "dddddd"))
    rp.add_pod(name="someone-else", env={}, networkVolumeId=attached["id"])
    rp.add_pod(name="someone-else-2", env={}, networkVolume={"id": nested["id"]})
    foreign = [
        rp.add_volume(name="my-other-project-weights"),
        rp.add_volume(name=f"{PREFIX}-cache"),
        rp.add_volume(name=name(PAST, "eeeeee")),  # a pod-style name on a volume
        rp.add_volume(name=f"{PREFIX}-vol-0123abcd-{PAST}-ffffff-old"),
    ]

    result = sweep(api(rp), NOW, prefix=PREFIX)

    assert result["reaped"] == [
        {
            "kind": "volume",
            "id": expired["id"],
            "name": expired["name"],
            "ttl": ttl_past.isoformat(),
        }
    ]
    assert deletes(rp) == [f"/v1/networkvolumes/{expired['id']}"]
    assert set(rp.volumes) == {fresh["id"], attached["id"], nested["id"]} | {
        v["id"] for v in foreign
    }


def test_volume_name_round_trips_and_refuses_bad_parts() -> None:
    ttl = datetime.fromtimestamp(FUTURE, UTC)
    assert volume_name(PREFIX, "0123abcd-long", ttl, "a1b2c3") == (
        f"{PREFIX}-vol-0123abcd-{FUTURE}-a1b2c3"
    )
    with pytest.raises(ValueError):
        volume_name(PREFIX, "0123abcd", ttl, "NOTHEX")
    with pytest.raises(ValueError):
        volume_name(PREFIX, "/etc", ttl, "a1b2c3")


# --- fail closed --------------------------------------------------------------------------


def raw_json(value: Any) -> tuple[int, bytes]:
    return 200, json.dumps(value).encode()


MALFORMED_PODS = {
    "object instead of list": raw_json({"pods": []}),
    "null": raw_json(None),
    "empty body": (200, b""),
    "not json": (200, b"<html>maintenance</html>"),
    "string element": raw_json(["fakepod000001"]),
    "element without name": raw_json([{"id": "fakepod000009"}]),
    "element with null name": raw_json([{"id": "fakepod000009", "name": None}]),
    "element with numeric id": raw_json([{"id": 9, "name": "x"}]),
    "loom pod without env": raw_json([{"id": "fakepod000009", "name": name(PAST, "999999")}]),
    "loom pod with env list": raw_json(
        [{"id": "fakepod000009", "name": name(PAST, "999999"), "env": ["LOOM_MANAGED=true"]}]
    ),
    "loom pod with odd id": raw_json(
        [{"id": "../pods", "name": name(PAST, "999999"), "env": loom_env(PAST)}]
    ),
}


@pytest.mark.parametrize("raw", MALFORMED_PODS.values(), ids=MALFORMED_PODS.keys())
def test_a_malformed_pod_listing_reaps_nothing(raw: tuple[int, bytes]) -> None:
    rp = FakeRunpod()
    loom_pod(rp, PAST, "aaaaaa")
    rp.add_volume(name=volume_name(PREFIX, "0123abcd", datetime.fromtimestamp(PAST, UTC), "bbbbbb"))
    real = list(rp.pods.values())

    def override(method: str, path: str) -> tuple[int, bytes] | None:
        if (method, path) != ("GET", "/v1/pods"):
            return None
        if raw[1].startswith(b"[") and raw[0] == 200:
            # The bad element arrives alongside the real, reapable pod.
            return 200, json.dumps([*real, *json.loads(raw[1])]).encode()
        return raw

    with pytest.raises((UnsafeToReap, RunpodError)):
        sweep(api(rp, override), NOW, prefix=PREFIX)
    assert deletes(rp) == []


MALFORMED_VOLUMES = {
    "object instead of list": raw_json({"volumes": []}),
    "element without name": raw_json([{"id": "fakevol000009", "size": 10}]),
    "loom volume with odd id": raw_json(
        [{"id": "x/y", "name": f"{PREFIX}-vol-0123abcd-{PAST}-999999"}]
    ),
}


@pytest.mark.parametrize("raw", MALFORMED_VOLUMES.values(), ids=MALFORMED_VOLUMES.keys())
def test_a_malformed_volume_listing_reaps_no_pods_either(raw: tuple[int, bytes]) -> None:
    rp = FakeRunpod()
    loom_pod(rp, PAST, "aaaaaa")

    def override(method: str, path: str) -> tuple[int, bytes] | None:
        return raw if (method, path) == ("GET", "/v1/networkvolumes") else None

    with pytest.raises(UnsafeToReap):
        sweep(api(rp, override), NOW, prefix=PREFIX)
    assert deletes(rp) == []


def test_more_targets_than_the_breaker_allows_reaps_nothing() -> None:
    rp = FakeRunpod()
    for i in range(4):
        loom_pod(rp, PAST, f"aaaaa{i}")
    with pytest.raises(UnsafeToReap, match="max_per_run=3"):
        sweep(api(rp), NOW, prefix=PREFIX, max_per_run=3)
    assert deletes(rp) == []
    assert len(sweep(api(rp), NOW, prefix=PREFIX, max_per_run=4)["reaped"]) == 4


def test_select_refuses_a_naive_now() -> None:
    with pytest.raises(ValueError, match="timezone"):
        select([], [], datetime(2026, 10, 10), prefix=PREFIX)


# --- dry run --------------------------------------------------------------------------------


def test_dry_run_lists_what_it_would_reap_and_deletes_nothing() -> None:
    rp = FakeRunpod()
    pod = loom_pod(rp, PAST, "aaaaaa")
    vol = rp.add_volume(
        name=volume_name(PREFIX, "0123abcd", datetime.fromtimestamp(PAST, UTC), "bbbbbb")
    )
    result = sweep(api(rp), NOW, prefix=PREFIX, dry_run=True)
    assert result["dry_run"] is True
    assert reaped_ids(result) == {pod["id"], vol["id"]}
    assert {r.method for r in rp.requests} == {"GET"}
    assert pod["id"] in rp.pods and vol["id"] in rp.volumes


# --- API errors -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, status", [("/v1/pods", 500), ("/v1/pods", 401), ("/v1/networkvolumes", 503)]
)
def test_a_failed_listing_reaps_nothing(path: str, status: int) -> None:
    rp = FakeRunpod()
    loom_pod(rp, PAST, "aaaaaa")
    rp.fail[("GET", path)] = [status]
    with pytest.raises(RunpodError, match=str(status)):
        sweep(api(rp), NOW, prefix=PREFIX)
    assert deletes(rp) == []


def test_transport_errors_reap_nothing_and_never_show_the_key() -> None:
    def broken(method: str, url: str, headers: Mapping[str, str], timeout_s: float):
        raise TimeoutError(f"timed out sending {headers['Authorization']}")

    with pytest.raises(RunpodError) as e:
        sweep(RunpodRest(FAKE_KEY, rest_url=REST, transport=broken), NOW, prefix=PREFIX)
    assert FAKE_KEY not in str(e.value) and "***" in str(e.value)


def test_error_bodies_are_redacted() -> None:
    rp = FakeRunpod()

    def override(method: str, path: str) -> tuple[int, bytes] | None:
        return 403, f"bad key {FAKE_KEY}".encode()

    with pytest.raises(RunpodError) as e:
        sweep(api(rp, override), NOW, prefix=PREFIX)
    assert FAKE_KEY not in str(e.value)


def test_one_failed_delete_does_not_stop_the_rest() -> None:
    rp = FakeRunpod()
    first = loom_pod(rp, PAST, "aaaaaa")
    second = loom_pod(rp, PAST, "bbbbbb")
    rp.fail[("DELETE", f"/v1/pods/{first['id']}")] = [500]
    result = sweep(api(rp), NOW, prefix=PREFIX)
    assert reaped_ids(result) == {second["id"]}
    assert [f["id"] for f in result["failed"]] == [first["id"]]
    assert first["id"] in rp.pods


def test_a_pod_already_gone_counts_as_reaped() -> None:
    rp = FakeRunpod()
    pod = loom_pod(rp, PAST, "aaaaaa")
    rp.fail[("DELETE", f"/v1/pods/{pod['id']}")] = [404]
    assert reaped_ids(sweep(api(rp), NOW, prefix=PREFIX)) == {pod["id"]}


def test_redirects_are_not_followed() -> None:
    """A redirect would carry the Authorization header to another host."""
    seen: list[tuple[str, str | None]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        client = RunpodRest(FAKE_KEY, rest_url=url, transport=urllib_transport, timeout_s=5)
        with pytest.raises(RunpodError, match="302"):
            client.list_pods()
    finally:
        server.shutdown()
        server.server_close()
    assert seen == [("/v1/pods", f"Bearer {FAKE_KEY}")]


# --- the Lambda handler --------------------------------------------------------------------


@pytest.fixture
def handler_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    from moto import mock_aws

    with mock_aws():
        sm = boto3.client("secretsmanager", region_name=REGION)
        arn = sm.create_secret(Name="loom/runpod-reaper-api-key", SecretString=FAKE_KEY + "\n")[
            "ARN"
        ]
        monkeypatch.setenv("LOOM_RUNPOD_KEY_SECRET_ID", arn)
        for var in (
            "LOOM_RUNPOD_REAPER_DRY_RUN",
            "LOOM_RUNPOD_REAPER_PREFIX",
            "LOOM_RUNPOD_REAPER_MAX_PER_RUN",
        ):
            monkeypatch.delenv(var, raising=False)
        yield arn


def run_handler(rp: FakeRunpod, event: dict[str, Any] | None) -> dict[str, Any]:
    send = transport(rp)

    def to_fake(method: str, url: str, headers: Mapping[str, str], timeout_s: float):
        return send(method, url.replace(rl.REST_URL, REST), headers, timeout_s)

    return lambda_handler(event, None, transport=to_fake)


def test_handler_defaults_to_dry_run(handler_env: str) -> None:
    rp = FakeRunpod()
    pod = loom_pod(rp, PAST, "aaaaaa")
    result = run_handler(rp, None)
    assert result["dry_run"] is True and reaped_ids(result) == {pod["id"]}
    assert pod["id"] in rp.pods


def test_handler_reaps_when_live_and_an_invoke_can_only_ask_for_a_dry_run(
    handler_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rp = FakeRunpod()
    pod = loom_pod(rp, PAST, "aaaaaa")
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "true")
    assert run_handler(rp, {"dry_run": False})["dry_run"] is True
    assert run_handler(rp, {"dry_run": "false"})["dry_run"] is True
    assert pod["id"] in rp.pods

    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "false")
    assert run_handler(rp, {"dry_run": True})["dry_run"] is True
    assert pod["id"] in rp.pods
    result = run_handler(rp, {})
    assert result["dry_run"] is False and reaped_ids(result) == {pod["id"]}
    assert pod["id"] not in rp.pods


def test_handler_takes_prefix_and_breaker_from_env(
    handler_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rp = FakeRunpod()
    loom_pod(rp, PAST, "aaaaaa")
    loom_pod(rp, PAST, "bbbbbb")
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_MAX_PER_RUN", "1")
    with pytest.raises(UnsafeToReap):
        run_handler(rp, {})
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_MAX_PER_RUN", "10")
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_PREFIX", "loom-other")
    assert run_handler(rp, {})["reaped"] == []
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_PREFIX", "")
    with pytest.raises(ValueError, match="prefix"):
        run_handler(rp, {})


def test_handler_raises_when_a_delete_failed(
    handler_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rp = FakeRunpod()
    pod = loom_pod(rp, PAST, "aaaaaa")
    rp.fail[("DELETE", f"/v1/pods/{pod['id']}")] = [500]
    monkeypatch.setenv("LOOM_RUNPOD_REAPER_DRY_RUN", "false")
    with pytest.raises(RuntimeError, match="failed to reap 1"):
        run_handler(rp, {})


def test_handler_without_a_secret_value_reaps_nothing(
    handler_env: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terraform creates the secret empty; until the key is put, runs fail and reap nothing."""
    boto3.client("secretsmanager", region_name=REGION).create_secret(Name="loom/empty")
    monkeypatch.setenv("LOOM_RUNPOD_KEY_SECRET_ID", "loom/empty")
    rp = FakeRunpod()
    loom_pod(rp, PAST, "aaaaaa")
    with pytest.raises(Exception, match=r"AWSCURRENT|ResourceNotFound|holds no"):
        run_handler(rp, {})
    assert rp.requests == []


def test_module_imports_only_stdlib_and_boto3() -> None:
    tree = ast.parse(Path(rl.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    stdlib = {"__future__", "json", "logging", "os", "re", "urllib", "collections"}
    assert names <= stdlib | {"dataclasses", "datetime", "typing", "boto3"}
