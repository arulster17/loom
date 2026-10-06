import json
import os
import subprocess
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from loom_bench.providers import runpod_api
from loom_bench.providers.runpod_api import (
    RunpodApi,
    RunpodApiError,
    check_pod_id,
    load_runpod_api_key,
    public_ssh,
    usd_to_micros,
)

from .fakes import FAKE_KEY, FakeRunpod, create_response_fixture

# Captured before the suite-wide fixture swaps it out, to test it with subprocess faked.
REAL_KEYCHAIN_LOOKUP = runpod_api._keychain_lookup


def test_suite_never_sees_a_real_key() -> None:
    assert runpod_api.KEY_ENV not in os.environ
    assert load_runpod_api_key() is None


def test_create_get_list_and_delete() -> None:
    rp = FakeRunpod()
    api = rp.api()
    pod = api.create_pod({"name": "loom-bench-x", "gpuCount": 1, "env": {"LOOM_MANAGED": "true"}})
    assert pod["id"].startswith("fakepod")
    assert pod["costPerHr"] == 1.09
    request = rp.requests[0]
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in str(request.url)
    assert rp.bodies("POST", "/v1/pods") == [
        {"name": "loom-bench-x", "gpuCount": 1, "env": {"LOOM_MANAGED": "true"}}
    ]
    assert api.get_pod(pod["id"]) == pod
    assert [p["id"] for p in api.list_pods()] == [pod["id"]]
    assert api.find_pods_by_name("loom-bench-x") == [pod]
    assert api.find_pods_by_name("other") == []
    api.delete_pod(pod["id"])
    assert api.get_pod(pod["id"]) is None
    api.delete_pod(pod["id"])  # already gone is fine


def test_fixture_pins_the_fields_the_provider_reads() -> None:
    pod = create_response_fixture()
    assert {"id", "name", "costPerHr", "desiredStatus", "env", "createdAt", "machine"} <= set(pod)
    assert pod["machine"]["dataCenterId"] == "EUR-IS-2"
    assert pod["volumeInGb"] == 0
    assert usd_to_micros(pod["costPerHr"]) == 1_090_000


def test_terminate_falls_back_to_graphql_when_rest_refuses_the_key() -> None:
    rp = FakeRunpod()
    pod = rp.add_pod()
    rp.rest_delete_status = 403
    rp.api().terminate_pod(pod["id"])
    assert pod["id"] not in rp.pods
    (body,) = rp.bodies("POST", "/graphql")
    assert "podTerminate" in body["query"]
    assert body["variables"] == {"podId": pod["id"]}
    assert all(r.url.path.endswith(("/graphql", pod["id"])) for r in rp.requests)


def test_terminate_prefers_rest_and_reraises_other_errors() -> None:
    rp = FakeRunpod()
    pod = rp.add_pod()
    rp.api().terminate_pod(pod["id"])
    assert pod["id"] not in rp.pods
    assert rp.bodies("POST", "/graphql") == []
    rp.rest_delete_status = 400
    other = rp.add_pod()
    with pytest.raises(RunpodApiError) as e:
        rp.api().terminate_pod(other["id"])
    assert e.value.status == 400


def test_idempotent_requests_retry_but_create_does_not() -> None:
    rp = FakeRunpod()
    pod = rp.add_pod()
    rp.fail[("GET", f"/v1/pods/{pod['id']}")] = [503, 502]
    assert rp.api().get_pod(pod["id"]) == pod
    rp.fail[("POST", "/v1/pods")] = [503]
    with pytest.raises(RunpodApiError) as e:
        rp.api().create_pod({"name": "x"})
    assert e.value.status == 503
    assert len(rp.bodies("POST", "/v1/pods")) == 1


def test_retries_give_up_after_three_attempts() -> None:
    rp = FakeRunpod()
    rp.fail[("GET", "/v1/pods")] = [503, 503, 503, 503]
    with pytest.raises(RunpodApiError, match="503"):
        rp.api().list_pods()
    assert len(rp.requests) == 3


def test_errors_and_reprs_never_contain_the_key() -> None:
    def echo_key(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {request.headers['Authorization']}")

    api = RunpodApi(SecretStr(FAKE_KEY), transport=httpx.MockTransport(echo_key))
    with pytest.raises(RunpodApiError) as e:
        api.list_pods()
    assert FAKE_KEY not in str(e.value)
    assert "***" in str(e.value)
    assert FAKE_KEY not in repr(api)
    assert FAKE_KEY not in repr(load_runpod_api_key({runpod_api.KEY_ENV: FAKE_KEY}))


def test_transport_errors_are_wrapped_and_redacted() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {request.headers['Authorization']}")

    api = RunpodApi(SecretStr(FAKE_KEY), transport=httpx.MockTransport(boom), sleep=lambda s: None)
    with pytest.raises(RunpodApiError) as e:
        api.get_pod("abcdef123")
    assert e.value.status is None
    assert FAKE_KEY not in str(e.value)


def test_graphql_errors_raise() -> None:
    rp = FakeRunpod()
    with pytest.raises(RunpodApiError, match="unsupported query"):
        rp.api().graphql("query { myself { id } }")


def test_bad_pod_ids_are_refused_before_any_request() -> None:
    rp = FakeRunpod()
    for bad in ["../pods", "abc/def", "", "UPPER123", "a" * 40]:
        with pytest.raises(ValueError):
            rp.api().get_pod(bad)
    assert rp.requests == []
    assert check_pod_id("9dhapj2120z6su") == "9dhapj2120z6su"


def test_empty_key_is_refused() -> None:
    with pytest.raises(ValueError):
        RunpodApi(SecretStr(""))


def test_key_from_env_then_keychain_then_none() -> None:
    def keychain() -> str | None:
        return "from-keychain"

    assert load_runpod_api_key({"RUNPOD_API_KEY": " from-env\n"}, keychain=keychain) == SecretStr(
        "from-env"
    )
    key = load_runpod_api_key({}, keychain=keychain)
    assert key is not None and key.get_secret_value() == "from-keychain"
    assert load_runpod_api_key({"RUNPOD_API_KEY": ""}, keychain=lambda: None) is None
    assert load_runpod_api_key({}, keychain=lambda: "   ") is None


def _completed(rc: int, out: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr="")


def test_keychain_lookup_calls_security_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _completed(0, "secret-from-keychain\n")

    monkeypatch.setattr(runpod_api.sys, "platform", "darwin")
    monkeypatch.setattr(runpod_api.shutil, "which", lambda name: "/usr/bin/security")
    monkeypatch.setattr(runpod_api.subprocess, "run", fake_run)
    monkeypatch.setenv("USER", "someone")
    assert REAL_KEYCHAIN_LOOKUP() == "secret-from-keychain"
    assert calls == [
        ["security", "find-generic-password", "-a", "someone", "-s", "RUNPOD_API_KEY", "-w"]
    ]
    monkeypatch.setattr(runpod_api.subprocess, "run", lambda argv, **kw: _completed(44, ""))
    assert REAL_KEYCHAIN_LOOKUP() is None


def test_keychain_lookup_is_off_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_run(*a: Any, **kw: Any) -> None:
        raise AssertionError("must not run security off macOS")

    monkeypatch.setattr(runpod_api.sys, "platform", "linux")
    monkeypatch.setattr(runpod_api.subprocess, "run", fail_run)
    assert REAL_KEYCHAIN_LOOKUP() is None


@pytest.mark.parametrize(
    ("value", "micros"),
    [
        (1.09, 1_090_000),
        ("4.36", 4_360_000),
        (Decimal("0.0000011"), 2),
        (0, 0),
        (2, 2_000_000),
    ],
)
def test_usd_to_micros_is_exact_and_rounds_up(value: Any, micros: int) -> None:
    assert usd_to_micros(value) == micros


def test_usd_to_micros_rejects_negative() -> None:
    with pytest.raises(ValueError):
        usd_to_micros(-0.5)


def test_public_ssh() -> None:
    assert public_ssh({"publicIp": "203.0.113.7", "portMappings": {"22": 28813}}) == (
        "203.0.113.7",
        28813,
    )
    assert public_ssh({"publicIp": "", "portMappings": {"22": 1}}) is None
    assert public_ssh({"publicIp": "203.0.113.7", "portMappings": {}}) is None
    assert public_ssh({"publicIp": "203.0.113.7", "portMappings": None}) is None
    assert public_ssh(create_response_fixture()) is None


def test_graphql_body_shape() -> None:
    rp = FakeRunpod()
    pod = rp.add_pod()
    rp.api().graphql(runpod_api.TERMINATE_MUTATION, {"podId": pod["id"]})
    (body,) = rp.bodies("POST", "/graphql")
    assert json.dumps(body)  # serialisable
    assert body["query"] == runpod_api.TERMINATE_MUTATION
