"""m3-h's route checks (M3 spec §10 m3-h) against a fake stack: no network, no TLS.

The public model list needs no key (contracts §"Public model list"): `GET /v1/models` on
`api.` answers 200. A keyed route without a key answers 401 with the OpenAI error shape and
code `invalid_api_key` (contracts §11). Ports other than 80/443 are closed, TLS is faked,
and every HTTP call goes through an `httpx.MockTransport`.
"""

from __future__ import annotations

import json
import socket
import ssl
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from loom_devstack import m3_acceptance as m3
from loom_devstack.acceptance import AcceptanceError, ItemResult

DOMAIN = "loom.example.test"
NAME = "Acme Inference"
HSTS = {"strict-transport-security": "max-age=31536000; includeSubDomains"}
UNAUTHORIZED = {
    "error": {
        "message": "missing API key",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}


class FakeConn:
    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def version(self) -> str:
        return "TLSv1.3"


class FakeTls:
    def wrap_socket(self, raw: object, server_hostname: str) -> FakeConn:
        return FakeConn()


def fake_connect(addr: tuple[str, int], timeout: float | None = None) -> FakeConn:
    if addr[1] != 443:
        raise ConnectionRefusedError(f"{addr} closed")
    return FakeConn()


Answer = Callable[[], httpx.Response]


def stack(overrides: dict[tuple[str, str], Answer]) -> Any:
    """A request handler for a correct stack; `overrides` replaces (method, URL) answers."""

    def handle(req: httpx.Request) -> httpx.Response:
        host, path, method = req.url.host, req.url.path, req.method
        key = (method, f"{req.url.scheme}://{host}{path}")
        if key in overrides:
            return overrides[key]()
        if req.url.scheme == "http":
            return httpx.Response(308, headers={"location": f"https://{host}{path}"})
        if (method, host, path) == ("GET", f"api.{DOMAIN}", "/v1/models"):
            return httpx.Response(200, json={"object": "list", "data": []}, headers=HSTS)
        if (method, host, path) == ("POST", f"api.{DOMAIN}", "/v1/chat/completions"):
            assert "authorization" not in req.headers, "m3-h sent a key"
            return httpx.Response(401, json=UNAUTHORIZED, headers=HSTS)
        if (method, host, path) == ("GET", f"hooks.{DOMAIN}", "/stripe/health"):
            return httpx.Response(200, json={"service": "stripe_webhook"}, headers=HSTS)
        if (method, host, path) == ("POST", f"hooks.{DOMAIN}", "/stripe/webhook"):
            return httpx.Response(400, json={"error": "unsigned"}, headers=HSTS)
        if (method, host, path) == ("GET", DOMAIN, "/signin"):
            page = f"<html><title>{NAME}</title><body><h1>{NAME}</h1></body></html>"
            return httpx.Response(200, text=page, headers=HSTS)
        return httpx.Response(404, text="not found", headers=HSTS)

    return handle


def run_m3_h(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[tuple[str, str], Answer]
) -> ItemResult:
    real_client = httpx.Client
    transport = httpx.MockTransport(stack(overrides))

    def client(**kw: Any) -> httpx.Client:
        return real_client(transport=transport, **kw)

    monkeypatch.setattr(socket, "create_connection", fake_connect)
    monkeypatch.setattr(ssl, "create_default_context", lambda *a, **k: FakeTls())
    monkeypatch.setattr(httpx, "Client", client)
    args = SimpleNamespace(domain=DOMAIN, brand_name=NAME)
    acc = cast(m3.M3Acceptance, SimpleNamespace(args=args))
    res = ItemResult("m3-h", "external")
    m3.M3Acceptance.item_m3_h(acc, res)
    return res


def test_m3_h_passes_on_a_stack_with_a_public_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    res = run_m3_h(monkeypatch, {})
    assert res.status == "PASS"
    assert res.numbers["routes_checked"]


MODELS = ("GET", f"https://api.{DOMAIN}/v1/models")
CHAT = ("POST", f"https://api.{DOMAIN}/v1/chat/completions")


def test_a_keyed_model_list_fails_m3_h(monkeypatch: pytest.MonkeyPatch) -> None:
    over = {MODELS: lambda: httpx.Response(401, json=UNAUTHORIZED, headers=HSTS)}
    with pytest.raises(AcceptanceError, match="/v1/models"):
        run_m3_h(monkeypatch, over)


def test_a_model_list_without_hsts_fails_m3_h(monkeypatch: pytest.MonkeyPatch) -> None:
    over = {MODELS: lambda: httpx.Response(200, json={"object": "list", "data": []})}
    with pytest.raises(AcceptanceError, match="HSTS"):
        run_m3_h(monkeypatch, over)


OTHER_CODE = {"error": {**UNAUTHORIZED["error"], "code": "x"}}


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(lambda: httpx.Response(200, json={"choices": []}, headers=HSTS), id="200"),
        pytest.param(lambda: httpx.Response(404, text="not found", headers=HSTS), id="404"),
        pytest.param(
            lambda: httpx.Response(401, json=OTHER_CODE, headers=HSTS), id="401-other-code"
        ),
        pytest.param(
            lambda: httpx.Response(401, text="unauthorized", headers=HSTS), id="401-no-json"
        ),
        pytest.param(
            lambda: httpx.Response(401, json={"code": "invalid_api_key"}, headers=HSTS),
            id="401-code-not-under-error",
        ),
    ],
)
def test_a_keyed_route_that_does_not_refuse_with_invalid_api_key_fails_m3_h(
    monkeypatch: pytest.MonkeyPatch, answer: Answer
) -> None:
    with pytest.raises(AcceptanceError, match="chat/completions"):
        run_m3_h(monkeypatch, {CHAT: answer})


def test_error_code_reads_the_openai_error_shape() -> None:
    assert m3.error_code(httpx.Response(401, json=UNAUTHORIZED)) == "invalid_api_key"
    assert m3.error_code(httpx.Response(401, text="nope")) is None
    assert m3.error_code(httpx.Response(401, json=["error"])) is None
    assert m3.error_code(httpx.Response(401, content=json.dumps({"error": "x"}))) is None
