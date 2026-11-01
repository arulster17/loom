"""docker-compose.yml stays consistent with the registry and the contracts (§2, §9, §10)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from loom_bench.registry import load_registry
from loom_platform.pricing import servable

ROOT = Path(__file__).resolve().parents[3]


def compose() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def published(service: dict[str, Any]) -> set[int]:
    """Default host ports: "${VAR:-8080}:8080" -> 8080."""
    out = set()
    for spec in service.get("ports", []):
        host = str(spec).rsplit(":", 1)[0]
        m = re.search(r":-(\d+)\}$", host) or re.fullmatch(r"(\d+)", host)
        assert m, spec
        out.add(int(m.group(1)))
    return out


def test_one_mock_pod_per_servable_model_and_poolsim_routes_to_it():
    services = compose()
    pods = {
        name: svc["environment"]["LOOM_MOCKPOD_MODELS"]
        for name, svc in services.items()
        if svc.get("command") == ["loom-mockpod"]
    }
    for name, svc in services.items():
        if name in pods:
            assert svc["environment"]["LOOM_MOCKPOD_ID"] == name  # certificate name = host
            assert not svc.get("ports"), "mock pods are reachable only on the compose network"
            assert svc["expose"] == ["8443"]
    servable_ids = {m.id for m in servable(load_registry())}
    assert set(pods.values()) == servable_ids

    spec = services["poolsim"]["environment"]["LOOM_POOLSIM_REPLICAS"]
    routes = dict(part.split("=", 1) for part in spec.split(","))
    assert routes == {model: f"{pod}@https://{pod}:8443" for pod, model in pods.items()}

    cmd = services["devcerts"]["command"]
    certified = {cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--pod"}
    assert certified == set(pods)


def test_ports_match_the_contracts():
    services = compose()
    expected = {
        "postgres": {5432},
        "redis": {6379},
        "gateway": {8080, 9100},
        "billing": {8090, 9101},
        "web": {3000},
        "mailpit": {8025, 1025},
    }
    for name, ports in expected.items():
        assert published(services[name]) == ports, name
    for name in ("migrate", "devcerts", "poolsim"):
        assert not services[name].get("ports"), name


def test_gateway_has_the_mtls_material():
    env = compose()["gateway"]["environment"]
    assert env["LOOM_GATEWAY_CLIENT_CERT"] == "/certs/gateway.crt"
    assert env["LOOM_GATEWAY_CLIENT_KEY"] == "/certs/gateway.key"
    assert env["LOOM_POD_CA_CERT"] == "/certs/ca.crt"
    assert env["LOOM_DATABASE_URL"].endswith("@postgres:5432/loom_platform")


def test_billing_wiring_matches_its_readme():
    services = compose()
    billing, stub = services["billing"], services["stripe-stub"]
    # The image's entrypoint is `loom-billing`; commands are its subcommands.
    assert billing["command"] == ["all"] and stub["command"] == ["stripe-stub"]
    assert billing["build"]["dockerfile"] == stub["build"]["dockerfile"]
    assert billing["environment"]["LOOM_STRIPE_STUB_URL"] == "http://stripe-stub:12111"
    stub_env = stub["environment"]
    assert stub_env["LOOM_STRIPE_STUB_WEBHOOK_URL"] == "http://billing:8090/stripe/webhook"
    assert stub_env["LOOM_STRIPE_STUB_PUBLIC_URL"].startswith("http://localhost:")
    assert published(stub) == {12111}
    # Stub only by explicit opt-in, with a secret generated per stack (never in code).
    assert billing["environment"]["LOOM_STRIPE_MODE"] == "${LOOM_STRIPE_MODE:-stub}"
    assert stub_env["LOOM_STRIPE_MODE"] == "stub"
    secret_file = "/certs/stripe-stub-webhook.secret"
    assert billing["environment"]["LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE"] == secret_file
    assert stub_env["LOOM_STRIPE_STUB_WEBHOOK_SECRET_FILE"] == secret_file
    cmd = services["devcerts"]["command"]
    assert cmd[cmd.index("--secret") + 1] == "stripe-stub-webhook"
    for svc in (billing, stub):
        assert svc["depends_on"]["devcerts"]["condition"] == "service_completed_successfully"
        assert "./.dev/certs:/certs:ro" in svc["volumes"]
    token = billing["environment"]["LOOM_BILLING_INTERNAL_TOKEN"]
    assert token and services["web"]["environment"]["LOOM_INTERNAL_TOKEN"] == token
    assert billing["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
