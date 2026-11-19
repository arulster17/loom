"""`deploy/compose.prod.yml`, the production stack on the box (M3 spec §6.2, A14).

Parsed, not run (A15 runs it). Images are referenced only by digest: a literal
`<name>@sha256:<64 hex>` (third-party images such as Redis), or a variable `loom-deploy`
fills from the release manifest (the Loom images, built per commit, §6.1) with no default
tag.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROD = ROOT / "deploy" / "compose.prod.yml"

SERVICES = {
    "caddy",
    "web",
    "gateway",
    "billing",
    "poolmgr",
    "redis",
    "prometheus",
    "grafana",
    "migrate",
}
DEV_ONLY = {"mailpit", "stripe-stub", "devcerts", "poolsim", "postgres"}
DIGEST = re.compile(r"^[^\s$]+@sha256:[0-9a-f]{64}$")
VARIABLE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*(:?\?[^}]*)?\}$")


def compose() -> dict[str, Any]:
    assert PROD.is_file(), "deploy/compose.prod.yml does not exist"
    doc = yaml.safe_load(PROD.read_text())
    assert isinstance(doc, dict) and isinstance(doc.get("services"), dict), "no services"
    return doc


def services() -> dict[str, dict[str, Any]]:
    return compose()["services"]


def env_names(svc: dict[str, Any]) -> set[str]:
    env = svc.get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {str(e).split("=", 1)[0] for e in env}


def port_specs(svc: dict[str, Any]) -> list[tuple[str | None, int]]:
    """(host ip or None, host port) for each published port, short and long syntax."""
    out: list[tuple[str | None, int]] = []
    for spec in svc.get("ports") or []:
        if isinstance(spec, dict):
            published = spec.get("published")
            assert published is not None, f"port {spec} publishes no host port"
            out.append((spec.get("host_ip"), int(str(published).split("-")[0])))
            continue
        text = str(spec).split("/")[0]
        parts = text.rsplit(":", 2)
        if len(parts) == 3:
            out.append((parts[0].strip("[]"), int(parts[1])))
        elif len(parts) == 2:
            out.append((None, int(parts[0])))
        else:
            raise AssertionError(f"port {spec!r} publishes a random host port")
    return out


def test_the_services_are_the_production_set() -> None:
    names = set(services())
    assert names == SERVICES, (
        f"missing {sorted(SERVICES - names)}, extra {sorted(names - SERVICES)}"
    )
    assert not names & DEV_ONLY


def test_migrate_runs_only_in_its_profile() -> None:
    assert services()["migrate"].get("profiles") == ["migrate"]
    for name, svc in services().items():
        if name != "migrate":
            assert not svc.get("profiles"), f"{name} is in a profile: it would not start"


def test_no_service_sets_loom_env() -> None:
    """Unset LOOM_ENV means prod: every service runs its production checks (§6.2)."""
    for name, svc in services().items():
        assert "LOOM_ENV" not in env_names(svc), f"{name} sets LOOM_ENV"
        for f in svc.get("env_file") or []:
            path = f["path"] if isinstance(f, dict) else f
            assert str(path).startswith("/run/loom/"), f"{name}: env file {path} outside /run/loom"
    assert "LOOM_ENV" not in PROD.read_text(), "LOOM_ENV appears in the production compose file"


def test_only_caddy_publishes_public_ports() -> None:
    for name, svc in services().items():
        for host_ip, port in port_specs(svc):
            if name == "caddy" and port in (80, 443):
                assert host_ip in (None, "", "0.0.0.0", "::"), f"caddy {port} on {host_ip}"
                continue
            assert host_ip == "127.0.0.1", (
                f"{name} publishes {port} on {host_ip or 'every address'}"
            )
            assert port not in (80, 443), f"{name} takes Caddy's port {port}"
    caddy = {p for _, p in port_specs(services()["caddy"])}
    assert caddy == {80, 443}, caddy


def test_metrics_prometheus_and_grafana_only_on_loopback() -> None:
    svcs = services()
    want = {"gateway": 9100, "billing": 9101, "poolmgr": 9102}
    for name, port in want.items():
        assert ("127.0.0.1", port) in port_specs(svcs[name]), f"{name} metrics {port}"
    assert ("127.0.0.1", 3001) in port_specs(svcs["grafana"]), "grafana on 127.0.0.1:3001"
    for name in ("web", "redis", "migrate"):
        assert not port_specs(svcs[name]), f"{name} publishes a port"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_images_by_digest(name: str) -> None:
    svc = services()[name]
    assert "build" not in svc, f"{name} builds on the box"
    image = str(svc.get("image", ""))
    assert DIGEST.match(image) or VARIABLE.match(image), (
        f"{name}: image {image!r} is neither name@sha256:<digest> nor a required variable"
    )
    assert ":-" not in image, f"{name}: image variable with a default ({image})"


def test_gateway_stops_gracefully_and_everything_restarts() -> None:
    svcs = services()
    grace = str(svcs["gateway"].get("stop_grace_period"))
    assert grace in ("30s", "30"), f"gateway stop_grace_period {grace}"
    for name, svc in svcs.items():
        if name == "migrate":
            continue
        assert svc.get("restart") == "unless-stopped", f"{name} restart {svc.get('restart')}"


def test_poolmgr_starts_after_a_healthy_gateway() -> None:
    """P4: the edge is the reverse of the dev file's (§6.2)."""
    svcs = services()
    dep = svcs["poolmgr"].get("depends_on") or {}
    assert isinstance(dep, dict) and "gateway" in dep, f"poolmgr depends_on {dep}"
    assert dep["gateway"].get("condition") == "service_healthy", dep["gateway"]
    gw_dep = svcs["gateway"].get("depends_on") or {}
    assert "poolmgr" not in gw_dep, "gateway depends on poolmgr: a cycle against P4"


def bind_sources(svc: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for v in svc.get("volumes") or []:
        if isinstance(v, dict):
            if v.get("type") == "bind":
                out.append((str(v["source"]), str(v["target"])))
            elif v.get("type") == "volume":
                out.append((f"volume:{v.get('source')}", str(v["target"])))
        else:
            src, _, rest = str(v).partition(":")
            out.append((src, rest.split(":")[0]))
    return out


def test_state_lives_on_the_data_volume() -> None:
    svcs = services()
    for name, sub in {
        "redis": "redis",
        "gateway": "gateway-spool",
        "caddy": "caddy",
        "poolmgr": "poolmgr",
        "prometheus": "prometheus",
    }.items():
        sources = [s for s, _ in bind_sources(svcs[name])]
        assert f"/var/lib/loom/{sub}" in sources, f"{name}: no /var/lib/loom/{sub} bind ({sources})"


def test_secrets_are_read_only_files_from_run_loom() -> None:
    for name, svc in services().items():
        for v in svc.get("volumes") or []:
            text = str(v)
            if "/run/loom" in text:
                if isinstance(v, dict):
                    assert v.get("read_only") is True, f"{name}: {v} not read-only"
                else:
                    assert str(v).endswith(":ro"), f"{name}: {v} not read-only"


def test_the_rds_ca_bundle_is_mounted_read_only() -> None:
    svcs = services()
    for name in ("gateway", "billing", "web", "poolmgr", "migrate"):
        mounts = [str(v) for v in svcs[name].get("volumes") or []]
        hit = [m for m in mounts if "/etc/loom/rds-ca.pem" in m]
        assert hit, f"{name} does not mount the RDS CA at /etc/loom/rds-ca.pem"
        assert all(m.endswith(":ro") or "'read_only': True" in m for m in hit), hit


def test_logs_go_to_cloudwatch() -> None:
    for name, svc in services().items():
        logging = svc.get("logging") or {}
        assert logging.get("driver") == "awslogs", f"{name}: log driver {logging.get('driver')}"
        assert (logging.get("options") or {}).get("awslogs-group") == "/loom/platform", name


def test_no_secret_value_in_the_file() -> None:
    text = PROD.read_text()
    for pat in (
        r"sk_(test|live)_[A-Za-z0-9]+",
        r"whsec_[A-Za-z0-9]+",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?i)password\s*[:=]\s*['\"]?[^'\"\s$]{6,}",
        r"postgres(ql)?(\+\w+)?://[^/\s$]*:[^@\s$]+@",
    ):
        assert not re.search(pat, text), f"compose.prod.yml holds a secret ({pat})"
