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
_VAR = r"(?:\$\{[A-Za-z_][A-Za-z0-9_]*(?::?\?[^}]*)?\}|\$[A-Za-z_][A-Za-z0-9_]*)"
# `<name>@sha256:<64 hex>`, where the name may hold variables and the digest may be a
# variable (`...@sha256:${X}` or `...@${X}`); or the whole reference is one variable.
DIGEST = re.compile(rf"^[^\s@]+@(?:sha256:(?:[0-9a-f]{{64}}|{_VAR})|{_VAR})$")
VARIABLE = re.compile(rf"^{_VAR}$")


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
        files = svc.get("env_file") or []
        for f in [files] if isinstance(files, (str, dict)) else files:
            path = f["path"] if isinstance(f, dict) else f
            assert str(path).startswith("/run/loom/"), f"{name}: env file {path} outside /run/loom"

    # Anywhere in the parsed file (extension fields, anchors), not in comments.
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert "LOOM_ENV" not in node, "the production compose file sets LOOM_ENV"
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)
        elif isinstance(node, str):
            assert not node.strip().startswith("LOOM_ENV="), "the file sets LOOM_ENV"

    walk(compose())


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


# §6.3: the services that read a secret Loom renders (database and Redis passwords, the
# internal token, auth_secret, Stripe, the Redis and pod certificates, Grafana's admin
# password). Caddy and Prometheus need none; migrate gets the owner URL from loom-deploy.
NEEDS_SECRETS = ("web", "gateway", "billing", "poolmgr", "redis", "grafana")
SECRETS_DIR = "/run/loom/secrets/"


def mounts(svc: dict[str, Any], doc: dict[str, Any]) -> list[tuple[str, bool]]:
    """(host source, read-only) of each bind mount and file-backed compose secret."""
    out: list[tuple[str, bool]] = []
    for v in svc.get("volumes") or []:
        if isinstance(v, dict):
            out.append((str(v.get("source", "")), v.get("read_only") is True))
        else:
            parts = str(v).split(":")
            out.append((parts[0], len(parts) >= 3 and "ro" in parts[2].split(",")))
    top = doc.get("secrets") or {}
    for s in svc.get("secrets") or []:
        ref = s.get("source") if isinstance(s, dict) else s
        spec = top.get(ref) or {}
        if isinstance(spec, dict) and spec.get("file"):
            out.append((str(spec["file"]), True))  # compose mounts secrets read-only
    return out


def secrets_owner(source: str) -> str | None:
    if not source.startswith(SECRETS_DIR):
        return None
    return source[len(SECRETS_DIR) :].split("/")[0]


def test_run_loom_mounts_are_read_only() -> None:
    doc = compose()
    for name, svc in doc["services"].items():
        for source, ro in mounts(svc, doc):
            if source.startswith("/run/loom"):
                assert ro, f"{name}: {source} is not mounted read-only"


@pytest.mark.parametrize("name", NEEDS_SECRETS)
def test_each_service_mounts_its_own_secrets_directory(name: str) -> None:
    doc = compose()
    own = [(s, ro) for s, ro in mounts(doc["services"][name], doc) if secrets_owner(s) == name]
    assert own, f"{name} mounts nothing from {SECRETS_DIR}{name}"
    assert all(ro for _, ro in own), f"{name}: its secrets are not read-only: {own}"


def test_no_service_mounts_another_services_secrets() -> None:
    doc = compose()
    for name, svc in doc["services"].items():
        for source, _ in mounts(svc, doc):
            owner = secrets_owner(source)
            assert owner in (None, name), f"{name} mounts {source}, {owner}'s secrets"


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


def words(value: Any) -> list[str]:
    """A command or entrypoint (string or list) as whitespace-separated words."""
    if value is None:
        return []
    if isinstance(value, str):
        return value.split()
    return [w for item in value for w in str(item).split()]


def redis_config() -> str:
    """Redis's settings as written in the compose file: its command and entrypoint, plus
    any config file from deploy/ it mounts (a `redis.conf` is as good as flags)."""
    doc = compose()
    svc = doc["services"]["redis"]
    parts = words(svc.get("entrypoint")) + words(svc.get("command"))
    sources = [
        str(v.get("source", "")) if isinstance(v, dict) else str(v).split(":")[0]
        for v in svc.get("volumes") or []
    ]
    for c in svc.get("configs") or []:
        spec = (doc.get("configs") or {}).get(c.get("source") if isinstance(c, dict) else c)
        if isinstance(spec, dict) and spec.get("file"):
            sources.append(str(spec["file"]))
    for source in sources:
        path = (PROD.parent / source).resolve()
        if not source.startswith("/") and path.is_file():
            for line in path.read_text().splitlines():
                if not line.lstrip().startswith("#"):
                    parts += line.split()
    return " ".join(parts)


@pytest.mark.parametrize(
    "setting",
    [
        ("tls-port", "6379"),
        ("port", "0"),
        ("appendonly", "yes"),
        ("maxmemory-policy", "noeviction"),
    ],
)
def test_redis_runs_tls_only_durable_and_never_evicts(setting: tuple[str, str]) -> None:
    """§6.5: `--port 0 --tls-port 6379`, `appendonly yes`, `maxmemory-policy noeviction`
    (contracts §2 checks the last with CONFIG GET)."""
    key, value = setting
    config = redis_config()
    pat = rf"(?:^|\s)(?:--)?{re.escape(key)}\s+[\"']?{re.escape(value)}[\"']?(?:\s|$)"
    assert re.search(pat, config), f"redis is not run with {key} {value}: {config!r}"


def test_the_redis_password_is_not_on_its_command_line() -> None:
    """`requirepass` comes from the secret (a mounted config file), never as an argument."""
    svc = services()["redis"]
    argv = " ".join(words(svc.get("entrypoint")) + words(svc.get("command")))
    assert "requirepass" not in argv, f"redis gets requirepass on its command line: {argv!r}"


def test_billing_runs_all() -> None:
    """§6.2: `billing` (`all`): the API, the webhook and the jobs in one service."""
    argv = words(services()["billing"].get("command"))
    assert "all" in argv, f"billing does not run `all`: {argv}"


def test_the_gateway_has_a_healthcheck() -> None:
    """poolmgr waits for `service_healthy` (P4), so the gateway needs a healthcheck: in the
    compose file, or the image's own (not disabled here)."""
    hc = services()["gateway"].get("healthcheck")
    if isinstance(hc, dict):
        assert hc.get("disable") is not True, "the gateway's healthcheck is disabled"
        test = hc.get("test")
        assert words(test)[:1] != ["NONE"], "the gateway's healthcheck is disabled"
        if test:
            return
    dockerfile = ROOT / "services" / "gateway" / "Dockerfile"
    assert dockerfile.is_file() and re.search(r"^HEALTHCHECK\s", dockerfile.read_text(), re.M), (
        "the gateway has no healthcheck (compose or image)"
    )


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
