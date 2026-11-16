"""docker-compose.yml runs the pool manager on the local backend (docs/phase1-m2-spec.md
§9 "Local backend", §11 settings, §13 item 25; decision 3 in §15).

The default stack (`docker compose up`) runs `poolmgr` with `provider: local` from the
committed `config/serving.yaml`; `poolsim`, `mock-pod-1` and `mock-pod-2` stay defined
(the locked `test_devstack_compose.py` reads them) under `profiles: ["poolsim"]`, so
nothing in the default stack starts or waits for them.

The gateway reaches the local pods' proxies on the poolmgr container
(`LOOM_POOLMGR_LOCAL_HOST`), with mTLS from the same dev CA it trusts, from a source
address the pods' allowlist admits: the network the gateway and poolmgr share has a fixed
`ipam` subnet, and `LOOM_POOLMGR_ALLOW_CIDRS` (dev only) holds CIDRs inside it, never
wider (§9). `LOOM_POOLMGR_IDLE_STOP_S` (dev only) shortens every model's idle stop for
the acceptance run, but stays longer than the ~4-minute M1 acceptance (§9: compose sets
600). poolmgr shares the gateway's Redis and Postgres database (it does not read the usage
stream: a settled request reaches it as `loom:pool:<model>:last_request`, spec §11).

Values written `${VAR:-default}` are read as their default.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from loom_platform.serving import load_serving

ROOT = Path(__file__).resolve().parents[3]
POOLSIM_PROFILE = "poolsim"
POOLSIM_SERVICES = ("poolsim", "mock-pod-1", "mock-pod-2")
METRICS_PORT = 9102
LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
M1_ACCEPTANCE_S = 240  # ~4 minutes: no model may go idle-cold during it
IDLE_STOP_MAX_S = 900

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def interpolated(value: str) -> str:
    """`${VAR:-default}` / `${VAR-default}` -> default; `${VAR}` -> ""."""
    value = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?-([^}]*)\}", r"\1", value)
    return re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", "", value)


def compose_doc() -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return doc


def services() -> dict[str, Any]:
    svcs: dict[str, Any] = compose_doc()["services"]
    return svcs


def poolmgr() -> dict[str, Any]:
    svcs = services()
    assert "poolmgr" in svcs, "docker-compose.yml has no poolmgr service"
    svc: dict[str, Any] = svcs["poolmgr"]
    return svc


def env_of(svc: dict[str, Any]) -> dict[str, str]:
    """The service's environment, `${VAR:-default}` read as its default."""
    env = svc.get("environment") or {}
    if isinstance(env, list):
        out = {}
        for item in env:
            k, _, v = str(item).partition("=")
            out[k] = interpolated(v)
        return out
    return {str(k): "" if v is None else interpolated(str(v)) for k, v in env.items()}


def depends_on(svc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    deps = svc.get("depends_on") or {}
    if isinstance(deps, list):
        return {str(d): {} for d in deps}
    return {str(k): dict(v or {}) for k, v in deps.items()}


def container_ports(svc: dict[str, Any]) -> set[int]:
    """Container ports a service publishes on the host ("${VAR:-9102}:9102" -> 9102)."""
    out: set[int] = set()
    for spec in svc.get("ports") or []:
        if isinstance(spec, dict):
            out.add(int(interpolated(str(spec["target"]))))
            continue
        target = interpolated(str(spec)).rsplit(":", 1)[-1].split("/", 1)[0]
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", target)
        assert m, spec
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        out.update(range(lo, hi + 1))
    return out


def host_path(svc: dict[str, Any], container_path: str) -> Path | None:
    """The repo path a container path maps to through the service's bind mounts."""
    want = PurePosixPath(container_path)
    for vol in svc.get("volumes") or []:
        if isinstance(vol, dict):
            source, target = vol.get("source", ""), vol.get("target", "")
        else:
            parts = str(vol).split(":")
            if len(parts) < 2:
                continue
            source, target = parts[0], parts[1]
        if not str(source).startswith("."):
            continue  # a named volume
        mount = PurePosixPath(target)
        if want == mount or mount in want.parents:
            return (ROOT / source / want.relative_to(mount)).resolve()
    return None


def profiles(svc: dict[str, Any]) -> list[str]:
    return [str(p) for p in svc.get("profiles") or []]


def default_services() -> dict[str, Any]:
    """What `docker compose up` starts with no profile."""
    return {n: s for n, s in services().items() if not profiles(s)}


def dockerfile_of(name: str) -> Path | None:
    """The Dockerfile a service is built from (its own `build`, else another service's
    building the same `image`)."""
    svcs = services()
    svc = svcs[name]
    candidates = [svc] + [
        s for s in svcs.values() if svc.get("image") and s.get("image") == svc.get("image")
    ]
    for s in candidates:
        build = s.get("build")
        if isinstance(build, dict) and build.get("dockerfile"):
            return ROOT / str(build.get("context", ".")) / str(build["dockerfile"])
    return None


# ------------------------------------------------------------------ poolsim profile


@pytest.mark.parametrize("name", POOLSIM_SERVICES)
def test_poolsim_and_the_mock_pods_stay_defined_under_the_poolsim_profile(name: str) -> None:
    svc = services().get(name)
    assert svc is not None, f"{name} must stay defined (test_devstack_compose.py reads it)"
    assert profiles(svc) == [POOLSIM_PROFILE], f"{name}: profiles {profiles(svc)}"


def test_nothing_in_the_default_stack_depends_on_the_poolsim_profile() -> None:
    for name, svc in default_services().items():
        bad = set(depends_on(svc)) & set(POOLSIM_SERVICES)
        assert not bad, f"{name} depends on {sorted(bad)}, which `docker compose up` skips"


def test_the_gateway_does_not_depend_on_poolsim() -> None:
    gateway = services()["gateway"]
    assert not profiles(gateway)
    assert "poolsim" not in depends_on(gateway)


# ------------------------------------------------------------------ the poolmgr service


def test_poolmgr_runs_in_the_default_stack() -> None:
    """`loom-poolmgr run`: in the command, or `run` with an entrypoint (compose's or the
    image's Dockerfile ENTRYPOINT) naming `loom-poolmgr`."""
    svc = poolmgr()
    assert not profiles(svc), "poolmgr must start with a plain `docker compose up`"

    def words(v: Any) -> list[str]:
        if not v:
            return []
        return [str(w) for w in v] if isinstance(v, list) else str(v).split()

    command, entrypoint = words(svc.get("command")), words(svc.get("entrypoint"))
    assert "run" in command, f"poolmgr command {command}"
    named = any("loom-poolmgr" in w for w in command + entrypoint)
    if not named:
        dockerfile = dockerfile_of("poolmgr")
        assert dockerfile is not None and dockerfile.is_file(), "poolmgr has no Dockerfile"
        entry = [
            line
            for line in dockerfile.read_text().splitlines()
            if line.strip().upper().startswith("ENTRYPOINT")
        ]
        named = bool(entry) and "loom-poolmgr" in entry[-1]
    assert named, f"neither the command {command} nor an entrypoint runs loom-poolmgr"


def test_poolmgr_has_a_healthcheck() -> None:
    """`docker compose up --wait` must mean the manager is healthy (item 25)."""
    svc = poolmgr()
    check = svc.get("healthcheck")
    if check is not None:
        assert not check.get("disable"), "poolmgr's healthcheck is disabled"
        assert check.get("test"), "poolmgr's healthcheck has no test"
        return
    dockerfile = dockerfile_of("poolmgr")
    assert dockerfile and "HEALTHCHECK" in dockerfile.read_text(), "poolmgr has no healthcheck"


def test_poolmgr_runs_with_loom_env_dev() -> None:
    """The local backend and the dev-only settings are refused outside LOOM_ENV=dev."""
    assert env_of(poolmgr()).get("LOOM_ENV") == "dev"


def test_poolmgr_publishes_no_port_but_its_metrics() -> None:
    """The local pods' proxies are reached on the compose network only; at most the
    Prometheus port (contracts §2: 9102) is published."""
    svc = poolmgr()
    assert svc.get("network_mode") != "host", "poolmgr must not share the host's network"
    assert container_ports(svc) <= {METRICS_PORT}, svc.get("ports")
    port = env_of(svc).get("LOOM_POOLMGR_METRICS_PORT")
    if port:
        assert port == str(METRICS_PORT)


def test_poolmgr_starts_after_the_migration_and_redis() -> None:
    """`serving_pods` (migration 0010) and the pool hashes must exist before it runs."""
    deps = depends_on(poolmgr())
    assert deps.get("migrate", {}).get("condition") == "service_completed_successfully"
    assert deps.get("redis", {}).get("condition") == "service_healthy"


@pytest.mark.parametrize("var", ["LOOM_REDIS_URL", "LOOM_DATABASE_URL"])
def test_poolmgr_shares_the_gateways_redis_and_database(var: str) -> None:
    """§9: the pool hashes, wake keys and last-request stamps the manager reads are the
    gateway's."""
    gateway = env_of(services()["gateway"])
    assert env_of(poolmgr()).get(var) == gateway.get(var), var
    assert gateway.get(var), f"the gateway sets no {var}"


def test_poolmgr_uses_the_committed_serving_yaml_with_the_local_provider() -> None:
    svc = poolmgr()
    env = env_of(svc)
    path = env.get("LOOM_POOLMGR_SERVING_YAML") or env.get("LOOM_SERVING_YAML")
    assert path, "poolmgr sets neither LOOM_POOLMGR_SERVING_YAML nor LOOM_SERVING_YAML"
    mapped = host_path(svc, path)
    assert mapped == (ROOT / "config" / "serving.yaml").resolve(), (path, mapped)
    assert load_serving(mapped).provider == "local"


def test_poolmgr_idle_stop_is_shortened_but_outlasts_the_m1_acceptance() -> None:
    value = env_of(poolmgr()).get("LOOM_POOLMGR_IDLE_STOP_S", "")
    assert value, "poolmgr does not set LOOM_POOLMGR_IDLE_STOP_S"
    assert M1_ACCEPTANCE_S < int(value) <= IDLE_STOP_MAX_S, value


def test_poolmgr_issues_pod_certificates_from_the_ca_the_gateway_trusts() -> None:
    """The dev certificates are mounted, and the CA the manager signs pod certificates
    with is the file the gateway verifies pods against (`LOOM_POD_CA_CERT`), so the
    gateway's mTLS to the local pods works."""
    svc, gateway = poolmgr(), services()["gateway"]
    env, gw_env = env_of(svc), env_of(gateway)
    certs = (ROOT / ".dev" / "certs").resolve()
    assert any(
        isinstance(v, str) and v.split(":")[0].rstrip("/") == "./.dev/certs"
        for v in svc.get("volumes") or []
    ), "poolmgr does not mount ./.dev/certs"
    for var in (
        "LOOM_POOLMGR_CA_CERT",
        "LOOM_POOLMGR_CA_KEY",
        "LOOM_POOLMGR_CLIENT_CERT",
        "LOOM_POOLMGR_CLIENT_KEY",
    ):
        assert env.get(var), f"poolmgr does not set {var}"
        mapped = host_path(svc, env[var])
        assert mapped is not None and certs in mapped.parents, (var, env[var], mapped)
    assert host_path(svc, env["LOOM_POOLMGR_CA_CERT"]) == host_path(
        gateway, gw_env["LOOM_POD_CA_CERT"]
    ), "the manager's CA is not the one the gateway trusts"
    dep = depends_on(svc).get("devcerts", {})
    assert dep.get("condition") == "service_completed_successfully"


def poolmgr_hostnames(svc: dict[str, Any]) -> set[str]:
    names = {"poolmgr"}
    for key in ("hostname", "container_name"):
        if svc.get(key):
            names.add(str(svc[key]))
    nets = svc.get("networks")
    if isinstance(nets, dict):
        for cfg in nets.values():
            names.update(str(a) for a in (cfg or {}).get("aliases") or [])
    return names


def test_the_local_pods_are_advertised_at_the_poolmgr_host() -> None:
    """The pool hash's replica URLs must point at the poolmgr container, not at the
    gateway's own loopback (the proxies listen on all interfaces and are advertised at
    `LOOM_POOLMGR_LOCAL_HOST`)."""
    svc = poolmgr()
    host = env_of(svc).get("LOOM_POOLMGR_LOCAL_HOST", "")
    assert host not in LOOPBACK, f"LOOM_POOLMGR_LOCAL_HOST: {host!r}"
    assert host in poolmgr_hostnames(svc), (host, poolmgr_hostnames(svc))


def networks_of(svc: dict[str, Any]) -> set[str]:
    n = svc.get("networks")
    if not n:
        return {"default"}
    return {str(x) for x in n}


def fixed_subnets(network: str) -> list[Network]:
    cfg = (compose_doc().get("networks") or {}).get(network) or {}
    return [
        ipaddress.ip_network(interpolated(str(c["subnet"])))
        for c in (cfg.get("ipam") or {}).get("config") or []
        if c.get("subnet")
    ]


def gateway_address(network: str) -> str | None:
    nets = services()["gateway"].get("networks")
    if isinstance(nets, dict):
        cfg = nets.get(network) or {}
        addr = cfg.get("ipv4_address") or cfg.get("ipv6_address")
        return interpolated(str(addr)) if addr else None
    return None


def test_the_pods_allowlist_is_the_shared_networks_fixed_subnet_never_wider() -> None:
    """§9: the network the gateway and poolmgr share has a fixed `ipam` subnet;
    `LOOM_POOLMGR_ALLOW_CIDRS` holds CIDRs inside it (never wider) that cover the
    gateway's address (its fixed address, else the whole subnet it may get one from)."""
    svc, gateway = poolmgr(), services()["gateway"]
    shared = networks_of(svc) & networks_of(gateway)
    assert shared, "the gateway and poolmgr share no network"
    fixed = {net: fixed_subnets(net) for net in shared}
    fixed = {net: subnets for net, subnets in fixed.items() if subnets}
    assert fixed, f"no network the gateway and poolmgr share ({sorted(shared)}) has a fixed subnet"

    raw = env_of(svc).get("LOOM_POOLMGR_ALLOW_CIDRS", "")
    cidrs = [c.strip() for c in raw.split(",") if c.strip()]
    assert cidrs, "poolmgr does not set LOOM_POOLMGR_ALLOW_CIDRS"
    allowed = [ipaddress.ip_network(c, strict=True) for c in cidrs]
    all_subnets = [s for subnets in fixed.values() for s in subnets]
    for a in allowed:
        assert any(
            a.version == s.version and a.subnet_of(s)  # type: ignore[arg-type]
            for s in all_subnets
        ), f"{a} is not inside the shared subnet(s) {[str(s) for s in all_subnets]}"

    def covered(target: Network) -> bool:
        return any(
            target.version == a.version and target.subnet_of(a)  # type: ignore[arg-type]
            for a in ipaddress.collapse_addresses(allowed)  # type: ignore[type-var]
        )

    for net, subnets in fixed.items():
        addr = gateway_address(net)
        if addr:
            assert covered(ipaddress.ip_network(addr)), f"the gateway's {addr} is not allowed"
        else:
            for s in subnets:
                assert covered(s), f"the gateway may get any address in {s}: {cidrs} misses some"
