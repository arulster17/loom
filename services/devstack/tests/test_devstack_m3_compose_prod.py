"""`deploy/compose.prod.yml`, the production stack on the box (M3 spec §6.2, A14).

Parsed, not run (A15 runs it). §6.12 fixes the rest of its interface:

- images by digest: the Loom images only as `${LOOM_IMAGE_<NAME>:?...}` (filled from the
  release manifest through `compose.env`; `migrate` and `acceptance` both the devstack
  image), the third-party ones (Redis, Prometheus, Grafana) as a literal
  `<name>@sha256:<64 hex>`;
- bind-mount sources on the box as `${LOOM_DEPLOY_ROOT:-}/run/loom/...` and
  `${LOOM_DEPLOY_ROOT:-}/var/lib/loom/...` (the prefix is empty on the box; tests set a
  scratch root), and each service's env file `${LOOM_DEPLOY_ROOT:-}/run/loom/env/<svc>.env`;
- `acceptance` (profile `acceptance`): the devstack image, the docker socket (the only
  service that mounts it), `deploy.log` read-only, `/var/lib/loom/acceptance` read-write,
  no AWS credentials;
- the Caddyfile comes from the bundle (mounted at `/etc/caddy/Caddyfile`, not baked into
  the image), and the Caddy image is built from `deploy/caddy/` (its `go.mod`/`go.sum` pin
  the rate-limit module).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROD = ROOT / "deploy" / "compose.prod.yml"
ROOT_VAR = "${LOOM_DEPLOY_ROOT:-}"

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
    "acceptance",
}
ONE_SHOT = {"migrate": "migrate", "acceptance": "acceptance"}  # service -> its profile
LOOM_IMAGES = {
    "gateway": "GATEWAY",
    "billing": "BILLING",
    "web": "WEB",
    "poolmgr": "POOLMGR",
    "migrate": "DEVSTACK",
    "acceptance": "DEVSTACK",
    "caddy": "CADDY",
}
THIRD_PARTY = ("redis", "prometheus", "grafana")
ENV_SERVICES = ("caddy", "web", "gateway", "billing", "poolmgr", "migrate", "acceptance")
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


def split_volume(spec: str) -> list[str]:
    """A short-syntax volume split on `:`, the `${LOOM_DEPLOY_ROOT:-}` prefix kept whole."""
    return [p.replace("\0", ROOT_VAR) for p in spec.replace(ROOT_VAR, "\0").split(":")]


def host_path(source: str) -> str:
    """A bind source or env file as the path on the box (the prefix is empty there)."""
    return source.removeprefix(ROOT_VAR)


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


def test_the_one_shot_services_run_only_in_their_profiles() -> None:
    for name, profile in ONE_SHOT.items():
        assert services()[name].get("profiles") == [profile], f"{name} profiles"
    for name, svc in services().items():
        if name not in ONE_SHOT:
            assert not svc.get("profiles"), f"{name} is in a profile: it would not start"


SECRET_KEY = re.compile(r"(?:^|_)(?:PASSWORD|SECRET|TOKEN|KEY)$", re.I)


def env_files(svc: dict[str, Any]) -> list[str]:
    files = svc.get("env_file") or []
    return [
        str(f["path"] if isinstance(f, dict) else f)
        for f in ([files] if isinstance(files, (str, dict)) else files)
    ]


def secret_assignments(text: str) -> list[str]:
    """`KEY=value` lines of an env file whose key looks secret (`*_PASSWORD`, `*_SECRET`,
    `*_TOKEN`, `*_KEY`) and whose value is not empty."""
    out = []
    for line in text.splitlines():
        line = line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if SECRET_KEY.search(key.strip()) and value.strip().strip("\"'"):
            out.append(key.strip())
    return out


def test_no_service_sets_loom_env() -> None:
    """Unset LOOM_ENV means prod: every service runs its production checks (§6.2).

    Env files: a secret one lives under `/run/loom` (§6.3, A13). One elsewhere must hold
    no secret: a file in the repository (a path relative to deploy/) is read and may not
    give a value to a `*_PASSWORD`, `*_SECRET`, `*_TOKEN` or `*_KEY` key; one that is not
    in the repository (an absolute path outside `/run/loom`, or a relative one written on
    the box or marked `required: false`) may not be named like a secret one. Neither may
    set LOOM_ENV.
    """
    for name, svc in services().items():
        assert "LOOM_ENV" not in env_names(svc), f"{name} sets LOOM_ENV"
        for path in env_files(svc):
            if host_path(path).startswith("/run/loom/"):
                continue
            local = (PROD.parent / path).resolve()
            if path.startswith("/") or not local.is_file():
                assert not re.search(r"(?i)secret|password|token|credential", path), (
                    f"{name}: env file {path} looks secret but is outside /run/loom"
                )
                continue
            text = local.read_text(encoding="utf-8")
            hits = secret_assignments(text)
            assert not hits, f"{name}: env file {path} outside /run/loom holds secrets {hits}"
            assert not re.search(r"^\s*(export\s+)?LOOM_ENV\s*=", text, re.M), (
                f"{name}: env file {path} sets LOOM_ENV"
            )

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
    for name in ("web", "redis", "migrate", "acceptance"):
        assert not port_specs(svcs[name]), f"{name} publishes a port"


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_images_by_digest(name: str) -> None:
    """§6.12: a Loom image is exactly `${LOOM_IMAGE_<NAME>:?<message>}` (compose fails when
    compose.env does not set it); a third-party one a literal `name@sha256:<64 hex>`."""
    svc = services()[name]
    assert "build" not in svc, f"{name} builds on the box"
    image = str(svc.get("image", ""))
    if name in LOOM_IMAGES:
        var = LOOM_IMAGES[name]
        assert re.fullmatch(rf"\$\{{LOOM_IMAGE_{var}:\?[^}}]*\}}", image), (
            f"{name}: image {image!r} is not ${{LOOM_IMAGE_{var}:?...}}"
        )
    else:
        assert name in THIRD_PARTY
        assert re.fullmatch(r"[^\s@$]+@sha256:[0-9a-f]{64}", image), (
            f"{name}: image {image!r} is not a literal name@sha256:<digest>"
        )
    assert DIGEST.match(image) or VARIABLE.match(image)


def test_gateway_stops_gracefully_and_everything_restarts() -> None:
    svcs = services()
    grace = str(svcs["gateway"].get("stop_grace_period"))
    assert grace in ("30s", "30"), f"gateway stop_grace_period {grace}"
    for name, svc in svcs.items():
        if name in ONE_SHOT:
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
    """(source as written, target) of each volume; named volumes as `volume:<name>`."""
    out = []
    for v in svc.get("volumes") or []:
        if isinstance(v, dict):
            if v.get("type") == "bind":
                out.append((str(v["source"]), str(v["target"])))
            elif v.get("type") == "volume":
                out.append((f"volume:{v.get('source')}", str(v["target"])))
        else:
            parts = split_volume(str(v))
            out.append((parts[0], parts[1] if len(parts) > 1 else parts[0]))
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
        want = f"{ROOT_VAR}/var/lib/loom/{sub}"
        assert want in sources, f"{name}: no {want} bind ({sources})"


# §6.3: the services that read a secret Loom renders (database and Redis passwords, the
# internal token, auth_secret, Stripe, the Redis and pod certificates, Grafana's admin
# password; the acceptance's Redis CA and webhook secret, §6.12). Caddy and Prometheus
# need none; migrate gets the owner URL from its env file.
NEEDS_SECRETS = ("web", "gateway", "billing", "poolmgr", "redis", "grafana", "acceptance")
SECRETS_DIR = "/run/loom/secrets/"


def mounts(svc: dict[str, Any], doc: dict[str, Any]) -> list[tuple[str, bool]]:
    """(host source on the box, read-only) of each bind mount and file-backed compose
    secret; the `${LOOM_DEPLOY_ROOT:-}` prefix is dropped (it is empty on the box)."""
    out: list[tuple[str, bool]] = []
    for v in svc.get("volumes") or []:
        if isinstance(v, dict):
            out.append((host_path(str(v.get("source", ""))), v.get("read_only") is True))
        else:
            parts = split_volume(str(v))
            out.append((host_path(parts[0]), len(parts) >= 3 and "ro" in parts[2].split(",")))
    top = doc.get("secrets") or {}
    for s in svc.get("secrets") or []:
        ref = s.get("source") if isinstance(s, dict) else s
        spec = top.get(ref) or {}
        if isinstance(spec, dict) and spec.get("file"):
            out.append((host_path(str(spec["file"])), True))  # compose mounts secrets ro
    return out


def raw_sources(svc: dict[str, Any], doc: dict[str, Any]) -> list[str]:
    """Every host path a service names as written: bind sources, file-backed secrets and
    env files."""
    out = [s for s, _ in bind_sources(svc) if not s.startswith("volume:")]
    top = doc.get("secrets") or {}
    for s in svc.get("secrets") or []:
        spec = top.get(s.get("source") if isinstance(s, dict) else s) or {}
        if isinstance(spec, dict) and spec.get("file"):
            out.append(str(spec["file"]))
    return out + env_files(svc)


def secrets_owner(source: str) -> str | None:
    if not source.startswith(SECRETS_DIR):
        return None
    return source[len(SECRETS_DIR) :].split("/")[0]


def test_box_paths_carry_the_deploy_root_prefix() -> None:
    """§6.12: a source under /run/loom or /var/lib/loom is written
    `${LOOM_DEPLOY_ROOT:-}/...`, so a test can move the whole box under a scratch root."""
    doc = compose()
    seen = 0
    for name, svc in doc["services"].items():
        for raw in raw_sources(svc, doc):
            path = host_path(raw)
            if path.startswith(("/run/loom", "/var/lib/loom")):
                seen += 1
                assert raw.startswith(ROOT_VAR + "/"), f"{name}: {raw} lacks {ROOT_VAR}"
    assert seen, "no source under /run/loom or /var/lib/loom: the check would be vacuous"


def test_run_loom_mounts_are_read_only() -> None:
    doc = compose()
    checked: set[str] = set()
    for name, svc in doc["services"].items():
        for source, ro in mounts(svc, doc):
            if source.startswith("/run/loom"):
                checked.add(name)
                assert ro, f"{name}: {source} is not mounted read-only"
    missing = set(NEEDS_SECRETS) - checked
    assert not missing, f"no /run/loom mount found for {sorted(missing)}: nothing was checked"


def test_each_service_reads_its_own_env_file() -> None:
    """§6.12: `env_file` `${LOOM_DEPLOY_ROOT:-}/run/loom/env/<svc>.env`, never mounted."""
    doc = compose()
    for name in ENV_SERVICES:
        want = f"{ROOT_VAR}/run/loom/env/{name}.env"
        assert want in env_files(doc["services"][name]), (
            f"{name}: env_file {env_files(doc['services'][name])}, not {want}"
        )
    for name, svc in doc["services"].items():
        for path in env_files(svc):
            if host_path(path).startswith("/run/loom/env/"):
                assert host_path(path) == f"/run/loom/env/{name}.env", f"{name} reads {path}"
        for source, _ in mounts(svc, doc):
            assert not source.startswith("/run/loom/env"), f"{name} mounts {source}"


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
    for name in ("gateway", "billing", "web", "poolmgr", "migrate", "acceptance"):
        mounts = [str(v) for v in svcs[name].get("volumes") or []]
        hit = [m for m in mounts if "/etc/loom/rds-ca.pem" in m]
        assert hit, f"{name} does not mount the RDS CA at /etc/loom/rds-ca.pem"
        assert all(m.endswith(":ro") or "'read_only': True" in m for m in hit), hit


DOCKER_SOCKET = ("/var/run/docker.sock", "/run/docker.sock")


def volume_specs(svc: dict[str, Any]) -> list[tuple[str, str, bool]]:
    """(host source on the box, target, read-only) of each volume entry."""
    out = []
    for v in svc.get("volumes") or []:
        if isinstance(v, dict):
            src = host_path(str(v.get("source", "")))
            out.append((src, str(v.get("target", "")), v.get("read_only") is True))
        else:
            parts = split_volume(str(v))
            ro = len(parts) >= 3 and "ro" in parts[2].split(",")
            out.append((host_path(parts[0]), parts[1] if len(parts) > 1 else parts[0], ro))
    return out


def test_only_the_acceptance_service_mounts_the_docker_socket() -> None:
    for name, svc in services().items():
        socket = [
            (s, t) for s, t, _ in volume_specs(svc) if s in DOCKER_SOCKET or t in DOCKER_SOCKET
        ]
        if name == "acceptance":
            assert socket, "acceptance does not mount the docker socket"
        else:
            assert not socket, f"{name} mounts the docker socket {socket}"


def test_the_acceptance_service() -> None:
    """§6.12: the deploy log read-only, its own directory read-write, no AWS credentials."""
    svc = services()["acceptance"]
    vols = {t: (s, ro) for s, t, ro in volume_specs(svc)}
    log = vols.get("/var/lib/loom/deploy.log")
    assert log == ("/var/lib/loom/deploy.log", True), f"deploy.log mount {log}"
    acc = vols.get("/var/lib/loom/acceptance")
    assert acc == ("/var/lib/loom/acceptance", False), f"/var/lib/loom/acceptance mount {acc}"
    aws_env = [n for n in env_names(svc) if n.startswith("AWS_")]
    assert not aws_env, f"acceptance gets AWS settings {aws_env}"
    assert not any(".aws" in s or ".aws" in t for s, t, _ in volume_specs(svc)), (
        "acceptance mounts AWS credentials"
    )


def test_the_caddyfile_comes_from_the_bundle() -> None:
    """§6.12: compose mounts the bundle's Caddyfile (beside compose.prod.yml) at
    /etc/caddy/Caddyfile, read-only; the image does not carry it."""
    hits = [
        (s, ro) for s, t, ro in volume_specs(services()["caddy"]) if t == "/etc/caddy/Caddyfile"
    ]
    assert len(hits) == 1, f"caddy mounts at /etc/caddy/Caddyfile: {hits}"
    source, ro = hits[0]
    assert not source.startswith(("/", "$", "~")), f"the Caddyfile {source} is not the bundle's"
    assert (PROD.parent / source).resolve() == (PROD.parent / "Caddyfile").resolve(), source
    assert ro, "the Caddyfile is not mounted read-only"


def test_the_caddy_image_is_built_from_deploy_caddy() -> None:
    """§6.7, §6.12: `deploy/caddy/Dockerfile`, build context `deploy/caddy/`, whose
    `go.mod` and `go.sum` pin the rate-limit module by version."""
    d = ROOT / "deploy" / "caddy"
    for f in ("Dockerfile", "go.mod", "go.sum"):
        assert (d / f).is_file(), f"deploy/caddy/{f} does not exist"
    gomod = (d / "go.mod").read_text()
    pin = re.search(r"^\s*(?:require\s+)?(\S*ratelimit\S*)\s+(v\d\S*)", gomod, re.M | re.I)
    assert pin, "deploy/caddy/go.mod does not pin a rate-limit module by version"
    assert pin.group(1) in (d / "go.sum").read_text(), "go.sum does not hold the module"


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


REDIS_SECRETS = SECRETS_DIR + "redis/"
DEVSTACK_SRC = ROOT / "services" / "devstack" / "src" / "loom_devstack"


def uncommented(text: str) -> list[str]:
    return [line for line in text.splitlines() if not line.lstrip().startswith("#")]


def rendered_redis_conf() -> list[str]:
    """Arguments of Redis's command line that point at a file loom-deploy renders under
    `/run/loom/secrets/redis/` (a config file that also holds `requirepass`, §6.3)."""
    svc = services()["redis"]
    argv = words(svc.get("entrypoint")) + words(svc.get("command"))
    return [a.strip("\"'") for a in argv if a.strip("\"'").startswith(REDIS_SECRETS)]


def redis_config() -> str:
    """Redis's settings as the repository states them: its command and entrypoint, plus
    any config file from deploy/ it mounts (a `redis.conf` is as good as flags).

    When Redis's argument is a config file loom-deploy renders under
    `/run/loom/secrets/redis/` (so `requirepass` can sit in it), the rendered file is not
    in the repository; its settings are then read from where it is rendered from: the
    files of deploy/ (a template) and loom-deploy's module (`loom_devstack/*deploy*.py`).
    The settings must be stated in one of those places: a static check cannot see a
    value computed at run time.
    """
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
            parts += [w for line in uncommented(path.read_text()) for w in line.split()]
    if rendered_redis_conf():
        renderers = [p for p in sorted(PROD.parent.rglob("*")) if p.is_file() and p != PROD]
        renderers += sorted(DEVSTACK_SRC.glob("*deploy*.py"))
        for path in renderers:
            if b"\0" not in path.read_bytes()[:4096]:
                text = path.read_text(encoding="utf-8", errors="replace")
                # A one-line template (`"port 0\ntls-port 6379\n"`) is read line by line.
                parts += uncommented(text.replace("\\n", "\n"))
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
    (contracts §2 checks the last with CONFIG GET). As flags, in a config file from
    deploy/, or (for a config file rendered under `/run/loom/secrets/redis/`) in what
    renders it: see `redis_config`. A15 checks the running Redis."""
    key, value = setting
    config = redis_config()
    # `key value`, `--key value`, or quoted/mapped as in a renderer (`"key": "value"`).
    pat = rf"(?<![\w-])(?:--)?{re.escape(key)}[\"']?[\s:=,]+[\"']?{re.escape(value)}(?![\w-])"
    assert re.search(pat, config), f"redis is not run with {key} {value}: {config[:2000]!r}"


def test_a_rendered_redis_config_file_keeps_the_password_out_of_the_compose_file() -> None:
    """If Redis reads a config file from `/run/loom/secrets/redis/`, that is the file
    loom-deploy renders with `requirepass` (§6.3): the compose file names it only by path
    and states no `requirepass` value of its own. (That Redis mounts its own secrets
    directory read-only is checked by the mount tests above; A13 checks the file.)"""
    confs = rendered_redis_conf()
    if not confs:
        return  # flags or a config file from deploy/: the tests above cover them
    text = "\n".join(uncommented(PROD.read_text()))
    assert not re.search(r"requirepass[\"']?[\s:=]+\S", text), (
        "compose.prod.yml states a requirepass value"
    )


def test_the_redis_password_is_not_on_its_command_line() -> None:
    """`requirepass` comes from the secret (a mounted config file), never as an argument."""
    svc = services()["redis"]
    argv = " ".join(words(svc.get("entrypoint")) + words(svc.get("command")))
    assert "requirepass" not in argv, f"redis gets requirepass on its command line: {argv!r}"


def dockerfile_cmd(path: Path) -> list[str]:
    """The last `CMD` of a Dockerfile, exec (`["all"]`) or shell form, as words."""
    cmds = re.findall(r"^\s*CMD\s+(.+?)\s*$", path.read_text(encoding="utf-8"), re.M)
    if not cmds:
        return []
    last = cmds[-1]
    if last.startswith("["):
        return [w for item in yaml.safe_load(last) for w in str(item).split()]
    return last.split()


def test_billing_runs_all() -> None:
    """§6.2: `billing` (`all`): the API, the webhook and the jobs in one service. In the
    compose file's command, or (with no `command` and no `entrypoint` override, which
    would reset it) the image's own `CMD` from services/billing/Dockerfile."""
    svc = services()["billing"]
    argv = words(svc.get("entrypoint")) + words(svc.get("command"))
    if svc.get("command") is None and svc.get("entrypoint") is None:
        argv = dockerfile_cmd(ROOT / "services" / "billing" / "Dockerfile")
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
