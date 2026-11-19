"""A15: the production stack comes up in production mode locally (M3 spec §6.12 "A15's
harness", A15, A29's live half, A35's sign-in page, A37's routes, P4's deploy log).

Acceptance-marked; CI job `m3-prod-stack` runs exactly this file with `-m acceptance
--timeout 3000`. Needs Docker with compose >= 2.24 (`!override`) and the network (the
third-party images are pulled by `up`).

The harness, as §6.12 fixes it:
- builds the Loom images from the repository's Dockerfiles (the web image without
  `LOOM_BRAND_NAME`, Caddy from `deploy/caddy/`) under local tags, and a bundle from
  `deploy/`; the manifest names the local tags and the third-party images as
  `deploy/compose.prod.yml` pins them;
- a fake `aws` serves the site parameter (`example.test`, a test name, `db_host` the
  stand-in Postgres), the manifest, the bundle, the RDS master secret and Secrets Manager
  (the user's Stripe, Google and SES values preset; `secrets init` makes the rest); a
  recording `chown` (nothing is chowned: every service runs as the test's uid) and a
  `docker` wrapper (the real docker, except `login` and `compose pull`, recorded and
  answered 0) are first on PATH;
- runs `loom-deploy secrets init`, then `apply`, with `LOOM_DEPLOY_ROOT` under /dev/shm,
  `LOOM_DEPLOY_PROJECT=loom-m3prod` and `LOOM_DEPLOY_COMPOSE_OVERRIDE` naming an override
  that adds a TLS Postgres standing in for RDS (a throwaway CA mounted at
  `/etc/loom/rds-ca.pem`, `migrate` depending on it, TCP connections without TLS rejected
  by its pg_hba), runs the services as the test's uid, moves Caddy's published ports off
  80/443 (to 127.0.0.1), sets `LOOM_CADDY_TLS=internal`, gives Caddy the network aliases
  `example.test`, `api.example.test` and `hooks.example.test`, and lets `acceptance` trust
  Caddy's internal root (`SSL_CERT_FILE` and the certifi bundle, both Caddy's root plus
  certifi's roots, mounted read-only);
- removes the project (`down -v`) and the root afterwards.

Where §6.12's "nothing else" cannot hold, this file's choices (the narrowest it found):
- `logging` is overridden to `json-file` for every service: the `awslogs` driver needs
  AWS credentials the local daemon does not have;
- `acceptance` keeps its image's uid (0, §6.12's table) instead of the test's: it needs the
  docker socket, which the test's uid may not open inside a container;
- A29's live half: Postgres rejects TCP without TLS (pg_hba `hostnossl ... reject`), so a
  service that connects at all connects with TLS; the test checks that the app role's
  connections are TLS ones and that every database URL rendered for the services is
  `verify-full` with the throwaway CA's path (a verify-full client that reached it
  verified the stand-in's certificate for the name `postgres`).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import traceback
import urllib.parse
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from devstack_m3_kit import (
    MASTER_ARN,
    ORDER,
    PROD,
    ROOT,
    Box,
    aws_op,
    deploy_root_base,
    read_env_file,
    rmtree,
    secret_values,
    site,
)

from loom_devstack.acceptance import visible

pytestmark = pytest.mark.acceptance

PROJECT = "loom-m3prod"
DOMAIN = "example.test"
NAME = "Prodstack Test Brand"
OPS = "ops@example.test"
SERVICES = ("caddy", "web", "gateway", "billing", "poolmgr", "redis", "prometheus", "grafana")
BUILDS = {  # manifest key -> (Dockerfile, build context), both from the repository root
    "gateway": ("services/gateway/Dockerfile", "."),
    "billing": ("services/billing/Dockerfile", "."),
    "web": ("apps/web/Dockerfile", "."),
    "poolmgr": ("services/poolmgr/Dockerfile", "."),
    "migrate": ("services/devstack/Dockerfile", "."),
    "caddy": ("deploy/caddy/Dockerfile", "deploy/caddy"),
}
THIRD_PARTY = ("redis", "prometheus", "grafana")
# The secret containers `secrets init` fills (§6.12, §5.2).
GENERATED = ("internal-token", "app-db-password", "auth", "redis", "pod-ca", "poolmgr-client",
             "grafana")  # fmt: skip
PG_IMAGE = "postgres:16"
BUNDLE_IN_CONTAINER = "/etc/loom-test/ca-bundle.pem"


def tag(svc: str) -> str:
    return f"{PROJECT}-test/{'devstack' if svc == 'migrate' else svc}:local"


def sh(*argv: str, timeout: float = 600, check: bool = True, **kw: Any) -> Any:
    r = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, **kw)
    if check and r.returncode != 0:
        raise AssertionError(
            f"{' '.join(argv[:6])}: exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
        )
    return r


@dataclass
class Stack:
    error: str | None = None
    box: Box | None = None
    sha: str = ""
    uid: int = field(default_factory=lambda: os.getuid())
    gid: int = field(default_factory=lambda: os.getgid())
    network: str = "default"
    ports: dict[int, int] = field(default_factory=dict)
    override: Path | None = None
    ca_bundle: Path | None = None
    caddy_root: Path | None = None
    apply_proc: Any = None
    base: Path | None = None  # the /dev/shm scratch root, set as soon as it exists
    notes: list[str] = field(default_factory=list)

    def need(self) -> Box:
        if self.error is not None:
            pytest.fail(self.error, pytrace=False)
        assert self.box is not None
        return self.box

    def compose(self, *args: str, timeout: float = 600, check: bool = True) -> Any:
        box = self.need()
        files = ["-f", str(box.var_lib / "releases" / self.sha / "compose.prod.yml")]
        files += ["-f", str(self.override)]
        return sh(
            "docker",
            "compose",
            "-p",
            PROJECT,
            *files,
            "--env-file",
            str(box.run_loom / "compose.env"),
            *args,
            timeout=timeout,
            check=check,
        )


# ---------------------------------------------------------------- set-up


def prerequisites() -> dict[str, Any]:
    """The cheap checks first, so the first failure names what is missing."""
    assert PROD.is_file(), "deploy/compose.prod.yml does not exist (the production stack, §6.2)"
    import importlib.util

    assert importlib.util.find_spec("loom_devstack.deploy"), "loom_devstack.deploy does not exist"
    assert (ROOT / "deploy" / "caddy" / "Dockerfile").is_file(), "deploy/caddy/Dockerfile missing"
    sh("docker", "info", "--format", "{{.ServerVersion}}", timeout=60)
    doc = yaml.safe_load(PROD.read_text())
    assert isinstance(doc, dict) and isinstance(doc.get("services"), dict), "no services"
    busy = []
    for name, svc in doc["services"].items():
        if name == "caddy":
            continue
        for port in host_ports(svc):
            if not port_free(port):
                busy.append(f"{name}:{port}")
    assert not busy, f"host ports in use (another stack running?): {busy}"
    return doc


def host_ports(svc: dict[str, Any]) -> list[int]:
    out = []
    for spec in svc.get("ports") or []:
        if isinstance(spec, dict):
            if spec.get("published") is not None:
                out.append(int(str(spec["published"]).split("-")[0]))
            continue
        parts = str(spec).split("/")[0].split(":")
        if len(parts) == 3:
            out.append(int(parts[1]))
        elif len(parts) == 2:
            out.append(int(parts[0]))
    return out


def port_free(port: int) -> bool:
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def caddy_network(doc: dict[str, Any]) -> str:
    nets = doc["services"]["caddy"].get("networks")
    if isinstance(nets, dict) and nets:
        return str(next(iter(nets)))
    if isinstance(nets, list) and nets:
        return str(nets[0])
    return "default"


def build_images() -> None:
    for svc, (dockerfile, context) in BUILDS.items():
        sh(
            "docker",
            "build",
            "-t",
            tag(svc),
            "-f",
            str(ROOT / dockerfile),
            str(ROOT / context),
            timeout=2400,
            env={**os.environ, "DOCKER_BUILDKIT": "1"},
        )


def pg_files(d: Path) -> None:
    """A throwaway CA and a server certificate for the name `postgres`."""
    d.mkdir()
    now = dt.datetime.now(dt.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "loom m3prod test RDS CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    crt = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "postgres")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=2))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("postgres")]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    (d / "ca.crt").write_bytes(ca.public_bytes(pem))
    (d / "server.crt").write_bytes(crt.public_bytes(pem))
    k = d / "server.key"
    k.write_bytes(
        key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    k.chmod(0o600)
    (d / "pg_hba.conf").write_text(
        "local all all trust\nhostssl all all all scram-sha-256\nhostnossl all all all reject\n"
    )


def user_store(master_pw: str) -> dict[str, str | None]:
    """The user's values (Stripe, SES) preset; `secrets init` makes the generated ones."""
    tok = lambda: secrets.token_urlsafe(24)  # noqa: E731
    return {
        "stripe": json.dumps(
            {"secret_key": "sk_test_" + tok(), "webhook_secret": "whsec_" + tok()}
        ),
        "auth": None,
        "internal-token": None,
        "app-db-password": None,
        "redis": None,
        "pod-ca": None,
        "poolmgr-client": None,
        "ses-smtp": json.dumps({"username": "AKIAM3PRODTEST0000", "password": tok()}),
        "grafana": None,
        "serving-runpod-key": None,
        MASTER_ARN: json.dumps({"username": "loom_owner", "password": master_pw}),
    }


def j(value: Any) -> str:
    return json.dumps(value)


def override_text(st: Stack, pg: Path) -> str:
    """The override of §6.12 (YAML; JSON values are YAML flow values)."""
    u = f"{st.uid}:{st.gid}"
    ca = f"{pg / 'ca.crt'}:/etc/loom/rds-ca.pem:ro"
    log = "logging: !override " + j({"driver": "json-file"})
    net = st.network
    out = ["services:"]

    def svc(name: str, *lines: str) -> None:
        out.append(f"  {name}:")
        out.extend(f"    {line}" for line in lines)

    svc(
        "postgres",
        f"image: {j(PG_IMAGE)}",
        f"user: {j(u)}",
        "environment: "
        + j(
            {
                "POSTGRES_USER": "loom_owner",
                "POSTGRES_PASSWORD_FILE": "/pg/password",
                "POSTGRES_DB": "postgres",
                "PGDATA": "/var/lib/postgresql/data/pgdata",
            }
        ),
        "command: "
        + j(
            [
                "postgres",
                "-c",
                "ssl=on",
                "-c",
                "ssl_cert_file=/pg/server.crt",
                "-c",
                "ssl_key_file=/pg/server.key",
                "-c",
                "hba_file=/pg/pg_hba.conf",
            ]
        ),
        f"volumes: {j([f'{pg}:/pg:ro'])}",
        f"networks: {j({net: {}})}",
        "healthcheck: "
        + j(
            {
                "test": ["CMD-SHELL", "pg_isready -U loom_owner -d postgres"],
                "interval": "2s",
                "timeout": "5s",
                "retries": 90,
            }
        ),
        "logging: " + j({"driver": "json-file"}),
    )
    svc(
        "migrate",
        f"user: {j(u)}",
        "depends_on: " + j({"postgres": {"condition": "service_healthy"}}),
        f"volumes: {j([ca])}",
        log,
    )
    for name in ("gateway", "billing", "web", "poolmgr"):
        svc(name, f"user: {j(u)}", f"volumes: {j([ca])}", log)
    for name in ("redis", "prometheus", "grafana"):
        svc(name, f"user: {j(u)}", log)
    svc(
        "caddy",
        f"user: {j(u)}",
        "ports: !override " + j([f"127.0.0.1:{st.ports[80]}:80", f"127.0.0.1:{st.ports[443]}:443"]),
        "environment: " + j({"LOOM_CADDY_TLS": "internal"}),
        "networks: " + j({net: {"aliases": [DOMAIN, f"api.{DOMAIN}", f"hooks.{DOMAIN}"]}}),
        log,
    )
    assert st.ca_bundle is not None
    certifi_path = certifi_in_devstack()
    svc(
        "acceptance",
        "environment: " + j({"SSL_CERT_FILE": BUNDLE_IN_CONTAINER}),
        "volumes: "
        + j(
            [
                ca,
                f"{st.ca_bundle}:{BUNDLE_IN_CONTAINER}:ro",
                f"{st.ca_bundle}:{certifi_path}:ro",
            ]
        ),
        log,
    )
    return "\n".join(out) + "\n"


def certifi_in_devstack() -> str:
    r = sh(
        "docker", "run", "--rm", "--entrypoint", "python", tag("migrate"),
        "-c", "import certifi; print(certifi.where())",
        timeout=120,
    )  # fmt: skip
    return str(r.stdout.strip())


def third_party_images(doc: dict[str, Any]) -> dict[str, str]:
    return {svc: str(doc["services"][svc]["image"]) for svc in THIRD_PARTY}


def deploy_env(st: Stack) -> dict[str, str | None]:
    # `Box.env` forwards DOCKER_HOST, DOCKER_CONTEXT (and the TLS variables) from the
    # test's environment, so the wrapper's real docker talks to the test's daemon.
    return {
        "LOOM_DEPLOY_PROJECT": PROJECT,
        "LOOM_DEPLOY_COMPOSE_OVERRIDE": str(st.override),
        "LOOM_DEPLOY_HEALTH_TIMEOUT_S": "600",
        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG") or str(Path.home() / ".docker"),
        "AWS_REGION": None,
    }


def find_caddy_root(st: Stack) -> bytes:
    box = st.need()
    found = sorted((box.var_lib / "caddy").rglob("pki/authorities/local/root.crt"))
    if found:
        return found[0].read_bytes()
    r = st.compose(
        "exec", "-T", "caddy", "sh", "-c",
        'cat "${XDG_DATA_HOME:-/data}/caddy/pki/authorities/local/root.crt"',
        timeout=60,
    )  # fmt: skip
    return str(r.stdout).encode()


def set_up(st: Stack) -> None:
    doc = prerequisites()
    st.network = caddy_network(doc)
    head = sh("git", "-C", str(ROOT), "rev-parse", "HEAD", timeout=30, check=False)
    st.sha = head.stdout.strip() if head.returncode == 0 else ""
    if not re.fullmatch(r"[0-9a-f]{40}", st.sha):  # a copy without .git: any release sha
        st.sha = secrets.token_hex(20)
    build_images()
    base = deploy_root_base()
    st.base = base
    master_pw = "own" + secrets.token_urlsafe(24)
    box = Box(
        base,
        fake_docker=False,
        store=user_store(master_pw),
        site_doc=site(domain=DOMAIN, brand_name=NAME, ops_email=OPS, db_host="postgres"),
    )
    st.box = box
    pg = base / "pg"
    pg_files(pg)
    pw = pg / "password"
    pw.write_text(master_pw)
    pw.chmod(0o600)
    st.ca_bundle = base / "ca-bundle.pem"
    st.ca_bundle.write_text("")  # filled once Caddy has made its root; a file, not a dir
    st.ports = {80: free_port(), 443: free_port()}
    st.override = base / "override.yml"
    st.override.write_text(override_text(st, pg))
    images = {svc: tag(svc) for svc in BUILDS} | third_party_images(doc)
    box.publish(st.sha, images)

    init = box.deploy("secrets", "init", timeout=300, **deploy_env(st))
    assert init.returncode == 0, f"secrets init: exit {init.returncode}\n{init.stderr[-3000:]}"
    store = box.load_store()
    auth = json.loads(str(store.get("auth") or "{}"))
    auth.update(
        google_id="m3prod-test.apps.googleusercontent.com", google_secret="GOCSPX-" + "x" * 28
    )
    store["auth"] = json.dumps(auth)
    box.store = store
    box.save_store()

    st.apply_proc = box.deploy("apply", "--release", st.sha, timeout=2700, **deploy_env(st))
    assert st.apply_proc.returncode == 0, (
        f"apply: exit {st.apply_proc.returncode}\n--- stdout\n{st.apply_proc.stdout[-4000:]}"
        f"\n--- stderr\n{st.apply_proc.stderr[-4000:]}\n--- ps\n"
        + sh("docker", "compose", "-p", PROJECT, "ps", "-a", check=False, timeout=60).stdout
    )
    root_pem = find_caddy_root(st)
    assert b"BEGIN CERTIFICATE" in root_pem, "Caddy's internal root certificate not found"
    st.caddy_root = base / "caddy-root.crt"
    st.caddy_root.write_bytes(root_pem)
    certifi_pem = sh(
        "docker", "run", "--rm", "--entrypoint", "cat", tag("migrate"), certifi_in_devstack(),
        timeout=120,
    ).stdout  # fmt: skip
    with open(st.ca_bundle, "w") as f:  # in place: the bind mount keeps the inode
        f.write(certifi_pem.rstrip("\n") + "\n" + root_pem.decode())


def tear_down(st: Stack) -> None:
    """Remove project `loom-m3prod` and the scratch root, from whatever point set-up
    reached (it may have been interrupted anywhere, e.g. by the timeout). Every step runs
    whatever an earlier one raised (a `down` that times out still leaves the label sweep
    and the root's removal to run); the first error is raised at the end, and a root left
    behind is an error too, so a leak stays visible."""
    box, base = st.box, st.base
    errors: list[BaseException] = []

    def step(fn: Any, *args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except BaseException as e:  # also pytest-timeout's Failed, KeyboardInterrupt
            errors.append(e)
            return None

    def down(box: Box) -> None:
        files: list[str] = []
        if st.override is not None:
            rel = box.var_lib / "releases" / st.sha / "compose.prod.yml"
            if rel.is_file() and (box.run_loom / "compose.env").is_file():
                env_file = str(box.run_loom / "compose.env")
                files = ["-f", str(rel), "-f", str(st.override), "--env-file", env_file]
        sh(
            "docker", "compose", "-p", PROJECT, *files,
            "--profile", "acceptance", "--profile", "migrate",
            "down", "-v", "--remove-orphans", "--timeout", "20",
            timeout=600, check=False, cwd=str(box.work),
        )  # fmt: skip

    def sweep(kind: str, ls: list[str]) -> None:
        ids = sh(*ls, check=False, timeout=60).stdout.split()
        if ids:
            rm = ["docker", "rm", "-f"] if kind == "container" else ["docker", kind, "rm"]
            sh(*rm, *ids, check=False, timeout=120)

    docker = shutil.which("docker") is not None
    if docker:
        if box is not None:  # (the label sweep below covers the rest)
            step(down, box)
        label = f"label=com.docker.compose.project={PROJECT}"
        for kind, ls in (
            ("container", ["docker", "ps", "-aq", "--filter", label]),
            ("volume", ["docker", "volume", "ls", "-q", "--filter", label]),
            ("network", ["docker", "network", "ls", "-q", "--filter", label]),
        ):
            step(sweep, kind, ls)
    if base is not None:
        step(rmtree, base)
        if docker and step(base.exists):  # files a root container wrote
            step(
                sh,
                "docker", "run", "--rm", "--entrypoint", "rm", "-v", f"{base}:/x",
                tag("migrate"), "-rf", "/x/root", "/x/state", "/x/pg",
                check=False, timeout=120,
            )  # fmt: skip
            step(rmtree, base)
        if step(base.exists):
            errors.append(AssertionError(f"tear-down left the scratch root {base} behind"))
    if errors:
        for e in errors[1:]:
            errors[0].add_note(f"tear-down, then: {type(e).__name__}: {e}")
        raise errors[0]


@pytest.fixture(scope="module")
def stack() -> Iterator[Stack]:
    st = Stack()
    # One try/finally around set-up and the tests: an interruption anywhere in set-up
    # (pytest-timeout's Failed, KeyboardInterrupt: BaseExceptions) still tears down.
    try:
        try:
            set_up(st)
        except Exception as e:  # recorded: each test fails with it (a failure, not an error)
            st.error = (
                f"{type(e).__name__}: {e}"
                if isinstance(e, AssertionError)
                else (traceback.format_exc()[-4000:])
            )
        yield st
    finally:
        tear_down(st)


# ---------------------------------------------------------------- the checks


def caddy_get(st: Stack, host: str, path: str, method: str = "GET", **kw: Any) -> httpx.Response:
    """A request through Caddy's moved HTTPS port, with the site's name as SNI and Host,
    verified against Caddy's internal root."""
    assert st.caddy_root is not None
    ctx = ssl.create_default_context(cafile=str(st.caddy_root))
    with httpx.Client(verify=ctx, timeout=30, trust_env=False, follow_redirects=False) as http:
        return http.request(
            method,
            f"https://127.0.0.1:{st.ports[443]}{path}",
            headers={"Host": host, **kw.pop("headers", {})},
            extensions={"sni_hostname": host},
            **kw,
        )


def test_a15_every_service_is_healthy_without_loom_env(stack: Stack) -> None:
    stack.need()
    out = stack.compose("ps", "-a", "--format", "json", timeout=60).stdout.strip()
    rows = (
        json.loads(out) if out.startswith("[") else [json.loads(x) for x in out.splitlines() if x]
    )
    by = {r["Service"]: r for r in rows}
    for svc in SERVICES:
        assert svc in by, f"no {svc} container ({sorted(by)})"
        r = by[svc]
        assert r.get("State") == "running", f"{svc}: {r.get('State')} {r.get('Status')}"
        assert r.get("Health") in ("healthy", ""), f"{svc}: health {r.get('Health')}"
        env = json.loads(sh("docker", "inspect", "-f", "{{json .Config.Env}}", r["ID"]).stdout)
        assert not [e for e in env if e.startswith("LOOM_ENV=")], f"{svc} runs with LOOM_ENV"


def test_a15_the_deploy_log_shows_the_p4_order(stack: Stack) -> None:
    from loom_devstack import m3_acceptance as m3

    box = stack.need()
    lines = (box.var_lib / "deploy.log").read_text().splitlines()
    m3.p4_log_order(lines, stack.sha)
    recs = [json.loads(x) for x in lines if x.strip()]
    mine = [r for r in recs if r.get("release") == stack.sha]
    assert not [r for r in mine if r["status"] in ("failed", "refused")], mine
    steps = [r["step"] for r in mine]
    for s in ("apply", "manifest", "render", "pull", "migrate", "update", "record"):
        assert s in steps, f"no {s} record"
    ups = [r["service"] for r in mine if r["step"] == "update" and r["status"] == "start"]
    assert ups == list(ORDER), ups
    # The manifest names local tags and docker.io images only: no image is in ECR, so no
    # `docker login` and no `aws ecr get-login-password` (§6.12 "ECR").
    logins = [c for c in box.calls("docker") if c[:1] == ["login"]]
    assert not logins, f"docker login without an ECR image: {logins}"
    ecr = [c for c in box.calls("aws") if aws_op(c) == ["ecr", "get-login-password"]]
    assert not ecr, f"ECR login password fetched without an ECR image: {ecr}"


ROUTED_404 = [
    (f"api.{DOMAIN}", "GET", "/internal/checkout-sessions"),
    (f"api.{DOMAIN}", "GET", "/health"),
    (f"api.{DOMAIN}", "GET", "/metrics"),
    (f"api.{DOMAIN}", "GET", "/"),
    (f"hooks.{DOMAIN}", "GET", "/internal/checkout-sessions"),
    (f"hooks.{DOMAIN}", "POST", "/internal/checkout-sessions"),
    (f"hooks.{DOMAIN}", "GET", "/health"),
    (f"hooks.{DOMAIN}", "GET", "/metrics"),
    (f"hooks.{DOMAIN}", "GET", "/stripe/webhook"),
    (f"hooks.{DOMAIN}", "GET", "/"),
    (DOMAIN, "GET", "/internal/checkout-sessions"),
    (DOMAIN, "GET", "/metrics"),
]


@pytest.mark.parametrize("host, method, path", ROUTED_404)
def test_a37_off_route_paths_answer_404_through_caddy(
    stack: Stack, host: str, method: str, path: str
) -> None:
    stack.need()
    r = caddy_get(stack, host, path, method)
    assert r.status_code == 404, f"{method} https://{host}{path}: {r.status_code}"
    assert "max-age=31536000" in r.headers.get("strict-transport-security", ""), "no HSTS"


def test_a37_the_public_routes_answer_through_caddy(stack: Stack) -> None:
    stack.need()
    health = caddy_get(stack, f"hooks.{DOMAIN}", "/stripe/health")
    assert health.status_code == 200 and "stripe_webhook" in health.text, (
        f"hooks. GET /stripe/health: {health.status_code} {health.text[:200]}"
    )
    hook = caddy_get(stack, f"hooks.{DOMAIN}", "/stripe/webhook", "POST", content=b"{}",
                     headers={"Content-Type": "application/json"})  # fmt: skip
    assert hook.status_code == 400, f"unsigned POST /stripe/webhook: {hook.status_code}"
    models = caddy_get(stack, f"api.{DOMAIN}", "/v1/models")
    assert models.status_code == 401, f"api. GET /v1/models without a key: {models.status_code}"
    for host in (DOMAIN, f"api.{DOMAIN}", f"hooks.{DOMAIN}"):
        with httpx.Client(timeout=30, trust_env=False, follow_redirects=False) as http:
            r = http.get(f"http://127.0.0.1:{stack.ports[80]}/", headers={"Host": host})
        assert r.status_code in (301, 302, 307, 308), f"http://{host}/: {r.status_code}"
        assert r.headers.get("location", "").startswith(f"https://{host}"), r.headers


def test_a35_the_signin_page_shows_the_brand_name(stack: Stack) -> None:
    """The web image was built without LOOM_BRAND_NAME; apply rendered it at run time."""
    stack.need()
    page = caddy_get(stack, DOMAIN, "/signin")
    assert page.status_code == 200, f"/signin: {page.status_code}"
    title = re.search(r"<title>(.*?)</title>", page.text, re.S)
    heading = re.search(r"<(h1|h2)(?:\s[^>]*)?>(.*?)</\1>", page.text, re.S)
    assert title and NAME in visible(title.group(1)), f"title {title and title.group(1)!r}"
    assert heading and NAME in visible(heading.group(2)), (
        f"heading {heading and heading.group(2)!r}"
    )


def test_a29_services_and_migrate_connect_with_tls(stack: Stack) -> None:
    box = stack.need()
    for p in (box.run_loom / "env").glob("*.env"):
        for k, v in read_env_file(p).items():
            if v.startswith("postgres"):
                q = urllib.parse.parse_qs(urllib.parse.urlsplit(v).query)
                assert q.get("sslmode") == ["verify-full"], f"{p.name} {k}"
                assert q.get("sslrootcert") == ["/etc/loom/rds-ca.pem"], f"{p.name} {k}"
    sql = (
        "SELECT a.usename, s.ssl FROM pg_stat_activity a JOIN pg_stat_ssl s USING (pid)"
        " WHERE a.backend_type = 'client backend' AND a.client_addr IS NOT NULL"
    )
    rows = stack.compose(
        "exec", "-T", "postgres", "psql", "-U", "loom_owner", "-d", "loom_platform",
        "-At", "-F", ",", "-c", sql,
        timeout=60,
    ).stdout.split()  # fmt: skip
    app = [r for r in rows if r.startswith("loom_app,")]
    assert app, f"no TCP connection of loom_app: {rows}"
    assert all(r.endswith(",t") for r in rows), f"connections without TLS: {rows}"
    su = stack.compose(
        "exec", "-T", "postgres", "psql", "-U", "loom_owner", "-d", "loom_platform", "-At",
        "-c", "SELECT rolsuper FROM pg_roles WHERE rolname = 'loom_app'",
        timeout=60,
    ).stdout.strip()  # fmt: skip
    assert su == "f", f"loom_app missing or a superuser ({su!r}): loom-migrate did not run"


def test_a15_m3_0_b_d_pass_in_the_acceptance_service(stack: Stack) -> None:
    box = stack.need()
    report = box.var_lib / "acceptance" / "report.json"
    r = stack.compose(
        "--profile", "acceptance", "run", "--rm", "-T", "acceptance",
        "loom-m3-acceptance",
        "--domain", DOMAIN,
        "--release", stack.sha,
        "--json", "/var/lib/loom/acceptance/report.json",
        "--deploy-log", "/var/lib/loom/deploy.log",
        "--admin-email", OPS,
        "--webhook-secret-file", "/run/loom/secrets/acceptance/webhook_secret",
        "--only", "m3-0,m3-b,m3-d",
        timeout=2400, check=False,
    )  # fmt: skip
    assert report.is_file(), (
        f"no report (exit {r.returncode})\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"
    )
    doc = json.loads(report.read_text())
    status = {i["item"]: i["status"] for i in doc["items"]}
    assert status == {"m3-0": "PASS", "m3-b": "PASS", "m3-d": "PASS"}, (
        f"{status}\n" + json.dumps([i for i in doc["items"] if i["status"] != "PASS"])[:4000]
    )
    assert r.returncode == 0, r.stdout[-3000:]


def test_a13_no_generated_secret_on_a_command_line(stack: Stack) -> None:
    """The real run's argv (aws, docker, chown) holds none of the values init made, each
    unwrapped from the JSON object it may be stored in."""
    box = stack.need()
    store = box.load_store()
    argv = box.all_argv()
    for name in GENERATED:
        assert store.get(name), f"secrets init left {name} without a value"
        values = secret_values({name: store[name]})
        assert values, f"no secret value found in {name}"
        for v in values:
            assert v not in argv, f"a value of {name} is on a command line"
