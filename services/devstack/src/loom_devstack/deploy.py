"""loom-deploy: the production deploy on the box (docs/phase1-m3-spec.md §6.3, §6.6, §6.12).

    loom-deploy secrets init              generate the secrets Loom makes (once; never overwrites)
    loom-deploy ses-smtp-password         derive the SES SMTP password from an IAM access key
    loom-deploy apply --release <sha>     fetch the release, render secrets, migrate, update
    loom-deploy status                    the current release and the compose services
    loom-deploy rollback                  apply the previous release
    loom-deploy acceptance --p4-release <sha> [-- <loom-m3-acceptance options>]

This is one module that uses **only the standard library** (a test enforces it), so it runs
on the box with the host's Python 3.12 and no packages: `python3.12 -S loom-deploy.py <command>`
(the release bundle carries a copy; it also runs on the system Python 3.9) or
`python -S -m loom_devstack.deploy`. It calls the `aws` CLI, `docker compose`, `openssl`
and `chown`; it imports nothing from its own package.

Every box path is under `LOOM_DEPLOY_ROOT` (unset on the box): `<root>/run/loom/` (rendered
secrets and env files; must be a tmpfs) and `<root>/var/lib/loom/` (the release record, the
deploy log, the extracted releases, the data volumes). The compose project is
`LOOM_DEPLOY_PROJECT` (default `loom`), always passed as `-p`. Exit codes: 0 done, 2 refused
(the reason on stderr, nothing changed), 1 any other failure.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets as pysecrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Any

SITE_PARAMETER = "/loom/platform/site"
SECRET_PREFIX = "loom/platform/"
RDS_CA_PATH = "/etc/loom/rds-ca.pem"
DATABASE = "loom_platform"
APP_USER = "loom_app"
DEFAULT_REGION = "us-east-1"
ACME_STAGING = "https://acme-staging-v02.api.letsencrypt.org/directory"

# §6.6 step 4: the update order. `redis` only when its image changed.
ORDER = ("redis", "billing", "gateway", "poolmgr", "web", "caddy", "prometheus", "grafana")
# §6.12: the compose.env variable of each Loom image, by manifest key.
IMAGE_VARS = {
    "LOOM_IMAGE_GATEWAY": "gateway",
    "LOOM_IMAGE_BILLING": "billing",
    "LOOM_IMAGE_WEB": "web",
    "LOOM_IMAGE_POOLMGR": "poolmgr",
    "LOOM_IMAGE_DEVSTACK": "migrate",
    "LOOM_IMAGE_CADDY": "caddy",
}
# §6.12: the uid each image runs as (the owner of its rendered secrets).
UIDS = {
    "gateway": 10001,
    "billing": 10001,
    "web": 1001,
    "poolmgr": 0,
    "migrate": 0,
    "acceptance": 0,
    "redis": 999,
    "grafana": 472,
}
# §6.12: the volume directories under <root>/var/lib/loom and their owner.
VOLUMES = {
    "redis": 999,
    "gateway-spool": 10001,
    "caddy": 0,
    "poolmgr": 0,
    "prometheus": 65534,
    "gateway-cert": 10001,
}
ENV_SERVICES = ("caddy", "web", "gateway", "billing", "poolmgr", "migrate", "acceptance")
REDIS_CLIENTS = ("gateway", "billing", "poolmgr", "web", "acceptance")
SITE_STRINGS = (
    "domain",
    "brand_name",
    "ops_email",
    "releases_bucket",
    "db_host",
    "db_master_secret_arn",
)
# The pool manager's name for the gateway's client certificate file (loom_poolmgr.pki).
GATEWAY_CERT_FILE = "loom-gateway.pem"
SHA = re.compile(r"[0-9a-f]{40}")
ECR_REGISTRY = re.compile(r"^(\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com)/")
DOMAIN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$", re.I)
HOST = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?$")
EMAIL = re.compile(r"^[^@\s<>\"',;]+@[^@\s<>\"',;]+$")
HEALTH_FORMAT = "{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}"
STARTED_FORMAT = "{{.State.StartedAt}}"


class Refused(Exception):
    """The apply (or acceptance) is refused: exit 2, nothing changed."""


class DeployError(Exception):
    """Any other failure: exit 1."""


# ---------------------------------------------------------------------------- environment


class Paths:
    """The box paths, under LOOM_DEPLOY_ROOT (unset on the box)."""

    def __init__(self) -> None:
        raw = os.environ.get("LOOM_DEPLOY_ROOT", "")
        self.root_text = raw
        self.root = Path(raw) if raw else Path("/")
        self.run_loom = self.root / "run" / "loom"
        self.var_lib = self.root / "var" / "lib" / "loom"
        self.release_json = self.var_lib / "release.json"
        self.deploy_log = self.var_lib / "deploy.log"
        self.acceptance = self.var_lib / "acceptance"
        self.compose_env = self.run_loom / "compose.env"


def region() -> str:
    return os.environ.get("AWS_REGION") or DEFAULT_REGION


def project() -> str:
    return os.environ.get("LOOM_DEPLOY_PROJECT") or "loom"


def health_timeout() -> float:
    try:
        return max(1.0, float(os.environ.get("LOOM_DEPLOY_HEALTH_TIMEOUT_S", "300")))
    except ValueError:
        return 300.0


def now() -> str:
    # dt.UTC is Python 3.11+; the box's system python3 is 3.9
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")  # noqa: UP017


def say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    inherit: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a tool. Nothing secret is ever an argument; secrets go in `input_text`. A failure
    names the tool and its first words only, with the tail of its stderr (never its input)."""
    try:
        r = subprocess.run(
            list(cmd),
            input=input_text if input_text is not None else ("" if not inherit else None),
            capture_output=not inherit,
            text=True,
            cwd=cwd,
        )
    except OSError as e:
        raise DeployError(f"{cmd[0]}: cannot run: {e}") from None
    if check and r.returncode != 0:
        words = " ".join(w for w in cmd[1:4] if not w.startswith("-"))
        tail = (r.stderr or "").strip()[-400:] if not inherit else ""
        raise DeployError(f"{cmd[0]} {words} failed (exit {r.returncode}) {tail}".strip())
    return r


def aws(
    *args: str, input_text: str | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        ["aws", *args, "--region", region(), "--no-cli-pager"], input_text=input_text, check=check
    )


# ---------------------------------------------------------------------------- the deploy log


class DeployLog:
    """`<root>/var/lib/loom/deploy.log`: JSON Lines, appended and printed (§6.12)."""

    def __init__(self, paths: Paths, release: str) -> None:
        self.paths, self.release = paths, release

    def record(self, step: str, status: str, **extra: Any) -> None:
        rec: dict[str, Any] = {"ts": now(), "release": self.release, "step": step, "status": status}
        for key in ("service", "only", "detail"):
            if key in extra and (extra[key] is not None or key == "only"):
                rec[key] = extra[key]
        line = json.dumps(rec)
        self.paths.var_lib.mkdir(parents=True, exist_ok=True)
        with open(self.paths.deploy_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)


# ---------------------------------------------------------------------------- checks


def mount_fstype(path: Path) -> str | None:
    """The filesystem type of the mount that holds `path` (or its nearest existing parent),
    from /proc/self/mountinfo; None when it cannot be read."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    real = os.path.realpath(p)
    best, fstype = "", None
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        left, sep, right = line.partition(" - ")
        fields = left.split()
        if not sep or len(fields) < 5 or not right.split():
            continue
        point = fields[4].replace("\\040", " ")
        inside = real == point or real.startswith(point.rstrip("/") + "/")
        if inside and len(point) >= len(best):
            best, fstype = point, right.split()[0]
    return fstype


def check_tmpfs(paths: Paths) -> None:
    kind = mount_fstype(paths.run_loom)
    if kind != "tmpfs":
        raise Refused(
            f"{paths.run_loom} is not on a tmpfs (filesystem: {kind or 'unknown'}): rendered "
            "secrets must never reach a disk"
        )


def check_database_url(url: str) -> None:
    """D10 (§6.4): a database URL must name `sslmode=verify-full` and the RDS CA file."""
    u = urllib.parse.urlsplit(url)
    if not (
        u.scheme == "postgres" or u.scheme == "postgresql" or u.scheme.startswith("postgresql+")
    ):
        raise Refused("a database URL is not a postgresql:// URL")
    query = urllib.parse.parse_qs(u.query, keep_blank_values=True)
    if query.get("sslmode") != ["verify-full"]:
        raise Refused("a database URL lacks sslmode=verify-full (D10)")
    if query.get("sslrootcert") != [RDS_CA_PATH]:
        raise Refused(f"a database URL lacks sslrootcert={RDS_CA_PATH} (D10)")


def validate_site(doc: Any) -> dict[str, Any]:
    """§6.11, §6.12: every key present and non-empty, and sane (the domain and the name end
    up in host names and a mail sender)."""
    if not isinstance(doc, dict):
        raise Refused(f"the site parameter {SITE_PARAMETER} is not a JSON object")
    problems: list[str] = []
    for key in SITE_STRINGS:
        value = doc.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"`{key}` is missing or empty")
    port = doc.get("db_port")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        problems.append("`db_port` is missing or not a port number")
    if not problems:
        if not DOMAIN.match(doc["domain"]):
            problems.append("`domain` is not a plain domain name")
        name = doc["brand_name"]
        # The name is the display name of EMAIL_FROM (`<name> <no-reply@<domain>>`), so it
        # holds none of the characters RFC 5322 treats as special there.
        if (
            len(name) > 64
            or any(c in name for c in '"<>\\,;:@()[]')
            or any(ord(c) < 32 for c in name)
        ):
            problems.append(
                "`brand_name` is longer than 64 characters or holds a control character or "
                'one of , ; : @ ( ) [ ] < > " \\'
            )
        if not EMAIL.match(doc["ops_email"]):
            problems.append("`ops_email` is not an email address")
        if not HOST.match(doc["db_host"]):
            problems.append("`db_host` is not a host name")
    if problems:
        raise Refused(f"the site parameter {SITE_PARAMETER}: " + "; ".join(problems))
    # The URLs apply renders from it must pass D10.
    check_database_url(db_url(APP_USER, "x", doc["db_host"], doc["db_port"]))
    return doc


# ---------------------------------------------------------------------------- plan_steps


def _running(current: dict[str, Any] | None) -> dict[str, str]:
    """The release each service runs: `current["running"]` (loom-deploy's own record), else
    every service runs `current["sha"]`."""
    if not current:
        return {}
    running = current.get("running")
    if isinstance(running, dict):
        return {str(k): str(v) for k, v in running.items()}
    return dict.fromkeys(ORDER, str(current.get("sha")))


def _image(manifest: dict[str, Any] | None, service: str) -> Any:
    return ((manifest or {}).get("images") or {}).get(service)


def plan_steps(
    current: dict[str, Any] | None,
    target: dict[str, Any],
    only: Collection[str] | None = None,
) -> list[tuple[str, ...]]:
    """The apply's steps, a pure function (§6.6): `("migrate",)`, then `("up", <service>)` in
    the P4 order (redis only if its image changed; with `only`, exactly those services).
    Raises `Refused` naming the gateway when the poolmgr would run another release than
    the gateway (P4). `current` is the running release's manifest (None before the first
    deploy), optionally with `running`: the release of each service."""
    if only is None:
        names = [
            s
            for s in ORDER
            if s != "redis" or current is None or _image(current, s) != _image(target, s)
        ]
    else:
        wanted = set(only)
        unknown = sorted(wanted - set(ORDER))
        if unknown:
            raise Refused(f"unknown service(s) in --only: {', '.join(unknown)}")
        names = [s for s in ORDER if s in wanted]
    sha = str(target["sha"])
    running = _running(current)
    gateway = sha if "gateway" in names else running.get("gateway")
    poolmgr = sha if "poolmgr" in names else running.get("poolmgr")
    if "poolmgr" in names and poolmgr != gateway:
        runs = f"release {str(gateway)[:12]}" if gateway else "nothing"
        raise Refused(
            f"P4: poolmgr would run release {sha[:12]} while the gateway runs {runs}; the "
            "gateway must be updated first (include gateway in --only, or apply the "
            "gateway's release before the poolmgr's)"
        )
    return [("migrate",), *(("up", s) for s in names)]


# ---------------------------------------------------------------------------- AWS reads


def secret_id(name: str) -> str:
    return SECRET_PREFIX + name


def get_secret(secret: str) -> str | None:
    """A secret's value, None when the container has no value (or does not exist)."""
    r = aws("secretsmanager", "get-secret-value", "--secret-id", secret, check=False)
    if r.returncode != 0:
        if "ResourceNotFoundException" in (r.stderr or ""):
            return None
        raise DeployError(f"aws secretsmanager get-secret-value failed: {r.stderr.strip()[-300:]}")
    try:
        value = json.loads(r.stdout)["SecretString"]
    except (ValueError, KeyError, TypeError):
        raise DeployError("aws secretsmanager get-secret-value: no SecretString") from None
    return str(value)


def has_value(name: str) -> bool:
    r = aws("secretsmanager", "describe-secret", "--secret-id", secret_id(name), check=False)
    if r.returncode != 0:
        raise DeployError(
            f"secret {secret_id(name)} cannot be read (is the Terraform applied?): "
            f"{(r.stderr or '').strip()[-300:]}"
        )
    try:
        doc = json.loads(r.stdout)
    except ValueError:
        raise DeployError("aws secretsmanager describe-secret: not JSON") from None
    return bool(doc.get("VersionIdsToStages"))


def put_secret(name: str, value: str) -> None:
    aws(
        "secretsmanager",
        "put-secret-value",
        "--secret-id",
        secret_id(name),
        "--secret-string",
        "file:///dev/stdin",
        input_text=value,
    )


def read_site() -> dict[str, Any]:
    r = aws("ssm", "get-parameter", "--name", SITE_PARAMETER)
    try:
        return json.loads(json.loads(r.stdout)["Parameter"]["Value"])  # type: ignore[no-any-return]
    except (ValueError, KeyError, TypeError):
        raise DeployError(f"{SITE_PARAMETER} is not a JSON document") from None


# ---------------------------------------------------------------------------- secrets init


def _crypto_dir() -> Path:
    base = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
    d = Path(tempfile.mkdtemp(prefix="loom-pki-", dir=base))
    d.chmod(0o700)
    return d


def _openssl(work: Path, *args: str) -> None:
    run_cmd(["openssl", *args], cwd=work)


def _genkey(work: Path, name: str) -> None:
    _openssl(
        work, "genpkey", "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:P-256", "-out", name
    )
    (work / name).chmod(0o600)


def make_ca(cn: str, days: int) -> tuple[str, str]:
    """A self-signed EC P-256 CA (path length 0): (certificate PEM, private key PEM)."""
    work = _crypto_dir()
    try:
        _genkey(work, "ca.key")
        _openssl(
            work, "req", "-x509", "-new", "-key", "ca.key", "-sha256", "-days", str(days),
            "-subj", f"/O=Loom/CN={cn}",
            "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-addext", "subjectKeyIdentifier=hash",
            "-out", "ca.crt",
        )  # fmt: skip
        return (work / "ca.crt").read_text(), (work / "ca.key").read_text()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def issue_cert(
    ca_crt: str, ca_key: str, cn: str, *, server: bool, dns: Sequence[str], days: int
) -> tuple[str, str]:
    """A leaf signed by the CA: (certificate PEM, private key PEM); serverAuth with the
    given names, or clientAuth."""
    work = _crypto_dir()
    try:
        for name, text in (("ca.crt", ca_crt), ("ca.key", ca_key)):
            p = work / name
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(text)
        _genkey(work, "leaf.key")
        _openssl(
            work, "req", "-new", "-key", "leaf.key", "-subj", f"/O=Loom/CN={cn}", "-out", "leaf.csr"
        )
        ext = [
            "basicConstraints=critical,CA:FALSE",
            "keyUsage=critical,digitalSignature",
            f"extendedKeyUsage={'serverAuth' if server else 'clientAuth'}",
            "subjectKeyIdentifier=hash",
            "authorityKeyIdentifier=keyid",
        ]
        if dns:
            ext.append("subjectAltName=" + ",".join(f"DNS:{d}" for d in dns))
        (work / "ext.cnf").write_text("\n".join(ext) + "\n")
        _openssl(
            work, "x509", "-req", "-in", "leaf.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
            "-CAcreateserial", "-days", str(days), "-sha256", "-extfile", "ext.cnf",
            "-out", "leaf.crt",
        )  # fmt: skip
        return (work / "leaf.crt").read_text(), (work / "leaf.key").read_text()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def token() -> str:
    return pysecrets.token_urlsafe(32)


def cmd_secrets_init() -> int:
    """§6.3: generate and store the values Loom makes, without echoing them; never overwrite a
    secret that has a value; never touch the user's (stripe, ses-smtp) or `serving-runpod-key`."""
    generated = (
        "internal-token",
        "app-db-password",
        "auth",
        "redis",
        "pod-ca",
        "poolmgr-client",
        "grafana",
    )
    present = {name: has_value(name) for name in generated}
    new: dict[str, str] = {}
    if not present["internal-token"]:
        new["internal-token"] = token()
    if not present["app-db-password"]:
        new["app-db-password"] = token()
    if not present["auth"]:
        new["auth"] = json.dumps({"auth_secret": pysecrets.token_urlsafe(48)})
    if not present["redis"]:
        ca_crt, ca_key = make_ca("Loom Redis CA", 3650)
        crt, key = issue_cert(
            ca_crt, ca_key, "redis", server=True, dns=["redis", "localhost"], days=3650
        )
        new["redis"] = json.dumps(
            {
                "password": pysecrets.token_urlsafe(32),
                "ca_crt": ca_crt,
                "server_crt": crt,
                "server_key": key,
            }
        )
    pod_ca: dict[str, str] | None = None
    if not present["pod-ca"]:
        crt, key = make_ca("Loom pod CA", 730)
        pod_ca = {"crt": crt, "key": key}
        new["pod-ca"] = json.dumps(pod_ca)
    if not present["poolmgr-client"]:
        if pod_ca is None:
            existing = get_secret(secret_id("pod-ca"))
            try:
                pod_ca = json.loads(existing or "")
                assert pod_ca and pod_ca["crt"] and pod_ca["key"]
            except (ValueError, KeyError, AssertionError, TypeError):
                raise DeployError(
                    "secret pod-ca has no crt and key to sign the client certificate"
                ) from None
        crt, key = issue_cert(
            pod_ca["crt"],
            pod_ca["key"],
            "loom-poolmgr",
            server=False,
            dns=["loom-poolmgr"],
            days=700,
        )
        new["poolmgr-client"] = json.dumps({"crt": crt, "key": key})
    if not present["grafana"]:
        new["grafana"] = json.dumps({"admin_password": pysecrets.token_urlsafe(24)})
    for name in generated:
        if name in new:
            put_secret(name, new[name])
            print(f"secrets init: generated {secret_id(name)}")
        else:
            print(f"secrets init: {secret_id(name)} already has a value, left as it is")
    return 0


def ses_smtp_password(secret_access_key: str, for_region: str) -> str:
    """AWS's documented derivation (SES developer guide, "Obtaining Amazon SES SMTP
    credentials"): the SigV4 key chain with the date 11111111 over SendRawEmail, version 4."""

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    sig = sign(("AWS4" + secret_access_key).encode(), "11111111")
    for part in (for_region, "ses", "aws4_request", "SendRawEmail"):
        sig = sign(sig, part)
    return base64.b64encode(bytes([0x04]) + sig).decode()


def cmd_ses_smtp_password() -> int:
    """Reads `aws iam create-access-key`'s JSON on stdin (never a command-line argument),
    stores {"username", "password"} in loom/platform/ses-smtp; prints neither."""
    try:
        key = json.loads(sys.stdin.read())["AccessKey"]
        key_id, secret = key["AccessKeyId"], key["SecretAccessKey"]
        if not (isinstance(key_id, str) and key_id and isinstance(secret, str) and secret):
            raise ValueError("empty")
    except (ValueError, KeyError, TypeError):
        raise DeployError(
            "stdin is not `aws iam create-access-key` output (AccessKey.AccessKeyId and "
            "SecretAccessKey)"
        ) from None
    put_secret(
        "ses-smtp",
        json.dumps({"username": key_id, "password": ses_smtp_password(secret, region())}),
    )
    print("ses-smtp-password: stored loom/platform/ses-smtp")
    return 0


# ---------------------------------------------------------------------------- the release


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def s3_get(bucket: str, key: str, dest: Path) -> None:
    aws("s3", "cp", f"s3://{bucket}/{key}", str(dest))


def fetch_release(site: dict[str, Any], sha: str, work: Path, paths: Paths) -> dict[str, Any]:
    """§6.12: the manifest `releases/<sha>.json`, the compose file and the bundle it names,
    both sha256s checked, the bundle extracted to `<root>/var/lib/loom/releases/<sha>/`
    (and its compose.prod.yml checked to be the named one). Returns the manifest."""
    bucket = site["releases_bucket"]
    mpath = work / "manifest.json"
    s3_get(bucket, f"releases/{sha}.json", mpath)
    try:
        manifest = json.loads(mpath.read_text())
    except ValueError:
        raise DeployError(f"releases/{sha}.json is not JSON") from None
    if not isinstance(manifest, dict) or manifest.get("sha") != sha:
        raise DeployError(f"releases/{sha}.json does not describe release {sha}")
    images = manifest.get("images")
    if not isinstance(images, dict) or not all(isinstance(v, str) and v for v in images.values()):
        raise DeployError("the manifest's images are not all non-empty strings")
    missing = [s for s in IMAGE_VARS.values() if not images.get(s)]
    if missing:
        raise DeployError(f"the manifest names no image for {', '.join(missing)}")
    files: dict[str, Path] = {}
    for what in ("compose", "bundle"):
        entry = manifest.get(what)
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str)
        ):
            raise DeployError(f"the manifest has no {what} path and sha256")
        rel = entry["path"]
        if rel.startswith("/") or ".." in Path(rel).parts:
            raise DeployError(f"the manifest's {what} path {rel!r} is not a release key")
        dest = work / what
        s3_get(bucket, rel, dest)
        if sha256_of(dest) != entry["sha256"]:
            raise DeployError(f"the {what} file does not match the manifest's sha256")
        files[what] = dest
    target = paths.var_lib / "releases" / sha
    target.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(files["bundle"], "r:gz") as tar:
            if hasattr(tarfile, "data_filter"):  # 3.12, and the 3.9.17+ / 3.10.12+ backports
                tar.extractall(target, filter="data")
            else:  # an older 3.9: the same rule by hand, regular files and directories only
                members = tar.getmembers()
                for member in members:
                    parts = Path(member.name).parts
                    if (
                        member.name.startswith("/")
                        or ".." in parts
                        or not (member.isfile() or member.isdir())
                    ):
                        raise DeployError(f"the bundle holds an unsafe member {member.name!r}")
                    member.mode &= ~0o7000  # no setuid, setgid or sticky bit (the data filter)
                    # never apply the archive's owner: tarfile chowns only when running as
                    # root, so blank it to root (the running user), as the data filter does
                    member.uid = member.gid = 0
                    member.uname = member.gname = ""
                tar.extractall(target, members=members)
    except DeployError:
        raise
    except (tarfile.TarError, OSError) as e:
        raise DeployError(f"the bundle does not extract: {e}") from None
    extracted = target / "compose.prod.yml"
    if not extracted.is_file() or sha256_of(extracted) != manifest["compose"]["sha256"]:
        raise DeployError("the bundle's compose.prod.yml is not the one the manifest names")
    return manifest  # type: ignore[no-any-return]


def load_release(paths: Paths) -> dict[str, Any] | None:
    try:
        doc = json.loads(paths.release_json.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def current_view(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    """`release.json` as plan_steps' `current`: the release's manifest and who runs what."""
    if not rec or not isinstance(rec.get("manifest"), dict):
        return None
    return {**rec["manifest"], "sha": rec.get("release"), "running": rec.get("services") or {}}


# ---------------------------------------------------------------------------- rendering


def quote_env(value: str) -> str:
    """One `KEY=value` value for a compose env file: literal (no `$` interpolation)."""
    if "\n" in value or "\r" in value or "\0" in value:
        raise DeployError("an environment value holds a newline")
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "$$") + '"'


def env_text(values: dict[str, str]) -> str:
    return "".join(f"{k}={quote_env(v)}\n" for k, v in values.items())


def write_file(path: Path, data: str, mode: int) -> None:
    """Replace `path` atomically with `data` at `mode` (no moment with another mode)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(fd, mode)
        os.write(fd, data.encode())
    finally:
        os.close(fd)
    os.replace(tmp, path)


def chown(owner: str, path: Path, *, recursive: bool = False) -> None:
    run_cmd(["chown", *(["-R"] if recursive else []), owner, str(path)])


def dir_digest(d: Path) -> str:
    h = hashlib.sha256()
    if d.is_dir():
        for p in sorted(d.iterdir()):
            if p.is_file():
                h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def db_url(user: str, password: str, host: str, port: int) -> str:
    q = urllib.parse.quote
    return (
        f"postgresql://{q(user, safe='')}:{q(password, safe='')}@{host}:{port}/{DATABASE}"
        f"?sslmode=verify-full&sslrootcert={RDS_CA_PATH}"
    )


def redis_url(password: str, service: str) -> str:
    return (
        f"rediss://:{urllib.parse.quote(password, safe='')}@redis:6379/0"
        f"?ssl_ca_certs=/run/loom/secrets/{service}/redis-ca.crt"
    )


def json_secret(store: dict[str, str | None], name: str, *keys: str) -> dict[str, str]:
    raw = store.get(name)
    if not raw:
        raise DeployError(f"secret {secret_id(name)} has no value")
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = None
    if not isinstance(doc, dict):
        raise DeployError(f"secret {secret_id(name)} is not a JSON object")
    missing = [k for k in keys if not isinstance(doc.get(k), str) or not doc[k]]
    if missing:
        raise DeployError(f"secret {secret_id(name)} lacks {', '.join(missing)}")
    return {k: str(v) for k, v in doc.items() if isinstance(v, str)}


def plain_secret(store: dict[str, str | None], name: str) -> str:
    raw = store.get(name)
    if not raw:
        raise DeployError(f"secret {secret_id(name)} has no value")
    with contextlib.suppress(ValueError):
        doc = json.loads(raw)
        if isinstance(doc, dict) and len(doc) == 1:
            raw = str(next(iter(doc.values())))
    return raw.rstrip("\r\n")


def read_secrets(site: dict[str, Any]) -> dict[str, str | None]:
    names = (
        "stripe",
        "auth",
        "internal-token",
        "app-db-password",
        "redis",
        "pod-ca",
        "poolmgr-client",
        "ses-smtp",
        "grafana",
    )
    store: dict[str, str | None] = {n: get_secret(secret_id(n)) for n in names}
    store["master"] = get_secret(site["db_master_secret_arn"])
    return store


def build_render(
    paths: Paths, site: dict[str, Any], manifest: dict[str, Any], store: dict[str, str | None]
) -> tuple[dict[str, str], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """(compose.env values, env files by service, secret files by service), all in memory."""
    domain, name = site["domain"], site["brand_name"]
    master = json_secret(store, "master", "username", "password")
    app_pw = plain_secret(store, "app-db-password")
    token_value = plain_secret(store, "internal-token")
    stripe = json_secret(store, "stripe", "secret_key", "webhook_secret")
    auth = json_secret(store, "auth", "auth_secret")
    redis = json_secret(store, "redis", "password", "ca_crt", "server_crt", "server_key")
    pod_ca = json_secret(store, "pod-ca", "crt", "key")
    client = json_secret(store, "poolmgr-client", "crt", "key")
    ses = json_secret(store, "ses-smtp", "username", "password")
    grafana = json_secret(store, "grafana", "admin_password")

    host, port = site["db_host"], site["db_port"]
    app_url = db_url(APP_USER, app_pw, host, port)
    owner_url = db_url(master["username"], master["password"], host, port)
    for url in (app_url, owner_url):
        check_database_url(url)
    rpw = redis["password"]
    q = urllib.parse.quote

    compose_env = {"LOOM_DEPLOY_ROOT": paths.root_text}
    for var, svc in IMAGE_VARS.items():
        compose_env[var] = manifest["images"][svc]

    caddy_env = {"LOOM_DOMAIN": domain, "LOOM_CADDY_TLS": site["ops_email"]}
    if os.environ.get("LOOM_DEPLOY_ACME_STAGING", "").lower() in ("1", "true", "yes"):
        caddy_env["LOOM_ACME_CA"] = ACME_STAGING
    web_env = {
        "LOOM_BRAND_NAME": name,
        "AUTH_URL": f"https://{domain}",
        "LOOM_PUBLIC_API_URL": f"https://api.{domain}/v1",
        "EMAIL_FROM": f"{name} <no-reply@{domain}>",
        "AUTH_SECRET": auth["auth_secret"],
        "EMAIL_SERVER": (
            f"smtps://{q(ses['username'], safe='')}:{q(ses['password'], safe='')}"
            f"@email-smtp.{region()}.amazonaws.com:465"
        ),
        "DATABASE_URL": app_url,
        "LOOM_DATABASE_URL": app_url,
        "LOOM_REDIS_URL": redis_url(rpw, "web"),
        "LOOM_INTERNAL_TOKEN": token_value,
    }
    if auth.get("google_id") and auth.get("google_secret"):
        web_env["AUTH_GOOGLE_ID"] = auth["google_id"]
        web_env["AUTH_GOOGLE_SECRET"] = auth["google_secret"]
    envs = {
        "caddy": caddy_env,
        "web": web_env,
        "gateway": {
            "LOOM_DATABASE_URL": app_url,
            "LOOM_REDIS_URL": redis_url(rpw, "gateway"),
            "LOOM_CORS_ORIGINS": f"https://{domain}",
        },
        "billing": {
            "LOOM_DATABASE_URL": app_url,
            "LOOM_REDIS_URL": redis_url(rpw, "billing"),
            "LOOM_BILLING_INTERNAL_TOKEN": token_value,
            "STRIPE_SECRET_KEY": stripe["secret_key"],
            "STRIPE_WEBHOOK_SECRET": stripe["webhook_secret"],
            "LOOM_BRAND_NAME": name,
        },
        "poolmgr": {
            "LOOM_DATABASE_URL": app_url,
            "LOOM_REDIS_URL": redis_url(rpw, "poolmgr"),
        },
        "migrate": {"LOOM_DATABASE_URL": owner_url, "LOOM_APP_DATABASE_PASSWORD": app_pw},
        "acceptance": {
            "LOOM_ACCEPT_DATABASE_URL": app_url,
            "LOOM_ACCEPT_REDIS_URL": redis_url(rpw, "acceptance"),
        },
    }
    ca = redis["ca_crt"]
    secret_files: dict[str, dict[str, str]] = {svc: {"redis-ca.crt": ca} for svc in REDIS_CLIENTS}
    secret_files["gateway"]["pod-ca.crt"] = pod_ca["crt"]
    secret_files["poolmgr"].update(
        {
            "ca.crt": pod_ca["crt"],
            "ca.key": pod_ca["key"],
            "client.crt": client["crt"],
            "client.key": client["key"],
            # M1's overhead probe (loom-m3-acceptance m3-d runs it in this container) takes
            # `ca.crt`, `gateway.crt` and `gateway.key` from /certs; the manager's client
            # certificate is signed by the same CA, which is all a pod proxy checks.
            "gateway.crt": client["crt"],
            "gateway.key": client["key"],
        }
    )
    secret_files["acceptance"]["webhook_secret"] = stripe["webhook_secret"]
    secret_files["redis"] = {
        "ca.crt": ca,
        "server.crt": redis["server_crt"],
        "server.key": redis["server_key"],
        "requirepass.conf": f"requirepass {rpw}\n",
    }
    secret_files["grafana"] = {"admin_password": grafana["admin_password"]}
    return compose_env, envs, secret_files


def bootstrap_gateway_cert(paths: Paths, ca_crt: str, ca_key: str) -> None:
    """Gives the gateway a client certificate before the manager's first issue: the manager,
    which issues and rotates it (§7), starts only after the gateway is healthy (P4). Written
    only when the file is missing (never replacing the manager's), as the manager writes it
    (`loom-gateway.pem`, key then certificate in one PEM, 0400, owned by the gateway's uid,
    atomic rename), signed by the current pod CA for 7 days, so the manager's start check
    accepts it and rotates it on its 24 h schedule. A failure is a warning, not an error."""
    path = paths.var_lib / "gateway-cert" / GATEWAY_CERT_FILE
    if path.exists():
        return
    try:
        crt, key = issue_cert(ca_crt, ca_key, "loom-gateway", server=False, dns=[], days=7)
    except DeployError as e:
        say(f"warning: no initial gateway client certificate ({e}); the gateway will not start")
        return
    # Its own temp name (never the manager's `.loom-gateway.pem.tmp`), complete and owned
    # before it is placed, and placed with a hard link, which fails if the manager's file
    # appeared meanwhile (an existing file is never replaced).
    uid = UIDS["gateway"]
    tmp = path.with_name(f".loom-gateway-bootstrap-{os.getpid()}.tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        try:
            os.write(fd, (key + crt).encode())
            try:
                os.fchown(fd, uid, uid)
            except PermissionError:  # not root (a test box): the chown tool, as for the rest
                chown(f"{uid}:{uid}", tmp)
            os.fsync(fd)
        finally:
            os.close(fd)
        with contextlib.suppress(FileExistsError):
            os.link(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def render(paths: Paths, site: dict[str, Any], manifest: dict[str, Any]) -> set[str]:
    """§6.3, §6.12: read the secrets with the instance role and render everything the
    compose file reads under <root>/run/loom (env files 0400, secret directories 0500 with
    0400 files owned by the service's uid), and create the volume directories. Returns the
    services whose secret files changed (they are recreated even with an unchanged image)."""
    store = read_secrets(site)
    compose_env, envs, secret_files = build_render(paths, site, manifest, store)
    run = paths.run_loom
    run.mkdir(parents=True, exist_ok=True)
    (run / "env").mkdir(exist_ok=True)
    (run / "env").chmod(0o700)
    write_file(paths.compose_env, env_text(compose_env), 0o400)
    for svc, values in envs.items():
        write_file(run / "env" / f"{svc}.env", env_text(values), 0o400)
    changed: set[str] = set()
    (run / "secrets").mkdir(exist_ok=True)
    for svc, files in secret_files.items():
        d = run / "secrets" / svc
        before = dir_digest(d)
        d.mkdir(exist_ok=True)
        d.chmod(0o700)
        for stale in list(d.iterdir()):
            if stale.name in files:
                continue
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        for fname, text in files.items():
            write_file(d / fname, text, 0o400)
        owner = f"{UIDS[svc]}:{UIDS[svc]}"
        chown(owner, d, recursive=True)
        d.chmod(0o500)
        if before != dir_digest(d):
            changed.add(svc)
    for sub, uid in VOLUMES.items():
        v = paths.var_lib / sub
        v.mkdir(parents=True, exist_ok=True)
        chown(f"{uid}:{uid}", v)
    paths.acceptance.mkdir(parents=True, exist_ok=True)
    paths.deploy_log.touch(exist_ok=True)
    bootstrap_gateway_cert(
        paths, secret_files["poolmgr"]["ca.crt"], secret_files["poolmgr"]["ca.key"]
    )
    return changed


# ---------------------------------------------------------------------------- docker compose


class Compose:
    """`docker compose` for the release: always `-p <project>`, the release's file (and
    LOOM_DEPLOY_COMPOSE_OVERRIDE, tests only), compose.env, run from the release directory."""

    def __init__(self, paths: Paths, sha: str) -> None:
        self.dir = paths.var_lib / "releases" / sha
        files = [self.dir / "compose.prod.yml"]
        override = os.environ.get("LOOM_DEPLOY_COMPOSE_OVERRIDE")
        if override:
            files.append(Path(override))
        self.base = ["docker", "compose", "-p", project()]
        for f in files:
            self.base += ["-f", str(f)]
        self.base += ["--env-file", str(paths.compose_env)]

    def run(
        self, *args: str, check: bool = True, inherit: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return run_cmd([*self.base, *args], check=check, inherit=inherit, cwd=self.dir)

    def container(self, service: str) -> str | None:
        r = self.run("ps", "-q", service, check=False)
        ids = r.stdout.split() if r.returncode == 0 else []
        return ids[0] if ids else None

    def status(self, service: str) -> str:
        cid = self.container(service)
        if not cid:
            return ""
        r = run_cmd(["docker", "inspect", "-f", HEALTH_FORMAT, cid], check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    def started_at(self, service: str) -> str | None:
        cid = self.container(service)
        if not cid:
            return None
        r = run_cmd(["docker", "inspect", "-f", STARTED_FORMAT, cid], check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    def wait_healthy(self, service: str, timeout: float) -> None:
        """Healthy (or running, for a service without a healthcheck) within `timeout`."""
        deadline = time.monotonic() + timeout
        last = ""
        while True:
            last = self.status(service)
            if last in ("healthy", "running"):
                return
            if last in ("exited", "dead"):
                raise DeployError(f"{service} is {last}")
            if time.monotonic() >= deadline:
                raise DeployError(
                    f"{service} not healthy after {timeout:.0f} s (last: {last or 'no container'})"
                )
            time.sleep(1)


def ecr_login(manifest: dict[str, Any]) -> None:
    """`docker login` once per ECR registry the images name; the password on stdin."""
    registries = sorted(
        {m.group(1) for v in manifest["images"].values() if (m := ECR_REGISTRY.match(str(v)))}
    )
    for registry in registries:
        password = aws("ecr", "get-login-password").stdout.strip()
        run_cmd(
            ["docker", "login", "--username", "AWS", "--password-stdin", registry],
            input_text=password,
        )


# ---------------------------------------------------------------------------- apply


def parse_only(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    return [s.strip() for v in values for s in v.split(",") if s.strip()]


def write_release(
    paths: Paths,
    manifest: dict[str, Any],
    previous: dict[str, Any] | None,
    names: Iterable[str],
    full: bool,
) -> None:
    """`release.json`: the current release (a full apply's; an `--only` apply keeps it), the
    one before it (for rollback), the release each service runs, and the release's manifest."""
    sha = manifest["sha"]
    old = previous or {}
    services = dict(old.get("services") or {})
    services.update(dict.fromkeys(ORDER if full else names, sha))
    if full or not old:
        release, release_manifest = sha, manifest
        before = old.get("release") if old.get("release") != sha else old.get("previous")
    else:
        release, release_manifest, before = old["release"], old["manifest"], old.get("previous")
    rec = {
        "release": release,
        "previous": before,
        "services": services,
        "applied_at": now(),
        "manifest": release_manifest,
    }
    write_file(paths.release_json, json.dumps(rec, indent=2) + "\n", 0o644)


def apply_release(sha: str, only: list[str] | None) -> int:
    """`loom-deploy apply` (§6.6, §6.12). Returns the exit code."""
    paths = Paths()
    log = DeployLog(paths, sha)
    try:
        check_tmpfs(paths)
        site = validate_site(read_site())
        if only is not None:
            # P4 needs only the release's sha, not its manifest: refuse before any fetch,
            # extract or write (§6.12: a refused apply logs one record and changes nothing).
            plan_steps(current_view(load_release(paths)), {"sha": sha}, only)
    except Refused as e:
        log.record("apply", "refused", detail=str(e))
        say(f"refused: {e}")
        return 2
    except (DeployError, OSError) as e:
        log.record("apply", "failed", detail=str(e))
        say(f"failed: {e}")
        return 1
    log.record("apply", "start", only=only)
    step = "manifest"
    try:
        log.record("manifest", "start")
        with tempfile.TemporaryDirectory(prefix="loom-deploy-") as tmp:
            manifest = fetch_release(site, sha, Path(tmp), paths)
        log.record("manifest", "ok")

        previous = load_release(paths)
        try:
            steps = plan_steps(current_view(previous), manifest, only)
        except Refused as e:
            log.record("apply", "refused", detail=str(e))
            say(f"refused: {e}")
            return 2

        step = "render"
        log.record("render", "start")
        changed = render(paths, site, manifest)
        log.record("render", "ok")

        compose = Compose(paths, sha)
        step = "pull"
        log.record("pull", "start")
        ecr_login(manifest)
        compose.run("--profile", "migrate", "--profile", "acceptance", "pull")
        log.record("pull", "ok")

        step = "migrate"
        log.record("migrate", "start")
        r = compose.run("run", "--rm", "-T", "migrate", check=False, inherit=True)
        if r.returncode != 0:
            raise DeployError(f"loom-migrate exited {r.returncode}")
        log.record("migrate", "ok")

        names = [s[1] for s in steps if s[0] == "up"]
        forced = set(changed)
        # A same-release apply after a reboot (the secret files were gone) must bring every
        # service back: a redis that is not healthy, or whose secret files changed, is
        # recreated too.
        if (
            only is None
            and "redis" not in names
            and (compose.status("redis") not in ("healthy", "running") or "redis" in changed)
        ):
            names.insert(0, "redis")
            forced.add("redis")
        timeout = health_timeout()
        step = "update"
        for svc in names:
            log.record("update", "start", service=svc)
            try:
                cmd = [
                    "up",
                    "-d",
                    "--no-deps",
                    *(["--force-recreate"] if svc in forced else []),
                    svc,
                ]
                compose.run(*cmd)
                compose.wait_healthy(svc, timeout)
            except DeployError as e:
                log.record("update", "failed", service=svc, detail=str(e))
                raise
            log.record("update", "ok", service=svc)

        step = "record"
        log.record("record", "start")
        write_release(paths, manifest, previous, names, full=only is None)
        log.record("record", "ok")
    except (DeployError, OSError) as e:  # OSError: render's mkdir/chmod/write, the record
        if step in ("manifest", "render", "pull", "migrate", "record"):
            log.record(step, "failed", detail=str(e))
        log.record("apply", "failed", detail=str(e))
        say(f"failed: {e}")
        return 1
    except Refused as e:  # a check inside a step (D10): nothing was rendered yet
        log.record("apply", "refused", detail=str(e))
        say(f"refused: {e}")
        return 2
    log.record("apply", "ok")
    return 0


def cmd_rollback() -> int:
    paths = Paths()
    rec = load_release(paths)
    previous = rec.get("previous") if rec else None
    if not isinstance(previous, str) or not SHA.fullmatch(previous):
        say("refused: no previous release recorded (release.json)")
        return 2
    return apply_release(previous, None)


def cmd_status() -> int:
    paths = Paths()
    rec = load_release(paths)
    if not rec:
        print("no release recorded")
        return 0
    print(f"release {rec.get('release')} (previous {rec.get('previous')})")
    for svc, sha in sorted((rec.get("services") or {}).items()):
        print(f"  {svc}: {sha}")
    compose = Compose(paths, str(rec.get("release")))
    r = compose.run("ps", check=False)
    print(r.stdout)
    return 0


# ---------------------------------------------------------------------------- acceptance


def selected(extra: Sequence[str], item: str) -> bool:
    """Whether `loom-m3-acceptance` runs `item` (§6.12 "Item selection"): the LAST `--only`
    (`--only X` or `--only=X`, comma-separated, spaces ignored) names the list, else every
    box item; `--external` runs only m3-h, so never m3-p."""
    if "--external" in extra:
        return item == "m3-h"
    value: str | None = None
    skip = False
    for i, word in enumerate(extra):
        if skip:
            skip = False
            continue
        if word == "--only" and i + 1 < len(extra):
            value = extra[i + 1]
            skip = True
        elif word.startswith("--only="):
            value = word.split("=", 1)[1]
    if value is None:
        return item != "m3-h"
    return item in [s.strip() for s in value.split(",")]


def check_p4_refusal(
    paths: Paths, compose: Compose, current: str, p4_release: str
) -> dict[str, Any]:
    """m3-p's host half: `apply --only poolmgr --release <p4-release>` exits 2 naming the
    gateway, and the poolmgr container is not recreated."""
    if p4_release == current:
        return {
            "ok": False,
            "detail": "--p4-release is the current release: apply --only poolmgr is not refused",
        }
    before = compose.started_at("poolmgr")
    code = apply_release(p4_release, ["poolmgr"])
    after = compose.started_at("poolmgr")
    last = ""
    with contextlib.suppress(OSError):
        lines = paths.deploy_log.read_text().splitlines()
        last = lines[-1] if lines else ""
    if code != 2:
        return {"ok": False, "detail": f"apply --only poolmgr exited {code}, not 2"}
    if "gateway" not in last.lower():
        return {"ok": False, "detail": "the refusal does not name the gateway"}
    if before != after:
        return {"ok": False, "detail": "the poolmgr container was recreated"}
    return {"ok": True, "detail": "refused, naming the gateway; poolmgr not recreated"}


def cmd_acceptance(p4_release: str, extra: list[str]) -> int:
    paths = Paths()
    try:
        check_tmpfs(paths)
    except Refused as e:
        say(f"refused: {e}")
        return 2
    rec = load_release(paths)
    if not rec or not SHA.fullmatch(str(rec.get("release"))):
        say("failed: no current release recorded (release.json): apply one first")
        return 1
    current = str(rec["release"])
    site = validate_site(read_site())
    compose = Compose(paths, current)
    cmd = [
        "--profile", "acceptance", "run", "--rm", "-T", "acceptance", "loom-m3-acceptance",
        "--domain", site["domain"],
        "--release", current,
        "--json", "/var/lib/loom/acceptance/report.json",
        "--deploy-log", "/var/lib/loom/deploy.log",
        "--admin-email", site["ops_email"],
        "--webhook-secret-file", "/run/loom/secrets/acceptance/webhook_secret",
        *extra,
    ]  # fmt: skip
    paths.acceptance.mkdir(parents=True, exist_ok=True)
    r = compose.run(*cmd, check=False, inherit=True)
    host: dict[str, Any] = {}
    if selected(extra, "m3-p"):
        host["m3-p"] = check_p4_refusal(paths, compose, current, p4_release)
    else:
        host["m3-p"] = {"ok": None, "detail": "not selected"}  # never true: it did not run
    write_file(paths.acceptance / "host.json", json.dumps(host, indent=2) + "\n", 0o644)
    for item, res in host.items():
        label = "SKIP" if res["ok"] is None else ("PASS" if res["ok"] else "FAIL")
        say(f"{item}: {label}: {res['detail']}")
    ok = r.returncode == 0 and all(v["ok"] is not False for v in host.values())
    return 0 if ok else 1


# ---------------------------------------------------------------------------- the command line


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loom-deploy",
        description="Loom's production deploy on the box (docs/phase1-m3-spec.md §6).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    secrets_p = sub.add_parser("secrets", help="manage the secrets Loom makes")
    ssub = secrets_p.add_subparsers(dest="secrets_command", required=True)
    ssub.add_parser("init", help="generate the secrets Loom makes (never overwrites a value)")
    sub.add_parser(
        "ses-smtp-password",
        help="derive the SES SMTP password from `aws iam create-access-key` JSON on stdin",
    )
    ap = sub.add_parser("apply", help="deploy a release (the P4 order)")
    ap.add_argument("--release", required=True, help="the release's git sha (40 hex)")
    ap.add_argument(
        "--only", action="append", metavar="SERVICE", help="update only these services (repeatable)"
    )
    sub.add_parser("status", help="the current release and the compose services")
    sub.add_parser("rollback", help="apply the previous release")
    cp = sub.add_parser(
        "acceptance",
        help="run loom-m3-acceptance on the box, then m3-p's host-side refusal",
        description="loom-deploy acceptance --p4-release <sha> [-- <loom-m3-acceptance options>]",
    )
    cp.add_argument(
        "--p4-release", required=True, help="any release but the current one (m3-p's refusal)"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in args_list:
        i = args_list.index("--")
        args_list, extra = args_list[:i], args_list[i + 1 :]
    args = parser().parse_args(args_list)
    if extra and args.command != "acceptance":
        parser().error("`--` options are for `acceptance`")
    try:
        if args.command == "secrets":
            return cmd_secrets_init()
        if args.command == "ses-smtp-password":
            return cmd_ses_smtp_password()
        if args.command == "apply":
            if not SHA.fullmatch(args.release):
                parser().error("--release must be a 40-character git sha")
            return apply_release(args.release, parse_only(args.only))
        if args.command == "status":
            return cmd_status()
        if args.command == "rollback":
            return cmd_rollback()
        if args.command == "acceptance":
            if not SHA.fullmatch(args.p4_release):
                parser().error("--p4-release must be a 40-character git sha")
            return cmd_acceptance(args.p4_release, extra)
    except Refused as e:
        say(f"refused: {e}")
        return 2
    except DeployError as e:
        say(f"failed: {e}")
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
