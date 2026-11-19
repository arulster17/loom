"""Shared fakes for the M3 deploy tests (`loom-deploy apply` and `acceptance`, A13, A15).

Not a test module (no `test_` prefix): imported by `test_devstack_m3_deploy_apply.py`,
`test_devstack_m3_deploy_acceptance.py` and `test_devstack_m3_prodstack.py`, which pytest
loads with their own directory on `sys.path`.

`Box` is a scratch `LOOM_DEPLOY_ROOT` (under `/dev/shm`, a tmpfs, as §6.12 asks) plus a
directory of fake tools put first on `PATH`:

- `aws`: the site parameter (`ssm get-parameter`), the releases bucket (`s3 cp`, `s3api
  get-object`), Secrets Manager (`get-secret-value`, `describe-secret`,
  `list-secret-version-ids`, `put-secret-value`), `ecr get-login-password`, `logs
  filter-log-events` (an event for each configured "hit" the filter pattern holds) and `sts
  get-caller-identity`;
- `chown`: records its argv and changes nothing (the test is not root);
- `docker`: either a fake (`fake_docker=True`: answers `login`, `compose pull`, `compose run
  --rm <svc>`, `compose up -d --no-deps <svc>`, `compose ps -q <svc>` and `inspect` as
  §6.12 describes, from a small config) or a wrapper around the real docker that records
  and answers only `login` and `compose pull` (A15's harness).

Every call of every tool is appended to one `timeline.jsonl` (`{"tool", "argv", ...}`), so
a test can check the order across tools. The tools' state lives in `<scratch>/state`,
outside the deploy root, and is never scanned for secrets (it holds them by design).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.parse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "services" / "devstack" / "src"
DEPLOY_DIR = ROOT / "deploy"
PROD = DEPLOY_DIR / "compose.prod.yml"

REGION_DEFAULT = "us-east-1"
BUCKET = "loom-test-releases"
MASTER_ARN = "arn:aws:secretsmanager:us-east-1:123456789012:secret:rds!db-0a1b2c3d-AbCdEf"
REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"

# §6.6 step 4.
ORDER = ["redis", "billing", "gateway", "poolmgr", "web", "caddy", "prometheus", "grafana"]
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
# §6.12: the volume directories `apply` creates under <root>/var/lib/loom, and their owner.
VOLUMES = {
    "redis": 999,
    "gateway-spool": 10001,
    "caddy": 0,
    "poolmgr": 0,
    "prometheus": 65534,
    "gateway-cert": 10001,
}
# §6.12 "What apply writes": compose.env's image variables, by manifest key.
IMAGE_VARS = {
    "LOOM_IMAGE_GATEWAY": "gateway",
    "LOOM_IMAGE_BILLING": "billing",
    "LOOM_IMAGE_WEB": "web",
    "LOOM_IMAGE_POOLMGR": "poolmgr",
    "LOOM_IMAGE_DEVSTACK": "migrate",
    "LOOM_IMAGE_CADDY": "caddy",
}
LOOM_IMAGES = ("gateway", "billing", "web", "poolmgr", "migrate", "caddy")
THIRD_PARTY = ("redis", "prometheus", "grafana")


def site(**over: Any) -> dict[str, Any]:
    """The site parameter of §6.12 (`db_port` a JSON number); `None` drops a key."""
    doc: dict[str, Any] = {
        "domain": "example.test",
        "brand_name": "Quillwork Test",
        "ops_email": "ops@example.test",
        "releases_bucket": BUCKET,
        "db_host": "db.internal.example.test",
        "db_port": 5432,
        "db_master_secret_arn": MASTER_ARN,
    }
    doc.update(over)
    return {k: v for k, v in doc.items() if v is not None}


def pem(kind: str) -> str:
    body = secrets.token_urlsafe(96)
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return f"-----BEGIN {kind}-----\n" + "\n".join(lines) + f"\n-----END {kind}-----\n"


def full_store() -> dict[str, str | None]:
    """Every secret container of §5.2 with a value (the user's and the generated ones),
    plus the RDS master secret; `serving-runpod-key` has a value the box must never read.
    The master password holds characters a URL must percent-encode."""
    tok = lambda: "s" + secrets.token_urlsafe(24)  # noqa: E731
    return {
        "stripe": json.dumps(
            {"secret_key": "sk_test_" + tok(), "webhook_secret": "whsec_" + tok()}
        ),
        "auth": json.dumps(
            {"auth_secret": tok(), "google_id": "gid-" + tok(), "google_secret": tok()}
        ),
        "internal-token": tok(),
        "app-db-password": tok(),
        "redis": json.dumps(
            {
                "password": tok(),
                "ca_crt": pem("CERTIFICATE"),
                "server_crt": pem("CERTIFICATE"),
                "server_key": pem("PRIVATE KEY"),
            }
        ),
        "pod-ca": json.dumps({"crt": pem("CERTIFICATE"), "key": pem("PRIVATE KEY")}),
        "poolmgr-client": json.dumps({"crt": pem("CERTIFICATE"), "key": pem("PRIVATE KEY")}),
        "ses-smtp": json.dumps(
            {"username": "AKIAEXAMPLE" + secrets.token_hex(4).upper(), "password": tok()}
        ),
        "grafana": json.dumps({"admin_password": tok()}),
        "serving-runpod-key": "rpa_" + tok(),
        MASTER_ARN: json.dumps({"username": "loom_owner", "password": "p@ss/w:rd%" + tok()}),
    }


# Fields that are secret (certificates and user names are not).
SECRET_FIELDS = {
    "secret_key",
    "webhook_secret",
    "auth_secret",
    "google_secret",
    "password",
    "server_key",
    "key",
    "admin_password",
}


def secret_values(store: dict[str, str | None]) -> list[str]:
    """Every secret value in a store, with the forms it may be rendered in (a private key's
    base64 lines, a password percent-encoded)."""
    out: list[str] = []
    for value in store.values():
        if not value:
            continue
        try:
            doc = json.loads(value)
        except ValueError:
            doc = value
        if isinstance(doc, dict):
            vals = [str(v) for k, v in doc.items() if k in SECRET_FIELDS and v]
        else:
            vals = [str(doc)]
        for v in vals:
            if "PRIVATE KEY" in v:
                out += [ln for ln in v.splitlines() if len(ln) >= 40]
            else:
                out.append(v)
                quoted = urllib.parse.quote(v, safe="")
                if quoted != v:
                    out.append(quoted)
    return [v for v in out if len(v) >= 12]


def field(store: dict[str, str | None], name: str, key: str) -> str:
    value = store[name]
    assert value, f"no value for {name}"
    return str(json.loads(value)[key])


# ---------------------------------------------------------------------------- the fakes

_LOG = r"""
def log(tool, argv, **extra):
    rec = {"tool": tool, "argv": argv}
    rec.update(extra)
    line = json.dumps(rec) + "\n"
    fd = os.open(os.path.join(D, "timeline.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)
"""

FAKE_AWS = (
    r"""#!__PYTHON__ -IS
import json, os, re, shutil, sys
D = __DIR__
"""
    + _LOG
    + r"""
argv = sys.argv[1:]
log("aws", argv)
opts, pos, i = {}, [], 0
while i < len(argv):
    a = argv[i]
    if a.startswith("--"):
        if "=" in a:
            k, v = a[2:].split("=", 1); opts[k] = v
        elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            opts[a[2:]] = argv[i + 1]; i += 1
        else:
            opts[a[2:]] = True
    else:
        pos.append(a)
    i += 1
svc, op = (pos + ["", ""])[:2]
rest = pos[2:]

def load(v):
    if isinstance(v, str) and v.startswith("file://"):
        path = v[len("file://"):]
        return sys.stdin.read() if path in ("/dev/stdin", "-") else open(path).read()
    return v

if isinstance(opts.get("cli-input-json"), str):
    for k, v in json.loads(load(opts.pop("cli-input-json"))).items():
        opts.setdefault(re.sub(r"(?<!^)(?=[A-Z])", "-", k).lower(), v)

def out(obj):
    q = opts.get("query")
    if isinstance(q, str) and q:
        for part in q.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
    if opts.get("output") == "text" and isinstance(obj, (str, int, float)):
        print(obj)
    else:
        print(json.dumps(obj))
    sys.exit(0)

def fail(code, msg, rc=254):
    sys.stderr.write(f"An error occurred ({code}) when calling the {op} operation: {msg}\n")
    sys.exit(rc)

def s3_path(bucket, key):
    return os.path.join(D, "s3", bucket, key)

if svc == "ssm" and op in ("get-parameter", "get-parameters"):
    names = [opts.get("name")] if op == "get-parameter" else rest or [opts.get("names")]
    site = os.path.join(D, "site.json")
    if "/loom/platform/site" not in names or not os.path.exists(site):
        fail("ParameterNotFound", "parameter not found")
    value = open(site).read()
    p = {"Name": "/loom/platform/site", "Type": "String", "Value": value, "Version": 1}
    out({"Parameter": p} if op == "get-parameter" else {"Parameters": [p], "InvalidParameters": []})

if svc == "s3" and op == "cp":
    src, dst = (rest + ["", ""])[:2]
    m = re.fullmatch(r"s3://([^/]+)/(.+)", src)
    if not m:
        fail("Usage", "only downloads are faked", 1)
    path = s3_path(m.group(1), m.group(2))
    if not os.path.isfile(path):
        sys.stderr.write("fatal error: An error occurred (404) when calling the HeadObject "
                         f"operation: Key {m.group(2)!r} does not exist\n")
        sys.exit(1)
    if dst == "-":
        sys.stdout.buffer.write(open(path, "rb").read()); sys.exit(0)
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(m.group(2)))
    shutil.copyfile(path, dst)
    print(f"download: {src} to {dst}"); sys.exit(0)

if svc == "s3api" and op == "get-object":
    path = s3_path(str(opts.get("bucket")), str(opts.get("key")))
    if not os.path.isfile(path):
        fail("NoSuchKey", "The specified key does not exist.")
    dst = rest[0] if rest else None
    if dst in (None, "-", "/dev/stdout"):
        sys.stdout.buffer.write(open(path, "rb").read()); sys.exit(0)
    shutil.copyfile(path, dst)
    print(json.dumps({"ContentLength": os.path.getsize(path)})); sys.exit(0)

if svc == "secretsmanager":
    STORE = os.path.join(D, "secrets.json")
    store = json.load(open(STORE))
    sid = str(opts.get("secret-id", ""))
    if sid in store:
        name = sid
    else:
        m = re.search(r"loom/platform/([A-Za-z0-9-]+)$", sid)
        name = m.group(1) if m else sid
        if name not in store and sid.startswith("arn:"):
            s = re.fullmatch(r"(.+)-[A-Za-z0-9]{6}", name)
            name = s.group(1) if s else name
    if name not in store:
        fail("ResourceNotFoundException", "Secrets Manager can't find the specified secret.")
    full = name if name.startswith("arn:") else f"loom/platform/{name}"
    prefix = "arn:aws:secretsmanager:us-east-1:123456789012:secret:"
    arn = name if name.startswith("arn:") else f"{prefix}{full}-AbCdEf"
    value = store[name]
    if op == "get-secret-value":
        if value is None:
            fail("ResourceNotFoundException", "Secrets Manager can't find the specified "
                 "secret value for staging label: AWSCURRENT")
        out({"ARN": arn, "Name": full, "VersionId": "v1", "SecretString": value,
             "VersionStages": ["AWSCURRENT"]})
    if op == "describe-secret":
        d = {"ARN": arn, "Name": full}
        if value is not None:
            d["VersionIdsToStages"] = {"v1": ["AWSCURRENT"]}
        out(d)
    if op == "list-secret-version-ids":
        versions = [] if value is None else [{"VersionId": "v1", "VersionStages": ["AWSCURRENT"]}]
        out({"ARN": arn, "Name": full, "Versions": versions})
    if op == "put-secret-value":
        s = load(opts.get("secret-string"))
        if not isinstance(s, str):
            fail("InvalidParameterException", "no --secret-string")
        store[name] = s
        json.dump(store, open(STORE, "w"))
        out({"ARN": arn, "Name": full, "VersionId": "v2", "VersionStages": ["AWSCURRENT"]})
    fail("InvalidAction", f"not faked: secretsmanager {op}")

if svc == "ecr" and op == "get-login-password":
    print(open(os.path.join(D, "ecr-token")).read()); sys.exit(0)

if svc == "logs" and op == "filter-log-events":
    hits_file = os.path.join(D, "log-hits.json")
    hits = json.load(open(hits_file)) if os.path.exists(hits_file) else []
    pattern = str(opts.get("filter-pattern", ""))
    events = [{"logStreamName": "gateway/abc", "timestamp": 1, "message": "... " + h + " ...",
               "eventId": str(n)} for n, h in enumerate(hits) if h and h in pattern]
    print(json.dumps({"events": events, "searchedLogStreams": []})); sys.exit(0)

if svc == "sts":
    out({"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/x/i-0",
         "UserId": "AROAEXAMPLE:i-0"})
fail("InvalidAction", f"not faked: {svc} {op}")
"""
)

CHOWN = (
    r"""#!__PYTHON__ -IS
import json, os, sys
D = __DIR__
"""
    + _LOG
    + r"""
log("chown", sys.argv[1:])
"""
)

# `docker`, faked (A13's unit tests).
FAKE_DOCKER = (
    r"""#!__PYTHON__ -IS
import json, os, sys
D = __DIR__
"""
    + _LOG
    + r"""
argv = sys.argv[1:]
cfg = json.load(open(os.path.join(D, "docker.json")))

def save():
    json.dump(cfg, open(os.path.join(D, "docker.json"), "w"))

COMPOSE_VALUED = {"-p", "--project-name", "-f", "--file", "--env-file", "--profile",
                  "--project-directory", "--ansi", "--progress", "--parallel"}
RUN_VALUED = {"-e", "--env", "--name", "--entrypoint", "-u", "--user", "-w", "--workdir",
              "-v", "--volume", "-l", "--label", "-p", "--publish", "--pull", "--cap-add",
              "--cap-drop"}

def positional(words, valued):
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if w == "--":
            out += words[i + 1:]; break
        if w.startswith("-"):
            if "=" not in w and w in valued:
                i += 1
        else:
            out.append(w)
        i += 1
    return out

def health(svc):
    return cfg.get("health", {}).get(svc, cfg.get("default_health", "healthy"))

if argv[:1] == ["login"]:
    data = sys.stdin.read()
    token = open(os.path.join(D, "ecr-token")).read().strip()
    log("docker", argv, stdin_is_token=(data.strip() == token))
    print("Login Succeeded"); sys.exit(0)

log("docker", argv)

if argv[:1] == ["compose"]:
    words = argv[1:]
    i = 0
    while i < len(words) and words[i].startswith("-"):
        i += 2 if ("=" not in words[i] and words[i] in COMPOSE_VALUED) else 1
    sub, rest = (words[i] if i < len(words) else ""), words[i + 1:]
    if sub == "version":
        print("Docker Compose version v2.30.0"); sys.exit(0)
    if sub == "pull":
        sys.exit(0)
    if sub == "run":
        names = positional(rest, RUN_VALUED)
        svc = names[0] if names else ""
        if svc == "acceptance" and cfg.get("canaries") is not None:
            d = os.path.join(cfg["root"], "var", "lib", "loom", "acceptance")
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "canaries.json")
            with open(p, "w") as f:
                json.dump(cfg["canaries"], f)
            os.chmod(p, 0o600)
        sys.exit(int(cfg.get("run_exit", {}).get(svc, 0)))
    if sub == "up":
        for svc in positional(rest, {"--timeout", "-t", "--scale", "--pull", "--wait-timeout"}):
            cfg.setdefault("started", {})[svc] = cfg.get("started", {}).get(svc, 0) + 1
        save(); sys.exit(0)
    if sub == "ps":
        for svc in positional(rest, {"--format", "--filter", "--status"}):
            print(f"cid-{svc}")
        sys.exit(0)
    sys.exit(0)

if argv[:1] == ["inspect"] or argv[:2] == ["container", "inspect"]:
    words = argv[1:] if argv[0] == "inspect" else argv[2:]
    fmt, ids, i = "", [], 0
    while i < len(words):
        w = words[i]
        if w in ("-f", "--format"):
            fmt = words[i + 1]; i += 2; continue
        if w.startswith("--format="):
            fmt = w.split("=", 1)[1]
        elif not w.startswith("-"):
            ids.append(w)
        i += 1
    for cid in ids:
        svc = cid[4:] if cid.startswith("cid-") else cid
        n = cfg.get("started", {}).get(svc, 0)
        if "StartedAt" in fmt:
            print(f"2026-11-18T00:00:{n:02d}.000000000Z")
        elif "Health" in fmt or "Status" in fmt:
            print(health(svc))
        elif "Id" in fmt:
            print(cid)
        else:
            print(json.dumps([{"Id": cid, "State": {"Status": "running",
                               "StartedAt": f"2026-11-18T00:00:{n:02d}Z",
                               "Health": {"Status": health(svc)}}}]))
    sys.exit(0)
sys.exit(0)
"""
)

# `docker`, the real one except `login` and `compose ... pull` (A15's harness, §6.12).
DOCKER_WRAPPER = (
    r"""#!__PYTHON__ -IS
import json, os, sys
D = __DIR__
REAL = __REAL__
"""
    + _LOG
    + r"""
argv = sys.argv[1:]
COMPOSE_VALUED = {"-p", "--project-name", "-f", "--file", "--env-file", "--profile",
                  "--project-directory", "--ansi", "--progress", "--parallel"}
if argv[:1] == ["login"]:
    sys.stdin.read()
    log("docker", argv, faked=True)
    print("Login Succeeded"); sys.exit(0)
if argv[:1] == ["compose"]:
    words, i = argv[1:], 0
    while i < len(words) and words[i].startswith("-"):
        i += 2 if ("=" not in words[i] and words[i] in COMPOSE_VALUED) else 1
    if i < len(words) and words[i] == "pull":
        log("docker", argv, faked=True)
        sys.exit(0)
log("docker", argv)
os.execv(REAL, [REAL, *argv])
"""
)


def _script(template: str, state: Path, **extra: str) -> str:
    text = template.replace("__PYTHON__", sys.executable).replace("__DIR__", repr(str(state)))
    for k, v in extra.items():
        text = text.replace(k, v)
    return text


def deploy_root_base() -> Path:
    """A fresh scratch directory under /dev/shm (a tmpfs, §6.12)."""
    base = Path("/dev/shm") / f"loom-m3test-{os.getpid()}-{secrets.token_hex(4)}"
    base.mkdir(mode=0o700)
    return base


def rmtree(path: Path) -> None:
    """Remove a tree that may hold 0500 directories and 0400 files."""
    if not path.exists():
        return
    for p in [path, *path.rglob("*")]:
        with contextlib.suppress(OSError):
            if p.is_dir() and not p.is_symlink():
                p.chmod(0o700)
    shutil.rmtree(path, ignore_errors=True)


def mount_fstype(path: Path) -> str:
    """The filesystem type of the mount holding `path` (from /proc/self/mountinfo)."""
    path = path.resolve()
    best, fstype = "", ""
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, _, right = line.partition(" - ")
        point = left.split()[4].replace("\\040", " ")
        if (str(path) == point or str(path).startswith(point.rstrip("/") + "/")) and len(
            point
        ) >= len(best):
            best, fstype = point, right.split()[0]
    return fstype


class Box:
    """A deploy root, the fake tools on PATH, and the release files in the fake bucket."""

    def __init__(
        self,
        base: Path,
        *,
        root: Path | None = None,
        fake_docker: bool = True,
        store: dict[str, str | None] | None = None,
        site_doc: dict[str, Any] | None = None,
    ) -> None:
        self.base = base
        self.root = root if root is not None else base / "root"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = base / "state"
        self.bin = base / "bin"
        self.home, self.tmp, self.work = base / "home", base / "tmp", base / "work"
        for d in (self.state, self.bin, self.home, self.tmp, self.work, self.state / "s3"):
            d.mkdir(parents=True, exist_ok=True)
        self.store: dict[str, str | None] = store if store is not None else full_store()
        self.save_store()
        self.set_site(site_doc if site_doc is not None else site())
        self.ecr_token = "ecrtok" + secrets.token_urlsafe(48)
        (self.state / "ecr-token").write_text(self.ecr_token)
        self.docker_cfg: dict[str, Any] = {"root": str(self.root), "health": {}}
        self.save_docker()
        tools = {"aws": _script(FAKE_AWS, self.state), "chown": _script(CHOWN, self.state)}
        if fake_docker:
            tools["docker"] = _script(FAKE_DOCKER, self.state)
        else:
            real = shutil.which("docker")
            assert real, "no docker on PATH"
            tools["docker"] = _script(DOCKER_WRAPPER, self.state, __REAL__=repr(real))
        for name, text in tools.items():
            p = self.bin / name
            p.write_text(text)
            p.chmod(0o755)
        self.releases: dict[str, dict[str, Any]] = {}

    # -- state of the fakes

    def save_store(self) -> None:
        (self.state / "secrets.json").write_text(json.dumps(self.store))

    def load_store(self) -> dict[str, str | None]:
        data: dict[str, str | None] = json.loads((self.state / "secrets.json").read_text())
        return data

    def set_site(self, doc: dict[str, Any] | None) -> None:
        p = self.state / "site.json"
        self.site = doc
        if doc is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(json.dumps(doc))

    def save_docker(self) -> None:
        (self.state / "docker.json").write_text(json.dumps(self.docker_cfg))

    def load_docker(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((self.state / "docker.json").read_text())
        return data

    def set_log_hits(self, hits: list[str]) -> None:
        (self.state / "log-hits.json").write_text(json.dumps(hits))

    # -- the release

    def put_object(self, key: str, data: bytes) -> None:
        p = self.state / "s3" / BUCKET / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def publish(
        self,
        sha: str,
        images: dict[str, str] | None = None,
        *,
        bundle_files: dict[str, bytes] | None = None,
        compose_bytes: bytes | None = None,
        manifest_over: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Upload `releases/<sha>.json`, `releases/<sha>/compose.prod.yml` and
        `releases/<sha>/deploy.tar.gz` (the `deploy/` directory, its contents at the
        archive's root) as the image workflow does (§6.12)."""
        files = bundle_files if bundle_files is not None else deploy_files()
        bundle = tar_gz(files)
        compose = compose_bytes if compose_bytes is not None else files["compose.prod.yml"]
        self.put_object(f"releases/{sha}/deploy.tar.gz", bundle)
        self.put_object(f"releases/{sha}/compose.prod.yml", compose)
        module = SRC / "loom_devstack" / "deploy.py"
        if module.is_file():
            self.put_object(f"releases/{sha}/loom-deploy.py", module.read_bytes())
        doc: dict[str, Any] = {
            "sha": sha,
            "images": images or ecr_images(sha),
            "migrations": 12,
            "compose": {
                "path": f"releases/{sha}/compose.prod.yml",
                "sha256": hashlib.sha256(compose).hexdigest(),
            },
            "bundle": {
                "path": f"releases/{sha}/deploy.tar.gz",
                "sha256": hashlib.sha256(bundle).hexdigest(),
            },
        }
        doc.update(manifest_over or {})
        self.put_object(f"releases/{sha}.json", json.dumps(doc).encode())
        self.releases[sha] = doc
        return doc

    # -- running loom-deploy

    def env(self, **over: str | None) -> dict[str, str]:
        env = {
            "PATH": f"{self.bin}:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(SRC),
            "HOME": str(self.home),
            "TMPDIR": str(self.tmp),
            "LOOM_DEPLOY_ROOT": str(self.root),
            "LOOM_DEPLOY_HEALTH_TIMEOUT_S": "20",
        }
        for k, v in over.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        return env

    def deploy(
        self, *args: str, timeout: float = 110, **env: str | None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-S", "-m", "loom_devstack.deploy", *args],
            env=self.env(**env),
            cwd=self.work,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    # -- what happened

    def timeline(self) -> list[dict[str, Any]]:
        p = self.state / "timeline.jsonl"
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

    def calls(self, tool: str) -> list[list[str]]:
        return [r["argv"] for r in self.timeline() if r["tool"] == tool]

    def all_argv(self) -> str:
        return "\n".join(" ".join([r["tool"], *r["argv"]]) for r in self.timeline())

    @property
    def run_loom(self) -> Path:
        return self.root / "run" / "loom"

    @property
    def var_lib(self) -> Path:
        return self.root / "var" / "lib" / "loom"

    def log_records(self) -> list[dict[str, Any]]:
        p = self.var_lib / "deploy.log"
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                assert isinstance(rec, dict), f"a deploy.log line is not a JSON object: {line}"
                out.append(rec)
        return out

    def close(self) -> None:
        rmtree(self.base)
        if not self.root.is_relative_to(self.base):
            rmtree(self.root)


def deploy_files() -> dict[str, bytes]:
    """The `deploy/` directory's files, by path relative to it."""
    assert PROD.is_file(), "deploy/compose.prod.yml does not exist"
    out = {}
    for p in sorted(DEPLOY_DIR.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            out[p.relative_to(DEPLOY_DIR).as_posix()] = p.read_bytes()
    return out


def tar_gz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 1_700_000_000
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def ecr_images(sha: str, changed: Iterable[str] = ()) -> dict[str, str]:
    """A manifest's `images`: the Loom ones in ECR by digest, the third-party ones as a
    compose file pins them."""
    out = {}
    for svc in LOOM_IMAGES:
        repo = "devstack" if svc == "migrate" else svc
        d = hashlib.sha256(f"{sha}/{svc}".encode()).hexdigest()
        out[svc] = f"{REGISTRY}/loom/{repo}@sha256:{d}"
    for svc in THIRD_PARTY:
        d = hashlib.sha256(f"third-party/{svc}".encode()).hexdigest()
        out[svc] = f"docker.io/library/{svc}@sha256:{d}"
    return out


# ---------------------------------------------------------------------------- reading


def read_env_file(path: Path) -> dict[str, str]:
    """`KEY=value` lines (compose `env_file`), surrounding quotes removed."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def mode(path: Path) -> int:
    return stat.S_IMODE(path.lstat().st_mode)


def chowns(timeline: list[dict[str, Any]]) -> list[tuple[str, Path, bool]]:
    """(owner, path, recursive) of each recorded `chown`."""
    out = []
    for r in timeline:
        if r["tool"] != "chown":
            continue
        flags = [a for a in r["argv"] if a.startswith("-")]
        words = [a for a in r["argv"] if not a.startswith("-")]
        recursive = any(f in ("-R", "--recursive") for f in flags)
        if len(words) >= 2:
            for p in words[1:]:
                out.append((words[0], Path(p), recursive))
    return out


def owner_of(path: Path, recorded: list[tuple[str, Path, bool]]) -> str | None:
    """The owner the last recorded `chown` naming `path` (or, recursively, a parent) gave."""
    got = None
    for owner, p, rec in recorded:
        if p == path or (rec and path.is_relative_to(p)):
            got = owner
    return got


def compose_calls(timeline: list[dict[str, Any]]) -> list[tuple[int, list[str], str, list[str]]]:
    """(timeline index, global options, subcommand, the rest) of each `docker compose`."""
    valued = {"-p", "--project-name", "-f", "--file", "--env-file", "--profile",
              "--project-directory", "--ansi", "--progress", "--parallel"}  # fmt: skip
    out = []
    for n, r in enumerate(timeline):
        if r["tool"] != "docker" or r["argv"][:1] != ["compose"]:
            continue
        words, i = r["argv"][1:], 0
        while i < len(words) and words[i].startswith("-"):
            i += 2 if ("=" not in words[i] and words[i] in valued) else 1
        out.append((n, words[:i], words[i] if i < len(words) else "", words[i + 1 :]))
    return out


def option_values(words: list[str], *names: str) -> list[str]:
    out = []
    for i, w in enumerate(words):
        for n in names:
            if w == n and i + 1 < len(words):
                out.append(words[i + 1])
            elif w.startswith(n + "="):
                out.append(w.split("=", 1)[1])
    return out


def non_options(words: list[str], valued: set[str]) -> list[str]:
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if w == "--":
            out += words[i + 1 :]
            break
        if w.startswith("-"):
            if "=" not in w and w in valued:
                i += 1
        else:
            out.append(w)
        i += 1
    return out


def ups(timeline: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(timeline index, service) of each `docker compose up`, in order."""
    out = []
    for n, _, sub, rest in compose_calls(timeline):
        if sub == "up":
            for svc in non_options(rest, {"--timeout", "-t", "--scale", "--pull"}):
                out.append((n, svc))
    return out


def runs(timeline: list[dict[str, Any]]) -> list[tuple[int, str, list[str]]]:
    """(timeline index, service, argv after `run`) of each `docker compose run`."""
    valued = {"-e", "--env", "--name", "--entrypoint", "-u", "--user", "-w", "--workdir",
              "-v", "--volume", "-l", "--label", "-p", "--publish", "--pull"}  # fmt: skip
    out = []
    for n, _, sub, rest in compose_calls(timeline):
        if sub == "run":
            names = non_options(rest, valued)
            out.append((n, names[0] if names else "", rest))
    return out


def text_files(*dirs: Path) -> dict[Path, str]:
    out = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_file() and not p.is_symlink():
                out[p] = p.read_bytes().decode("utf-8", errors="replace")
    return out
