"""`loom-deploy secrets init` and `loom-deploy ses-smtp-password` (M3 spec §5.2, §6.3, A13)
against a fake `aws` on PATH.

`loom-deploy` runs as the box's Python does: `python -S` (no site-packages) with
`loom_devstack` on PYTHONPATH, `-m loom_devstack.deploy`. The fake `aws` keeps a small
Secrets Manager: the secret containers of §5.2 (`loom/platform/<name>`), each with or
without a value; it answers `get-secret-value` (ResourceNotFoundException for a container
with no value, as AWS does), `describe-secret`, `list-secret-version-ids` and
`put-secret-value` (`--secret-string` inline, or `file://...` such as
`file:///dev/stdin`), and logs every call's argv.

`secrets init` generates the values Loom makes (internal-token, app-db-password,
`auth.auth_secret`, the Redis password, CA and server certificate with SAN `redis`, the
pod CA valid 2 years, the manager's client certificate) and never overwrites a secret
that has a value; it writes none of the user's (stripe, ses-smtp, serving-runpod-key).
Nothing secret is on a command line or in its output, and no file with a secret is left
behind (its HOME and TMPDIR are scanned).

`ses-smtp-password` reads the access key the runbook created, as `aws iam
create-access-key` prints it (`{"AccessKey": {"AccessKeyId", "SecretAccessKey", ...}}`),
on stdin (this test's reading: the spec fixes neither the channel nor a flag; stdin keeps
the key off the command line), derives the SES SMTP password with AWS's documented
HMAC-SHA256 chain for us-east-1, and stores `{"username": <key id>, "password": ...}` in
`loom/platform/ses-smtp`, printing neither the key's secret nor the password.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtensionOID

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "services" / "devstack" / "src"

CONTAINERS = (
    "stripe",
    "auth",
    "internal-token",
    "app-db-password",
    "redis",
    "pod-ca",
    "poolmgr-client",
    "ses-smtp",
    "serving-runpod-key",
)
GENERATED = ("internal-token", "app-db-password", "auth", "redis", "pod-ca", "poolmgr-client")
USERS = ("stripe", "ses-smtp", "serving-runpod-key")

FAKE_AWS = r"""#!__PYTHON__
import json, os, re, sys, shutil
D = os.environ["FAKE_AWS_DIR"]
argv = sys.argv[1:]
with open(os.path.join(D, "calls.jsonl"), "a") as f:
    f.write(json.dumps(argv) + "\n")
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
STORE = os.path.join(D, "secrets.json")
store = json.load(open(STORE))

def out(obj):
    q = opts.get("query")
    if q:
        for part in q.split("."):
            obj = obj.get(part) if isinstance(obj, dict) else None
    if opts.get("output") == "text" and isinstance(obj, (str, int, float)):
        print(obj)
    else:
        print(json.dumps(obj))
    sys.exit(0)

def fail(code, msg):
    sys.stderr.write(f"An error occurred ({code}) when calling the {op} operation: {msg}\n")
    sys.exit(254)

def name_of(sid):
    m = re.search(r"loom/platform/([a-z0-9-]+?)(-[A-Za-z0-9]{6})?$", sid)
    return m.group(1) if m else sid

if svc == "secretsmanager":
    sid = str(opts.get("secret-id", ""))
    name = name_of(sid)
    if name not in store:
        fail("ResourceNotFoundException", "Secrets Manager can't find the specified secret.")
    arn = f"arn:aws:secretsmanager:us-east-1:123456789012:secret:loom/platform/{name}-AbCdEf"
    value = store[name]
    if op == "get-secret-value":
        if value is None:
            fail("ResourceNotFoundException", "Secrets Manager can't find the specified "
                 "secret value for staging label: AWSCURRENT")
        out({"ARN": arn, "Name": f"loom/platform/{name}", "VersionId": "v1",
             "SecretString": value, "VersionStages": ["AWSCURRENT"]})
    if op == "describe-secret":
        d = {"ARN": arn, "Name": f"loom/platform/{name}"}
        if value is not None:
            d["VersionIdsToStages"] = {"v1": ["AWSCURRENT"]}
        out(d)
    if op == "list-secret-version-ids":
        versions = [] if value is None else [{"VersionId": "v1", "VersionStages": ["AWSCURRENT"]}]
        out({"ARN": arn, "Name": f"loom/platform/{name}", "Versions": versions})
    if op == "put-secret-value":
        s = opts.get("secret-string")
        if not isinstance(s, str):
            fail("InvalidParameterException", "no --secret-string")
        if s.startswith("file://"):
            path = s[len("file://"):]
            s = sys.stdin.read() if path in ("/dev/stdin", "-") else open(path).read()
        store[name] = s
        json.dump(store, open(STORE, "w"))
        with open(os.path.join(D, "puts.jsonl"), "a") as f:
            f.write(json.dumps(name) + "\n")
        out({"ARN": arn, "Name": f"loom/platform/{name}", "VersionId": "v2",
             "VersionStages": ["AWSCURRENT"]})
    if op == "create-secret":
        fail("ResourceExistsException", "the secret already exists")
    print("{}"); sys.exit(0)
if svc == "sts":
    out({"Account": "123456789012", "Arn": "arn:aws:sts::123456789012:assumed-role/x/i-0",
         "UserId": "AROAEXAMPLE:i-0"})
print("{}")
"""


class FakeAws:
    def __init__(self, root: Path, preset: dict[str, str | None] | None = None) -> None:
        self.dir = root / "fake-aws"
        self.bin = root / "bin"
        self.dir.mkdir()
        self.bin.mkdir()
        store: dict[str, str | None] = dict.fromkeys(CONTAINERS)
        store.update(preset or {})
        (self.dir / "secrets.json").write_text(json.dumps(store))
        aws = self.bin / "aws"
        aws.write_text(FAKE_AWS.replace("__PYTHON__", sys.executable))
        aws.chmod(0o755)

    @property
    def store(self) -> dict[str, str | None]:
        data: dict[str, str | None] = json.loads((self.dir / "secrets.json").read_text())
        return data

    def calls(self) -> list[list[str]]:
        p = self.dir / "calls.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def puts(self) -> list[str]:
        p = self.dir / "puts.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def run_deploy(
    tmp: Path, fake: FakeAws, *args: str, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    home, tmpdir = tmp / "home", tmp / "tmp"
    home.mkdir(exist_ok=True)
    tmpdir.mkdir(exist_ok=True)
    env = {
        "PATH": f"{fake.bin}:/usr/bin:/bin",
        "PYTHONPATH": str(SRC),
        "FAKE_AWS_DIR": str(fake.dir),
        "HOME": str(home),
        "TMPDIR": str(tmpdir),
        "AWS_REGION": "us-east-1",
        "AWS_DEFAULT_REGION": "us-east-1",
    }
    return subprocess.run(
        [sys.executable, "-S", "-m", "loom_devstack.deploy", *args],
        env=env,
        input=stdin if stdin is not None else "",
        capture_output=True,
        text=True,
        timeout=120,
    )


def secret_strings(value: str) -> list[str]:
    """The value and every secret-looking piece of it (each JSON field, each line of a
    PEM key), so a part of a secret on a command line is found too."""
    parts = [value]
    try:
        doc = json.loads(value)
    except ValueError:
        doc = None
    fields = list(doc.values()) if isinstance(doc, dict) else []
    for f in fields:
        if isinstance(f, str) and f:
            parts.append(f)
            if "PRIVATE KEY" in f:
                parts += [ln for ln in f.splitlines() if len(ln) >= 40]
    return [p for p in parts if len(p) >= 12]


def assert_nowhere(secrets: list[str], fake: FakeAws, proc: Any, tmp: Path) -> None:
    argv = "\n".join(" ".join(c) for c in fake.calls())
    output = proc.stdout + proc.stderr
    left = [
        p.read_text(errors="replace")
        for d in (tmp / "home", tmp / "tmp")
        for p in d.rglob("*")
        if p.is_file()
    ]
    for s in secrets:
        assert s not in argv, "a secret value is on an aws command line"
        assert s not in output, "loom-deploy printed a secret value"
        assert not any(s in t for t in left), "a secret value was left in a file (HOME/TMPDIR)"


def field(value: str | None, key: str) -> str:
    assert value, "no value"
    doc = json.loads(value)
    assert isinstance(doc, dict) and doc.get(key), f"no {key!r} in the secret"
    return str(doc[key])


def single(value: str | None) -> str:
    assert value, "no value"
    try:
        doc = json.loads(value)
    except ValueError:
        return value
    if isinstance(doc, dict) and len(doc) == 1:
        return str(next(iter(doc.values())))
    return value


def cert(pem: str) -> x509.Certificate:
    return x509.load_pem_x509_certificate(pem.encode())


def matches(key_pem: str, crt: x509.Certificate) -> bool:
    key = serialization.load_pem_private_key(key_pem.encode(), password=None)
    fmt = serialization.PublicFormat.SubjectPublicKeyInfo
    enc = serialization.Encoding.DER
    return key.public_key().public_bytes(enc, fmt) == crt.public_key().public_bytes(enc, fmt)


def is_ca(c: x509.Certificate) -> bool:
    return bool(c.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS).value.ca)


def signed_by(c: x509.Certificate, ca: x509.Certificate) -> bool:
    try:
        c.verify_directly_issued_by(ca)
    except Exception:
        return False
    return True


# ---------------------------------------------------------------- secrets init


def test_init_generates_the_values_loom_makes(tmp_path: Path) -> None:
    fake = FakeAws(tmp_path)
    proc = run_deploy(tmp_path, fake, "secrets", "init")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    store = fake.store
    puts = fake.puts()
    for name in GENERATED:
        assert puts.count(name) == 1, f"{name} put {puts.count(name)} times"
    for name in USERS:
        assert name not in puts and store[name] is None, f"init wrote the user's {name}"

    token, dbpw = single(store["internal-token"]), single(store["app-db-password"])
    auth_secret = field(store["auth"], "auth_secret")
    redis_pw = field(store["redis"], "password")
    for v in (token, dbpw, auth_secret, redis_pw):
        assert len(v) >= 24, "a generated secret shorter than 24 characters"
    assert len({token, dbpw, auth_secret, redis_pw}) == 4, "two generated secrets are equal"

    redis_ca = cert(field(store["redis"], "ca_crt"))
    redis_crt = cert(field(store["redis"], "server_crt"))
    assert is_ca(redis_ca) and signed_by(redis_crt, redis_ca)
    san = redis_crt.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
    assert "redis" in san.value.get_values_for_type(x509.DNSName), "Redis cert SAN lacks redis"
    assert matches(field(store["redis"], "server_key"), redis_crt)

    pod_ca = cert(field(store["pod-ca"], "crt"))
    assert is_ca(pod_ca) and matches(field(store["pod-ca"], "key"), pod_ca)
    days = (pod_ca.not_valid_after_utc - pod_ca.not_valid_before_utc) / dt.timedelta(days=1)
    assert 700 <= days <= 740, f"pod CA valid {days:.0f} days, not 2 years"

    client = cert(field(store["poolmgr-client"], "crt"))
    assert signed_by(client, pod_ca), "the manager's client certificate is not the pod CA's"
    assert matches(field(store["poolmgr-client"], "key"), client)

    generated = [v for n in GENERATED if (v := store[n])]
    assert_nowhere([s for v in generated for s in secret_strings(v)], fake, proc, tmp_path)


def test_init_never_overwrites_a_secret_that_has_a_value(tmp_path: Path) -> None:
    preset = {
        "internal-token": "preset-internal-token-0123456789abcdef",
        "redis": json.dumps({"password": "preset-redis-password-0123456789"}),
        "auth": json.dumps({"auth_secret": "preset-auth-secret-0123456789abcdef"}),
    }
    fake = FakeAws(tmp_path, preset)
    proc = run_deploy(tmp_path, fake, "secrets", "init")
    assert "No module named" not in proc.stderr, proc.stderr
    asked = " ".join(" ".join(c) for c in fake.calls())
    for name in preset:
        assert f"loom/platform/{name}" in asked, f"init never looked at {name}"
    puts = fake.puts()
    for name, value in preset.items():
        assert name not in puts, f"init wrote {name}, which had a value"
        assert fake.store[name] == value
    assert_nowhere([s for v in preset.values() for s in secret_strings(v)], fake, proc, tmp_path)


def test_init_twice_changes_nothing_the_second_time(tmp_path: Path) -> None:
    fake = FakeAws(tmp_path)
    first = run_deploy(tmp_path, fake, "secrets", "init")
    assert first.returncode == 0, first.stdout + first.stderr
    before, n_puts = fake.store, len(fake.puts())
    assert n_puts == len(GENERATED)
    run_deploy(tmp_path, fake, "secrets", "init")
    assert fake.store == before, "a second init changed a secret"
    assert len(fake.puts()) == n_puts, "a second init wrote a secret"


# ---------------------------------------------------------------- SES SMTP password


def ses_smtp_password(secret_access_key: str, region: str) -> str:
    """AWS's documented derivation (SES developer guide, "Obtaining Amazon SES SMTP
    credentials": the SigV4 key chain with date 11111111 over SendRawEmail, version 4)."""

    def sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    sig = sign(("AWS4" + secret_access_key).encode(), "11111111")
    for part in (region, "ses", "aws4_request", "SendRawEmail"):
        sig = sign(sig, part)
    return base64.b64encode(bytes([0x04]) + sig).decode()


def test_the_derivation_matches_the_documented_algorithm() -> None:
    """The oracle itself: deterministic, versioned, 44 base64 characters (1 + 32 bytes)."""
    pw = ses_smtp_password("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "us-east-1")
    assert len(pw) == 44 and base64.b64decode(pw)[0] == 4
    assert pw != ses_smtp_password("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "eu-west-1")


def test_ses_smtp_password_stores_the_derived_password(tmp_path: Path) -> None:
    key_id, secret = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    created = {
        "AccessKey": {
            "UserName": "loom-ses-smtp",
            "AccessKeyId": key_id,
            "Status": "Active",
            "SecretAccessKey": secret,
            "CreateDate": "2026-11-18T00:00:00+00:00",
        }
    }
    fake = FakeAws(tmp_path)
    proc = run_deploy(tmp_path, fake, "ses-smtp-password", stdin=json.dumps(created))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert fake.puts() == ["ses-smtp"], fake.puts()
    stored = json.loads(fake.store["ses-smtp"] or "{}")
    want = ses_smtp_password(secret, "us-east-1")
    assert stored == {"username": key_id, "password": want}, "wrong SES SMTP credentials"
    assert_nowhere([secret, want], fake, proc, tmp_path)
    for name in CONTAINERS:
        if name != "ses-smtp":
            assert fake.store[name] is None, f"ses-smtp-password wrote {name}"


def test_ses_smtp_password_refuses_input_without_a_key(tmp_path: Path) -> None:
    fake = FakeAws(tmp_path)
    proc = run_deploy(tmp_path, fake, "ses-smtp-password", stdin="{}")
    assert "No module named" not in proc.stderr and "invalid choice" not in proc.stderr, proc.stderr
    assert proc.returncode != 0
    assert fake.puts() == [], "stored SES credentials from input without an access key"
