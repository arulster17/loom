"""A27: `loom-poolmgr new-ca --out DIR` writes a new pod CA outside dev; the dev-only
`issue-ca` stays dev-only (docs/phase1-m3-spec.md §7; contracts §9).

Runbook step 1 runs `new-ca` on the operator's machine, so the command is run outside
dev with no reachable Postgres or Redis. "A valid CA" is checked as the manager uses
one: its certificate and key load through `CertificateAuthority.load` (the code behind
`LOOM_POOLMGR_CA_CERT` / `_KEY`), it is valid now and for longer than alarm P3's 30 days,
and a manager client certificate issued from it loads with it (`Pki.load`). The file
names inside DIR are the command's choice.
"""

from __future__ import annotations

import time
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from poolmgr_m3_kit import (  # noqa: F401
    PROD_DATABASE_URL,
    PROD_REDIS_URL,
    PoolmgrRun,
    poolmgr_run_fixture,
)

from loom_poolmgr.pki import POOLMGR_CN, CertificateAuthority, Pki, write_issued

DAY_S = 86400


def outside_dev(h: PoolmgrRun) -> None:
    h.mp.setenv("LOOM_ENV", "prod")
    h.mp.setenv("LOOM_DATABASE_URL", PROD_DATABASE_URL)  # nothing listens there
    h.mp.setenv("LOOM_REDIS_URL", PROD_REDIS_URL)


def find_ca(directory: Path) -> tuple[Path, Path]:
    """(the CA certificate's file, the file with its private key) among the files written."""
    files = [p for p in directory.rglob("*") if p.is_file()]
    assert files, f"nothing written in {directory}"
    certs = []
    for p in files:
        data = p.read_bytes()
        if b"BEGIN CERTIFICATE" in data:
            cert = x509.load_pem_x509_certificate(data)
            try:
                bc = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
            except x509.ExtensionNotFound:
                continue
            if bc.ca:
                certs.append((p, cert))
    assert len(certs) == 1, f"expected one CA certificate in {directory}: {files}"
    cert_path, cert = certs[0]
    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    for p in files:
        if b"PRIVATE KEY" not in p.read_bytes():
            continue
        try:
            key = serialization.load_pem_private_key(p.read_bytes(), password=None)
        except (ValueError, TypeError):
            continue
        if (
            key.public_key().public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            == spki
        ):
            return cert_path, p
    raise AssertionError(f"no file in {directory} holds the CA's private key: {files}")


def test_new_ca_outside_dev_writes_a_ca_the_manager_can_use(poolmgr_run: PoolmgrRun) -> None:
    h = poolmgr_run
    outside_dev(h)
    out = h.tmp / "pod-ca-next"
    code = h.main("new-ca", "--out", str(out))
    assert code in (0, None), f"loom-poolmgr new-ca exited {code}"
    cert_path, key_path = find_ca(out)
    ca = CertificateAuthority.load(cert_path, key_path)
    now = time.time()
    assert ca.cert.not_valid_before_utc.timestamp() <= now
    assert ca.cert.not_valid_after_utc.timestamp() - now > 30 * DAY_S, "expires within P3's 30 days"
    ca.cert.verify_directly_issued_by(ca.cert)  # self-signed root
    client = h.tmp / "client"
    client.mkdir()
    write_issued(client, POOLMGR_CN, ca.issue_client(now=now))
    pki = Pki.load(cert_path, key_path, client / f"{POOLMGR_CN}.crt", client / f"{POOLMGR_CN}.key")
    pki.client_ssl_context()
    server = ca.issue_server("podm3newca", days=7, now=now)
    server.cert.verify_directly_issued_by(ca.cert)


def test_new_ca_makes_a_new_key_each_time(poolmgr_run: PoolmgrRun) -> None:
    h = poolmgr_run
    outside_dev(h)
    keys = []
    for name in ("one", "two"):
        out = h.tmp / name
        assert h.main("new-ca", "--out", str(out)) in (0, None)
        cert_path, key_path = find_ca(out)
        keys.append(CertificateAuthority.load(cert_path, key_path).cert.public_key())
    der = [
        k.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        for k in keys
    ]
    assert der[0] != der[1]


def test_issue_ca_stays_dev_only(poolmgr_run: PoolmgrRun) -> None:
    h = poolmgr_run
    outside_dev(h)
    out = h.tmp / "dev-ca"
    assert h.main("issue-ca", "--out", str(out)) not in (0, None)
    assert not out.exists() or not any(
        b"PRIVATE KEY" in p.read_bytes() for p in out.rglob("*") if p.is_file()
    )


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
