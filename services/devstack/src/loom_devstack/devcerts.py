"""Local development certificates for the gateway-to-pod mTLS path (contracts §9).

Writes into one directory (default `.dev/certs`, gitignored; never commit it):

| File | What |
|---|---|
| `ca.crt`, `ca.key` | Loom dev CA (EC P-256, 1 year) |
| `gateway.crt`, `gateway.key` | Gateway client certificate (CN `loom-gateway`, clientAuth) |
| `<pod>.crt`, `<pod>.key` | Pod proxy server certificate, SAN DNS `<pod>` (serverAuth) |
| `<pod>.token` | That pod's `X-Loom-Pod-Token` value |

Idempotent: existing files are kept while they are valid for at least another day and
chain to the CA, so re-running (every `docker compose up`) changes nothing. `--force`
starts over with a new CA. Production issues its own (CA key in Secrets Manager, 7-day
leaves, phase1-runpod-plan.md §4); nothing here is used outside local dev.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

DEFAULT_DIR = Path(".dev/certs")
DEFAULT_PODS = ("mock-pod-1", "mock-pod-2")
GATEWAY_CN = "loom-gateway"
CA_DAYS = 365
LEAF_DAYS = 30
RENEW_BEFORE = dt.timedelta(days=1)


@dataclass(frozen=True)
class Issued:
    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _name(cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Loom dev"),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def make_ca(cn: str = "Loom dev CA", days: int = CA_DAYS) -> Issued:
    key = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
    cert = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(_name(cn))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(ski, critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski), critical=False
        )
        .sign(key, hashes.SHA256())
    )
    return Issued(cert, key)


def issue(
    ca: Issued,
    cn: str,
    *,
    server: bool,
    dns_names: tuple[str, ...] = (),
    ips: tuple[str, ...] = (),
    days: int = LEAF_DAYS,
) -> Issued:
    """A leaf signed by `ca`: a server certificate (serverAuth, SANs) or a client one."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = _now()
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(cn))
        .issuer_name(ca.cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(min(now + dt.timedelta(days=days), ca.cert.not_valid_after_utc))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]
            ),
            critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca.key.public_key()),
            critical=False,
        )
    )
    sans: list[x509.GeneralName] = [x509.DNSName(n) for n in dns_names]
    sans += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    if server and not sans:
        sans = [x509.DNSName(cn)]
    if sans:
        builder = builder.add_extension(x509.SubjectAlternativeName(sans), critical=False)
    return Issued(builder.sign(ca.key, hashes.SHA256()), key)


# ---------------------------------------------------------------- files


def _write(path: Path, data: bytes, mode: int) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, mode)
    tmp.replace(path)


def write_issued(out: Path, stem: str, issued: Issued, key_mode: int) -> None:
    key_pem = issued.key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    _write(out / f"{stem}.key", key_pem, key_mode)
    _write(out / f"{stem}.crt", issued.cert.public_bytes(serialization.Encoding.PEM), 0o644)


def load_issued(out: Path, stem: str) -> Issued | None:
    crt, key = out / f"{stem}.crt", out / f"{stem}.key"
    if not (crt.is_file() and key.is_file()):
        return None
    cert = x509.load_pem_x509_certificate(crt.read_bytes())
    private = serialization.load_pem_private_key(key.read_bytes(), password=None)
    if not isinstance(private, ec.EllipticCurvePrivateKey):
        return None
    return Issued(cert, private)


def _usable(leaf: Issued | None, ca: Issued, dns: tuple[str, ...]) -> bool:
    if leaf is None:
        return False
    if leaf.cert.not_valid_after_utc - _now() < RENEW_BEFORE:
        return False
    if leaf.cert.issuer != ca.cert.subject:
        return False
    try:
        leaf.cert.verify_directly_issued_by(ca.cert)
    except (ValueError, TypeError, Exception):
        return False
    if dns:
        try:
            sans = leaf.cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        except x509.ExtensionNotFound:
            return False
        if set(sans.value.get_values_for_type(x509.DNSName)) != set(dns):
            return False
    return True


def ensure_dev_certs(
    out: Path,
    pods: tuple[str, ...] = DEFAULT_PODS,
    *,
    force: bool = False,
    key_mode: int = 0o600,
) -> list[str]:
    """Create or refresh everything in `out`; returns the file stems (re)written."""
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    ca = load_issued(out, "ca")
    if ca is None or ca.cert.not_valid_after_utc - _now() < RENEW_BEFORE * 30:
        ca = make_ca()
        write_issued(out, "ca", ca, 0o600)
        written.append("ca")

    if not _usable(load_issued(out, "gateway"), ca, ()):
        write_issued(out, "gateway", issue(ca, GATEWAY_CN, server=False), key_mode)
        written.append("gateway")
    for pod in pods:
        if not _usable(load_issued(out, pod), ca, (pod,)):
            write_issued(out, pod, issue(ca, pod, server=True, dns_names=(pod,)), key_mode)
            written.append(pod)
        token = out / f"{pod}.token"
        if not token.is_file() or not token.read_text().strip():
            _write(token, secrets.token_urlsafe(32).encode() + b"\n", key_mode)
            written.append(f"{pod}.token")
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="loom-devcerts", description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path(os.environ.get("LOOM_DEV_CERTS", DEFAULT_DIR)))
    p.add_argument(
        "--pod",
        action="append",
        help=f"pod id to issue a server certificate for (default: {', '.join(DEFAULT_PODS)})",
    )
    p.add_argument("--force", action="store_true", help="discard everything, new CA")
    p.add_argument(
        "--readable-keys",
        action="store_true",
        help="leaf keys and tokens mode 0644 instead of 0600, for containers that run as "
        "another user than the one writing (dev only)",
    )
    args = p.parse_args(argv)
    pods = tuple(args.pod or DEFAULT_PODS)
    written = ensure_dev_certs(
        args.out, pods, force=args.force, key_mode=0o644 if args.readable_keys else 0o600
    )
    print(f"{args.out}: " + (", ".join(written) + " written" if written else "up to date"))


if __name__ == "__main__":
    main()
