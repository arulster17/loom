"""The Loom pod CA: load it, issue pod server certificates and client certificates
(docs/phase1-m2-spec.md §3, §5 "Certificates").

- A pod's **server** certificate: SAN DNS = the pod id (the name the gateway and the
  manager send as `server_name`), `serverAuth`, `server_cert_days` long.
- A **client** certificate (`clientAuth`): the manager's own, for its health checks and
  probes through each pod's proxy (the gateway's path), and in dev the gateway's.

Keys are EC P-256, as `loom-devcerts`. Nothing here writes a secret anywhere but the
files `write_dev_ca` is asked for; certificates and keys for pods go to the pod over SSH
stdin only (`loom_poolmgr.bootstrap`). Every issue takes `now` (epoch seconds) so the
clock is injectable.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import ssl
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

ORGANIZATION = "Loom"
CA_DAYS = 365
NEW_CA_DAYS = 730  # `new-ca`: as the first CA (`loom-deploy secrets init`)
CLIENT_DAYS = 30
POOLMGR_CN = "loom-poolmgr"
GATEWAY_CN = "loom-gateway"
# Issued slightly in the past, so a pod whose clock lags a little accepts it.
BACKDATE = dt.timedelta(minutes=5)


def _utc(now: float | None) -> dt.datetime:
    return dt.datetime.fromtimestamp(time.time() if now is None else now, dt.UTC)


def _name(cn: str) -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, ORGANIZATION),
            x509.NameAttribute(NameOID.COMMON_NAME, cn),
        ]
    )


def _key_usage(*, ca: bool) -> x509.KeyUsage:
    return x509.KeyUsage(
        digital_signature=True,
        content_commitment=False,
        key_encipherment=False,
        data_encipherment=False,
        key_agreement=False,
        key_cert_sign=ca,
        crl_sign=ca,
        encipher_only=False,
        decipher_only=False,
    )


@dataclass(frozen=True)
class IssuedCert:
    """A certificate and its private key."""

    cert: x509.Certificate
    key: ec.EllipticCurvePrivateKey

    def cert_pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)

    def key_pem(self) -> bytes:
        return self.key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    @property
    def not_after(self) -> dt.datetime:
        return self.cert.not_valid_after_utc


class CertificateAuthority:
    """The CA the manager is given (`LOOM_POOLMGR_CA_CERT` / `LOOM_POOLMGR_CA_KEY`)."""

    def __init__(self, cert: x509.Certificate, key: ec.EllipticCurvePrivateKey) -> None:
        if cert.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ) != key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ):
            raise ValueError("the CA key does not match the CA certificate")
        self.cert = cert
        self.key = key

    @classmethod
    def load(cls, cert_path: Path, key_path: Path) -> CertificateAuthority:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError(f"{key_path}: not an EC private key")
        try:
            constraints = cert.extensions.get_extension_for_class(x509.BasicConstraints).value
        except x509.ExtensionNotFound:
            constraints = None
        if constraints is None or not constraints.ca:
            raise ValueError(f"{cert_path}: not a CA certificate")
        return cls(cert, key)

    @classmethod
    def generate(
        cls, cn: str = "Loom pod CA", *, days: int = CA_DAYS, now: float | None = None
    ) -> CertificateAuthority:
        """A new self-signed CA (dev and tests; production's comes from M3's secrets)."""
        key = ec.generate_private_key(ec.SECP256R1())
        start = _utc(now)
        ski = x509.SubjectKeyIdentifier.from_public_key(key.public_key())
        cert = (
            x509.CertificateBuilder()
            .subject_name(_name(cn))
            .issuer_name(_name(cn))
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(start - BACKDATE)
            .not_valid_after(start + dt.timedelta(days=days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(_key_usage(ca=True), critical=True)
            .add_extension(ski, critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_subject_key_identifier(ski),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )
        return cls(cert, key)

    def cert_pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)

    def _issue(
        self, cn: str, *, server: bool, dns_names: tuple[str, ...], days: float, now: float | None
    ) -> IssuedCert:
        if days <= 0:
            raise ValueError("days must be positive")
        key = ec.generate_private_key(ec.SECP256R1())
        start = _utc(now)
        builder = (
            x509.CertificateBuilder()
            .subject_name(_name(cn))
            .issuer_name(self.cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(start - BACKDATE)
            .not_valid_after(min(start + dt.timedelta(days=days), self.cert.not_valid_after_utc))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(_key_usage(ca=False), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage(
                    [ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]
                ),
                critical=False,
            )
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(self.key.public_key()),
                critical=False,
            )
        )
        if dns_names:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.DNSName(n) for n in dns_names]), critical=False
            )
        return IssuedCert(builder.sign(self.key, hashes.SHA256()), key)

    def issue_server(self, pod_id: str, *, days: float, now: float | None = None) -> IssuedCert:
        """A pod proxy's server certificate: SAN DNS = `pod_id`, serverAuth."""
        if not pod_id:
            raise ValueError("empty pod id")
        return self._issue(pod_id, server=True, dns_names=(pod_id,), days=days, now=now)

    def issue_client(
        self, cn: str = POOLMGR_CN, *, days: float = CLIENT_DAYS, now: float | None = None
    ) -> IssuedCert:
        """A client certificate (clientAuth) the pod proxies accept."""
        return self._issue(cn, server=False, dns_names=(), days=days, now=now)


@dataclass(frozen=True)
class Pki:
    """What the manager holds: the CA (to issue pod server certificates) and its own
    client certificate as files (to talk to pod proxies with mTLS)."""

    ca: CertificateAuthority
    ca_cert_path: Path
    client_cert_path: Path
    client_key_path: Path

    @classmethod
    def load(cls, ca_cert: Path, ca_key: Path, client_cert: Path, client_key: Path) -> Pki:
        """From the `LOOM_POOLMGR_CA_*` and `LOOM_POOLMGR_CLIENT_*` files. The client
        certificate must chain to the CA."""
        ca = CertificateAuthority.load(ca_cert, ca_key)
        client = x509.load_pem_x509_certificate(client_cert.read_bytes())
        client.verify_directly_issued_by(ca.cert)
        return cls(ca, ca_cert, client_cert, client_key)

    @classmethod
    def generate(cls, directory: Path, *, now: float | None = None) -> Pki:
        """A fresh CA and manager client certificate written into `directory` (tests and
        the local backend without configured files)."""
        write_dev_ca(directory, now=now)
        return cls.load(
            directory / "ca.crt",
            directory / "ca.key",
            directory / f"{POOLMGR_CN}.crt",
            directory / f"{POOLMGR_CN}.key",
        )

    def client_ssl_context(self) -> ssl.SSLContext:
        """TLS 1.3, the pod CA as the only trust root, the manager's client certificate.
        The server name (the pod id) is checked per request (`sni_hostname`), as the
        gateway does."""
        ctx = ssl.create_default_context(cafile=str(self.ca_cert_path))
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.load_cert_chain(str(self.client_cert_path), str(self.client_key_path))
        return ctx


def _write(path: Path, data: bytes, mode: int) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)  # a stale one from a crashed run
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, mode)  # the umask may have cleared bits of `mode`
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_issued(directory: Path, stem: str, issued: IssuedCert, *, key_mode: int = 0o600) -> None:
    _write(directory / f"{stem}.key", issued.key_pem(), key_mode)
    _write(directory / f"{stem}.crt", issued.cert_pem(), 0o644)


def write_dev_ca(directory: Path, *, now: float | None = None) -> list[str]:
    """`loom-poolmgr issue-ca --out DIR` (dev only): a new CA plus the manager's and the
    gateway's client certificates. Writes `ca.{crt,key}`, `loom-poolmgr.{crt,key}` and
    `loom-gateway.{crt,key}` (keys 0600); returns the stems written."""
    directory.mkdir(parents=True, exist_ok=True)
    ca = CertificateAuthority.generate("Loom dev pod CA", now=now)
    _write(directory / "ca.key", IssuedCert(ca.cert, ca.key).key_pem(), 0o600)
    _write(directory / "ca.crt", ca.cert_pem(), 0o644)
    for cn in (POOLMGR_CN, GATEWAY_CN):
        write_issued(directory, cn, ca.issue_client(cn, now=now))
    return ["ca", POOLMGR_CN, GATEWAY_CN]


# ---------------------------------------------------------------- the gateway's certificate

GATEWAY_CERT_FILE = "loom-gateway.pem"
GATEWAY_CERT_DAYS = 7
# Rotated every 24 h; at start also when less than this is left (spec §7).
GATEWAY_RENEW_BEFORE_S = 6 * 86400.0
GATEWAY_ROTATE_EVERY_S = 86400.0

_PEM_KEY = re.compile(rb"-----BEGIN ([A-Z ]*PRIVATE KEY)-----.*?-----END \1-----", re.S)
_PEM_CERT = re.compile(rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.S)
_SPKI = (serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)


def gateway_cert_is_current(path: Path, ca: CertificateAuthority, now: float) -> bool:
    """The file at `path` holds a key and a certificate that load together, issued by
    `ca` (the signature, not just the name), with at least `GATEWAY_RENEW_BEFORE_S` left
    on the manager's clock `now`. Anything else (missing, broken, another CA, too old)
    is not current."""
    try:
        data = path.read_bytes()
        key_match, cert_match = _PEM_KEY.search(data), _PEM_CERT.search(data)
        if key_match is None or cert_match is None:
            return False
        key = serialization.load_pem_private_key(key_match.group(0), password=None)
        cert = x509.load_pem_x509_certificate(cert_match.group(0))
        if key.public_key().public_bytes(*_SPKI) != cert.public_key().public_bytes(*_SPKI):
            return False
        cert.verify_directly_issued_by(ca.cert)
        return cert.not_valid_after_utc.timestamp() - now >= GATEWAY_RENEW_BEFORE_S
    except Exception:
        return False


def write_gateway_cert(
    directory: Path, ca: CertificateAuthority, *, now: float | None = None
) -> IssuedCert:
    """Issue the gateway's client certificate (`clientAuth`, 7 days from `now`) into
    `directory/loom-gateway.pem`: key and certificate in one PEM, written under a
    temporary name (mode 0400, owned by the directory's owner so the gateway's uid can
    read it) and renamed into place, so a reader never sees a key without its certificate."""
    directory.mkdir(parents=True, exist_ok=True)
    issued = ca.issue_client(GATEWAY_CN, days=GATEWAY_CERT_DAYS, now=now)
    final = directory / GATEWAY_CERT_FILE
    tmp = directory / f".{GATEWAY_CERT_FILE}.tmp"
    tmp.unlink(missing_ok=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(issued.key_pem() + issued.cert_pem())
            f.flush()
            os.fsync(f.fileno())
        owner = directory.stat()
        if os.geteuid() == 0 and (owner.st_uid, owner.st_gid) != (0, 0):
            os.chown(tmp, owner.st_uid, owner.st_gid)
        tmp.replace(final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return issued


def write_new_ca(directory: Path, *, now: float | None = None) -> list[str]:
    """`loom-poolmgr new-ca --out DIR`: a new pod CA (`ca.crt`, `ca.key`, mode 0600) and
    the manager's client certificate from it (`loom-poolmgr.crt`/`.key`, valid as long
    as the CA), for the secrets `pod-ca` and `poolmgr-client` (docs/runbook-pod-ca.md).
    Refuses to overwrite an existing CA. Returns the stems written."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("ca.crt", "ca.key"):
        if (directory / name).exists():
            raise FileExistsError(f"{directory / name} exists: not overwriting a CA")
    ca = CertificateAuthority.generate("Loom pod CA", days=NEW_CA_DAYS, now=now)
    _write(directory / "ca.key", IssuedCert(ca.cert, ca.key).key_pem(), 0o600)
    _write(directory / "ca.crt", ca.cert_pem(), 0o644)
    write_issued(directory, POOLMGR_CN, ca.issue_client(POOLMGR_CN, days=NEW_CA_DAYS, now=now))
    return ["ca", POOLMGR_CN]
