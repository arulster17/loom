from __future__ import annotations

import shutil
import stat
import subprocess

import pytest
from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID

from loom_devstack.devcerts import ensure_dev_certs, main


def load(path):
    return x509.load_pem_x509_certificate(path.read_bytes())


def test_creates_ca_gateway_and_pod_certs(tmp_path):
    out = tmp_path / "certs"
    written = ensure_dev_certs(out, ("mock-pod-1", "mock-pod-2"))
    assert written == [
        "ca",
        "gateway",
        "mock-pod-1",
        "mock-pod-1.token",
        "mock-pod-2",
        "mock-pod-2.token",
    ]
    ca = load(out / "ca.crt")
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    gateway = load(out / "gateway.crt")
    gateway.verify_directly_issued_by(ca)
    eku = gateway.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert list(eku) == [ExtendedKeyUsageOID.CLIENT_AUTH]
    pod = load(out / "mock-pod-1.crt")
    pod.verify_directly_issued_by(ca)
    san = pod.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["mock-pod-1"]
    assert list(pod.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value) == [
        ExtendedKeyUsageOID.SERVER_AUTH
    ]
    for name in ("ca.key", "gateway.key", "mock-pod-1.key", "mock-pod-1.token"):
        assert stat.S_IMODE((out / name).stat().st_mode) == 0o600
    tokens = {(out / f"mock-pod-{i}.token").read_text().strip() for i in (1, 2)}
    assert len(tokens) == 2 and all(len(t) >= 40 for t in tokens)


def test_idempotent_and_adds_new_pods(tmp_path):
    ensure_dev_certs(tmp_path, ("mock-pod-1",))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert ensure_dev_certs(tmp_path, ("mock-pod-1",)) == []
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert ensure_dev_certs(tmp_path, ("mock-pod-1", "mock-pod-3")) == [
        "mock-pod-3",
        "mock-pod-3.token",
    ]
    assert (tmp_path / "mock-pod-1.crt").read_bytes() == before["mock-pod-1.crt"]


def test_force_replaces_the_ca_and_reissues(tmp_path):
    ensure_dev_certs(tmp_path, ("mock-pod-1",))
    old_ca = (tmp_path / "ca.crt").read_bytes()
    old_pod = (tmp_path / "mock-pod-1.crt").read_bytes()
    ensure_dev_certs(tmp_path, ("mock-pod-1",), force=True)
    assert (tmp_path / "ca.crt").read_bytes() != old_ca
    assert (tmp_path / "mock-pod-1.crt").read_bytes() != old_pod
    load(tmp_path / "mock-pod-1.crt").verify_directly_issued_by(load(tmp_path / "ca.crt"))


def test_leaf_from_another_ca_is_reissued(tmp_path):
    ensure_dev_certs(tmp_path, ("mock-pod-1",))
    (tmp_path / "ca.crt").unlink()  # a new CA on the next run: every leaf must follow
    written = ensure_dev_certs(tmp_path, ("mock-pod-1",))
    assert written == ["ca", "gateway", "mock-pod-1"]


def test_readable_keys_cli(tmp_path, capsys):
    main(["--out", str(tmp_path), "--pod", "p1", "--readable-keys"])
    assert "written" in capsys.readouterr().out
    assert stat.S_IMODE((tmp_path / "p1.key").stat().st_mode) == 0o644
    assert stat.S_IMODE((tmp_path / "ca.key").stat().st_mode) == 0o600
    main(["--out", str(tmp_path), "--pod", "p1"])
    assert "up to date" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("openssl") is None, reason="no openssl")
def test_openssl_verifies_the_chain(tmp_path):
    ensure_dev_certs(tmp_path, ("mock-pod-1",))
    for leaf, purpose in (("gateway", "sslclient"), ("mock-pod-1", "sslserver")):
        done = subprocess.run(
            [
                "openssl",
                "verify",
                "-purpose",
                purpose,
                "-CAfile",
                str(tmp_path / "ca.crt"),
                str(tmp_path / f"{leaf}.crt"),
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout + done.stderr
