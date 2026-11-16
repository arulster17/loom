"""Pod certificates (docs/phase1-m2-spec.md §13 item 16; §5 "Certificates"; §9 SIGHUP):
issued with SAN = pod id and the right key usages; renewed over SSH before expiry, reaching
the proxy through a reload while a stream stays open."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from cryptography import x509
from cryptography.x509.oid import ExtendedKeyUsageOID
from poolmgr_fakes import (
    FakeServing,
    ProxyStream,
    bring_up,
    env_fixture,  # noqa: F401
    peer_certificate,
    proxy_get,
    serving_config,
    tick,
    unique_model_id,
)

from loom_platform.testing import pg_url, redis_client  # noqa: F401
from loom_poolmgr.pki import Pki


def check_server_cert(cert: x509.Certificate, pod_id: str, pki: Pki) -> None:
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == [pod_id]
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert list(eku) == [ExtendedKeyUsageOID.SERVER_AUTH]
    usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    assert usage.digital_signature and not usage.key_cert_sign
    assert not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    cert.verify_directly_issued_by(pki.ca.cert)


def test_pki_issues_server_and_client_certificates(tmp_path: Path) -> None:
    """Item 16 at the pki level: a server certificate (SAN = pod id, serverAuth) and the
    manager's client certificate (clientAuth), both from the given CA."""
    pki = Pki.generate(tmp_path)
    check_server_cert(pki.ca.issue_server("podabc123", days=7).cert, "podabc123", pki)
    client = x509.load_pem_x509_certificate(pki.client_cert_path.read_bytes())
    eku = client.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert list(eku) == [ExtendedKeyUsageOID.CLIENT_AUTH]
    client.verify_directly_issued_by(pki.ca.cert)


async def test_delivered_certificate_is_the_pods_own(env: FakeServing) -> None:
    """Item 16: the certificate the manager delivers, and the proxy presents, names the pod
    id, is for serverAuth, lasts server_cert_days, and is recorded as cert_not_after."""
    model = unique_model_id()
    manager = env.manager(serving_config([model]))  # server_cert_days 7
    await manager.start()
    pod = await bring_up(env, manager, model)
    delivered = x509.load_pem_x509_certificate(pod.payloads[-1]["server.crt"])
    check_server_cert(delivered, pod.pod_id, env.pki)
    (boot,) = [c for c in env.pods.calls if c.kind == "serve_bootstrap"]
    lifetime = delivered.not_valid_after_utc.timestamp() - boot.at
    assert abs(lifetime - 7 * 86400) < 600, lifetime
    (replica,) = (await env.pool(model)).replicas
    presented = x509.load_der_x509_certificate(await peer_certificate(replica, env.pki))
    assert presented == delivered
    (row,) = await env.rows(model)
    assert abs((row["cert_not_after"] - delivered.not_valid_after_utc).total_seconds()) < 1


async def test_renewal_before_expiry_reaches_proxy_with_stream_open(env: FakeServing) -> None:
    """Item 16: within server_cert_renew_before_s of expiry the manager writes a new
    certificate over SSH and the proxy reloads it (SIGHUP): new connections get it, a
    stream open across the reload completes, and the row's cert_not_after moves."""
    model = unique_model_id()
    # One-day certificates renewed when less than a day minus 60 s is left: due 60 s in.
    # The progress rule is pushed out of the way of the long open stream.
    config = serving_config(
        [model],
        lifecycle={"server_cert_days": 1, "server_cert_renew_before_s": 86400 - 60},
        health={"progress_stall_s": 600},
    )
    manager = env.manager(config)
    await manager.start()
    pod = await bring_up(env, manager, model)
    (boot,) = [c for c in env.pods.calls if c.kind == "serve_bootstrap"]
    (replica,) = (await env.pool(model)).replicas
    old = x509.load_der_x509_certificate(await peer_certificate(replica, env.pki))

    stream = await ProxyStream(env.client(), replica, model, max_tokens=400).start()
    if env.clock() < boot.at + 45:
        env.clock.set(boot.at + 45)
        await tick(manager, env.clock)
        assert pod.bootstraps == ["started"], "renewed before it was due"
    env.clock.set(max(env.clock(), boot.at + 90))
    await tick(manager, env.clock)
    assert pod.bootstraps == ["started", "reloaded"]
    assert not stream.finished, "the stream ended before the reload (test too slow?)"

    (replica,) = (await env.pool(model)).replicas
    new = x509.load_der_x509_certificate(await peer_certificate(replica, env.pki))
    assert new.serial_number != old.serial_number
    assert new.not_valid_after_utc > old.not_valid_after_utc
    check_server_cert(new, pod.pod_id, env.pki)
    assert (await proxy_get(env.client(), replica, "/health")).status_code == 200
    (row,) = await env.rows(model)
    assert abs((row["cert_not_after"] - new.not_valid_after_utc).total_seconds()) < 1
    assert row["cert_not_after"] > dt.datetime.fromtimestamp(boot.at + 86400, dt.UTC)

    result = await stream.finish()
    assert result.complete and result.completion_tokens == 400, result
