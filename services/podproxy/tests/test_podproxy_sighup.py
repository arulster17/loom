"""SIGHUP reloads the pod proxy's files (docs/phase1-m2-spec.md §9, §13 item 23).

A real `loom-podproxy` process with its certificate, key, client CA, token and allowlist
in files under tmp_path. After SIGHUP new connections use the new files, an open stream
survives, and a reload that fails keeps the whole old configuration and logs at error.

The allowlist comes from `--allow-ip-file PATH`, one CIDR or address per line (§9). An
empty or blank allowlist file is a parse failure, never "any source": the proxy refuses
to start on one, and a reload with one keeps the old allowlist.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from podproxy_testlib import POD, Upstream, pod_ext

from loom_devstack.devcerts import Issued, issue, make_ca, write_issued

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP is POSIX only")

CHAT = "/v1/chat/completions"
ERROR_LINE = re.compile(r"\bERROR\b")


def pem(issued: Issued) -> bytes:
    return issued.cert.public_bytes(serialization.Encoding.PEM)


def der(path: Path) -> bytes:
    return ssl.PEM_cert_to_DER_cert(path.read_text())


class Files:
    """The proxy's files, plus what a client needs to talk to it."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "proxy"
        self.dir.mkdir()
        self.trust = root / "trust"  # the client's own copy of the CA (verifies the server)
        self.trust.mkdir()
        self.ca = make_ca("Loom test CA")
        (self.dir / "ca.crt").write_bytes(pem(self.ca))
        (self.trust / "ca.crt").write_bytes(pem(self.ca))
        write_issued(self.dir, POD, issue(self.ca, POD, server=True), 0o600)
        write_issued(self.trust, "gateway", issue(self.ca, "loom-gateway", server=False), 0o600)
        self.write_token(secrets.token_urlsafe(24))
        self.allow.write_text("127.0.0.1/32\n")

    @property
    def cert(self) -> Path:
        return self.dir / f"{POD}.crt"

    @property
    def key(self) -> Path:
        return self.dir / f"{POD}.key"

    @property
    def client_ca(self) -> Path:
        return self.dir / "ca.crt"

    @property
    def token_file(self) -> Path:
        return self.dir / "pod.token"

    @property
    def allow(self) -> Path:
        return self.dir / "allow.txt"

    def token(self) -> str:
        return self.token_file.read_text().strip()

    def write_token(self, value: str) -> None:
        self.token_file.write_text(value + "\n")

    def new_server_cert(self, ca: Issued | None = None) -> None:
        write_issued(self.dir, POD, issue(ca or self.ca, POD, server=True), 0o600)

    def client(self, stem: str = "gateway", where: Path | None = None) -> ssl.SSLContext:
        """Verifies the server with the original CA; presents `where/stem.crt`."""
        ctx = ssl.create_default_context(cafile=self.trust / "ca.crt")
        where = where or self.trust
        ctx.load_cert_chain(where / f"{stem}.crt", where / f"{stem}.key")
        return ctx


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _proxy_command() -> list[str]:
    script = shutil.which("loom-podproxy", path=str(Path(sys.executable).parent))
    return [script] if script else [sys.executable, "-m", "loom_podproxy.proxy"]


class ProxyProcess:
    def __init__(self, files: Files, upstream_port: int, log: Path, allow_file: bool) -> None:
        self.files = files
        self.port = _free_port()
        self.log_path = log
        args = [
            *_proxy_command(),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--upstream",
            f"http://127.0.0.1:{upstream_port}",
            "--cert",
            str(files.cert),
            "--key",
            str(files.key),
            "--client-ca",
            str(files.client_ca),
            "--token-file",
            str(files.token_file),
            "--log-level",
            "info",
        ]
        if allow_file:
            args += ["--allow-ip-file", str(files.allow)]
        env = {k: v for k, v in os.environ.items() if not k.startswith("LOOM_POD")}
        self._log = log.open("wb")
        self.proc = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT, env=env)

    def ready(self) -> None:
        token = self.files.token()
        self.wait_for(lambda: self.status(token) == 200, "the proxy to serve")

    def log(self) -> str:
        return self.log_path.read_text(errors="replace")

    def errors(self) -> int:
        return sum(1 for line in self.log().splitlines() if ERROR_LINE.search(line))

    def hup(self) -> None:
        assert self.proc.poll() is None, f"proxy not running before SIGHUP:\n{self.log()}"
        os.kill(self.proc.pid, signal.SIGHUP)

    def wait_for(self, cond: Callable[[], bool], what: str, timeout: float = 15) -> None:
        deadline = time.monotonic() + timeout
        while not cond():
            code = self.proc.poll()
            assert code is None, f"proxy exited ({code}) waiting for {what}:\n{self.log()}"
            assert time.monotonic() < deadline, f"timed out waiting for {what}:\n{self.log()}"
            time.sleep(0.05)

    def status(self, token: str, ctx: ssl.SSLContext | None = None) -> int | None:
        """A new connection's status for GET /v1/models; None when refused at TLS/TCP."""
        ctx = ctx or self.files.client()
        try:
            with httpx.Client(verify=ctx, timeout=5, trust_env=False) as http:
                r = http.get(
                    f"https://127.0.0.1:{self.port}/v1/models",
                    headers={"X-Loom-Pod-Token": token},
                    extensions=pod_ext(),
                )
        except httpx.TransportError:
            return None
        return r.status_code

    def presented_cert(self) -> bytes | None:
        """The server certificate a new TLS connection sees (DER)."""
        ctx = self.files.client()
        try:
            with (
                socket.create_connection(("127.0.0.1", self.port), timeout=5) as raw,
                ctx.wrap_socket(raw, server_hostname=POD) as tls,
            ):
                return tls.getpeercert(binary_form=True)
        except (OSError, ssl.SSLError):
            return None

    def closed_before_handshake(self) -> bool:
        """True when a new TCP connection is closed without a TLS handshake (allowlist)."""
        try:
            raw = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        except OSError:
            return False  # not listening (wait_for reports a dead process)
        with raw:
            raw.settimeout(1.0)
            try:
                return raw.recv(1) == b""
            except TimeoutError:
                return False  # waiting for our ClientHello: accepted
            except ConnectionResetError:
                return True

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self._log.close()


@pytest.fixture
def files(tmp_path: Path) -> Files:
    return Files(tmp_path)


@pytest.fixture
def start(
    files: Files, upstream: tuple[Upstream, int], tmp_path: Path
) -> Iterator[Callable[..., ProxyProcess]]:
    procs: list[ProxyProcess] = []

    def factory(*, allow_file: bool = False, wait: bool = True) -> ProxyProcess:
        log = tmp_path / f"proxy-{len(procs)}.log"
        proc = ProxyProcess(files, upstream[1], log, allow_file)
        procs.append(proc)  # registered before the wait: stopped even if never ready
        if wait:
            proc.ready()
        return proc

    yield factory
    for p in procs:
        p.stop()


# ---------------------------------------------------------------- reloads


def test_sighup_reloads_the_server_certificate(files: Files, start: Any) -> None:
    proxy = start()
    old = der(files.cert)
    assert proxy.presented_cert() == old

    files.new_server_cert()
    new = der(files.cert)
    assert new != old
    proxy.hup()

    proxy.wait_for(lambda: proxy.presented_cert() == new, "the new certificate")
    assert proxy.status(files.token()) == 200
    assert proxy.errors() == 0


def test_sighup_reloads_the_token(files: Files, start: Any) -> None:
    proxy = start()
    old = files.token()
    new = secrets.token_urlsafe(24)
    files.write_token(new)
    assert proxy.status(old) == 200  # nothing changes before the signal

    proxy.hup()

    proxy.wait_for(lambda: proxy.status(new) == 200, "the new token")
    assert proxy.status(old) == 403
    assert proxy.status("") == 403


def test_sighup_reloads_the_client_ca(files: Files, start: Any, tmp_path: Path) -> None:
    proxy = start()
    other = make_ca("Loom next CA")
    later = tmp_path / "next"
    later.mkdir()
    write_issued(later, "gateway", issue(other, "loom-gateway", server=False), 0o600)
    assert proxy.status(files.token(), files.client("gateway", later)) is None

    files.client_ca.write_bytes(pem(other))
    proxy.hup()

    proxy.wait_for(
        lambda: proxy.status(files.token(), files.client("gateway", later)) == 200,
        "a client certificate from the new CA",
    )
    assert proxy.status(files.token()) is None  # the old CA's client certificate


def test_sighup_reloads_the_allowlist(files: Files, start: Any) -> None:
    proxy = start(allow_file=True)
    assert not proxy.closed_before_handshake()

    files.allow.write_text("10.0.0.0/8\n")
    proxy.hup()
    proxy.wait_for(proxy.closed_before_handshake, "127.0.0.1 to be refused")
    assert proxy.status(files.token()) is None

    files.allow.write_text("10.0.0.0/8\n127.0.0.1\n")
    proxy.hup()
    proxy.wait_for(lambda: proxy.status(files.token()) == 200, "127.0.0.1 to be allowed again")


BLANK = ["", "\n", "  \n\t\n"]


@pytest.mark.parametrize("blank", BLANK, ids=["empty", "newline", "whitespace"])
def test_a_blank_allowlist_file_refuses_to_start(files: Files, start: Any, blank: str) -> None:
    first = start(allow_file=True)  # the same command line with a valid file serves
    first.stop()

    files.allow.write_text(blank)
    proxy = start(allow_file=True, wait=False)
    try:
        code = proxy.proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        code = None
    assert code not in (None, 0), f"proxy started on a blank allowlist:\n{proxy.log()}"


@pytest.mark.parametrize("blank", BLANK, ids=["empty", "newline", "whitespace"])
def test_a_blank_allowlist_file_on_reload_keeps_the_old_allowlist(
    files: Files, start: Any, blank: str
) -> None:
    """A truncated allowlist must not open the proxy to any source."""
    proxy = start(allow_file=True)
    files.allow.write_text("10.0.0.0/8\n")
    proxy.hup()
    proxy.wait_for(proxy.closed_before_handshake, "127.0.0.1 to be refused")
    assert proxy.errors() == 0

    files.allow.write_text(blank)
    proxy.hup()

    proxy.wait_for(lambda: proxy.errors() > 0, "the failed reload to be logged at error")
    assert proxy.closed_before_handshake()  # still outside the old allowlist
    assert proxy.status(files.token()) is None
    assert proxy.proc.poll() is None


def test_an_open_stream_survives_a_reload(
    files: Files, start: Any, upstream: tuple[Upstream, int]
) -> None:
    """Certificate and token rotate, and the allowlist drops the stream's own source
    address: the stream already open completes in full anyway."""
    app, _ = upstream
    app.chunks = [b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"]
    app.hold_after = 1
    proxy = start(allow_file=True)
    old_token = files.token()

    received: list[bytes] = []
    with (
        httpx.Client(verify=files.client(), timeout=30, trust_env=False) as http,
        http.stream(
            "POST",
            f"https://127.0.0.1:{proxy.port}{CHAT}",
            json={"stream": True},
            headers={"X-Loom-Pod-Token": old_token},
            extensions=pod_ext(),
        ) as r,
    ):
        assert r.status_code == 200
        chunks = r.iter_raw()
        received.append(next(chunks))
        held_since = time.monotonic()

        files.new_server_cert()
        new_cert = der(files.cert)
        files.write_token(secrets.token_urlsafe(24))
        files.allow.write_text("10.0.0.0/8\n")
        proxy.hup()
        proxy.wait_for(proxy.closed_before_handshake, "the reload (127.0.0.1 refused)")
        # The reload landed while the stream was still held open by the upstream (it waits
        # at most 10 s for `release`), so the rest of the stream crosses the reload.
        assert not app.release.is_set()
        assert time.monotonic() - held_since < 8, "the upstream hold ran out before the reload"

        app.release.set()
        received.extend(chunks)

    assert b"".join(received) == b"".join(app.chunks)

    files.allow.write_text("127.0.0.1\n")
    proxy.hup()
    proxy.wait_for(lambda: proxy.status(files.token()) == 200, "the next reload")
    assert proxy.presented_cert() == new_cert
    assert proxy.status(old_token) == 403


# ---------------------------------------------------------------- a bad reload


def _break(files: Files, case: str) -> None:
    if case == "cert-garbage":
        files.cert.write_text("not a certificate\n")
    elif case == "cert-missing":
        files.cert.unlink()
    elif case == "key-mismatch":
        other = ec.generate_private_key(ec.SECP256R1())
        files.key.write_bytes(
            other.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    elif case == "key-garbage":
        files.key.write_text("-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----\n")
    elif case == "ca-garbage":
        files.client_ca.write_text("not a CA\n")
    elif case == "token-empty":
        files.token_file.write_text("\n")
    elif case == "allowlist-garbage":
        files.allow.write_text("127.0.0.1/32\nnot-a-cidr\n")
    else:
        raise AssertionError(case)


BAD = [
    "cert-garbage",
    "cert-missing",
    "key-mismatch",
    "key-garbage",
    "ca-garbage",
    "token-empty",
    "allowlist-garbage",
]


@pytest.mark.parametrize("case", BAD)
def test_a_bad_reload_keeps_the_whole_old_configuration(
    files: Files, start: Any, case: str
) -> None:
    """Every other file gets a valid new version, then one is broken: none of the new
    values take effect, the error is logged, and a later good reload applies them."""
    proxy = start(allow_file=True)
    old_cert, old_token = der(files.cert), files.token()
    assert proxy.errors() == 0

    files.new_server_cert()
    new_cert = der(files.cert)
    new_token = secrets.token_urlsafe(24)
    files.write_token(new_token)
    _break(files, case)
    proxy.hup()

    proxy.wait_for(lambda: proxy.errors() > 0, "the failed reload to be logged at error")
    assert proxy.presented_cert() == old_cert
    assert proxy.status(old_token) == 200
    if case != "token-empty":
        assert proxy.status(new_token) == 403
    assert proxy.proc.poll() is None

    # Fixed: the next SIGHUP applies everything.
    files.new_server_cert()
    new_cert = der(files.cert)
    files.write_token(new_token)
    files.client_ca.write_bytes(pem(files.ca))
    files.allow.write_text("127.0.0.1/32\n")
    proxy.hup()
    proxy.wait_for(lambda: proxy.status(new_token) == 200, "the good reload")
    assert proxy.presented_cert() == new_cert
    assert proxy.status(old_token) == 403
