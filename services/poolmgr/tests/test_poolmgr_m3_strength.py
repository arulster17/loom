"""M3 pod PKI and mock-backend settings: boundaries and branches the other M3 pool
manager tests leave open (docs/phase1-m3-spec.md §6.8, §6.12, §7, §13 A25, A27;
contracts §2 `LOOM_POOLMGR_*`, §9).

- A25 / §6.8: with the mock backend outside dev, `LOOM_POOLMGR_ALLOW_CIDRS` takes only
  private (RFC 1918) networks; an IPv6 network is not one, and is refused with that
  problem (by `production_problems` and by `loom-poolmgr run`, exit 2), not a crash.
- §7 settings: a blank `LOOM_POOLMGR_GATEWAY_CERT_DIR` or `LOOM_POOLMGR_LEASE_BUCKET`
  counts as unset (no gateway certificate is written anywhere; the bucket is "not set").
- A27 "at start when the current one has under 6 days left": the boundary, on the
  manager's injected clock (a `FakeClock` at a whole second, so the certificate's
  `notAfter` sits exactly where the test puts it): 3 s under 6 days is re-issued, 3 s
  over is kept.
- §7 / §6.12 "owned by the directory's owner": the gateway file's owner. The tests do not
  run as root, so the writer's euid, the directory's ownership and `os.chown` are faked
  (the fake `chown` refuses, as the kernel does, a non-root process giving a file away).
- §7 `loom-poolmgr new-ca --out DIR`: `ca.key` (and the manager's client key) mode 0600
  (docs/runbook-pod-ca.md step 1), and an existing CA in DIR (either file of it) is never
  overwritten.
"""

from __future__ import annotations

import errno
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from poolmgr_fakes import CLOCK_LAG_S, FakeClock
from poolmgr_m3_kit import (
    DAY_S,
    PROD_DATABASE_URL,
    PROD_REDIS_URL,
    PoolmgrRun,
    check_gateway_cert,
    combined_pem,
    gateway_pems,
    poolmgr_run_fixture,  # noqa: F401
    serving_file,
    write_ca,
)

from loom_platform.serving import load_serving
from loom_platform.testing import pg_url  # noqa: F401
from loom_poolmgr import pki as pki_module
from loom_poolmgr.manager import PoolManager
from loom_poolmgr.pki import GATEWAY_CN, CertificateAuthority
from loom_poolmgr.settings import PoolmgrSettings

FLAG = "LOOM_POOLMGR_MOCK_BACKEND"
ALLOW = "LOOM_POOLMGR_ALLOW_CIDRS"
CERT_DIR = "LOOM_POOLMGR_GATEWAY_CERT_DIR"
BUCKET = "LOOM_POOLMGR_LEASE_BUCKET"
# contracts §2 `LOOM_POOLMGR_GATEWAY_CERT_DIR`: the file the gateway reads.
GATEWAY_FILE = "loom-gateway.pem"
RENEW_BEFORE_S = 6 * DAY_S  # §7: re-issued at start with under 6 days left


def problems_for(h: PoolmgrRun, provider: str = "local") -> list[str]:
    path = serving_file(h.tmp / "serving", provider=provider)
    h.mp.setenv("LOOM_POOLMGR_SERVING_YAML", str(path))
    return PoolmgrSettings().production_problems(load_serving(path, check_registry=False))


# ------------------------------------------------------------------ A25: IPv6 ALLOW_CIDRS

IPV6 = ["fd00::/8", "::/0", "2001:db8::/32", "10.1.0.0/24,fd12:3456::/48"]


@pytest.mark.parametrize("cidrs", IPV6)
def test_item1_ipv6_allow_cidrs_with_the_mock_backend_is_refused_as_not_private(
    poolmgr_run: PoolmgrRun, cidrs: str
) -> None:
    """A25 / §6.8: outside dev with the mock backend, an IPv6 network in
    `LOOM_POOLMGR_ALLOW_CIDRS` (even a unique-local one) is not an RFC 1918 network: the
    production checks report the private-networks-only problem naming it, and do not
    raise."""
    h = poolmgr_run
    h.prod()
    h.mp.setenv(FLAG, "true")
    h.mp.setenv(ALLOW, cidrs)
    found = problems_for(h)
    v6 = [c.strip() for c in cidrs.split(",") if ":" in c]
    lines = [p for p in found if ALLOW in p and "private" in p.lower()]
    assert lines, f"no private-networks-only problem for {cidrs!r}: {found}"
    for net in v6:
        assert any(net in line for line in lines), f"{net} not named in {lines}"


def test_item1_run_refuses_ipv6_allow_cidrs_with_exit_2(
    poolmgr_run: PoolmgrRun, capsys: pytest.CaptureFixture[str]
) -> None:
    """A25: `loom-poolmgr run` with the same settings exits 2 before building a manager,
    and its refusal carries the private-networks-only line."""
    h = poolmgr_run
    h.prod()
    h.mp.setenv(FLAG, "true")
    h.mp.setenv(ALLOW, "fd00::/8")
    code = h.main("run")
    assert code == 2, f"loom-poolmgr run exited {code}"
    assert h.constructed == [], "a manager was built"
    err = capsys.readouterr().err
    assert ALLOW in err and "fd00::/8" in err and "private" in err.lower(), err


# ------------------------------------------------------------------ §7: blank is unset


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
def test_item2_blank_gateway_cert_dir_is_unset(poolmgr_run: PoolmgrRun, blank: str) -> None:
    """§7 / contracts §2: `LOOM_POOLMGR_GATEWAY_CERT_DIR` set to nothing (or spaces) is
    the same as not set: no directory (not the working directory, not one named by the
    spaces)."""
    poolmgr_run.mp.setenv(CERT_DIR, blank)
    assert PoolmgrSettings().poolmgr_gateway_cert_dir is None


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "spaces"])
@pytest.mark.parametrize("where", ["prod_local", "dev_runpod"])
def test_item2_blank_lease_bucket_is_not_set(
    poolmgr_run: PoolmgrRun, blank: str, where: str
) -> None:
    """A blank `LOOM_POOLMGR_LEASE_BUCKET` is refused as not set where a bucket is
    required: outside dev without the mock backend, and with `provider: runpod` in dev."""
    h = poolmgr_run
    if where == "prod_local":
        h.prod()
    h.mp.setenv(BUCKET, blank)
    assert PoolmgrSettings().poolmgr_lease_bucket is None
    found = problems_for(h, "local" if where == "prod_local" else "runpod")
    assert any(BUCKET in p and "not set" in p for p in found), found


@pytest.mark.postgres
@pytest.mark.redis
def test_item2_run_with_a_blank_gateway_cert_dir_writes_no_gateway_certificate(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
    tmp_path: Path,
) -> None:
    """`loom-poolmgr run` (dev) with `LOOM_POOLMGR_GATEWAY_CERT_DIR=` starts and issues
    no gateway certificate: none in its working directory (what an empty path names)."""
    h = poolmgr_run
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    h.mp.chdir(cwd)
    h.mp.setenv(CERT_DIR, "")
    h.live(pg_url)
    h.serve_metrics()

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        await run.wait_tick(0)

    h.scenario = scenario
    assert h.main("run") in (0, None)
    assert h.constructed, "no manager was built"
    written = [p for p in cwd.rglob("*") if p.is_file() and p in gateway_pems(p.parent)]
    assert written == [], f"a gateway certificate was written under the working dir: {written}"
    assert gateway_pems(h.cert_dir) == []


# ------------------------------------------------------------------ A27: the 6-day boundary


@pytest.mark.postgres
@pytest.mark.redis
@pytest.mark.parametrize(
    ("delta_s", "replaced"), [(-3, True), (3, False)], ids=["just_under_6d", "just_over_6d"]
)
def test_item3_at_start_reissued_just_under_six_days_kept_just_over(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
    delta_s: int,
    replaced: bool,
) -> None:
    """A27 "at start when the current one has under 6 days left", on the manager's
    injected clock: the gateway file holds a certificate from the manager's CA with
    6 days + `delta_s` left at the clock's time. 3 s under: the start replaces it with a
    fresh 7-day one; 3 s over: the start keeps the very file."""
    h = poolmgr_run
    h.live(pg_url)
    start = float(int(time.time() - CLOCK_LAG_S))  # a whole second
    h.clock = FakeClock(start)
    h.ca = write_ca(h.tmp / "ca-boundary", now=start - 2 * DAY_S)
    h.set_ca(h.ca)
    issued = h.ca.ca.issue_client(GATEWAY_CN, days=7, now=start - DAY_S + delta_s)
    left = issued.cert.not_valid_after_utc.timestamp() - start
    assert left == RENEW_BEFORE_S + delta_s, left
    path = h.cert_dir / GATEWAY_FILE
    path.write_bytes(combined_pem(issued))
    before = path.read_bytes()
    inode = path.stat().st_ino
    h.serve_metrics()

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        await run.wait_tick(0)

    h.scenario = scenario
    assert h.main("run") in (0, None)
    assert gateway_pems(h.cert_dir) == [path], gateway_pems(h.cert_dir)
    if not replaced:
        assert path.read_bytes() == before, f"replaced with 6 days + {delta_s} s left"
        assert path.stat().st_ino == inode
        return
    assert path.read_bytes() != before, f"kept with 6 days - {-delta_s} s left"
    cert = check_gateway_cert(path, h.ca.ca.cert)
    assert cert.serial_number != issued.cert.serial_number
    new_left = cert.not_valid_after_utc.timestamp() - start
    assert 7 * DAY_S - 60 <= new_left <= 7 * DAY_S + 60, new_left / DAY_S


# ------------------------------------------------------------------ §6.12: the file's owner


class FakeOwnership:
    """The writer's euid, the directory's (uid, gid) and `os.chown`, faked. The file the
    writer creates is owned by (euid, euid's group: the same number here) until a chown
    on it; the fake chown refuses (EPERM) a non-root writer giving a file to another uid,
    as the kernel does. `owner_of(path)` is the owner the file ends up with."""

    def __init__(
        self, mp: pytest.MonkeyPatch, directory: Path, euid: int, dir_owner: tuple[int, int]
    ) -> None:
        self.directory = directory
        self.euid = euid
        self.chowns: list[tuple[Path, int, int]] = []
        real_stat = os.stat

        def fake_stat(path: Any, *a: Any, **kw: Any) -> os.stat_result:
            st = real_stat(path, *a, **kw)
            if isinstance(path, (str, os.PathLike)) and os.fspath(path) == os.fspath(directory):
                fields = list(st[:10])
                fields[4], fields[5] = dir_owner
                return os.stat_result(fields)
            return st

        def fake_chown(path: Any, uid: int, gid: int, *a: Any, **kw: Any) -> None:
            if self.euid != 0 and uid not in (-1, self.euid):
                raise PermissionError(errno.EPERM, "Operation not permitted", os.fspath(path))
            self.chowns.append((Path(os.fspath(path)), uid, gid))

        mp.setattr(os, "geteuid", lambda: euid)
        mp.setattr(os, "stat", fake_stat)
        mp.setattr(os, "chown", fake_chown)

    def owner_of(self, final: Path) -> tuple[int, int]:
        """Owner of `final` after the write: the writer's, changed by each chown of
        `final` or of a file in the directory that is gone (renamed into `final`)."""
        uid, gid = self.euid, self.euid
        for path, u, g in self.chowns:
            if path.parent == self.directory and (path == final or not path.exists()):
                uid = uid if u == -1 else u
                gid = gid if g == -1 else g
        return uid, gid


@pytest.mark.parametrize(
    "dir_owner", [(10001, 10001), (0, 1), (1, 0), (0, 0)], ids=lambda o: f"dir_{o[0]}_{o[1]}"
)
def test_item4_as_root_the_gateway_file_gets_the_directorys_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dir_owner: tuple[int, int]
) -> None:
    """§7 / §6.12: written by root, the gateway file ends up owned by the directory's
    owner, uid and gid (the gateway's uid 10001 on the box; also a directory whose uid or
    gid alone is root's), mode 0400, holding the key and a certificate from the CA."""
    ca = CertificateAuthority.generate("Loom strength CA")
    directory = tmp_path / "gateway-cert"
    directory.mkdir()
    fake = FakeOwnership(monkeypatch, directory, euid=0, dir_owner=dir_owner)
    pki_module.write_gateway_cert(directory, ca)
    final = directory / GATEWAY_FILE
    assert fake.owner_of(final) == dir_owner, (fake.chowns, dir_owner)
    monkeypatch.undo()
    assert stat.S_IMODE(final.stat().st_mode) == 0o400
    check_gateway_cert(final, ca.cert)


def test_item4_not_root_the_file_is_written_without_giving_it_away(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§6.12: a non-root writer (euid 4242) in a directory owned by another uid cannot
    give the file away (the kernel refuses): the write still succeeds, the file is the
    writer's, in place, mode 0400."""
    ca = CertificateAuthority.generate("Loom strength CA")
    directory = tmp_path / "gateway-cert"
    directory.mkdir()
    fake = FakeOwnership(monkeypatch, directory, euid=4242, dir_owner=(10001, 10001))
    pki_module.write_gateway_cert(directory, ca)
    final = directory / GATEWAY_FILE
    assert fake.owner_of(final) == (4242, 4242)
    monkeypatch.undo()
    assert stat.S_IMODE(final.stat().st_mode) == 0o400
    check_gateway_cert(final, ca.cert)
    assert [p.name for p in directory.iterdir()] == [GATEWAY_FILE]


# ------------------------------------------------------------------ §7: new-ca


def outside_dev(h: PoolmgrRun) -> None:
    h.mp.setenv("LOOM_ENV", "prod")
    h.mp.setenv("LOOM_DATABASE_URL", PROD_DATABASE_URL)  # nothing listens there
    h.mp.setenv("LOOM_REDIS_URL", PROD_REDIS_URL)


def test_item10_new_ca_keys_are_mode_0600(poolmgr_run: PoolmgrRun) -> None:
    """§7, runbook step 1: `new-ca --out DIR` writes `ca.key` and the manager's client
    key `loom-poolmgr.key` mode 0600, whatever the umask (0 here)."""
    h = poolmgr_run
    outside_dev(h)
    out = h.tmp / "pod-ca-next"
    old = os.umask(0)
    try:
        code = h.main("new-ca", "--out", str(out))
    finally:
        os.umask(old)
    assert code in (0, None), f"loom-poolmgr new-ca exited {code}"
    for name in ("ca.key", "loom-poolmgr.key"):
        mode = stat.S_IMODE((out / name).stat().st_mode)
        assert mode == 0o600, f"{name} has mode {mode:o}, not 600"
    CertificateAuthority.load(out / "ca.crt", out / "ca.key")


@pytest.mark.parametrize("left", ["both", "ca.crt", "ca.key"])
def test_item10_new_ca_refuses_to_overwrite_an_existing_ca(
    poolmgr_run: PoolmgrRun, left: str
) -> None:
    """§7, runbook step 1: `new-ca` into a DIR that already holds a CA (both files, or
    either one of them) fails, and the files there are left exactly as they were."""
    h = poolmgr_run
    outside_dev(h)
    out = h.tmp / "pod-ca-next"
    assert h.main("new-ca", "--out", str(out)) in (0, None)
    if left != "both":
        other = "ca.key" if left == "ca.crt" else "ca.crt"
        (out / other).unlink()
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    code = h.main("new-ca", "--out", str(out))
    assert code not in (0, None), f"new-ca over an existing CA ({left}) exited {code}"
    after = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    assert after == before, f"new-ca changed the files of an existing CA ({left})"


# fu-flake-proof: no-op line so flake-check runs this file (draft PR, never merged).
