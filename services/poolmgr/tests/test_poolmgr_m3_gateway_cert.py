"""A27, the manager's half: the gateway's client certificate (docs/phase1-m3-spec.md §7;
contracts §2 `LOOM_POOLMGR_GATEWAY_CERT_DIR`, §9).

The manager issues the gateway's client certificate (`clientAuth`, 7 days, from its pod
CA) every 24 h, or at start when the current one has under 6 days left, is missing,
fails to load or was not issued by the current CA (§6.13), into
`LOOM_POOLMGR_GATEWAY_CERT_DIR`: one PEM file with key and certificate, written to a
temporary name (mode `0400`, owned by the directory's owner, §6.12) and renamed into
place (a reader never sees a key without its certificate). Runbook step 5 (§7,
`docs/runbook-pod-ca.md`): after the switch to a new CA the manager issues the gateway's
certificate from the new CA on start.

Driven through `loom-poolmgr run` (`poolmgr_m3_kit.PoolmgrRun`): the file name inside the
directory is the manager's choice, so the tests find the one file holding a key and a
certificate. "Renamed into place" is observed as a new inode on each write (an in-place
rewrite keeps the inode, and a reader could see it half written). The 24 h schedule and
the certificate's 7-day validity run on the manager's injected clock (`clock=`, a
`FakeClock` in the 24 h test), as the rest of the manager's timing does; the tests at
start use the default clock (real time).
"""

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path

import pytest
from cryptography.x509.oid import NameOID
from poolmgr_fakes import FakeClock
from poolmgr_m3_kit import (
    DAY_S,
    PoolmgrRun,
    assert_pem_unchanged,
    check_gateway_cert,
    combined_pem,
    gateway_pems,
    pem_identity,
    poolmgr_run_fixture,  # noqa: F401
    wait_for_new_pem,
    write_ca,
)

from loom_platform.testing import pg_url  # noqa: F401
from loom_poolmgr.manager import PoolManager
from loom_poolmgr.pki import GATEWAY_CN

pytestmark = [pytest.mark.postgres, pytest.mark.redis]


def the_file(directory: Path) -> Path:
    found = gateway_pems(directory)
    assert len(found) == 1, f"expected one PEM with key and certificate in {directory}: {found}"
    return found[0]


async def wait_for_file(directory: Path, timeout_s: float = 10.0) -> Path:
    deadline = time.monotonic() + timeout_s
    while not gateway_pems(directory):
        if time.monotonic() > deadline:
            raise AssertionError(f"no gateway certificate in {directory} after the start")
        await asyncio.sleep(0.05)
    return the_file(directory)


def run_once(h: PoolmgrRun, *, replaced: bool = True) -> None:
    """`loom-poolmgr run` until the manager's loop has finished at least one tick and
    the gateway file is as expected, then stop. The file's identity (inode, serial) is
    taken before the start. With no file before: wait (real time) for one. With a file
    and `replaced`: wait until it is a different one (a write during the start or the
    first ticks both count). With a file and not `replaced`: it stays the same file for
    1 s after the first tick."""
    old = pem_identity(h.cert_dir)
    if old is None:
        assert gateway_pems(h.cert_dir) == [], "a gateway file is there but does not parse"

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        await run.wait_tick(0)
        if old is None:
            await wait_for_file(run.cert_dir)
        elif replaced:
            await wait_for_new_pem(run.cert_dir, old)
        else:
            await assert_pem_unchanged(run.cert_dir, old, for_s=1.0)

    h.scenario = scenario
    h.serve_metrics()
    assert h.main("run") in (0, None)


def test_issued_at_start_into_the_gateway_cert_dir(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """An empty directory: after the start it holds one PEM file with the key and a
    7-day `clientAuth` certificate from the manager's CA, loadable as one file."""
    h = poolmgr_run
    h.live(pg_url)
    assert gateway_pems(h.cert_dir) == []
    run_once(h)
    cert = check_gateway_cert(the_file(h.cert_dir), h.ca.ca.cert)
    assert cert.not_valid_after_utc.timestamp() - time.time() > 6.9 * DAY_S


@pytest.mark.parametrize(
    ("days_left", "replaced"), [(6.5, False), (5.0, True), (-1.0, True)], ids=str
)
def test_at_start_replaced_only_under_six_days_left(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
    days_left: float,
    replaced: bool,
) -> None:
    """A second start finds the file the first one wrote, now holding a certificate
    from the same CA with `days_left` left: kept as it is at 6.5 days; replaced at 5
    days (and when expired) by a fresh 7-day one, renamed into place."""
    h = poolmgr_run
    h.live(pg_url)
    run_once(h)
    path = the_file(h.cert_dir)
    issued = h.ca.ca.issue_client(GATEWAY_CN, days=7, now=time.time() - (7 - days_left) * DAY_S)
    tmp = path.with_name(path.name + ".test")
    tmp.write_bytes(combined_pem(issued))
    os.replace(tmp, path)
    before = path.read_bytes()
    inode = path.stat().st_ino

    run_once(h, replaced=replaced)
    path = the_file(h.cert_dir)
    if not replaced:
        assert path.read_bytes() == before, "a certificate with 6.5 days left was replaced"
        return
    assert path.read_bytes() != before, f"a certificate with {days_left} days left was kept"
    cert = check_gateway_cert(path, h.ca.ca.cert)
    assert cert.serial_number != issued.cert.serial_number
    assert cert.not_valid_after_utc.timestamp() - time.time() > 6.9 * DAY_S
    assert path.stat().st_ino != inode, "rewritten in place, not renamed into place"


def test_at_start_a_certificate_from_another_ca_is_replaced(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """Runbook step 5 (§7): after the manager is switched to a new CA, it issues the
    gateway's certificate from the new CA on start, even though the old CA's certificate
    still has more than 6 days left."""
    h = poolmgr_run
    h.live(pg_url)
    run_once(h)
    old = check_gateway_cert(the_file(h.cert_dir), h.ca.ca.cert)
    new_ca = write_ca(h.tmp / "ca-next")
    h.set_ca(new_ca)
    run_once(h)
    cert = check_gateway_cert(the_file(h.cert_dir), new_ca.ca.cert)
    assert cert.serial_number != old.serial_number


def replace_file(path: Path, data: bytes) -> None:
    """Put `data` at `path` (renamed into place, the test's own default mode)."""
    tmp = path.with_name(path.name + ".test")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def first_start(h: PoolmgrRun, database_url: str) -> Path:
    """A first `run` into the empty directory; the file it wrote (from the manager's CA)."""
    h.live(database_url)
    run_once(h)
    path = the_file(h.cert_dir)
    check_gateway_cert(path, h.ca.ca.cert)
    return path


def assert_fresh_at(h: PoolmgrRun, path: Path, *, not_serial: int | None = None) -> None:
    """After a second start: `path` (the name the gateway reads) holds a fresh 7-day
    certificate from the manager's current CA, the only gateway file in the directory."""
    assert the_file(h.cert_dir) == path, (
        f"the gateway reads {path.name}, the manager wrote {the_file(h.cert_dir).name}"
    )
    cert = check_gateway_cert(path, h.ca.ca.cert)
    assert cert.not_valid_after_utc.timestamp() - time.time() > 6.9 * DAY_S
    if not_serial is not None:
        assert cert.serial_number != not_serial


def test_at_start_a_missing_file_is_reissued(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """§7/§6.13: the file the first start wrote is gone (deleted by hand, a new volume):
    the next start issues one again, under the same name."""
    h = poolmgr_run
    path = first_start(h, pg_url)
    old_serial = check_gateway_cert(path, h.ca.ca.cert).serial_number
    path.unlink()
    run_once(h)
    assert_fresh_at(h, path, not_serial=old_serial)


@pytest.mark.parametrize("broken", ["key_only", "mismatched_key", "not_pem", "empty"])
def test_at_start_a_file_that_fails_to_load_is_replaced(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
    broken: str,
) -> None:
    """§7/§6.13: the gateway's file is there but does not load as a key and certificate
    (a key without its certificate, a key that is not the certificate's, not PEM, empty),
    even though the certificate part would have more than 6 days left: the next start
    replaces it with a fresh one from the current CA."""
    h = poolmgr_run
    path = first_start(h, pg_url)
    a = h.ca.ca.issue_client(GATEWAY_CN, days=7)
    b = h.ca.ca.issue_client(GATEWAY_CN, days=7)
    data = {
        "key_only": a.key_pem(),
        "mismatched_key": a.key_pem() + b.cert_pem(),
        "not_pem": b"this is not a PEM file\n",
        "empty": b"",
    }[broken]
    replace_file(path, data)
    run_once(h)
    assert_fresh_at(h, path, not_serial=b.cert.serial_number)


@pytest.mark.parametrize("same_name", [True, False], ids=["same_ca_name", "other_ca_name"])
def test_at_start_a_certificate_not_issued_by_the_current_ca_is_replaced(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
    same_name: bool,
) -> None:
    """§7/§6.13: the file holds a valid 7-day `clientAuth` certificate with its key, but
    from another CA than the manager's (one with the same common name, which only the
    signature tells apart, or another name): the next start replaces it with one from
    the current CA. The manager's CA itself is unchanged."""
    h = poolmgr_run
    path = first_start(h, pg_url)
    current_cn = h.ca.ca.cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    other = write_ca(h.tmp / "ca-other", cn=str(current_cn) if same_name else "Some other CA")
    assert (other.ca.cert.subject == h.ca.ca.cert.subject) is same_name
    foreign = other.ca.issue_client(GATEWAY_CN, days=7)
    replace_file(path, combined_pem(foreign))
    run_once(h)
    assert_fresh_at(h, path, not_serial=foreign.cert.serial_number)


def assert_mode_and_owner(path: Path) -> None:
    st = path.stat()
    mode = stat.S_IMODE(st.st_mode)
    assert mode == 0o400, f"{path.name} has mode {mode:o}, not 400"
    owner = path.parent.stat().st_uid
    assert st.st_uid == owner, f"{path.name} is owned by uid {st.st_uid}, the directory by {owner}"


def test_the_file_is_mode_0400_and_owned_by_the_directorys_owner(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """§7, §6.12: the gateway file is written mode `0400`, owned by the directory's owner
    (so the gateway's uid, which owns the directory on the box, can read it): at the
    first start, and when a later start replaces a certificate with under 6 days left
    that the test wrote with its own default mode. Run as root, the directory is first
    given to uid 10001 (the gateway's, §6.12), so the owner is not just the writer's;
    otherwise the directory's owner is the test's own uid."""
    h = poolmgr_run
    if os.geteuid() == 0:
        os.chown(h.cert_dir, 10001, 10001)
    path = first_start(h, pg_url)
    assert_mode_and_owner(path)

    issued = h.ca.ca.issue_client(GATEWAY_CN, days=7, now=time.time() - 2 * DAY_S)
    replace_file(path, combined_pem(issued))
    assert stat.S_IMODE(path.stat().st_mode) != 0o400
    run_once(h)
    assert_fresh_at(h, path, not_serial=issued.cert.serial_number)
    assert_mode_and_owner(path)


def test_reissued_every_24_hours_and_renamed_into_place(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """On the manager's injected clock (a `FakeClock`): issued at start; the same file
    12 h later; a new certificate (a new inode) after 24 h, and again after 48 h, each
    valid 7 days from the clock's time of the rotation. After each clock move the test
    polls the file (real time): up to 10 s for a due rotation, and for 1 s that nothing
    changes when none is due."""
    h = poolmgr_run
    h.live(pg_url)
    h.clock = FakeClock()
    h.ca = write_ca(h.tmp / "ca-fake-clock", now=h.clock())
    h.set_ca(h.ca)
    h.serve_metrics()
    seen: list[tuple[int, int, float]] = []  # (serial, inode, fake time) per check

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        clock = run.clock
        assert clock is not None
        start = clock()
        path = await wait_for_file(run.cert_dir)
        first = check_gateway_cert(path, run.ca.ca.cert)
        seen.append((first.serial_number, path.stat().st_ino, clock()))

        identity = pem_identity(run.cert_dir)
        assert identity is not None
        await run.advance_to(start + DAY_S / 2)
        await assert_pem_unchanged(run.cert_dir, identity, for_s=1.0)
        path = the_file(run.cert_dir)
        assert check_gateway_cert(path, run.ca.ca.cert).serial_number == first.serial_number, (
            "re-issued 12 h after the start"
        )

        for n in (1, 2):
            identity = pem_identity(run.cert_dir)
            assert identity is not None
            await run.advance_to(start + n * DAY_S + 60 * n)
            try:
                path = await wait_for_new_pem(run.cert_dir, identity)
            except AssertionError as e:
                raise AssertionError(f"not re-issued {n * 24} h after the start: {e}") from e
            cert = check_gateway_cert(path, run.ca.ca.cert)
            seen.append((cert.serial_number, path.stat().st_ino, clock()))
            assert cert.serial_number != seen[-2][0], f"not re-issued {n * 24} h after the start"
            assert seen[-1][1] != seen[-2][1], "rewritten in place, not renamed into place"
            # Issued on the manager's clock: valid 7 days from the rotation.
            left = cert.not_valid_after_utc.timestamp() - clock()
            assert 6.9 * DAY_S < left <= 7 * DAY_S + 3600, left / DAY_S

    h.scenario = scenario
    assert h.main("run") in (0, None)
    assert len(seen) == 3, seen
