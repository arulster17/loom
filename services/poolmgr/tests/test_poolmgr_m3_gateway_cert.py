"""A27, the manager's half: the gateway's client certificate (docs/phase1-m3-spec.md §7;
contracts §2 `LOOM_POOLMGR_GATEWAY_CERT_DIR`, §9).

The manager issues the gateway's client certificate (`clientAuth`, 7 days, from its pod
CA) every 24 h, or at start when the current one has under 6 days left, into
`LOOM_POOLMGR_GATEWAY_CERT_DIR`: one PEM file with key and certificate, written to a
temporary name and renamed into place (a reader never sees a key without its
certificate). Runbook step 5 (§7, `docs/runbook-pod-ca.md`): after the switch to a new CA
the manager issues the gateway's certificate from the new CA on start.

Driven through `loom-poolmgr run` (`poolmgr_m3_kit.PoolmgrRun`): the file name inside the
directory is the manager's choice, so the tests find the one file holding a key and a
certificate. "Renamed into place" is observed as a new inode on each write (an in-place
rewrite keeps the inode, and a reader could see it half written). The 24 h schedule runs
on the manager's injected clock (`FakeClock`), as the rest of the manager's timing does.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from poolmgr_fakes import FakeClock
from poolmgr_m3_kit import (
    DAY_S,
    PoolmgrRun,
    check_gateway_cert,
    combined_pem,
    gateway_pems,
    poolmgr_run_fixture,  # noqa: F401
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


def run_once(h: PoolmgrRun) -> None:
    """`loom-poolmgr run` until the manager's loop has ticked once, then stop."""

    async def scenario(manager: PoolManager, run: PoolmgrRun) -> None:
        await wait_for_file(run.cert_dir)

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

    run_once(h)
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


def test_reissued_every_24_hours_and_renamed_into_place(
    poolmgr_run: PoolmgrRun,
    pg_url: str,  # noqa: F811
) -> None:
    """On the manager's clock: issued at start; the same file 12 h later; a new
    certificate (a new inode) after 24 h, and again after 48 h."""
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

        await run.advance_to(start + DAY_S / 2)
        path = the_file(run.cert_dir)
        assert check_gateway_cert(path, run.ca.ca.cert).serial_number == first.serial_number, (
            "re-issued 12 h after the start"
        )

        for n in (1, 2):
            await run.advance_to(start + n * DAY_S + 60 * n)
            path = the_file(run.cert_dir)
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
