"""B3 after the third review (docs/m3-review-fixes-spec.md, "Billing", B3, "Only an outage
counts as unreachable"): exit 2 only on the explicit misconfiguration messages
(`password authentication failed`, `does not exist`, `no pg_hba.conf entry`); "every
other connection error is retried" with a warning.

Each case starts the billing worker outside dev (Redis reachable through `TlsRedis`)
with a LOOM_DATABASE_URL whose connections fail in one way. The worker must warn that
Postgres is unreachable and keep running (the role check deferred to the gate, which
keeps retrying): it must not exit 2. The cases of one test run as concurrent workers to
keep the waits down.

- Client-side failures: a host name that does not resolve (`.invalid`), network
  unreachable (an IPv6 documentation address, on a host without an IPv6 route), a Unix
  socket path that does not exist.
- Server answers (`ForgedPg`, a server that answers the startup like Postgres in that
  state): `the database system is in recovery mode` / `... is not accepting
  connections`, a connection closed without an answer, and an error not on either list;
  and, as a guard, the states the old retry list already named (`starting up`, `too many
  clients`)."""

from __future__ import annotations

import errno
import socket
import uuid
from pathlib import Path

import pytest
from platform_m3_review4_kit import error_response, forged, report, run_workers

pytestmark = [pytest.mark.redis]

WARN_S = 40  # the unreachable-at-start warning, all workers started at once
WATCH_S = 8  # still running: the gate retries every 0.5-5 s, several times in this
UNROUTED_V6 = "2001:db8::1"  # RFC 3849 documentation prefix


def _network_unreachable() -> bool:
    """Whether connecting to UNROUTED_V6 fails at once with 'network unreachable' or
    'no route to host' (a host without an IPv6 route), rather than timing out."""
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    s.settimeout(2)
    try:
        s.connect((UNROUTED_V6, 5432))
    except OSError as e:
        return e.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH)
    finally:
        s.close()
    return False


def _assert_all_kept_running(outcomes, what: str) -> None:
    bad = [n for n, o in outcomes.items() if not o.kept_running]
    exited_2 = [n for n in bad if outcomes[n].returncode == 2]
    assert not bad, (
        f"{what}: the worker did not warn and keep running for {bad} (exit 2 for "
        f"{exited_2}; spec B3: only `password authentication failed`, `does not exist` and "
        "`no pg_hba.conf entry` exit 2, every other connection error is retried):\n"
        + report(outcomes, bad)
    )


@pytest.mark.timeout(110)
def test_item3_client_side_outages_are_retried(tmp_path: Path):
    """done-when B3 (retried outages): a host name that does not resolve, network
    unreachable (when the host has no IPv6 route) and a missing Unix socket. The worker
    warns and keeps running for each; none exits 2."""
    tag = uuid.uuid4().hex[:8]
    urls = {
        "unresolvable_host": f"postgresql://loom_t_r4:pw@loom-r4-{tag}.invalid:5432/loom_review4",
        "missing_socket": (
            f"postgresql://loom_t_r4:pw@/loom_review4?host=/nonexistent/loom-r4-{tag}"
        ),
    }
    if _network_unreachable():
        urls["network_unreachable"] = f"postgresql://loom_t_r4:pw@[{UNROUTED_V6}]:5432/loom_review4"
    outcomes = run_workers(urls, tmp_path, warn_s=WARN_S, watch_s=WATCH_S)
    _assert_all_kept_running(outcomes, "client-side connection failures")


# Postgres's own SQLSTATEs and wording for these states.
UNLISTED = {
    "in_recovery": error_response("57P03", "the database system is in recovery mode"),
    "not_accepting": error_response("57P03", "the database system is not accepting connections"),
    "closed_without_answer": None,
    "unknown_error": error_response("XX000", "loom review4: an unexpected server condition"),
}
LISTED = {
    "starting_up": error_response("57P03", "the database system is starting up"),
    "too_many_clients": error_response("53300", "sorry, too many clients already"),
}


def _run_forged(cases: dict[str, bytes | None], tmp_path: Path, what: str) -> None:
    with forged(cases) as servers:
        urls = {name: s.url() for name, s in servers.items()}
        outcomes = run_workers(urls, tmp_path, warn_s=WARN_S, watch_s=WATCH_S)
        _assert_all_kept_running(outcomes, what)
        once = [n for n, s in servers.items() if s.startups < 2]
        assert not once, (
            f"{what}: no retry reached the server for {once} within {WATCH_S}s of the "
            "start warning (spec B3: the gate keeps retrying):\n" + report(outcomes, once)
        )


@pytest.mark.timeout(110)
def test_item3_server_states_not_on_the_old_list_are_retried(tmp_path: Path):
    """done-when B3 (retried outages): a server answering `the database system is in
    recovery mode` or `... is not accepting connections`, closing the connection without
    an answer, or sending an error on neither list. The worker warns and keeps running
    (it retries); none exits 2."""
    _run_forged(UNLISTED, tmp_path, "server answers not on the old retry list")


@pytest.mark.timeout(110)
def test_item3_guard_listed_server_states_are_retried(tmp_path: Path):
    """done-when B3 (retried outages), guard: `the database system is starting up` and
    `too many clients`, retried before this change too, still are."""
    _run_forged(LISTED, tmp_path, "server answers on the old retry list")
