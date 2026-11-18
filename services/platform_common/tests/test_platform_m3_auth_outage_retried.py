"""B3, "Every other connection error is retried" (docs/m3-review-fixes-spec.md,
"Billing"), next to the user decision of 2026-11-17 that made any `authentication
failed` and `pg_hba.conf rejects` exit 2 (test_platform_m3_auth_misconfig.py): an
outage that happens during the authentication exchange is still an outage. The wider
match must not catch it.

Each case starts the billing worker outside dev (Redis reachable through `TlsRedis`)
against an `AuthPg` (platform_m3_auth_kit) that fails the login in one way:

- it asks for a password, reads it, then closes the connection (the server died);
- it asks for a password, then sends `terminating connection due to administrator
  command` (57P01, a fast shutdown) or `the database system is shutting down` (57P03);
- it sends `canceling authentication due to timeout` (57014: an overloaded server timed
  the login out; the message says "authentication" but not "authentication failed");
- it asks for a password, then sends `role "..." is not permitted to log in` (28000,
  the SQLSTATE of the `authentication failed` and `pg_hba.conf` rejects, which the
  decision left retried: matching on SQLSTATE 28000 or class 28 would exit 2 for it).

The worker must warn that Postgres is unreachable and keep running, retrying: it must
not exit 2. The cases run as concurrent workers to keep the waits down."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_m3_auth_kit import AUTH_OUTAGES, USER, auth_servers
from platform_m3_review4_kit import error_response, report, run_workers

pytestmark = [pytest.mark.redis]

WARN_S = 40  # the unreachable-at-start warning, all workers started at once
WATCH_S = 8  # still running: the gate retries every 0.5-5 s, several times in this

# The kit's outages, plus a login the server refuses after the password (as Postgres
# checks rolcanlogin once authentication has passed) with SQLSTATE 28000 but neither
# `authentication failed` nor `pg_hba.conf rejects`: retried, per the decision.
RETRIED: dict[str, tuple[bytes | None, bool]] = {
    **AUTH_OUTAGES,
    "not_permitted_to_log_in": (
        error_response("28000", f'role "{USER}" is not permitted to log in'),
        True,
    ),
}


@pytest.mark.timeout(110)
def test_item3_outages_during_authentication_are_retried(tmp_path: Path):
    """done-when B3 (retried outages, with the wider misconfiguration list): a server
    that dies, shuts down or times the login out during the authentication exchange,
    or refuses a role not permitted to log in (28000). The worker warns and keeps
    running for each (it retries); none exits 2."""
    with auth_servers(RETRIED) as servers:
        urls = {name: s.url() for name, s in servers.items()}
        outcomes = run_workers(urls, tmp_path, warn_s=WARN_S, watch_s=WATCH_S)
        bad = [n for n, o in outcomes.items() if not o.kept_running]
        assert not bad, (
            f"the worker did not warn and keep running for {bad} (exit 2 for "
            f"{[n for n in bad if outcomes[n].returncode == 2]}): an outage during the "
            "authentication exchange is retried; only `authentication failed`, `pg_hba.conf "
            "rejects` and the other misconfigurations exit 2:\n" + report(outcomes, bad)
        )
        asked = {n for n, (_, ask) in RETRIED.items() if ask}
        unanswered = [n for n in sorted(asked) if servers[n].passwords < 1]
        assert not unanswered, (
            f"the worker never sent its password for {unanswered}: the failure did not come "
            "during the authentication exchange\n" + report(outcomes, unanswered)
        )
        once = [n for n, s in servers.items() if s.startups < 2]
        assert not once, (
            f"no retry reached the server for {once} within {WATCH_S}s of the start "
            "warning (spec B3: the gate keeps retrying):\n" + report(outcomes, once)
        )
