"""B3, "Only an outage counts as unreachable" (docs/m3-review-fixes-spec.md, "Billing"),
with the user decision of 2026-11-17: the shared connection-error classifier itself,
`loom_platform.settings.connection_outage` / `connection_problem`, without a server.

Every service decides with it whether a failed Postgres connection is retried (an
outage) or exits 2 (a misconfiguration), at start and in the deferred gate. The service
tests (test_platform_m3_auth_*.py) reach it through the gateway, the billing worker and
the pool manager at start; this file also covers the callers no service test reaches
(the pool manager's deferred gate), by calling it directly on errors shaped as psycopg
raises them (libpq's text, one line per address tried).

Misconfigurations (exit 2): an `authentication failed` of any method (password, peer,
ident, certificate, LDAP, the method-less form), `pg_hba.conf rejects`, `no pg_hba.conf
entry`, a role or database that `does not exist`. Outages (retried): refused, timed out,
a host name that does not resolve, network unreachable, a missing socket, any "the
database system is ..." state, too many clients, a connection closed unexpectedly,
`canceling authentication due to timeout`, and a role `not permitted to log in`."""

from __future__ import annotations

import psycopg
import pytest

from loom_platform.settings import connection_outage, connection_problem

USER = "loom_t_auth_svc"
DB = "loom_platform"
AT = 'connection to server at "127.0.0.1", port 5432 failed: '
SOCKET = 'connection to server on socket "/var/run/postgresql/.s.PGSQL.5432" failed: '

# case -> (libpq's message, a fragment of it the exit-2 problem must name)
MISCONFIGURATION: dict[str, tuple[str, str]] = {
    "password": (
        AT + f'FATAL:  password authentication failed for user "{USER}"',
        "password authentication failed",
    ),
    "peer": (
        SOCKET + f'FATAL:  Peer authentication failed for user "{USER}"',
        "Peer authentication failed",
    ),
    "ident": (
        AT + f'FATAL:  Ident authentication failed for user "{USER}"',
        "Ident authentication failed",
    ),
    "cert": (
        AT + f'FATAL:  certificate authentication failed for user "{USER}"',
        "certificate authentication failed",
    ),
    "ldap": (
        AT + f'FATAL:  LDAP authentication failed for user "{USER}"',
        "LDAP authentication failed",
    ),
    "method_less": (
        AT + f'FATAL:  authentication failed for user "{USER}": host rejected',
        "authentication failed",
    ),
    "pg_hba_rejects": (
        AT + f'FATAL:  pg_hba.conf rejects connection for host "127.0.0.1", user "{USER}", '
        f'database "{DB}", no encryption',
        "pg_hba.conf rejects connection",
    ),
    "no_pg_hba_entry": (
        AT + f'FATAL:  no pg_hba.conf entry for host "10.1.2.3", user "{USER}", '
        f'database "{DB}", SSL encryption',
        "no pg_hba.conf entry",
    ),
    "role_missing": (AT + f'FATAL:  role "{USER}" does not exist', "does not exist"),
    "database_missing": (AT + f'FATAL:  database "{DB}" does not exist', "does not exist"),
}

OUTAGE: dict[str, str] = {
    "refused": AT + "Connection refused\n\tIs the server running on that host and accepting "
    "TCP/IP connections?",
    "timeout": 'connection to server at "10.9.9.9", port 5432 failed: timeout expired',
    "dns": 'could not translate host name "db.internal" to address: Name or service not known',
    "dns_temporary": (
        'could not translate host name "db.internal" to address: Temporary failure in name '
        "resolution"
    ),
    "network_unreachable": 'connection to server at "10.9.9.9", port 5432 failed: Network is '
    "unreachable\n\tIs the server running on that host and accepting TCP/IP connections?",
    "missing_socket": SOCKET + "No such file or directory\n\tIs the server running locally and "
    "accepting connections on that socket?",
    "starting_up": AT + "FATAL:  the database system is starting up",
    "shutting_down": AT + "FATAL:  the database system is shutting down",
    "recovery": AT + "FATAL:  the database system is in recovery mode",
    "not_accepting": AT + "FATAL:  the database system is not yet accepting connections\n"
    "DETAIL:  Consistent recovery state has not been yet reached.",
    "too_many_clients": AT + "FATAL:  sorry, too many clients already",
    "closed_unexpectedly": AT + "server closed the connection unexpectedly\n\tThis probably "
    "means the server terminated abnormally\n\tbefore or while processing the request.",
    "authentication_timeout": AT + "FATAL:  canceling authentication due to timeout",
    "not_permitted_to_log_in": AT + f'FATAL:  role "{USER}" is not permitted to log in',
    "admin_shutdown": AT + "FATAL:  terminating connection due to administrator command",
}


def error(message: str) -> psycopg.OperationalError:
    """A connection error as `psycopg.connect` raises it."""
    return psycopg.OperationalError(message)


@pytest.mark.parametrize("case", sorted(MISCONFIGURATION))
def test_item_b3_misconfiguration_is_no_outage(case):
    """done-when B3 (with the decision): each misconfiguration exits 2, not retried."""
    message, _ = MISCONFIGURATION[case]
    assert connection_outage(error(message)) is False, message


@pytest.mark.parametrize("case", sorted(MISCONFIGURATION))
def test_item_b3_misconfiguration_problem_names_it(case):
    """done-when B3: the exit-2 problem names the misconfiguration as libpq said it."""
    message, named = MISCONFIGURATION[case]
    problem = connection_problem(error(message))
    assert named.lower() in problem.lower(), problem


@pytest.mark.parametrize("case", sorted(MISCONFIGURATION))
def test_item_b3_misconfiguration_on_a_later_address(case):
    """done-when B3: libpq tries every address and lists each failure. A
    misconfiguration on a later line exits 2 and is the one the problem names, though
    the first address only refused."""
    message, named = MISCONFIGURATION[case]
    first = 'connection to server at "::1", port 5432 failed: Connection refused\n\tIs the '
    first += "server running on that host and accepting TCP/IP connections?\n"
    e = error(first + message)
    assert connection_outage(e) is False, e
    problem = connection_problem(e)
    assert named.lower() in problem.lower(), problem


@pytest.mark.parametrize("case", sorted(OUTAGE))
def test_item_b3_outage_is_retried(case):
    """done-when B3: every other connection error is an outage, retried, including the
    two that say "authentication" or carry SQLSTATE 28000 without being one of the
    misconfigurations (`canceling authentication due to timeout`, `not permitted to log
    in`)."""
    assert connection_outage(error(OUTAGE[case])) is True, OUTAGE[case]


def test_item_b3_unforeseen_error_is_retried():
    """done-when B3: an error no one foresaw (and an empty one) is an outage."""
    assert connection_outage(error("something odd happened")) is True
    assert connection_outage(error("")) is True
