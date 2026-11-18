"""B3, "Only an outage counts as unreachable" (docs/m3-review-fixes-spec.md, "Billing"),
extended by the user decision of 2026-11-17: besides `password authentication failed`,
`does not exist` and `no pg_hba.conf entry`, a Postgres connection error whose message
says `authentication failed` (any method: peer, ident, certificate, LDAP, ...) or
`pg_hba.conf rejects` is a misconfiguration. Outside dev it exits 2, naming the error,
at start and in the deferred gate, for every Python service. It is not an outage: the
service must not report Postgres as unreachable and keep retrying.

Postgres is an `AuthPg` (platform_m3_auth_kit) answering the login with Postgres's own
wording for that failure, so the test does not depend on the test server's pg_hba.

- At start (Postgres answers from the start): the gateway and the billing worker, with
  every static production problem cleared (Redis reachable through `TlsRedis`), exit 2
  naming the error. The pool manager cannot have its static problems cleared outside
  dev (RunPod only), so it exits 2 regardless; its exit-2 message must list the
  connection error as a problem, not warn that Postgres is unreachable.
- In the gate: the gateway and the billing worker start with Postgres unreachable
  (warn, keep running), then Postgres answers with the failure; they exit 2 naming it.
  (The pool manager's gate cannot be reached outside dev, for the same reason.)
- Guard: `password authentication failed`, on the list before, behaves the same."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from platform_m3_auth_kit import LISTED, MISCONFIG, POOLMGR, AuthPg, misconfig_error, prod_env
from platform_m3_review2_kit import PG_WARNING
from platform_m3_review3_kit import COMMANDS, RedisFront, pg_unreachable_lines, redis_front
from platform_m3_review_proxy import Service

pytestmark = [pytest.mark.redis]

EXIT_S = 45  # a refusal takes seconds; generous for a loaded CI runner
WARN_S = 30  # the unreachable-at-start warning (gate tests); 75 s in all with EXIT_S
ALL_COMMANDS = {**COMMANDS, "poolmgr": POOLMGR}


@pytest.fixture
def rf(tmp_path: Path) -> Iterator[RedisFront]:
    with redis_front(tmp_path / "redis") as front:
        yield front


def _naming_lines(output: str, named: re.Pattern[str]) -> list[str]:
    """Lines naming the error that are not 'Postgres unreachable' lines."""
    unreachable = set(pg_unreachable_lines(output))
    return [line for line in output.splitlines() if named.search(line) and line not in unreachable]


def _assert_refused(kind: str, code: int | None, out: str, named: re.Pattern[str], what: str):
    assert code is not None, (
        f"the {kind} kept running {EXIT_S}s after Postgres answered '{what}' (user "
        "decision 2026-11-17: `authentication failed` and `pg_hba.conf rejects` are "
        f"misconfigurations, they exit 2 and are not retried):\n{out}"
    )
    assert code == 2, f"the {kind} exited {code}, not 2:\n{out}"
    assert _naming_lines(out, named), (
        f"the exit-2 message does not name the error ({named.pattern}) as a problem:\n{out}"
    )


def _refuses_at_start(kind: str, case: str, rf: RedisFront, tmp_path: Path) -> None:
    error, ask, named = misconfig_error(case)
    server = AuthPg(error, ask_password=ask)
    svc = Service(ALL_COMMANDS[kind], prod_env(kind, rf, server.url(), tmp_path))
    try:
        code = svc.wait_exit(EXIT_S)
        out = svc.output
        assert server.startups >= 1, f"the {kind} never reached Postgres:\n{out}"
        _assert_refused(kind, code, out, named, case)
        unreachable = pg_unreachable_lines(out)
        assert unreachable == [], (
            f"the {kind} reported the misconfiguration ({case}) as Postgres unreachable:\n"
            f"{unreachable}"
        )
    finally:
        svc.stop()
        server.close()


@pytest.mark.parametrize("case", sorted(MISCONFIG))
@pytest.mark.parametrize("kind", ["gateway", "worker", "poolmgr"])
def test_item2_auth_failure_at_start_exits_2(rf, tmp_path, kind, case):
    """done-when B3 (misconfiguration at start, user decision 2026-11-17): Postgres
    answers the login with `<method> authentication failed` / `pg_hba.conf rejects`;
    the service exits 2 naming it, without calling Postgres unreachable."""
    _refuses_at_start(kind, case, rf, tmp_path)


@pytest.mark.parametrize("case", sorted(LISTED))
@pytest.mark.parametrize("kind", ["gateway", "worker", "poolmgr"])
def test_item2_guard_password_failure_at_start_exits_2(rf, tmp_path, kind, case):
    """done-when B3 (guard): `password authentication failed`, on the list before the
    decision, exits 2 the same way (so the checks above can pass)."""
    _refuses_at_start(kind, case, rf, tmp_path)


def _refuses_in_gate(kind: str, case: str, rf: RedisFront, tmp_path: Path) -> None:
    error, ask, named = misconfig_error(case)
    server = AuthPg(error, ask_password=ask, listening=False)
    svc = Service(ALL_COMMANDS[kind], prod_env(kind, rf, server.url(), tmp_path))
    try:
        warned = svc.wait_for(PG_WARNING, WARN_S)
        assert svc.returncode is None, (
            f"the {kind} exited ({svc.returncode}) while Postgres was unreachable at start "
            f"(spec: start with a warning):\n{svc.output}"
        )
        assert warned, f"no warning about Postgres being unreachable at start:\n{svc.output}"
        assert server.startups == 0, "reached Postgres although it refused connections"

        server.open()
        code = svc.wait_exit(EXIT_S)
        out = svc.output
        assert server.startups >= 1, f"the {kind} never tried Postgres once it answered:\n{out}"
        _assert_refused(kind, code, out, named, case)
    finally:
        svc.stop()
        server.close()


@pytest.mark.parametrize("case", sorted(MISCONFIG))
@pytest.mark.parametrize("kind", ["gateway", "worker"])
def test_item2_auth_failure_in_gate_exits_2(rf, tmp_path, kind, case):
    """done-when B3 (misconfiguration in the deferred gate, user decision 2026-11-17):
    started with Postgres unreachable, then Postgres answers `<method> authentication
    failed` / `pg_hba.conf rejects`; the service exits 2 naming it rather than
    retrying it as an outage."""
    _refuses_in_gate(kind, case, rf, tmp_path)


@pytest.mark.parametrize("case", sorted(LISTED))
@pytest.mark.parametrize("kind", ["gateway", "worker"])
def test_item2_guard_password_failure_in_gate_exits_2(rf, tmp_path, kind, case):
    """done-when B3 (guard): `password authentication failed` in the gate exits 2."""
    _refuses_in_gate(kind, case, rf, tmp_path)
