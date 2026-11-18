"""B3, "When the deferred check fails, the service exits 2 through its normal shutdown
path" (docs/m3-review-fixes-spec.md, "Billing"), as contracts §3 "Roles" states it: "The
failed check decides the exit code even when something else fails on the way out (an
error during the shutdown, or the gateway's port taken: uvicorn would exit 3); a gateway
whose check has failed by the end of its startup does not bind its port."

The regression these pin: a gateway whose deferred check failed during its startup while
its port was taken exited 3 (uvicorn's startup failure), not 2.

- Gateway, port held (subprocess, needs Redis): the test binds and listens on the
  gateway's port itself for the whole test, so the bind would fail every time. The real
  `python -m loom_gateway` starts with `LOOM_ENV=prod` (every static production problem
  cleared, review3 kit), Postgres unreachable at start (an `AuthPg` that is bound but not
  listening), then Postgres answers the login with `LDAP authentication failed` as soon
  as the start warning is out. The check fails within the app's startup (the auth
  listener's 5 s wait, while the gate retries at 0.5/1/2 s steps). The gateway must exit
  2, print the refusal naming the error, and never try to bind (no bind error logged).
- Gateway `main` (in process, no Postgres/Redis): uvicorn's `Server.startup` is replaced
  by one that lets the gate fail for real (a fresh `RoleGate` deferred to an `AuthPg`
  misconfiguration, patched in for the process's gate) and then fails the way uvicorn's
  does when the bind fails (`sys.exit(3)`), or raises. This is the gate failing after the
  pre-bind check: `main` must still exit 2. Guards: with the gate not failed, the same
  exits are not turned into 2.
- Pool manager (in process): the real `main(["run"])` in dev with the gate deferred to a
  misconfiguration; `_run` stops at the gate and its `finally` runs `PoolManager.stop`,
  which here fails after doing its work. The exit must still be 2. Guards: the plain
  failed gate exits 2; a failing stop with no failed gate is not turned into 2.
- Billing CLI (in process): guard, the real `main(["worker"])` with a failed gate exits 2
  (the old code did too: nothing in billing's run paths can fail on the way out after
  `open_pool` refuses). And the contract's case, injected: a worker run that, after the
  gate failed, ends in another error; `main` must return 2 rather than raise it."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
import uvicorn
from platform_m3_auth_kit import AuthPg, misconfig_error
from platform_m3_review2_kit import PG_WARNING, REPO_ROOT
from platform_m3_review3_kit import GATEWAY, RedisFront, gateway_env, redis_front
from platform_m3_review_proxy import Service

import loom_billing.cli as billing_cli
import loom_gateway.__main__ as gateway_main
import loom_platform.db as platform_db
import loom_poolmgr.cli as poolmgr_cli
from loom_platform.db import RoleCheckFailed, RoleGate
from loom_poolmgr.manager import ManagerLocked, PoolManager

WARN_S = 30  # the unreachable-at-start warning
EXIT_S = 45  # after Postgres answers: the listener's 5 s wait, the shutdown; generous
CASE = "ldap"  # `LDAP authentication failed for user ...`: a misconfiguration, exit 2
REFUSAL = re.compile(r"stopping with LOOM_ENV=", re.IGNORECASE)
BIND_ERROR = re.compile(
    r"address already in use|error while attempting to bind|errno 98", re.IGNORECASE
)
SERVER_EXITED = re.compile(r"server (exited|failed) .*after the failed database role check")


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def rf(tmp_path: Path) -> Iterator[RedisFront]:
    with redis_front(tmp_path / "redis") as front:
        yield front


@pytest.fixture
def held_port() -> Iterator[int]:
    """A port on 127.0.0.1 this test binds and listens on until it ends."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(16)
        yield int(s.getsockname()[1])


@pytest.fixture
def misconfig_pg() -> Iterator[AuthPg]:
    """Postgres answering every login with `LDAP authentication failed`."""
    error, ask, _ = misconfig_error(CASE)
    server = AuthPg(error, ask_password=ask)
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> RoleGate:
    """A fresh gate in place of the process's one (`loom_platform.db.role_gate`, and the
    entrypoints' own references to it), so a failed check does not outlive the test."""
    fresh = RoleGate()
    module: ModuleType
    for module in (platform_db, gateway_main, poolmgr_cli, billing_cli):
        monkeypatch.setattr(module, "role_gate", fresh)
    return fresh


def _pytest_handler(h: logging.Handler) -> bool:
    return type(h).__module__ == "_pytest.logging"


@pytest.fixture
def clean_logging() -> Iterator[None]:
    """Put back what the entrypoints' logging setup changes: the root logger's handlers
    and level (a handler on this test's captured stdout/stderr, closed afterwards), and
    the loggers they retune."""
    root = logging.getLogger()
    level, before = root.level, [h for h in root.handlers if not _pytest_handler(h)]
    named = {
        name: (lg.level, list(lg.handlers), lg.propagate)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "httpcore")
        for lg in [logging.getLogger(name)]
    }
    yield
    added = [h for h in root.handlers if not _pytest_handler(h) and h not in before]
    root.handlers[:] = [h for h in root.handlers if _pytest_handler(h)] + before
    root.setLevel(level)
    for h in added:
        h.close()
    for name, (lvl, handlers, propagate) in named.items():
        lg = logging.getLogger(name)
        lg.setLevel(lvl)
        lg.handlers[:] = handlers
        lg.propagate = propagate


@pytest.fixture
def dev_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clean_logging) -> pytest.MonkeyPatch:
    """No LOOM_* variable from the caller; dev; the committed config files."""
    for var in list(os.environ):
        if var.startswith("LOOM_"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOOM_ENV", "dev")
    monkeypatch.setenv("LOOM_MODELS_YAML", str(REPO_ROOT / "config" / "models.yaml"))
    monkeypatch.setenv("LOOM_SERVING_YAML", str(REPO_ROOT / "config" / "serving.yaml"))
    monkeypatch.setenv("LOOM_POOLMGR_STATE_DIR", str(tmp_path / "poolmgr-state"))
    monkeypatch.setenv("LOOM_BILLING_METRICS_PORT", "0")
    monkeypatch.setenv("LOOM_LOG_LEVEL", "INFO")
    monkeypatch.chdir(REPO_ROOT)
    return monkeypatch


def _defer(gate: RoleGate, server: AuthPg, service: str) -> None:
    """Postgres was unreachable at start: the check is left to the gate, which will meet
    the misconfiguration on its first connection."""
    gate.defer(server.url(), service, "prod")


def _named(out: str) -> bool:
    return bool(misconfig_error(CASE)[2].search(out))


# ------------------------------------------------------------------ gateway, port held


@pytest.mark.redis
@pytest.mark.timeout(110)
def test_item_b3_gateway_port_held_gate_fails_in_startup_exits_2_without_binding(
    rf, held_port, tmp_path
):
    """done-when B3 (exit 2 through the normal shutdown; contracts §3: a gateway whose
    check has failed by the end of its startup does not bind its port): the port is held
    by the test throughout; the deferred check fails during the app's startup; the
    gateway exits 2 (not uvicorn's 3), prints the refusal, and logs no bind error."""
    error, ask, named = misconfig_error(CASE)
    server = AuthPg(error, ask_password=ask, listening=False)
    env = gateway_env(rf, server.url(), port=held_port, spool=tmp_path / "spool.jsonl")
    svc = Service(GATEWAY, env)
    try:
        warned = svc.wait_for(PG_WARNING, WARN_S)
        assert svc.returncode is None and warned, (
            f"the gateway did not start with Postgres unreachable (exit {svc.returncode}):\n"
            f"{svc.output}"
        )
        server.open()  # the gate's next attempt meets the misconfiguration
        code = svc.wait_exit(EXIT_S)
        out = svc.output
        assert server.startups >= 1, f"the gateway never tried Postgres once it answered:\n{out}"
        assert code is not None, (
            f"the gateway kept running {EXIT_S}s after its deferred check met '{CASE}':\n{out}"
        )
        assert code == 2, (
            f"the gateway exited {code}, not 2, after its deferred check failed while its "
            f"port ({held_port}) was taken (3 is uvicorn's startup failure: the bind was "
            f"tried, and its failure decided the exit):\n{out}"
        )
        assert REFUSAL.search(out) and named.search(out), (
            f"the refusal naming the error ({named.pattern}) was not printed:\n{out}"
        )
        bind_lines = [ln for ln in out.splitlines() if BIND_ERROR.search(ln)]
        assert bind_lines == [], (
            "the gateway tried to bind its port after its check had failed in startup "
            f"(contracts §3: it does not bind):\n{bind_lines}\n{out}"
        )
        exited = [ln for ln in out.splitlines() if SERVER_EXITED.search(ln)]
        assert exited == [], (
            f"the server ended some other way after the failed check (expected a straight "
            f"shutdown, no bind):\n{exited}\n{out}"
        )
    finally:
        svc.stop()
        server.close()


# ------------------------------------------------------------------ gateway main


def _gateway_exit(monkeypatch: pytest.MonkeyPatch, startup) -> int | str | None:
    """`python -m loom_gateway`'s exit code, in process, with uvicorn's `Server.startup`
    replaced by `startup` (the metrics listener is not started)."""
    monkeypatch.setattr(gateway_main, "start_http_server", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn.Server, "startup", startup)
    try:
        gateway_main.main()
    except SystemExit as e:
        return e.code
    return 0


async def _fail_gate(gate: RoleGate) -> None:
    with pytest.raises(RoleCheckFailed):
        await gate.wait()
    assert gate.failed


def test_item_b3_gateway_main_bind_fails_after_gate_failed_exits_2(
    dev_env, gate, misconfig_pg, capsys
):
    """done-when B3 (contracts §3: the failed check decides the exit code; the port taken:
    uvicorn would exit 3): the gate fails after the pre-bind check, then uvicorn's bind
    fails (`sys.exit(3)`); `main` exits 2."""
    _defer(gate, misconfig_pg, "loom-gateway")

    async def gate_fails_then_bind_fails(self, sockets=None):
        await _fail_gate(gate)
        sys.exit(3)  # uvicorn's STARTUP_FAILURE, as for a bind OSError

    code = _gateway_exit(dev_env, gate_fails_then_bind_fails)
    out = capsys.readouterr()
    assert code == 2, (
        f"the gateway exited {code!r}, not 2: uvicorn's startup failure (3) after the "
        f"failed check decided the exit code\n{out.out}{out.err}"
    )
    assert _named(out.err), f"the refusal was not printed:\n{out.err}"


def test_item_b3_gateway_main_server_error_after_gate_failed_exits_2(
    dev_env, gate, misconfig_pg, capsys
):
    """done-when B3 (contracts §3: even when something else fails on the way out): the
    server raises after the gate failed; `main` exits 2 rather than raise it."""
    _defer(gate, misconfig_pg, "loom-gateway")

    async def gate_fails_then_server_errors(self, sockets=None):
        await _fail_gate(gate)
        raise RuntimeError("server failed after the check")

    try:
        code = _gateway_exit(dev_env, gate_fails_then_server_errors)
    except RuntimeError as e:
        pytest.fail(
            f"the server's error ({e}) escaped after the failed check instead of exit 2:\n"
            f"{capsys.readouterr().err}"
        )
    out = capsys.readouterr()
    assert code == 2, f"the gateway exited {code!r}, not 2\n{out.out}{out.err}"
    assert _named(out.err), f"the refusal was not printed:\n{out.err}"


def test_item_b3_guard_gateway_main_bind_failure_without_failed_gate_exits_3(dev_env, gate):
    """Guard (passes before and after the fix): with no failed check, a bind failure
    keeps uvicorn's exit 3 and a server error propagates; 2 is not every exit."""

    async def bind_fails(self, sockets=None):
        sys.exit(3)

    assert _gateway_exit(dev_env, bind_fails) == 3

    async def server_errors(self, sockets=None):
        raise RuntimeError("server failed")

    with pytest.raises(RuntimeError, match="server failed"):
        _gateway_exit(dev_env, server_errors)
    assert not gate.failed


# ------------------------------------------------------------------ pool manager


def _poolmgr_run_exit() -> int | str | None:
    with pytest.raises(SystemExit) as exit_:
        poolmgr_cli.main(["run"])
    return exit_.value.code


@pytest.fixture
def stop_fails(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """`PoolManager.stop` does its work, then raises (an error on the way out)."""
    calls: list[int] = []
    real_stop = PoolManager.stop

    async def stop_then_fail(self: PoolManager) -> None:
        calls.append(1)
        await real_stop(self)
        raise RuntimeError("closing the manager failed")

    monkeypatch.setattr(PoolManager, "stop", stop_then_fail)
    return calls


def test_item_b3_poolmgr_shutdown_error_after_gate_failed_exits_2(
    dev_env, gate, misconfig_pg, stop_fails, capsys
):
    """done-when B3 (contracts §3: an error during the shutdown): `loom-poolmgr run`'s
    deferred check fails, its shutdown (`PoolManager.stop` in `_run`'s `finally`) then
    fails; the exit is still 2."""
    _defer(gate, misconfig_pg, "loom-poolmgr")
    try:
        code = _poolmgr_run_exit()
    except RuntimeError as e:
        pytest.fail(
            f"the shutdown's error ({e}) escaped after the failed check instead of the "
            f"exit 2:\n{capsys.readouterr().err}"
        )
    err = capsys.readouterr().err
    assert stop_fails, "the manager's shutdown did not run"
    assert gate.failed
    assert code == 2, f"loom-poolmgr run exited {code!r}, not 2:\n{err}"
    assert _named(err), f"the refusal was not printed:\n{err}"


def test_item_b3_guard_poolmgr_failed_gate_exits_2(dev_env, gate, misconfig_pg, capsys):
    """Guard (passes before and after the fix): a failed check with a clean shutdown
    exits 2."""
    _defer(gate, misconfig_pg, "loom-poolmgr")
    assert _poolmgr_run_exit() == 2
    assert gate.failed and _named(capsys.readouterr().err)


def test_item_b3_guard_poolmgr_shutdown_error_without_failed_gate_propagates(
    dev_env, gate, stop_fails, monkeypatch
):
    """Guard (passes before and after the fix): with no failed check (none deferred), a
    shutdown error is not turned into exit 2."""

    async def locked(self: PoolManager) -> None:
        raise ManagerLocked("another manager holds the lock")

    monkeypatch.setattr(PoolManager, "start", locked)
    with pytest.raises(RuntimeError, match="closing the manager failed"):
        poolmgr_cli.main(["run"])
    assert stop_fails and not gate.failed


# ------------------------------------------------------------------ billing


def test_item_b3_guard_billing_worker_failed_gate_exits_2(dev_env, gate, misconfig_pg, capsys):
    """Guard (passes before and after the fix): `loom-billing worker` whose deferred
    check fails exits 2, through the real run path (`open_pool` refuses first)."""
    _defer(gate, misconfig_pg, "loom-billing worker")
    assert billing_cli.main(["worker"]) == 2
    assert gate.failed and _named(capsys.readouterr().err)


def test_item_b3_billing_error_on_the_way_out_after_gate_failed_exits_2(
    dev_env, gate, misconfig_pg, capsys, monkeypatch
):
    """done-when B3 (contracts §3: even when something else fails on the way out),
    injected: billing's own run paths cannot fail after `open_pool` refuses, so a worker
    run is replaced by one whose unwinding, after the failed check, raises another error.
    `main` returns 2 rather than raise it."""
    _defer(gate, misconfig_pg, "loom-billing worker")

    async def worker_unwind_fails(settings) -> None:
        try:
            await platform_db.open_pool(settings.database_url)
        except RoleCheckFailed as e:
            raise RuntimeError("closing the worker failed") from e

    monkeypatch.setattr(billing_cli, "run_worker", worker_unwind_fails)
    try:
        code = billing_cli.main(["worker"])
    except RuntimeError as e:
        pytest.fail(
            f"the error on the way out ({e}) escaped after the failed check instead of "
            f"exit 2:\n{capsys.readouterr().err}"
        )
    assert gate.failed
    assert code == 2, f"loom-billing worker returned {code!r}, not 2"
    assert _named(capsys.readouterr().err)


def test_item_b3_guard_billing_error_without_failed_gate_propagates(dev_env, gate, monkeypatch):
    """Guard (passes before and after the fix): with no failed check, a worker error is
    not turned into exit 2."""

    async def worker_fails(settings) -> None:
        raise RuntimeError("worker failed")

    monkeypatch.setattr(billing_cli, "run_worker", worker_fails)
    with pytest.raises(RuntimeError, match="worker failed"):
        billing_cli.main(["worker"])
    assert not gate.failed


def test_kit_gate_deferred_to_misconfig_fails_on_first_connection(gate, misconfig_pg, capsys):
    """The setup the in-process tests rest on: the fresh gate, deferred, fails for real
    on the misconfiguration (printed), and keeps failing."""
    _defer(gate, misconfig_pg, "loom-test")
    asyncio.run(_fail_gate(gate))
    with pytest.raises(RoleCheckFailed):
        asyncio.run(gate.wait())
    assert misconfig_pg.startups >= 1 and _named(capsys.readouterr().err)
