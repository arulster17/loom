"""Shared kit for the M3 Stripe-area tests (`test_billing_m3s_*.py`, docs/phase1-m3-spec.md
§8, §13 A16-A34). Importable from any test file in `services/billing/tests` (and from
elsewhere with `PYTHONPATH=services/billing/tests`). It offers:

- `ep(name, module)`: a §8.11 entry point, imported when the test runs, so a missing one
  fails that test (ImportError/AttributeError naming it) instead of the module's
  collection.
- `m3s_settings(**kw)`: `BillingSettings` in stub mode with the Stripe timeouts scaled
  down for tests (§8.3 "Tests scale the three settings down").
- Event builders with random ids (`checkout_event`, `charge_event`, `dispute_event`) and
  `deliver` (signed POST to `/stripe/webhook`), `billing_client` (the API app in-process).
- Row helpers: `event_row` (every `stripe_events` column M3 defines), `purchases`,
  `refunds`, `ledger_count`, `store_raw` (insert an event row directly, "as a racing
  delivery stored it"), `make_org`, `make_admin`.
- Metrics: `sample(name, **labels)` reads the process registry.
- Isolation (§16): `m3s_template` (session: a database migrated once, dropped at the end)
  and `scratch_url` (function: a clone of it, dropped at teardown, also on failure),
  `scratch_pool` (a pool on the clone). A test module binds them with
  `m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()`.
- Concurrency without sleeps: `Holder` (a transaction on its own connection, held until
  released) and `wait_blocked_by` (polls `pg_blocking_pids` until a backend waits on the
  holder).
- Faults at the connection level (§8.11), never by patching code:
  * `install_faults(url)` adds two triggers to a scratch database: `poison(...)` makes
    every write of an event row that clears `last_error` raise (a handler bug on that
    event: the second transaction, which sets `last_error`, still stores it), or every
    write at all (`hard=True`: the error store fails too); `gate(...)` adds a deferred
    (commit-time) check on a row of `stripe_events`, `ledger_transactions`,
    `stripe_payments` or `admin_audit_log` that either fails the COMMIT with a chosen
    SQLSTATE or waits on an advisory lock the test holds (a transaction held at its
    COMMIT).
  * `FaultProxy`: a TCP proxy in front of Postgres that can refuse connections, drop a
    connection at its next COMMIT (before or after the COMMIT reaches the server), drop
    one inside a savepoint, and records every statement.
  * `Blackhole`: a TCP server that accepts and never answers (a frozen Redis).
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import importlib
import json
import re
import secrets
import socket
import struct
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import psycopg
import pytest
from billing_fixtures import STUB, STUB_WEBHOOK_SECRET
from prometheus_client import REGISTRY
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from loom_billing.settings import BillingSettings
from loom_billing.stripe_sig import sign
from loom_platform.db import open_pool
from loom_platform.ledger import STRIPE_CLEARING
from loom_platform.migrate import upgrade
from loom_platform.testing import create_org, create_user

UTC = dt.UTC
TOKEN = "internal-test-token"

# ------------------------------------------------------------------ entry points


def ep(name: str, module: str = "loom_billing.webhooks") -> Any:
    """The §8.11 entry point `module.name`, imported now (at test run time)."""
    return getattr(importlib.import_module(module), name)


# ------------------------------------------------------------------ settings

# Scaled-down Stripe timeouts (§8.3: 3000 ms / 5000 ms / 5 s by default).
LOCK_TIMEOUT_MS = 2000
STATEMENT_TIMEOUT_MS = 4000
DB_TIMEOUT_S = 1


def m3s_settings(**kw: Any) -> BillingSettings:
    base: dict[str, Any] = {
        **STUB,
        "billing_internal_token": TOKEN,
        "stripe_lock_timeout_ms": LOCK_TIMEOUT_MS,
        "stripe_statement_timeout_ms": STATEMENT_TIMEOUT_MS,
        "stripe_webhook_db_timeout_s": DB_TIMEOUT_S,
    }
    base.update(kw)
    return BillingSettings(**base)


def now() -> dt.datetime:
    return dt.datetime.now(UTC)


# ------------------------------------------------------------------ events


def rid(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(10)}"


def _event(typ: str, obj: dict[str, Any], evt_id: str | None = None) -> dict[str, Any]:
    return {
        "id": evt_id or rid("evt_test"),
        "object": "event",
        "type": typ,
        "livemode": False,
        "created": int(time.time()),
        "data": {"object": obj},
    }


def checkout_event(
    org: uuid.UUID | str,
    cents: int = 2000,
    *,
    typ: str = "checkout.session.completed",
    paid: bool = True,
    session_id: str | None = None,
    payment_intent: str | None = None,
    evt_id: str | None = None,
    **over: Any,
) -> dict[str, Any]:
    session: dict[str, Any] = {
        "id": session_id or rid("cs_test"),
        "object": "checkout.session",
        "amount_total": cents,
        "currency": "usd",
        "client_reference_id": str(org),
        "metadata": {
            "org_id": str(org),
            "credit_micros": str(cents * 10_000),
            "purpose": "loom_credits",
        },
        "mode": "payment",
        "payment_intent": payment_intent,
        "payment_status": "paid" if paid else "unpaid",
        "status": "complete",
        "livemode": False,
    }
    session.update(over)
    return _event(typ, session, evt_id)


def charge_event(
    pi: str | None, amount_refunded: int, cents: int = 2000, *, evt_id: str | None = None
) -> dict[str, Any]:
    charge = {
        "id": rid("ch_test"),
        "object": "charge",
        "amount": cents,
        "currency": "usd",
        "amount_refunded": amount_refunded,
        "payment_intent": pi,
        "livemode": False,
    }
    return _event("charge.refunded", charge, evt_id)


def dispute_event(
    pi: str | None, cents: int, *, dispute_id: str | None = None, evt_id: str | None = None
) -> dict[str, Any]:
    dispute = {
        "id": dispute_id or rid("dp_test"),
        "object": "dispute",
        "amount": cents,
        "currency": "usd",
        "payment_intent": pi,
        "charge": rid("ch_test"),
    }
    return _event("charge.dispute.created", dispute, evt_id)


def body_of(event: dict[str, Any]) -> bytes:
    return json.dumps(event).encode()


async def deliver(
    client: httpx.AsyncClient,
    event: dict[str, Any] | None = None,
    *,
    body: bytes | None = None,
    secret: str = STUB_WEBHOOK_SECRET,
    timestamp: int | None = None,
    header: str | None = None,
) -> httpx.Response:
    """POST a signed event (or a raw `body`) to `/stripe/webhook`. Each call signs afresh
    (a new `t`), as Stripe's retries and resends do."""
    raw = body if body is not None else body_of(event or {})
    sig = header if header is not None else sign(raw, secret, timestamp)
    return await client.post(
        "/stripe/webhook",
        content=raw,
        headers={"Content-Type": "application/json", "Stripe-Signature": sig},
    )


@contextlib.asynccontextmanager
async def billing_client(
    settings: BillingSettings, pool: Any, redis: Any, **kw: Any
) -> AsyncIterator[httpx.AsyncClient]:
    from loom_billing.api import create_app

    app = create_app(settings, pool=pool, redis=redis, **kw)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://billing"
    ) as c:
        yield c


def error_code(r: httpx.Response) -> str | None:
    try:
        return r.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return None


# ------------------------------------------------------------------ rows

EVENT_COLUMNS = (
    "id",
    "type",
    "received_at",
    "processed_at",
    "result",
    "last_error",
    "last_error_at",
    "error_count",
    "stale_reported_at",
)


async def event_row(pool: AsyncConnectionPool, evt_id: str) -> dict[str, Any] | None:
    """The event's row with every M3 column, or None when nothing is stored."""
    q = sql.SQL("SELECT {} FROM stripe_events WHERE id = %s").format(
        sql.SQL(", ").join(map(sql.Identifier, EVENT_COLUMNS))
    )
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(q, (evt_id,))
        row = await cur.fetchone()
    return None if row is None else dict(zip(EVENT_COLUMNS, row, strict=True))


async def purchases(pool: AsyncConnectionPool, org: uuid.UUID) -> list[int]:
    """Credit amounts (micros) of the org's `purchase` transactions."""
    return await _org_amounts(pool, org, "purchase")


async def refunds(pool: AsyncConnectionPool, org: uuid.UUID) -> list[int]:
    """Clawback amounts (negative micros on the org's account), oldest first."""
    return await _org_amounts(pool, org, "refund")


async def _org_amounts(pool: AsyncConnectionPool, org: uuid.UUID, kind: str) -> list[int]:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT e.amount_micros FROM ledger_transactions t "
            "JOIN ledger_entries e ON e.transaction_id = t.id "
            "WHERE t.org_id = %s AND t.kind = %s AND e.account_id <> %s "
            "ORDER BY t.created_at, t.id",
            (org, kind, STRIPE_CLEARING),
        )
        return [int(r[0]) for r in await cur.fetchall()]


async def ledger_count(pool: AsyncConnectionPool, evt_id: str) -> int:
    """Ledger transactions keyed by this event (`stripe:<event id>`)."""
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT count(*) FROM ledger_transactions WHERE idempotency_key = %s",
            (f"stripe:{evt_id}",),
        )
        row = await cur.fetchone()
    assert row is not None
    return int(row[0])


async def store_raw(
    pool: AsyncConnectionPool,
    event: dict[str, Any],
    *,
    received_at: dt.datetime | None = None,
    **cols: Any,
) -> None:
    """Insert an unprocessed event row directly (as a delivery that raced its purchase
    stored it); `cols` sets other M3 columns (e.g. `last_error`)."""
    names = ["id", "type", "payload"]
    values: list[Any] = [event["id"], event["type"], json.dumps(event)]
    if received_at is not None:
        names.append("received_at")
        values.append(received_at)
    for k, v in cols.items():
        names.append(k)
        values.append(v)
    q = sql.SQL("INSERT INTO stripe_events ({}) VALUES ({})").format(
        sql.SQL(", ").join(map(sql.Identifier, names)),
        sql.SQL(", ").join(sql.Placeholder() * len(names)),
    )
    async with pool.connection() as conn:
        await conn.execute(q, values)


async def make_org(pool: AsyncConnectionPool, status: str = "active") -> uuid.UUID:
    async with pool.connection() as conn, conn.transaction():
        return await create_org(conn, status=status)


async def make_admin(pool: AsyncConnectionPool, role: str = "admin") -> tuple[uuid.UUID, str]:
    email = f"{secrets.token_hex(6)}@example.test"
    async with pool.connection() as conn, conn.transaction():
        uid = await create_user(conn, role=role, email=email)
    return uid, email


async def scalar(pool: AsyncConnectionPool, query: str, *params: Any) -> Any:
    async with pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(query, params)  # type: ignore[arg-type]
        row = await cur.fetchone()
    return None if row is None else row[0]


# ------------------------------------------------------------------ metrics


def sample(name: str, **labels: str) -> float:
    """A sample of the process's Prometheus registry (0 when absent)."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def gauge(name: str, **labels: str) -> float | None:
    """A gauge's current value, or None when the series does not exist."""
    return REGISTRY.get_sample_value(name, labels)


WEBHOOKS = "loom_billing_stripe_webhooks_total"
WEBHOOK_ERRORS = "loom_billing_stripe_webhook_errors_total"  # S2c {status}
HANDLER_FAILURES = "loom_billing_stripe_handler_failures_total"  # S4 {path}
STALE_TOTAL = "loom_billing_stripe_unapplied_stale_total"  # S1a {type}
STALE_EVENTS = "loom_billing_stripe_unapplied_stale_events"  # S1b
RETRY_LAST_SUCCESS = "loom_billing_stripe_retry_last_success_seconds"  # S5
CREDITED_MICROS = "loom_billing_stripe_credited_micros_total"


# ------------------------------------------------------------------ scratch databases


def _url_with(pg_url: str, **over: Any) -> str:
    info = psycopg.conninfo.conninfo_to_dict(pg_url)
    return psycopg.conninfo.make_conninfo("", **{**info, **over})


def _admin(pg_url: str) -> str:
    return _url_with(pg_url, dbname="postgres")


def _drop(pg_url: str, name: str) -> None:
    with psycopg.connect(_admin(pg_url), autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )


@pytest.fixture(scope="session")
def m3s_template(pg_url: str) -> Iterator[str]:
    """Name of a database migrated once per session, the template of `scratch_url`."""
    name = f"loom_m3s_tpl_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_admin(pg_url), autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        upgrade(_url_with(pg_url, dbname=name))
        yield name
    finally:
        _drop(pg_url, name)


@pytest.fixture
def scratch_url(pg_url: str, m3s_template: str) -> Iterator[str]:
    """A fresh, migrated database of this test's own (a clone of the template; no
    migrations run), dropped at teardown."""
    name = f"loom_m3s_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(_admin(pg_url), autocommit=True) as conn:
        for attempt in range(50):
            try:
                conn.execute(
                    sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                        sql.Identifier(name), sql.Identifier(m3s_template)
                    )
                )
                break
            except psycopg.errors.ObjectInUse:  # another clone being made from it
                if attempt == 49:
                    raise
                time.sleep(0.1)
    try:
        yield _url_with(pg_url, dbname=name)
    finally:
        _drop(pg_url, name)


@pytest.fixture
async def scratch_pool(scratch_url: str) -> AsyncIterator[AsyncConnectionPool]:
    p = await open_pool(scratch_url, min_size=1, max_size=8)
    try:
        yield p
    finally:
        await p.close()


def m3s_scratch_fixtures() -> tuple[Any, Any, Any]:
    """The scratch-database fixtures, for a test module to bind by name:

    m3s_template, scratch_url, scratch_pool = m3s_scratch_fixtures()
    """
    return m3s_template, scratch_url, scratch_pool


# ------------------------------------------------------------------ holders and barriers


class Holder:
    """A transaction on its own connection, open until `commit()`/`rollback()`."""

    def __init__(self, conn: psycopg.AsyncConnection, pid: int) -> None:
        self.conn = conn
        self.pid = pid

    async def run(self, query: str, *params: Any) -> list[tuple[Any, ...]]:
        async with self.conn.cursor() as cur:
            await cur.execute(query, params)  # type: ignore[arg-type]
            return await cur.fetchall() if cur.description else []

    async def commit(self) -> None:
        await self.conn.commit()

    async def rollback(self) -> None:
        await self.conn.rollback()


@contextlib.asynccontextmanager
async def holder(url: str) -> AsyncIterator[Holder]:
    conn = await psycopg.AsyncConnection.connect(url)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SELECT pg_backend_pid()")
            row = await cur.fetchone()
        assert row is not None
        await conn.commit()
        yield Holder(conn, int(row[0]))
    finally:
        with contextlib.suppress(Exception):
            await conn.rollback()
        await conn.close()


async def wait_blocked_by(
    url: str, pid: int, *, waiter: int | None = None, timeout_s: float = 20.0
) -> int:
    """Wait until some backend of this database (or the backend `waiter`) waits on a
    lock `pid` holds; returns the waiter's pid (a barrier: polls, never sleeps for a
    guessed time)."""
    deadline = time.monotonic() + timeout_s
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        while True:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT pid FROM pg_stat_activity WHERE datname = current_database() "
                    "AND %s = ANY(pg_blocking_pids(pid)) AND (%s::int IS NULL OR pid = %s) "
                    "ORDER BY backend_start LIMIT 1",
                    (pid, waiter, waiter),
                )
                row = await cur.fetchone()
            if row is not None:
                return int(row[0])
            if time.monotonic() > deadline:
                raise AssertionError(f"no backend waited on pid {pid} within {timeout_s} s")
            await asyncio.sleep(0.01)


async def wait_task_or_blocked(
    url: str,
    pid: int,
    task: asyncio.Task[Any],
    *,
    waiter_pid: int | None = None,
    timeout_s: float = 20.0,
) -> int:
    """Like `wait_blocked_by`, but fails at once (with the task's own error) if `task`
    finishes before anything waits on `pid`."""
    waiter = asyncio.ensure_future(
        wait_blocked_by(url, pid, waiter=waiter_pid, timeout_s=timeout_s)
    )
    done, _ = await asyncio.wait({waiter, task}, return_when=asyncio.FIRST_COMPLETED)
    if waiter in done:
        return waiter.result()
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter
    result = task.result()  # re-raises the task's error
    raise AssertionError(f"finished without waiting on pid {pid}: {result!r}")


# ------------------------------------------------------------------ scratch-db faults

FAULTS_SQL = """
CREATE TABLE m3s_poison (
    id       text PRIMARY KEY,
    hard     boolean NOT NULL DEFAULT false,
    sqlstate text NOT NULL DEFAULT 'P0001',
    sleep_s  float8
);
CREATE FUNCTION m3s_poison_fn() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE p m3s_poison;
BEGIN
    SELECT * INTO p FROM m3s_poison WHERE id = NEW.id;
    IF FOUND AND (p.hard OR NEW.last_error IS NULL) THEN
        IF p.sleep_s IS NOT NULL THEN
            PERFORM pg_sleep(p.sleep_s);
        END IF;
        RAISE EXCEPTION 'm3s poison: %', NEW.id USING ERRCODE = p.sqlstate;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER m3s_poison BEFORE INSERT OR UPDATE ON stripe_events
    FOR EACH ROW EXECUTE FUNCTION m3s_poison_fn();

CREATE TABLE m3s_gate (
    tbl      text NOT NULL,
    id       text NOT NULL,
    action   text NOT NULL CHECK (action IN ('wait', 'fail')),
    key      bigint,
    sqlstate text NOT NULL DEFAULT 'P0001',
    PRIMARY KEY (tbl, id)
);
CREATE FUNCTION m3s_gate_fn() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE g m3s_gate;
BEGIN
    SELECT * INTO g FROM m3s_gate
    WHERE tbl = TG_TABLE_NAME AND id = to_jsonb(NEW) ->> TG_ARGV[0];
    IF FOUND THEN
        IF g.action = 'wait' THEN
            PERFORM pg_advisory_xact_lock(g.key);
        ELSE
            RAISE EXCEPTION 'm3s deferred check failed: %', g.id USING ERRCODE = g.sqlstate;
        END IF;
    END IF;
    RETURN NULL;
END $$;
CREATE CONSTRAINT TRIGGER m3s_gate AFTER INSERT OR UPDATE ON stripe_events
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION m3s_gate_fn('id');
CREATE CONSTRAINT TRIGGER m3s_gate AFTER INSERT ON ledger_transactions
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION m3s_gate_fn('idempotency_key');
CREATE CONSTRAINT TRIGGER m3s_gate AFTER INSERT OR UPDATE ON stripe_payments
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION m3s_gate_fn('payment_intent');
CREATE CONSTRAINT TRIGGER m3s_gate AFTER INSERT ON admin_audit_log
    DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION m3s_gate_fn('target_id');
"""


async def install_faults(url: str) -> None:
    """Add the poison and gate triggers to a scratch database (never the session one)."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(FAULTS_SQL)  # type: ignore[arg-type]


async def poison(
    url: str,
    evt_id: str,
    *,
    hard: bool = False,
    sqlstate: str = "P0001",
    sleep_s: float | None = None,
) -> None:
    """Make writes of this event's row raise: those that leave `last_error` NULL (every
    attempt that would record an outcome: a handler bug), or all of them (`hard`)."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(
            "INSERT INTO m3s_poison (id, hard, sqlstate, sleep_s) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET hard = EXCLUDED.hard, "
            "sqlstate = EXCLUDED.sqlstate, sleep_s = EXCLUDED.sleep_s",
            (evt_id, hard, sqlstate, sleep_s),
        )


async def cure(url: str, evt_id: str) -> None:
    """The release that fixes the handler for this event."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute("DELETE FROM m3s_poison WHERE id = %s", (evt_id,))


async def gate(
    url: str,
    table: str,
    row_id: str,
    *,
    fail: bool = False,
    key: int | None = None,
    sqlstate: str = "P0001",
) -> None:
    """A commit-time check on one row of `table` (by its id column): `fail` raises
    `sqlstate` at COMMIT; otherwise the COMMIT waits on advisory lock `key`."""
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute(
            "INSERT INTO m3s_gate (tbl, id, action, key, sqlstate) VALUES (%s, %s, %s, %s, %s)",
            (table, row_id, "fail" if fail else "wait", key, sqlstate),
        )


async def ungate(url: str, table: str, row_id: str) -> None:
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        await conn.execute("DELETE FROM m3s_gate WHERE tbl = %s AND id = %s", (table, row_id))


# ------------------------------------------------------------------ TCP faults

_SSL_REQUEST = 80877103
_GSSENC_REQUEST = 80877104
_CANCEL_REQUEST = 80877102
_COMMIT = re.compile(r"^\s*(COMMIT|END)\b", re.IGNORECASE)


def _shut(s: socket.socket | None) -> None:
    if s is None:
        return
    with contextlib.suppress(OSError):
        s.shutdown(socket.SHUT_RDWR)
    with contextlib.suppress(OSError):
        s.close()


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf += chunk
    return buf


class _Conn:
    def __init__(self, client: socket.socket) -> None:
        self.client = client
        self.upstream: socket.socket | None = None
        self.kill_on_reply = False
        self.kill_next_statement = False
        self.dead = False

    def kill(self) -> None:
        self.dead = True
        _shut(self.client)
        _shut(self.upstream)


class FaultProxy:
    """A Postgres TCP proxy (127.0.0.1) with faults:

    - until `open()` the port is bound but not listening (connections refused);
    - `refuse(True)`: new connections are accepted and closed at once;
    - `kill_all()`: every live connection is cut;
    - `arm_commit(landed=..., then_refuse=...)`: the next COMMIT on any connection is
      either dropped before it reaches the server (`landed=False`: the transaction rolls
      back) or forwarded and its result never returned (`landed=True`); that connection
      is then cut and, with `then_refuse`, every other one too and new ones refused;
    - `arm_kill_after(pattern)`: once a statement matching `pattern` was forwarded, the
      connection is cut at its next statement (e.g. inside a savepoint);
    - `statements()`: the SQL of every simple query and Parse sent through it.

    The proxy answers SSL/GSSAPI requests with "no", so its URL needs no `sslmode`."""

    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self.upstream = (upstream_host, upstream_port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        self._lock = threading.Lock()
        self._conns: list[_Conn] = []
        self._statements: list[str] = []
        self._refusing = False
        self._commit_arm: tuple[bool, bool] | None = None
        self._after_arm: re.Pattern[str] | None = None
        self.fired = threading.Event()
        self._closed = False

    @classmethod
    def for_url(cls, pg_url: str) -> FaultProxy:
        info = psycopg.conninfo.conninfo_to_dict(pg_url)
        return cls(str(info.get("host") or "localhost"), int(info.get("port") or 5432))

    def url(self, pg_url: str) -> str:
        return _url_with(pg_url, host="127.0.0.1", port=str(self.port))

    # control

    def open(self) -> None:
        self._sock.listen(64)
        threading.Thread(target=self._accept, daemon=True).start()

    def refuse(self, on: bool = True) -> None:
        with self._lock:
            self._refusing = on

    def kill_all(self) -> None:
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            c.kill()

    def arm_commit(self, *, landed: bool, then_refuse: bool = False) -> None:
        with self._lock:
            self._commit_arm = (landed, then_refuse)
        self.fired.clear()

    def arm_kill_after(self, pattern: str) -> None:
        with self._lock:
            self._after_arm = re.compile(pattern, re.IGNORECASE)
        self.fired.clear()

    def statements(self) -> list[str]:
        with self._lock:
            return list(self._statements)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conns = list(self._conns)
        _shut(self._sock)
        for c in conns:
            c.kill()

    # forwarding

    def _accept(self) -> None:
        while True:
            try:
                client, _ = self._sock.accept()
            except OSError:
                return
            with self._lock:
                refusing = self._refusing or self._closed
            if refusing:
                _shut(client)
                continue
            c = _Conn(client)
            with self._lock:
                self._conns.append(c)
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _fire(self, c: _Conn, then_refuse: bool) -> None:
        if then_refuse:
            with self._lock:
                self._refusing = True
                others = [o for o in self._conns if o is not c]
            for o in others:
                o.kill()
        self.fired.set()

    def _serve(self, c: _Conn) -> None:
        client = c.client
        try:
            while True:  # startup: untyped messages
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!i", head)
                body = _recv_exact(client, length - 4)
                if body is None:
                    return
                (code,) = struct.unpack("!i", body[:4])
                if code in (_SSL_REQUEST, _GSSENC_REQUEST):
                    client.sendall(b"N")
                    continue
                c.upstream = socket.create_connection(self.upstream, timeout=10)
                c.upstream.settimeout(None)
                c.upstream.sendall(head + body)
                if code == _CANCEL_REQUEST:
                    return
                break
            threading.Thread(target=self._pump, args=(c,), daemon=True).start()
            in_commit = False
            commit_mode: tuple[bool, bool] | None = None
            while not c.dead:
                kind = _recv_exact(client, 1)
                if kind is None:
                    return
                head = _recv_exact(client, 4)
                if head is None:
                    return
                (length,) = struct.unpack("!i", head)
                body = _recv_exact(client, length - 4) if length > 4 else b""
                if body is None:
                    return
                text: str | None = None
                if kind == b"Q":
                    text = body.split(b"\0", 1)[0].decode(errors="replace")
                elif kind == b"P":
                    text = body.split(b"\0", 2)[1].decode(errors="replace")
                if text is not None:
                    with self._lock:
                        self._statements.append(text)
                        if c.kill_next_statement:
                            c.kill_next_statement = False
                            fire_kill = True
                        else:
                            fire_kill = False
                        if self._after_arm is not None and self._after_arm.search(text):
                            self._after_arm = None
                            c.kill_next_statement = True
                        if self._commit_arm is not None and _COMMIT.match(text):
                            commit_mode, self._commit_arm = self._commit_arm, None
                            in_commit = True
                    if fire_kill:
                        c.kill()
                        self._fire(c, False)
                        return
                if in_commit and commit_mode is not None:
                    landed, then_refuse = commit_mode
                    if not landed:
                        c.kill()
                        self._fire(c, then_refuse)
                        return
                    # Forward the COMMIT; cut the connection when its result comes back.
                    if kind in (b"Q", b"S"):
                        c.kill_on_reply = True
                        in_commit = False
                        assert c.upstream is not None
                        c.upstream.sendall(kind + head + body)
                        if then_refuse:
                            with self._lock:
                                self._refusing = True
                        continue
                assert c.upstream is not None
                c.upstream.sendall(kind + head + body)
        except OSError:
            return
        finally:
            if not c.kill_on_reply:
                c.kill()

    def _pump(self, c: _Conn) -> None:
        try:
            while True:
                assert c.upstream is not None
                data = c.upstream.recv(65536)
                if not data:
                    break
                if c.kill_on_reply:
                    # The server answered the COMMIT: it landed; the client never hears.
                    c.kill()
                    with self._lock:
                        refusing = self._refusing
                    self._fire(c, refusing)
                    return
                c.client.sendall(data)
        except OSError:
            pass
        finally:
            c.kill()


@contextlib.contextmanager
def fault_proxy(pg_url: str, *, opened: bool = True) -> Iterator[FaultProxy]:
    p = FaultProxy.for_url(pg_url)
    if opened:
        p.open()
    try:
        yield p
    finally:
        p.close()


class Blackhole:
    """A TCP server on 127.0.0.1 that accepts connections, reads, and never answers."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self.port: int = self._sock.getsockname()[1]
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                s, _ = self._sock.accept()
            except OSError:
                return
            with self._lock:
                self._clients.append(s)
            threading.Thread(target=self._drain, args=(s,), daemon=True).start()

    @staticmethod
    def _drain(s: socket.socket) -> None:
        with contextlib.suppress(OSError):
            while s.recv(65536):
                pass

    def close(self) -> None:
        _shut(self._sock)
        with self._lock:
            clients = list(self._clients)
        for s in clients:
            _shut(s)


def closed_port() -> int:
    """A local port nothing listens on (bound, then released)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port
