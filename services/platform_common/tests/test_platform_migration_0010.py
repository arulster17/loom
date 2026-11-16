"""Migration 0010 (`serving_pods`, `serving_spend_alerts`; docs/phase1-m2-spec.md §5,
§13 item 4): applies on a fresh database and on one already migrated to 0009."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import uuid
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import errors, sql

from loom_platform import migrate
from loom_platform.migrate import load_migrations, upgrade

pytestmark = pytest.mark.postgres

SERVING_PODS = {
    "id": ("uuid", "NO"),
    "model_id": ("text", "NO"),
    "pod_name": ("text", "NO"),
    "pod_id": ("text", "YES"),
    "state": ("text", "NO"),
    "gpu_type_id": ("text", "YES"),
    "gpu_count": ("integer", "YES"),
    "data_center": ("text", "YES"),
    "cost_per_hr_micros": ("bigint", "YES"),
    "public_ip": ("text", "YES"),
    "proxy_port": ("integer", "YES"),
    "ssh_port": ("integer", "YES"),
    "cert_not_after": ("timestamp with time zone", "YES"),
    "lease_until": ("timestamp with time zone", "YES"),
    "stages": ("jsonb", "NO"),
    "failure": ("text", "YES"),
    "terminated_by": ("text", "YES"),
    "created_at": ("timestamp with time zone", "NO"),
    "ready_at": ("timestamp with time zone", "YES"),
    "drain_started_at": ("timestamp with time zone", "YES"),
    "terminated_at": ("timestamp with time zone", "YES"),
}
# (type, nullable); None: the spec leaves nullability open (sent_at may be
# NOT NULL DEFAULT now()).
SPEND_ALERTS: dict[str, tuple[str, str | None]] = {
    "month": ("date", "NO"),
    "fraction": ("numeric", "NO"),
    "sent_at": ("timestamp with time zone", None),
}


@contextlib.contextmanager
def scratch_database(pg_url: str) -> Iterator[str]:
    """An empty database next to the session's, dropped afterwards."""
    name = f"loom_m0010_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(pg_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    info = psycopg.conninfo.conninfo_to_dict(pg_url)
    url = psycopg.conninfo.make_conninfo("", **{**info, "dbname": name})
    try:
        yield url
    finally:
        with psycopg.connect(pg_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


def tables(url: str) -> set[str]:
    with psycopg.connect(url) as conn:
        return {
            r[0]
            for r in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
        }


def columns(url: str, table: str) -> dict[str, tuple[str, str]]:
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s",
            (table,),
        ).fetchall()
    return {name: (dtype, nullable) for name, dtype, nullable in rows}


def check_schema(url: str) -> None:
    assert {"serving_pods", "serving_spend_alerts"} <= tables(url)
    have = columns(url, "serving_pods")
    for col, want in SERVING_PODS.items():
        assert have.get(col) == want, (col, have.get(col), want)
    have = columns(url, "serving_spend_alerts")
    for col, (dtype, nullable) in SPEND_ALERTS.items():
        got = have.get(col)
        assert got is not None and got[0] == dtype, (col, got, dtype)
        if nullable is not None:
            assert got[1] == nullable, (col, got, nullable)


def test_0010_is_the_serving_pods_migration() -> None:
    by_version = {m.version: m for m in load_migrations()}
    assert 10 in by_version
    assert by_version[10].name == "serving_pods"


def test_applies_on_a_fresh_database(pg_url: str) -> None:
    with scratch_database(pg_url) as url:
        done = upgrade(url)
        assert 10 in [m.version for m in done]
        check_schema(url)
        assert upgrade(url) == []  # and is recorded


def test_applies_on_a_database_at_0009(pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    every = load_migrations()
    assert [m.version for m in every][:9] == list(range(1, 10))
    with scratch_database(pg_url) as url:
        # The database as M1 left it: 0001-0009 applied, with data in it.
        with monkeypatch.context() as m:
            m.setattr(migrate, "load_migrations", lambda: every[:9])
            assert [mig.version for mig in upgrade(url)] == list(range(1, 10))
        assert "serving_pods" not in tables(url)
        with psycopg.connect(url) as conn:
            conn.execute("INSERT INTO orgs (name, status) VALUES ('before 0010', 'active')")
            conn.commit()
        done = [mig.version for mig in upgrade(url)]
        assert done[:1] == [10]
        assert done == [m.version for m in every if m.version >= 10]
        check_schema(url)
        with psycopg.connect(url) as conn:
            n = conn.execute("SELECT count(*) FROM orgs WHERE name = 'before 0010'").fetchone()
        assert n == (1,)


def _insert_pod(conn: psycopg.Connection, **over: object) -> uuid.UUID:
    row: dict[str, object] = {
        "id": uuid.uuid4(),
        "model_id": "qwen3-8b",
        "pod_name": f"loom-serve-qwen3-8b-1763200000-{uuid.uuid4().hex[:6]}",
        "state": "creating",
        "created_at": dt.datetime.now(dt.UTC),
    }
    row.update(over)
    conn.execute(
        sql.SQL("INSERT INTO serving_pods ({}) VALUES ({})").format(
            sql.SQL(", ").join(map(sql.Identifier, row)),
            sql.SQL(", ").join(sql.Placeholder() * len(row)),
        ),
        list(row.values()),
    )
    return row["id"]  # type: ignore[return-value]


def test_serving_pods_constraints(pg_url: str) -> None:
    with psycopg.connect(pg_url) as conn:
        try:
            # stages defaults to {}, pod_id is null until RunPod answers; several rows may
            # wait for their pod id at once.
            first = _insert_pod(conn)
            _insert_pod(conn)
            stages, pod_id = conn.execute(
                "SELECT stages, pod_id FROM serving_pods WHERE id = %s", (first,)
            ).fetchone()
            assert stages == {} and pod_id is None
            conn.execute(
                "UPDATE serving_pods SET stages = %s, pod_id = %s, state = 'ready' WHERE id = %s",
                (json.dumps({"pod_created": 0.0, "first_token": 212.5}), "abc123xyz", first),
            )
            with pytest.raises(errors.UniqueViolation), conn.transaction():
                _insert_pod(conn, pod_id="abc123xyz")
            name = f"loom-serve-qwen3-8b-1763200000-{uuid.uuid4().hex[:6]}"
            _insert_pod(conn, pod_name=name)
            with pytest.raises(errors.UniqueViolation), conn.transaction():
                _insert_pod(conn, pod_name=name)
            for col in ("model_id", "pod_name", "state", "created_at"):
                with pytest.raises(errors.NotNullViolation), conn.transaction():
                    _insert_pod(conn, **{col: None})
            with pytest.raises(errors.UniqueViolation), conn.transaction():
                _insert_pod(conn, id=first)
        finally:
            conn.rollback()


def test_spend_alerts_once_per_month_and_fraction(pg_url: str) -> None:
    with psycopg.connect(pg_url) as conn:
        try:
            q = "INSERT INTO serving_spend_alerts (month, fraction, sent_at) VALUES (%s, %s, now())"
            conn.execute(q, (dt.date(2031, 1, 1), 0.5))
            conn.execute(q, (dt.date(2031, 1, 1), 0.8))
            conn.execute(q, (dt.date(2031, 2, 1), 0.5))
            with pytest.raises(errors.UniqueViolation), conn.transaction():
                conn.execute(q, (dt.date(2031, 1, 1), 0.5))
            n = conn.execute(
                "INSERT INTO serving_spend_alerts (month, fraction, sent_at) "
                "VALUES ('2031-01-01', 0.8, now()) ON CONFLICT DO NOTHING"
            ).rowcount
            assert n == 0
        finally:
            conn.rollback()
