"""Apply the platform's SQL migrations: `loom-migrate` / `python -m loom_platform.migrate`.

Migrations are numbered plain SQL files in `loom_platform/migrations/` (NNNN_name.sql),
applied in order, each in its own transaction, and recorded in `schema_migrations` with
a SHA-256 of the file. An applied file that has since changed is an error: write a new
migration instead. A session advisory lock keeps two runners from racing.

Plain SQL keeps the schema readable from the web app's side (Auth.js reads the same
tables) without a Python ORM in between.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from importlib import resources

import psycopg
from psycopg import sql

from loom_platform.db import connect_sync, libpq_url
from loom_platform.settings import PlatformSettings

_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
# Arbitrary constant: pg_advisory_lock key for the migration runner.
_LOCK_KEY = 0x4C4F4F4D_4D494752  # "LOOMMIGR"


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode()).hexdigest()


class MigrationError(RuntimeError):
    pass


def load_migrations() -> list[Migration]:
    out: list[Migration] = []
    for entry in resources.files("loom_platform.migrations").iterdir():
        m = _FILE_RE.match(entry.name)
        if m is None:
            continue
        out.append(Migration(int(m.group(1)), m.group(2), entry.read_text(encoding="utf-8")))
    out.sort(key=lambda mig: mig.version)
    versions = [mig.version for mig in out]
    if versions != list(range(1, len(out) + 1)):
        raise MigrationError(f"migration numbers must run 1..n without gaps: {versions}")
    return out


def _ensure_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    int PRIMARY KEY,
            name       text NOT NULL,
            checksum   text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def applied(conn: psycopg.Connection) -> dict[int, tuple[str, str]]:
    rows = conn.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    return {v: (n, c) for v, n, c in rows}


def upgrade(url: str) -> list[Migration]:
    """Apply pending migrations; returns the ones applied now."""
    migrations = load_migrations()
    done: list[Migration] = []
    with connect_sync(url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
        try:
            _ensure_table(conn)
            have = applied(conn)
            for mig in migrations:
                if mig.version in have:
                    name, checksum = have[mig.version]
                    if checksum != mig.checksum:
                        raise MigrationError(
                            f"migration {mig.version:04d}_{mig.name} changed after it was "
                            f"applied (recorded as {name}); add a new migration instead"
                        )
                    continue
                with conn.transaction():
                    conn.execute(mig.sql.encode())  # bytes: no %-placeholder parsing
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name, checksum) "
                        "VALUES (%s, %s, %s)",
                        (mig.version, mig.name, mig.checksum),
                    )
                done.append(mig)
            unknown = sorted(set(have) - {m.version for m in migrations})
            if unknown:
                raise MigrationError(f"database has migrations this code does not: {unknown}")
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
    return done


def create_database(url: str) -> bool:
    """Create the URL's database if it does not exist (connecting to 'postgres')."""
    info = psycopg.conninfo.conninfo_to_dict(libpq_url(url))
    dbname = info.get("dbname")
    if not dbname:
        raise MigrationError("database URL names no database")
    admin = {**info, "dbname": "postgres"}
    with psycopg.connect(psycopg.conninfo.make_conninfo("", **admin), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if exists:
            return False
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(str(dbname))))
        return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="loom-migrate", description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", help="default: $LOOM_DATABASE_URL")
    parser.add_argument(
        "--create-database", action="store_true", help="create the database if missing"
    )
    args = parser.parse_args(argv)
    url = args.database_url or PlatformSettings().database_url
    if args.create_database and create_database(url):
        print("created database")
    done = upgrade(url)
    for mig in done:
        print(f"applied {mig.version:04d}_{mig.name}")
    if not done:
        print("up to date")


if __name__ == "__main__":
    main()
