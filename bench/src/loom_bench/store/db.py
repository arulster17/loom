"""Engine, sessions and schema migrations for the results store."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://loom:loom@localhost:5432/loom"
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def database_url(url: str | None = None) -> str:
    return url or os.environ.get("LOOM_DATABASE_URL") or DEFAULT_DATABASE_URL


def get_engine(url: str | None = None) -> Engine:
    """Shared engine for `url`, else $LOOM_DATABASE_URL, else the local compose database."""
    return _engine(database_url(url))


@cache
def _engine(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _sqlite_foreign_keys)
    return engine


def _sqlite_foreign_keys(dbapi_conn: Any, _record: Any) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@cache
def _sessionmaker(url: str) -> sessionmaker[Session]:
    return sessionmaker(_engine(url), expire_on_commit=False)


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Session committed on success and rolled back on error."""
    session = _sessionmaker(database_url(url))()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def alembic_config(url: str | None = None) -> Config:
    """Alembic config built in code; no alembic.ini needed.

    New revisions: `command.revision(alembic_config(url), message, autogenerate=True)`
    against a database already at head.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    cfg.attributes["url"] = database_url(url)
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)
