"""Database fixtures. SQLite always; Postgres too when LOOM_TEST_DATABASE_URL is set.

Each Postgres test gets its own throwaway schema, so the target database is
never modified outside it.
"""

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, make_url, text
from sqlalchemy.orm import Session

from loom_bench.store.db import get_engine, session_scope, upgrade

PG_ENV = "LOOM_TEST_DATABASE_URL"


@pytest.fixture
def sqlite_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'loom.db'}"


@pytest.fixture
def pg_url() -> Iterator[str]:
    base = os.environ.get(PG_ENV)
    if not base:
        pytest.skip(f"{PG_ENV} not set")
    schema = f"loom_test_{uuid.uuid4().hex[:12]}"
    admin = create_engine(base)
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    url = (
        make_url(base)
        .update_query_dict({"options": f"-csearch_path={schema}"})
        .render_as_string(hide_password=False)
    )
    try:
        yield url
    finally:
        get_engine(url).dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def empty_db_url(request) -> str:
    return request.getfixturevalue("sqlite_url" if request.param == "sqlite" else "pg_url")


@pytest.fixture
def db_url(empty_db_url) -> str:
    upgrade(empty_db_url)
    return empty_db_url


@pytest.fixture
def session(db_url) -> Iterator[Session]:
    with session_scope(db_url) as s:
        yield s
