import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import inspect

from loom_bench.store.db import downgrade, get_engine, upgrade
from loom_bench.store.models import Base

pytestmark = pytest.mark.filterwarnings(
    # SQLite cannot reflect the lower(email) index; Postgres compares it.
    "ignore:.*expression-based index.*:UserWarning",
    "ignore:.*expression-based index.*",
)


def _tables(url: str) -> set[str]:
    return set(inspect(get_engine(url)).get_table_names())


def test_migrations_match_models(db_url):
    with get_engine(db_url).connect() as conn:
        ctx = MigrationContext.configure(
            conn, opts={"compare_type": True, "compare_server_default": True}
        )
        assert compare_metadata(ctx, Base.metadata) == []


def test_upgrade_downgrade_round_trip(empty_db_url):
    upgrade(empty_db_url)
    assert set(Base.metadata.tables) <= _tables(empty_db_url)
    downgrade(empty_db_url)
    assert _tables(empty_db_url) == {"alembic_version"}
    upgrade(empty_db_url)
    assert set(Base.metadata.tables) <= _tables(empty_db_url)
