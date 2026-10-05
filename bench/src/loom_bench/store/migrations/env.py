"""Alembic environment, configured from code by `loom_bench.store.db.alembic_config`."""

from alembic import context

from loom_bench.store.db import get_engine
from loom_bench.store.models import Base

config = context.config
url = config.attributes["url"]


def run_offline() -> None:
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    with get_engine(url).connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=Base.metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
