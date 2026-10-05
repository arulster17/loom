"""Gate decision 'review': divergence beyond the noise-calibrated limits, tasks passing

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None

TABLE = "bench_gate_decisions"
NAME = "decision"


def _replace_check(values: tuple[str, ...]) -> None:
    allowed = ", ".join(f"'{v}'" for v in values)
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_constraint(op.f(f"ck_{TABLE}_{NAME}"), type_="check")
        batch.create_check_constraint(op.f(f"ck_{TABLE}_{NAME}"), f"decision IN ({allowed})")


def upgrade() -> None:
    _replace_check(("pass", "fail", "inconclusive", "review"))


def downgrade() -> None:
    reviews = op.get_bind().execute(
        sa.text(f"SELECT count(*) FROM {TABLE} WHERE decision = 'review'")
    )
    if reviews.scalar_one():
        raise RuntimeError(f"{TABLE} holds 'review' decisions, which revision 0001 cannot store")
    _replace_check(("pass", "fail", "inconclusive"))
