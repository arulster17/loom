"""Initial Phase 0 schema

Revision ID: 0001
Revises:
Create Date: 2026-10-04 22:28:04.040705
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | None = None
depends_on: str | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
TIMESTAMPTZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "bench_experiments",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("spec", JSON, nullable=False),
        sa.Column("spec_hash", sa.Text(), nullable=False),
        sa.Column("git_sha", sa.Text(), nullable=True),
        sa.Column("git_dirty", sa.Boolean(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("budget_micros", sa.BigInteger(), nullable=True),
        sa.Column("spent_micros", sa.BigInteger(), server_default=sa.text("0"), nullable=False),
        sa.Column("abort_reason", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False),
        sa.Column("finished_at", TIMESTAMPTZ, nullable=True),
        sa.CheckConstraint(
            "status IN ('planned', 'running', 'completed', 'aborted', 'failed')",
            name=op.f("ck_bench_experiments_status"),
        ),
        sa.CheckConstraint(
            "spent_micros >= 0", name=op.f("ck_bench_experiments_spent_nonnegative")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_experiments")),
    )
    op.create_index(
        op.f("ix_bench_experiments_spec_hash"), "bench_experiments", ["spec_hash"], unique=False
    )
    op.create_table(
        "waitlist_signups",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_waitlist_signups")),
    )
    op.create_index(
        "uq_waitlist_signups_email_lower",
        "waitlist_signups",
        [sa.literal_column("lower(email)")],
        unique=True,
    )
    op.create_table(
        "bench_cold_starts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("resource_id", sa.Text(), nullable=True),
        sa.Column("config_hash", sa.Text(), nullable=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("stages", JSON, nullable=False),
        sa.Column("total_s", sa.Double(), nullable=False),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False),
        sa.CheckConstraint("kind IN ('cold', 'warm')", name=op.f("ck_bench_cold_starts_kind")),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["bench_experiments.id"],
            name=op.f("fk_bench_cold_starts_experiment_id_bench_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_cold_starts")),
    )
    op.create_index(
        op.f("ix_bench_cold_starts_experiment_id"),
        "bench_cold_starts",
        ["experiment_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_bench_cold_starts_config_hash"),
        "bench_cold_starts",
        ["config_hash"],
        unique=False,
    )
    op.create_table(
        "bench_eval_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("task_version", sa.Text(), nullable=True),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("ci_low", sa.Double(), nullable=True),
        sa.Column("ci_high", sa.Double(), nullable=True),
        sa.Column("provenance", JSON, nullable=False),
        sa.Column("samples_uri", sa.Text(), nullable=True),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["bench_experiments.id"],
            name=op.f("fk_bench_eval_runs_experiment_id_bench_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_eval_runs")),
    )
    op.create_index(
        op.f("ix_bench_eval_runs_config_hash_task"),
        "bench_eval_runs",
        ["config_hash", "task"],
        unique=False,
    )
    op.create_index(
        op.f("ix_bench_eval_runs_experiment_id"), "bench_eval_runs", ["experiment_id"], unique=False
    )
    op.create_table(
        "bench_gate_decisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("baseline_config_hash", sa.Text(), nullable=False),
        sa.Column("candidate_config_hash", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("details", JSON, nullable=False),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False),
        sa.CheckConstraint(
            "decision IN ('pass', 'fail', 'inconclusive')",
            name=op.f("ck_bench_gate_decisions_decision"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["bench_experiments.id"],
            name=op.f("fk_bench_gate_decisions_experiment_id_bench_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_gate_decisions")),
    )
    op.create_index(
        op.f("ix_bench_gate_decisions_candidate_config_hash"),
        "bench_gate_decisions",
        ["candidate_config_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_bench_gate_decisions_experiment_id"),
        "bench_gate_decisions",
        ["experiment_id"],
        unique=False,
    )
    op.create_table(
        "bench_resources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("resource_type", sa.Text(), nullable=False),
        sa.Column("resource_id", sa.Text(), nullable=False),
        sa.Column("region", sa.Text(), nullable=True),
        sa.Column("experiment_id", sa.Uuid(), nullable=True),
        sa.Column("tags", JSON, nullable=False),
        sa.Column("created_at", TIMESTAMPTZ, nullable=False),
        sa.Column("ttl_at", TIMESTAMPTZ, nullable=False),
        sa.Column("terminated_at", TIMESTAMPTZ, nullable=True),
        sa.Column("terminated_by", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "terminated_by IN ('runner', 'reaper', 'self-ttl')",
            name=op.f("ck_bench_resources_terminated_by"),
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["bench_experiments.id"],
            name=op.f("fk_bench_resources_experiment_id_bench_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_resources")),
    )
    op.create_index(
        op.f("ix_bench_resources_experiment_id"), "bench_resources", ["experiment_id"], unique=False
    )
    op.create_index(
        "ix_bench_resources_live_ttl_at",
        "bench_resources",
        ["ttl_at"],
        unique=False,
        postgresql_where=sa.text("terminated_at IS NULL"),
        sqlite_where=sa.text("terminated_at IS NULL"),
    )
    op.create_index(
        op.f("ix_bench_resources_provider_resource_id"),
        "bench_resources",
        ["provider", "resource_id"],
        unique=True,
    )
    op.create_table(
        "bench_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("cell_key", sa.Text(), nullable=True),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("workload", sa.Text(), nullable=True),
        sa.Column("load_mode", sa.Text(), nullable=True),
        sa.Column("load_value", sa.Double(), nullable=True),
        sa.Column("repetition", sa.Integer(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("provenance", JSON, nullable=False),
        sa.Column("summary", JSON, nullable=True),
        sa.Column("requests_uri", sa.Text(), nullable=True),
        sa.Column("started_at", TIMESTAMPTZ, nullable=True),
        sa.Column("finished_at", TIMESTAMPTZ, nullable=True),
        sa.CheckConstraint(
            "load_mode IN ('open_loop', 'closed_loop')", name=op.f("ck_bench_runs_load_mode")
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["bench_experiments.id"],
            name=op.f("fk_bench_runs_experiment_id_bench_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_runs")),
    )
    op.create_index(op.f("ix_bench_runs_config_hash"), "bench_runs", ["config_hash"], unique=False)
    op.create_index(
        op.f("ix_bench_runs_experiment_id"), "bench_runs", ["experiment_id"], unique=False
    )
    op.create_index(
        op.f("ix_bench_runs_experiment_id_cell_key_repetition"),
        "bench_runs",
        ["experiment_id", "cell_key", "repetition"],
        unique=False,
    )
    op.create_table(
        "bench_spend",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("experiment_id", sa.Uuid(), nullable=False),
        sa.Column("resource_id", sa.Text(), nullable=True),
        sa.Column("amount_micros", sa.BigInteger(), nullable=False),
        sa.Column("basis", JSON, nullable=False),
        sa.Column("recorded_at", TIMESTAMPTZ, nullable=False),
        sa.CheckConstraint("amount_micros >= 0", name=op.f("ck_bench_spend_amount_nonnegative")),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["bench_experiments.id"],
            name=op.f("fk_bench_spend_experiment_id_bench_experiments"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_bench_spend")),
    )
    op.create_index(
        op.f("ix_bench_spend_experiment_id"), "bench_spend", ["experiment_id"], unique=False
    )


def downgrade() -> None:
    op.drop_table("bench_spend")
    op.drop_table("bench_runs")
    op.drop_table("bench_resources")
    op.drop_table("bench_gate_decisions")
    op.drop_table("bench_eval_runs")
    op.drop_table("bench_cold_starts")
    op.drop_table("waitlist_signups")
    op.drop_table("bench_experiments")
