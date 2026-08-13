"""Persist the original server-validated query for durable query resumes.

Revision ID: 0006_query_run_question
Revises: 0005_ingestion_dead_letter_state
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0006_query_run_question"
down_revision: str | None = "0005_ingestion_dead_letter_state"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("question", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "question")
