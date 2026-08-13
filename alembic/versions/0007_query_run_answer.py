"""Persist the audited public answer projection for query Run APIs.

Revision ID: 0007_query_run_answer
Revises: 0006_query_run_question
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0007_query_run_answer"
down_revision: str | None = "0006_query_run_question"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("answer", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "answer")

