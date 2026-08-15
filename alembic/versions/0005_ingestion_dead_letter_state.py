"""Persist ingestion dead-letter delivery intent.

Revision ID: 0005_ingestion_dead_letter_state
Revises: 0004_document_deletion_status
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0005_ingestion_dead_letter_state"
down_revision: str | None = "0004_document_deletion_status"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("dead_letter_status", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("dead_letter_reason", sa.String(length=128), nullable=True),
    )
    op.create_check_constraint(
        "ck_ingestion_jobs_dead_letter_status",
        "ingestion_jobs",
        "(dead_letter_status IS NULL AND dead_letter_reason IS NULL) OR "
        "(status = 'failed' AND dead_letter_status IN ('pending','published') "
        "AND dead_letter_reason IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_ingestion_jobs_dead_letter_status", "ingestion_jobs", type_="check"
    )
    op.drop_column("ingestion_jobs", "dead_letter_reason")
    op.drop_column("ingestion_jobs", "dead_letter_status")
