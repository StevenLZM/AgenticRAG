"""Track durable physical deletion reconciliation.

Revision ID: 0004_document_deletion_status
Revises: 0003_parent_ast_locator_longtext
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0004_document_deletion_status"
down_revision: str | None = "0003_parent_ast_locator_longtext"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("deletion_status", sa.String(length=16), nullable=True)
    )
    op.add_column(
        "documents", sa.Column("deletion_fenced_at", sa.DateTime(), nullable=True)
    )
    op.create_check_constraint(
        "ck_documents_deletion_status",
        "documents",
        "deletion_status IS NULL OR "
        "deletion_status IN ('pending','fenced','completed')",
    )
    op.execute(
        "UPDATE documents SET deletion_status = 'pending' WHERE status = 'deleted'"
    )


def downgrade() -> None:
    op.drop_constraint("ck_documents_deletion_status", "documents", type_="check")
    op.drop_column("documents", "deletion_fenced_at")
    op.drop_column("documents", "deletion_status")
