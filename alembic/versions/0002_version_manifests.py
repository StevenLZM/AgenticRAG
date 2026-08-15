"""Associate immutable staging manifests with document versions.

Revision ID: 0002_version_manifests
Revises: 0001_initial_schema
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0002_version_manifests"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("manifest_path", sa.String(1024), nullable=True),
    )
    op.add_column(
        "document_versions",
        sa.Column("manifest_hash", sa.String(64), nullable=True),
    )
    op.create_check_constraint(
        "ck_document_versions_manifest_pair",
        "document_versions",
        "((manifest_path IS NULL AND manifest_hash IS NULL) OR "
        "(manifest_path IS NOT NULL AND manifest_hash IS NOT NULL))",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_document_versions_manifest_pair",
        "document_versions",
        type_="check",
    )
    op.drop_column("document_versions", "manifest_hash")
    op.drop_column("document_versions", "manifest_path")
