"""Allow complete multi-span Parent AST locators.

Revision ID: 0003_parent_ast_locator_longtext
Revises: 0002_version_manifests
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql


revision: str = "0003_parent_ast_locator_longtext"
down_revision: str | None = "0002_version_manifests"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.alter_column(
        "parent_chunks",
        "ast_locator",
        existing_type=sa.String(512),
        type_=mysql.LONGTEXT(),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "parent_chunks",
        "ast_locator",
        existing_type=mysql.LONGTEXT(),
        type_=sa.String(512),
        existing_nullable=False,
    )
