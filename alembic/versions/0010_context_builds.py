"""Frozen additive grounded-context build accounting and result pointers."""

from alembic import op
from sqlalchemy import JSON, Column, Float, Integer, String, Table, MetaData

revision = "0010_context_builds"
down_revision = "0009_supervisor_runtime"
branch_labels = None
depends_on = None
metadata = MetaData()

context_build_budgets = Table(
    "context_build_budgets",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("document_version_id", String(36), nullable=False),
    Column("build_fingerprint", String(64), nullable=False),
    Column("request_count", Integer, nullable=False),
    Column("reserved_tokens", Integer, nullable=False),
    Column("consumed_tokens", Integer, nullable=False),
    Column("reserved_cost", Float, nullable=False),
    Column("consumed_cost", Float, nullable=False),
)
context_build_items = Table(
    "context_build_items",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("build_id", String(64), nullable=False, index=True),
    Column("source_child_id", String(255), nullable=False),
    Column("status", String(16), nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("artifact_ref", JSON),
    Column("accepted_generation", Integer),
)
context_build_reservations = Table(
    "context_build_reservations",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("build_id", String(64), nullable=False, index=True),
    Column("item_id", String(64), nullable=False),
    Column("claim_generation", Integer, nullable=False),
    Column("tokens", Integer, nullable=False),
    Column("cost", Float, nullable=False),
    Column("status", String(16), nullable=False),
)


for table in metadata.tables.values():
    table.dialect_options["mysql"].update(
        engine="InnoDB", charset="utf8mb4", collate="utf8mb4_bin"
    )


def upgrade() -> None:
    metadata.create_all(op.get_bind(), checkfirst=False)


def downgrade() -> None:
    metadata.drop_all(op.get_bind(), checkfirst=False)
