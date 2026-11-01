"""Frozen additive retrieval registry, rebuild leases and maintenance release controls."""

from alembic import op
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Table,
    UniqueConstraint,
    MetaData,
)

revision = "0011_context_release"
down_revision = "0010_context_builds"
branch_labels = None
depends_on = None
metadata = MetaData()

retrieval_builds = Table(
    "retrieval_builds",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("document_version_id", String(36), nullable=False),
    Column("index_generation", String(128), nullable=False),
    Column("manifest_ref", JSON, nullable=False),
    Column("representation_version", String(32)),
    Column("build_fingerprint", String(64)),
    Column("build_config_hash", String(64)),
    Column("parent_hash", String(64), nullable=False),
    Column("child_identity_digest", String(64), nullable=False),
    Column("state", String(16), nullable=False),
    UniqueConstraint(
        "user_id",
        "document_version_id",
        "index_generation",
        name="uq_retrieval_build_generation",
    ),
)

retrieval_rebuild_jobs = Table(
    "retrieval_rebuild_jobs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("document_id", String(36), nullable=False),
    Column("document_version_id", String(36), nullable=False),
    Column("index_generation", String(128), nullable=False),
    Column("payload", JSON),
    Column("status", String(24), nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("lease_owner", String(255)),
    Column("lease_expires_at", DateTime(timezone=True)),
    Column("error_code", String(128)),
    UniqueConstraint(
        "user_id",
        "document_version_id",
        "index_generation",
        name="uq_rebuild_generation",
    ),
)

context_gate = Table(
    "context_release_gate",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("state", String(16), nullable=False),
    Column("generation", String(128), nullable=False),
    Column("config_hash", String(64), nullable=False),
    Column("epoch", Integer, nullable=False),
    Column("roles", JSON, nullable=False),
    Column("inventory", JSON, nullable=False),
    Column("receipt_hash", String(64), nullable=False),
    Column("maintenance_receipt", JSON, nullable=False),
    Column("receipts", JSON, nullable=False),
    Column("rollback_generation", String(128)),
)

release_journal = Table(
    "context_release_journal",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("stage", String(32), nullable=False),
    Column("source_generation", String(128), nullable=False),
    Column("target_generation", String(128), nullable=False),
    Column("source_config_hash", String(64), nullable=False),
    Column("target_config_hash", String(64), nullable=False),
    Column("target_snapshot", JSON, nullable=False),
    Column("before_mapping", JSON, nullable=False),
    Column("after_mapping", JSON, nullable=False),
    Column("corpus_hash", String(64), nullable=False),
    Column("receipt_hash", String(64), nullable=False),
    Column("maintenance_receipt", JSON, nullable=False),
)

maintenance_nonces = Table(
    "context_maintenance_nonces",
    metadata,
    Column("nonce", String(128), primary_key=True),
    Column("receipt_hash", String(64), nullable=False),
    Column("operation", String(128), nullable=False),
)

context_model_capacity = Table(
    "context_model_capacity",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("capacity", Integer, nullable=False),
    Column("version", Integer, nullable=False),
    Column("leases", JSON, nullable=False),
)

context_build_observations = Table(
    "context_build_observations",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("build_id", String(64), nullable=False, index=True),
    Column("item_id", String(64), nullable=False),
    Column("event", JSON, nullable=False),
)

for table in metadata.tables.values():
    table.dialect_options["mysql"].update(
        engine="InnoDB", charset="utf8mb4", collate="utf8mb4_bin"
    )


def upgrade():
    metadata.create_all(op.get_bind())


def downgrade():
    metadata.drop_all(op.get_bind())
