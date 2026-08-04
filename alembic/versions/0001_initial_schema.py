"""Create the approved initial MySQL schema.

Revision ID: 0001_initial_schema
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql


revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("active_version_id", sa.String(36), nullable=True),
        sa.Column(
            "source_trust", sa.String(32), nullable=False, server_default="untrusted"
        ),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.Column(
            "updated_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.CheckConstraint(
            "status IN ('processing','active','failed','deleted')",
            name="ck_documents_status",
        ),
        sa.CheckConstraint(
            "source_trust IN ('untrusted','trusted_curated')",
            name="ck_documents_source_trust",
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_documents_user_status", "documents", ["user_id", "status"])
    op.create_index(
        "ix_documents_user_content_hash", "documents", ["user_id", "content_hash"]
    )

    op.create_table(
        "document_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("version_no", sa.Integer, nullable=False),
        sa.Column("parser_version", sa.String(128), nullable=False),
        sa.Column("pipeline_version", sa.String(128), nullable=False),
        sa.Column("canonical_ast_path", sa.String(1024), nullable=True),
        sa.Column("canonical_ast_hash", sa.String(64), nullable=True),
        sa.Column("parent_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("child_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("embedding_version", sa.String(128), nullable=False),
        sa.Column("index_generation", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.CheckConstraint(
            "status IN ('uploaded','building','active','quarantined','failed','inactive')",
            name="ck_document_versions_status",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_versions_document",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_id", "version_no", name="uq_versions_document_number"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_versions_document_status", "document_versions", ["document_id", "status"]
    )
    op.create_foreign_key(
        "fk_documents_active_version",
        "documents",
        "document_versions",
        ["active_version_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "parent_chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("document_version_id", sa.String(36), nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("heading_path", mysql.JSON, nullable=False),
        sa.Column("content_type", sa.String(64), nullable=False),
        sa.Column("content", mysql.LONGTEXT, nullable=False),
        sa.Column("page_from", sa.Integer, nullable=True),
        sa.Column("page_to", sa.Integer, nullable=True),
        sa.Column("ast_locator", sa.String(512), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.CheckConstraint(
            "status IN ('active','inactive')", name="ck_parent_chunks_status"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_parents_document",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_parents_version",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_version_id", "ordinal", name="uq_parents_version_ordinal"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index("ix_parents_user_status", "parent_chunks", ["user_id", "status"])
    op.create_index(
        "ix_parents_user_document", "parent_chunks", ["user_id", "document_id"]
    )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("document_version_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("lease_owner", sa.String(255), nullable=True),
        sa.Column("lease_expires_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("heartbeat_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_error_detail_ref", sa.String(1024), nullable=True),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.Column(
            "updated_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.CheckConstraint(
            "status IN ('queued','running','completed','quarantined','failed')",
            name="ck_ingestion_jobs_status",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_jobs_document",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_jobs_version",
            ondelete="CASCADE",
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_jobs_status_lease", "ingestion_jobs", ["status", "lease_expires_at"]
    )
    op.create_index("ix_jobs_user_created", "ingestion_jobs", ["user_id", "created_at"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("thread_id", sa.String(255), nullable=False),
        sa.Column("checkpoint_thread_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("active_slot", sa.Integer, nullable=True),
        sa.Column("lease_owner", sa.String(255), nullable=True),
        sa.Column("lease_expires_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("heartbeat_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("route", sa.String(64), nullable=True),
        sa.Column("runtime_config_snapshot_id", sa.String(64), nullable=False),
        sa.Column("runtime_config_snapshot", mysql.JSON, nullable=False),
        sa.Column("result_ref", sa.String(1024), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("termination_reason", sa.String(255), nullable=True),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.Column("started_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("finished_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued','running','cancel_requested','cancelled','completed','failed')",
            name="ck_agent_runs_status",
        ),
        sa.CheckConstraint(
            "active_slot IS NULL OR active_slot = 1", name="ck_active_slot"
        ),
        sa.UniqueConstraint(
            "user_id", "thread_id", "active_slot", name="uq_runs_active_slot"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_runs_status_lease", "agent_runs", ["status", "lease_expires_at"]
    )
    op.create_index(
        "ix_runs_user_thread_created",
        "agent_runs",
        ["user_id", "thread_id", "created_at"],
    )

    op.create_table(
        "task_outbox",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("aggregate_type", sa.String(32), nullable=False),
        sa.Column("aggregate_id", sa.String(36), nullable=False),
        sa.Column("stream_name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("next_attempt_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.Column("dispatched_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.CheckConstraint(
            "aggregate_type IN ('query_run','ingestion_job')",
            name="ck_outbox_aggregate_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending','dispatched')", name="ck_outbox_status"
        ),
        sa.UniqueConstraint(
            "aggregate_type", "aggregate_id", name="uq_outbox_aggregate"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_outbox_pending", "task_outbox", ["status", "next_attempt_at", "id"]
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("thread_id", sa.String(255), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", mysql.LONGTEXT, nullable=True),
        sa.Column("payload_ref", sa.String(1024), nullable=True),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.CheckConstraint(
            "role IN ('user','assistant','tool')", name="ck_messages_role"
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["agent_runs.id"], name="fk_messages_run", ondelete="CASCADE"
        ),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_messages_user_thread_created",
        "messages",
        ["user_id", "thread_id", "created_at"],
    )

    op.create_table(
        "agent_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("event_key", sa.String(64), nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=False),
        sa.Column("run_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("node_name", sa.String(128), nullable=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("summary", sa.String(1024), nullable=False),
        sa.Column("payload_ref", sa.String(1024), nullable=True),
        sa.Column("runtime_config_snapshot_id", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            mysql.DATETIME(fsp=6),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(6)"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["agent_runs.id"], name="fk_events_run", ondelete="CASCADE"
        ),
        sa.UniqueConstraint("event_key", name="uq_events_event_key"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_events_user_run_id", "agent_events", ["user_id", "run_id", "id"]
    )

    op.create_table(
        "memory_tombstones",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("memory_id", sa.String(255), nullable=False),
        sa.Column("requested_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completed_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.Column("last_error_detail_ref", sa.String(1024), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending','completed','failed')", name="ck_tombstones_status"
        ),
        sa.UniqueConstraint("user_id", "memory_id", name="uq_tombstones_user_memory"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )
    op.create_index(
        "ix_tombstones_status_requested",
        "memory_tombstones",
        ["status", "requested_at"],
    )


def downgrade() -> None:
    op.drop_table("memory_tombstones")
    op.drop_table("agent_events")
    op.drop_table("messages")
    op.drop_table("task_outbox")
    op.drop_table("agent_runs")
    op.drop_table("ingestion_jobs")
    op.drop_table("parent_chunks")
    op.drop_constraint("fk_documents_active_version", "documents", type_="foreignkey")
    op.drop_table("document_versions")
    op.drop_table("documents")
