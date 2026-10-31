"""Additive SQL tables for the supervisor runtime, independent of SQLite checkpoints."""

from sqlalchemy import (
    JSON, Column, DateTime, Index, Integer, MetaData, String, Table, UniqueConstraint,
)
from sqlalchemy.dialects import mysql

metadata = MetaData()
timestamp = DateTime().with_variant(mysql.DATETIME(fsp=6), "mysql")

runs = Table(
    "supervisor_runs", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(255), nullable=False),
    Column("version", Integer, nullable=False),
    Column("status", String(32), nullable=False),
    Column("deadline", timestamp, nullable=False),
    Column("data", JSON, nullable=False),
    Column("created_at", timestamp, nullable=False),
    Index("ix_supervisor_runs_status_deadline", "status", "deadline"),
)
work_items = Table(
    "agent_work_items", metadata,
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("logical_key", String(255), nullable=False),
    Column("stage", String(32), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("status", String(32), nullable=False),
    Column("owner", String(255)),
    Column("lease_token", Integer, nullable=False),
    Column("lease_until", timestamp),
    Column("attempt_count", Integer, nullable=False),
    Column("dispatch_generation", Integer, nullable=False),
    Column("not_before", timestamp, nullable=False),
    Column("error_code", String(128)),
    Column("created_at", timestamp, nullable=False),
    UniqueConstraint("run_id", "logical_key", name="uq_agent_work_logical"),
    Index("ix_agent_work_ready", "stage", "status", "not_before"),
    Index("ix_agent_work_run", "run_id", "status"),
)
results = Table(
    "agent_task_results", metadata,
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("work_id", String(36), nullable=False, unique=True),
    Column("stage", String(32), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("created_at", timestamp, nullable=False),
    Index("ix_agent_results_run", "run_id"),
)
decisions = Table(
    "supervisor_decisions", metadata,
    Column("run_id", String(36), primary_key=True),
    Column("decision_id", String(80), primary_key=True),
    Column("input_version", Integer, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("work_ids", JSON, nullable=False),
)
snapshots = Table(
    "evidence_input_snapshots", metadata,
    Column("id", String(64), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("group_key", String(80), nullable=False),
    Column("group_revision", Integer, nullable=False),
    Column("aggregation_version", Integer, nullable=False),
    Column("data", JSON, nullable=False),
    UniqueConstraint("run_id", "group_key", "group_revision", "aggregation_version", name="uq_evidence_snapshot_version"),
)
inbox = Table(
    "supervisor_inbox", metadata,
    Column("run_id", String(36), primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("result_id", String(36), nullable=False, unique=True),
    Column("received_at", timestamp, nullable=False),
)
outbox = Table(
    "supervisor_outbox", metadata,
    Column("id", String(36), primary_key=True),
    Column("run_id", String(36), nullable=False),
    Column("stream", String(255), nullable=False),
    Column("aggregate_id", String(80), nullable=False),
    Column("dedupe_key", String(255), nullable=False, unique=True),
    Column("status", String(16), nullable=False),
    Column("next_attempt_at", timestamp, nullable=False),
    Column("created_at", timestamp, nullable=False),
    Index("ix_supervisor_outbox_pending", "status", "next_attempt_at"),
)
chunks = Table(
    "answer_stream_chunks", metadata,
    Column("run_id", String(36), primary_key=True),
    Column("epoch", Integer, primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("event_type", String(40), nullable=False),
    Column("data", JSON, nullable=False),
    Column("created_at", timestamp, nullable=False),
)
resources = Table(
    "agent_resource_leases", metadata,
    Column("resource", String(255), primary_key=True),
    Column("slot", Integer, primary_key=True),
    Column("owner", String(255)),
    Column("token", Integer, nullable=False),
    Column("lease_until", timestamp),
)

# Model-generated identifiers are case-sensitive, including unique work keys.
for table in metadata.tables.values():
    table.dialect_options["mysql"].update(
        engine="InnoDB", charset="utf8mb4", collate="utf8mb4_bin",
    )
