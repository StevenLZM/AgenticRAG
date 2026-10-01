"""Persist chat session metadata and optional per-turn request/source fields."""
from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

revision: str = "0008_chat_sessions"
down_revision: str | None = "0007_query_run_answer"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The public limit counts Unicode characters; TEXT only holds 64 KiB.
    op.alter_column("agent_runs", "question", existing_type=sa.Text(), type_=mysql.MEDIUMTEXT(), existing_nullable=False)
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("creation_request_id", sa.String(36), nullable=False),
        sa.Column("title", sa.String(100), nullable=False),
        sa.Column("title_source", sa.String(16), nullable=False),
        sa.Column("created_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("updated_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("last_activity_at", mysql.DATETIME(fsp=6), nullable=False),
        sa.Column("deleted_at", mysql.DATETIME(fsp=6), nullable=True),
        sa.CheckConstraint("title_source IN ('default','first_question','manual')", name="ck_chat_session_title_source"),
        sa.UniqueConstraint("user_id", "creation_request_id", name="uq_chat_session_creation"),
        mysql_engine="InnoDB", mysql_charset="utf8mb4",
    )
    op.create_index("ix_chat_sessions_user_activity", "chat_sessions", ["user_id", "deleted_at", "last_activity_at", "id"])
    op.add_column("agent_runs", sa.Column("client_request_id", sa.String(36), nullable=True))
    op.add_column("agent_runs", sa.Column("answer_sources", sa.JSON(), nullable=True))
    op.create_unique_constraint("uq_runs_client_request", "agent_runs", ["user_id", "thread_id", "client_request_id"])


def downgrade() -> None:
    # Retain the backward-compatible wider question column to avoid data loss.
    op.drop_constraint("uq_runs_client_request", "agent_runs", type_="unique")
    op.drop_column("agent_runs", "answer_sources")
    op.drop_column("agent_runs", "client_request_id")
    op.drop_table("chat_sessions")
