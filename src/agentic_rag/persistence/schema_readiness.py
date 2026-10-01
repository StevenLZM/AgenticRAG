"""Read-only migration readiness; never upgrades a deployment implicitly."""
from sqlalchemy import inspect
from sqlalchemy.dialects.mysql import MEDIUMTEXT, LONGTEXT
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncEngine


class ChatSchemaUnavailable(RuntimeError):
    """Required chat migration has not been applied."""


def _check(connection: Connection) -> None:
    schema = inspect(connection)
    missing = not {"chat_sessions", "agent_runs"}.issubset(schema.get_table_names())
    if not missing:
        run_columns = schema.get_columns("agent_runs")
        columns = {c["name"] for c in run_columns}
        session_columns = {c["name"] for c in schema.get_columns("chat_sessions")}
        run_keys = {tuple(c["column_names"]) for c in schema.get_unique_constraints("agent_runs")}
        session_keys = {tuple(c["column_names"]) for c in schema.get_unique_constraints("chat_sessions")}
        indexes = {i["name"] for i in schema.get_indexes("chat_sessions")}
        missing = (not {"client_request_id", "answer_sources"}.issubset(columns)
                   or not {"id", "user_id", "creation_request_id", "title", "title_source", "created_at", "updated_at", "last_activity_at", "deleted_at"}.issubset(session_columns)
                   or ("user_id", "thread_id", "client_request_id") not in run_keys
                   or ("user_id", "creation_request_id") not in session_keys
                   or "ix_chat_sessions_user_activity" not in indexes)
        if connection.dialect.name == "mysql":
            question = next((column["type"] for column in run_columns if column["name"] == "question"), None)
            missing = missing or not isinstance(question, (MEDIUMTEXT, LONGTEXT))
    if missing:
        raise ChatSchemaUnavailable("Chat schema unavailable: apply migration 0008_chat_sessions before starting queries.")


async def check_chat_schema(engine: AsyncEngine) -> None:
    async with engine.connect() as connection:
        await connection.run_sync(_check)
