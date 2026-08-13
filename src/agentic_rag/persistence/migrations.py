"""Shared database URL resolution for Alembic and application settings."""

from agentic_rag.config import Settings


def migration_database_url() -> str:
    """Return the configured application MySQL DSN for command-line Alembic."""
    return Settings().mysql_dsn
