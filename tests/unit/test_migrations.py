from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa

from agentic_rag.persistence.migrations import migration_database_url


def test_migration_database_url_uses_application_mysql_settings(monkeypatch) -> None:
    monkeypatch.setenv(
        "AGENTIC_RAG_MYSQL_DSN",
        "mysql+asyncmy://configured:secret@127.0.0.1:3306/agentic_rag",
    )
    monkeypatch.setenv(
        "AGENTIC_RAG_DEEPSEEK_BASE_URL", "https://models.example.invalid/v1"
    )
    monkeypatch.setenv(
        "AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL",
        "https://embeddings.example.invalid/v1",
    )

    assert migration_database_url() == (
        "mysql+asyncmy://configured:secret@127.0.0.1:3306/agentic_rag"
    )


def test_query_question_migration_avoids_text_server_default(monkeypatch) -> None:
    migration_path = (
        Path(__file__).parents[2] / "alembic" / "versions" / "0006_query_run_question.py"
    )
    spec = importlib.util.spec_from_file_location("migration_0006", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    calls: list[tuple[str, object]] = []

    class FakeOperations:
        def add_column(self, table: str, column: sa.Column[str]) -> None:
            calls.append(("add_column", (table, column)))

        def execute(self, statement: object) -> None:
            calls.append(("execute", statement))

        def alter_column(self, *args: object, **kwargs: object) -> None:
            calls.append(("alter_column", (args, kwargs)))

    monkeypatch.setattr(migration, "op", FakeOperations())
    migration.upgrade()

    add_table, add_payload = next(item for item in calls if item[0] == "add_column")
    assert add_table == "add_column"
    table, column = add_payload
    assert table == "agent_runs"
    assert column.name == "question"
    assert column.nullable is True
    assert column.server_default is None
    assert any(item[0] == "execute" for item in calls)
    assert any(item[0] == "alter_column" for item in calls)
