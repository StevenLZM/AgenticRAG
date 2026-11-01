"""Frozen additive migration can be created/dropped in isolated SQLite."""

import importlib.util
from pathlib import Path
from sqlalchemy import create_engine, inspect


def test_context_migration_frozen_schema_matches_runtime():
    from agentic_rag.persistence.context_builds import (
        context_build_budgets,
        context_build_items,
        context_build_reservations,
    )

    path = Path("alembic/versions/0010_context_builds.py")
    assert path.exists()
    spec = importlib.util.spec_from_file_location("context_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.down_revision == "0009_supervisor_runtime"
    assert "agentic_rag" not in path.read_text()
    engine = create_engine("sqlite://")
    module.metadata.create_all(engine)
    for table in (
        context_build_budgets,
        context_build_items,
        context_build_reservations,
    ):
        assert set(table.c.keys()) == {
            c["name"] for c in inspect(engine).get_columns(table.name)
        }
    module.metadata.drop_all(engine)
    assert inspect(engine).get_table_names() == []


def test_release_migration_has_only_additive_frozen_tables():
    from agentic_rag.persistence import context_registry, context_release, context_limiter
    from agentic_rag.persistence.context_builds import context_build_observations
    tables = (context_registry.retrieval_builds, context_registry.retrieval_rebuild_jobs,
        context_release.context_gate, context_release.release_journal, context_release.maintenance_nonces,
        context_limiter.context_model_capacity, context_build_observations)
    path = Path("alembic/versions/0011_context_release.py")
    spec = importlib.util.spec_from_file_location("release_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.down_revision == "0010_context_builds"
    assert "agentic_rag" not in path.read_text()
    engine = create_engine("sqlite://")
    module.metadata.create_all(engine)
    assert set(inspect(engine).get_table_names()) == {table.name for table in tables}
    for table in tables:
        assert set(table.c.keys()) == {c["name"] for c in inspect(engine).get_columns(table.name)}
    module.metadata.drop_all(engine)
