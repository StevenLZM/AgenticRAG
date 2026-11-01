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
    from agentic_rag.persistence import (
        context_registry,
        context_release,
        context_limiter,
    )
    from agentic_rag.persistence.context_builds import context_build_observations

    tables = (
        context_registry.retrieval_builds,
        context_registry.retrieval_rebuild_jobs,
        context_release.context_gate,
        context_release.release_journal,
        context_release.maintenance_nonces,
        context_limiter.context_model_capacity,
        context_build_observations,
    )
    path = Path("alembic/versions/0011_context_release.py")
    spec = importlib.util.spec_from_file_location("release_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.down_revision == "0010_context_builds"
    assert "agentic_rag" not in path.read_text()
    engine = create_engine("sqlite://")
    module.metadata.create_all(engine)
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    fence_path = Path("alembic/versions/0012_context_execution_fence.py")
    fence_spec = importlib.util.spec_from_file_location("fence_migration", fence_path)
    fence_module = importlib.util.module_from_spec(fence_spec)
    fence_spec.loader.exec_module(fence_module)
    assert fence_module.down_revision == "0011_context_release"
    assert "agentic_rag" not in fence_path.read_text()
    with engine.begin() as connection:
        fence_module.op = Operations(MigrationContext.configure(connection))
        fence_module.upgrade()
    assert set(inspect(engine).get_table_names()) == {table.name for table in tables}
    for table in tables:
        assert set(table.c.keys()) == {
            c["name"] for c in inspect(engine).get_columns(table.name)
        }
    with engine.begin() as connection:
        fence_module.op = Operations(MigrationContext.configure(connection))
        fence_module.downgrade()
    assert "execution_token" not in {
        c["name"] for c in inspect(engine).get_columns("context_release_gate")
    }
    module.metadata.drop_all(engine)


def test_0010_through_0012_and_armed_upgrade_refusal_before_any_ddl():
    import pytest
    from sqlalchemy import select
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    modules = []
    for revision in (
        "0010_context_builds",
        "0011_context_release",
        "0012_context_execution_fence",
    ):
        spec = importlib.util.spec_from_file_location(
            revision, Path(f"alembic/versions/{revision}.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    for armed in (False, True):
        engine = create_engine("sqlite://")
        with engine.begin() as connection:
            for module in modules[:2]:
                module.op = Operations(MigrationContext.configure(connection))
                module.upgrade()
            gate = modules[1].context_gate
            if armed:
                connection.execute(
                    gate.insert().values(
                        id=1,
                        state="frozen",
                        generation="old-v1",
                        config_hash="a" * 64,
                        epoch=0,
                        roles=["api"],
                        inventory=[],
                        receipt_hash="b" * 64,
                        maintenance_receipt={},
                        receipts={},
                    )
                )
            before = {
                name: inspect(connection).get_columns(name)
                for name in inspect(connection).get_table_names()
            }
            before_rows = connection.execute(select(gate)).mappings().all()
            modules[2].op = Operations(MigrationContext.configure(connection))
            if armed:
                with pytest.raises(RuntimeError, match="unarmed"):
                    modules[2].upgrade()
                after = {
                    name: inspect(connection).get_columns(name)
                    for name in inspect(connection).get_table_names()
                }
                assert {
                    k: [str(v) for v in values] for k, values in before.items()
                } == {k: [str(v) for v in values] for k, values in after.items()}
                assert connection.execute(select(gate)).mappings().all() == before_rows
            else:
                modules[2].upgrade()
                assert "execution_token" in {
                    c["name"] for c in inspect(connection).get_columns(gate.name)
                }
        engine.dispose()
