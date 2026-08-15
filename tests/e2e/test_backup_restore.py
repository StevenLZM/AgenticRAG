"""Deterministic local backup and restore contracts."""

from __future__ import annotations

import json
import asyncio
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import make_url

from agentic_rag.persistence.mysql import create_mysql_engine
from scripts.backup_local import (
    BackupError,
    BackupSpec,
    _validate_redis_key_prefix,
    create_backup,
)
from scripts.restore_local import RestoreError, ServiceRestoreSpec, restore_backup
from scripts.verify_acceptance import verify_acceptance
import scripts.restore_local as restore_module
from agentic_rag.persistence.artifacts import LocalArtifactStore


pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("prefix", ["agentic:*", "agentic?", "agentic[0]", "agentic\\"])
def test_redis_backup_prefix_cannot_expand_scan_namespace(prefix: str) -> None:
    with pytest.raises(BackupError):
        _validate_redis_key_prefix(prefix)


def _seed_state(root: Path) -> tuple[Path, Path, Path]:
    artifacts = root / "artifacts"
    sqlite_path = root / "query.sqlite"
    store = LocalArtifactStore(artifacts)
    store.put_json(
        "documents/doc-1/manifest.json",
        {"document_id": "doc-1", "active": True, "citation_coverage": 1.0},
    )
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute("CREATE TABLE answers (answer TEXT NOT NULL)")
        connection.execute("INSERT INTO answers VALUES ('known answer')")
    return artifacts, sqlite_path, root / "ingestion.sqlite"


def test_backup_restores_hash_checked_queryable_local_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifacts, query_checkpoint, ingestion_checkpoint = _seed_state(source)
    backup = tmp_path / "backup"

    create_backup(
        BackupSpec(
            output=backup,
            artifact_root=artifacts,
            checkpoint_paths=(query_checkpoint, ingestion_checkpoint),
            app_version="test",
            schema_generation="test-schema",
            index_generation="e2e-generation",
        )
    )

    restored = tmp_path / "restored"
    restore_backup(backup, restored)

    restored_store = LocalArtifactStore(restored / "artifacts")
    ref = restored_store.describe("artifact://documents/doc-1/manifest.json")
    assert restored_store.read_json(ref)["citation_coverage"] == 1.0
    with sqlite3.connect(restored / "checkpoints" / "query.sqlite") as connection:
        assert connection.execute("SELECT answer FROM answers").fetchone() == ("known answer",)


def test_restore_rejects_a_tampered_backup_before_creating_target(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifacts, query_checkpoint, ingestion_checkpoint = _seed_state(source)
    backup = tmp_path / "backup"
    create_backup(
        BackupSpec(
            output=backup,
            artifact_root=artifacts,
            checkpoint_paths=(query_checkpoint, ingestion_checkpoint),
            app_version="test",
            schema_generation="test-schema",
            index_generation="e2e-generation",
        )
    )
    (backup / "artifacts" / "documents" / "doc-1" / "manifest.json").write_text(
        '{"tampered":true}', encoding="utf-8"
    )

    target = tmp_path / "must-not-exist"
    with pytest.raises(RestoreError, match="integrity"):
        restore_backup(backup, target)
    assert not target.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("user_leak_count", 1),
        ("citation_coverage", 0.99),
        ("unaudited_answer_count", 1),
        ("recovery_drill_passed", False),
        ("backup_restore_passed", False),
    ],
)
def test_acceptance_fails_on_any_hard_gate_violation(field: str, value: object) -> None:
    summary: dict[str, object] = {
        "user_leak_count": 0,
        "citation_coverage": 1.0,
        "unaudited_answer_count": 0,
        "recovery_drill_passed": True,
        "backup_restore_passed": True,
    }
    summary[field] = value

    assert verify_acceptance(summary) == 1


def test_acceptance_rejects_missing_or_non_json_hard_gate_values() -> None:
    assert verify_acceptance({}) == 1
    assert verify_acceptance(json.loads('{"citation_coverage":"1.0"}')) == 1


async def test_mysql_restore_uses_subprocesses_inside_an_async_test_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch a nested ``asyncio.run`` regression in service restore tooling."""
    dump = tmp_path / "dump.sql"
    dump.write_text("-- isolated test dump\n", encoding="utf-8")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=b"")

    monkeypatch.setattr(restore_module.subprocess, "run", fake_run)

    restore_module._restore_mysql(
        dump,
        "mysql+asyncmy://tester:password@127.0.0.1:3306/agentic_rag_restore_test",
        "mysql",
    )

    assert any("SHOW TABLES" in args for args, _kwargs in calls)
    assert any(args[1:4] == ["-m", "alembic", "upgrade"] for args, _kwargs in calls)


async def test_restore_async_adapter_can_run_while_a_test_loop_is_active() -> None:
    async def available() -> dict[str, str]:
        return {"mysql": "available"}

    assert restore_module._run_awaitable(available()) == {"mysql": "available"}


async def test_opt_in_real_services_restore_only_disposable_database_index_and_redis_prefix(
    tmp_path: Path,
) -> None:
    """Exercise actual MySQL/ES tooling without touching configured app state."""
    if os.getenv("AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE") != "1":
        pytest.skip(
            "set AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1 to use a generated isolated "
            "MySQL database, Elasticsearch index, and Redis key prefix"
        )
    admin_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN")
    redis_dsn = os.getenv("AGENTIC_RAG_TEST_REDIS_DSN")
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL")
    if not admin_dsn or not redis_dsn or not elasticsearch_url:
        pytest.skip(
            "missing AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN, AGENTIC_RAG_TEST_REDIS_DSN, "
            "or AGENTIC_RAG_TEST_ELASTICSEARCH_URL for isolated real backup restore"
        )
    _assert_loopback_services(admin_dsn, redis_dsn, elasticsearch_url)
    suffix = uuid4().hex
    source_database = f"agentic_rag_backup_{suffix}"
    restored_database = f"agentic_rag_restore_{suffix}"
    source_dsn = make_url(admin_dsn).set(database=source_database).render_as_string(
        hide_password=False
    )
    restored_dsn = make_url(admin_dsn).set(database=restored_database).render_as_string(
        hide_password=False
    )
    source_generation = f"e2e-backup-{suffix}"
    restored_generation = f"e2e-restore-{suffix}"
    source_index = f"agenticrag-children-{source_generation}"
    restored_index = f"agenticrag-children-{restored_generation}"
    alias = f"agenticrag-active-{suffix}"
    redis_prefix = f"agentic-rag:e2e-backup:{suffix}"
    restored_redis_prefix = f"agentic-rag:e2e-restore:{suffix}"
    redis = Redis.from_url(redis_dsn)
    from elasticsearch import AsyncElasticsearch

    elasticsearch = AsyncElasticsearch(elasticsearch_url)
    source_exists = False
    restored_exists = False
    try:
        _mysql_admin(admin_dsn, f"CREATE DATABASE `{source_database}`")
        _mysql_admin(admin_dsn, f"CREATE DATABASE `{restored_database}`")
        source_exists = restored_exists = True
        await asyncio.to_thread(_wait_for_mysql_cli, source_dsn)
        await asyncio.to_thread(_alembic_upgrade, source_dsn)
        await _wait_for_asyncmy(source_dsn)
        source_engine = create_mysql_engine(source_dsn)
        try:
            async with source_engine.begin() as connection:
                await connection.execute(text("CREATE TABLE backup_marker (value VARCHAR(64))"))
                await connection.execute(text("INSERT INTO backup_marker VALUES ('known-answer')"))
        finally:
            await source_engine.dispose()
        await elasticsearch.indices.create(
            index=source_index,
            mappings={"properties": {"marker": {"type": "keyword"}}},
        )
        await elasticsearch.index(index=source_index, id="known-answer", document={"marker": "known-answer"}, refresh="wait_for")
        await elasticsearch.indices.put_alias(index=source_index, name=alias)
        await redis.set(f"{redis_prefix}:probe", "isolated")

        backup = tmp_path / "services-backup"
        await asyncio.to_thread(
            create_backup,
            BackupSpec(
                output=backup,
                artifact_root=tmp_path / "source-artifacts",
                checkpoint_paths=(),
                app_version="test",
                schema_generation="head",
                index_generation=source_generation,
                mysql_dsn=source_dsn,
                elasticsearch_url=elasticsearch_url,
                redis_dsn=redis_dsn,
                redis_key_prefix=redis_prefix,
            ),
        )

        _mysql_admin(admin_dsn, f"DROP DATABASE `{source_database}`")
        source_exists = False
        await elasticsearch.indices.delete(index=source_index)
        await redis.delete(f"{redis_prefix}:probe")

        await asyncio.to_thread(
            restore_backup,
            backup,
            tmp_path / "services-restored",
            services=ServiceRestoreSpec(
                mysql_dsn=restored_dsn,
                elasticsearch_url=elasticsearch_url,
                index_generation=restored_generation,
                redis_dsn=redis_dsn,
                redis_key_prefix=restored_redis_prefix,
            ),
        )
        restored_engine = create_mysql_engine(restored_dsn)
        try:
            async with restored_engine.connect() as connection:
                assert (await connection.execute(text("SELECT value FROM backup_marker"))).scalar_one() == "known-answer"
        finally:
            await restored_engine.dispose()
        assert (await elasticsearch.count(index=restored_index, query={"match_all": {}})).body["count"] == 1
        assert restored_index in (await elasticsearch.indices.get_alias(name=alias)).body
        assert await redis.get(f"{restored_redis_prefix}:probe") == b"isolated"
    finally:
        await elasticsearch.indices.delete(index=source_index, ignore_unavailable=True)
        await elasticsearch.indices.delete(index=restored_index, ignore_unavailable=True)
        await redis.delete(f"{redis_prefix}:probe")
        await redis.delete(f"{restored_redis_prefix}:probe")
        if source_exists:
            _mysql_admin(admin_dsn, f"DROP DATABASE `{source_database}`")
        if restored_exists:
            _mysql_admin(admin_dsn, f"DROP DATABASE `{restored_database}`")
        await elasticsearch.close()
        await redis.aclose()



def _assert_loopback_services(mysql_dsn: str, redis_dsn: str, elasticsearch_url: str) -> None:
    mysql = make_url(mysql_dsn)
    endpoints = (mysql.host, urlparse(redis_dsn).hostname, urlparse(elasticsearch_url).hostname)
    if any(host not in {"127.0.0.1", "localhost", "::1"} for host in endpoints):
        pytest.fail("real backup test refuses non-local MySQL, Redis, or Elasticsearch endpoints")


def _mysql_admin(dsn: str, statement: str) -> None:
    """Use the configured TCP CLI account only for generated test DB names."""
    url = make_url(dsn)
    if not statement.startswith(("CREATE DATABASE `agentic_rag_", "DROP DATABASE `agentic_rag_")):
        raise AssertionError("test helper refuses an unscoped MySQL admin statement")
    environment = dict(os.environ)
    if url.password is not None:
        environment["MYSQL_PWD"] = url.password
    subprocess.run(
        [
            "mysql",
            "--protocol=TCP",
            f"--host={url.host or 'localhost'}",
            f"--port={url.port or 3306}",
            f"--user={url.username or ''}",
            "--execute",
            statement,
        ],
        check=True,
        capture_output=True,
        env=environment,
    )


def _wait_for_mysql_cli(dsn: str, *, timeout_seconds: float = 10.0) -> None:
    """Wait for a just-created disposable DB to accept authenticated TCP work."""
    url = make_url(dsn)
    environment = dict(os.environ)
    if url.password is not None:
        environment["MYSQL_PWD"] = url.password
    deadline = time.monotonic() + timeout_seconds
    last_error: subprocess.CalledProcessError | OSError | None = None
    while time.monotonic() < deadline:
        try:
            subprocess.run(
                [
                    "mysql",
                    "--protocol=TCP",
                    f"--host={url.host or 'localhost'}",
                        f"--port={url.port or 3306}",
                        f"--user={url.username or ''}",
                        url.database or "",
                        "--execute",
                        "SELECT 1",
                    ],
                check=True,
                capture_output=True,
                env=environment,
            )
            return
        except (OSError, subprocess.CalledProcessError) as error:
            last_error = error
            time.sleep(0.2)
    raise RuntimeError("generated MySQL test database did not become ready") from last_error


def _alembic_upgrade(dsn: str) -> None:
    environment = dict(os.environ)
    environment["AGENTIC_RAG_MYSQL_DSN"] = dsn
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
        capture_output=True,
        env=environment,
    )


async def _wait_for_asyncmy(dsn: str, *, timeout_seconds: float = 10.0) -> None:
    """Avoid racing the first asyncmy connection after test-DB creation."""
    deadline = time.monotonic() + timeout_seconds
    last_error: OperationalError | None = None
    while time.monotonic() < deadline:
        engine = create_mysql_engine(dsn)
        try:
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            return
        except OperationalError as error:
            last_error = error
            await asyncio.sleep(0.2)
        finally:
            await engine.dispose()
    raise RuntimeError("asyncmy did not authenticate against generated MySQL test database") from last_error
