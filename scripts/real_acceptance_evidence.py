"""Service-aware evidence helpers for real Query acceptance only.

Unlike the hermetic local drill, this module backs up and restores the exact
isolated MySQL database, Redis namespace, Elasticsearch generation, Artifact
root, and Query checkpoint used by a real provider/API Run.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import sqlite3
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import make_url

from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.mysql import create_mysql_engine
from agentic_rag.domain.models import RunStatus, UserScope
from agentic_rag.runtime.query_worker import QUERY_STREAM
from scripts.backup_local import BackupSpec, create_backup
from scripts.restore_local import ServiceRestoreSpec, restore_backup


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_DATABASE = re.compile(r"agentic_rag_acceptance_(?:source|restore)_[a-f0-9]{8,32}")


class RealAcceptanceEvidenceError(RuntimeError):
    """A configured real recovery/backup evidence operation failed closed."""


def service_backup_configuration_issue() -> str | None:
    """Return an explicit unavailable reason without falling back to app DSNs."""
    if os.getenv("AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE") != "1":
        return "set AGENTIC_RAG_RUN_REAL_BACKUP_RESTORE=1 for service-aware backup/restore"
    admin_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN", "").strip()
    if not admin_dsn:
        return "missing AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN for isolated MySQL resources"
    try:
        url = make_url(admin_dsn)
    except ValueError:
        return "AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN is invalid"
    if url.drivername != "mysql+asyncmy" or url.host not in _LOCAL_HOSTS:
        return "AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN must be a loopback mysql+asyncmy DSN"
    return None


def require_service_backup_admin_dsn() -> str:
    issue = service_backup_configuration_issue()
    if issue is not None:
        raise RealAcceptanceEvidenceError(f"real backup/restore unavailable: {issue}")
    return os.environ["AGENTIC_RAG_TEST_MYSQL_ADMIN_DSN"].strip()


def isolated_mysql_dsn(admin_dsn: str, database: str) -> str:
    _validate_database(database)
    return make_url(admin_dsn).set(database=database).render_as_string(
        hide_password=False
    )


def create_isolated_mysql_database(admin_dsn: str, database: str) -> None:
    _mysql_admin(admin_dsn, f"CREATE DATABASE `{database}`")


def drop_isolated_mysql_database(admin_dsn: str, database: str) -> None:
    _mysql_admin(admin_dsn, f"DROP DATABASE IF EXISTS `{database}`")


@dataclass(frozen=True, slots=True)
class ServiceBackupResources:
    working_root: Path
    artifact_root: Path
    query_checkpoint_path: Path
    source_mysql_dsn: str
    source_mysql_database: str
    admin_mysql_dsn: str
    redis_dsn: str
    source_redis_prefix: str
    elasticsearch_url: str
    source_index_generation: str
    source_run_id: str
    checkpoint_thread_id: str
    parent_id: str
    runtime_config_snapshot_id: str
    app_version: str = "0.1.0"
    schema_generation: str = "head"


async def exercise_query_worker_recovery(
    *,
    client: Any,
    container: Any,
    broker: Any,
    user_id: str,
    snapshot_id: str,
    parent_id: str,
    stop_worker: Callable[[], Awaitable[None]],
    start_worker: Callable[[], Awaitable[None]],
    timeout_seconds: float = 180.0,
) -> tuple[dict[str, object], str]:
    """Restart the current worker around a real API Run and prove idempotency."""
    await stop_worker()
    thread_id = f"recovery-{uuid4().hex}"
    created = await client.post(
        "/v1/query-runs",
        json={
            "query": "How many days notice does the seeded document require?",
            "thread_id": thread_id,
            "wait_seconds": 0,
        },
    )
    if created.status_code != 202:
        raise RealAcceptanceEvidenceError(
            f"recovery Query API create failed with HTTP {created.status_code}"
        )
    payload = created.json()
    run_id = payload.get("run_id") if isinstance(payload, Mapping) else None
    if not isinstance(run_id, str) or not run_id:
        raise RealAcceptanceEvidenceError("recovery Query API omitted a Run ID")
    await broker.publish(QUERY_STREAM, run_id, _utc_now(), dedupe_key=None)
    await start_worker()
    terminal = await _wait_for_terminal(client, run_id, timeout_seconds)
    answer = terminal.get("answer")
    parent_ids = answer.get("evidence_parent_ids") if isinstance(answer, Mapping) else None
    if (
        terminal.get("status") != RunStatus.COMPLETED.value
        or terminal.get("runtime_config_snapshot_id") != snapshot_id
        or not isinstance(answer, Mapping)
        or answer.get("audited") is not True
        or not isinstance(parent_ids, list)
        or parent_id not in parent_ids
    ):
        raise RealAcceptanceEvidenceError(
            "recovery Run did not finish with the audited source evidence"
        )
    duplicate_message_id = await broker.publish(
        QUERY_STREAM, run_id, _utc_now(), dedupe_key=None
    )
    duplicate_acked = await _wait_for_duplicate_ack(
        container.redis,
        broker.query_stream,
        broker.query_group,
        duplicate_message_id,
    )
    events = await container.event_repository.list_after(
        run_id,
        UserScope(user_id=user_id),
        0,
        200,
    )
    terminal_event_count = sum(
        event.event_type in {"RUN_COMPLETED", "RUN_FAILED", "RUN_CANCELLED"}
        for event in events
    )
    if not duplicate_acked or terminal_event_count != 1:
        raise RealAcceptanceEvidenceError(
            "recovery duplicate delivery was not ACKed idempotently"
        )
    return (
        {
            "provider_e2e": True,
            "source_run_id": run_id,
            "runtime_config_snapshot_id": snapshot_id,
            "worker_restarted": True,
            "duplicate_delivery_injected": True,
            "duplicate_delivery_acked": True,
            "terminal_status": RunStatus.COMPLETED.value,
            "terminal_event_count": terminal_event_count,
            "final_answer_audited": True,
        },
        f"query:{user_id}:{thread_id}",
    )


async def run_service_backup_restore(
    resources: ServiceBackupResources,
) -> dict[str, object]:
    """Backup, restore, query, and clean a distinct set of service targets."""
    _validate_resources(resources)
    suffix = uuid4().hex[:12]
    restored_database = f"agentic_rag_acceptance_restore_{suffix}"
    restored_mysql_dsn = isolated_mysql_dsn(
        resources.admin_mysql_dsn, restored_database
    )
    restored_generation = f"real-query-restore-{suffix}"
    restored_redis_prefix = f"agenticrag:e2e:restore-{suffix}"
    restored_index = f"agenticrag-children-{restored_generation}"
    marker_name = hashlib.sha256(resources.source_run_id.encode()).hexdigest()
    marker_ref = LocalArtifactStore(resources.artifact_root).put_json(
        f"acceptance/backup/{marker_name}.json",
        {
            "run_id": resources.source_run_id,
            "runtime_config_snapshot_id": resources.runtime_config_snapshot_id,
        },
    )
    backup = resources.working_root / f"service-backup-{suffix}"
    restored = resources.working_root / f"service-restore-{suffix}"
    database_created = False
    body_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    try:
        await asyncio.to_thread(
            create_isolated_mysql_database,
            resources.admin_mysql_dsn,
            restored_database,
        )
        database_created = True
        await asyncio.to_thread(
            create_backup,
            BackupSpec(
                output=backup,
                artifact_root=resources.artifact_root,
                checkpoint_paths=(resources.query_checkpoint_path,),
                app_version=resources.app_version,
                schema_generation=resources.schema_generation,
                index_generation=resources.source_index_generation,
                mysql_dsn=resources.source_mysql_dsn,
                elasticsearch_url=resources.elasticsearch_url,
                redis_dsn=resources.redis_dsn,
                redis_key_prefix=resources.source_redis_prefix,
                elasticsearch_include_global_metadata=False,
            ),
        )
        await asyncio.to_thread(
            restore_backup,
            backup,
            restored,
            services=ServiceRestoreSpec(
                mysql_dsn=restored_mysql_dsn,
                elasticsearch_url=resources.elasticsearch_url,
                index_generation=restored_generation,
                redis_dsn=resources.redis_dsn,
                redis_key_prefix=restored_redis_prefix,
            ),
        )
        await _verify_mysql_restore(restored_mysql_dsn, resources)
        await _verify_elasticsearch_restore(
            resources.elasticsearch_url,
            restored_index,
            resources.parent_id,
        )
        await _verify_redis_restore(
            resources.redis_dsn,
            restored_redis_prefix,
            resources.source_redis_prefix,
            resources.source_run_id,
        )
        _verify_artifact_restore(restored, marker_ref.uri, resources)
        _verify_checkpoint_restore(restored, resources)
        return {
            "provider_e2e": True,
            "source_run_id": resources.source_run_id,
            "runtime_config_snapshot_id": resources.runtime_config_snapshot_id,
            "source_scope": {
                "mysql_database": resources.source_mysql_database,
                "redis_prefix": resources.source_redis_prefix,
                "index_generation": resources.source_index_generation,
                "checkpoint_thread_id": resources.checkpoint_thread_id,
                "artifact_marker": marker_ref.uri,
            },
            "restore_scope": {
                "mysql_database": restored_database,
                "redis_prefix": restored_redis_prefix,
                "index_generation": restored_generation,
            },
            "services": {
                "mysql": True,
                "redis": True,
                "elasticsearch": True,
                "artifacts": True,
                "checkpoints": True,
            },
        }
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            await _delete_redis_prefix(resources.redis_dsn, restored_redis_prefix)
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            await _delete_elasticsearch_index(resources.elasticsearch_url, restored_index)
        except BaseException as error:
            cleanup_errors.append(error)
        if database_created:
            try:
                await asyncio.to_thread(
                    drop_isolated_mysql_database,
                    resources.admin_mysql_dsn,
                    restored_database,
                )
            except BaseException as error:
                cleanup_errors.append(error)
        if body_error is None and cleanup_errors:
            raise RealAcceptanceEvidenceError(
                "service-aware restore target cleanup failed: "
                + ", ".join(type(error).__name__ for error in cleanup_errors)
            ) from cleanup_errors[0]


def _validate_resources(resources: ServiceBackupResources) -> None:
    _validate_database(resources.source_mysql_database)
    source_url = make_url(resources.source_mysql_dsn)
    if source_url.database != resources.source_mysql_database:
        raise RealAcceptanceEvidenceError("source MySQL DSN does not match its isolated database")
    if make_url(resources.admin_mysql_dsn).host not in _LOCAL_HOSTS:
        raise RealAcceptanceEvidenceError("MySQL admin DSN must remain loopback-only")
    if urlparse(resources.redis_dsn).hostname not in _LOCAL_HOSTS:
        raise RealAcceptanceEvidenceError("Redis restore requires a loopback test DSN")
    if urlparse(resources.elasticsearch_url).hostname not in _LOCAL_HOSTS:
        raise RealAcceptanceEvidenceError("Elasticsearch restore requires a loopback test URL")
    if not resources.source_redis_prefix.startswith("agenticrag:e2e:"):
        raise RealAcceptanceEvidenceError("Redis backup prefix is not acceptance-isolated")
    if not resources.source_index_generation.startswith("real-query-"):
        raise RealAcceptanceEvidenceError("Elasticsearch generation is not acceptance-isolated")
    if not resources.checkpoint_thread_id.startswith("query:"):
        raise RealAcceptanceEvidenceError("Query checkpoint thread is invalid")
    if not resources.query_checkpoint_path.is_file():
        raise RealAcceptanceEvidenceError("real Query checkpoint is unavailable for backup")


async def _wait_for_terminal(
    client: Any, run_id: str, timeout_seconds: float
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/v1/query-runs/{run_id}")
        if response.status_code != 200:
            raise RealAcceptanceEvidenceError(
                f"recovery Run lookup failed with HTTP {response.status_code}"
            )
        payload = response.json()
        if isinstance(payload, Mapping) and payload.get("status") in {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        }:
            return dict(payload)
        await asyncio.sleep(0.1)
    raise RealAcceptanceEvidenceError("recovery Run did not become terminal")


async def _wait_for_duplicate_ack(
    redis: Any,
    stream: str,
    group: str,
    message_id: str,
    timeout_seconds: float = 10.0,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        groups = await redis.xinfo_groups(stream)
        for item in groups:
            name = _redis_text(_redis_field(item, "name"))
            if name != group:
                continue
            last_delivered = _redis_text(_redis_field(item, "last-delivered-id"))
            pending = _redis_field(item, "pending")
            if last_delivered == message_id and pending == 0:
                return True
        await asyncio.sleep(0.1)
    return False


def _redis_field(value: Mapping[object, object], name: str) -> object:
    return value.get(name, value.get(name.encode()))


def _redis_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _utc_now() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)


async def _verify_mysql_restore(
    dsn: str, resources: ServiceBackupResources
) -> None:
    engine = create_mysql_engine(dsn)
    try:
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    text(
                        "SELECT runtime_config_snapshot_id, checkpoint_thread_id "
                        "FROM agent_runs WHERE id = :run_id"
                    ),
                    {"run_id": resources.source_run_id},
                )
            ).one_or_none()
        if row != (
            resources.runtime_config_snapshot_id,
            resources.checkpoint_thread_id,
        ):
            raise RealAcceptanceEvidenceError("restored MySQL omitted the source Query Run")
    finally:
        await engine.dispose()


async def _verify_elasticsearch_restore(
    endpoint: str, index: str, parent_id: str
) -> None:
    from elasticsearch import AsyncElasticsearch

    client = AsyncElasticsearch(endpoint)
    try:
        response = await client.count(
            index=index,
            query={"term": {"parent_id": parent_id}},
        )
        if int(response.body.get("count", 0)) < 1:
            raise RealAcceptanceEvidenceError(
                "restored Elasticsearch omitted the source evidence parent"
            )
    finally:
        await client.close()


async def _verify_redis_restore(
    endpoint: str,
    restored_prefix: str,
    source_prefix: str,
    run_id: str,
) -> None:
    from redis.asyncio import Redis

    client = Redis.from_url(endpoint, decode_responses=False)
    source_stream = f"{source_prefix}:query"
    restored_stream = f"{restored_prefix}{source_stream.removeprefix(source_prefix)}"
    try:
        rows = await client.xrange(restored_stream)
        aggregate_ids = {
            fields.get(b"aggregate_id", fields.get("aggregate_id"))
            for _message_id, fields in rows
        }
        if run_id.encode() not in aggregate_ids and run_id not in aggregate_ids:
            raise RealAcceptanceEvidenceError(
                "restored Redis stream omitted the source Query Run delivery"
            )
    finally:
        await client.aclose()


def _verify_artifact_restore(
    restored: Path, marker_uri: str, resources: ServiceBackupResources
) -> None:
    store = LocalArtifactStore(restored / "artifacts")
    payload = store.read_json(store.describe(marker_uri))
    if payload != {
        "run_id": resources.source_run_id,
        "runtime_config_snapshot_id": resources.runtime_config_snapshot_id,
    }:
        raise RealAcceptanceEvidenceError("restored Artifact marker did not verify")


def _verify_checkpoint_restore(
    restored: Path, resources: ServiceBackupResources
) -> None:
    checkpoint = restored / "checkpoints" / resources.query_checkpoint_path.name
    try:
        with sqlite3.connect(checkpoint) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            count = connection.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (resources.checkpoint_thread_id,),
            ).fetchone()
    except sqlite3.Error as error:
        raise RealAcceptanceEvidenceError("restored Query checkpoint is not queryable") from error
    if integrity != ("ok",) or count is None or count[0] < 1:
        raise RealAcceptanceEvidenceError("restored Query checkpoint omitted the source Run")


async def _delete_redis_prefix(endpoint: str, prefix: str) -> None:
    from redis.asyncio import Redis

    client = Redis.from_url(endpoint, decode_responses=False)
    try:
        keys = [key async for key in client.scan_iter(match=f"{prefix}*")]
        if keys:
            await client.delete(*keys)
    finally:
        await client.aclose()


async def _delete_elasticsearch_index(endpoint: str, index: str) -> None:
    from elasticsearch import AsyncElasticsearch

    client = AsyncElasticsearch(endpoint)
    try:
        await client.indices.delete(index=index, ignore_unavailable=True)
    finally:
        await client.close()


def _mysql_admin(dsn: str, statement: str) -> None:
    url = make_url(dsn)
    database_match = re.search(r"`([^`]+)`", statement)
    if database_match is None:
        raise RealAcceptanceEvidenceError("MySQL admin statement omitted a database")
    _validate_database(database_match.group(1))
    if not statement.startswith(("CREATE DATABASE `", "DROP DATABASE IF EXISTS `")):
        raise RealAcceptanceEvidenceError("MySQL admin operation is not allowlisted")
    environment = dict(os.environ)
    if url.password is not None:
        environment["MYSQL_PWD"] = url.password
    try:
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
    except (OSError, subprocess.CalledProcessError) as error:
        raise RealAcceptanceEvidenceError("isolated MySQL database operation failed") from error


def _validate_database(database: str) -> None:
    if _DATABASE.fullmatch(database) is None:
        raise RealAcceptanceEvidenceError("MySQL database is not an isolated acceptance name")


__all__ = [
    "RealAcceptanceEvidenceError",
    "ServiceBackupResources",
    "create_isolated_mysql_database",
    "drop_isolated_mysql_database",
    "exercise_query_worker_recovery",
    "isolated_mysql_dsn",
    "require_service_backup_admin_dsn",
    "run_service_backup_restore",
    "service_backup_configuration_issue",
]
