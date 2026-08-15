"""Fail-closed restore for backups written by :mod:`scripts.backup_local`."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


class RestoreError(RuntimeError):
    """Raised when a backup cannot be fully verified and restored."""


ReadinessCheck = Callable[[], Awaitable[Mapping[str, str]] | Mapping[str, str]]


@dataclass(frozen=True)
class ServiceRestoreSpec:
    """Explicit empty service targets; omitted services are never touched."""

    mysql_dsn: str | None = None
    elasticsearch_url: str | None = None
    index_generation: str | None = None
    redis_dsn: str | None = None
    redis_key_prefix: str | None = None
    mysql_command: str = "mysql"


def restore_backup(
    backup: Path,
    target: Path,
    *,
    readiness_check: ReadinessCheck | None = None,
    services: ServiceRestoreSpec | None = None,
) -> Path:
    """Restore a verified backup to a previously non-existent target directory.

    Publishing is one ``os.replace`` operation.  Existing targets are rejected
    so a typo cannot overwrite working data, even when it happens to be empty.
    """
    backup = Path(backup)
    target = Path(target)
    if target.exists():
        raise RestoreError(f"restore target must not exist: {target}")
    manifest = _verify_backup(backup)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Keep the no-overwrite guarantee across the slow copy/service phase.
        # ``os.replace`` below replaces this reserved empty directory with the
        # fully verified staging tree in one filesystem operation.
        target.mkdir()
    except FileExistsError as error:
        raise RestoreError(f"restore target must not exist: {target}") from error
    reserved_target = True
    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore-", dir=target.parent))
        for entry in manifest["files"]:
            relative = _safe_relative(entry["path"])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(backup / relative, destination, follow_symlinks=False)
        _validate_restored_artifacts(staging / "artifacts")
        _restore_services(staging, manifest, services)
        _run_readiness(
            readiness_check
            if readiness_check is not None
            else _default_readiness_check(staging, services)
        )
        _fsync_tree(staging)
        assert staging is not None
        os.replace(staging, target)
        reserved_target = False
        _fsync_directory(target.parent)
        return target
    except Exception as error:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if reserved_target:
            _remove_empty_directory(target)
        if isinstance(error, RestoreError):
            raise
        raise RestoreError(f"restore failed: {error}") from error


def _restore_services(
    staging: Path, manifest: Mapping[str, Any], services: ServiceRestoreSpec | None
) -> None:
    if not isinstance(manifest.get("services", {}), Mapping):
        raise RestoreError("backup service inventory is invalid")
    mysql_dump = staging / "mysql" / "dump.sql"
    es_export = staging / "elasticsearch" / "export.json"
    if mysql_dump.exists():
        if services is None or services.mysql_dsn is None:
            raise RestoreError("backup contains MySQL state; an explicit empty MySQL target is required")
        _restore_mysql(mysql_dump, services.mysql_dsn, services.mysql_command)
    if es_export.exists():
        if services is None or not services.elasticsearch_url or not services.index_generation:
            raise RestoreError(
                "backup contains Elasticsearch state; an explicit empty index generation is required"
            )
        _run_async(
            _restore_elasticsearch(
                es_export,
                services.elasticsearch_url,
                services.index_generation,
                str(manifest["index_generation"]),
            )
        )
    redis_export = staging / "redis" / "export.json"
    if redis_export.exists():
        if services is None or not services.redis_dsn or not services.redis_key_prefix:
            raise RestoreError(
                "backup contains Redis state; an explicit target DSN and key prefix are required"
            )
        _run_async(
            _restore_redis(
                redis_export,
                services.redis_dsn,
                services.redis_key_prefix,
            )
        )


def _restore_mysql(dump: Path, dsn: str, command: str) -> None:
    """Import only into a verified-empty DB, then bring it to Alembic head."""
    url = make_url(dsn)
    if url.drivername != "mysql+asyncmy" or not url.database:
        raise RestoreError("MySQL restore requires a mysql+asyncmy DSN with an empty database")
    _assert_mysql_database_name(url.database)
    _assert_empty_mysql(dsn, command)
    args = [*_mysql_command(url, command), url.database]
    environment = dict(os.environ)
    if url.password is not None:
        environment["MYSQL_PWD"] = url.password
    try:
        subprocess.run(
            args,
            input=dump.read_bytes(),
            check=True,
            env=environment,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RestoreError("MySQL restore import failed") from error
    _upgrade_mysql(dsn)


def _assert_empty_mysql(dsn: str, command: str) -> None:
    """Check the explicit target with the same TCP CLI used for restoration."""
    url = make_url(dsn)
    environment = _mysql_environment(url)
    try:
        probe = subprocess.run(
            [
                *_mysql_command(url, command),
                url.database or "",
                "--batch",
                "--skip-column-names",
                "--execute",
                "SHOW TABLES",
            ],
            check=True,
            env=environment,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RestoreError("MySQL restore target cannot be checked") from error
    if probe.stdout.strip():
        raise RestoreError("MySQL restore target is not empty")


def _upgrade_mysql(dsn: str) -> None:
    environment = dict(os.environ)
    environment["AGENTIC_RAG_MYSQL_DSN"] = dsn
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SOURCE_ROOT), existing_pythonpath) if path
    )
    try:
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            check=True,
            env=environment,
            cwd=PROJECT_ROOT,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RestoreError("Alembic migration after MySQL restore failed") from error


def _mysql_command(url: Any, command: str) -> list[str]:
    _assert_mysql_database_name(url.database)
    return [
        command,
        "--protocol=TCP",
        f"--host={url.host or 'localhost'}",
        f"--port={url.port or 3306}",
        f"--user={url.username or ''}",
    ]


def _assert_mysql_database_name(value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_]{1,64}", value) is None:
        raise RestoreError("MySQL database name contains unsafe characters")


def _mysql_environment(url: Any) -> dict[str, str]:
    environment = dict(os.environ)
    if url.password is not None:
        environment["MYSQL_PWD"] = url.password
    return environment


def _run_async(value: Any) -> Any:
    """Run a coroutine from synchronous restore code, even in an active loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)

    result: list[Any] = []
    failure: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as error:  # pragma: no cover - re-raised below
            failure.append(error)

    thread = threading.Thread(target=runner, name="agentic-rag-restore-async")
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return result[0] if result else None


def _run_awaitable(value: Any) -> Any:
    """Compatibility name used by operational smoke tests."""
    return _run_async(value)


async def _restore_elasticsearch(
    export_path: Path,
    endpoint: str,
    target_generation: str,
    source_generation: str,
) -> None:
    from elasticsearch import AsyncElasticsearch

    from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore

    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RestoreError("Elasticsearch export is invalid") from error
    if not isinstance(payload, Mapping):
        raise RestoreError("Elasticsearch export is invalid")
    source_index = payload.get("index")
    definition = payload.get("index_definition")
    documents = payload.get("documents")
    expected_source_index = f"agenticrag-children-{source_generation}"
    if (
        not isinstance(source_index, str)
        or source_index != expected_source_index
        or not isinstance(definition, Mapping)
        or not isinstance(documents, list)
    ):
        raise RestoreError("Elasticsearch export is incomplete")
    # ``backup_local`` stores the single controlled index definition directly;
    # accept the raw Elasticsearch response shape as well for older backups.
    source_definition = (
        definition.get(source_index)
        if source_index in definition
        else definition
    )
    if not isinstance(source_definition, Mapping):
        raise RestoreError("Elasticsearch export lacks source index definition")
    mappings = source_definition.get("mappings")
    if not isinstance(mappings, Mapping):
        raise RestoreError("Elasticsearch export lacks index mapping")
    target_index = ElasticsearchChildIndexStore.index_name(target_generation)
    client = AsyncElasticsearch(endpoint)
    try:
        if await client.indices.exists(index=target_index):
            raise RestoreError("Elasticsearch restore target index already exists")
        await client.indices.create(index=target_index, mappings=dict(mappings))
        operations: list[Mapping[str, object]] = []
        for document in documents:
            if not isinstance(document, Mapping):
                raise RestoreError("Elasticsearch export contains an invalid document")
            identifier = document.get("id")
            source = document.get("source")
            if not isinstance(identifier, str) or not isinstance(source, Mapping):
                raise RestoreError("Elasticsearch export contains an invalid document")
            operations.extend(({"index": {"_index": target_index, "_id": identifier}}, dict(source)))
        if operations:
            response = await client.bulk(operations=operations, refresh="wait_for")
            if response.body.get("errors"):
                raise RestoreError("Elasticsearch bulk restore failed")
        await _restore_templates(client, payload.get("index_templates"))
        await _restore_aliases(client, payload.get("aliases"), source_index, target_index)
        count = await client.count(index=target_index, query={"match_all": {}})
        if int(count.body["count"]) != len(documents):
            raise RestoreError("Elasticsearch restored document count does not match export")
        mapping = await client.indices.get_mapping(index=target_index)
        if target_index not in mapping.body:
            raise RestoreError("Elasticsearch restored index mapping is unavailable")
    finally:
        await client.close()


async def _restore_redis(export_path: Path, endpoint: str, target_prefix: str) -> None:
    """Restore only into an explicitly named, empty Redis namespace."""
    _validate_redis_key_prefix(target_prefix)
    try:
        payload = json.loads(export_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RestoreError("Redis export is invalid") from error
    if not isinstance(payload, Mapping):
        raise RestoreError("Redis export is invalid")
    source_prefix = payload.get("key_prefix")
    rows = payload.get("keys")
    if (
        not isinstance(source_prefix, str)
        or re.fullmatch(r"[A-Za-z0-9:_-]{1,256}", source_prefix) is None
        or not isinstance(rows, list)
    ):
        raise RestoreError("Redis export prefix or key inventory is invalid")
    if source_prefix == target_prefix:
        raise RestoreError("Redis restore target prefix must differ from the source prefix")
    from redis.asyncio import Redis

    client = Redis.from_url(endpoint, decode_responses=False)
    decoded: list[tuple[bytes, bytes, int]] = []
    written_keys: list[bytes] = []
    try:
        await client.ping()
        existing_keys = [
            key
            async for key in client.scan_iter(match=f"{target_prefix}*")
        ]
        if existing_keys:
            raise RestoreError("Redis restore target namespace is not empty")
        source_prefix_bytes = source_prefix.encode()
        target_prefix_bytes = target_prefix.encode()
        seen_keys: set[bytes] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise RestoreError("Redis export contains an invalid key row")
            raw_key = row.get("key_b64")
            raw_value = row.get("payload_b64")
            raw_ttl = row.get("ttl_ms")
            if (
                not isinstance(raw_key, str)
                or not isinstance(raw_value, str)
                or isinstance(raw_ttl, bool)
                or not isinstance(raw_ttl, int)
            ):
                raise RestoreError("Redis export contains invalid encoded data")
            try:
                key = base64.b64decode(raw_key, validate=True)
                value = base64.b64decode(raw_value, validate=True)
                ttl_ms = raw_ttl
            except (TypeError, ValueError) as error:
                raise RestoreError("Redis export contains invalid encoded data") from error
            if not key.startswith(source_prefix_bytes) or ttl_ms < -1:
                raise RestoreError("Redis export contains an unsafe key or TTL")
            target_key = target_prefix_bytes + key[len(source_prefix_bytes) :]
            if target_key in seen_keys:
                raise RestoreError("Redis export contains duplicate keys")
            seen_keys.add(target_key)
            decoded.append((target_key, value, ttl_ms))
        for key, value, ttl_ms in decoded:
            await client.restore(key, max(ttl_ms, 0), value, replace=False)
            written_keys.append(key)
        for key, value, _ttl_ms in decoded:
            if not await client.exists(key) or await client.dump(key) != value:
                raise RestoreError("Redis restored key did not verify")
    except Exception as error:
        if written_keys:
            try:
                await client.delete(*written_keys)
            except Exception:
                pass
        if isinstance(error, RestoreError):
            raise
        raise RestoreError("Redis namespace restore failed") from error
    finally:
        await client.aclose()


def _validate_redis_key_prefix(value: object) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9:_-]{1,256}", value) is None
    ):
        raise RestoreError("Redis key prefix must be a non-empty safe string")


async def _restore_templates(client: Any, raw_templates: object) -> None:
    if not isinstance(raw_templates, list):
        raise RestoreError("Elasticsearch template export is invalid")
    for item in raw_templates:
        if not isinstance(item, Mapping):
            raise RestoreError("Elasticsearch template export is invalid")
        name, template = item.get("name"), item.get("index_template")
        if not isinstance(name, str) or not name.startswith("agenticrag-") or not isinstance(template, Mapping):
            raise RestoreError("Elasticsearch template export is unsafe")
        if await client.indices.exists_index_template(name=name):
            raise RestoreError("Elasticsearch restore template already exists")
        await client.indices.put_index_template(name=name, index_template=dict(template))
        verified = await client.indices.get_index_template(name=name)
        if not verified.body.get("index_templates"):
            raise RestoreError("Elasticsearch restored template did not verify")


async def _restore_aliases(client: Any, raw_aliases: object, source: str, target: str) -> None:
    if not isinstance(raw_aliases, Mapping):
        raise RestoreError("Elasticsearch alias export is invalid")
    source_aliases = raw_aliases.get(source, {})
    aliases = source_aliases.get("aliases", {}) if isinstance(source_aliases, Mapping) else {}
    if not isinstance(aliases, Mapping):
        raise RestoreError("Elasticsearch alias export is invalid")
    from elasticsearch import NotFoundError

    for alias, options in aliases.items():
        if not isinstance(alias, str) or not alias.startswith("agenticrag-") or not isinstance(options, Mapping):
            raise RestoreError("Elasticsearch alias export is unsafe")
        try:
            existing = await client.indices.get_alias(name=alias, allow_no_indices=True)
        except NotFoundError:
            existing = None
        if existing is not None and existing.body:
            raise RestoreError("Elasticsearch restore alias already exists")
        await client.indices.put_alias(index=target, name=alias, **dict(options))
        verified = await client.indices.get_alias(name=alias)
        if target not in verified.body:
            raise RestoreError("Elasticsearch restored alias did not verify")


def _verify_backup(backup: Path) -> dict[str, Any]:
    if not backup.is_dir() or backup.is_symlink():
        raise RestoreError("backup must be a real directory")
    manifest_path = backup / "manifest.json"
    digest_path = backup / "manifest.sha256"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not digest_path.is_file()
        or digest_path.is_symlink()
    ):
        raise RestoreError("backup manifest is missing")
    try:
        expected_digest = digest_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise RestoreError("backup manifest digest is invalid") from error
    if len(expected_digest) != 64 or _sha256(manifest_path) != expected_digest:
        raise RestoreError("backup manifest integrity check failed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RestoreError("backup manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("format_version")) is not int
        or manifest.get("format_version") != 1
    ):
        raise RestoreError("backup format is unsupported")
    for field in ("app_version", "schema_generation", "index_generation"):
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip():
            raise RestoreError(f"backup manifest field {field} is invalid")
    from agentic_rag.models.indexing import validate_index_generation

    try:
        validate_index_generation(manifest["index_generation"])
    except ValueError as error:
        raise RestoreError("backup manifest index_generation is invalid") from error
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RestoreError("backup manifest has no file inventory")
    expected_paths: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise RestoreError("backup manifest file entry is invalid")
        path = _safe_relative(entry.get("path"))
        if path.as_posix() in expected_paths:
            raise RestoreError("backup manifest has duplicate file paths")
        expected_paths.add(path.as_posix())
        source = backup / path
        if not source.is_file() or source.is_symlink():
            raise RestoreError(f"backup file is missing: {path}")
        if (
            type(entry.get("sha256")) is not str
            or len(entry["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in entry["sha256"])
            or type(entry.get("size_bytes")) is not int
            or entry["size_bytes"] < 0
            or entry["sha256"] != _sha256(source)
            or entry["size_bytes"] != source.stat().st_size
        ):
            raise RestoreError(f"backup integrity check failed: {path}")
    actual_paths: set[str] = set()
    for path in backup.rglob("*"):
        relative = path.relative_to(backup).as_posix()
        if relative in {"manifest.json", "manifest.sha256"}:
            continue
        if path.is_symlink():
            raise RestoreError("backup contains an unsafe symlink")
        if path.is_file():
            actual_paths.add(relative)
        elif not path.is_dir():
            raise RestoreError("backup contains an unsupported filesystem entry")
    if actual_paths != expected_paths:
        raise RestoreError("backup contains files outside its hash-verified inventory")
    return manifest


def _validate_restored_artifacts(root: Path) -> None:
    """Reject malformed copied Artifact trees before publishing the restore."""
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RestoreError("restored artifact tree contains an unsafe path")


def _run_readiness(check: ReadinessCheck) -> None:
    result = check()
    if inspect.isawaitable(result):
        result = _run_async(_await_readiness(result))
    if (
        not isinstance(result, Mapping)
        or not result
        or any(value != "available" for value in result.values())
    ):
        raise RestoreError("readiness checks did not all pass")


async def _await_readiness(value: Awaitable[Mapping[str, str]]) -> Mapping[str, str]:
    return await value


def _default_readiness_check(
    staging: Path, services: ServiceRestoreSpec | None
) -> ReadinessCheck:
    """Probe the restored target before publishing it.

    Service endpoints are always deployment-supplied.  When a backup contains
    no external service state, the local tree probe still prevents an empty
    readiness mapping from being treated as success.
    """

    async def check() -> Mapping[str, str]:
        from agentic_rag.api.health import (
            ReadinessChecks,
            check_elasticsearch,
            check_mysql,
            check_redis,
        )

        engines: list[Any] = []
        redis_clients: list[Any] = []
        elasticsearch_clients: list[Any] = []
        checks: dict[str, Any] = {
            "restore_target": lambda: _check_restore_tree(staging)
        }
        try:
            if services is not None and services.mysql_dsn:
                from agentic_rag.persistence.mysql import create_mysql_engine

                engine = create_mysql_engine(services.mysql_dsn, pool_pre_ping=True)
                engines.append(engine)
                checks["mysql"] = lambda engine=engine: check_mysql(engine)
            if services is not None and services.redis_dsn:
                from redis.asyncio import Redis

                client = Redis.from_url(services.redis_dsn)
                redis_clients.append(client)
                checks["redis"] = lambda client=client: check_redis(client)
            if services is not None and services.elasticsearch_url:
                from elasticsearch import AsyncElasticsearch

                client = AsyncElasticsearch(services.elasticsearch_url)
                elasticsearch_clients.append(client)
                checks["elasticsearch"] = lambda client=client: check_elasticsearch(client)
            return await ReadinessChecks(checks).require_ready()
        finally:
            for client in redis_clients:
                await client.aclose()
            for client in elasticsearch_clients:
                await client.close()
            for engine in engines:
                await engine.dispose()

    return check


async def _check_restore_tree(staging: Path) -> None:
    if not staging.is_dir() or staging.is_symlink():
        raise RuntimeError("restored staging tree is unavailable")
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("restored staging tree contains a symlink")


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RestoreError("backup contains an unsafe relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {".", ""}:
        raise RestoreError("backup contains an unsafe relative path")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _remove_empty_directory(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        # Never delete data another process may have placed in the reserved
        # path while this restore was running.
        pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mysql-dsn")
    parser.add_argument("--elasticsearch-url")
    parser.add_argument("--index-generation")
    parser.add_argument("--redis-dsn")
    parser.add_argument("--redis-key-prefix")
    args = parser.parse_args(argv)
    try:
        services = ServiceRestoreSpec(
            mysql_dsn=args.mysql_dsn,
            elasticsearch_url=args.elasticsearch_url,
            index_generation=args.index_generation,
            redis_dsn=args.redis_dsn,
            redis_key_prefix=args.redis_key_prefix,
        )
        restored = restore_backup(args.backup, args.target, services=services)
    except RestoreError as error:
        parser.error(str(error))
    print(restored)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
