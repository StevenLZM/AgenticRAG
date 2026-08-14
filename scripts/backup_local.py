"""Create deterministic, integrity-checked backups of local Agentic RAG state.

The command never replaces an existing backup.  Operators must stop workers
before running it so SQLite checkpointing and in-flight claims are quiescent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy.engine import make_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agentic_rag.config import Settings  # noqa: E402
from agentic_rag.models.indexing import validate_index_generation  # noqa: E402


BACKUP_FORMAT_VERSION = 1


class BackupError(RuntimeError):
    """Raised when a complete, verified backup cannot be produced."""


@dataclass(frozen=True)
class BackupSpec:
    """Explicit local paths and optional service endpoints to include."""

    output: Path
    artifact_root: Path
    checkpoint_paths: tuple[Path, ...]
    app_version: str
    schema_generation: str
    index_generation: str
    mysql_dsn: str | None = None
    elasticsearch_url: str | None = None
    mysqldump_command: str = "mysqldump"


def create_backup(spec: BackupSpec) -> Path:
    """Write one fully checked backup by atomically publishing a new directory."""
    output = Path(spec.output)
    if output.exists():
        raise BackupError(f"backup output already exists: {output}")
    if not spec.app_version.strip() or not spec.schema_generation.strip():
        raise BackupError("app_version and schema_generation must be non-empty")
    validate_index_generation(spec.index_generation)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
    try:
        _copy_artifacts(Path(spec.artifact_root), staging / "artifacts")
        _backup_checkpoints(spec.checkpoint_paths, staging / "checkpoints")
        service_metadata: dict[str, object] = {}
        if spec.mysql_dsn is not None:
            _dump_mysql(spec.mysql_dsn, staging / "mysql" / "dump.sql", spec.mysqldump_command)
            service_metadata["mysql"] = {"included": True}
        if spec.elasticsearch_url is not None:
            service_metadata["elasticsearch"] = asyncio.run(
                _export_elasticsearch(spec.elasticsearch_url, spec.index_generation, staging / "elasticsearch" / "export.json")
            )
        manifest = {
            "format_version": BACKUP_FORMAT_VERSION,
            "app_version": spec.app_version,
            "schema_generation": spec.schema_generation,
            "index_generation": spec.index_generation,
            "files": _file_manifest(staging),
            "services": service_metadata,
        }
        _write_canonical_json(staging / "manifest.json", manifest)
        (staging / "manifest.sha256").write_text(
            _sha256(staging / "manifest.json") + "\n", encoding="ascii"
        )
        _fsync_tree(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return output
    except Exception as error:
        if staging.exists():
            shutil.rmtree(staging)
        if isinstance(error, BackupError):
            raise
        raise BackupError(f"backup failed: {error}") from error


def _copy_artifacts(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if not source.is_dir() or source.is_symlink():
        raise BackupError("artifact root must be a real directory")
    for path in _regular_files(source):
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target, follow_symlinks=False)


def _backup_checkpoints(paths: Sequence[Path], destination: Path) -> None:
    seen: set[str] = set()
    for configured in paths:
        source = Path(configured)
        if not source.exists():
            continue
        if source.is_symlink() or not source.is_file():
            raise BackupError(f"checkpoint must be a regular SQLite file: {source}")
        if source.name in seen:
            raise BackupError("checkpoint file names must be unique")
        seen.add(source.name)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        _sqlite_backup(source, target)


def _sqlite_backup(source: Path, target: Path) -> None:
    """Checkpoint WAL and use SQLite's backup API instead of a racy file copy."""
    try:
        with sqlite3.connect(source) as source_connection:
            source_connection.execute("PRAGMA wal_checkpoint(FULL)")
            with sqlite3.connect(target) as target_connection:
                source_connection.backup(target_connection)
    except sqlite3.Error as error:
        raise BackupError(f"SQLite checkpoint backup failed for {source}: {error}") from error


def _dump_mysql(dsn: str, target: Path, command: str) -> None:
    url = make_url(dsn)
    if url.drivername != "mysql+asyncmy" or not url.database:
        raise BackupError("MySQL backup requires a mysql+asyncmy DSN with a database")
    target.parent.mkdir(parents=True, exist_ok=True)
    args = [
        command,
        "--single-transaction",
        "--routines",
        "--events",
        "--no-tablespaces",
        f"--host={url.host or 'localhost'}",
        f"--port={url.port or 3306}",
        f"--user={url.username or ''}",
        "--result-file=" + str(target),
        url.database,
    ]
    environment = dict(os.environ)
    if url.password is not None:
        environment["MYSQL_PWD"] = url.password
    try:
        subprocess.run(args, check=True, env=environment, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise BackupError("consistent MySQL dump failed") from error
    if not target.is_file() or target.stat().st_size == 0:
        raise BackupError("consistent MySQL dump produced no data")


async def _export_elasticsearch(
    endpoint: str, index_generation: str, target: Path
) -> dict[str, object]:
    """Export the controlled indices plus aliases/templates in canonical order."""
    from elasticsearch import AsyncElasticsearch

    client = AsyncElasticsearch(endpoint)
    index = f"agenticrag-children-{index_generation}"
    try:
        index_data = _response_body(
            await client.indices.get(index=index, allow_no_indices=True)
        )
        aliases = _response_body(
            await client.indices.get_alias(index="agenticrag-*", allow_no_indices=True)
        )
        templates: Mapping[str, object]
        try:
            templates = _response_body(
                await client.indices.get_index_template(name="agenticrag-*")
            )
        except Exception:
            templates = {"index_templates": []}
        documents: list[dict[str, object]] = []
        if index in index_data:
            documents = await _export_documents(client, index)
        payload = {
            "index": index,
            "index_definition": index_data.get(index, {}),
            "aliases": aliases,
            "index_templates": templates.get("index_templates", []),
            "documents": documents,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_canonical_json(target, payload)
        return {"included": True, "index": index, "document_count": len(documents)}
    finally:
        await client.close()


async def _export_documents(client: Any, index: str) -> list[dict[str, object]]:
    """Read every document, then sort IDs so export bytes are reproducible."""
    response = await client.search(
        index=index,
        size=1_000,
        scroll="1m",
        query={"match_all": {}},
        sort=["_doc"],
    )
    scroll_id: str | None = None
    rows: list[dict[str, object]] = []
    try:
        while True:
            body = response.body
            candidate = body.get("_scroll_id")
            if isinstance(candidate, str):
                scroll_id = candidate
            hits = body.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                identifier = hit.get("_id")
                source = hit.get("_source", {})
                if not isinstance(identifier, str) or not isinstance(source, Mapping):
                    raise BackupError("Elasticsearch export returned an invalid document")
                rows.append({"id": identifier, "source": dict(source)})
            if scroll_id is None:
                raise BackupError("Elasticsearch export did not return a scroll ID")
            response = await client.scroll(scroll_id=scroll_id, scroll="1m")
    finally:
        if scroll_id is not None:
            await client.clear_scroll(scroll_id=scroll_id)
    return sorted(rows, key=lambda row: str(row["id"]))


def _response_body(value: Any) -> Mapping[str, object]:
    """Normalize Elasticsearch client responses for canonical JSON export."""
    body = getattr(value, "body", value)
    if not isinstance(body, Mapping):
        raise BackupError("Elasticsearch response body is not a JSON object")
    return body


def _regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise BackupError(f"backup input contains a symlink: {path}")
        if path.is_file():
            yield path
        elif not path.is_dir():
            raise BackupError(f"backup input contains an unsupported path: {path}")


def _file_manifest(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _regular_files(root)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fsync_tree(root: Path) -> None:
    for path in _regular_files(root):
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


def _default_spec(output: Path, settings: Settings, include_services: bool) -> BackupSpec:
    from alembic.script import ScriptDirectory
    from alembic.config import Config
    return BackupSpec(
        output=output,
        artifact_root=settings.artifact_root,
        checkpoint_paths=(settings.query_checkpoint_path, settings.ingestion_checkpoint_path),
        app_version="0.1.0",
        schema_generation=ScriptDirectory.from_config(Config("alembic.ini")).get_current_head() or "unknown",
        index_generation=settings.index_generation,
        mysql_dsn=settings.mysql_dsn if include_services else None,
        elasticsearch_url=settings.elasticsearch_url if include_services else None,
    )


def run_backup_restore_drill() -> bool:
    """Exercise the signed local-state restore path in an isolated temp root."""
    from scripts.restore_local import RestoreError, restore_backup

    try:
        with tempfile.TemporaryDirectory(prefix="agentic-rag-backup-drill-") as root:
            base = Path(root)
            artifacts = base / "source" / "artifacts"
            checkpoints = base / "source" / "query.sqlite"
            from agentic_rag.persistence.artifacts import LocalArtifactStore

            store = LocalArtifactStore(artifacts)
            store.put_json("drill/manifest.json", {"citation_coverage": 1.0})
            with sqlite3.connect(checkpoints) as connection:
                connection.execute("CREATE TABLE drill (value TEXT NOT NULL)")
                connection.execute("INSERT INTO drill VALUES ('queryable')")
            backup = create_backup(
                BackupSpec(
                    output=base / "backup",
                    artifact_root=artifacts,
                    checkpoint_paths=(checkpoints,),
                    app_version="drill",
                    schema_generation="drill",
                    index_generation="drill-v1",
                )
            )
            restored = restore_backup(backup, base / "restored")
            restored_store = LocalArtifactStore(restored / "artifacts")
            ref = restored_store.describe("artifact://drill/manifest.json")
            if restored_store.read_json(ref).get("citation_coverage") != 1.0:
                return False
            with sqlite3.connect(restored / "checkpoints" / "query.sqlite") as connection:
                return connection.execute("SELECT value FROM drill").fetchone() == ("queryable",)
    except (BackupError, RestoreError, OSError, sqlite3.Error, ValueError):
        return False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-services", action="store_true")
    args = parser.parse_args(argv)
    try:
        backup = create_backup(
            _default_spec(args.output, Settings(), args.include_services)  # type: ignore[call-arg]
        )
    except BackupError as error:
        parser.error(str(error))
    print(backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
