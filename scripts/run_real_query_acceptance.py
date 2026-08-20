"""Run one opt-in, production-composed Query API acceptance case.

This command is intentionally separate from the hermetic E2E fixtures.  When
``AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1`` is set it composes the real
QueryGraph/Worker, DeepSeek, Qwen, reranker, Mem0, MySQL, Redis and
Elasticsearch boundaries in an isolated user/index namespace.  It writes the
same summary format consumed by ``verify_acceptance.py`` and fails closed on
missing credentials, provider failures, audit failures or cleanup failures.
"""

# The source-layout bootstrap below intentionally precedes application imports.
# Keep the script runnable directly without an editable install.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, select
from sqlalchemy.engine import make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.api.app import create_app
from agentic_rag.bootstrap import AppContainer, build_container
from agentic_rag.config import Settings
from agentic_rag.domain.models import DocumentVersionStatus, RunStatus, UserScope
from agentic_rag.ingestion.chunker import AstLocator, AstSpan, ChildChunk, ParentChunk
from agentic_rag.ingestion.indexer import EMBEDDING_MODEL, IndexWriter, StagingContext
from agentic_rag.ingestion.publisher import VersionPublisher
from agentic_rag.memory.models import PublicMessage
from agentic_rag.memory.service import MemoryService
from agentic_rag.persistence.elasticsearch import ElasticsearchChildIndexStore
from agentic_rag.persistence.lifecycle import SqlAlchemyPublicationRepository
from agentic_rag.observability.logging import emit_degradation, event_emission_scope
from agentic_rag.persistence.outbox import OutboxDispatcher
from agentic_rag.persistence.repositories import (
    SqlAlchemyDocumentRepository,
    agent_events,
    agent_runs,
    documents,
    memory_tombstones,
    parent_chunks,
    task_outbox,
)
from agentic_rag.persistence.staging import SqlAlchemyParentStagingStore
from agentic_rag.query.graph import QueryGraphDependencies
from agentic_rag.runtime.query_composition import (
    build_query_dependencies,
    close_query_dependencies,
)
from agentic_rag.runtime.query_worker import QueryWorker, build_graph_factory
from agentic_rag.runtime.run_manager import RunManager, TransactionalRunRepository
from agentic_rag.testing.isolated_query_broker import IsolatedQueryBroker
from agentic_rag.testing.real_provider_config import (
    explicit_provider_configuration_issue,
    provider_configuration_issue,
    provider_environment_from_process,
)
from evals.clients import HttpQueryClient
from evals.models import EvaluationCase
from evals.report import write_summary
from evals.run import EvalRunner
from scripts.backup_local import run_backup_restore_drill
from scripts.run_query_worker import TransactionalQueryOutboxAdapter
from scripts.run_recovery_drill import run_recovery_drill
from scripts.verify_acceptance import verify_acceptance


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PUBLIC_DEGRADATION_ATTRIBUTES = frozenset(
    {"attempt", "component", "reason", "outcome", "retryable"}
)
_CONTROLLED_DEGRADATION = {
    "attempt": 1,
    "component": "retrieval",
    "outcome": "degraded",
    "reason": "circuit_open",
    "retryable": True,
}
_SENSITIVE_MARKERS = ("Bearer ", "sk-", "chain_of_thought", "prompt", "tool_input")


class AcceptanceTeardownError(RuntimeError):
    """An owned acceptance boundary failed after the query gates passed."""


async def _record_teardown_error(
    failures: list[tuple[str, BaseException]],
    boundary: str,
    operation: Awaitable[object],
) -> None:
    """Run every cleanup step, preserving failures for a final fail-closed gate.

    ``CancelledError`` derives from ``BaseException``.  It must be retained
    long enough to release every other owned boundary, then re-raised below;
    allowing it to escape here would leave the current acceptance attempt
    partially cleaned and could retain a reportable PASS summary.
    """
    try:
        await operation
    except BaseException as error:
        failures.append((boundary, error))


def _raise_teardown_failures(failures: Sequence[tuple[str, BaseException]]) -> None:
    if not failures:
        return
    for _, error in failures:
        # Never translate cancellation or process-control flow into an
        # ordinary acceptance error.  The cleanup sequence above has already
        # exhausted the remaining owned boundaries before this is re-raised.
        if not isinstance(error, Exception):
            raise error
    details = ", ".join(
        f"{boundary}: {type(error).__name__}" for boundary, error in failures
    )
    raise AcceptanceTeardownError(
        f"real Query acceptance teardown failed ({details})"
    ) from failures[0][1]


class _FixedEmbedding:
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 1023 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0] + [0.0] * 1023


def _require_real_provider_base_settings() -> tuple[Settings, str, str, str]:
    """Read explicit disposable-service inputs without falling back to app DSNs."""
    if os.environ.get("AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E") != "1":
        raise RuntimeError(
            "set AGENTIC_RAG_RUN_REAL_QUERY_PROVIDER_E2E=1 to permit live provider calls"
        )
    mysql_dsn = os.getenv("AGENTIC_RAG_TEST_MYSQL_DSN", "").strip()
    redis_url = os.getenv("AGENTIC_RAG_TEST_REDIS_DSN", "").strip()
    elasticsearch_url = os.getenv("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", "").strip()
    missing = [
        name
        for name, value in (
            ("AGENTIC_RAG_TEST_MYSQL_DSN", mysql_dsn),
            ("AGENTIC_RAG_TEST_REDIS_DSN", redis_url),
            ("AGENTIC_RAG_TEST_ELASTICSEARCH_URL", elasticsearch_url),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "missing explicit disposable real-service settings: " + ", ".join(missing)
        )
    try:
        mysql_host = make_url(mysql_dsn).host
        redis_host = urlparse(redis_url).hostname
        elasticsearch_host = urlparse(elasticsearch_url).hostname
    except ValueError as error:
        raise RuntimeError("real Query acceptance service URL is invalid") from error
    if mysql_host not in _LOCAL_HOSTS:
        raise RuntimeError("real Query acceptance requires a loopback MySQL test DSN")
    if redis_host not in _LOCAL_HOSTS:
        raise RuntimeError("real Query acceptance requires a loopback Redis test DSN")
    if elasticsearch_host not in _LOCAL_HOSTS:
        raise RuntimeError(
            "real Query acceptance requires a loopback Elasticsearch test URL"
        )
    try:
        base = Settings(  # type: ignore[call-arg]
            mysql_dsn=mysql_dsn,
            redis_url=redis_url,
            elasticsearch_url=elasticsearch_url,
        )
    except Exception as error:
        explicit_issue = explicit_provider_configuration_issue(
            provider_environment_from_process()
        )
        if explicit_issue is not None:
            _, fields = explicit_issue
            raise RuntimeError(
                "invalid DeepSeek/Qwen/Mem0 real acceptance configuration: "
                + ", ".join(fields)
            ) from error
        raise RuntimeError(
            "configured DeepSeek/Qwen/Mem0 variables are required for real Query acceptance"
        ) from error
    issue = provider_configuration_issue(base)
    if issue is not None:
        kind, fields = issue
        raise RuntimeError(
            f"{kind} DeepSeek/Qwen/Mem0 real acceptance configuration: "
            + ", ".join(fields)
        )
    return base, mysql_dsn, redis_url, elasticsearch_url


async def _wait_for_terminal(
    client: httpx.AsyncClient,
    run_id: str,
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    terminal = {status.value for status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/v1/query-runs/{run_id}")
        if response.status_code != 200:
            raise RuntimeError(
                f"Query API status lookup failed with HTTP {response.status_code}"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError("Query API returned a non-object run response")
        result = dict(payload)
        if result.get("status") in terminal:
            return result
        await asyncio.sleep(0.25)
    raise RuntimeError("Query API Run did not reach a terminal status before timeout")


async def _read_sse(
    client: httpx.AsyncClient,
    run_id: str,
) -> tuple[list[dict[str, object]], int]:
    """Read the reconnectable public projection after a Run is terminal."""
    response = await client.get(f"/v1/query-runs/{run_id}/events")
    if response.status_code != 200:
        raise RuntimeError(f"Query SSE endpoint failed with HTTP {response.status_code}")
    events: list[dict[str, object]] = []
    current_id: int | None = None
    current_type = "PROGRESS"
    last_event_id = 0
    for line in response.text.splitlines():
        if line.startswith("id: "):
            try:
                current_id = int(line.removeprefix("id: ").strip())
            except ValueError as error:
                raise RuntimeError("Query SSE emitted a non-integer event cursor") from error
        elif line.startswith("event: "):
            current_type = line.removeprefix("event: ").strip() or "PROGRESS"
        elif line.startswith("data: "):
            try:
                payload = json.loads(line.removeprefix("data: "))
            except json.JSONDecodeError as error:
                raise RuntimeError("Query SSE emitted invalid JSON") from error
            if not isinstance(payload, Mapping) or current_id is None:
                raise RuntimeError("Query SSE omitted a public event object or cursor")
            event = dict(payload)
            event["id"] = current_id
            event.setdefault("event_type", current_type)
            events.append(event)
            last_event_id = max(last_event_id, current_id)
            current_id = None
            current_type = "PROGRESS"
    if not events:
        raise RuntimeError("Query SSE emitted no durable events")
    return events, last_event_id


def _assert_safe_public_events(events: Sequence[Mapping[str, object]]) -> None:
    """Reject an API projection containing any raw model/provider/tool material."""
    encoded = json.dumps(list(events), ensure_ascii=False, sort_keys=True)
    if any(marker in encoded for marker in _SENSITIVE_MARKERS):
        raise RuntimeError("Query SSE exposed a sensitive prompt, secret, or tool field")
    for event in events:
        attributes = event.get("attributes")
        if attributes is not None and (
            not isinstance(attributes, Mapping)
            or not set(attributes).issubset(_PUBLIC_DEGRADATION_ATTRIBUTES)
        ):
            raise RuntimeError("Query SSE exposed a non-allowlisted degradation attribute")


def _assert_audited_answer(
    run: Mapping[str, object],
    *,
    parent_id: str,
    snapshot_id: str,
) -> None:
    if run.get("status") != RunStatus.COMPLETED.value:
        raise RuntimeError("Query API Run did not complete")
    if run.get("runtime_config_snapshot_id") != snapshot_id:
        raise RuntimeError("Query API Run used a different runtime snapshot")
    answer = run.get("answer")
    if not isinstance(answer, Mapping) or answer.get("audited") is not True:
        raise RuntimeError("completed Query API Run omitted an audited answer")
    segments = answer.get("segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        raise RuntimeError("completed Query API Run omitted audited answer segments")
    if not all(isinstance(segment, Mapping) for segment in segments):
        raise RuntimeError("completed Query API Run emitted malformed answer segments")
    parent_ids = answer.get("evidence_parent_ids")
    if not isinstance(parent_ids, Sequence) or isinstance(parent_ids, (str, bytes)):
        raise RuntimeError("completed Query API Run omitted server evidence parent IDs")
    if parent_id not in parent_ids:
        raise RuntimeError("completed Query API Run did not cite the seeded server Parent")


async def _exercise_memory_boundary(
    memory_service: MemoryService,
    settings: Settings,
) -> tuple[dict[str, bool], str]:
    """Use the production MemoryService for a real, scoped Mem0 read and write."""
    if not bool(getattr(memory_service, "available", False)):
        raise RuntimeError("real Query acceptance requires an available Mem0 provider")
    scope = UserScope(user_id=settings.default_user_id)
    marker = f"memory-boundary-{uuid4().hex}"
    context = await memory_service.load_context(scope, marker)
    if bool(getattr(context, "degraded", True)):
        raise RuntimeError("Mem0 read boundary degraded during real Query acceptance")
    await memory_service.extract_and_store(
        scope,
        f"memory-boundary-{uuid4().hex}",
        [
            PublicMessage(
                id="memory-boundary-message",
                role="user",
                content=(
                    "Please remember this durable acceptance preference marker: "
                    f"{marker}."
                ),
            )
        ],
    )
    deadline = asyncio.get_running_loop().time() + 30.0
    while asyncio.get_running_loop().time() < deadline:
        records = await memory_service.list(scope)
        if any(marker in str(getattr(record, "text", "")) for record in records):
            return {"read": True, "write": True}, marker
        await asyncio.sleep(0.25)
    raise RuntimeError("Mem0 write boundary did not return the stored acceptance marker")


async def _assert_memory_api_boundary(
    client: httpx.AsyncClient,
    *,
    marker: str,
) -> None:
    response = await client.get("/v1/memories")
    if response.status_code != 200:
        raise RuntimeError(f"Mem0 API boundary failed with HTTP {response.status_code}")
    payload = response.json()
    memories = payload.get("memories") if isinstance(payload, Mapping) else None
    if not isinstance(memories, Sequence) or isinstance(memories, (str, bytes)):
        raise RuntimeError("Mem0 API boundary returned malformed memories")
    if not any(
        isinstance(memory, Mapping) and marker in str(memory.get("text", ""))
        for memory in memories
    ):
        raise RuntimeError("Mem0 API boundary did not expose the scoped stored marker")


async def _emit_controlled_degradation(
    dependencies: QueryGraphDependencies,
    *,
    run_id: str,
    user_id: str,
    snapshot_id: str,
) -> None:
    """Persist one safe circuit notice, exercising logging and event artifacts."""
    emitter = dependencies.event_emitter
    if emitter is None:
        raise RuntimeError("real Query acceptance requires a durable event emitter")
    async with event_emission_scope(
        emitter,
        run_id,
        "console.acceptance",
        user_id=user_id,
    ):
        await emit_degradation(
            component="retrieval",
            reason="circuit_open",
            run_id=run_id,
            snapshot_id=snapshot_id,
            attempt=1,
            retryable=True,
            outcome="degraded",
        )


def _controlled_degradation_events(
    events: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for event in events:
        attributes = event.get("attributes")
        if event.get("event_type") == "CIRCUIT_OPEN" and attributes == _CONTROLLED_DEGRADATION:
            matches.append(
                {
                    "event_type": "CIRCUIT_OPEN",
                    "attributes": dict(_CONTROLLED_DEGRADATION),
                }
            )
    if not matches:
        raise RuntimeError("Query SSE omitted the durable controlled circuit degradation")
    return matches


def console_acceptance_passed(summary: Mapping[str, object]) -> bool:
    """Fail closed on the console-specific live evidence absent from generic eval gates."""
    try:
        memory_boundary = summary.get("memory_boundary")
        sse_last_event_id = summary.get("sse_last_event_id")
        return bool(
            summary.get("console_page_status") == 200
            and type(sse_last_event_id) is int
            and sse_last_event_id > 0
            and summary.get("client_provenance") == HttpQueryClient.provenance
            and isinstance(summary.get("runtime_config_snapshot_id"), str)
            and bool(str(summary["runtime_config_snapshot_id"]).strip())
            and summary.get("console_run_snapshot_id")
            == summary.get("runtime_config_snapshot_id")
            and summary.get("memory_provider_available") is True
            and isinstance(memory_boundary, Mapping)
            and memory_boundary.get("read") is True
            and memory_boundary.get("write") is True
            and _has_safe_controlled_degradation(summary.get("degradation_events"))
            and type(summary.get("citation_coverage")) in {int, float}
            and not isinstance(summary.get("citation_coverage"), bool)
            and summary.get("citation_coverage") == 1.0
            and type(summary.get("user_leak_count")) is int
            and summary.get("user_leak_count") == 0
            and type(summary.get("unaudited_answer_count")) is int
            and summary.get("unaudited_answer_count") == 0
            and summary.get("recovery_drill_passed") is True
            and summary.get("backup_restore_passed") is True
        )
    except (KeyError, TypeError, ValueError):
        return False


def _has_safe_controlled_degradation(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for event in value:
        if not isinstance(event, Mapping) or event.get("event_type") != "CIRCUIT_OPEN":
            continue
        attributes = event.get("attributes")
        if isinstance(attributes, Mapping) and dict(attributes) == _CONTROLLED_DEGRADATION:
            return True
    return False


def _invalidate_previous_summary(output: Path) -> None:
    """Move a prior report aside before attempting a live acceptance run.

    ``verify_acceptance.py`` conventionally reads ``summary.json``.  Leaving a
    former PASS in that location when setup or a provider fails would let a
    caller validate stale evidence as though it belonged to this attempt.
    Preserve the old report for diagnosis under a non-default name instead.
    """
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.json"
    if not summary.exists() and not summary.is_symlink():
        return
    if not summary.is_file() and not summary.is_symlink():
        raise RuntimeError("real Query acceptance output summary path is not a file")
    stale = output / f"summary.stale-{uuid4().hex}.json"
    summary.replace(stale)


def _locator() -> AstLocator:
    span = AstSpan(
        canonical_path="#/text_blocks/0",
        block_id="real-provider-block",
        page_from=1,
        page_to=1,
        char_from=0,
        char_to=128,
        parent_char_from=0,
        parent_char_to=128,
    )
    return AstLocator(
        spans=(span,),
        segment_ordinal=0,
        parent_char_from=0,
        parent_char_to=128,
    )


async def _seed_document(
    container: AppContainer,
    settings: Settings,
    scope: UserScope,
) -> str:
    """Seed one real Parent/Child pair through the production indexer ports."""
    factory = container.repositories.session_factory
    async with factory.begin() as session:
        document, version = await SqlAlchemyDocumentRepository().create(
            scope,
            source_type="text",
            filename="real-query-acceptance.txt",
            mime_type="text/plain",
            content_hash=hashlib.sha256(b"real-query-acceptance").hexdigest(),
            parser_version="acceptance-parser-v1",
            pipeline_version="acceptance-pipeline-v1",
            embedding_version=EMBEDDING_MODEL,
            index_generation=settings.index_generation,
            version_status=DocumentVersionStatus.UPLOADED,
            transaction=session,
        )

    content = "The acceptance document requires thirty days notice before cancellation."
    locator = _locator()
    parent_id = f"acceptance-parent-{uuid4().hex}"
    child = ChildChunk(
        id=f"acceptance-child-{uuid4().hex}",
        parent_id=parent_id,
        parent_ordinal=0,
        document_id=document.id,
        document_version_id=version.id,
        user_id=scope.user_id,
        ordinal=0,
        heading_path=("Notice",),
        heading_ast_locators=(),
        content_type="text",
        content=content,
        contextualized_content=content,
        token_count=12,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
    )
    parent = ParentChunk(
        id=parent_id,
        document_id=document.id,
        document_version_id=version.id,
        user_id=scope.user_id,
        ordinal=0,
        heading_path=("Notice",),
        heading_ast_locators=(),
        content_type="text",
        content=content,
        token_count=12,
        page_from=1,
        page_to=1,
        ast_locator=locator,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        children=(child,),
    )
    context = StagingContext(
        user_id=scope.user_id,
        document_id=document.id,
        document_version_id=version.id,
        version_no=version.version_no,
        pipeline_version="acceptance-pipeline-v1",
        embedding_version=EMBEDDING_MODEL,
        index_generation=settings.index_generation,
    )
    canonical = container.artifacts.put_json(
        f"documents/{scope.user_id}/{document.id}/{version.id}/canonical/source.json",
        {"text": content},
    )
    await IndexWriter(
        embedding=_FixedEmbedding(),
        parent_store=SqlAlchemyParentStagingStore(factory),
        child_store=ElasticsearchChildIndexStore(container.elasticsearch),
        artifacts=cast(Any, container.artifacts),
    ).stage((parent,), context=context, canonical_ast=canonical)
    await VersionPublisher(
        repository=SqlAlchemyPublicationRepository(factory),
        parent_store=SqlAlchemyParentStagingStore(factory),
        child_store=ElasticsearchChildIndexStore(container.elasticsearch),
        artifacts=cast(Any, container.artifacts),
    ).publish(version.id)
    return parent_id


async def _cleanup_mysql_state(container: AppContainer, settings: Settings) -> None:
    """Delete only this acceptance user's MySQL rows."""
    run_ids: list[str] = []
    async with container.repositories.session_factory.begin() as session:
        run_ids = list(
            (
                await session.execute(
                    select(agent_runs.c.id).where(
                        agent_runs.c.user_id == settings.default_user_id
                    )
                )
            ).scalars()
        )
        if run_ids:
            await session.execute(
                delete(task_outbox).where(
                    task_outbox.c.aggregate_type == "query_run",
                    task_outbox.c.aggregate_id.in_(run_ids),
                )
            )
        await session.execute(
            delete(memory_tombstones).where(
                memory_tombstones.c.user_id == settings.default_user_id
            )
        )
        await session.execute(
            delete(agent_events).where(agent_events.c.user_id == settings.default_user_id)
        )
        await session.execute(
            delete(agent_runs).where(agent_runs.c.user_id == settings.default_user_id)
        )
        await session.execute(
            delete(parent_chunks).where(parent_chunks.c.user_id == settings.default_user_id)
        )
        await session.execute(
            delete(documents).where(documents.c.user_id == settings.default_user_id)
        )


async def _cleanup_redis_state(
    container: AppContainer,
    broker: IsolatedQueryBroker | None,
) -> None:
    """Delete only the private stream and dedupe keys owned by this attempt."""
    if broker is not None:
        # The adapter's stream/group is private to this acceptance attempt.
        # Delete entire keys rather than scanning production queue names, so
        # teardown cannot observe, ACK or alter another worker's delivery.
        await container.redis.delete(*broker.cleanup_keys)


async def _cleanup_query_index(container: AppContainer, settings: Settings) -> None:
    """Delete this acceptance attempt's isolated child index."""
    await container.elasticsearch.indices.delete(
        index=f"agenticrag-children-{settings.index_generation}",
        ignore_unavailable=True,
    )


async def _cleanup_mem0_index(container: AppContainer, settings: Settings) -> None:
    """Delete this acceptance attempt's isolated Mem0 index."""
    await container.elasticsearch.indices.delete(
        index=settings.mem0_collection,
        ignore_unavailable=True,
    )


async def _cleanup_durable_boundaries(
    failures: list[tuple[str, BaseException]],
    *,
    container: AppContainer,
    settings: Settings,
    broker: IsolatedQueryBroker | None,
) -> None:
    """Attempt every durable-state cleanup even if an earlier backend fails."""
    await _record_teardown_error(
        failures, "acceptance MySQL state", _cleanup_mysql_state(container, settings)
    )
    await _record_teardown_error(
        failures, "acceptance Redis state", _cleanup_redis_state(container, broker)
    )
    await _record_teardown_error(
        failures, "acceptance query index", _cleanup_query_index(container, settings)
    )
    await _record_teardown_error(
        failures, "acceptance Mem0 index", _cleanup_mem0_index(container, settings)
    )


async def run(output: Path) -> dict[str, object]:
    _invalidate_previous_summary(output)
    base, mysql_dsn, redis_url, elasticsearch_url = _require_real_provider_base_settings()
    root = Path(tempfile.mkdtemp(prefix="agentic-rag-real-query-"))
    suffix = uuid4().hex[:12]
    settings = base.model_copy(
        update={
            "mysql_dsn": mysql_dsn,
            "redis_url": redis_url,
            "elasticsearch_url": elasticsearch_url,
            "default_user_id": f"real-query-{suffix}",
            "index_generation": f"real-query-{suffix}",
            "artifact_root": root / "artifacts",
            "query_checkpoint_path": root / "query.sqlite",
            "ingestion_checkpoint_path": root / "ingestion.sqlite",
            "mem0_collection": f"agent_memories_{suffix}",
            "mem0_history_db_path": root / "mem0" / "history.db",
            "query_run_timeout_seconds": 180,
        }
    )
    container: AppContainer | None = None
    dependencies: QueryGraphDependencies | None = None
    checkpoint_context: Any = None
    stop = asyncio.Event()
    worker_task: asyncio.Task[None] | None = None
    app_lifespan: Any = None
    http_client: httpx.AsyncClient | None = None
    isolated_broker: IsolatedQueryBroker | None = None
    accepted_summary: dict[str, object] | None = None
    body_error: BaseException | None = None
    try:
        migration = Config(str((PROJECT_ROOT / "alembic.ini").resolve()))
        migration.set_main_option("sqlalchemy.url", settings.mysql_dsn)
        await asyncio.to_thread(command.upgrade, migration, "head")
        container = build_container(settings)
        isolated_broker = IsolatedQueryBroker(
            container.broker, namespace=f"real-query-{suffix}"
        )
        # The public API must write its Run/Outbox record to the private
        # stream within the same transaction.  Mapping at dispatcher time is
        # too late: a normal worker could otherwise claim the global row.
        container.run_manager = RunManager(
            session_factory=container.repositories.session_factory,
            runs=TransactionalRunRepository(
                container.repositories.session_factory,
                outbox_stream_name=isolated_broker.query_stream,
            ),
        )
        await container.redis.ping()
        await container.elasticsearch.info()
        async with container.mysql_engine.connect() as connection:
            await connection.execute(select(1))
        memory_service = container.memory_service
        if not settings.mem0_enabled or memory_service is None or not bool(
            getattr(memory_service, "available", False)
        ):
            raise RuntimeError("real Query acceptance requires an available Mem0 provider")
        parent_id = await _seed_document(
            container, settings, UserScope(user_id=settings.default_user_id)
        )
        dependencies = await build_query_dependencies(
            container,
            settings,
            child_index=f"agenticrag-children-{settings.index_generation}",
        )
        checkpoint_context = container.checkpoints.open_query()
        checkpointer = await checkpoint_context.__aenter__()
        worker = QueryWorker(
            runs=TransactionalRunRepository(container.repositories.session_factory),
            broker=isolated_broker,
            graph_factory=build_graph_factory(dependencies, checkpointer),
            worker_id=f"real-query-{suffix}",
            concurrency=dependencies.concurrency,
            trace_recorder=dependencies.trace_recorder,
            event_emitter=dependencies.event_emitter,
            block_ms=50,
            heartbeat_interval_seconds=0.5,
            lease_seconds=10,
            run_timeout_seconds=180,
            outbox_dispatcher=OutboxDispatcher(
                TransactionalQueryOutboxAdapter(
                    container.repositories.session_factory,
                    user_id=settings.default_user_id,
                    stream_name=isolated_broker.query_stream,
                ),
                isolated_broker,
                aggregate_type="query_run",
            ),
            outbox_interval_seconds=0.1,
        )
        worker_task = asyncio.create_task(worker.run_forever(stop_event=stop))

        app = create_app(settings, container=container)
        app_lifespan = app.router.lifespan_context(app)
        await app_lifespan.__aenter__()
        http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://real-query"
        )
        snapshot = container.runtime_snapshot
        emitter = dependencies.event_emitter
        if emitter is None:
            raise RuntimeError("real Query acceptance requires a durable event emitter")
        if emitter.runtime_config_snapshot_id != snapshot.snapshot_id:
            raise RuntimeError("Query Worker event emitter does not match the current snapshot")
        page = await http_client.get("/")
        if page.status_code != 200:
            raise RuntimeError(f"console page failed with HTTP {page.status_code}")
        memory_boundary, memory_marker = await _exercise_memory_boundary(
            memory_service, settings
        )
        await _assert_memory_api_boundary(http_client, marker=memory_marker)

        question = (
            "How many days notice does the acceptance document require before "
            "cancellation?"
        )
        created = await http_client.post(
            "/v1/query", json={"query": question, "wait_seconds": 30}
        )
        if created.status_code not in {200, 202}:
            raise RuntimeError(f"console Query API failed with HTTP {created.status_code}")
        created_payload = created.json()
        run_id = created_payload.get("run_id") if isinstance(created_payload, Mapping) else None
        if not isinstance(run_id, str) or not run_id.strip():
            raise RuntimeError("console Query API omitted a durable run ID")
        direct_run = await _wait_for_terminal(http_client, run_id)
        _assert_audited_answer(
            direct_run,
            parent_id=parent_id,
            snapshot_id=snapshot.snapshot_id,
        )
        await _emit_controlled_degradation(
            dependencies,
            run_id=run_id,
            user_id=settings.default_user_id,
            snapshot_id=snapshot.snapshot_id,
        )
        sse_events, sse_last_event_id = await _read_sse(http_client, run_id)
        _assert_safe_public_events(sse_events)
        degradation_events = _controlled_degradation_events(sse_events)

        case = EvaluationCase.model_validate(
            {
                "case_id": f"real-api-{suffix}",
                "user_id": settings.default_user_id,
                "question": question,
                "reference_answer": "The acceptance document requires thirty days notice before cancellation.",
                "reference_parent_ids": [parent_id],
                "expected_route": "fast_rag",
                "tags": ["real-provider", "api", "mem0"],
                "runtime_config_snapshot_id": snapshot.snapshot_id,
            }
        )
        client = HttpQueryClient(
            "http://real-query", http_client=http_client, timeout_seconds=180
        )
        recovery_ok = (await asyncio.to_thread(run_recovery_drill)).gate_passed
        backup_ok = await asyncio.to_thread(run_backup_restore_drill)
        summary = await EvalRunner(
            client,
            # EvalRunner writes intermediate rows as it goes.  Keep those in
            # the disposable root so a failed console gate can never leave a
            # separately-verifiable PASS report at the requested destination.
            output_dir=root / "evaluation",
            evaluation_mode="api",
            client_provenance=HttpQueryClient.provenance,
        ).run([case], recovery_drill_passed=recovery_ok, backup_restore_passed=backup_ok)
        if summary.get("runtime_config_snapshot_id") != snapshot.snapshot_id:
            raise RuntimeError("EvalRunner summary used a different runtime snapshot")
        if summary.get("client_provenance") != HttpQueryClient.provenance:
            raise RuntimeError("EvalRunner summary did not use the real Query API client")
        summary.update(
            {
                "console_page_status": page.status_code,
                "sse_last_event_id": sse_last_event_id,
                "console_run_snapshot_id": direct_run["runtime_config_snapshot_id"],
                "memory_provider_available": bool(
                    getattr(memory_service, "available", False)
                ),
                "memory_boundary": memory_boundary,
                "degradation_events": degradation_events,
            }
        )
        summary["memory_provider"] = {
            "enabled": settings.mem0_enabled,
            "available": bool(getattr(memory_service, "available", False)),
            "degraded": bool(getattr(memory_service, "degraded", True)),
        }
        if verify_acceptance(summary) != 0:
            raise RuntimeError("real API acceptance gates failed")
        if not console_acceptance_passed(summary):
            raise RuntimeError("real console acceptance gates failed")
        accepted_summary = dict(summary)
    except BaseException as error:
        body_error = error
        raise
    finally:
        teardown_failures: list[tuple[str, BaseException]] = []
        stop.set()
        if worker_task is not None:
            # Query execution failures are observed through the public Run
            # before acceptance can pass.  Teardown must still reach durable
            # cleanup if the background worker exits while another boundary
            # is already failing.
            await _record_teardown_error(
                teardown_failures, "Query Worker", worker_task
            )
        if http_client is not None:
            await _record_teardown_error(
                teardown_failures, "Query API client", http_client.aclose()
            )
        if app_lifespan is not None:
            await _record_teardown_error(
                teardown_failures,
                "Query API lifespan",
                app_lifespan.__aexit__(None, None, None),
            )
        if checkpoint_context is not None:
            await _record_teardown_error(
                teardown_failures,
                "Query checkpoint",
                checkpoint_context.__aexit__(None, None, None),
            )
        if dependencies is not None:
            await _record_teardown_error(
                teardown_failures,
                "Query dependencies",
                close_query_dependencies(dependencies),
            )
        if container is not None:
            await _cleanup_durable_boundaries(
                teardown_failures,
                container=container,
                settings=settings,
                broker=isolated_broker,
            )
            await _record_teardown_error(
                teardown_failures,
                "acceptance container",
                container.close(raise_on_error=True),
            )
        await _record_teardown_error(
            teardown_failures,
            "acceptance temporary root",
            asyncio.to_thread(shutil.rmtree, root),
        )
        if body_error is None:
            _raise_teardown_failures(teardown_failures)
    if accepted_summary is None:
        raise RuntimeError("real Query acceptance did not produce a verified summary")
    # Promote the report only after every owned client, stream, index and
    # database record has been cleaned successfully.  A teardown failure must
    # leave no current PASS report for verify_acceptance to reuse.
    write_summary(output / "summary.json", accepted_summary)
    return accepted_summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(run(args.output))
    except Exception as error:
        print(f"REAL QUERY ACCEPTANCE FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"REAL QUERY ACCEPTANCE PASSED: {args.output / 'summary.json'}")
    print(f"snapshot={summary.get('runtime_config_snapshot_id', '<runner-summary>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
