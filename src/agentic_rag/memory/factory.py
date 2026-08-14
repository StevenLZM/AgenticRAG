"""Production Mem0 composition and fail-closed memory fallback.

The application imports Mem0 only when the setting explicitly enables it.  This
keeps local/unit startup independent of the optional provider while making a
misconfigured enabled provider visible through readiness and structured logs.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import Any, cast
from urllib.parse import urlsplit

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.mem0_adapter import Mem0Adapter
from agentic_rag.memory.models import (
    MemoryContext,
    MemoryExtractor,
    MemoryRecord,
    MemoryTombstoneStore,
    Tombstone,
)
from agentic_rag.memory.service import MemoryService, MemoryServiceImpl, ModelGatewayMemoryExtractor
from agentic_rag.persistence.repositories import SqlAlchemyMemoryTombstoneRepository
from agentic_rag.runtime.model_gateway import ModelGateway
from agentic_rag.runtime.models import RuntimeConfigSnapshot

logger = logging.getLogger(__name__)

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
class MemoryCompositionError(RuntimeError):
    """Raised when an enabled Mem0 provider cannot be composed safely."""


class UnavailableMemoryService:
    """A visible no-op boundary used when memory is disabled or unavailable."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.available = False
        self.degraded = True
        self._owned_resources: tuple[object, ...] = ()

    async def load_context(
        self, scope: UserScope, query: str, limit: int = 10
    ) -> MemoryContext:
        del scope, query, limit
        return MemoryContext(degraded=True)

    async def extract_and_store(
        self, scope: UserScope, run_id: str, messages: object
    ) -> None:
        del scope, run_id, messages

    async def list(self, scope: UserScope) -> list[MemoryRecord]:
        del scope
        raise OSError(f"memory provider unavailable: {self.reason}")

    async def delete(self, scope: UserScope, memory_id: str) -> None:
        del scope, memory_id
        raise OSError(f"memory provider unavailable: {self.reason}")

    async def reconcile_deletions(self) -> None:
        return None


class _TransactionalTombstones(MemoryTombstoneStore):
    """Bind the existing SQL tombstone repository to short transactions."""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    async def request(self, scope: UserScope, memory_id: str) -> Tombstone:
        async with self._factory.begin() as session:
            item = await SqlAlchemyMemoryTombstoneRepository(session).request(
                scope, memory_id
            )
        return _to_tombstone(item)

    async def list_pending(self, limit: int = 100) -> list[Tombstone]:
        async with self._factory() as session:
            items = await SqlAlchemyMemoryTombstoneRepository(session).list_pending(
                limit
            )
        return [_to_tombstone(item) for item in items]

    async def mark_completed(self, scope: UserScope, memory_id: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyMemoryTombstoneRepository(session).mark_completed(
                scope, memory_id
            )

    async def mark_retry(self, scope: UserScope, memory_id: str, error: str) -> None:
        async with self._factory.begin() as session:
            await SqlAlchemyMemoryTombstoneRepository(session).mark_retry(
                scope, memory_id, error
            )


def build_mem0_config(settings: object) -> dict[str, object]:
    """Build the strict Mem0 configuration from process-owned settings."""
    collection = _required_string(settings, "mem0_collection")
    if not _COLLECTION_RE.fullmatch(collection):
        raise MemoryCompositionError("mem0_collection contains unsafe characters")
    elasticsearch_url = _required_string(settings, "elasticsearch_url")
    parsed = urlsplit(elasticsearch_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise MemoryCompositionError("elasticsearch_url must be an http(s) URL")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 9200)
    except ValueError as error:
        raise MemoryCompositionError("elasticsearch_url has an invalid port") from error

    embedder_base_url = _required_string(settings, "mem0_embedding_base_url")
    embedder_api_key = _credential(
        getattr(settings, "mem0_embedding_api_key", None),
        "mem0_embedding_api_key",
    )
    embedding_model = _required_string(settings, "mem0_embedding_model")
    dimensions = getattr(settings, "embedding_dimensions", 1024)
    if dimensions != 1024:
        raise MemoryCompositionError("Mem0 requires 1024-dimensional embeddings")

    vector_config: dict[str, object] = {
        "collection_name": collection,
        # Mem0's Elasticsearch adapter concatenates host + port and the
        # elastic client requires the resulting host to include a scheme.
        "host": f"{parsed.scheme}://{parsed.hostname}",
        "port": port,
        "embedding_model_dims": 1024,
        "use_ssl": parsed.scheme == "https",
        "verify_certs": bool(getattr(settings, "mem0_elasticsearch_verify_certs", True)),
        "auto_create_index": True,
    }
    es_api_key = _credential(
        getattr(settings, "mem0_elasticsearch_api_key", None),
        "mem0_elasticsearch_api_key",
        required=False,
    )
    es_user = _optional_string(getattr(settings, "mem0_elasticsearch_user", None))
    es_password = _credential(
        getattr(settings, "mem0_elasticsearch_password", None),
        "mem0_elasticsearch_password",
        required=False,
    )
    if es_api_key:
        vector_config["api_key"] = es_api_key
    elif es_user and es_password:
        vector_config["user"] = es_user
        vector_config["password"] = es_password
    else:
        raise MemoryCompositionError(
            "Mem0 Elasticsearch authentication is required (API key or user/password)"
        )

    config: dict[str, object] = {
        "vector_store": {"provider": "elasticsearch", "config": vector_config},
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "api_key": embedder_api_key,
                "openai_base_url": embedder_base_url,
                "embedding_dims": 1024,
            },
        },
        "history_db_path": str(
            getattr(settings, "mem0_history_db_path", "var/mem0/history.db")
        ),
    }
    if bool(getattr(settings, "mem0_llm_enabled", False)):
        llm_base_url = _required_string(settings, "mem0_llm_base_url")
        llm_api_key = _credential(
            getattr(settings, "mem0_llm_api_key", None), "mem0_llm_api_key"
        )
        llm_model = _required_string(settings, "mem0_llm_model")
        config["llm"] = {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": llm_api_key,
                "openai_base_url": llm_base_url,
            },
        }
    else:
        # Mem0 2.x eagerly constructs an LLM even when every application call
        # uses infer=False.  LM Studio's local OpenAI-compatible client has no
        # credential requirement and is never called in this mode; keeping it
        # explicit prevents the SDK from implicitly selecting cloud OpenAI.
        config["llm"] = {
            "provider": "lmstudio",
            "config": {
                "model": "mem0-infer-disabled",
                "lmstudio_base_url": "http://127.0.0.1:1234/v1",
            },
        }
    return config


def _build_memory_service_sync(
    container: object,
    settings: object,
    snapshot: RuntimeConfigSnapshot,
    extractor: MemoryExtractor | None = None,
) -> MemoryService:
    if not bool(getattr(settings, "mem0_enabled", False)):
        logger.warning(
            "memory_provider_degraded",
            extra={"component": "mem0", "reason": "disabled", "outcome": "degraded"},
        )
        return UnavailableMemoryService("disabled")  # type: ignore[return-value]

    config = build_mem0_config(settings)
    try:
        from mem0 import AsyncMemory  # type: ignore[import-untyped]
    except ImportError as error:
        raise MemoryCompositionError("mem0ai is not installed in the runtime environment") from error
    try:
        async_memory = AsyncMemory.from_config(config)
    except Exception as error:
        raise MemoryCompositionError(
            f"Mem0 provider construction failed: {type(error).__name__}"
        ) from error

    repositories = getattr(container, "repositories", None)
    factory = getattr(repositories, "session_factory", None)
    if factory is None:
        _close_sync(async_memory)
        raise MemoryCompositionError("memory composition requires a MySQL session factory")

    owned_resources: list[object] = [async_memory]
    if extractor is None:
        deepseek_key = _credential(
            getattr(settings, "deepseek_api_key", None), "deepseek_api_key"
        )
        deepseek_base_url = _required_string(settings, "deepseek_base_url")
        client = AsyncOpenAI(
            api_key=deepseek_key,
            base_url=deepseek_base_url,
            timeout=30.0,
            max_retries=0,
        )
        extractor = ModelGatewayMemoryExtractor(ModelGateway(client), snapshot)
        owned_resources.insert(0, client)
    service = MemoryServiceImpl(
        Mem0Adapter(async_memory),
        tombstones=_TransactionalTombstones(factory),
        policy_version="memory-v1",
        extractor=extractor,
    )
    # AppContainer.close discovers these without exposing provider objects to
    # graph state or the public MemoryService protocol.
    service._owned_resources = tuple(owned_resources)  # type: ignore[attr-defined]
    service.available = True  # type: ignore[attr-defined]
    service.degraded = False  # type: ignore[attr-defined]
    return service


async def build_memory_service(
    container: object,
    settings: object,
    snapshot: RuntimeConfigSnapshot,
    *,
    extractor: MemoryExtractor | None = None,
) -> MemoryService:
    """Build the enabled provider off the event loop; disabled mode is local."""
    return await asyncio.to_thread(
        _build_memory_service_sync, container, settings, snapshot, extractor
    )


def build_memory_service_sync(
    container: object,
    settings: object,
    snapshot: RuntimeConfigSnapshot,
    *,
    extractor: MemoryExtractor | None = None,
) -> MemoryService:
    """Synchronous composition hook used before an async application starts."""
    return _build_memory_service_sync(container, settings, snapshot, extractor)


def _to_tombstone(item: object) -> Tombstone:
    status = str(getattr(item, "status"))
    if status not in {"pending", "completed", "failed"}:
        status = "pending"
    return Tombstone(
        user_id=str(getattr(item, "user_id")),
        memory_id=str(getattr(item, "memory_id")),
        status=cast(Any, status),
    )


def _required_string(settings: object, name: str) -> str:
    value = _optional_string(getattr(settings, name, None))
    if not value or value.startswith("replace-with-"):
        raise MemoryCompositionError(f"{name} is required")
    return value


def _credential(value: object, name: str, *, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise MemoryCompositionError(f"{name} is required")
        return None
    raw = value.get_secret_value() if hasattr(value, "get_secret_value") else value
    if not isinstance(raw, str) or not raw.strip() or raw.strip().startswith("replace-with-"):
        if required:
            raise MemoryCompositionError(f"{name} is required")
        return None
    return raw.strip()


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _close_sync(resource: object) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        value = close()
        if inspect.isawaitable(value):
            raise MemoryCompositionError("provider close unexpectedly requires an event loop")


__all__ = [
    "MemoryCompositionError",
    "UnavailableMemoryService",
    "build_mem0_config",
    "build_memory_service",
    "build_memory_service_sync",
]
