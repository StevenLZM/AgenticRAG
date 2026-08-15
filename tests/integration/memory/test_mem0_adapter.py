"""Opt-in real Mem0/Elasticsearch tenant and deletion contract."""

from __future__ import annotations

import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agentic_rag.config import Settings
from agentic_rag.domain.models import UserScope
from agentic_rag.memory.factory import build_memory_service
from agentic_rag.memory.models import MemoryCandidate, MemoryType, PublicMessage
from agentic_rag.persistence.mysql import create_mysql_engine, create_session_factory
from agentic_rag.runtime.query_composition import build_query_snapshot


def _required_environment() -> dict[str, str] | None:
    if os.environ.get("AGENTIC_RAG_TEST_MEM0_ENABLED") != "1":
        return None
    names = (
        "AGENTIC_RAG_TEST_MYSQL_DSN",
        "AGENTIC_RAG_TEST_ELASTICSEARCH_URL",
        "AGENTIC_RAG_TEST_MEM0_EMBEDDING_BASE_URL",
        "AGENTIC_RAG_TEST_MEM0_EMBEDDING_API_KEY",
        "AGENTIC_RAG_TEST_MEM0_ELASTICSEARCH_API_KEY",
    )
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip("missing explicit Mem0 fixture settings: " + ", ".join(missing))
    return values


class _DeterministicExtractor:
    def __init__(self, text: str) -> None:
        self._candidate = MemoryCandidate(
            text=text,
            memory_type=MemoryType.SEMANTIC,
            source_message_ids=("source-message",),
        )

    async def extract(self, messages: list[PublicMessage]) -> tuple[MemoryCandidate, ...]:
        assert messages[0].id == "source-message"
        return (self._candidate,)


async def _close_service(service: object) -> None:
    resources = getattr(service, "_owned_resources", ())
    for resource in reversed(tuple(resources)):
        close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if callable(close):
            value = close()
            if hasattr(value, "__await__"):
                await value


@pytest.mark.integration
@pytest.mark.e2e
async def _run_real_mem0_contract() -> None:
    values = _required_environment()
    if values is None:
        pytest.skip("set AGENTIC_RAG_TEST_MEM0_ENABLED=1 with explicit Mem0/ES settings")

    settings = Settings(
        mysql_dsn=values["AGENTIC_RAG_TEST_MYSQL_DSN"],
        redis_url=os.environ.get(
            "AGENTIC_RAG_TEST_REDIS_DSN", "redis://127.0.0.1:6379/15"
        ),
        elasticsearch_url=values["AGENTIC_RAG_TEST_ELASTICSEARCH_URL"],
        deepseek_base_url=os.environ.get(
            "AGENTIC_RAG_DEEPSEEK_BASE_URL", "https://models.invalid/v1"
        ),
        deepseek_api_key=os.environ.get("AGENTIC_RAG_DEEPSEEK_API_KEY", "test-key"),
        qwen_embedding_base_url=os.environ.get(
            "AGENTIC_RAG_QWEN_EMBEDDING_BASE_URL", "https://embeddings.invalid/v1"
        ),
        qwen_api_key=os.environ.get("AGENTIC_RAG_QWEN_API_KEY", "test-key"),
        mem0_enabled=True,
        mem0_embedding_base_url=values["AGENTIC_RAG_TEST_MEM0_EMBEDDING_BASE_URL"],
        mem0_embedding_api_key=values["AGENTIC_RAG_TEST_MEM0_EMBEDDING_API_KEY"],
        mem0_elasticsearch_api_key=values[
            "AGENTIC_RAG_TEST_MEM0_ELASTICSEARCH_API_KEY"
        ],
        mem0_collection=f"agent_memories_e2e_{uuid4().hex[:12]}",
    )
    snapshot = build_query_snapshot(settings)
    engine = create_mysql_engine(settings.mysql_dsn, pool_pre_ping=True)
    session_factory = create_session_factory(engine)
    container = SimpleNamespace(
        repositories=SimpleNamespace(session_factory=session_factory)
    )
    text = f"e2e preference {uuid4().hex}"
    service = await build_memory_service(
        container,
        settings,
        snapshot,
        extractor=_DeterministicExtractor(text),
    )
    user = UserScope(user_id=f"mem0-e2e-{uuid4().hex}")
    other_user = UserScope(user_id=f"mem0-other-{uuid4().hex}")
    try:
        await service.extract_and_store(
            user,
            f"run-{uuid4().hex}",
            [PublicMessage(id="source-message", role="user", content=text)],
        )
        records = await service.list(user)
        assert [record.text for record in records] == [text]
        assert await service.list(other_user) == []

        await service.delete(user, records[0].id)
        assert await service.list(user) == []
        await service.reconcile_deletions()
    finally:
        await _close_service(service)
        await engine.dispose()


@pytest.mark.integration
@pytest.mark.e2e
async def test_local_mem0_and_elasticsearch_contract_is_opt_in() -> None:
    await _run_real_mem0_contract()
