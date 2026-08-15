"""Production Mem0 composition contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.memory.models import MemoryContext
from agentic_rag.runtime.models import RuntimeConfigSnapshot


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "mem0_enabled": True,
        "mem0_collection": "agent_memories_v1",
        "mem0_embedding_base_url": "https://qwen.example/v1",
        "mem0_embedding_api_key": "qwen-key",
        "mem0_embedding_model": "text-embedding-v3",
        "mem0_llm_enabled": False,
        "mem0_llm_model": "deepseek-v4-flash",
        "mem0_llm_base_url": "https://deepseek.example/v1",
        "mem0_llm_api_key": "deepseek-key",
        "elasticsearch_url": "http://127.0.0.1:9200",
        "mem0_elasticsearch_api_key": "local-key",
        "mem0_history_db_path": "/tmp/agentic-rag-mem0-history.db",
        "embedding_dimensions": 1024,
        "embedding_model": "text-embedding-v3",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="query-v1",
    prompt_version="prompt-v1",
    main_model_id="main",
    light_model_id="light",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="reranker",
    retrieval_config_version="retrieval-v1",
    index_generation="index-v2",
    memory_config_version="agent_memories_v1-v1",
)


def test_mem0_config_contains_scoped_elasticsearch_and_qwen_embedding() -> None:
    from agentic_rag.memory.factory import build_mem0_config

    config = build_mem0_config(_settings())

    assert config["vector_store"]["config"]["collection_name"] == "agent_memories_v1"
    assert config["vector_store"]["config"]["embedding_model_dims"] == 1024
    assert config["embedder"]["config"]["openai_base_url"] == "https://qwen.example/v1"
    assert config["embedder"]["config"]["embedding_dims"] == 1024
    assert "llm" not in config or config["llm"]["config"].get("api_key") is None


def test_mem0_config_falls_back_to_qwen_embedding_and_allows_loopback_without_auth() -> None:
    from agentic_rag.memory.factory import build_mem0_config

    settings = _settings(
        mem0_embedding_base_url=None,
        mem0_embedding_api_key=None,
        qwen_embedding_base_url="https://qwen.example/v1",
        qwen_api_key="qwen-fallback-key",
        mem0_elasticsearch_api_key=None,
        mem0_elasticsearch_user=None,
        mem0_elasticsearch_password=None,
        elasticsearch_url="http://localhost:9200",
    )

    config = build_mem0_config(settings)

    assert config["embedder"]["config"]["openai_base_url"] == "https://qwen.example/v1"
    assert config["embedder"]["config"]["api_key"] == "qwen-fallback-key"
    assert "api_key" not in config["vector_store"]["config"]
    assert "user" not in config["vector_store"]["config"]


def test_mem0_config_rejects_unauthenticated_remote_elasticsearch() -> None:
    from agentic_rag.memory.factory import MemoryCompositionError, build_mem0_config

    settings = _settings(
        mem0_elasticsearch_api_key=None,
        mem0_elasticsearch_user=None,
        mem0_elasticsearch_password=None,
        elasticsearch_url="https://search.example:9243",
    )

    with pytest.raises(MemoryCompositionError, match="authentication"):
        build_mem0_config(settings)


@pytest.mark.asyncio
async def test_disabled_mem0_returns_degraded_service_without_provider_import() -> None:
    from agentic_rag.memory.factory import build_memory_service

    service = await build_memory_service(
        SimpleNamespace(memory_tombstones=None),
        _settings(mem0_enabled=False),
        SNAPSHOT,
    )

    context = await service.load_context(UserScope(user_id="u1"), "query")
    assert isinstance(context, MemoryContext)
    assert context.degraded is True


def test_mem0_adapter_uses_filter_namespace_and_disables_provider_inference() -> None:
    from agentic_rag.memory.mem0_adapter import Mem0Adapter

    class StrictAsyncMemory:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        async def add(
            self,
            messages: list[dict[str, str]],
            *,
            user_id: str,
            metadata: dict[str, object],
            infer: bool,
        ) -> None:
            self.calls.append(("add", (messages, user_id, metadata, infer)))

        async def search(
            self, query: str, *, top_k: int, filters: dict[str, str]
        ) -> dict[str, object]:
            self.calls.append(("search", (query, top_k, filters)))
            return {"results": []}

        async def get_all(
            self, *, top_k: int, filters: dict[str, str]
        ) -> dict[str, object]:
            self.calls.append(("get_all", (top_k, filters)))
            return {"results": []}

        async def delete(self, memory_id: str) -> None:
            self.calls.append(("delete", memory_id))

    async def exercise() -> list[tuple[str, object]]:
        client = StrictAsyncMemory()
        adapter = Mem0Adapter(client)  # type: ignore[arg-type]
        await adapter.add(
            [{"role": "user", "content": "preference"}],
            user_id="u1",
            metadata={"policy_version": "v1"},
        )
        await adapter.search("preference", user_id="u1", limit=7)
        await adapter.get_all(user_id="u1")
        return client.calls

    calls = __import__("asyncio").run(exercise())
    assert calls[0][0] == "add"
    assert calls[0][1][-1] is False
    assert calls[1][1][-1] == {"user_id": "u1"}
    assert calls[2][1][-1] == {"user_id": "u1"}


@pytest.mark.asyncio
async def test_enabled_mem0_builds_provider_and_light_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mem0

    from agentic_rag.memory.factory import build_memory_service

    class FakeAsyncMemory:
        config: dict[str, object] | None = None

        @classmethod
        def from_config(cls, config: dict[str, object]) -> "FakeAsyncMemory":
            instance = cls()
            instance.config = config
            return instance

        def close(self) -> None:
            return None

    monkeypatch.setattr(mem0, "AsyncMemory", FakeAsyncMemory)
    settings = _settings(
        deepseek_base_url="https://deepseek.example/v1",
        deepseek_api_key="deepseek-key",
    )
    container = SimpleNamespace(
        repositories=SimpleNamespace(session_factory=object())
    )

    service = await build_memory_service(container, settings, SNAPSHOT)

    assert getattr(service, "available") is True
    assert getattr(service, "_extractor") is not None
    assert getattr(service, "_mem0")._client.config["vector_store"]["config"]["host"] == "http://127.0.0.1"
