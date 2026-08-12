"""Configuration and narrow adapter for an injected ``mem0.AsyncMemory``.

The import is intentionally deferred: application startup owns third-party
construction while tests may pass a small in-process fake without installing
or contacting Mem0, Elasticsearch, or an embedding provider.
"""

from __future__ import annotations

from agentic_rag.memory.models import MemoryClient


MEMORY_COLLECTION = "agent_memories_v1"
QWEN_EMBEDDING_DIMENSIONS = 1024


def mem0_config(*, embedding_model: str = "text-embedding-v3") -> dict[str, object]:
    """Return the fixed collection/dimension configuration for AsyncMemory.

    The embedding provider endpoint and credentials are supplied by the
    process-level Mem0 configuration.  The model is Qwen-compatible and must
    keep the project's 1024-dimensional vector contract.
    """
    return {
        "vector_store": {
            "provider": "elasticsearch",
            "config": {
                "collection_name": MEMORY_COLLECTION,
                "embedding_model_dims": QWEN_EMBEDDING_DIMENSIONS,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "embedding_dims": QWEN_EMBEDDING_DIMENSIONS,
            },
        },
    }


class Mem0Adapter:
    """Keep the application dependent on a tiny, fake-friendly async surface."""

    def __init__(self, client: MemoryClient) -> None:
        self._client = client

    async def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, object],
    ) -> object:
        return await self._client.add(messages, user_id=user_id, metadata=metadata)

    async def search(self, query: str, *, user_id: str, limit: int) -> object:
        return await self._client.search(query, user_id=user_id, limit=limit)

    async def get_all(self, *, user_id: str) -> object:
        return await self._client.get_all(user_id=user_id)

    async def delete(self, memory_id: str) -> object:
        return await self._client.delete(memory_id)


def as_mem0_adapter(client: MemoryClient | Mem0Adapter) -> Mem0Adapter:
    """Avoid wrapping an adapter twice when the composition root owns it."""
    return client if isinstance(client, Mem0Adapter) else Mem0Adapter(client)
