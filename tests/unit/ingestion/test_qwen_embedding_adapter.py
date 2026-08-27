"""Provider batch limits for the production ingestion embedding adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from scripts.run_ingestion_worker import QwenEmbeddingAdapter


class _BatchLimitedEmbeddings:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def create(self, **kwargs: Any) -> Any:
        values = kwargs["input"]
        texts = (values,) if isinstance(values, str) else tuple(values)
        self.calls.append(texts)
        if len(texts) > 10:
            raise ValueError("batch size is invalid, it should not be larger than 10")
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(index)]) for index, _ in enumerate(texts)]
        )


class _BatchLimitedClient:
    def __init__(self, embeddings: _BatchLimitedEmbeddings) -> None:
        self.embeddings = embeddings


@pytest.mark.asyncio
async def test_embedding_adapter_splits_dashscope_batches_at_ten() -> None:
    embeddings = _BatchLimitedEmbeddings()
    adapter = object.__new__(QwenEmbeddingAdapter)
    adapter._model = "text-embedding-v3"
    adapter._dimensions = 1024
    adapter._client = _BatchLimitedClient(embeddings)

    vectors = await adapter.embed_documents([f"text-{index}" for index in range(11)])

    assert len(vectors) == 11
    assert [len(call) for call in embeddings.calls] == [10, 1]
