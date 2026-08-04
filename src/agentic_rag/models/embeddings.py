"""Embedding provider boundary.

Concrete Qwen/OpenAI-compatible clients are process-owned adapters. They are not
part of LangGraph state and are deliberately absent from this protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingPort(Protocol):
    """Minimal asynchronous contract shared by ingestion and retrieval."""

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
