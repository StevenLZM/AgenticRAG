"""Ports for the independently replaceable retrieval lanes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agentic_rag.retrieval.models import ChildHit, SearchFilter


class VectorIndex(Protocol):
    """Search Child chunks by their query embedding."""

    async def search(
        self,
        query_vector: Sequence[float],
        filter: SearchFilter,
        top_k: int,
    ) -> list[ChildHit]: ...


class LexicalIndex(Protocol):
    """Search Child chunks with lexical relevance."""

    async def search(
        self,
        query_text: str,
        filter: SearchFilter,
        top_k: int,
    ) -> list[ChildHit]: ...
