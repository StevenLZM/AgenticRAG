"""Bounded, degradable Cross-Encoder reranking."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from agentic_rag.retrieval.models import ChildHit


class CrossEncoder(Protocol):
    """Minimal local Cross-Encoder interface used by the reranking boundary."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Reranked hits and observability data for the retrieval pipeline."""

    hits: list[ChildHit]
    degraded: bool
    model_version: str


class Reranker:
    """Run injected Cross-Encoder inference without blocking the event loop."""

    max_candidates = 30

    def __init__(
        self,
        model: CrossEncoder,
        *,
        model_version: str,
        timeout_seconds: float = 30.0,
        max_concurrent_reranks: int = 1,
        executor: Executor | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_concurrent_reranks <= 0:
            raise ValueError("max_concurrent_reranks must be positive")
        self.model = model
        self._model_version = model_version
        self._timeout_seconds = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_reranks)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_concurrent_reranks,
            thread_name_prefix="agentic-rag-reranker",
        )

    async def rerank(
        self, query: str, candidates: Sequence[ChildHit], *, limit: int = 10
    ) -> RerankResult:
        """Rerank up to thirty candidates or preserve RRF order when degraded."""
        if limit < 0:
            raise ValueError("limit must be non-negative")

        input_hits = list(candidates[: self.max_candidates])
        result_limit = min(limit, self.max_candidates)
        fallback = input_hits[:result_limit]
        if not input_hits or result_limit == 0:
            return RerankResult(
                hits=fallback,
                degraded=False,
                model_version=self._model_version,
            )

        pairs = [(query, hit.content) for hit in input_hits]
        try:
            async with self._semaphore:
                loop = asyncio.get_running_loop()
                prediction = loop.run_in_executor(
                    self._executor, self.model.predict, pairs
                )
                scores = await asyncio.wait_for(
                    prediction, timeout=self._timeout_seconds
                )
            if len(scores) != len(input_hits):
                raise ValueError("Cross-Encoder returned a score count mismatch")
            ranked = [
                hit
                for _, hit in sorted(
                    enumerate(input_hits),
                    key=lambda indexed_hit: (
                        -float(scores[indexed_hit[0]]),
                        indexed_hit[0],
                    ),
                )
            ]
            return RerankResult(
                hits=ranked[:result_limit],
                degraded=False,
                model_version=self._model_version,
            )
        except (asyncio.TimeoutError, Exception):
            return RerankResult(
                hits=fallback,
                degraded=True,
                model_version=self._model_version,
            )
