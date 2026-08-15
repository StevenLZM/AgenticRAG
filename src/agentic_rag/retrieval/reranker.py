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


class RerankerUnavailable(RuntimeError):
    """An operational model failure that permits RRF-order degradation."""


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
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_concurrent_reranks,
            thread_name_prefix="agentic-rag-reranker",
        )
        self._closed = False

    @property
    def closed(self) -> bool:
        """Return whether this service has been closed."""
        return self._closed

    def close(self) -> None:
        """Stop the internally-owned executor after its active work completes."""
        if self._closed:
            return
        self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    async def aclose(self) -> None:
        """Asynchronously stop internally-owned worker threads."""
        await asyncio.to_thread(self.close)

    async def rerank(
        self, query: str, candidates: Sequence[ChildHit], *, limit: int = 10
    ) -> RerankResult:
        """Rerank up to thirty candidates or preserve RRF order when degraded."""
        if self._closed:
            raise RuntimeError("Reranker is closed")
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
        prediction: asyncio.Future[Sequence[float]] | None = None
        await self._semaphore.acquire()
        release_slot = True
        try:
            loop = asyncio.get_running_loop()
            prediction = loop.run_in_executor(self._executor, self.model.predict, pairs)
            try:
                scores = await asyncio.wait_for(
                    asyncio.shield(prediction), timeout=self._timeout_seconds
                )
            except asyncio.TimeoutError:
                prediction.add_done_callback(lambda _: self._semaphore.release())
                release_slot = False
                return RerankResult(
                    hits=fallback,
                    degraded=True,
                    model_version=self._model_version,
                )
            except (RerankerUnavailable, OSError):
                return RerankResult(
                    hits=fallback,
                    degraded=True,
                    model_version=self._model_version,
                )
        finally:
            if prediction is not None and not prediction.done() and release_slot:
                prediction.add_done_callback(lambda _: self._semaphore.release())
                release_slot = False
            if release_slot:
                self._semaphore.release()

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
