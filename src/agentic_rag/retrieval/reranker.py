"""Bounded, degradable Cross-Encoder reranking."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag.retrieval.models import ChildHit


class CrossEncoder(Protocol):
    """Minimal local Cross-Encoder interface used by the reranking boundary."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class RerankerUnavailable(RuntimeError):
    """An operational model failure that permits RRF-order degradation."""


class RerankInputTooLong(RerankerUnavailable):
    """The full document and complete query cannot fit in at most two pairs."""


@dataclass(frozen=True, slots=True)
class QueryWindows:
    """Original body slices, with explicit complete coverage or impossibility."""

    windows: tuple[str, ...]
    complete: bool


class PairTokenizer(Protocol):
    """Process-owned tokenizer contract, including all pair special tokens."""

    max_pair_tokens: int

    def count_pair(self, query: str, document: str) -> int: ...

    def query_windows(
        self, body: str, *, query_prefix: str, document: str, max_windows: int
    ) -> QueryWindows: ...


class TokenizerPairAdapter:
    """Wrap an already-loaded fast tokenizer; never download or clip text.

    The composition owner must use the same tokenizer and effective capacity as
    the Cross-Encoder. Offset mappings split original query text losslessly.
    """

    def __init__(
        self, tokenizer: Callable[..., Mapping[str, Any]], *, max_pair_tokens: int
    ) -> None:
        if type(max_pair_tokens) is not int or max_pair_tokens <= 0:
            raise ValueError("max_pair_tokens must be a positive integer")
        self.max_pair_tokens = max_pair_tokens
        self._tokenizer = tokenizer

    def count_pair(self, query: str, document: str) -> int:
        encoded = self._tokenizer(
            query, document, add_special_tokens=True, truncation=False
        )
        return len(encoded["input_ids"])

    def query_windows(
        self, body: str, *, query_prefix: str, document: str, max_windows: int = 2
    ) -> QueryWindows:
        if max_windows not in {1, 2}:
            raise ValueError("query windows must be bounded to at most two")
        if self.count_pair(query_prefix + body, document) <= self.max_pair_tokens:
            return QueryWindows((body,), complete=True)
        if max_windows == 1 or not body:
            return QueryWindows((), complete=False)
        encoded = self._tokenizer(
            body,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True,
        )
        boundaries: set[int] = set()
        for start, end in encoded["offset_mapping"]:
            if not 0 <= start <= end <= len(body):
                raise ValueError("tokenizer returned invalid query offsets")
            if 0 < end < len(body):
                boundaries.add(end)
        # Prefer balanced complete windows. Each actual pair is measured because
        # prefixes, separators, and special tokens need not count additively.
        for boundary in sorted(boundaries, key=lambda n: (abs(n - len(body) / 2), n)):
            windows = (body[:boundary], body[boundary:])
            if all(
                self.count_pair(query_prefix + window, document) <= self.max_pair_tokens
                for window in windows
            ):
                return QueryWindows(windows, complete=True)
        return QueryWindows((), complete=False)


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Reranked hits and observability data for the retrieval pipeline."""

    hits: list[ChildHit]
    degraded: bool
    model_version: str
    degradation_reason: str | None = None
    window_counts: tuple[int, ...] = ()


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
        max_candidates: int = 30,
        pair_tokenizer: PairTokenizer | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_concurrent_reranks <= 0:
            raise ValueError("max_concurrent_reranks must be positive")
        if type(max_candidates) is not int or not 1 <= max_candidates <= 200:
            raise ValueError("max_candidates must be in [1, 200]")
        self.max_candidates = max_candidates
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
        self._pair_tokenizer = pair_tokenizer

    def _predict(
        self, query: str, input_hits: Sequence[ChildHit], query_prefix: str, allow_query_windows: bool
    ) -> tuple[list[float], tuple[int, ...]]:
        pairs: list[tuple[str, str]] = []
        counts: list[int] = []
        for hit in input_hits:
            document = hit.search_content
            windows = (query,)
            tokenizer = self._pair_tokenizer
            if tokenizer is not None:
                if tokenizer.count_pair(query, document) > tokenizer.max_pair_tokens:
                    if not allow_query_windows:
                        raise RerankInputTooLong("rerank_input_too_long")
                    plan = tokenizer.query_windows(
                        query[len(query_prefix) :],
                        query_prefix=query_prefix,
                        document=document,
                        max_windows=2,
                    )
                    if not plan.complete:
                        raise RerankInputTooLong("rerank_input_too_long")
                    if (
                        not 1 <= len(plan.windows) <= 2
                        or "".join(plan.windows) != query[len(query_prefix) :]
                        or any(not window for window in plan.windows)
                    ):
                        raise ValueError("query window coverage mismatch")
                    windows = tuple(query_prefix + window for window in plan.windows)
                    if any(
                        tokenizer.count_pair(window, document)
                        > tokenizer.max_pair_tokens
                        for window in windows
                    ):
                        raise RerankInputTooLong("rerank_input_too_long")
            pairs.extend((window, document) for window in windows)
            counts.append(len(windows))
        # No inference runs until every candidate's complete text passes preflight.
        scores = self.model.predict(pairs)
        if len(scores) != len(pairs):
            raise ValueError("Cross-Encoder returned a score count mismatch")
        aggregated: list[float] = []
        offset = 0
        for count in counts:
            aggregated.append(
                max(float(score) for score in scores[offset : offset + count])
            )
            offset += count
        return aggregated, tuple(counts)

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
        self,
        query: str,
        candidates: Sequence[ChildHit],
        *,
        limit: int = 10,
        protected_query_context: str = "",
        allow_query_windows: bool = True,
    ) -> RerankResult:
        """Rerank up to thirty candidates or preserve RRF order when degraded."""
        if self._closed:
            raise RuntimeError("Reranker is closed")
        if limit < 0:
            raise ValueError("limit must be non-negative")
        query_prefix = (
            protected_query_context + "\n\n" if protected_query_context else ""
        )
        if query_prefix and not query.startswith(query_prefix):
            raise ValueError("protected query context must be the exact query prefix")

        input_hits = list(candidates[: self.max_candidates])
        result_limit = min(limit, self.max_candidates)
        fallback = input_hits[:result_limit]
        if not input_hits or result_limit == 0:
            return RerankResult(
                hits=fallback,
                degraded=False,
                model_version=self._model_version,
            )

        prediction: asyncio.Future[tuple[list[float], tuple[int, ...]]] | None = None
        await self._semaphore.acquire()
        release_slot = True
        try:
            loop = asyncio.get_running_loop()
            prediction = loop.run_in_executor(
                self._executor, self._predict, query, input_hits, query_prefix, allow_query_windows
            )
            try:
                scores, window_counts = await asyncio.wait_for(
                    asyncio.shield(prediction), timeout=self._timeout_seconds
                )
            except asyncio.TimeoutError:
                prediction.add_done_callback(lambda _: self._semaphore.release())
                release_slot = False
                return RerankResult(
                    hits=fallback,
                    degraded=True,
                    model_version=self._model_version,
                    degradation_reason="reranker_unavailable",
                )
            except RerankInputTooLong:
                return RerankResult(
                    hits=fallback,
                    degraded=True,
                    model_version=self._model_version,
                    degradation_reason="rerank_input_too_long",
                )
            except (RerankerUnavailable, OSError):
                return RerankResult(
                    hits=fallback,
                    degraded=True,
                    model_version=self._model_version,
                    degradation_reason="reranker_unavailable",
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
            hit.model_copy(update={"rerank_score": float(scores[position])})
            for position, hit in sorted(
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
            window_counts=window_counts,
        )
