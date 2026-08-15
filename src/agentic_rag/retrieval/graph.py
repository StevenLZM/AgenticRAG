"""Fixed tenant-scoped hybrid retrieval graph with explicit degradation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import NodeCancelledError

from agentic_rag.domain.models import UserScope
from agentic_rag.models.embeddings import EmbeddingPort
from agentic_rag.observability.logging import emit_degradation
from agentic_rag.retrieval.filters import FilterBuilder
from agentic_rag.retrieval.fusion import rrf_fuse
from agentic_rag.retrieval.models import (
    ChildHit,
    EvidenceBatch,
    ParentEvidence,
    RetrievalRequest,
)
from agentic_rag.retrieval.parents import aggregate_parents
from agentic_rag.retrieval.ports import LexicalIndex, VectorIndex
from agentic_rag.retrieval.reranker import RerankResult
from agentic_rag.retrieval.state import LaneFailure, RetrievalState
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class RerankerPort(Protocol):
    """Minimal reranking boundary required by the retrieval graph."""

    async def rerank(
        self, query: str, candidates: Sequence[ChildHit], *, limit: int
    ) -> RerankResult: ...


class ParentHydrator(Protocol):
    """Parent boundary that preserves selection provenance during hydration."""

    async def hydrate(
        self, selected: Sequence[ParentEvidence], scope: UserScope
    ) -> list[ParentEvidence]: ...


class RetrievalUnavailable(RuntimeError):
    """Raised when no recall lane can safely supply evidence."""

    def __init__(self, failures: dict[str, LaneFailure]) -> None:
        self.failures = dict(failures)
        components = ", ".join(sorted(failures))
        super().__init__(f"all retrieval lanes failed: {components}")


@dataclass(frozen=True, slots=True)
class RetrievalDependencies:
    """Process-owned dependencies captured when the graph is compiled."""

    embedding: EmbeddingPort
    vector: VectorIndex
    lexical: LexicalIndex
    reranker: RerankerPort
    parent_fetcher: ParentHydrator
    filter_builder: FilterBuilder = field(default_factory=FilterBuilder)
    lane_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.lane_timeout_seconds <= 0:
            raise ValueError("lane_timeout_seconds must be positive")


class RetrievalService:
    """The sole Agent-facing entry point for tenant-scoped retrieval."""

    def __init__(self, dependencies: RetrievalDependencies) -> None:
        self._graph = build_retrieval_graph(dependencies)

    @property
    def graph(
        self,
    ) -> CompiledStateGraph[RetrievalState, None, RetrievalState, RetrievalState]:
        """Expose the compiled graph for runtime observability and tests."""
        return self._graph

    async def retrieve(
        self,
        request: RetrievalRequest,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> EvidenceBatch:
        """Run the immutable retrieval path and return safely scoped evidence."""
        try:
            result = await self._graph.ainvoke(
                {"request": request, "scope": scope, "snapshot": snapshot}
            )
        except NodeCancelledError as error:
            # LangGraph turns a node's deliberate CancelledError into its own
            # node failure. The service boundary restores cancellation so
            # callers can stop work rather than mistake it for degradation.
            raise asyncio.CancelledError() from error
        return cast(EvidenceBatch, result["evidence_batch"])


def build_retrieval_graph(
    dependencies: RetrievalDependencies,
) -> CompiledStateGraph[RetrievalState, None, RetrievalState, RetrievalState]:
    """Compile the non-agent-configurable retrieval subgraph.

    Dense and BM25 are parallel inside the ``recall`` node.  Their outcomes
    are materialized as state values before the deterministic sequential
    stages begin: RRF, rerank, parent selection, scoped hydration, and batch
    construction.
    """

    async def validate_and_filter(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        search_filter = dependencies.filter_builder.build(
            state["request"], state["scope"], state["snapshot"]
        )
        return {
            "search_filter": search_filter,
            "degraded_components": (),
            "lane_failures": {},
            "candidate_counts": {},
            "timings_ms": {"validate_and_filter": _elapsed_ms(started)},
        }

    async def recall(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        request = state["request"]
        search_filter = state["search_filter"]
        recall_limit = request.top_k_override or 40

        async def dense() -> list[ChildHit]:
            vector = await dependencies.embedding.embed_query(request.query)
            return await dependencies.vector.search(vector, search_filter, recall_limit)

        async def bm25() -> list[ChildHit]:
            return await dependencies.lexical.search(
                request.query, search_filter, recall_limit
            )

        dense_result, bm25_result = await asyncio.gather(
            _with_timeout(dense(), dependencies.lane_timeout_seconds),
            _with_timeout(bm25(), dependencies.lane_timeout_seconds),
            return_exceptions=True,
        )
        results = {"dense": dense_result, "bm25": bm25_result}
        _raise_cancellation_or_control_flow(results)
        _raise_input_errors(results)

        failures: dict[str, LaneFailure] = {}
        hits: dict[str, list[ChildHit]] = {}
        for component, result in results.items():
            if isinstance(result, Exception):
                failures[component] = _lane_failure(component, result)
            else:
                assert isinstance(result, list)
                hits[component] = result

        for component, failure in failures.items():
            await emit_degradation(
                component=component,
                reason=(
                    "lane_timeout"
                    if failure.error_type in {"TimeoutError", "CancelledError"}
                    else "lane_failure"
                ),
                run_id=None,
                snapshot_id=state["snapshot"].snapshot_id,
                attempt=1,
                retryable=True,
                outcome="degraded",
                event_type="RETRIEVAL_DEGRADED",
            )

        if len(failures) == len(results):
            raise RetrievalUnavailable(failures)

        degraded = tuple(
            component for component in ("dense", "bm25") if component in failures
        )
        timings = dict(state["timings_ms"])
        timings["recall"] = _elapsed_ms(started)
        return {
            "dense_hits": hits.get("dense", []),
            "bm25_hits": hits.get("bm25", []),
            "degraded_components": degraded,
            "lane_failures": failures,
            "candidate_counts": {
                "dense": len(hits.get("dense", [])),
                "bm25": len(hits.get("bm25", [])),
            },
            "timings_ms": timings,
        }

    async def rrf_fusion(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        fused = rrf_fuse(
            [state.get("dense_hits", []), state.get("bm25_hits", [])], limit=30
        )
        return _stage_update(
            state, "rrf_fusion", started, fused_hits=fused, rrf=len(fused)
        )

    async def cross_encoder(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        result = await dependencies.reranker.rerank(
            state["request"].query, state["fused_hits"], limit=10
        )
        degraded = state["degraded_components"]
        if result.degraded:
            await emit_degradation(
                component="reranker",
                reason="reranker_unavailable",
                run_id=None,
                snapshot_id=state["snapshot"].snapshot_id,
                attempt=1,
                retryable=True,
                outcome="degraded",
            )
            degraded = (*degraded, "reranker")
        return _stage_update(
            state,
            "cross_encoder",
            started,
            reranked_hits=result.hits,
            reranker=len(result.hits),
            degraded_components=degraded,
        )

    async def parent_aggregation(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        selected = aggregate_parents(
            state["reranked_hits"], max_children_per_parent=2, limit=6
        )
        return _stage_update(
            state,
            "parent_aggregation",
            started,
            selected_parents=selected,
            parents_selected=len(selected),
        )

    async def parent_fetch(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        hydrated = await dependencies.parent_fetcher.hydrate(
            state["selected_parents"], state["scope"]
        )
        return _stage_update(
            state,
            "parent_fetch",
            started,
            hydrated_parents=hydrated,
            parents_hydrated=len(hydrated),
        )

    async def evidence_batch(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        batch = EvidenceBatch(
            query=state["request"].query,
            parents=tuple(state["hydrated_parents"]),
            degraded_components=state["degraded_components"],
            document_ids=state["request"].document_ids,
        )
        return _stage_update(state, "evidence_batch", started, evidence_batch=batch)

    builder = StateGraph(RetrievalState)
    builder.add_node("validate_and_filter", validate_and_filter)
    builder.add_node("recall", recall)
    builder.add_node("rrf_fusion", rrf_fusion)
    builder.add_node("cross_encoder", cross_encoder)
    builder.add_node("parent_aggregation", parent_aggregation)
    builder.add_node("parent_fetch", parent_fetch)
    builder.add_node("evidence_batch", evidence_batch)
    builder.add_edge(START, "validate_and_filter")
    builder.add_edge("validate_and_filter", "recall")
    builder.add_edge("recall", "rrf_fusion")
    builder.add_edge("rrf_fusion", "cross_encoder")
    builder.add_edge("cross_encoder", "parent_aggregation")
    builder.add_edge("parent_aggregation", "parent_fetch")
    builder.add_edge("parent_fetch", "evidence_batch")
    builder.add_edge("evidence_batch", END)
    return builder.compile(name="RetrievalPipelineGraph")


async def _with_timeout(
    operation: Awaitable[list[ChildHit]], timeout_seconds: float
) -> list[ChildHit]:
    """Apply the configured timeout without obscuring the lane exception."""
    return await asyncio.wait_for(operation, timeout=timeout_seconds)


def _raise_input_errors(results: Mapping[str, object]) -> None:
    """Never recast invalid selectors or configuration as a degraded lane."""
    for result in results.values():
        if isinstance(result, ValueError):
            raise result


def _raise_cancellation_or_control_flow(results: Mapping[str, object]) -> None:
    """Propagate cancellation and interpreter control flow out of the graph."""
    for result in results.values():
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result


def _lane_failure(component: str, error: BaseException) -> LaneFailure:
    """Reduce an operational exception to safe degradation metadata."""
    return LaneFailure(
        component=component,
        error_type=type(error).__name__,
        message=str(error) or type(error).__name__,
    )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _stage_update(
    state: RetrievalState,
    stage: str,
    started: float,
    **update: object,
) -> dict[str, object]:
    """Attach one timing and optional candidate counts to a node update."""
    timings = dict(state["timings_ms"])
    timings[stage] = _elapsed_ms(started)
    counts = dict(state["candidate_counts"])
    for key in ("rrf", "reranker", "parents_selected", "parents_hydrated"):
        value = update.pop(key, None)
        if value is not None:
            counts[key] = cast(int, value)
    update["timings_ms"] = timings
    update["candidate_counts"] = counts
    return update
