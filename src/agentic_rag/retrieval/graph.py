"""Fixed tenant-scoped hybrid retrieval graph with explicit degradation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol, cast, TYPE_CHECKING

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.errors import NodeCancelledError

from agentic_rag.domain.models import UserScope
from agentic_rag.models.embeddings import EmbeddingPort
from agentic_rag.observability.logging import emit_degradation
from agentic_rag.retrieval.filters import FilterBuilder
from agentic_rag.retrieval.query_compiler import QueryCompiler
from agentic_rag.retrieval.query_context import QueryBusinessContext
from agentic_rag.retrieval.fusion import rrf_fuse
from agentic_rag.retrieval.models import (
    ChildHit,
    EvidenceBatch,
    ParentEvidence,
    RetrievalRequest,
    RankedHitRef,
    RetrievalObservation,
    RankedCandidates,
)
from agentic_rag.retrieval.parents import aggregate_parents
from agentic_rag.retrieval.ports import LexicalIndex, VectorIndex
from agentic_rag.retrieval.reranker import RerankResult
from agentic_rag.retrieval.state import LaneFailure, RetrievalState
from agentic_rag.runtime.models import RetrievalBudget, RuntimeConfigSnapshot


if TYPE_CHECKING:
    from agentic_rag.retrieval.context_evidence import ContextSourceHydrator


class RerankerPort(Protocol):
    """Minimal reranking boundary required by the retrieval graph."""

    async def rerank(
        self, query: str, candidates: Sequence[ChildHit], *, limit: int,
        protected_query_context: str = "", allow_query_windows: bool = True,
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


def _budget(state: RetrievalState) -> RetrievalBudget:
    evaluation = state["snapshot"].evaluation
    return evaluation.retrieval_budget if evaluation and evaluation.retrieval_budget else RetrievalBudget()


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
    query_compiler: QueryCompiler | None = None
    context_sources: ContextSourceHydrator | None = None

    def __post_init__(self) -> None:
        if self.lane_timeout_seconds <= 0:
            raise ValueError("lane_timeout_seconds must be positive")


class RetrievalService:
    """The sole Agent-facing entry point for tenant-scoped retrieval."""

    def __init__(self, dependencies: RetrievalDependencies) -> None:
        self._graph = build_retrieval_graph(dependencies)
        self._candidate_graph = build_retrieval_graph(dependencies, candidates_only=True)

    async def retrieve_candidates(
        self, request: RetrievalRequest, scope: UserScope, snapshot: RuntimeConfigSnapshot,
        *, business_context: QueryBusinessContext | None = None,
    ) -> RankedCandidates:
        """Stop at rerank; the external evidence stage owns parent hydration."""
        try:
            state = await self._candidate_graph.ainvoke(
                {"request": request, "scope": scope, "snapshot": snapshot, "business_context": business_context}
            )
        except NodeCancelledError as error:
            raise asyncio.CancelledError() from error
        return RankedCandidates(
            query=request.query, hits=tuple(state["reranked_hits"]),
            **_plan_output(state),
            user_id=scope.user_id, snapshot_id=snapshot.snapshot_id,
            index_generation=snapshot.index_generation, document_ids=request.document_ids,
            degraded_components=state["degraded_components"],
            observation=_observation(cast(RetrievalState, state)),
        )

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
        *, business_context: QueryBusinessContext | None = None,
    ) -> EvidenceBatch:
        """Run the immutable retrieval path and return safely scoped evidence."""
        try:
            result = await self._graph.ainvoke(
                {"request": request, "scope": scope, "snapshot": snapshot, "business_context": business_context}
            )
        except NodeCancelledError as error:
            # LangGraph turns a node's deliberate CancelledError into its own
            # node failure. The service boundary restores cancellation so
            # callers can stop work rather than mistake it for degradation.
            raise asyncio.CancelledError() from error
        return cast(EvidenceBatch, result["evidence_batch"])


def build_retrieval_graph(
    dependencies: RetrievalDependencies,
    *, candidates_only: bool = False,
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
        compiled = {}
        if dependencies.query_compiler is not None:
            plan = await dependencies.query_compiler.compile(state["request"], state["scope"], state["snapshot"], state.get("business_context"))
            if plan.authorization_filter != search_filter:
                raise ValueError("compiler authorization differs from retrieval authorization policy")
            search_filter = plan.effective_filter
            compiled = {"query_plan": plan, "query_plan_ref": dependencies.query_compiler.persist(plan), "query_plan_hash": plan.plan_hash}
        elif state.get("business_context") is not None:
            raise ValueError("business context requires an enabled QueryCompiler")
        return {
            **compiled,
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
        budget = _budget(state)
        fixed = state["snapshot"].evaluation is not None and state["snapshot"].evaluation.retrieval_budget is not None
        dense_limit = budget.dense_k if fixed else request.top_k_override or budget.dense_k
        bm25_limit = budget.bm25_k if fixed else request.top_k_override or budget.bm25_k

        plan = state.get("query_plan")

        async def dense() -> list[ChildHit]:
            if search_filter.match_none:
                return []
            vector = await dependencies.embedding.embed_query(plan.dense_query if plan else request.query)
            return await dependencies.vector.search(vector, search_filter, dense_limit)

        async def bm25() -> list[ChildHit]:
            if search_filter.match_none:
                return []
            return await dependencies.lexical.search(
                plan.lexical_core_query if plan else request.query, search_filter, bm25_limit,
                **({"scope_terms": plan.lexical_scope_terms} if plan and plan.lexical_scope_terms else {})
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
            [state.get("dense_hits", []), state.get("bm25_hits", [])],
            limit=_budget(state).rrf_k, k=_budget(state).rrf_constant
        )
        return _stage_update(
            state, "rrf_fusion", started, fused_hits=fused, rrf=len(fused)
        )

    async def cross_encoder(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        plan = state.get("query_plan")
        options = {}
        if plan and plan.protected_query_context:
            options["protected_query_context"] = plan.protected_query_context
        if plan and not plan.allow_query_windows:
            options["allow_query_windows"] = False
        result = await dependencies.reranker.rerank(
            plan.rerank_query if plan else state["request"].query, state["fused_hits"], limit=_budget(state).rerank_k,
            **options,
        )
        degraded = state["degraded_components"]
        if result.degraded:
            await emit_degradation(
                component="reranker",
                reason=result.degradation_reason or "reranker_unavailable",
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
            rerank_window_counts=result.window_counts,
            rerank_pairs=sum(result.window_counts),
            reranker=len(result.hits),
            degraded_components=degraded,
        )

    async def parent_aggregation(state: RetrievalState) -> dict[str, object]:
        started = perf_counter()
        selected = aggregate_parents(
            state["reranked_hits"], max_children_per_parent=2, limit=_budget(state).parent_k
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
        stage_hits: dict[Literal["dense", "bm25", "rrf", "rerank"], list[ChildHit]] = {
            "dense": state.get("dense_hits", []),
            "bm25": state.get("bm25_hits", []),
            "rrf": state.get("fused_hits", []),
            "rerank": state.get("reranked_hits", []),
        }
        from agentic_rag.models.contextual import SupportingContext
        supporting = SupportingContext()
        selected_hits = [h for p in state["hydrated_parents"] for h in p.child_hits]
        if dependencies.context_sources is not None:
            supporting = await dependencies.context_sources.hydrate(selected_hits, state["scope"], state["snapshot"], state.get("query_plan"))
        elif any(h.context_ref for h in selected_hits):
            raise ValueError("context source hydration required")
        batch = EvidenceBatch(
            query=state["request"].query,
            **_plan_output(state),
            parents=tuple(state["hydrated_parents"]),
            supporting_context=supporting,
            degraded_components=state["degraded_components"],
            document_ids=state["request"].document_ids,
            observation=RetrievalObservation(
                user_id=state["scope"].user_id,
                snapshot_id=state["snapshot"].snapshot_id,
                index_generation=state["snapshot"].index_generation,
                stages={name: tuple(RankedHitRef(
                    child_id=hit.child_id, parent_id=hit.parent_id, user_id=hit.user_id,
                    document_id=hit.document_id, document_version_id=hit.document_version_id,
                    retrieval_score=hit.retrieval_score, rrf_score=hit.rrf_score, rerank_score=hit.rerank_score,
                ) for hit in hits) for name, hits in stage_hits.items()},
                selected_parent_ids=tuple(p.parent_id for p in state["selected_parents"]),
                hydrated_parent_ids=tuple(p.parent_id for p in state["hydrated_parents"]),
                candidate_budget=_budget(state).model_dump(), timings_ms=state["timings_ms"],
                candidate_counts=state["candidate_counts"],
            ),
        )
        return _stage_update(state, "evidence_batch", started, evidence_batch=batch)

    builder = StateGraph(RetrievalState)
    builder.add_node("validate_and_filter", validate_and_filter)
    builder.add_node("recall", recall)
    builder.add_node("rrf_fusion", rrf_fusion)
    builder.add_node("cross_encoder", cross_encoder)
    if candidates_only:
        builder.add_edge(START, "validate_and_filter")
        builder.add_edge("validate_and_filter", "recall")
        builder.add_edge("recall", "rrf_fusion")
        builder.add_edge("rrf_fusion", "cross_encoder")
        builder.add_edge("cross_encoder", END)
        return builder.compile(name="RankedRetrievalGraph", checkpointer=False)
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


def _observation(state: RetrievalState) -> RetrievalObservation:
    stages: dict[Literal["dense", "bm25", "rrf", "rerank"], list[ChildHit]] = {
        "dense": state.get("dense_hits", []), "bm25": state.get("bm25_hits", []),
        "rrf": state.get("fused_hits", []), "rerank": state.get("reranked_hits", []),
    }
    return RetrievalObservation(
        user_id=state["scope"].user_id, snapshot_id=state["snapshot"].snapshot_id,
        index_generation=state["snapshot"].index_generation,
        stages={name: tuple(RankedHitRef(
            child_id=hit.child_id, parent_id=hit.parent_id, user_id=hit.user_id,
            document_id=hit.document_id, document_version_id=hit.document_version_id,
            retrieval_score=hit.retrieval_score, rrf_score=hit.rrf_score,
            rerank_score=hit.rerank_score,
        ) for hit in hits) for name, hits in stages.items()},
        selected_parent_ids=(), hydrated_parent_ids=(),
        candidate_budget=_budget(state).model_dump(),
        timings_ms=state["timings_ms"], candidate_counts=state["candidate_counts"],
    )


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
    for key in ("rrf", "reranker", "rerank_pairs", "parents_selected", "parents_hydrated"):
        value = update.pop(key, None)
        if value is not None:
            counts[key] = cast(int, value)
    update["timings_ms"] = timings
    update["candidate_counts"] = counts
    return update


def _plan_output(state: Mapping[str, object]) -> dict[str, object]:
    return {key: state[key] for key in ("query_plan", "query_plan_ref", "query_plan_hash") if key in state}
