"""Behavioral coverage for the fixed, degradable retrieval subgraph."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import AgentEventEmitter, event_emission_scope
from agentic_rag.persistence.artifacts import LocalArtifactStore
from agentic_rag.persistence.repositories import AgentEvent
from agentic_rag.retrieval.graph import (
    RetrievalDependencies,
    RetrievalService,
    RetrievalUnavailable,
    build_retrieval_graph,
)
from agentic_rag.retrieval.models import ChildHit, ParentEvidence, RetrievalRequest
from agentic_rag.retrieval.reranker import RerankResult
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="test",
    prompt_version="test",
    main_model_id="test",
    light_model_id="test",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="test",
    retrieval_config_version="test",
    index_generation="index-test",
    memory_config_version="test",
)
REQUEST = RetrievalRequest(query="termination clause")
SCOPE = UserScope(user_id="user-1")


def hit(child_id: str, *, lane: str = "dense", score: float = 1.0) -> ChildHit:
    return ChildHit(
        child_id=child_id,
        parent_id=f"parent-{child_id}",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        content=f"child content {child_id}",
        ast_locator=f"section-{child_id}",
        lane=lane,  # type: ignore[arg-type]
        lane_rank=1,
        score=score,
    )


class FakeEmbedding:
    async def embed_query(self, text: str) -> list[float]:
        assert text == REQUEST.query
        return [0.0] * 1024


class FakeVector:
    def __init__(self, hits: Sequence[ChildHit]) -> None:
        self.hits = list(hits)
        self.error: BaseException | None = None
        self.calls: list[tuple[list[float], object, int]] = []

    async def search(
        self, vector: Sequence[float], filter: object, top_k: int
    ) -> list[ChildHit]:
        self.calls.append((list(vector), filter, top_k))
        if self.error is not None:
            raise self.error
        return self.hits


class FakeLexical:
    def __init__(self, hits: Sequence[ChildHit]) -> None:
        self.hits = list(hits)
        self.error: BaseException | None = None
        self.calls: list[tuple[str, object, int]] = []

    async def search(self, query: str, filter: object, top_k: int) -> list[ChildHit]:
        self.calls.append((query, filter, top_k))
        if self.error is not None:
            raise self.error
        return self.hits


class FakeReranker:
    def __init__(self, *, degraded: bool = False) -> None:
        self.degraded = degraded
        self.calls: list[tuple[str, list[ChildHit], int]] = []

    async def rerank(
        self, query: str, candidates: Sequence[ChildHit], *, limit: int
    ) -> RerankResult:
        self.calls.append((query, list(candidates), limit))
        return RerankResult(
            hits=list(candidates[:limit]),
            degraded=self.degraded,
            model_version="fake",
        )


class FakeParentFetcher:
    def __init__(self) -> None:
        self.calls: list[tuple[list[ParentEvidence], UserScope]] = []

    async def hydrate(
        self, selected: Sequence[ParentEvidence], scope: UserScope
    ) -> list[ParentEvidence]:
        self.calls.append((list(selected), scope))
        return [
            evidence.model_copy(update={"content": f"parent {evidence.parent_id}"})
            for evidence in selected
        ]


class RecordingEvents:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def append(self, event: AgentEvent) -> int:
        self.events.append(replace(event, id=len(self.events) + 1))
        return len(self.events)

    async def list_after(self, *args: object, **kwargs: object) -> list[AgentEvent]:
        del args, kwargs
        return []


@pytest.fixture
def deps() -> RetrievalDependencies:
    return RetrievalDependencies(
        embedding=FakeEmbedding(),
        vector=FakeVector([hit("dense")]),
        lexical=FakeLexical([hit("bm25", lane="bm25")]),
        reranker=FakeReranker(),
        parent_fetcher=FakeParentFetcher(),
    )


def state() -> dict[str, object]:
    return {"request": REQUEST, "scope": SCOPE, "snapshot": SNAPSHOT}


async def test_batch_keeps_separate_observed_rankings(deps: RetrievalDependencies) -> None:
    class ReverseReranker:
        async def rerank(self, query, candidates, *, limit):
            return RerankResult(hits=list(reversed(candidates))[:limit], degraded=False, model_version="contract")

    batch = await RetrievalService(replace(deps, reranker=ReverseReranker())).retrieve(REQUEST, SCOPE, SNAPSHOT)
    trace = batch.model_dump(mode="json").get("observation")
    assert trace is not None
    assert trace["user_id"] == "user-1"
    assert trace["snapshot_id"] == SNAPSHOT.snapshot_id
    assert trace["stages"]["dense"][0]["child_id"] == "dense"
    assert trace["stages"]["bm25"][0]["child_id"] == "bm25"
    assert trace["stages"]["rerank"] == list(reversed(trace["stages"]["rrf"]))
    assert trace["hydrated_parent_ids"] == [p.parent_id for p in batch.parents]


async def test_service_runs_fixed_pipeline_and_retains_parent_provenance(
    deps: RetrievalDependencies,
) -> None:
    result = await RetrievalService(deps).retrieve(REQUEST, SCOPE, SNAPSHOT)

    assert result.query == REQUEST.query
    assert [parent.parent_id for parent in result.parents] == [
        "parent-bm25",
        "parent-dense",
    ]
    assert all(parent.content.startswith("parent ") for parent in result.parents)
    assert all(parent.child_hits for parent in result.parents)
    assert result.degraded_components == ()
    assert deps.vector.calls[0][2] == 40
    assert deps.lexical.calls[0][2] == 40


async def test_service_propagates_direct_document_selector_to_evidence_batch(
    deps: RetrievalDependencies,
) -> None:
    request = RetrievalRequest(query="termination clause", document_ids=("document-1",))

    result = await RetrievalService(deps).retrieve(request, SCOPE, SNAPSHOT)

    assert result.document_ids == ("document-1",)


async def test_graph_degrades_to_bm25_when_dense_fails(
    deps: RetrievalDependencies,
) -> None:
    deps.vector.error = OSError("dense unavailable")

    result = await build_retrieval_graph(deps).ainvoke(state())

    batch = result["evidence_batch"]
    assert batch.degraded_components == ("dense",)
    assert [parent.parent_id for parent in batch.parents] == ["parent-bm25"]
    assert result["lane_failures"]["dense"].error_type == "OSError"


async def test_single_lane_degradation_emits_bounded_durable_signal(
    deps: RetrievalDependencies,
    tmp_path,
) -> None:
    events = RecordingEvents()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    emitter = AgentEventEmitter(
        events,
        artifacts,
        runtime_config_snapshot_id=SNAPSHOT.snapshot_id,
    )
    deps.vector.error = OSError("private provider response must not be persisted")

    async with event_emission_scope(emitter, "run-1", "retrieval", user_id=SCOPE.user_id):
        result = await RetrievalService(deps).retrieve(REQUEST, SCOPE, SNAPSHOT)

    assert result.degraded_components == ("dense",)
    assert events.events[-1].event_type == "RETRIEVAL_DEGRADED"
    assert events.events[-1].payload_ref is not None
    payload = artifacts.read_json(artifacts.describe(events.events[-1].payload_ref))
    assert "private provider response" not in str(payload)


async def test_graph_fails_closed_when_both_lanes_fail(
    deps: RetrievalDependencies,
) -> None:
    deps.vector.error = OSError("dense unavailable")
    deps.lexical.error = TimeoutError("bm25 unavailable")

    with pytest.raises(RetrievalUnavailable) as raised:
        await build_retrieval_graph(deps).ainvoke(state())

    assert set(raised.value.failures) == {"dense", "bm25"}


async def test_graph_propagates_lane_cancellation_without_degrading(
    deps: RetrievalDependencies,
) -> None:
    deps.vector.error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await RetrievalService(deps).retrieve(REQUEST, SCOPE, SNAPSHOT)


async def test_graph_propagates_invalid_selector_error_without_degradation(
    deps: RetrievalDependencies,
) -> None:
    deps.vector.error = ValueError("date_range is unsupported")

    with pytest.raises(ValueError, match="date_range is unsupported"):
        await RetrievalService(deps).retrieve(REQUEST, SCOPE, SNAPSHOT)


async def test_graph_records_reranker_degradation(
    deps: RetrievalDependencies,
) -> None:
    deps.reranker.degraded = True  # type: ignore[union-attr]

    result = await RetrievalService(deps).retrieve(REQUEST, SCOPE, SNAPSHOT)

    assert result.degraded_components == ("reranker",)


async def test_top_k_override_is_applied_only_to_recall_lanes(
    deps: RetrievalDependencies,
) -> None:
    request = RetrievalRequest(query="termination clause", top_k_override=7)

    await RetrievalService(deps).retrieve(request, SCOPE, SNAPSHOT)

    assert deps.vector.calls[0][2] == 7
    assert deps.lexical.calls[0][2] == 7
    assert deps.reranker.calls[0][2] == 10  # type: ignore[union-attr]
