"""Component integration coverage for the fixed retrieval service boundary."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.retrieval.graph import RetrievalDependencies, RetrievalService
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


class Embedding:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1024


class Index:
    def __init__(self, lane: str) -> None:
        self.lane = lane

    async def search(self, *args: object) -> list[ChildHit]:
        return [
            ChildHit(
                child_id=f"child-{self.lane}",
                parent_id=f"parent-{self.lane}",
                user_id="user-1",
                document_id="document-1",
                document_version_id="version-1",
                content="selection child",
                ast_locator="heading:1",
                lane=self.lane,  # type: ignore[arg-type]
                lane_rank=1,
                score=1.0,
            )
        ]


class Reranker:
    async def rerank(
        self, query: str, candidates: Sequence[ChildHit], *, limit: int
    ) -> RerankResult:
        return RerankResult(list(candidates[:limit]), False, "test")


class Parents:
    async def hydrate(
        self, selected: Sequence[ParentEvidence], scope: UserScope
    ) -> list[ParentEvidence]:
        return [item.model_copy(update={"content": "hydrated parent"}) for item in selected]


@pytest.mark.integration
async def test_pipeline_hydrates_selected_parent_evidence() -> None:
    service = RetrievalService(
        RetrievalDependencies(
            embedding=Embedding(),
            vector=Index("dense"),
            lexical=Index("bm25"),
            reranker=Reranker(),
            parent_fetcher=Parents(),
        )
    )

    result = await service.retrieve(
        RetrievalRequest(query="policy"), UserScope(user_id="user-1"), SNAPSHOT
    )

    assert len(result.parents) == 2
    assert {item.content for item in result.parents} == {"hydrated parent"}
    assert {item.child_hits[0].child_id for item in result.parents} == {
        "child-dense",
        "child-bm25",
    }
