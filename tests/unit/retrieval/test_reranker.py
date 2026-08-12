"""Unit coverage for bounded, degradable cross-encoder reranking."""

import time

import pytest

from agentic_rag.retrieval.models import ChildHit
from agentic_rag.retrieval.reranker import Reranker


def hit(child_id: str) -> ChildHit:
    """Build a small but valid retrieval hit."""
    return ChildHit(
        child_id=child_id,
        parent_id=f"parent-{child_id}",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        content=f"content for {child_id}",
        ast_locator="segment-0",
        lane="dense",
        lane_rank=1,
        score=1.0,
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.raise_timeout = False
        self.raise_unavailable = False
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(pairs)
        if self.raise_timeout:
            time.sleep(0.05)
        if self.raise_unavailable:
            raise RuntimeError("model unavailable")
        return self.scores[: len(pairs)]


@pytest.fixture
def candidates() -> list[ChildHit]:
    return [hit("a"), hit("b"), hit("c")]


@pytest.fixture
def service() -> Reranker:
    return Reranker(
        FakeCrossEncoder([0.2, 0.9, 0.5]),
        model_version="fake-cross-encoder-v1",
        timeout_seconds=0.01,
        max_concurrent_reranks=1,
    )


async def test_reranker_orders_hits_by_cross_encoder_score(
    service: Reranker, candidates: list[ChildHit]
) -> None:
    result = await service.rerank("query", candidates, limit=2)

    assert [item.child_id for item in result.hits] == ["b", "c"]
    assert result.degraded is False
    assert result.model_version == "fake-cross-encoder-v1"
    assert service.model.calls == [[("query", item.content) for item in candidates]]


async def test_reranker_timeout_returns_rrf_order(
    service: Reranker, candidates: list[ChildHit]
) -> None:
    service.model.raise_timeout = True

    result = await service.rerank("q", candidates, limit=2)

    assert result.degraded is True
    assert result.hits == candidates[:2]


async def test_reranker_unavailable_returns_rrf_order(
    service: Reranker, candidates: list[ChildHit]
) -> None:
    service.model.raise_unavailable = True

    result = await service.rerank("q", candidates, limit=2)

    assert result.degraded is True
    assert result.hits == candidates[:2]


async def test_reranker_limits_model_input_to_thirty_candidates() -> None:
    candidates = [hit(str(index)) for index in range(31)]
    model = FakeCrossEncoder([float(index) for index in range(30)])
    service = Reranker(model, model_version="fake-cross-encoder-v1")

    result = await service.rerank("q", candidates, limit=30)

    assert len(model.calls[0]) == 30
    assert len(result.hits) == 30
