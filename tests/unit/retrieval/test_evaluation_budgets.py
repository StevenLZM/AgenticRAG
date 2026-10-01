import pytest

from agentic_rag.retrieval.graph import RetrievalService
from agentic_rag.retrieval.reranker import Reranker
from agentic_rag.runtime.models import EvaluationMetadata, RetrievalBudget
from tests.unit.retrieval.test_graph import FakeEmbedding, FakeLexical, FakeParentFetcher, FakeReranker, FakeVector, REQUEST, SCOPE, SNAPSHOT, RetrievalDependencies, hit


@pytest.mark.asyncio
async def test_budgets_reach_real_stage_and_observation_not_hidden_30():
    deps = RetrievalDependencies(embedding=FakeEmbedding(), vector=FakeVector([hit(str(i)) for i in range(80)]),
        lexical=FakeLexical([hit(str(i), lane="bm25") for i in range(80)]),
        reranker=FakeReranker(), parent_fetcher=FakeParentFetcher())
    budget = RetrievalBudget(dense_k=80, bm25_k=80, rrf_k=50, rerank_k=20, parent_k=10)
    evaluation = EvaluationMetadata(session_id="e", dataset_sha256="a" * 64, corpus_snapshot_id="b" * 64, retrieval_budget=budget)
    batch = await RetrievalService(deps).retrieve(REQUEST, SCOPE, SNAPSHOT.model_copy(update={"evaluation": evaluation}))
    assert deps.vector.calls[0][2] == 80
    assert deps.lexical.calls[0][2] == 80
    assert len(deps.reranker.calls[0][1]) == 50
    assert deps.reranker.calls[0][2] == 20
    assert len(batch.parents) == 10
    assert batch.observation.candidate_budget["rrf_k"] == 50
    assert batch.observation.timings_ms["cross_encoder"] >= 0


@pytest.mark.asyncio
async def test_cross_encoder_configured_cap_accepts_50_inputs():
    class Model:
        def predict(self, pairs):
            assert len(pairs) == 50
            return list(range(50))
    scorer = Reranker(Model(), model_version="test", max_candidates=200)
    try:
        result = await scorer.rerank("q", [hit(str(i)) for i in range(50)], limit=20)
        assert len(result.hits) == 20
        assert result.hits[0].child_id == "49"
    finally:
        await scorer.aclose()
