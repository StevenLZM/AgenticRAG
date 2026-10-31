"""Multi-agent evidence preparation preserves provenance and bounded coverage."""

from importlib import import_module

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.orchestration.contracts import Criterion
from agentic_rag.retrieval.models import ChildHit, RankedCandidates
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test", graph_version="test", prompt_version="test",
    main_model_id="test", light_model_id="test", embedding_model="test",
    embedding_dimensions=1024, reranker_version="test", retrieval_config_version="test",
    index_generation="index-test", memory_config_version="test",
)
SCOPE = UserScope(user_id="user-1")


def ranked(parent="p1", *, user="user-1", score=1.0, content="Revenue is 120 million."):
    hit = ChildHit(
        child_id=f"c-{parent}", parent_id=parent, user_id=user,
        document_id=f"d-{parent}", document_version_id=f"v-{parent}",
        content=content, ast_locator="page/2/table/1", lane="dense", lane_rank=1,
        score=score, rerank_score=score,
    )
    return RankedCandidates(
        query="revenue", hits=(hit,), user_id=user, snapshot_id=SNAPSHOT.snapshot_id,
        index_generation=SNAPSHOT.index_generation,
    )


def candidate(key, criterion, batch):
    cls = import_module("agentic_rag.orchestration.contracts").CandidateResult
    return cls(origin_key=key, criterion_ids=(criterion,), candidates=(batch,))


class Hydrator:
    def __init__(self, text=None):
        self.selected = []
        self.text = text

    async def hydrate(self, selected, scope):
        self.selected.extend(selected)
        return [p.model_copy(update={"content": self.text or p.child_hits[0].content}) for p in selected]


async def test_shared_hit_is_one_source_with_both_task_and_criterion_associations():
    hydrator = Hydrator()
    prepare = import_module("agentic_rag.orchestration.evidence").EvidencePreparer(hydrator)
    result = await prepare.prepare(
        [candidate("r1", "revenue", ranked()), candidate("r2", "comparison", ranked())],
        [Criterion(id="revenue", description="Revenue"), Criterion(id="comparison", description="Comparison value")],
        SCOPE, SNAPSHOT,
    )
    assert len(hydrator.selected) == 1
    assert len(result.packed.items) == 1
    item = result.packed.items[0]
    assert set(item.covered_target_ids) == {"revenue", "comparison"}
    assert result.source_map[item.evidence_id] == ("r1", "r2")


async def test_foreign_candidates_are_rejected_before_hydration():
    hydrator = Hydrator()
    prepare = import_module("agentic_rag.orchestration.evidence").EvidencePreparer(hydrator)
    with pytest.raises(ValueError, match="scope|tenant"):
        await prepare.prepare(
            [candidate("r1", "revenue", ranked(user="other"))],
            [Criterion(id="revenue", description="Revenue")], SCOPE, SNAPSHOT,
        )
    assert hydrator.selected == []


async def test_large_high_scoring_query_does_not_consume_another_targets_budget():
    hydrator = Hydrator(text="Revenue context. " * 1000)
    prepare = import_module("agentic_rag.orchestration.evidence").EvidencePreparer(hydrator)
    bounded = SNAPSHOT.model_copy(update={"max_evidence_tokens": 1200})
    result = await prepare.prepare(
        [candidate("r1", "one", ranked("p1", score=9999).model_copy(update={"snapshot_id": bounded.snapshot_id})),
         candidate("r2", "two", ranked("p2", score=0.001).model_copy(update={"snapshot_id": bounded.snapshot_id}))],
        [Criterion(id="one", description="Company one"), Criterion(id="two", description="Company two")],
        SCOPE, bounded,
    )
    assert {item.parent_id for item in result.packed.items} == {"p1", "p2"}
    assert result.packed.token_count <= 1200


async def test_wrong_generation_never_becomes_a_source():
    prepare = import_module("agentic_rag.orchestration.evidence").EvidencePreparer(Hydrator())
    wrong = ranked().model_copy(update={"index_generation": "index-old"})
    with pytest.raises(ValueError, match="generation|snapshot"):
        await prepare.prepare(
            [candidate("r1", "revenue", wrong)],
            [Criterion(id="revenue", description="Revenue")], SCOPE, SNAPSHOT,
        )
