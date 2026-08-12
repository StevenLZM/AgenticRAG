"""Unit coverage for deterministic parent aggregation and scoped hydration."""

from __future__ import annotations

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.repositories import ParentChunk
from agentic_rag.retrieval.models import ChildHit
from agentic_rag.retrieval.parents import (
    ParentFetcher,
    ParentScopeViolation,
    aggregate_parents,
)


def hit(
    child_id: str,
    *,
    parent_id: str,
    score: float,
    lane_rank: int = 1,
) -> ChildHit:
    """Build a valid hit with an explicit rerank score."""
    return ChildHit(
        child_id=child_id,
        parent_id=parent_id,
        user_id="user-1",
        document_id=f"document-{parent_id}",
        document_version_id=f"version-{parent_id}",
        content=f"content for {child_id}",
        ast_locator=f"locator-{child_id}",
        lane="dense",
        lane_rank=lane_rank,
        score=score,
    )


RERANKED_HITS = [
    hit("a-1", parent_id="parent-a", score=0.98),
    hit("b-1", parent_id="parent-b", score=0.95),
    hit("a-2", parent_id="parent-a", score=0.91),
    hit("a-3", parent_id="parent-a", score=0.89),
    hit("c-1", parent_id="parent-c", score=0.87),
]


def test_parent_aggregation_caps_children_and_keeps_best_parent_order() -> None:
    parents = aggregate_parents(RERANKED_HITS, max_children_per_parent=2, limit=6)

    assert [parent.parent_id for parent in parents] == [
        "parent-a",
        "parent-b",
        "parent-c",
    ]
    assert [hit.child_id for hit in parents[0].child_hits] == ["a-1", "a-2"]
    assert all(len(parent.child_hits) <= 2 for parent in parents)
    assert len(parents) <= 6


def test_parent_aggregation_breaks_equal_scores_by_first_reranked_child_then_id() -> None:
    parents = aggregate_parents(
        [
            hit("z-child", parent_id="parent-z", score=0.8),
            hit("a-child", parent_id="parent-a", score=0.8),
            hit("a-second", parent_id="parent-a", score=0.7),
        ]
    )

    assert [parent.parent_id for parent in parents] == ["parent-z", "parent-a"]


def test_parent_aggregation_honors_parent_limit() -> None:
    parents = aggregate_parents(RERANKED_HITS, limit=2)

    assert [parent.parent_id for parent in parents] == ["parent-a", "parent-b"]


class RecordingParents:
    def __init__(self, rows: list[ParentChunk]) -> None:
        self.rows = rows
        self.calls: list[tuple[list[str], UserScope]] = []

    async def get_many(
        self, parent_ids: list[str], scope: UserScope
    ) -> list[ParentChunk]:
        self.calls.append((parent_ids, scope))
        return self.rows


def parent(parent_id: str, *, user_id: str = "user-1") -> ParentChunk:
    return ParentChunk(
        id=parent_id,
        user_id=user_id,
        document_id=f"document-{parent_id}",
        document_version_id=f"version-{parent_id}",
        ordinal=0,
        content=f"full content for {parent_id}",
        status="active",
    )


async def test_parent_fetch_batches_scoped_lookup_and_preserves_requested_order() -> None:
    repository = RecordingParents([parent("parent-b"), parent("parent-a")])
    fetcher = ParentFetcher(repository)

    evidence = await fetcher.fetch(
        ["parent-a", "parent-b", "parent-a"], UserScope(user_id="user-1")
    )

    assert repository.calls == [
        (["parent-a", "parent-b", "parent-a"], UserScope(user_id="user-1"))
    ]
    assert [item.parent_id for item in evidence] == [
        "parent-a",
        "parent-b",
        "parent-a",
    ]
    assert [item.content for item in evidence] == [
        "full content for parent-a",
        "full content for parent-b",
        "full content for parent-a",
    ]
    assert all(item.child_hits == () for item in evidence)


async def test_parent_fetch_fails_closed_when_any_requested_id_is_missing() -> None:
    fetcher = ParentFetcher(RecordingParents([parent("parent-a")]))

    with pytest.raises(ParentScopeViolation, match="missing, inactive, or outside scope"):
        await fetcher.fetch(["parent-a", "parent-b"], UserScope(user_id="user-1"))


async def test_parent_fetch_fails_closed_when_repository_returns_wrong_owner() -> None:
    fetcher = ParentFetcher(RecordingParents([parent("parent-a", user_id="user-2")]))

    with pytest.raises(ParentScopeViolation, match="missing, inactive, or outside scope"):
        await fetcher.fetch(["parent-a"], UserScope(user_id="user-1"))


async def test_parent_fetch_fails_closed_when_repository_returns_duplicate_ids() -> None:
    fetcher = ParentFetcher(RecordingParents([parent("parent-a"), parent("parent-a")]))

    with pytest.raises(ParentScopeViolation, match="missing, inactive, or outside scope"):
        await fetcher.fetch(["parent-a"], UserScope(user_id="user-1"))
