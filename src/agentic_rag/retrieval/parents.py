"""Deterministic parent selection and fail-closed parent hydration."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.domain.models import UserScope
from agentic_rag.persistence.repositories import ParentChunk, ParentRepository
from agentic_rag.retrieval.models import ChildHit, ParentEvidence


class ParentScopeViolation(RuntimeError):
    """Raised when a parent cannot be safely hydrated within the user scope."""


def aggregate_parents(
    hits: Sequence[ChildHit], *, max_children_per_parent: int = 2, limit: int = 6
) -> list[ParentEvidence]:
    """Group reranked children into a deterministic, bounded parent list.

    The input order is the reranker order.  Children are therefore retained in
    that order, while parent ranking uses each parent's highest rerank score and
    the first matching child as its stable tie breaker.
    """
    if max_children_per_parent < 1:
        raise ValueError("max_children_per_parent must be positive")
    if limit < 1:
        raise ValueError("limit must be positive")

    selections: dict[str, _ParentSelection] = {}
    for position, hit in enumerate(hits):
        selection = selections.get(hit.parent_id)
        if selection is None:
            selection = _ParentSelection(best_hit=hit, first_position=position)
            selections[hit.parent_id] = selection
        elif hit.score > selection.best_hit.score:
            selection.best_hit = hit

        if len(selection.child_hits) < max_children_per_parent:
            selection.child_hits.append(hit)

    ordered = sorted(
        selections.values(),
        key=lambda selection: (
            -selection.best_hit.score,
            selection.first_position,
            selection.best_hit.parent_id,
        ),
    )
    return [selection.to_evidence() for selection in ordered[:limit]]


class _ParentSelection:
    """Mutable aggregation state kept private to the pure public API."""

    def __init__(self, *, best_hit: ChildHit, first_position: int) -> None:
        self.best_hit = best_hit
        self.first_position = first_position
        self.child_hits: list[ChildHit] = []

    def to_evidence(self) -> ParentEvidence:
        return ParentEvidence(
            parent_id=self.best_hit.parent_id,
            document_id=self.best_hit.document_id,
            document_version_id=self.best_hit.document_version_id,
            content="",
            child_hits=tuple(self.child_hits),
            rerank_score=self.best_hit.score,
        )


class ParentFetcher:
    """Hydrate selected parents through the scoped persistence boundary."""

    def __init__(self, parents: ParentRepository) -> None:
        self._parents = parents

    async def fetch(
        self, parent_ids: Sequence[str], scope: UserScope
    ) -> list[ParentEvidence]:
        """Fetch every requested active parent or fail the complete batch closed."""
        requested_ids = list(parent_ids)
        if not requested_ids:
            return []

        rows = await self._parents.get_many(requested_ids, scope)
        expected_ids = set(requested_ids)
        by_id = {row.id: row for row in rows}
        if (
            set(by_id) != expected_ids
            or len(by_id) != len(rows)
            or any(
                row.user_id != scope.user_id or row.status != "active" for row in rows
            )
        ):
            raise ParentScopeViolation(
                "one or more parents are missing, inactive, or outside scope"
            )

        return [_to_parent_evidence(by_id[parent_id]) for parent_id in requested_ids]


def _to_parent_evidence(parent: ParentChunk) -> ParentEvidence:
    """Translate the repository row without coupling retrieval to SQLAlchemy."""
    return ParentEvidence(
        parent_id=parent.id,
        document_id=parent.document_id,
        document_version_id=parent.document_version_id,
        content=parent.content,
        child_hits=(),
        rerank_score=0.0,
    )
