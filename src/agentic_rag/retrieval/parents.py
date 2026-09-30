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

    Children retain input order. Parent ranking uses each parent's latest
    available stage score (rerank, then RRF, then raw retrieval) and the best
    matching child's position as its stable tie breaker.
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
        elif hit.ranking_score > selection.best_hit.ranking_score:
            selection.best_hit = hit
            selection.best_position = position

        if len(selection.child_hits) < max_children_per_parent:
            selection.child_hits.append(hit)

    ordered = sorted(
        selections.values(),
        key=lambda selection: (
            -selection.best_hit.ranking_score,
            selection.best_position,
            selection.best_hit.parent_id,
        ),
    )
    return [selection.to_evidence() for selection in ordered[:limit]]


class _ParentSelection:
    """Mutable aggregation state kept private to the pure public API."""

    def __init__(self, *, best_hit: ChildHit, first_position: int) -> None:
        self.best_hit = best_hit
        self.best_position = first_position
        self.child_hits: list[ChildHit] = []

    def to_evidence(self) -> ParentEvidence:
        return ParentEvidence(
            parent_id=self.best_hit.parent_id,
            document_id=self.best_hit.document_id,
            document_version_id=self.best_hit.document_version_id,
            content="",
            child_hits=tuple(self.child_hits),
            retrieval_score=self.best_hit.retrieval_score,
            rrf_score=self.best_hit.rrf_score,
            rerank_score=self.best_hit.rerank_score,
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

        by_id = await self._fetch_by_id(requested_ids, scope)

        return [_to_parent_evidence(by_id[parent_id]) for parent_id in requested_ids]

    async def hydrate(
        self, selected: Sequence[ParentEvidence], scope: UserScope
    ) -> list[ParentEvidence]:
        """Fill selected parent content while retaining selection provenance."""
        if not selected:
            return []

        by_id = await self._fetch_by_id(
            [evidence.parent_id for evidence in selected], scope
        )
        hydrated: list[ParentEvidence] = []
        for evidence in selected:
            parent = by_id[evidence.parent_id]
            if (
                parent.document_id != evidence.document_id
                or parent.document_version_id != evidence.document_version_id
            ):
                raise ParentScopeViolation(
                    "fetched parent provenance does not match the selected evidence"
                )
            hydrated.append(
                ParentEvidence(
                    parent_id=evidence.parent_id,
                    document_id=evidence.document_id,
                    document_version_id=evidence.document_version_id,
                    content=parent.content,
                    child_hits=evidence.child_hits,
                    retrieval_score=evidence.retrieval_score,
                    rrf_score=evidence.rrf_score,
                    rerank_score=evidence.rerank_score,
                    heading_path=parent.heading_path,
                )
            )
        return hydrated

    async def _fetch_by_id(
        self, parent_ids: Sequence[str], scope: UserScope
    ) -> dict[str, ParentChunk]:
        unique_ids = list(dict.fromkeys(parent_ids))
        rows = await self._parents.get_many(unique_ids, scope)
        by_id = {row.id: row for row in rows}
        if (
            set(by_id) != set(unique_ids)
            or len(by_id) != len(rows)
            or any(
                row.user_id != scope.user_id or row.status != "active" for row in rows
            )
        ):
            raise ParentScopeViolation(
                "one or more parents are missing, inactive, or outside scope"
            )
        return by_id


def _to_parent_evidence(parent: ParentChunk) -> ParentEvidence:
    """Translate the repository row without coupling retrieval to SQLAlchemy."""
    return ParentEvidence(
        parent_id=parent.id,
        document_id=parent.document_id,
        document_version_id=parent.document_version_id,
        content=parent.content,
        child_hits=(),
        retrieval_score=0.0,
        heading_path=parent.heading_path,
    )
