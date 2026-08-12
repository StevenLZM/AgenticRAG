"""Deterministic reciprocal-rank fusion for retrieval lanes."""

from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.retrieval.models import ChildHit


def reciprocal_rank_score(rank: int, k: int = 60) -> float:
    """Return the reciprocal-rank contribution for a one-based lane rank."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    if k < 0:
        raise ValueError("k must be non-negative")
    return 1.0 / (k + rank)


def rrf_fuse(
    lanes: Sequence[Sequence[ChildHit]], *, k: int = 60, limit: int = 30
) -> list[ChildHit]:
    """Fuse retrieval lanes, retaining one deterministic hit per Child ID.

    Each lane contributes a reciprocal-rank score.  When the same Child is
    returned by multiple lanes, its contributions are summed and the first hit
    instance is retained because it already carries the retrieval metadata
    needed by later pipeline stages.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if k < 0:
        raise ValueError("k must be non-negative")

    scores: dict[str, float] = {}
    best_ranks: dict[str, int] = {}
    hits: dict[str, ChildHit] = {}
    for lane in lanes:
        for hit in lane:
            rank = hit.lane_rank
            if rank <= 0:
                raise ValueError("ChildHit lane_rank must be positive")
            scores[hit.child_id] = scores.get(hit.child_id, 0.0) + reciprocal_rank_score(
                rank, k
            )
            best_ranks[hit.child_id] = min(best_ranks.get(hit.child_id, rank), rank)
            hits.setdefault(hit.child_id, hit)

    child_ids = sorted(
        hits,
        key=lambda child_id: (-scores[child_id], best_ranks[child_id], child_id),
    )
    return [hits[child_id] for child_id in child_ids[:limit]]
