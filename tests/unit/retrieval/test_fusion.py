"""Unit coverage for deterministic reciprocal-rank fusion."""

from agentic_rag.retrieval.fusion import reciprocal_rank_score, rrf_fuse
from agentic_rag.retrieval.models import ChildHit


def hit(
    child_id: str,
    *,
    lane: str = "dense",
    lane_rank: int = 1,
    score: float = 1.0,
) -> ChildHit:
    """Build a small but valid retrieval hit."""
    return ChildHit(
        child_id=child_id,
        parent_id=f"parent-{child_id}",
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        content=f"content for {child_id}",
        ast_locator="segment-0",
        lane=lane,  # type: ignore[arg-type]
        lane_rank=lane_rank,
        score=score,
    )


def test_reciprocal_rank_score_uses_one_based_rank() -> None:
    assert reciprocal_rank_score(1, k=60) == 1 / 61


def test_rrf_deduplicates_and_uses_stable_tie_break() -> None:
    fused = rrf_fuse(
        [[hit("a", lane_rank=1), hit("b", lane_rank=2)], [hit("b", lane_rank=1), hit("c", lane_rank=2)]],
        k=60,
        limit=30,
    )

    assert [item.child_id for item in fused] == ["b", "a", "c"]


def test_rrf_breaks_equal_scores_by_best_lane_rank_then_child_id() -> None:
    fused = rrf_fuse(
        [[hit("z", lane_rank=1), hit("a", lane_rank=2), hit("b", lane_rank=2)]],
        k=60,
        limit=30,
    )

    assert [item.child_id for item in fused] == ["z", "a", "b"]


def test_rrf_honors_the_result_limit() -> None:
    fused = rrf_fuse([[hit("a"), hit("b", lane_rank=2)]], limit=1)

    assert [item.child_id for item in fused] == ["a"]


def test_rrf_propagates_raw_retrieval_and_fused_scores() -> None:
    fused = rrf_fuse(
        [
            [hit("shared", lane="dense", lane_rank=1, score=0.91)],
            [hit("shared", lane="bm25", lane_rank=1, score=12.0)],
        ],
        k=60,
    )

    assert fused[0].retrieval_score == 0.91
    assert fused[0].rrf_score == 2 / 61
    assert fused[0].rerank_score is None


def test_rrf_rejects_a_non_positive_child_lane_rank() -> None:
    invalid_hit = hit("a", lane_rank=0)

    try:
        rrf_fuse([[invalid_hit]])
    except ValueError as error:
        assert str(error) == "ChildHit lane_rank must be positive"
    else:
        raise AssertionError("rrf_fuse accepted an invalid ChildHit lane rank")
