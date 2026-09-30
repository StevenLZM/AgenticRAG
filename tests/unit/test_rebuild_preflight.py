"""Safety gates for the non-destructive v2/v3 rebuild preflight."""

from __future__ import annotations

import pytest


def test_preflight_resolves_only_the_explicit_v3_physical_index() -> None:
    from scripts.rebuild_preflight import run_preflight

    report = run_preflight(
        index_generation="index-v3",
        physical_index="agenticrag-children-index-v3",
        backup_verified=True,
        workers_quiescent=True,
        sources_staged=True,
        reorder_events=(
            {
                "source_id": "doc-1",
                "reordered": True,
                "triggered": True,
                "reason": "consistent_same_row_date_band",
            },
            {"source_id": "doc-2", "reordered": False, "triggered": False},
        ),
        parent_count=2,
        parent_token_distribution=(120, 180),
        source_span_coverage={"before": 4, "after": 4},
    )

    assert report["ready"] is True
    assert report["pipeline_version"] == "ingestion-v2"
    assert report["index_generation"] == "index-v3"
    assert report["physical_index"] == "agenticrag-children-index-v3"
    assert report["alias"] == "agenticrag-children-active"
    assert report["parent_count"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"index_generation": "agenticrag-children-*"}, "wildcard"),
        ({"index_generation": "index-v2"}, "index-v3"),
    ],
)
def test_preflight_rejects_ambiguous_or_wrong_generation_targets(
    kwargs: dict[str, object], message: str
) -> None:
    from scripts.rebuild_preflight import RebuildPreflightError, run_preflight

    with pytest.raises(RebuildPreflightError, match=message):
        run_preflight(
            physical_index="agenticrag-children-index-v3",
            backup_verified=True,
            workers_quiescent=True,
            sources_staged=True,
            **kwargs,
        )


def test_preflight_rejects_reorder_without_a_design_trigger_before_any_delete() -> None:
    from scripts.rebuild_preflight import RebuildPreflightError, run_preflight

    with pytest.raises(RebuildPreflightError, match="non-triggering"):
        run_preflight(
            index_generation="index-v3",
            physical_index="agenticrag-children-index-v3",
            backup_verified=True,
            workers_quiescent=True,
            sources_staged=True,
            reorder_events=(
                {"source_id": "report", "reordered": True, "triggered": False},
            ),
        )


def test_preflight_requires_backup_quiescence_and_staging() -> None:
    from scripts.rebuild_preflight import RebuildPreflightError, run_preflight

    with pytest.raises(RebuildPreflightError, match="backup"):
        run_preflight(
            index_generation="index-v3",
            physical_index="agenticrag-children-index-v3",
            backup_verified=False,
            workers_quiescent=True,
            sources_staged=True,
        )
