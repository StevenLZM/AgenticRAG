"""Security boundary tests for the only answer shape exposed by the API."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_public_answer_schema_rejects_unknown_and_raw_provider_fields() -> None:
    from agentic_rag.query.public_answer import PublicAnswer

    with pytest.raises(ValidationError):
        PublicAnswer.model_validate(
            {
                "audited": True,
                "segments": [
                    {
                        "kind": "content",
                        "text": "safe answer",
                        "evidence_ids": ["e1"],
                        "tool_input": "Authorization: Bearer segment-secret",
                    }
                ],
                "provider_response": "Bearer top-level-secret",
            }
        )


@pytest.mark.parametrize("extra", [
    {"audited": True}, {"evidence_parent_ids": ["p1"]},
    {"citation_coverage": 1},
    {"segments": [{"kind": "content", "text": "hello", "evidence_ids": ["e1"]}]},
])
def test_chat_cannot_claim_document_audit_or_citations(extra: dict[str, object]) -> None:
    from agentic_rag.query.public_answer import project_public_answer

    assert project_public_answer({
        "route": "chat", "segments": [{"kind": "content", "text": "hello"}],
        **extra,
    }) is None


def test_chat_does_not_bypass_explicit_audited_projection() -> None:
    from agentic_rag.query.public_answer import project_public_answer

    answer = {"route": "chat", "segments": [{"kind": "content", "text": "hello"}]}
    assert project_public_answer(answer) is not None
    assert project_public_answer(answer, require_audited=True) is None


def test_public_answer_projection_keeps_only_reviewed_fields() -> None:
    from agentic_rag.query.public_answer import project_public_answer

    projected = project_public_answer(
        {
            "audited": True,
            "segments": [
                {
                    "kind": "content",
                    "text": "safe answer",
                    "evidence_ids": ["e1"],
                    "prompt": "never expose nested prompt",
                }
            ],
            "prompt": "never expose prompt",
            "provider_response": "Bearer never expose provider response",
            "tool_input": {"authorization": "secret"},
            "raw": "secret raw response",
            "unknown": "secret unknown value",
        },
        evidence_parent_ids=("parent-1",),
        route="research",
        runtime_config_snapshot_id="snapshot-1",
        require_audited=True,
    )

    assert projected is not None
    assert projected.model_dump(mode="json", exclude_none=True) == {
        "audited": True,
        "segments": [
            {"kind": "content", "text": "safe answer", "evidence_ids": ["e1"]}
        ],
        "evidence_parent_ids": ["parent-1"],
        "route": "research",
        "runtime_config_snapshot_id": "snapshot-1",
    }


@pytest.mark.parametrize(
    "answer",
    [
        {"segments": [{"kind": "content", "text": "not audited"}]},
        {"audited": True, "segments": []},
        {"audited": True, "segments": [{"kind": "content", "text": "x" * 8_001}]},
    ],
)
def test_completed_projection_fails_closed_without_valid_audited_segments(
    answer: dict[str, object],
) -> None:
    from agentic_rag.query.public_answer import project_public_answer

    assert project_public_answer(answer, require_audited=True) is None
