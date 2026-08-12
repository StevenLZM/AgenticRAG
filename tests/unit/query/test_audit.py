"""Behavioral coverage for evidence-grounded answer generation and audits."""

from __future__ import annotations

import asyncio

import pytest

from agentic_rag.domain.models import UserScope
from agentic_rag.query.evidence_builder import (
    EvidenceItem,
    EvidenceManifestEntry,
    PackedEvidence,
)
from agentic_rag.runtime.models import RuntimeConfigSnapshot


SCOPE = UserScope(user_id="user-1")
SNAPSHOT = RuntimeConfigSnapshot(
    app_version="test",
    graph_version="test",
    prompt_version="test",
    main_model_id="main-test",
    light_model_id="light-test",
    embedding_model="text-embedding-v3",
    embedding_dimensions=1024,
    reranker_version="test",
    retrieval_config_version="test",
    index_generation="index-current",
    memory_config_version="test",
)


def packed_evidence(*, index_generation: str = "index-current") -> PackedEvidence:
    item = EvidenceItem(
        evidence_id="evidence-1",
        parent_id="parent-1",
        document_id="document-1",
        document_version_id="version-1",
        ast_locator="#/paragraphs/1",
        content="The contract term is three years.",
        covered_target_ids=(),
    )
    entry = EvidenceManifestEntry(
        evidence_id=item.evidence_id,
        parent_id=item.parent_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        ast_locator=item.ast_locator,
    )
    return PackedEvidence(
        items=(item,),
        manifest={item.evidence_id: entry},
        rendered_context="evidence",
        token_count=8,
        index_generation=index_generation,
    )


def authorization(*, user_id: str = "user-1", index_generation: str = "index-current") -> dict[str, object]:
    return {
        "evidence-1": {
            "user_id": user_id,
            "index_generation": index_generation,
            "is_active": True,
            "parent_id": "parent-1",
            "document_id": "document-1",
            "document_version_id": "version-1",
            "ast_locator": "#/paragraphs/1",
        }
    }


def test_answer_segment_is_immutable_bounded_and_forbids_unknown_fields() -> None:
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment

    segment = AnswerSegment(
        kind="content", text="Three years", evidence_ids=("evidence-1",)
    )
    assert AnswerDraft(segments=(segment,)).segments == (segment,)
    with pytest.raises((TypeError, ValueError)):
        AnswerSegment(kind="content", text="x", unexpected="not allowed")
    with pytest.raises((TypeError, ValueError)):
        segment.text = "mutated"


def test_citation_validator_requires_evidence_for_every_content_segment() -> None:
    from agentic_rag.query.audit import CitationValidator
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment

    result = CitationValidator().validate(
        AnswerDraft(segments=(AnswerSegment(kind="content", text="期限为三年"),)),
        packed_evidence(),
        SCOPE,
        SNAPSHOT,
        authorization(),
    )

    assert result.passed is False
    assert "missing_evidence" in result.reasons


def test_citation_validator_rejects_invented_and_cross_scope_or_stale_evidence() -> None:
    from agentic_rag.query.audit import CitationValidator
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment

    validator = CitationValidator()
    draft = AnswerDraft(
        segments=(
            AnswerSegment(
                kind="content", text="The term is three years.", evidence_ids=("evidence-1",)
            ),
        )
    )
    assert not validator.validate(
        draft, packed_evidence(), SCOPE, SNAPSHOT, authorization(user_id="user-2")
    ).passed
    assert not validator.validate(
        draft, packed_evidence(), SCOPE, SNAPSHOT, authorization(index_generation="old")
    ).passed
    invented = draft.model_copy(
        update={"segments": (draft.segments[0].model_copy(update={"evidence_ids": ("invented",)}),)}
    )
    assert not validator.validate(
        invented, packed_evidence(), SCOPE, SNAPSHOT, authorization()
    ).passed


def test_citation_validator_rejects_conflicting_manifest_metadata() -> None:
    from agentic_rag.query.audit import CitationValidator
    from agentic_rag.query.generation import AnswerDraft, AnswerSegment

    evidence = packed_evidence().model_copy(
        update={
            "manifest": {
                "evidence-1": EvidenceManifestEntry(
                    evidence_id="evidence-1",
                    parent_id="different-parent",
                    document_id="document-1",
                    document_version_id="version-1",
                    ast_locator="#/paragraphs/1",
                )
            }
        }
    )
    result = CitationValidator().validate(
        AnswerDraft(segments=(AnswerSegment(kind="content", text="term", evidence_ids=("evidence-1",)),)),
        evidence,
        SCOPE,
        SNAPSHOT,
        authorization(),
    )
    assert result.passed is False
    assert "manifest_mismatch" in result.reasons


class FakeGateway:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[object] = []

    async def complete_structured(self, call: object, schema: type[object]) -> object:
        self.calls.append(call)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return type("Response", (), {"value": response})()


@pytest.mark.asyncio
async def test_evidence_grader_fails_closed_without_evidence() -> None:
    from agentic_rag.query.audit import EvidenceGrader

    gateway = FakeGateway([{"decision": "sufficient", "gaps": []}])
    empty = packed_evidence().model_copy(update={"items": (), "manifest": {}})
    grade = await EvidenceGrader(gateway).grade("What is the term?", empty, scope=SCOPE, snapshot=SNAPSHOT)

    assert grade.decision == "insufficient"
    assert gateway.calls == []


@pytest.mark.asyncio
async def test_generation_and_audit_repair_once_then_refuses_without_draft() -> None:
    from agentic_rag.query.audit import (
        CitationValidator,
        FaithfulnessAuditor,
        generate_with_mandatory_audits,
    )
    from agentic_rag.query.generation import AnswerGenerator

    gateway = FakeGateway([
        {"segments": [{"kind": "content", "text": "Unsupported claim", "evidence_ids": ["evidence-1"]}]},
        {"passed": False, "unsupported_claim_ids": ["claim-1"], "reasons": ["unsupported"]},
        {"segments": [{"kind": "content", "text": "Still unsupported", "evidence_ids": ["evidence-1"]}]},
        {"passed": False, "unsupported_claim_ids": ["claim-2"], "reasons": ["still unsupported"]},
    ])
    result = await generate_with_mandatory_audits(
        question="What is the term?",
        state={"revision_count": 0, "audit_results": [], "errors": []},
        packed_evidence=packed_evidence(),
        scope=SCOPE,
        snapshot=SNAPSHOT,
        generator=AnswerGenerator(gateway),
        faithfulness_auditor=FaithfulnessAuditor(gateway),
        citation_validator=CitationValidator(),
        authorization=authorization(),
    )

    assert result["termination_reason"] == "audit_failed"
    assert result["answer"] == {}
    assert result["revision_count"] == 1
    assert len(gateway.calls) == 4


@pytest.mark.asyncio
async def test_generation_operational_failure_refuses_but_cancellation_propagates() -> None:
    from agentic_rag.query.generation import AnswerGenerator, AnswerGenerationUnavailable

    generator = AnswerGenerator(FakeGateway([OSError("provider unavailable")]))
    with pytest.raises(AnswerGenerationUnavailable):
        await generator.generate("question", packed_evidence(), SNAPSHOT)

    cancelled = AnswerGenerator(FakeGateway([asyncio.CancelledError()]))
    with pytest.raises(asyncio.CancelledError):
        await cancelled.generate("question", packed_evidence(), SNAPSHOT)
