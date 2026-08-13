"""Mandatory semantic and deterministic answer-evidence gates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentic_rag.domain.models import UserScope
from agentic_rag.models.schemas import EvidenceGrade
from agentic_rag.persistence.repositories import ParentRepository
from agentic_rag.query.evidence_builder import EvidenceItem, EvidenceManifestEntry, PackedEvidence
from agentic_rag.query.generation import AnswerDraft, AnswerGenerationUnavailable, AnswerGenerator
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, load_prompt
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class FaithfulnessAudit(BaseModel):
    """Semantic support decision; citation shape belongs to CitationValidator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    unsupported_claim_ids: tuple[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)], ...
    ] = Field(default=(), max_length=32)
    reasons: tuple[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)], ...
    ] = Field(default=(), max_length=32)


class CitationValidation(BaseModel):
    """Deterministic citation decision with bounded machine-readable reasons."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reasons: tuple[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)], ...
    ] = Field(default=(), max_length=64)


class StructuredGateway(Protocol):
    async def complete_structured(self, call: ModelCall, schema: type[object]) -> object: ...


class EvidenceAuthorizationResolver(Protocol):
    """Repository-backed recheck for every manifest parent before publication."""

    async def resolve(
        self,
        manifest: Mapping[str, EvidenceManifestEntry],
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> Mapping[str, object]: ...


class ParentRepositoryAuthorizationResolver:
    """Adapt the scoped active-parent repository to the audit resolver port."""

    def __init__(self, parents: ParentRepository) -> None:
        self._parents = parents

    async def resolve(
        self,
        manifest: Mapping[str, EvidenceManifestEntry],
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> Mapping[str, object]:
        parent_ids = list(dict.fromkeys(entry.parent_id for entry in manifest.values()))
        rows = await self._parents.get_many(parent_ids, scope)
        by_id = {row.id: row for row in rows}
        return {
            evidence_id: _parent_record(by_id.get(entry.parent_id), scope, snapshot)
            for evidence_id, entry in manifest.items()
        }


class EvidenceGrader:
    """Light-model evidence coverage decision, conservative for empty evidence."""

    def __init__(self, gateway: StructuredGateway | ModelGateway) -> None:
        self._gateway = gateway

    async def grade(
        self,
        question: str,
        packed_evidence: PackedEvidence,
        *,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> EvidenceGrade:
        if not packed_evidence.items or not packed_evidence.manifest:
            return EvidenceGrade(decision="insufficient", gaps=("no verified evidence is available",))
        payload = {
            "question": question,
            "evidence_manifest": {key: value.model_dump(mode="json") for key, value in packed_evidence.manifest.items()},
            "packed_evidence_metadata": {"item_count": len(packed_evidence.items), "index_generation": packed_evidence.index_generation},
            "user_scope": scope.model_dump(mode="json"),
        }
        call = ModelCall(
            model_role="light",
            snapshot=snapshot,
            messages=(
                {"role": "system", "content": load_prompt("evidence_grader_v1").content},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ),
        )
        try:
            response = await self._gateway.complete_structured(call, EvidenceGrade)
            value = getattr(response, "value", response)
            grade = value if isinstance(value, EvidenceGrade) else EvidenceGrade.model_validate(value)
            if grade.decision == "sufficient" and not packed_evidence.items:
                return EvidenceGrade(decision="insufficient", gaps=("no verified evidence is available",))
            return grade
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return EvidenceGrade(decision="insufficient", gaps=("evidence grading unavailable",))


class FaithfulnessAuditor:
    """Light-model semantic support audit, deliberately not a citation parser."""

    def __init__(self, gateway: StructuredGateway | ModelGateway) -> None:
        self._gateway = gateway

    async def audit(
        self,
        question: str,
        draft: AnswerDraft,
        packed_evidence: PackedEvidence,
        *,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
    ) -> FaithfulnessAudit:
        payload = {
            "question": question,
            "answer_segments": [segment.model_dump(mode="json") for segment in draft.segments],
            "evidence_manifest": {key: value.model_dump(mode="json") for key, value in packed_evidence.manifest.items()},
            "packed_context": packed_evidence.rendered_context,
            "user_scope": scope.model_dump(mode="json"),
        }
        call = ModelCall(
            model_role="light",
            snapshot=snapshot,
            messages=(
                {"role": "system", "content": load_prompt("faithfulness_v1").content},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ),
        )
        try:
            response = await self._gateway.complete_structured(call, FaithfulnessAudit)
            value = getattr(response, "value", response)
            return value if isinstance(value, FaithfulnessAudit) else FaithfulnessAudit.model_validate(value)
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            return FaithfulnessAudit(passed=False, reasons=(f"faithfulness_audit_unavailable:{type(error).__name__}",))


class CitationValidator:
    """Validate provenance and scope deterministically without a model call."""

    def validate(
        self,
        draft: AnswerDraft,
        packed_evidence: PackedEvidence,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        authorization: Mapping[str, object],
    ) -> CitationValidation:
        reasons: list[str] = []
        items_by_id = _unique_items(packed_evidence.items, reasons)
        manifest = packed_evidence.manifest
        if packed_evidence.index_generation != snapshot.index_generation:
            reasons.append("stale_index_generation")
        for evidence_id, entry in manifest.items():
            item = items_by_id.get(evidence_id)
            if item is None or not _entry_matches_item(entry, item):
                reasons.append("manifest_mismatch")
                continue
            record = authorization.get(evidence_id)
            if not _authorization_allows(record, entry, scope, snapshot):
                reasons.append("evidence_not_authorized")
        for segment in draft.segments:
            if segment.kind == "content" and not segment.evidence_ids:
                reasons.append("missing_evidence")
            for evidence_id in segment.evidence_ids:
                if evidence_id not in manifest or evidence_id not in items_by_id:
                    reasons.append("unknown_evidence")
                elif not _authorization_allows(authorization.get(evidence_id), manifest[evidence_id], scope, snapshot):
                    reasons.append("evidence_not_authorized")
        return CitationValidation(passed=not reasons, reasons=tuple(dict.fromkeys(reasons)))

    async def validate_async(
        self,
        draft: AnswerDraft,
        packed_evidence: PackedEvidence,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        authorization: Mapping[str, object],
        *,
        authorization_resolver: EvidenceAuthorizationResolver,
    ) -> CitationValidation:
        """Recheck manifest provenance through a scoped repository before publication."""
        try:
            resolved = await authorization_resolver.resolve(
                packed_evidence.manifest, scope, snapshot
            )
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return CitationValidation(passed=False, reasons=("authorization_unavailable",))
        if not isinstance(resolved, Mapping):
            return CitationValidation(passed=False, reasons=("authorization_malformed",))
        # The resolver result is authoritative; the mapping is only a local cache
        # and cannot make an inactive, cross-tenant, or stale row valid.
        return self.validate(draft, packed_evidence, scope, snapshot, resolved)


async def generate_with_mandatory_audits(
    *,
    question: str,
    state: Mapping[str, object],
    packed_evidence: PackedEvidence,
    scope: UserScope,
    snapshot: RuntimeConfigSnapshot,
    generator: AnswerGenerator,
    faithfulness_auditor: FaithfulnessAuditor,
    citation_validator: CitationValidator,
    authorization: Mapping[str, object],
    authorization_resolver: EvidenceAuthorizationResolver | None = None,
) -> dict[str, object]:
    """Generate at most twice; never return an unapproved draft in state."""
    raw_revision_count = state.get("revision_count", 0)
    revision_count = raw_revision_count if isinstance(raw_revision_count, int) else 0
    prior_audits = _as_list(state.get("audit_results"))
    prior_errors = _as_list(state.get("errors"))
    repair_issues: tuple[str, ...] = ()
    for attempt in range(2):
        try:
            draft = await generator.generate(question, packed_evidence, snapshot, repair_issues=repair_issues)
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except AnswerGenerationUnavailable as error:
            return _refusal(revision_count, prior_audits, [*prior_errors, {"code": "generation_unavailable", "detail": str(error)}])
        faithfulness = await faithfulness_auditor.audit(question, draft, packed_evidence, scope=scope, snapshot=snapshot)
        if authorization_resolver is None:
            citation = CitationValidation(passed=False, reasons=("authorization_resolver_required",))
        else:
            citation = await citation_validator.validate_async(
                draft,
                packed_evidence,
                scope,
                snapshot,
                authorization,
                authorization_resolver=authorization_resolver,
            )
        audit_record = {
            "citation": citation.model_dump(mode="json"),
            "faithfulness": faithfulness.model_dump(mode="json"),
        }
        prior_audits.append(audit_record)
        if citation.passed and faithfulness.passed:
            return {
                "answer": {"segments": [segment.model_dump(mode="json") for segment in draft.segments]},
                "audit_results": prior_audits,
                "revision_count": revision_count,
                "errors": prior_errors,
                "termination_reason": None,
                "next_node": "end",
            }
        repair_issues = (*citation.reasons, *faithfulness.reasons, *faithfulness.unsupported_claim_ids)
        if attempt == 0 and revision_count < snapshot.max_answer_revisions:
            revision_count += 1
            continue
        return _refusal(revision_count, prior_audits, prior_errors)
    raise AssertionError("audit repair loop must return")


def _refusal(revision_count: int, audits: list[object], errors: list[object]) -> dict[str, object]:
    return {
        "answer": {},
        "audit_results": audits,
        "revision_count": revision_count,
        "errors": errors,
        "termination_reason": "audit_failed",
        "next_node": "end",
    }


def _as_list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def _unique_items(items: tuple[EvidenceItem, ...], reasons: list[str]) -> dict[str, EvidenceItem]:
    result: dict[str, EvidenceItem] = {}
    for item in items:
        if item.evidence_id in result:
            reasons.append("duplicate_evidence")
        result[item.evidence_id] = item
    return result


def _entry_matches_item(entry: EvidenceManifestEntry, item: EvidenceItem) -> bool:
    return (
        entry.evidence_id == item.evidence_id
        and entry.parent_id == item.parent_id
        and entry.document_id == item.document_id
        and entry.document_version_id == item.document_version_id
        and entry.ast_locator == item.ast_locator
    )


def _authorization_allows(
    value: object,
    entry: EvidenceManifestEntry,
    scope: UserScope,
    snapshot: RuntimeConfigSnapshot,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        value.get("user_id") == scope.user_id
        and value.get("index_generation") == snapshot.index_generation
        and value.get("is_active") is True
        and value.get("parent_id") == entry.parent_id
        and value.get("document_id") == entry.document_id
        and value.get("document_version_id") == entry.document_version_id
        and value.get("ast_locator") == entry.ast_locator
    )


def _parent_record(
    parent: object,
    scope: UserScope,
    snapshot: RuntimeConfigSnapshot,
) -> Mapping[str, object]:
    if parent is None:
        return {}
    user_id = getattr(parent, "user_id", None)
    status = getattr(parent, "status", None)
    return {
        "user_id": user_id if user_id is not None else scope.user_id,
        "index_generation": getattr(parent, "index_generation", None),
        "is_active": status == "active",
        "parent_id": getattr(parent, "id", None),
        "document_id": getattr(parent, "document_id", None),
        "document_version_id": getattr(parent, "document_version_id", None),
        "ast_locator": getattr(parent, "ast_locator", None),
    }
