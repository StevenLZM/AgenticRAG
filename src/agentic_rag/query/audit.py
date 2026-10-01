"""Mandatory semantic and deterministic answer-evidence gates."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from agentic_rag.domain.models import UserScope
from agentic_rag.observability.logging import emit_degradation
from agentic_rag.models.schemas import EvidenceGrade, EvidenceGradeV2
from agentic_rag.query.routing_context import RoutingContext
from agentic_rag.query.routing_policy import RuntimeCapabilities
from agentic_rag.persistence.repositories import ParentRepository
from agentic_rag.query.evidence_builder import (
    EvidenceItem,
    EvidenceManifestEntry,
    PackedEvidence,
    model_evidence_manifest,
)
from agentic_rag.query.generation import (
    AnswerDraft,
    AnswerGenerationUnavailable,
    AnswerGenerator,
)
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, load_prompt
from agentic_rag.runtime.models import RuntimeConfigSnapshot


EVIDENCE_GRADER_MAX_OUTPUT_TOKENS = 99_999
FAITHFULNESS_AUDITOR_MAX_OUTPUT_TOKENS = 99_999


class FaithfulnessAudit(BaseModel):
    """Semantic support decision; citation shape belongs to CitationValidator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    unsupported_claim_ids: tuple[
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
        ],
        ...,
    ] = Field(default=(), max_length=32)
    reasons: tuple[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
        ],
        ...,
    ] = Field(default=(), max_length=32)


class CitationValidation(BaseModel):
    """Deterministic citation decision with bounded machine-readable reasons."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    reasons: tuple[
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
        ],
        ...,
    ] = Field(default=(), max_length=64)


class StructuredGateway(Protocol):
    async def complete_structured(
        self, call: ModelCall, schema: type[object]
    ) -> object: ...


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


class EvidenceGradingUnavailable(RuntimeError):
    """A technical grading failure, never a claim about document coverage."""


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
        routing_context: RoutingContext | None = None,
        capabilities: RuntimeCapabilities | None = None,
    ) -> EvidenceGrade:
        if not packed_evidence.items or not packed_evidence.manifest:
            if capabilities is not None:
                return EvidenceGradeV2(decision="insufficient", gap_type="unknown",
                                       gaps=("no verified evidence is available",))
            return EvidenceGrade(
                decision="insufficient", gaps=("no verified evidence is available",)
            )
        payload = {
            "question": question,
            "evidence_manifest": model_evidence_manifest(packed_evidence.manifest),
            # EvidenceBuilder has already applied the server-side token and
            # tenant/provenance limits.  The grader needs the bounded text to
            # decide sufficiency; it must still treat that text as untrusted
            # data, never as instructions.
            "packed_context": packed_evidence.rendered_context,
            "packed_evidence_metadata": {
                "item_count": len(packed_evidence.items),
                "index_generation": packed_evidence.index_generation,
            },
            "user_scope": scope.model_dump(mode="json"),
            "routing_context": routing_context.model_dump(mode="json") if routing_context else None,
            "capabilities": capabilities.model_dump(mode="json") if capabilities else None,
        }
        call = ModelCall(
            model_role="light",
            snapshot=snapshot,
            max_output_tokens=EVIDENCE_GRADER_MAX_OUTPUT_TOKENS,
            messages=(
                {
                    "role": "system",
                    "content": load_prompt("evidence_grader_v2" if capabilities is not None else "evidence_grader_v1").content,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ),
        )
        try:
            schema = EvidenceGradeV2 if capabilities is not None else EvidenceGrade
            response = await self._gateway.complete_structured(call, schema)
            value = getattr(response, "value", response)
            grade = schema.model_validate(value)
            if grade.decision == "sufficient" and not packed_evidence.items:
                return EvidenceGrade(
                    decision="insufficient", gaps=("no verified evidence is available",)
                )
            return grade
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            if capabilities is not None:
                raise EvidenceGradingUnavailable("evidence_grader_unavailable") from error
            return EvidenceGrade(
                decision="insufficient", gaps=("evidence grading unavailable",)
            )


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
            "answer_segments": [
                segment.model_dump(mode="json") for segment in draft.segments
            ],
            "evidence_manifest": model_evidence_manifest(packed_evidence.manifest),
            "packed_context": packed_evidence.rendered_context,
            "user_scope": scope.model_dump(mode="json"),
        }
        call = ModelCall(
            model_role="light",
            snapshot=snapshot,
            max_output_tokens=FAITHFULNESS_AUDITOR_MAX_OUTPUT_TOKENS,
            messages=(
                {"role": "system", "content": load_prompt("faithfulness_v1").content},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            ),
        )
        try:
            response = await self._gateway.complete_structured(call, FaithfulnessAudit)
            value = getattr(response, "value", response)
            return (
                value
                if isinstance(value, FaithfulnessAudit)
                else FaithfulnessAudit.model_validate(value)
            )
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            return FaithfulnessAudit(
                passed=False,
                reasons=(f"faithfulness_audit_unavailable:{type(error).__name__}",),
            )


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
                elif not _authorization_allows(
                    authorization.get(evidence_id),
                    manifest[evidence_id],
                    scope,
                    snapshot,
                ):
                    reasons.append("evidence_not_authorized")
        return CitationValidation(
            passed=not reasons, reasons=tuple(dict.fromkeys(reasons))
        )

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
            return CitationValidation(
                passed=False, reasons=("authorization_unavailable",)
            )
        if not isinstance(resolved, Mapping):
            return CitationValidation(
                passed=False, reasons=("authorization_malformed",)
            )
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
    raw_run_id = state.get("run_id")
    run_id = raw_run_id if isinstance(raw_run_id, str) else None
    repair_issues: tuple[str, ...] = ()
    for attempt in range(2):
        try:
            draft = await generator.generate(
                question, packed_evidence, snapshot, repair_issues=repair_issues
            )
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except AnswerGenerationUnavailable as error:
            await emit_degradation(
                component="generation",
                reason="generation_unavailable",
                run_id=run_id,
                snapshot_id=snapshot.snapshot_id,
                attempt=attempt + 1,
                retryable=True,
                outcome="refused",
            )
            return _refusal(
                revision_count,
                prior_audits,
                [
                    *prior_errors,
                    {"code": "generation_unavailable", "detail": type(error).__name__},
                ],
            )
        faithfulness = await faithfulness_auditor.audit(
            question, draft, packed_evidence, scope=scope, snapshot=snapshot
        )
        if authorization_resolver is None:
            citation = CitationValidation(
                passed=False, reasons=("authorization_resolver_required",)
            )
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
                "answer": {
                    "segments": [
                        segment.model_dump(mode="json") for segment in draft.segments
                    ],
                    "audited": True,
                },
                "audit_results": prior_audits,
                "revision_count": revision_count,
                "errors": prior_errors,
                "termination_reason": None,
                "next_node": "end",
            }
        repair_issues = (
            *citation.reasons,
            *faithfulness.reasons,
            *faithfulness.unsupported_claim_ids,
        )
        if attempt == 0 and revision_count < snapshot.max_answer_revisions:
            revision_count += 1
            continue
        await emit_degradation(
            component="audit",
            reason="audit_failed",
            run_id=run_id,
            snapshot_id=snapshot.snapshot_id,
            attempt=attempt + 1,
            retryable=False,
            outcome="refused",
            event_type="AUDIT_REFUSED",
        )
        return _refusal(revision_count, prior_audits, prior_errors)
    raise AssertionError("audit repair loop must return")


def _refusal(
    revision_count: int, audits: list[object], errors: list[object]
) -> dict[str, object]:
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


def _unique_items(
    items: tuple[EvidenceItem, ...], reasons: list[str]
) -> dict[str, EvidenceItem]:
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
        and entry.heading_path == item.heading_path
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
        and _heading_path_is_authorized(value.get("heading_path"), entry.heading_path)
        and _ast_locator_is_authorized(value.get("ast_locator"), entry.ast_locator)
    )


def _heading_path_is_authorized(value: object, expected: tuple[str, ...]) -> bool:
    """Require repository heading metadata to match the immutable manifest."""
    if value is None:
        # Legacy test doubles/rows pre-dating heading metadata represent the
        # empty path only; never let an omitted value authorize a non-empty path.
        return expected == ()
    if isinstance(value, tuple):
        actual = value
    elif isinstance(value, list):
        actual = tuple(value)
    else:
        return False
    return all(isinstance(part, str) for part in actual) and actual == expected


def _ast_locator_is_authorized(parent_locator: object, child_locator: str) -> bool:
    """Allow a Child locator only when it is contained by its authorized Parent."""
    if not isinstance(parent_locator, str):
        return False
    if parent_locator == child_locator:
        # Keep opaque legacy locators valid when the repository and Manifest agree.
        return True

    parent = _decode_ast_locator(parent_locator)
    child = _decode_ast_locator(child_locator)
    if parent is None or child is None:
        return False

    parent_range = _ast_range(parent, "parent_char_from", "parent_char_to")
    child_range = _ast_range(child, "parent_char_from", "parent_char_to")
    if parent_range is None or child_range is None:
        return False
    if not _range_within(child_range, parent_range):
        return False

    parent_spans = parent.get("spans")
    child_spans = child.get("spans")
    if not isinstance(parent_spans, list) or not isinstance(child_spans, list):
        return False
    if not parent_spans or not child_spans:
        return False

    previous_parent_index = -1
    for child_span in child_spans:
        matching_parent_index: int | None = None
        for parent_index in range(previous_parent_index + 1, len(parent_spans)):
            parent_span = parent_spans[parent_index]
            if _span_within(child_span, parent_span, child_range, parent_range):
                matching_parent_index = parent_index
                break
        if matching_parent_index is None:
            return False
        previous_parent_index = matching_parent_index
    return True


def _decode_ast_locator(value: str) -> Mapping[str, object] | None:
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _ast_range(
    value: Mapping[str, object], start_key: str, end_key: str
) -> tuple[int, int] | None:
    start = value.get(start_key)
    end = value.get(end_key)
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    return start, end


def _range_within(inner: tuple[int, int], outer: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _span_within(
    child: object,
    parent: object,
    child_locator_range: tuple[int, int],
    parent_locator_range: tuple[int, int],
) -> bool:
    if not isinstance(child, Mapping) or not isinstance(parent, Mapping):
        return False
    if child.get("block_id") != parent.get("block_id") or child.get(
        "canonical_path"
    ) != parent.get("canonical_path"):
        return False

    child_canonical_range = _ast_range(child, "char_from", "char_to")
    parent_canonical_range = _ast_range(parent, "char_from", "char_to")
    child_parent_range = _ast_range(child, "parent_char_from", "parent_char_to")
    parent_parent_range = _ast_range(parent, "parent_char_from", "parent_char_to")
    if (
        child_canonical_range is None
        or parent_canonical_range is None
        or child_parent_range is None
        or parent_parent_range is None
    ):
        return False
    return (
        _range_within(child_canonical_range, parent_canonical_range)
        and _range_within(child_parent_range, parent_parent_range)
        and _range_within(child_parent_range, child_locator_range)
        and _range_within(parent_parent_range, parent_locator_range)
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
        "heading_path": tuple(getattr(parent, "heading_path", ()) or ()),
    }
