"""Deterministic, tenant-safe packing of retrieved evidence."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from agentic_rag.domain.models import UserScope
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.safety.context import DataEnvelope


MAX_PACKED_EVIDENCE_TOKENS = 12_000
MAX_ITEMS_PER_DOCUMENT = 3


class EvidenceCoverageTarget(BaseModel):
    """A question or Todo that retrieval evidence should preferentially cover."""

    model_config = ConfigDict(frozen=True)

    target_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class EvidenceItem(BaseModel):
    """One prompt-safe parent fragment and its immutable source provenance."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    parent_id: str
    document_id: str
    document_version_id: str
    content: str
    ast_locator: str
    covered_target_ids: tuple[str, ...]


class EvidenceManifestEntry(BaseModel):
    """Citation-validatable metadata for an included evidence item only."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    parent_id: str
    document_id: str
    document_version_id: str
    ast_locator: str


class PackedEvidence(BaseModel):
    """The bounded evidence payload passed to later query-runtime nodes."""

    model_config = ConfigDict(frozen=True)

    items: tuple[EvidenceItem, ...]
    manifest: Mapping[str, EvidenceManifestEntry]
    rendered_context: str
    token_count: int
    index_generation: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    parent: ParentEvidence
    locator: str
    target_ids: tuple[str, ...]
    is_direct_single_document: bool
    batch_position: int
    parent_position: int


class EvidenceBuilder:
    """Pack only internally consistent, tenant-scoped parent evidence.

    Token accounting is intentionally local and deterministic: each Unicode
    code point counts as one token. This deliberately conservative upper-bound
    avoids undercounting mixed-script evidence without downloading a tokenizer;
    downstream model gateways can apply any model-specific limit separately.
    """

    def build(
        self,
        batches: Sequence[EvidenceBatch],
        coverage_targets: Sequence[EvidenceCoverageTarget],
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        max_tokens: int = MAX_PACKED_EVIDENCE_TOKENS,
    ) -> PackedEvidence:
        """Return bounded data envelopes for valid evidence, or an empty pack."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        capacity = min(
            max_tokens, snapshot.max_evidence_tokens, MAX_PACKED_EVIDENCE_TOKENS
        )
        candidates = self._candidates(batches, coverage_targets, scope)
        descriptions_by_id = {
            target.target_id: target.description.casefold()
            for target in coverage_targets
        }

        items: list[EvidenceItem] = []
        rendered: list[str] = []
        document_counts: Counter[str] = Counter()
        selected_keys: set[tuple[str, str]] = set()

        for target in coverage_targets:
            for candidate in candidates:
                key = (candidate.parent.parent_id, candidate.parent.document_version_id)
                if target.target_id not in candidate.target_ids or key in selected_keys:
                    continue
                packed = self._pack_candidate(
                    candidate, rendered, capacity, descriptions_by_id
                )
                if (
                    packed is None
                    or target.target_id not in packed[0].covered_target_ids
                    or not _within_document_limit(candidate, document_counts)
                ):
                    continue
                item, rendered_envelope = packed
                items.append(item)
                rendered.append(rendered_envelope)
                selected_keys.add(key)
                document_counts[item.document_id] += 1
                break

        for candidate in candidates:
            key = (candidate.parent.parent_id, candidate.parent.document_version_id)
            if key in selected_keys:
                continue
            if (
                not candidate.is_direct_single_document
                and document_counts[candidate.parent.document_id]
                >= MAX_ITEMS_PER_DOCUMENT
            ):
                continue
            packed = self._pack_candidate(
                candidate, rendered, capacity, descriptions_by_id
            )
            if packed is None:
                continue
            item, rendered_envelope = packed
            items.append(item)
            rendered.append(rendered_envelope)
            document_counts[item.document_id] += 1

        manifest = {
            item.evidence_id: EvidenceManifestEntry(
                evidence_id=item.evidence_id,
                parent_id=item.parent_id,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                ast_locator=item.ast_locator,
            )
            for item in items
        }
        rendered_context = "\n".join(rendered)
        return PackedEvidence(
            items=tuple(items),
            manifest=manifest,
            rendered_context=rendered_context,
            token_count=_estimate_tokens(rendered_context),
            index_generation=snapshot.index_generation,
        )

    def _pack_candidate(
        self,
        candidate: _Candidate,
        rendered: Sequence[str],
        capacity: int,
        descriptions_by_id: Mapping[str, str],
    ) -> tuple[EvidenceItem, str] | None:
        item = self._item(candidate)
        envelope = DataEnvelope(
            source_label=f"document:{item.document_id}",
            evidence_id=item.evidence_id,
            content=item.content,
        )
        included = self._fit(
            envelope,
            rendered,
            capacity,
            candidate.parent.child_hits[0].content,
        )
        if included is None:
            return None
        fitted_content, rendered_envelope = included
        retained_target_ids = tuple(
            target_id
            for target_id in item.covered_target_ids
            if descriptions_by_id[target_id] in fitted_content.casefold()
        )
        return (
            item.model_copy(
                update={
                    "content": fitted_content,
                    "covered_target_ids": retained_target_ids,
                }
            ),
            rendered_envelope,
        )

    def _candidates(
        self,
        batches: Sequence[EvidenceBatch],
        targets: Sequence[EvidenceCoverageTarget],
        scope: UserScope,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for batch_position, batch in enumerate(batches):
            for parent_position, parent in enumerate(batch.parents):
                child_hits = _valid_child_hits(parent, scope)
                if not child_hits:
                    continue
                locator = child_hits[0].ast_locator
                haystack = "\n".join(
                    (parent.content, *(child.content for child in child_hits))
                ).casefold()
                target_ids = tuple(
                    target.target_id
                    for target in targets
                    if target.description.casefold() in haystack
                )
                candidates.append(
                    _Candidate(
                        parent=parent,
                        locator=locator,
                        target_ids=target_ids,
                        is_direct_single_document=(
                            len(batch.document_ids) == 1
                            and batch.document_ids[0] == parent.document_id
                        ),
                        batch_position=batch_position,
                        parent_position=parent_position,
                    )
                )

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.parent.rerank_score,
                candidate.batch_position,
                candidate.parent_position,
                candidate.parent.parent_id,
            ),
        )
        unique: list[_Candidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in ranked:
            key = (candidate.parent.parent_id, candidate.parent.document_version_id)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    @staticmethod
    def _item(candidate: _Candidate) -> EvidenceItem:
        parent = candidate.parent
        evidence_id = hashlib.sha256(
            "\x1f".join(
                (
                    parent.parent_id,
                    parent.document_id,
                    parent.document_version_id,
                    candidate.locator,
                )
            ).encode()
        ).hexdigest()[:32]
        return EvidenceItem(
            evidence_id=evidence_id,
            parent_id=parent.parent_id,
            document_id=parent.document_id,
            document_version_id=parent.document_version_id,
            content=_crop_around_child(
                parent.content, candidate.parent.child_hits[0].content
            ),
            ast_locator=candidate.locator,
            covered_target_ids=candidate.target_ids,
        )

    @staticmethod
    def _fit(
        envelope: DataEnvelope,
        rendered: Sequence[str],
        capacity: int,
        matching_child_content: str,
    ) -> tuple[str, str] | None:
        separator = "\n" if rendered else ""
        existing = "\n".join(rendered)
        full = envelope.render()
        if _estimate_tokens(f"{existing}{separator}{full}") <= capacity:
            return envelope.content, full

        low, high = 0, len(envelope.content)
        best: tuple[str, str] | None = None
        while low <= high:
            length = (low + high) // 2
            cropped = _crop_around_child(
                envelope.content, matching_child_content, length
            )
            candidate = envelope.model_copy(update={"content": cropped}).render()
            if _estimate_tokens(f"{existing}{separator}{candidate}") <= capacity:
                best = (cropped, candidate)
                low = length + 1
            else:
                high = length - 1
        return best


def _valid_child_hits(parent: ParentEvidence, scope: UserScope) -> tuple[ChildHit, ...]:
    """Require child-backed tenant and provenance proof for every parent."""
    valid = tuple(
        child
        for child in parent.child_hits
        if child.user_id == scope.user_id
        and child.parent_id == parent.parent_id
        and child.document_id == parent.document_id
        and child.document_version_id == parent.document_version_id
    )
    return valid if len(valid) == len(parent.child_hits) else ()


def _within_document_limit(
    candidate: _Candidate, document_counts: Counter[str]
) -> bool:
    """Apply diversity only when the request is not a direct document lookup."""
    return (
        candidate.is_direct_single_document
        or document_counts[candidate.parent.document_id] < MAX_ITEMS_PER_DOCUMENT
    )


def _crop_around_child(content: str, child_content: str, limit: int = 1_200) -> str:
    """Deterministically preserve a matching child passage when clipping a parent."""
    if limit < 1:
        return ""
    if len(content) <= limit:
        return content
    position = content.find(child_content) if child_content else len(content) // 2
    if position < 0:
        position = len(content) // 2
    start = max(0, position - limit // 2)
    end = min(len(content), start + limit)
    start = max(0, end - limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(content) else ""
    available = max(0, limit - len(prefix) - len(suffix))
    end = min(len(content), start + available)
    return f"{prefix}{content[start:end]}{suffix}"


def _estimate_tokens(text: str) -> int:
    """Return the documented conservative codepoint token upper bound."""
    return len(text)
