"""Deterministic, tenant-safe packing of retrieved evidence."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal

from agentic_rag.models.contextual import (
    CanonicalSourceRef,
    ContextArtifactRef,
    SupportingContext,
    EvidenceRequirement,
    text_sha256,
)

from agentic_rag.domain.models import UserScope
from agentic_rag.retrieval.models import ChildHit, EvidenceBatch, ParentEvidence
from agentic_rag.runtime.models import RuntimeConfigSnapshot
from agentic_rag.safety.context import DataEnvelope


MAX_PACKED_EVIDENCE_TOKENS = 12_000
MAX_ITEMS_PER_DOCUMENT = 3
MAX_ITEMS_SINGLE_DOCUMENT = 6


class EvidenceCoverageTarget(BaseModel):
    """A question or Todo that retrieval evidence should preferentially cover."""

    model_config = ConfigDict(frozen=True)

    target_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class EvidenceManifestEntry(BaseModel):
    """Versioned source union; metadata-only evidence is deliberately disabled."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1, 2] = 1
    evidence_id: str
    source_kind: Literal["parent_chunk", "canonical_excerpt"] = "parent_chunk"
    parent_id: str | None
    document_id: str
    document_version_id: str
    ast_locator: str
    heading_path: tuple[str, ...] = ()
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_excerpt_length: int | None = Field(
        default=None, ge=1, le=MAX_PACKED_EVIDENCE_TOKENS
    )
    source_ref: CanonicalSourceRef | None = None
    build_fingerprint: str | None = None
    context_refs: tuple[ContextArtifactRef, ...] = ()
    supports_claim_ids: tuple[str, ...] = ()
    support_source_refs: tuple[CanonicalSourceRef, ...] = ()
    source_child_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def source_contract(self):
        if self.schema_version == 1 and (
            self.parent_excerpt_length is not None
            or self.content_hash is not None
            or self.context_refs
            or self.source_ref is not None
            or self.build_fingerprint is not None
            or self.supports_claim_ids
            or self.support_source_refs
            or self.source_child_ids
        ):
            raise ValueError("context provenance requires evidence schema v2")
        if self.schema_version == 2 and self.content_hash is None:
            raise ValueError("v2 evidence requires content hash")
        if self.source_kind == "parent_chunk":
            if not self.parent_id or self.source_ref is not None:
                raise ValueError("parent evidence requires real Parent identity")
            if self.context_refs and self.parent_excerpt_length is None:
                raise ValueError(
                    "contextual Parent requires reproducible excerpt length"
                )
        else:
            ref = self.source_ref
            if (
                self.schema_version != 2
                or self.parent_id is not None
                or ref is None
                or not self.build_fingerprint
                or not self.context_refs
                or not self.supports_claim_ids
                or not self.source_child_ids
                or self.support_source_refs
                or self.parent_excerpt_length is not None
            ):
                raise ValueError(
                    "canonical evidence requires source/build/support provenance"
                )
            if (
                ref.document_id != self.document_id
                or ref.document_version_id != self.document_version_id
                or ref.quote_sha256 != self.content_hash
                or ref.canonical_path != self.ast_locator
            ):
                raise ValueError("canonical evidence provenance mismatch")
        return self


class EvidenceItem(EvidenceManifestEntry):
    content: str
    covered_target_ids: tuple[str, ...]

    @model_validator(mode="after")
    def content_integrity(self):
        if self.parent_excerpt_length is not None and self.parent_excerpt_length != len(
            self.content
        ):
            raise ValueError("Parent excerpt length mismatch")
        if self.content_hash is not None and self.content_hash != text_sha256(
            self.content
        ):
            raise ValueError("evidence content hash mismatch")
        return self


def evidence_projection(item: EvidenceItem) -> dict:
    return item.model_dump(exclude={"content", "covered_target_ids"})


def matches_manifest(item: EvidenceItem, entry: EvidenceManifestEntry | None) -> bool:
    return entry is not None and evidence_projection(item) == entry.model_dump()


def merge_evidence_items(left: EvidenceItem, right: EvidenceItem) -> EvidenceItem:
    """Union support relations only after checking the complete source identity."""
    relations = {
        "supports_claim_ids",
        "support_source_refs",
        "source_child_ids",
        "context_refs",
        "covered_target_ids",
    }
    if left.model_dump(exclude=relations) != right.model_dump(exclude=relations):
        raise ValueError("conflicting evidence source identity")
    return EvidenceItem(
        **{
            **left.model_dump(),
            **{
                name: tuple(
                    dict.fromkeys((*getattr(left, name), *getattr(right, name)))
                )
                for name in relations
            },
        }
    )


class PackedEvidence(BaseModel):
    """The bounded evidence payload passed to later query-runtime nodes."""

    model_config = ConfigDict(frozen=True)

    items: tuple[EvidenceItem, ...]
    manifest: Mapping[str, EvidenceManifestEntry]
    rendered_context: str
    token_count: int
    index_generation: str
    supporting_context: SupportingContext = Field(default_factory=SupportingContext)
    evidence_requirements: tuple[EvidenceRequirement, ...] = ()
    scope_evidence_missing: tuple[str, ...] = ()


def support_links(packed: PackedEvidence) -> dict[str, tuple[str, ...]]:
    """Exact source closure for every body; reused by grading and final freeze."""
    links = {}
    for item in packed.items:
        if item.source_kind != "parent_chunk":
            continue
        required = [
            r
            for r in packed.evidence_requirements
            if r.child_id in item.source_child_ids
        ]
        refs = []
        for requirement in required:
            if (
                requirement.status != "resolved"
                or requirement.id in packed.scope_evidence_missing
            ):
                raise ValueError("scope_evidence_missing")
            for source_ref in requirement.source_refs:
                source = next(
                    (
                        s
                        for s in packed.supporting_context.sources
                        if s.source_ref == source_ref
                    ),
                    None,
                )
                match = next(
                    (
                        candidate
                        for candidate in packed.items
                        if candidate.source_ref == source_ref
                        or (
                            source is not None
                            and candidate.source_kind == "parent_chunk"
                            and source_ref in candidate.support_source_refs
                            and set(requirement.claim_ids)
                            <= set(candidate.supports_claim_ids)
                            and source.content in candidate.content
                        )
                    ),
                    None,
                )
                if match is None:
                    raise ValueError("scope_evidence_missing")
                if match.evidence_id != item.evidence_id:
                    refs.append(match.evidence_id)
        if refs:
            links[item.evidence_id] = tuple(dict.fromkeys(refs))
    return links


def expand_support_ids(ids, links):
    selected = set(ids)
    pending = list(selected)
    while pending:
        for dependency in links.get(pending.pop(), ()):
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return selected


def model_evidence_manifest(
    manifest: Mapping[str, EvidenceManifestEntry],
) -> dict[str, dict[str, object]]:
    """Return the allowlisted provenance fields needed by model prompts.

    The authoritative manifest, including ``ast_locator``, remains on
    ``PackedEvidence`` for deterministic citation and authorization checks.
    """
    return {
        evidence_id: {
            "evidence_id": entry.evidence_id,
            "parent_id": entry.parent_id,
            "document_id": entry.document_id,
            "document_version_id": entry.document_version_id,
            "heading_path": entry.heading_path,
            **(
                {
                    "source_kind": entry.source_kind,
                    "content_hash": entry.content_hash,
                    "supports_claim_ids": entry.supports_claim_ids,
                    "source_child_ids": entry.source_child_ids,
                }
                if entry.schema_version == 2
                else {}
            ),
        }
        for evidence_id, entry in manifest.items()
    }


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
        *,
        rank_across_batches: bool = True,
        balance_targets: bool = False,
    ) -> PackedEvidence:
        """Return bounded data envelopes for valid evidence, or an empty pack."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        capacity = min(
            max_tokens, snapshot.max_evidence_tokens, MAX_PACKED_EVIDENCE_TOKENS
        )
        if any(
            b.supporting_context.requirements
            or b.supporting_context.sources
            or b.query_plan is not None
            or any(h.context_ref for p in b.parents for h in p.child_hits)
            for b in batches
        ):
            return self._build_contextual(
                batches,
                coverage_targets,
                scope,
                snapshot,
                capacity,
                rank_across_batches,
                balance_targets,
            )
        candidates = self._candidates(
            batches, coverage_targets, scope, rank_across_batches=rank_across_batches
        )
        allow_single_document = (
            bool(candidates)
            and len({candidate.parent.document_id for candidate in candidates}) == 1
        )

        items: list[EvidenceItem] = []
        rendered: list[str] = []
        document_counts: Counter[str] = Counter()
        selected_keys: set[tuple[str, str]] = set()
        covered_target_ids: set[str] = set()

        for target in coverage_targets:
            if target.target_id in covered_target_ids:
                continue
            for candidate in candidates:
                key = (candidate.parent.parent_id, candidate.parent.document_version_id)
                if target.target_id not in candidate.target_ids or key in selected_keys:
                    continue
                target_capacity = capacity
                if balance_targets:
                    remaining_targets = len(
                        {t.target_id for t in coverage_targets} - covered_target_ids
                    )
                    used = _estimate_tokens("\n".join(rendered))
                    target_capacity = used + (capacity - used) // max(
                        1, remaining_targets
                    )
                packed = self._pack_candidate(candidate, rendered, target_capacity)
                if (
                    packed is None
                    or target.target_id not in packed[0].covered_target_ids
                    or not _within_document_limit(
                        candidate,
                        document_counts,
                        allow_single_document=allow_single_document,
                    )
                ):
                    continue
                item, rendered_envelope = packed
                items.append(item)
                rendered.append(rendered_envelope)
                selected_keys.add(key)
                document_counts[item.document_id] += 1
                covered_target_ids.update(item.covered_target_ids)
                break

        for candidate in candidates:
            key = (candidate.parent.parent_id, candidate.parent.document_version_id)
            if key in selected_keys:
                continue
            if not _within_document_limit(
                candidate,
                document_counts,
                allow_single_document=allow_single_document,
            ):
                continue
            packed = self._pack_candidate(candidate, rendered, capacity)
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
                heading_path=item.heading_path,
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

    def _build_contextual(
        self, batches, targets, scope, snapshot, capacity, rank, balance
    ):
        from agentic_rag.retrieval.context_evidence import (
            EvidenceRequirementResolver,
            ContextSourceHydrator,
        )
        import json

        requirements = {}
        sources = {}
        for batch in batches:
            ContextSourceHydrator._check_plan(batch.query_plan, scope, snapshot)
            for requirement in batch.supporting_context.requirements:
                prior = requirements.get(requirement.id)
                if prior is not None and prior != requirement:
                    raise ValueError("conflicting evidence requirements")
                requirements[requirement.id] = requirement
            for source in batch.supporting_context.sources:
                old = sources.get(source.evidence_id)
                if old is not None:
                    if old.model_dump(
                        exclude={
                            "supports_claim_ids",
                            "source_child_ids",
                            "context_refs",
                        }
                    ) != source.model_dump(
                        exclude={
                            "supports_claim_ids",
                            "source_child_ids",
                            "context_refs",
                        }
                    ):
                        raise ValueError("conflicting supporting source")
                    source = source.model_copy(
                        update={
                            name: tuple(
                                dict.fromkeys(
                                    (*getattr(old, name), *getattr(source, name))
                                )
                            )
                            for name in (
                                "supports_claim_ids",
                                "source_child_ids",
                                "context_refs",
                            )
                        }
                    )
                sources[source.evidence_id] = source
            hydrated = {r.child_id for r in batch.supporting_context.requirements}
            for parent in batch.parents:
                for hit in parent.child_hits:
                    plan = batch.query_plan
                    if plan is not None and (
                        plan.effective_filter.match_none
                        or (
                            plan.effective_filter.document_ids
                            and hit.document_id
                            not in plan.effective_filter.document_ids
                        )
                    ):
                        raise ValueError("source outside frozen query plan")
                    if (
                        hit.context_ref is not None
                        and hit.child_id
                        not in batch.supporting_context.verified_child_ids
                    ):
                        raise ValueError("context source hydration required")
                    if hit.child_id in hydrated:
                        continue
                    for req in (
                        EvidenceRequirementResolver()
                        .resolve(batch.query_plan, hit, None)
                        .entries
                    ):
                        requirements[req.id] = req
        items, rendered, selected_sources = [], [], {}
        included_excerpt_ids = set()
        gaps = set()
        support_tokens = 0
        counts = Counter()
        target_covered = set()
        candidates = self._candidates(batches, targets, scope, rank_across_batches=rank)
        single = len({c.parent.document_id for c in candidates}) == 1
        for candidate in candidates:
            if not _within_document_limit(
                candidate, counts, allow_single_document=single
            ):
                continue
            children = {h.child_id for h in candidate.parent.child_hits}
            required = [r for r in requirements.values() if r.child_id in children]
            if any(r.status != "resolved" for r in required):
                gaps.update(r.id for r in required if r.status != "resolved")
                continue
            refs = {ref for r in required for ref in r.source_refs}
            needed = [
                s
                for s in sources.values()
                if s.source_ref in refs and children.intersection(s.source_child_ids)
            ]
            if refs != {s.source_ref for s in needed} or any(
                len(
                    {
                        ref
                        for r in required
                        if r.child_id == cid
                        for ref in r.source_refs
                    }
                )
                > 4
                for cid in children
            ):
                gaps.update(r.id for r in required)
                continue
            item = self._item(candidate)
            parent_content = candidate.parent.content
            same, outside = [], []
            for source in needed:
                ref = source.source_ref
                covered = False
                if candidate.parent.ast_locator and source.content in parent_content:
                    locator = json.loads(candidate.parent.ast_locator)
                    covered = any(
                        span.get("block_id") == ref.block_id
                        and span.get("canonical_path") == ref.canonical_path
                        and span.get("char_from", -1)
                        <= ref.char_from
                        < ref.char_to
                        <= span.get("char_to", -1)
                        for span in locator.get("spans", [])
                    )
                (same if covered else outside).append(source)
            new_sources = [
                source
                for source in outside
                if source.evidence_id not in included_excerpt_ids
            ]
            extra = [
                EvidenceItem(
                    schema_version=2,
                    source_kind="canonical_excerpt",
                    parent_id=None,
                    evidence_id=source.evidence_id,
                    document_id=source.source_ref.document_id,
                    document_version_id=source.source_ref.document_version_id,
                    content=source.content,
                    content_hash=source.source_ref.quote_sha256,
                    source_ref=source.source_ref,
                    ast_locator=source.source_ref.canonical_path,
                    build_fingerprint=source.build_fingerprint,
                    context_refs=source.context_refs,
                    supports_claim_ids=source.supports_claim_ids,
                    source_child_ids=source.source_child_ids,
                    covered_target_ids=(),
                )
                for source in new_sources
            ]
            extra_rendered = [self._render_item(i) for i in extra]
            extra_cost = sum(len(v) + 1 for v in extra_rendered)
            if support_tokens + extra_cost > capacity // 4:
                gaps.update(r.id for r in required)
                continue
            envelope = DataEnvelope(
                source_label=f"document:{item.document_id}",
                evidence_id=item.evidence_id,
                content=parent_content,
                heading_path=item.heading_path,
            )
            group_capacity = capacity
            if balance and set(candidate.target_ids) - target_covered:
                remaining = {target.target_id for target in targets} - target_covered
                used = len("\n".join(rendered))
                group_capacity = used + (capacity - used) // max(1, len(remaining))
            from agentic_rag.retrieval.context_evidence import parent_excerpt_range

            try:
                selected_ranges = (
                    [
                        parent_excerpt_range(
                            candidate.parent.ast_locator, h.ast_locator, parent_content
                        )
                        for h in candidate.parent.child_hits
                    ]
                    if candidate.parent.ast_locator
                    else [(0, len(parent_content), None, None)]
                )
            except ValueError:
                gaps.update(r.id for r in required)
                continue
            anchor_offset = selected_ranges[0][0]
            fitted = self._fit(
                envelope,
                (*rendered, *extra_rendered),
                group_capacity,
                candidate.parent.child_hits[0].content,
                matching_child_offset=anchor_offset,
            )
            crop_range = (
                _crop_bounds(parent_content, len(fitted[0]), anchor_offset)
                if fitted
                else (0, 0)
            )
            if (
                fitted is None
                or any(
                    not crop_range[0] <= start < end <= crop_range[1]
                    for start, end, _, _ in selected_ranges
                )
                or any(h.content not in fitted[0] for h in candidate.parent.child_hits)
                or any(s.content not in fitted[0] for s in same)
            ):
                gaps.update(r.id for r in required)
                continue
            item = EvidenceItem(
                **{
                    **item.model_dump(),
                    "schema_version": 2,
                    "content": fitted[0],
                    "content_hash": text_sha256(fitted[0]),
                    "parent_excerpt_length": len(fitted[0]),
                    "source_child_ids": tuple(sorted(children)),
                    "context_refs": tuple(
                        dict.fromkeys(
                            h.context_ref
                            for h in candidate.parent.child_hits
                            if h.context_ref
                        )
                    ),
                    "build_fingerprint": next(
                        (
                            h.build_fingerprint
                            for h in candidate.parent.child_hits
                            if h.build_fingerprint
                        ),
                        None,
                    ),
                    "supports_claim_ids": tuple(
                        dict.fromkeys(
                            cid
                            for source in same
                            for requirement in required
                            if source.source_ref in requirement.source_refs
                            for cid in requirement.claim_ids
                        )
                    ),
                    "support_source_refs": tuple(source.source_ref for source in same),
                }
            )
            items.extend((item, *extra))
            rendered.extend((self._render_item(item), *extra_rendered))
            selected_sources.update((s.evidence_id, s) for s in needed)
            included_excerpt_ids.update(source.evidence_id for source in new_sources)
            support_tokens += extra_cost
            counts[item.document_id] += 1
            target_covered.update(item.covered_target_ids)
        context = SupportingContext(
            sources=tuple(selected_sources.values()),
            requirements=tuple(requirements.values()),
            verified_child_ids=tuple(
                dict.fromkeys(
                    cid
                    for b in batches
                    for cid in b.supporting_context.verified_child_ids
                )
            ),
        )
        text = "\n".join(rendered)
        return PackedEvidence(
            items=tuple(items),
            manifest={
                i.evidence_id: EvidenceManifestEntry(**evidence_projection(i))
                for i in items
            },
            rendered_context=text,
            token_count=len(text),
            index_generation=snapshot.index_generation,
            supporting_context=context,
            evidence_requirements=tuple(requirements.values()),
            scope_evidence_missing=tuple(sorted(gaps)),
        )

    @staticmethod
    def _render_item(item):
        return DataEnvelope(
            source_label=f"document:{item.document_id}",
            evidence_id=item.evidence_id,
            content=item.content,
            heading_path=item.heading_path,
        ).render()

    def _pack_candidate(
        self,
        candidate: _Candidate,
        rendered: Sequence[str],
        capacity: int,
    ) -> tuple[EvidenceItem, str] | None:
        item = self._item(candidate)
        envelope = DataEnvelope(
            source_label=f"document:{item.document_id}",
            evidence_id=item.evidence_id,
            content=item.content,
            heading_path=item.heading_path,
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
        # Coverage is provenance attached to the retrieval batch, not a text
        # match inferred from untrusted document content.  Cropping may remove
        # the human-readable target description while retaining usable source
        # content, so preserve the server-owned IDs in that case.
        retained_target_ids = (
            item.covered_target_ids if _has_usable_content(fitted_content) else ()
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
        *,
        rank_across_batches: bool = True,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        known_target_ids = {target.target_id for target in targets}
        for batch_position, batch in enumerate(batches):
            for parent_position, parent in enumerate(batch.parents):
                child_hits = _valid_child_hits(parent, scope)
                if not child_hits:
                    continue
                locator = child_hits[0].ast_locator
                candidate_target_ids = tuple(
                    target_id
                    for target_id in dict.fromkeys(batch.target_ids)
                    if target_id in known_target_ids
                )
                candidates.append(
                    _Candidate(
                        parent=parent,
                        locator=locator,
                        target_ids=candidate_target_ids,
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
                -candidate.parent.ranking_score
                if rank_across_batches
                else candidate.parent_position,
                candidate.batch_position,
                candidate.parent_position,
                candidate.parent.parent_id,
            ),
        )
        unique: list[_Candidate] = []
        positions: dict[tuple[str, str], int] = {}
        for candidate in ranked:
            key = (candidate.parent.parent_id, candidate.parent.document_version_id)
            if key not in positions:
                positions[key] = len(unique)
                unique.append(candidate)
            else:
                position = positions[key]
                original = unique[position]
                if (
                    original.parent.document_id,
                    original.parent.content,
                    original.parent.heading_path,
                ) != (
                    candidate.parent.document_id,
                    candidate.parent.content,
                    candidate.parent.heading_path,
                ):
                    raise ValueError("conflicting parent source identity")
                children = {hit.child_id: hit for hit in original.parent.child_hits}
                for hit in candidate.parent.child_hits:
                    if (
                        hit.child_id in children
                        and children[hit.child_id].search_identity
                        != hit.search_identity
                    ):
                        raise ValueError("conflicting candidate search identity")
                    children[hit.child_id] = hit
                unique[position] = replace(
                    original,
                    parent=original.parent.model_copy(
                        update={"child_hits": tuple(children.values())}
                    ),
                    target_ids=tuple(
                        dict.fromkeys((*original.target_ids, *candidate.target_ids))
                    ),
                )
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
            heading_path=parent.heading_path,
        )

    @staticmethod
    def _fit(
        envelope: DataEnvelope,
        rendered: Sequence[str],
        capacity: int,
        matching_child_content: str,
        *,
        matching_child_offset: int | None = None,
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
                envelope.content,
                matching_child_content,
                length,
                anchor_offset=matching_child_offset,
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
    candidate: _Candidate,
    document_counts: Counter[str],
    *,
    allow_single_document: bool | None = None,
) -> bool:
    """Apply diversity only when the request is not a direct document lookup."""
    single_document = (
        candidate.is_direct_single_document
        if allow_single_document is None
        else allow_single_document
    )
    return document_counts[candidate.parent.document_id] < (
        MAX_ITEMS_SINGLE_DOCUMENT if single_document else MAX_ITEMS_PER_DOCUMENT
    )


def _has_usable_content(content: str) -> bool:
    """Return whether a fitted fragment still contains source characters."""
    return any(character not in {"…", " ", "\n", "\r", "\t"} for character in content)


def _crop_bounds(content: str, limit: int, position: int) -> tuple[int, int]:
    if len(content) <= limit:
        return 0, len(content)
    start = max(0, position - limit // 2)
    end = min(len(content), start + limit)
    start = max(0, end - limit)
    available = max(0, limit - int(start > 0) - int(end < len(content)))
    if end < len(content):
        end = min(len(content), start + available)
    else:
        start = max(start, end - available)
    return start, end


def _crop_around_child(
    content: str,
    child_content: str,
    limit: int = 1_200,
    *,
    anchor_offset: int | None = None,
) -> str:
    """Contextual crops use verified coordinates; legacy callers retain text lookup."""
    if limit < 1:
        return ""
    if len(content) <= limit:
        return content
    position = anchor_offset
    if position is None:
        position = content.find(child_content) if child_content else len(content) // 2
        if position < 0:
            position = len(content) // 2
    start, end = _crop_bounds(content, limit, position)
    return (
        f"{'…' if start else ''}{content[start:end]}{'…' if end < len(content) else ''}"
    )


def _estimate_tokens(text: str) -> int:
    """Return the documented conservative codepoint token upper bound."""
    return len(text)
