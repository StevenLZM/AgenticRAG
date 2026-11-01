"""Authorized original source hydration shared by retrieval and evidence workers."""

from __future__ import annotations

from typing import Protocol, Sequence

from sqlalchemy import select

from agentic_rag.domain.models import UserScope
from agentic_rag.ingestion.assembler import CanonicalAst
from agentic_rag.ingestion.chunker import AstLocator
from agentic_rag.ingestion.contextualization.sources import SourceCatalog
from agentic_rag.ingestion.manifest import VersionManifest, validate_context_artifacts
from agentic_rag.models.contextual import (
    CanonicalExcerpt,
    ContextArtifact,
    ContextArtifactRef,
    EvidenceRequirement,
    EvidenceRequirements,
    SupportingContext,
    text_sha256,
)
from agentic_rag.persistence.artifacts import ArtifactStore
from agentic_rag.retrieval.models import ChildHit, QueryPlan
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class ContextSourceViolation(ValueError):
    """An original source or its durable authorization could not be verified."""


class ContextSourceAuthority(Protocol):
    async def manifest(
        self,
        document_id: str,
        version_id: str,
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        *,
        build_fingerprint: str | None = None,
    ) -> ContextArtifactRef: ...


class SqlContextSourceAuthority:
    """Current single-build authority; never accepts an ES supplied manifest."""

    def __init__(self, session_factory, artifacts: ArtifactStore):
        self.session_factory, self.artifacts = session_factory, artifacts

    async def manifest(
        self, document_id, version_id, scope, snapshot, *, build_fingerprint=None
    ):
        from agentic_rag.persistence.repositories import documents, document_versions

        async with self.session_factory() as db:
            row = (
                (
                    await db.execute(
                        select(
                            document_versions.c.manifest_path,
                            document_versions.c.manifest_hash,
                        )
                        .select_from(
                            documents.join(
                                document_versions,
                                documents.c.active_version_id == document_versions.c.id,
                            )
                        )
                        .where(
                            documents.c.id == document_id,
                            documents.c.user_id == scope.user_id,
                            documents.c.status == "active",
                            documents.c.deletion_status.is_(None),
                            document_versions.c.id == version_id,
                            document_versions.c.document_id == document_id,
                            document_versions.c.status == "active",
                            document_versions.c.index_generation
                            == snapshot.index_generation,
                        )
                    )
                )
                .mappings()
                .one_or_none()
            )
        if row is None or not row["manifest_path"] or not row["manifest_hash"]:
            raise ContextSourceViolation(
                "source missing, revoked, stale or outside scope"
            )
        ref = self.artifacts.describe(row["manifest_path"])
        if ref.sha256 != row["manifest_hash"]:
            raise ContextSourceViolation("durable manifest hash differs")
        if build_fingerprint is not None:
            manifest = VersionManifest.model_validate(self.artifacts.read_json(ref))
            if manifest.build_fingerprint != build_fingerprint:
                raise ContextSourceViolation(
                    "source build is not the active manifest member"
                )
        return ContextArtifactRef(**ref.__dict__)


class EvidenceRequirementResolver:
    def resolve(
        self,
        query_plan: QueryPlan | None,
        hit: ChildHit,
        artifact: ContextArtifact | None,
    ) -> EvidenceRequirements:
        entries = []
        accepted = (
            tuple(c for c in artifact.claims if c.status == "accepted")
            if artifact
            else ()
        )
        plan_hash = query_plan.plan_hash if query_plan else None
        namespace = text_sha256(
            "|".join(
                (
                    hit.document_version_id,
                    artifact.source_child_id if artifact else hit.child_id,
                    artifact.build_fingerprint if artifact else "legacy",
                    hit.context_ref.sha256 if hit.context_ref else "none",
                )
            )
        )
        if query_plan is not None and query_plan.resolution_status in {
            "unresolved",
            "ambiguous",
        }:
            entries.append(
                EvidenceRequirement(
                    id=text_sha256(f"{namespace}|{plan_hash}|query_scope"),
                    child_id=hit.child_id,
                    query_plan_hash=plan_hash,
                    field="query_scope",
                    value="",
                    status="missing"
                    if query_plan.resolution_status == "unresolved"
                    else "conflicting",
                    reason=f"query_scope_{query_plan.resolution_status}",
                )
            )
        for claim in accepted:
            claim_id = f"{namespace}:{claim.id}"
            entries.append(
                EvidenceRequirement(
                    id=text_sha256(f"{claim_id}|{plan_hash}"),
                    child_id=hit.child_id,
                    query_plan_hash=plan_hash,
                    field=claim.field,
                    value=claim.value,
                    claim_ids=(claim_id,),
                    source_refs=claim.source_refs,
                    exclusion_conditions=claim.exclusion_conditions,
                    status="resolved",
                )
            )
        for target in query_plan.effective_business_scope if query_plan else ():
            field = "business_period" if target.field == "period" else target.field
            claims = tuple(c for c in accepted if c.field == field)
            matching = tuple(c for c in claims if c.value == target.value)
            uncertain = tuple(
                c
                for c in (artifact.claims if artifact else ())
                if c.field == field and c.status != "accepted"
            )
            contrary = tuple(
                c for c in uncertain if c.verification_status == "contradicted"
            )
            if (
                matching
                and not uncertain
                and len({c.value for c in claims}) == 1
                and target.resolution_status in {"resolved", "defaulted"}
            ):
                continue
            # Literal original body is an available input to the semantic grader,
            # never an accepted generated claim or proof of applicability.
            available = (
                not claims
                and not uncertain
                and target.value in hit.content
                and target.resolution_status in {"resolved", "defaulted"}
            )
            status = (
                "conflicting"
                if claims or contrary
                else "resolved"
                if available
                else "missing"
            )
            entries.append(
                EvidenceRequirement(
                    id=text_sha256(f"{namespace}|{plan_hash}|{field}|{target.value}"),
                    child_id=hit.child_id,
                    query_plan_hash=plan_hash,
                    field=field,
                    value=target.value,
                    status=status,
                    reason=(
                        "original_body_requires_semantic_review"
                        if available
                        else "; ".join(
                            c.rejection_reason or c.verification_status or c.status
                            for c in uncertain
                        )
                        if uncertain
                        else "scope_evidence_missing"
                    ),
                )
            )
        return EvidenceRequirements(entries=tuple(entries))


def parent_excerpt_range(parent_locator: str, child_locator: str, content: str):
    """Resolve a Child's actual Parent offsets and pages from trusted Parent spans.

    Opaque legacy locators can designate only the exact whole Parent. They do
    not grant an invented Child offset or a first textual occurrence.
    """
    from agentic_rag.query.audit import _ast_locator_is_authorized

    if not _ast_locator_is_authorized(parent_locator, child_locator):
        raise ContextSourceViolation("Parent excerpt locator is outside source")
    try:
        parent = AstLocator.model_validate_json(parent_locator)
        child = AstLocator.model_validate_json(child_locator)
    except ValueError:
        if parent_locator == child_locator:
            return 0, len(content), None, None
        raise ContextSourceViolation("Parent excerpt has no verified coordinates")
    for span in child.spans:
        matches = [
            p
            for p in parent.spans
            if p.block_id == span.block_id
            and p.canonical_path == span.canonical_path
            and p.char_from <= span.char_from < span.char_to <= p.char_to
            and p.parent_char_from
            <= span.parent_char_from
            < span.parent_char_to
            <= p.parent_char_to
        ]
        if not matches or not any(
            (p.page_from, p.page_to) == (span.page_from, span.page_to) for p in matches
        ):
            raise ContextSourceViolation(
                "Parent excerpt pages differ from original spans"
            )
        # Ordinary spans have an exact affine map between canonical and Parent
        # offsets. Table header repetition has its own chunker mapping; current
        # Parent bounds and original page metadata still constrain those rows.
        if not any(
            p.table_header_row_to
            or (
                span.char_from - p.char_from
                == span.parent_char_from - p.parent_char_from
                and span.char_to - p.char_from
                == span.parent_char_to - p.parent_char_from
            )
            for p in matches
        ):
            raise ContextSourceViolation("Parent excerpt coordinates disagree")
    start = child.parent_char_from - parent.parent_char_from
    end = child.parent_char_to - parent.parent_char_from
    if not 0 <= start < end <= len(content):
        raise ContextSourceViolation("Parent excerpt range is outside original bytes")
    return (
        start,
        end,
        min(span.page_from for span in child.spans),
        max(span.page_to for span in child.spans),
    )


def reconstruct_parent_excerpt(entry, parent):
    """Return authenticated crop bytes; neither the frozen hash nor length is authority."""
    from agentic_rag.query.evidence_builder import _crop_around_child, _crop_bounds

    locator = (
        parent.ast_locator.model_dump_json()
        if isinstance(parent.ast_locator, AstLocator)
        else parent.ast_locator
    )
    start, end, page_from, page_to = parent_excerpt_range(
        locator, entry.ast_locator, parent.content
    )
    if (
        entry.heading_path != tuple(parent.heading_path)
        or entry.parent_excerpt_length is None
    ):
        raise ContextSourceViolation(
            "Parent excerpt heading/length differs from original"
        )
    content = _crop_around_child(
        parent.content, "", entry.parent_excerpt_length, anchor_offset=start
    )
    crop_from, crop_to = _crop_bounds(
        parent.content, entry.parent_excerpt_length, start
    )
    if (
        len(content) != entry.parent_excerpt_length
        or text_sha256(content) != entry.content_hash
        or not crop_from <= start < end <= crop_to
    ):
        raise ContextSourceViolation(
            "Parent excerpt differs from authoritative original content"
        )
    return dict(
        verified_content=content,
        verified_heading_path=tuple(parent.heading_path),
        verified_page_from=page_from,
        verified_page_to=page_to,
        content_hash=entry.content_hash,
        parent_excerpt_length=entry.parent_excerpt_length,
    )


class ContextSourceHydrator:
    def __init__(
        self,
        *,
        artifacts: ArtifactStore,
        authority: ContextSourceAuthority,
        resolver: EvidenceRequirementResolver | None = None,
    ):
        self.artifacts, self.authority = artifacts, authority
        self.resolver = resolver or EvidenceRequirementResolver()

    async def _load(
        self, document_id, version_id, scope, snapshot, build_fingerprint=None
    ):
        ref = await self.authority.manifest(
            document_id,
            version_id,
            scope,
            snapshot,
            build_fingerprint=build_fingerprint,
        )
        manifest = VersionManifest.model_validate(self.artifacts.read_json(ref))
        if (
            manifest.schema_version != 2
            or manifest.index_generation != snapshot.index_generation
            or (
                build_fingerprint is not None
                and manifest.build_fingerprint != build_fingerprint
            )
        ):
            raise ContextSourceViolation("source manifest generation/schema mismatch")
        parents = validate_context_artifacts(
            manifest, self.artifacts, (scope.user_id, document_id, version_id)
        )
        root = self.artifacts.read_json(manifest.context_artifact_ref)
        canonical_ref = ContextArtifactRef.model_validate(root["canonical_ref"])
        ast = CanonicalAst.model_validate(self.artifacts.read_json(canonical_ref))
        return manifest, parents, SourceCatalog.from_ast(ast, canonical_ref.sha256)

    async def hydrate(
        self,
        hits: Sequence[ChildHit],
        scope: UserScope,
        snapshot: RuntimeConfigSnapshot,
        query_plan: QueryPlan | None,
    ) -> SupportingContext:
        self._check_plan(query_plan, scope, snapshot)
        sources: dict[str, CanonicalExcerpt] = {}
        requirements = []
        loaded = {}
        for hit in hits:
            if hit.user_id != scope.user_id:
                raise ContextSourceViolation("source outside scope")
            if query_plan is not None and (
                query_plan.effective_filter.match_none
                or (
                    query_plan.effective_filter.document_ids
                    and hit.document_id not in query_plan.effective_filter.document_ids
                )
            ):
                raise ContextSourceViolation("source outside frozen query plan")
            if hit.context_ref is None:
                requirements.extend(
                    self.resolver.resolve(query_plan, hit, None).entries
                )
                continue
            key = hit.document_id, hit.document_version_id, hit.build_fingerprint
            if key not in loaded:
                loaded[key] = await self._load(
                    hit.document_id,
                    hit.document_version_id,
                    scope,
                    snapshot,
                    hit.build_fingerprint,
                )
            manifest, parents, catalog = loaded[key]
            child = next(
                (c for p in parents for c in p.children if c.id == hit.child_id), None
            )
            if child is None:
                raise ContextSourceViolation(
                    "child is not a member of authorized build"
                )
            rep = child.search_representation
            expected = dict(parent_id=child.parent_id, content=child.content)
            if AstLocator.model_validate_json(hit.ast_locator) != child.ast_locator:
                raise ContextSourceViolation("candidate locator differs from build")
            expected.update(
                {
                    name: getattr(rep, name)
                    for name in (
                        "context_ref",
                        "content_hash",
                        "source_child_id",
                        "source_content_hash",
                        "build_fingerprint",
                        "contextualized_content",
                        "context_claim_ids",
                        "context_prefix",
                        "enrichment_status",
                        "representation_version",
                    )
                }
            )
            if any(getattr(hit, name) != value for name, value in expected.items()):
                raise ContextSourceViolation(
                    "candidate differs from authorized build member"
                )
            artifact = ContextArtifact.model_validate(
                self.artifacts.read_json(hit.context_ref)
            )
            for claim in artifact.claims:
                if claim.status == "accepted" and not catalog.covers_child(
                    claim.scope_region.source_refs, child
                ):
                    raise ContextSourceViolation(
                        "accepted claim does not cover selected child"
                    )
            resolved = self.resolver.resolve(query_plan, hit, artifact)
            requirements.extend(resolved.entries)
            for requirement in resolved.entries:
                for source in requirement.source_refs:
                    content = catalog.verify_source(source)
                    eid = text_sha256(
                        source.model_dump_json() + manifest.build_fingerprint
                    )[:32]
                    old = sources.get(eid)
                    sources[eid] = CanonicalExcerpt(
                        evidence_id=eid,
                        content=content,
                        source_ref=source,
                        build_fingerprint=manifest.build_fingerprint,
                        context_refs=tuple(
                            dict.fromkeys(
                                (*(old.context_refs if old else ()), hit.context_ref)
                            )
                        ),
                        supports_claim_ids=tuple(
                            dict.fromkeys(
                                (
                                    *(old.supports_claim_ids if old else ()),
                                    *requirement.claim_ids,
                                )
                            )
                        ),
                        source_child_ids=tuple(
                            dict.fromkeys(
                                (*(old.source_child_ids if old else ()), hit.child_id)
                            )
                        ),
                    )
        return SupportingContext(
            sources=tuple(sources.values()),
            requirements=tuple(requirements),
            verified_child_ids=tuple(h.child_id for h in hits if h.context_ref),
        )

    @staticmethod
    def _check_plan(plan, scope, snapshot):
        if plan is not None and (
            plan.snapshot_id != snapshot.snapshot_id
            or plan.authorization_filter.user_id != scope.user_id
            or plan.authorization_filter.index_generation != snapshot.index_generation
        ):
            raise ContextSourceViolation("query plan scope/snapshot mismatch")

    async def authorize_parent(self, entry, scope, snapshot):
        manifest, parents, catalog = await self._load(
            entry.document_id,
            entry.document_version_id,
            scope,
            snapshot,
            entry.build_fingerprint,
        )
        parent = next((p for p in parents if p.id == entry.parent_id), None)
        if parent is None or entry.build_fingerprint != manifest.build_fingerprint:
            raise ContextSourceViolation("parent outside context build")
        members = {c.id: c.search_representation.context_ref for c in parent.children}
        if (
            not entry.source_child_ids
            or not set(entry.source_child_ids) <= set(members)
            or set(entry.context_refs)
            != {members[cid] for cid in entry.source_child_ids}
        ):
            raise ContextSourceViolation("parent context members differ")
        locator = AstLocator.model_validate_json(entry.ast_locator)
        anchor = next(
            (
                child
                for child in parent.children
                if child.id in entry.source_child_ids and child.ast_locator == locator
            ),
            None,
        )
        if anchor is None:
            raise ContextSourceViolation(
                "Parent locator differs from authoritative Child"
            )
        verified = reconstruct_parent_excerpt(entry, parent)
        content = verified["verified_content"]
        from agentic_rag.query.evidence_builder import _crop_bounds

        anchor_offset = (
            anchor.ast_locator.parent_char_from - parent.ast_locator.parent_char_from
        )
        crop_from, crop_to = _crop_bounds(
            parent.content, entry.parent_excerpt_length, anchor_offset
        )
        for child in parent.children:
            if child.id in entry.source_child_ids and not (
                crop_from
                <= child.ast_locator.parent_char_from
                - parent.ast_locator.parent_char_from
                < child.ast_locator.parent_char_to - parent.ast_locator.parent_char_from
                <= crop_to
            ):
                raise ContextSourceViolation(
                    "Parent crop drops selected Child coordinates"
                )
        for source_ref in entry.support_source_refs:
            text = catalog.verify_source(source_ref)
            if text not in content or not any(
                span.block_id == source_ref.block_id
                and span.canonical_path == source_ref.canonical_path
                and span.char_from
                <= source_ref.char_from
                < source_ref.char_to
                <= span.char_to
                and crop_from
                <= span.parent_char_from
                - parent.ast_locator.parent_char_from
                + source_ref.char_from
                - span.char_from
                < span.parent_char_from
                - parent.ast_locator.parent_char_from
                + source_ref.char_to
                - span.char_from
                <= crop_to
                for span in parent.ast_locator.spans
            ):
                raise ContextSourceViolation(
                    "parent does not contain exact supporting source"
                )
        accepted_relations = set()
        for ref in entry.context_refs:
            artifact = ContextArtifact.model_validate(self.artifacts.read_json(ref))
            namespace = text_sha256(
                "|".join(
                    (
                        entry.document_version_id,
                        artifact.source_child_id,
                        artifact.build_fingerprint,
                        ref.sha256,
                    )
                )
            )
            for claim in artifact.claims:
                if claim.status == "accepted" and set(claim.source_refs).intersection(
                    entry.support_source_refs
                ):
                    accepted_relations.add(f"{namespace}:{claim.id}")
        if not set(entry.supports_claim_ids) <= accepted_relations:
            raise ContextSourceViolation(
                "parent support relation is outside accepted artifact"
            )
        return dict(
            **verified,
            context_refs=entry.context_refs,
            build_fingerprint=entry.build_fingerprint,
            source_child_ids=entry.source_child_ids,
            support_source_refs=entry.support_source_refs,
            supports_claim_ids=entry.supports_claim_ids,
        )

    async def authorize(self, entry, scope, snapshot):
        """Reload durable authority and exact accepted-source membership at audit."""
        manifest, parents, catalog = await self._load(
            entry.document_id,
            entry.document_version_id,
            scope,
            snapshot,
            entry.build_fingerprint,
        )
        if (
            entry.build_fingerprint != manifest.build_fingerprint
            or entry.source_ref is None
            or entry.heading_path != ()
        ):
            raise ContextSourceViolation("excerpt build/source mismatch")
        members = {
            c.search_representation.context_ref: c for p in parents for c in p.children
        }
        claims, child_ids = set(), set()
        for ref in entry.context_refs:
            if ref not in members:
                raise ContextSourceViolation(
                    "excerpt artifact outside authorized build"
                )
            child = members[ref]
            artifact = ContextArtifact.model_validate(self.artifacts.read_json(ref))
            # Recreate stable relation namespace using the persisted identity.
            namespace = text_sha256(
                "|".join(
                    (
                        entry.document_version_id,
                        artifact.source_child_id,
                        artifact.build_fingerprint,
                        ref.sha256,
                    )
                )
            )
            matched = False
            for claim in artifact.claims:
                if claim.status == "accepted" and entry.source_ref in claim.source_refs:
                    if not catalog.covers_child(claim.scope_region.source_refs, child):
                        raise ContextSourceViolation(
                            "excerpt claim scope does not cover child"
                        )
                    claims.add(f"{namespace}:{claim.id}")
                    matched = True
            if matched:
                child_ids.add(child.id)
        content = catalog.verify_source(entry.source_ref)
        if (
            not entry.context_refs
            or not entry.supports_claim_ids
            or not set(entry.supports_claim_ids) <= claims
            or set(entry.source_child_ids) != child_ids
            or entry.content_hash != text_sha256(content)
        ):
            raise ContextSourceViolation("excerpt support membership/hash mismatch")
        return dict(
            user_id=scope.user_id,
            index_generation=snapshot.index_generation,
            is_active=True,
            verified_content=content,
            verified_heading_path=(),
            verified_page_from=entry.source_ref.page_from,
            verified_page_to=entry.source_ref.page_to,
            **entry.model_dump(),
        )
