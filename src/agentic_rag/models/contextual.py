"""Lightweight immutable contracts shared by ingestion, retrieval and evidence.

These contracts deliberately depend on neither the chunker nor Docling. Source
coordinates and permissions are server-owned data, never model output schemas.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA256_PATTERN = r"^[0-9a-f]{64}$"
REPRESENTATION_VERSION = "context-v1"
CONTEXT_SEPARATOR = "\n\n"


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_min_length=1)


class ContextArtifactRef(_FrozenModel):
    uri: str = Field(pattern=r"^artifact://[^\s]+$")
    sha256: str = Field(pattern=SHA256_PATTERN)
    size_bytes: int = Field(ge=0)


class CanonicalSourceRef(_FrozenModel):
    """A half-open Python-character range of an immutable canonical block."""

    source_id: str
    user_id: str
    document_id: str
    document_version_id: str
    canonical_ast_sha256: str = Field(pattern=SHA256_PATTERN)
    block_id: str
    canonical_path: str = Field(pattern=r"^#/text_blocks/(0|[1-9][0-9]*)$")
    char_from: int = Field(ge=0)
    char_to: int = Field(gt=0)
    quote_sha256: str = Field(pattern=SHA256_PATTERN)
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)

    @model_validator(mode="after")
    def _ranges(self) -> CanonicalSourceRef:
        if self.char_to <= self.char_from or self.page_to < self.page_from:
            raise ValueError("source range is empty or reversed")
        return self


class ContextScopeRegion(_FrozenModel):
    """Server-resolved applicability spans, distinct from supporting quotations.

    A document/section label alone grants no coverage. ``source_refs`` must
    cover every body span of the child, including all selected table body rows.
    Header quotations remain supporting references and grant no body coverage.
    """

    kind: Literal["document", "section", "table", "rows", "spans"]
    source_refs: tuple[CanonicalSourceRef, ...] = ()


class ContextClaim(_FrozenModel):
    id: str
    field: Literal[
        "company", "population", "business_period", "topic", "term", "currency", "unit"
    ]
    value: str
    display_value: str
    scope_region: ContextScopeRegion
    source_refs: tuple[CanonicalSourceRef, ...] = ()
    derivation: Literal[
        "structural", "explicit_rule", "trusted_mapping", "llm", "human"
    ]
    status: Literal["accepted", "unknown", "ambiguous", "rejected"]
    verification_method: (
        Literal[
            "structural",
            "explicit_rule",
            "trusted_mapping",
            "llm_supported",
            "human_confirmed",
        ]
        | None
    ) = None
    verification_status: Literal["supported", "contradicted", "insufficient"] | None = (
        None
    )
    allowed_uses: tuple[Literal["search", "business_filter"], ...] = ()
    exclusion_conditions: tuple[str, ...] = ()
    rejection_reason: str | None = None

    @model_validator(mode="after")
    def _permissions_and_support(self) -> ContextClaim:
        if self.derivation == "llm" or self.verification_method == "llm_supported":
            if set(self.allowed_uses) - {"search"}:
                raise ValueError("LLM claims are search-only")
        if self.derivation == "llm":
            if self.verification_method not in {None, "llm_supported"}:
                raise ValueError("LLM claims require a distinct semantic verifier")
        if self.status == "accepted":
            if not self.source_refs or not self.scope_region.source_refs:
                raise ValueError("accepted claims require supporting and scope sources")
            if self.verification_method is None:
                raise ValueError("accepted claims require verification")
            if self.verification_status in {"contradicted", "insufficient"}:
                raise ValueError("unsupported claims cannot be accepted")
            if (
                self.derivation == "llm" or self.verification_method == "llm_supported"
            ) and self.verification_status != "supported":
                raise ValueError(
                    "accepted LLM claims require a supported verifier result"
                )
        elif self.allowed_uses:
            raise ValueError("unaccepted claims cannot have allowed uses")
        if (
            self.verification_method == "structural"
            and "business_filter" in self.allowed_uses
        ):
            raise ValueError("structural context cannot authorize business filters")
        return self


class ContextArtifact(_FrozenModel):
    user_id: str
    document_id: str
    document_version_id: str
    canonical_ast_sha256: str = Field(pattern=SHA256_PATTERN)
    source_child_id: str
    build_fingerprint: str
    claims: tuple[ContextClaim, ...] = ()
    rendered_claim_ids: tuple[str, ...] = ()
    representation_version: Literal["context-v1"] = REPRESENTATION_VERSION

    @model_validator(mode="after")
    def _dependencies(self) -> ContextArtifact:
        ids = [claim.id for claim in self.claims]
        if len(ids) != len(set(ids)) or len(self.rendered_claim_ids) != len(
            set(self.rendered_claim_ids)
        ):
            raise ValueError("claim IDs must be unique")
        usable = {
            claim.id
            for claim in self.claims
            if claim.status == "accepted" and "search" in claim.allowed_uses
        }
        if not set(self.rendered_claim_ids) <= usable:
            raise ValueError("rendered claim IDs must name accepted search claims")
        identity = (
            self.user_id,
            self.document_id,
            self.document_version_id,
            self.canonical_ast_sha256,
        )
        for claim in self.claims:
            for ref in (*claim.source_refs, *claim.scope_region.source_refs):
                if (
                    ref.user_id,
                    ref.document_id,
                    ref.document_version_id,
                    ref.canonical_ast_sha256,
                ) != identity:
                    raise ValueError("claim source identity differs from artifact")
        return self


class SearchRepresentation(_FrozenModel):
    """Draft until its immutable context artifact is attached and gated."""

    source_child_id: str
    id: str
    content: str
    source_content_hash: str = Field(pattern=SHA256_PATTERN)
    base_contextualized_content: str
    context_prefix: str = Field(min_length=0)
    contextualized_content: str
    content_hash: str = Field(pattern=SHA256_PATTERN)
    context_claim_ids: tuple[str, ...] = ()
    representation_version: Literal["context-v1"] = REPRESENTATION_VERSION
    enrichment_status: Literal["enriched", "structural_only", "partial"]
    token_count: int = Field(ge=1, le=384)
    build_fingerprint: str
    context_ref: ContextArtifactRef | None = None

    @model_validator(mode="after")
    def _coherent_text(self) -> SearchRepresentation:
        if text_sha256(self.content) != self.source_content_hash:
            raise ValueError("original content hash mismatch")
        if self.content not in self.base_contextualized_content:
            raise ValueError("base representation must retain the original content")
        expected = (
            self.context_prefix + CONTEXT_SEPARATOR if self.context_prefix else ""
        ) + self.base_contextualized_content
        if (
            expected != self.contextualized_content
            or text_sha256(expected) != self.content_hash
        ):
            raise ValueError("contextualized text/hash mismatch")
        if bool(self.context_prefix) != bool(self.context_claim_ids):
            raise ValueError("semantic context requires rendered claim IDs and prefix")
        if len(self.context_claim_ids) != len(set(self.context_claim_ids)):
            raise ValueError("rendered claim IDs must be unique")
        if (self.enrichment_status == "structural_only") != (
            not self.context_claim_ids
        ):
            raise ValueError("enrichment status does not match rendered claims")
        return self

    def with_context_ref(self, ref: ContextArtifactRef) -> SearchRepresentation:
        """Attach persistence metadata without changing text identity."""
        payload = self.model_dump()
        payload["context_ref"] = ref
        return type(self).model_validate(payload)

    def require_persisted(self) -> SearchRepresentation:
        """Publication boundary: every context-v1 record needs its artifact."""
        validated = type(self).model_validate(self.model_dump())
        if validated.context_ref is None:
            raise ValueError("context-v1 publication requires context_ref")
        return self
