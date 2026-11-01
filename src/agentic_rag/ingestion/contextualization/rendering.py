"""Render verified whole claims with actual tokenizer budgets and stable IDs."""

from __future__ import annotations

from typing import Callable, Sequence

from agentic_rag.ingestion.chunker import ChildChunk
from agentic_rag.ingestion.contextualization.sources import (
    SourceIntegrityError,
    covers_child_spans,
)
from agentic_rag.models.contextual import (
    CONTEXT_SEPARATOR,
    REPRESENTATION_VERSION,
    ContextArtifactRef,
    ContextClaim,
    SearchRepresentation,
    text_sha256,
)
from agentic_rag.runtime.ids import content_id


BASE_MAX_TOKENS = 320
PREFIX_MAX_TOKENS = 64
FINAL_MAX_TOKENS = 384
_PRIORITY = {
    "company": 0,
    "population": 1,
    "business_period": 2,
    "currency": 3,
    "unit": 4,
    "term": 5,
    "topic": 6,
}


class ContextRenderingError(ValueError):
    """The original body or base representation cannot fit without alteration."""


def render_context(
    child: ChildChunk,
    claims: Sequence[ContextClaim],
    *,
    count_tokens: Callable[[str], int],
    build_fingerprint: str,
    context_ref: ContextArtifactRef | None,
) -> SearchRepresentation:
    # model_copy bypasses Pydantic validation: validate all boundary inputs.
    child = ChildChunk.model_validate(child.model_dump())
    validated = [ContextClaim.model_validate(claim.model_dump()) for claim in claims]
    if len({claim.id for claim in validated}) != len(validated):
        raise ContextRenderingError("duplicate claim IDs")
    identity = (child.user_id, child.document_id, child.document_version_id)
    for claim in validated:
        for ref in (*claim.source_refs, *claim.scope_region.source_refs):
            if (ref.user_id, ref.document_id, ref.document_version_id) != identity:
                raise SourceIntegrityError(
                    "claim source does not belong to target child identity"
                )
    base = child.contextualized_content
    base_tokens = count_tokens(base)
    usable = [
        claim
        for claim in validated
        if claim.status == "accepted"
        and "search" in claim.allowed_uses
        and covers_child_spans(claim.scope_region.source_refs, child)
    ]
    if base_tokens > FINAL_MAX_TOKENS or (usable and base_tokens > BASE_MAX_TOKENS):
        raise ContextRenderingError(
            "base representation exceeds token budget; rebuild child without truncation"
        )
    prefix = ""
    rendered = []
    for claim in sorted(usable, key=lambda item: (_PRIORITY[item.field], item.id)):
        field = f"{claim.field}: {claim.display_value}"
        if claim.exclusion_conditions:
            field += " (exclusions: " + "; ".join(claim.exclusion_conditions) + ")"
        candidate = "\n".join(part for part in (prefix, field) if part)
        combined = candidate + CONTEXT_SEPARATOR + base
        if (
            count_tokens(candidate + CONTEXT_SEPARATOR) <= PREFIX_MAX_TOKENS
            and count_tokens(combined) <= FINAL_MAX_TOKENS
        ):
            prefix = candidate
            rendered.append(claim.id)
    final = (prefix + CONTEXT_SEPARATOR if prefix else "") + base
    digest = text_sha256(final)
    identifier = (
        "chi_"
        + content_id(
            child.document_version_id,
            child.parent_id,
            child.id,
            REPRESENTATION_VERSION,
            build_fingerprint,
            digest,
        )[:32]
    )
    return SearchRepresentation(
        source_child_id=child.id,
        id=identifier,
        content=child.content,
        source_content_hash=text_sha256(child.content),
        base_contextualized_content=base,
        context_prefix=prefix,
        contextualized_content=final,
        content_hash=digest,
        context_claim_ids=tuple(rendered),
        representation_version=REPRESENTATION_VERSION,
        enrichment_status=(
            "structural_only"
            if not rendered
            else "enriched"
            if len(rendered) == len(validated)
            else "partial"
        ),
        token_count=count_tokens(final),
        build_fingerprint=build_fingerprint,
        context_ref=context_ref,
    )
