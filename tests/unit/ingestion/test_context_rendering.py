"""Whole-field context budgeting and immutable representation contracts."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from agentic_rag.models.contextual import (
    ContextArtifact,
    ContextArtifactRef,
    ContextClaim,
    ContextScopeRegion,
    SearchRepresentation,
)
from agentic_rag.ingestion.contextualization.rendering import (
    ContextRenderingError,
    render_context,
)
from tests.unit.ingestion.test_context_sources import _catalog, _child


def _claim(
    *,
    id: str = "company-claim",
    field: str = "company",
    value: str = "星河公司",
    exclusions: tuple[str, ...] = (),
    status: str = "accepted",
    body: str = "policy body",
) -> ContextClaim:
    catalog = _catalog(body)
    source = catalog.resolve_quote(catalog.source_ids[0], body)
    return ContextClaim(
        id=id,
        field=field,
        value=value,
        display_value=value,
        scope_region=ContextScopeRegion(kind="document", source_refs=(source,)),
        source_refs=(source,),
        derivation="llm",
        status=status,
        verification_method="llm_supported",
        verification_status="supported",
        allowed_uses=("search",) if status == "accepted" else (),
        exclusion_conditions=exclusions,
    )


def _render(child=None, claims=(), **kwargs) -> SearchRepresentation:
    return render_context(
        child or _child("policy body"),
        claims,
        count_tokens=lambda text: len(text.split()),
        build_fingerprint="build-1",
        context_ref=None,
        **kwargs,
    )


def test_overflow_drops_whole_claim_preserving_exclusion_and_body() -> None:
    # Truncating the prefix would lose the exception or corrupt the body.
    child = _child(" ".join(["body"] * 320))
    huge = _claim(
        value=" ".join(["company"] * 63),
        exclusions=("不含 外包 人员",),
        body=child.content,
    )
    small = _claim(
        id="audience",
        field="population",
        value="正式 员工",
        exclusions=("不含 外包",),
        body=child.content,
    )
    result = _render(child, (huge, small))
    assert result.content == child.content
    assert result.base_contextualized_content == child.contextualized_content
    assert result.context_prefix == "population: 正式 员工 (exclusions: 不含 外包)"
    assert result.context_claim_ids == ("audience",)
    assert result.token_count == 326
    assert result.token_count <= 384
    assert result.enrichment_status == "partial"


def test_context_changes_hash_and_id_but_artifact_attachment_does_not() -> None:
    first = _render(claims=(_claim(),))
    second = _render(claims=(_claim(value="远航公司"),))
    assert (
        first.content_hash
        == hashlib.sha256(first.contextualized_content.encode()).hexdigest()
    )
    assert first.source_content_hash == hashlib.sha256(b"policy body").hexdigest()
    assert first.content_hash != second.content_hash
    assert first.id != second.id
    ref = ContextArtifactRef(
        uri="artifact://context/abc.json", sha256="a" * 64, size_bytes=12
    )
    attached = first.with_context_ref(ref)
    assert attached.id == first.id
    assert attached.contextualized_content == first.contextualized_content
    assert attached.context_ref == ref
    assert attached.require_persisted() is attached
    with pytest.raises(ValueError, match="context_ref"):
        first.require_persisted()
    with pytest.raises(ValidationError):
        first.with_context_ref({"uri": "broken", "sha256": "invalid", "size_bytes": -1})


def test_unusable_and_partial_scope_claims_are_omitted() -> None:
    catalog = _catalog("policy body")
    partial = _claim().model_copy(
        update={
            "scope_region": ContextScopeRegion(
                kind="spans",
                source_refs=(catalog.resolve_quote(catalog.source_ids[0], "policy "),),
            )
        }
    )
    result = _render(claims=(partial, _claim(id="unknown", status="unknown")))
    assert result.context_prefix == ""
    assert result.context_claim_ids == ()
    assert result.contextualized_content == "policy body"
    assert result.enrichment_status == "structural_only"


@pytest.mark.parametrize(
    "update",
    [
        {"content": "forged"},
        {"content_hash": "0" * 64},
        {"contextualized_content": "forged"},
        {"context_claim_ids": ()},
    ],
)
def test_representation_rejects_text_hash_and_claim_incoherence(update: dict) -> None:
    payload = _render(claims=(_claim(),)).model_dump()
    payload.update(update)
    with pytest.raises(ValidationError):
        SearchRepresentation.model_validate(payload)


def test_llm_claim_cannot_grant_filter_permission_or_skip_verification() -> None:
    payload = _claim().model_dump()
    for update in (
        {"allowed_uses": ("search", "business_filter")},
        {"verification_status": "insufficient"},
        {"source_refs": ()},
        {"verification_method": "explicit_rule"},
    ):
        with pytest.raises(ValidationError):
            ContextClaim.model_validate(payload | update)


def test_llm_verified_claim_cannot_gain_permissions_by_changing_derivation() -> None:
    # Relabeling derivation must not upgrade an LLM-supported result to a filter.
    payload = _claim().model_dump()
    with pytest.raises(ValidationError):
        ContextClaim.model_validate(
            payload
            | {
                "derivation": "trusted_mapping",
                "allowed_uses": ("search", "business_filter"),
            }
        )


def test_artifact_retains_rejected_and_unrendered_accepted_dependencies() -> None:
    accepted = _claim()
    rejected = _claim(id="rejected", status="rejected")
    artifact = ContextArtifact(
        user_id="user-1",
        document_id="document-1",
        document_version_id="version-1",
        canonical_ast_sha256=accepted.source_refs[0].canonical_ast_sha256,
        source_child_id="original-child",
        build_fingerprint="build-1",
        claims=(accepted, rejected),
        rendered_claim_ids=(),
    )
    restored = ContextArtifact.model_validate_json(artifact.model_dump_json())
    assert restored.claims == (accepted, rejected)
    with pytest.raises(ValidationError):
        ContextArtifact.model_validate(
            artifact.model_dump() | {"rendered_claim_ids": ("forged",)}
        )
    with pytest.raises(ValidationError):
        ContextArtifact.model_validate(
            artifact.model_dump() | {"rendered_claim_ids": ("rejected",)}
        )


def test_legacy_child_is_accepted_without_prefix_but_cannot_be_silently_truncated() -> (
    None
):
    child = _child(" ".join(["word"] * 384))
    assert _render(child).content == child.content
    with pytest.raises(ContextRenderingError):
        _render(child, (_claim(body=child.content),))


def test_final_budget_measures_combined_text_with_separators() -> None:
    child = _child("policy body")

    def count(text: str) -> int:
        return 385 if "\n\n" in text else len(text.split())

    result = render_context(
        child, (_claim(),), count_tokens=count, build_fingerprint="b", context_ref=None
    )
    assert result.context_prefix == ""
    assert result.contextualized_content == child.content


def test_prefix_budget_includes_separator_tokens() -> None:
    # A prefix-only measurement misses separator tokens consumed by the template.
    def count(text: str) -> int:
        if text == "policy body":
            return 2
        if text.endswith("\n\n"):
            return 65
        return 64

    result = render_context(
        _child("policy body"),
        (_claim(),),
        count_tokens=count,
        build_fingerprint="b",
        context_ref=None,
    )
    assert result.context_prefix == ""


def test_changed_build_and_parent_produce_distinct_final_ids() -> None:
    child = _child("policy body")
    first = _render(child)
    changed_parent = _render(child.model_copy(update={"parent_id": "parent-2"}))
    changed_build = render_context(
        child,
        (),
        count_tokens=lambda text: len(text.split()),
        build_fingerprint="build-2",
        context_ref=None,
    )
    assert len({first.id, changed_parent.id, changed_build.id}) == 3
    assert {
        first.content_hash,
        changed_parent.content_hash,
        changed_build.content_hash,
    } == {first.content_hash}
