"""Deterministic manifest for one fully staged document version."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


from agentic_rag.models.contextual import ContextArtifactRef


class VersionManifest(BaseModel):
    """Counts and immutable versions that gate later publication.

    ``manifest_hash`` is derived from the canonical JSON payload and cannot be
    supplied by callers. The Artifact Store writes exactly ``payload()`` so its
    SHA-256 is the same value.
    """

    model_config = ConfigDict(frozen=True)

    canonical_ast_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_count: int = Field(gt=0)
    child_count: int = Field(gt=0)
    embedding_model: str = Field(min_length=1, max_length=255, pattern=r"^\S+$")
    embedding_dimensions: Literal[1024]
    index_generation: str

    schema_version: Literal[1, 2] = 1
    context_artifact_ref: ContextArtifactRef | None = None
    enriched_chunks_ref: ContextArtifactRef | None = None
    build_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    representation_version: Literal["context-v1"] | None = None
    child_id_hash_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    enrichment_counts: dict[str, int] | None = None

    @model_validator(mode="after")
    def _schema_contract(self):
        values = (
            self.context_artifact_ref,
            self.enriched_chunks_ref,
            self.build_fingerprint,
            self.representation_version,
            self.child_id_hash_digest,
            self.enrichment_counts,
        )
        if self.schema_version == 1 and any(v is not None for v in values):
            raise ValueError("v1 manifest cannot contain context fields")
        if self.schema_version == 2 and any(v is None for v in values):
            raise ValueError("v2 manifest requires all context bindings")
        if self.enrichment_counts is not None and (
            set(self.enrichment_counts) != {"enriched", "partial", "structural_only"}
            or any(v < 0 for v in self.enrichment_counts.values())
            or sum(self.enrichment_counts.values()) != self.child_count
        ):
            raise ValueError("invalid enrichment counts")
        return self

    @field_validator("index_generation")
    @classmethod
    def _index_generation_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("index_generation must not be blank")
        return value

    def payload(self) -> dict[str, Any]:
        """Return the stable payload persisted as the Manifest Artifact."""
        payload = {
            "canonical_ast_sha256": self.canonical_ast_sha256,
            "parent_count": self.parent_count,
            "child_count": self.child_count,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "index_generation": self.index_generation,
        }

        if self.schema_version == 2:
            payload.update(
                self.model_dump(
                    mode="json",
                    include={
                        "schema_version",
                        "context_artifact_ref",
                        "enriched_chunks_ref",
                        "build_fingerprint",
                        "representation_version",
                        "child_id_hash_digest",
                        "enrichment_counts",
                    },
                )
            )
        return payload

    @computed_field  # type: ignore[prop-decorator]
    @property
    def manifest_hash(self) -> str:
        encoded = json.dumps(
            self.payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def child_identity_digest(identities) -> str:
    pairs = sorted(tuple(value) for value in identities)
    if len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("duplicate child identity")
    return hashlib.sha256(
        json.dumps(pairs, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def validate_context_artifacts(manifest, artifacts, identity):
    """Re-read every artifact and original source before embedding/publication."""
    from agentic_rag.ingestion.assembler import CanonicalAst
    from agentic_rag.ingestion.chunker import ParentChunk, resolve_ast_locator
    from agentic_rag.ingestion.contextualization.sources import SourceCatalog
    from agentic_rag.models.contextual import ContextArtifact

    root = artifacts.read_json(manifest.context_artifact_ref)
    enriched = artifacts.read_json(manifest.enriched_chunks_ref)
    if (
        root["build_fingerprint"] != manifest.build_fingerprint
        or enriched["build_fingerprint"] != manifest.build_fingerprint
    ):
        raise ValueError("context build fingerprint mismatch")
    if enriched["context_artifact_ref"] != manifest.context_artifact_ref.model_dump(
        mode="json"
    ):
        raise ValueError("enriched context artifact root mismatch")
    canonical_ref = ContextArtifactRef.model_validate(root["canonical_ref"])
    if canonical_ref.sha256 != manifest.canonical_ast_sha256:
        raise ValueError("context canonical hash mismatch")
    ast = CanonicalAst.model_validate(artifacts.read_json(canonical_ref))
    catalog = SourceCatalog.from_ast(ast, canonical_ref.sha256)
    if (catalog.user_id, catalog.document_id, catalog.document_version_id) != identity:
        raise ValueError("context canonical identity mismatch")
    parents = tuple(ParentChunk.model_validate(p) for p in enriched["parents"])
    pairs = []
    refs = []
    counts = dict(enriched=0, partial=0, structural_only=0)
    for parent in parents:
        if (parent.user_id, parent.document_id, parent.document_version_id) != identity:
            raise ValueError("context parent identity mismatch")
        for child in parent.children:
            if (
                child.user_id,
                child.document_id,
                child.document_version_id,
            ) != identity or child.parent_id != parent.id:
                raise ValueError("context child identity mismatch")
            rep = child.search_representation
            if rep is None:
                raise ValueError("context manifest contains legacy child")
            rep.require_persisted()
            if (
                rep.build_fingerprint != manifest.build_fingerprint
                or resolve_ast_locator(ast, child.ast_locator) != child.content
            ):
                raise ValueError("context original source or build differs")
            artifact = ContextArtifact.model_validate(
                artifacts.read_json(rep.context_ref)
            )
            if (
                (artifact.user_id, artifact.document_id, artifact.document_version_id)
                != identity
                or artifact.source_child_id != rep.source_child_id
                or artifact.build_fingerprint != manifest.build_fingerprint
                or artifact.canonical_ast_sha256 != manifest.canonical_ast_sha256
            ):
                raise ValueError("context claim artifact identity mismatch")
            if artifact.rendered_claim_ids != rep.context_claim_ids:
                raise ValueError("context rendered claim IDs mismatch")
            for claim in artifact.claims:
                for ref in (*claim.source_refs, *claim.scope_region.source_refs):
                    catalog.verify_source(ref)
                if claim.id in rep.context_claim_ids and not catalog.covers_child(
                    claim.scope_region.source_refs, child
                ):
                    raise ValueError("context claim does not cover child")
            refs.append(rep.context_ref.model_dump(mode="json"))
            pairs.append((child.id, child.content_hash))
            counts[rep.enrichment_status] += 1
    if (
        refs != root["context_refs"]
        or child_identity_digest(pairs) != manifest.child_id_hash_digest
    ):
        raise ValueError("context artifact/child identity digest mismatch")
    if (
        len(parents) != manifest.parent_count
        or len(pairs) != manifest.child_count
        or counts != manifest.enrichment_counts
    ):
        raise ValueError("context manifest counts differ")
    return parents
