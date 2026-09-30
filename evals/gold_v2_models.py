"""Frozen gold and corpus contracts, separate from legacy Parent-only cases."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, field_validator, model_validator

Text = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Digest = Annotated[str, StringConstraints(strict=True, pattern=r"^[a-f0-9]{64}$")]
Identifier = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str).encode()).hexdigest()


def finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected a finite numeric value, not a boolean/string")
    return float(value)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CorpusSnapshot(FrozenModel):
    schema_version: Literal[2]
    captured_at: Text
    snapshot_id: Digest
    scope: dict[str, Any]
    status_counts: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    parents: list[dict[str, Any]]
    children: list[dict[str, Any]]

    @model_validator(mode="before")
    @classmethod
    def verify_hash(cls, value):
        if isinstance(value, dict):
            unsigned = {k: v for k, v in value.items() if k != "snapshot_id"}
            if canonical_hash(unsigned) != value.get("snapshot_id"):
                raise ValueError("corpus snapshot hash mismatch")
        return value

    @model_validator(mode="after")
    def verify_members(self):
        user = self.scope.get("user_id")
        if not isinstance(user, str) or not user.strip():
            raise ValueError("snapshot must bind user scope")
        versions = {(d["document_id"], d["document_version_id"]) for d in self.documents}
        if len(versions) != len(self.documents):
            raise ValueError("duplicate snapshot document/version")
        for rows, key in ((self.parents, "id"), (self.children, "child_id")):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError("duplicate snapshot chunk")
            if any(r["user_id"] != user or (r["document_id"], r["document_version_id"]) not in versions for r in rows):
                raise ValueError("foreign snapshot chunk")
        parent_ids = {p["id"] for p in self.parents}
        if any(c["parent_id"] not in parent_ids for c in self.children):
            raise ValueError("snapshot child has no parent")
        return self


class SourceDocument(FrozenModel):
    document_id: Identifier
    document_version_id: Identifier
    sha256: Digest
    filename: Text
    searchable_at_snapshot: StrictBool


class SourcePointer(FrozenModel):
    """Original location, not a parsed AST masquerading as original truth."""
    page: int | None = Field(default=None, strict=True, ge=1)
    char_from: int | None = Field(default=None, strict=True, ge=0)
    char_to: int | None = Field(default=None, strict=True, ge=0)
    sheet: Text | None = None
    cells: Text | None = None

    @model_validator(mode="after")
    def bounds(self):
        if self.char_from is not None and (self.char_to is None or self.char_to <= self.char_from):
            raise ValueError("invalid original text range")
        return self


class EvidenceGroup(FrozenModel):
    fact_id: Identifier
    equivalent_fact_id: Identifier | None = None
    claim: Text
    document_version_id: Identifier
    source_anchors: list[Text] = Field(min_length=1)
    source_pointers: list[SourcePointer] = Field(default_factory=list)
    subject: Text | None = None
    predicate: Text | None = None
    value: str | int | float | None = None
    unit: Text | None = None
    polarity: Literal["positive", "negative", "unknown"] = "positive"
    conditions: dict[str, str | int] = Field(default_factory=dict)
    parent_ids_any_of: list[Identifier]
    child_ids_any_of: list[Identifier]
    child_evidence_sets_any_of: list[list[Identifier]]
    child_mapping_methods: list[Text]
    mapping_status: Literal["mapped", "mapped_multi_child", "parent_only", "unmapped"]

    @model_validator(mode="after")
    def sufficient_sets(self):
        sets = self.child_evidence_sets_any_of
        if any(not s or len(set(s)) != len(s) for s in sets):
            raise ValueError("sufficient evidence sets must be nonempty and unique")
        singles = {s[0] for s in sets if len(s) == 1}
        if set(self.child_ids_any_of) != singles:
            raise ValueError("singleton aliases must match sufficient evidence sets")
        if self.mapping_status.startswith("mapped") and not sets:
            raise ValueError("mapped fact must have sufficient Child evidence")
        if self.mapping_status in {"unmapped", "parent_only"} and sets:
            raise ValueError("unmapped fact cannot claim sufficient Child evidence")
        return self


class CriticalField(FrozenModel):
    name: Identifier
    value: float
    unit: Text
    tolerance: float = Field(ge=0)
    conditions: dict[str, str | int] = Field(default_factory=dict)

    @field_validator("value", "tolerance", mode="before")
    @classmethod
    def numeric(cls, value):
        return finite_number(value)


class Derivation(FrozenModel):
    operation: Text
    inputs: list[Any] | dict[str, Any]
    sheet: Text | None = None
    source_cells: Text | None = None


class GradedQrel(FrozenModel):
    level: Literal["child", "parent"]
    chunk_id: Identifier
    relevance: int = Field(strict=True, ge=0, le=3)
    reviewer: Text
    reason: Text


class GoldCaseV2(FrozenModel):
    schema_version: Literal[2]
    case_id: Identifier
    snapshot_id: Digest
    user_id: Identifier
    split: Literal["dev", "validation", "test"]
    category: Text
    question: Text
    answerable: StrictBool
    reference_answer: Text
    expected_route: Literal["fast_rag", "research", "fast_rag_or_research", "chat"]
    source_documents: list[SourceDocument] = Field(min_length=1)
    required_evidence_groups_all_of: list[EvidenceGroup] = Field(min_length=1)
    critical_fields: list[CriticalField]
    derivation: Derivation | None
    review_status: Literal["original_verified_human_review_pending", "human_approved", "human_rejected"]
    graded_qrels: list[GradedQrel] = Field(default_factory=list)
    question_family: Text | None = None

    @model_validator(mode="after")
    def distinct(self):
        for values in ([f.fact_id for f in self.required_evidence_groups_all_of],
                       [f.name for f in self.critical_fields],
                       [(s.document_id, s.document_version_id) for s in self.source_documents],
                       [(q.level, q.chunk_id) for q in self.graded_qrels]):
            if len(set(values)) != len(values):
                raise ValueError("duplicate gold identifier")
        return self
