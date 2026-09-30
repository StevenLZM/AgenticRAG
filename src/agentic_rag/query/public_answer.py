"""Strict public projection for audited answers and safe terminal outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator


SafeIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
PublicTerminalStatus = Literal[
    "clarify",
    "refuse",
    "cannot_answer",
    "research_action_invalid",
    "audit_failed",
    "research_round_limit",
]
PublicRoute = Literal["chat", "fast_rag", "research"]
PublicClientProvenance = Literal["api", "fixture", "real_query_api", "real_query_graph"]


class PublicAnswerSegment(BaseModel):
    """One audited display segment; no provider or tool fields are accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["content", "heading", "separator", "references"]
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000),
    ]
    evidence_ids: tuple[SafeIdentifier, ...] = Field(default=(), max_length=32)


class PublicAnswer(BaseModel):
    """The only answer representation permitted across the public API boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PublicTerminalStatus | None = None
    audited: Literal[True] | None = None
    segments: tuple[PublicAnswerSegment, ...] = Field(default=(), max_length=64)
    evidence_parent_ids: tuple[SafeIdentifier, ...] = Field(default=(), max_length=256)
    route: PublicRoute | None = None
    runtime_config_snapshot_id: SafeIdentifier | None = None
    client_provenance: PublicClientProvenance | None = None
    citation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _audited_segments_only(self) -> "PublicAnswer":
        if self.route == "chat":
            if self.audited is not None or self.evidence_parent_ids or self.citation_coverage is not None:
                raise ValueError("chat must not claim document audits or citations")
            if any(s.evidence_ids or s.kind != "content" for s in self.segments):
                raise ValueError("chat permits only uncited content")
        elif self.segments and self.audited is not True:
            raise ValueError("public answer segments must be audited")
        if self.status is None and not self.segments:
            raise ValueError("public answer must contain audited segments or a safe status")
        return self


def project_public_answer(
    value: object,
    *,
    evidence_parent_ids: Iterable[str] | None = None,
    route: str | None = None,
    runtime_config_snapshot_id: str | None = None,
    require_audited: bool = False,
) -> PublicAnswer | None:
    """Allowlist and strictly validate an untrusted stored or graph answer.

    Unknown fields are deliberately discarded before strict validation.  The
    strict model remains useful at typed call sites where unknown fields must
    be rejected, while persistence/API boundaries can safely salvage reviewed
    fields without ever returning raw provider or tool material.
    """
    if not isinstance(value, Mapping):
        return None
    candidate: dict[str, object] = {}
    for field_name in (
        "status",
        "audited",
        "route",
        "runtime_config_snapshot_id",
        "client_provenance",
        "citation_coverage",
    ):
        if field_name in value:
            candidate[field_name] = value[field_name]

    segments = value.get("segments")
    if isinstance(segments, (list, tuple)):
        candidate["segments"] = [
            {
                field_name: item[field_name]
                for field_name in ("kind", "text", "evidence_ids")
                if field_name in item
            }
            for item in segments
            if isinstance(item, Mapping)
        ]

    if evidence_parent_ids is None:
        stored_parent_ids = value.get("evidence_parent_ids")
        if isinstance(stored_parent_ids, (list, tuple)):
            candidate["evidence_parent_ids"] = list(stored_parent_ids)
    else:
        candidate["evidence_parent_ids"] = list(dict.fromkeys(evidence_parent_ids))
    if route is not None:
        candidate["route"] = route
    if runtime_config_snapshot_id is not None:
        candidate["runtime_config_snapshot_id"] = runtime_config_snapshot_id

    try:
        projected = PublicAnswer.model_validate(candidate)
    except (ValidationError, TypeError, ValueError):
        return None
    if require_audited and (projected.audited is not True or not projected.segments):
        return None
    return projected
