"""Bounded, immutable excerpts of the evidence actually cited by one answer."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from agentic_rag.ingestion.chunker import AstLocator
from agentic_rag.query.evidence_builder import PackedEvidence
from agentic_rag.query.public_answer import PublicAnswer

MAX_BYTES = 256 * 1024
SourceId = Annotated[
    str, StringConstraints(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_.:-]+$")
]


class InvalidAnswerSources(ValueError):
    pass


class AnswerSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_id: SourceId
    document_id: SourceId
    document_version_id: SourceId
    parent_id: SourceId
    heading_path: tuple[str, ...] = Field(default=(), max_length=16)
    heading_truncated: bool = False
    page_from: int | None = Field(default=None, ge=1, le=2147483647)
    page_to: int | None = Field(default=None, ge=1, le=2147483647)
    excerpt: str = Field(default="", max_length=2000)
    truncated: bool = False
    excerpt_omitted: bool = False

    @model_validator(mode="after")
    def bounded_heading(self):
        if sum(map(len, self.heading_path)) > 128:
            raise ValueError("heading too long")
        if (self.page_from is None) != (self.page_to is None) or (
            self.page_from is not None and self.page_to < self.page_from
        ):
            raise ValueError("invalid page range")
        return self


class AnswerSources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    run_id: SourceId
    runtime_config_snapshot_id: SourceId
    items: tuple[AnswerSource, ...] = Field(max_length=64)
    omitted_source_count: int = Field(default=0, ge=0, le=2048)

    @model_validator(mode="after")
    def bounded_storage(self):
        if not within_budget(self.model_dump(mode="json")):
            raise ValueError("source snapshot exceeds byte budget")
        if len({item.evidence_id for item in self.items}) != len(self.items):
            raise ValueError("duplicate source")
        return self


def within_budget(value: dict) -> bool:
    return (
        max(
            len(json.dumps(value).encode()),
            len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()),
        )
        <= MAX_BYTES
    )


def cited_ids(answer: PublicAnswer | None) -> tuple[str, ...]:
    if answer is None or answer.audited is not True:
        return ()
    return tuple(
        dict.fromkeys(
            eid for segment in answer.segments for eid in segment.evidence_ids
        )
    )


def build_answer_sources(
    answer: PublicAnswer, packed: PackedEvidence, *, run_id: str, snapshot_id: str
) -> AnswerSources | None:
    ids = cited_ids(answer)
    if not ids:
        return None
    try:
        lookup = {item.evidence_id: item for item in packed.items}
        if len(lookup) != len(packed.items):
            raise InvalidAnswerSources("duplicate packed evidence")
        sources, contents = [], []
        for eid in ids:
            item, manifest = lookup.get(eid), packed.manifest.get(eid)
            if (
                item is None
                or manifest is None
                or item.model_dump(exclude={"content", "covered_target_ids"})
                != manifest.model_dump()
            ):
                raise InvalidAnswerSources("packed evidence and manifest disagree")
            headings, remaining = [], 128
            for heading in item.heading_path[:16]:
                if remaining <= 0:
                    break
                headings.append(heading[:remaining])
                remaining -= len(headings[-1])
            pages = (None, None)
            try:
                locator = AstLocator.model_validate_json(item.ast_locator)
                low, high = (
                    min(s.page_from for s in locator.spans),
                    max(s.page_to for s in locator.spans),
                )
                if high <= 2147483647:
                    pages = (low, high)
            except (ValidationError, ValueError, TypeError):
                pass
            source = AnswerSource(
                evidence_id=eid,
                document_id=item.document_id,
                document_version_id=item.document_version_id,
                parent_id=item.parent_id,
                heading_path=tuple(headings),
                heading_truncated=tuple(headings) != item.heading_path,
                page_from=pages[0],
                page_to=pages[1],
                excerpt_omitted=bool(item.content),
                truncated=bool(item.content),
            )
            if len(sources) < 64:
                sources.append(source.model_dump(mode="json"))
                contents.append(item.content)
        # Reserve all bounded metadata before allocating text in citation order.
        candidate = dict(
            schema_version=1,
            run_id=run_id,
            runtime_config_snapshot_id=snapshot_id,
            items=sources,
            omitted_source_count=max(0, len(ids) - 64),
        )
        if not within_budget(candidate):
            raise InvalidAnswerSources("metadata exceeds source budget")
        for source, content in zip(sources, contents):
            low, high = 0, min(2000, len(content))
            while low < high:
                mid = (low + high + 1) // 2
                source.update(
                    excerpt=content[:mid],
                    excerpt_omitted=False,
                    truncated=mid < len(content),
                )
                if within_budget(candidate):
                    low = mid
                else:
                    high = mid - 1
            source.update(
                excerpt=content[:low],
                excerpt_omitted=bool(content) and low == 0,
                truncated=low < len(content),
            )
        return AnswerSources.model_validate(candidate)
    except (ValidationError, ValueError, TypeError) as error:
        raise InvalidAnswerSources("invalid source snapshot") from error
