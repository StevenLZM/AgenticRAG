"""Structured, evidence-bounded answer generation."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from agentic_rag.query.evidence_builder import PackedEvidence
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway, StructuredOutputValidationError, load_prompt
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class AnswerSegment(BaseModel):
    """One bounded display segment; citations are checked by CitationValidator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["content", "heading", "separator", "references"]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8_000)]
    evidence_ids: tuple[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)], ...
    ] = Field(default=(), max_length=32)


class AnswerDraft(BaseModel):
    """Immutable model output that is never exposed before the audit gates pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: tuple[AnswerSegment, ...] = Field(min_length=1, max_length=64)


class StructuredGateway(Protocol):
    async def complete_structured(self, call: ModelCall, schema: type[object]) -> object: ...


class AnswerGenerationUnavailable(RuntimeError):
    """Generation or its strict response parsing failed safely."""


class AnswerGenerator:
    """Generate only structured drafts whose cited identifiers are manifest-owned."""

    def __init__(self, gateway: StructuredGateway | ModelGateway) -> None:
        self._gateway = gateway

    async def generate(
        self,
        question: str,
        packed_evidence: PackedEvidence,
        snapshot: RuntimeConfigSnapshot,
        *,
        repair_issues: tuple[str, ...] = (),
    ) -> AnswerDraft:
        payload = {
            "question": question,
            "packed_context": packed_evidence.rendered_context,
            "evidence_manifest": {
                evidence_id: entry.model_dump(mode="json")
                for evidence_id, entry in packed_evidence.manifest.items()
            },
            "repair_issues": list(repair_issues),
        }
        call = ModelCall(
            model_role="main",
            snapshot=snapshot,
            messages=(
                {"role": "system", "content": load_prompt("generator_v1").content},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
            ),
        )
        try:
            response = await self._gateway.complete_structured(call, AnswerDraft)
            value = getattr(response, "value", response)
            draft = value if isinstance(value, AnswerDraft) else AnswerDraft.model_validate(value)
        except asyncio.CancelledError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except (OSError, TimeoutError, ConnectionError, StructuredOutputValidationError, ValidationError, TypeError, ValueError) as error:
            raise AnswerGenerationUnavailable("answer generation was unavailable or invalid") from error
        except Exception as error:
            raise AnswerGenerationUnavailable("answer generation failed") from error
        known_ids = set(packed_evidence.manifest)
        cited_ids = {evidence_id for segment in draft.segments for evidence_id in segment.evidence_ids}
        if not cited_ids.issubset(known_ids):
            raise AnswerGenerationUnavailable("model cited an evidence identifier outside the manifest")
        return draft


def render_final_answer(draft: AnswerDraft) -> str:
    """Render approved display text only after all gates succeeded."""
    return "\n".join(segment.text for segment in draft.segments)
