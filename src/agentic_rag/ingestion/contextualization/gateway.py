"""Constrained model port and one-request production gateway adapter."""

from __future__ import annotations
from typing import Literal, Protocol
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from agentic_rag.ingestion.contextualization.assembly import ContextInput
from agentic_rag.runtime.model_gateway import ModelCall, ModelGateway
from agentic_rag.runtime.models import RuntimeConfigSnapshot


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Quote(Frozen):
    source_id: str
    quote: str = Field(min_length=1)


class CandidateClaim(Frozen):
    field: Literal[
        "company", "population", "business_period", "topic", "term", "currency", "unit"
    ]
    value: str = Field(min_length=1, max_length=256)
    source_quotes: tuple[Quote, ...] = Field(min_length=1, max_length=16)
    scope_region_id: str
    exclusion_conditions: tuple[str, ...] = ()


class ExtractionPayload(Frozen):
    target_source_child_id: str
    claims: tuple[CandidateClaim, ...] = Field(default=(), max_length=16)


class Usage(Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    complete: bool = False


class ExtractionResponse(ExtractionPayload):
    usage: Usage = Usage()


class VerificationInput(Frozen):
    context: ContextInput
    candidate: CandidateClaim


class VerificationPayload(Frozen):
    status: Literal["supported", "contradicted", "insufficient"]
    source_ids: tuple[str, ...] = ()
    reason: str = ""


class VerificationResponse(VerificationPayload):
    usage: Usage = Usage()


class ContextModelPort(Protocol):
    async def extract(self, input: ContextInput) -> ExtractionResponse: ...
    async def verify(self, input: VerificationInput) -> VerificationResponse: ...


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def prompt_text(kind: str) -> str:
    return (PROMPT_DIR / f"context_{kind}_v1.md").read_text(encoding="utf-8")


class GatewayContextModel:
    """Client must expose max_retries=0; gateway itself also has no retries.

    A max_provider_attempts=1 call explicitly suppresses structured repair.
    Each method therefore dispatches at most the single reserved request.
    """

    def __init__(self, *, client: object, snapshot: RuntimeConfigSnapshot, build):
        if getattr(client, "max_retries", None) != 0:
            raise ValueError("context provider SDK max_retries must explicitly be 0")
        self._gateway = ModelGateway(client, max_retries=0)
        self._snapshot = snapshot
        self._build = build
        from agentic_rag.models.contextual import text_sha256

        if (
            text_sha256(prompt_text("extract")) != build.extraction_prompt_hash
            or text_sha256(prompt_text("verify")) != build.verification_prompt_hash
        ):
            raise ValueError("context prompt hash differs from frozen build")
        if snapshot.provider_config_fingerprint != build.provider_fingerprint:
            raise ValueError("context provider fingerprint differs from snapshot")

    async def _call(self, kind, input, schema):
        model = (
            self._build.extraction_model
            if kind == "extract"
            else self._build.verification_model
        )
        snapshot = self._snapshot.model_copy(update={"light_model_id": model})
        call = ModelCall(
            messages=(
                {"role": "system", "content": prompt_text(kind)},
                {"role": "user", "content": input.model_dump_json()},
            ),
            model_role="light",
            snapshot=snapshot,
            temperature=0,
            max_output_tokens=self._build.max_output_tokens,
            max_provider_attempts=1,
            timeout_seconds=self._build.timeout_seconds,
        )
        result = await self._gateway.complete_structured(call, schema)
        usage = Usage(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            complete=result.usage_complete,
        )
        return result.value, usage

    async def extract(self, input: ContextInput) -> ExtractionResponse:
        value, usage = await self._call("extract", input, ExtractionPayload)
        return ExtractionResponse(**value.model_dump(), usage=usage)

    async def verify(self, input: VerificationInput) -> VerificationResponse:
        value, usage = await self._call("verify", input, VerificationPayload)
        return VerificationResponse(**value.model_dump(), usage=usage)
