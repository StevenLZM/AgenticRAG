"""Blind numbered-clause selection; exact source offsets are computed, never generated."""
from __future__ import annotations

import re
from typing import Annotated

from pydantic import Field, field_validator

from evals.answer_fields import ExtractedFields
from evals.gold_v2_models import FrozenModel, Identifier, Text, finite_number
from evals.grounded_facts import GroundedExtraction, SegmentReview, SourceSpan, answer_segments, validate_extraction_coverage

SELECTION_PROTOCOL = "numbered_sentence_clause_quote_v1"
ROLES = ("subject_spans", "predicate_spans", "value_spans", "unit_spans",
         "condition_spans", "polarity_spans")


class SpanSelection(FrozenModel):
    span_id: str = Field(strict=True, pattern=r"^[rq][0-9]+$")
    quote: str = Field(strict=True, min_length=1, max_length=32000)


class SelectedCoreference(FrozenModel):
    mention: SpanSelection
    antecedent: SpanSelection


class SelectedFact(FrozenModel):
    fact_id: int = Field(strict=True, ge=0)
    evidence_segment_ids: list[Annotated[int, Field(strict=True, ge=0)]] = Field(min_length=1, max_length=32)
    subject_spans: list[SpanSelection] = Field(max_length=16)
    predicate_spans: list[SpanSelection] = Field(max_length=16)
    value_spans: list[SpanSelection] = Field(max_length=16)
    unit_spans: list[SpanSelection] = Field(max_length=16)
    condition_spans: list[SpanSelection] = Field(max_length=32)
    polarity_spans: list[SpanSelection] = Field(max_length=16)
    coreference_links: list[SelectedCoreference] = Field(max_length=16)


class SelectedExtraction(FrozenModel):
    facts: list[SelectedFact] = Field(max_length=128)
    segments: list[SegmentReview] = Field(max_length=512)


class SelectedValue(FrozenModel):
    value: float
    unit: Text | None
    conditions: dict[str, str | int]
    evidence: SpanSelection
    # Required schema properties; absence cannot silently become empty evidence.
    unit_evidence: list[SpanSelection] = Field(max_length=8)
    condition_evidence: dict[str, SpanSelection]

    @field_validator("value", mode="before")
    @classmethod
    def finite(cls, value):
        return finite_number(value)


class SelectedField(FrozenModel):
    name: Identifier
    values: list[SelectedValue] = Field(max_length=16)


class SelectedFields(FrozenModel):
    fields: list[SelectedField] = Field(max_length=128)


def source_catalog(question: str, answer: str) -> dict[str, SourceSpan]:
    """Clause boundaries affect addressing only, never the text or factual coverage."""
    result = {}
    for prefix, source, text in (("r", "response", answer), ("q", "question", question)):
        start, index = 0, 0
        # Do not split decimal points or a comma between digits (e.g. 1,500).
        boundaries = [m.end() for m in re.finditer(r"[，；;：:。！？!?\n]|(?<!\d),|,(?!\d)", text)]
        for end in [*boundaries, len(text)]:
            if text[start:end].strip():
                result[f"{prefix}{index}"] = SourceSpan(source=source, start=start, end=end, quote=text[start:end])
                index += 1
            start = end
    return result


def catalog_payload(catalog):
    return [{"span_id": key, "source": span.source, "text": span.quote} for key, span in catalog.items()]


def resolve_selection(selection: SpanSelection, catalog: dict[str, SourceSpan]) -> SourceSpan:
    parent = catalog.get(selection.span_id)
    if parent is None:
        raise ValueError("unknown_span_id")
    start = parent.quote.find(selection.quote)
    if start < 0:
        raise ValueError("selection_quote_missing")
    if parent.quote.find(selection.quote, start + 1) >= 0:
        raise ValueError("selection_quote_ambiguous")
    return SourceSpan(source=parent.source, start=parent.start + start,
                      end=parent.start + start + len(selection.quote), quote=selection.quote)


def resolve_facts(question, answer, extracted: SelectedExtraction, catalog):
    facts = []
    segments = answer_segments(answer)
    for fact in extracted.facts:
        if (len(set(fact.evidence_segment_ids)) != len(fact.evidence_segment_ids)
                or any(i >= len(segments) for i in fact.evidence_segment_ids)):
            raise ValueError("invalid_evidence_segment_ids")
        evidence = [SourceSpan(source="response", start=segments[i]["start"], end=segments[i]["end"],
                    quote=segments[i]["quote"]) for i in fact.evidence_segment_ids]
        facts.append({"fact_id": fact.fact_id,
            "evidence_spans": evidence,
            **{role: [resolve_selection(s, catalog) for s in getattr(fact, role)] for role in ROLES},
            "coreference_links": [{"mention": resolve_selection(c.mention, catalog),
                "antecedent": resolve_selection(c.antecedent, catalog)} for c in fact.coreference_links]})
    result = GroundedExtraction(facts=facts, segments=extracted.segments)
    return validate_extraction_coverage(question, answer, result)


def resolve_fields(extracted: SelectedFields, catalog) -> ExtractedFields:
    fields = []
    for field in extracted.fields:
        values = []
        for value in field.values:
            span = resolve_selection(value.evidence, catalog)
            if span.source != "response":
                raise ValueError("field_value_response_only")
            values.append({"value": value.value, "unit": value.unit, "conditions": value.conditions,
                "quote": span.quote, "start": span.start, "end": span.end,
                "unit_evidence": [resolve_selection(s, catalog) for s in value.unit_evidence],
                "condition_evidence": {k: resolve_selection(s, catalog) for k, s in value.condition_evidence.items()}})
        fields.append({"name": field.name, "values": values})
    return ExtractedFields(fields=fields)
