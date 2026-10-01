"""Fact identity is an immutable selection of source text, never an LLM rewrite."""
from __future__ import annotations

from decimal import Decimal
import re
from typing import Literal

from pydantic import Field, StrictBool

from evals.gold_v2_models import FrozenModel


class SourceSpan(FrozenModel):
    source: Literal["response", "question"]
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, ge=1)
    quote: str = Field(strict=True, min_length=1, max_length=32000)


class CoreferenceLink(FrozenModel):
    mention: SourceSpan
    antecedent: SourceSpan


class GroundedFact(FrozenModel):
    fact_id: int = Field(strict=True, ge=0)
    evidence_spans: list[SourceSpan] = Field(min_length=1, max_length=32)
    subject_spans: list[SourceSpan] = Field(default_factory=list, max_length=16)
    predicate_spans: list[SourceSpan] = Field(default_factory=list, max_length=16)
    value_spans: list[SourceSpan] = Field(default_factory=list, max_length=16)
    unit_spans: list[SourceSpan] = Field(default_factory=list, max_length=16)
    condition_spans: list[SourceSpan] = Field(default_factory=list, max_length=32)
    polarity_spans: list[SourceSpan] = Field(default_factory=list, max_length=16)
    coreference_links: list[CoreferenceLink] = Field(default_factory=list, max_length=16)


class SegmentReview(FrozenModel):
    segment_id: int = Field(strict=True, ge=0)
    factual: StrictBool


class GroundedExtraction(FrozenModel):
    facts: list[GroundedFact] = Field(max_length=128)
    segments: list[SegmentReview] = Field(max_length=512)


def resolve_span(span: SourceSpan, question: str, answer: str, containers=()) -> SourceSpan:
    text = answer if span.source == "response" else question
    if span.end > span.start and text[span.start:span.end] == span.quote:
        return span
    if text.count(span.quote) == 1:
        start = text.index(span.quote)
        return span.model_copy(update={"start": start, "end": start + len(span.quote)})
    positions = {c.start + m.start() for c in containers if c.source == span.source
                 for m in re.finditer(re.escape(span.quote), text[c.start:c.end])}
    if len(positions) == 1:
        start = positions.pop()
        return span.model_copy(update={"start": start, "end": start + len(span.quote)})
    raise ValueError("source_quote_missing_or_ambiguous")


def _inside(span, containers):
    return any(span.source == c.source and c.start <= span.start < span.end <= c.end for c in containers)


def validate_grounded_facts(question: str, answer: str, facts: list[GroundedFact]) -> list[GroundedFact]:
    if not facts:
        raise ValueError("empty_factual_extraction")
    if len({f.fact_id for f in facts}) != len(facts):
        raise ValueError("duplicate_fact_id")
    result = []
    roles = ("subject_spans", "predicate_spans", "value_spans", "unit_spans", "condition_spans", "polarity_spans")
    for fact in facts:
        evidence = [resolve_span(s, question, answer) for s in fact.evidence_spans]
        if any(s.source != "response" for s in evidence):
            raise ValueError("evidence_response_only")
        links = [CoreferenceLink(mention=resolve_span(c.mention, question, answer, evidence),
            antecedent=resolve_span(c.antecedent, question, answer)) for c in fact.coreference_links]
        if any(not _inside(c.mention, evidence) for c in links):
            raise ValueError("coreference_mention_outside_evidence")
        update = {"evidence_spans": evidence, "coreference_links": links}
        for role in roles:
            spans = [resolve_span(s, question, answer, evidence) for s in getattr(fact, role)]
            for s in spans:
                if s.source == "question":
                    if role not in {"subject_spans", "condition_spans"}:
                        raise ValueError("value_predicate_unit_polarity_response_only")
                elif not _inside(s, evidence) and not (role in {"subject_spans", "condition_spans"}
                        and any(s == c.antecedent for c in links)):
                    raise ValueError("role_outside_evidence")
            update[role] = spans
        result.append(fact.model_copy(update=update))
    return result


def answer_segments(answer: str) -> list[dict]:
    # Keep original offsets/punctuation. Decimal points are not boundaries.
    return [{"segment_id": i, "start": m.start(), "end": m.end(), "quote": m.group()}
            for i, m in enumerate(m for m in re.finditer(r"[^。！？\n]+[。！？\n]*|[。！？\n]+", answer) if m.group().strip())]


def validate_extraction_coverage(question: str, answer: str, extraction: GroundedExtraction) -> GroundedExtraction:
    facts = validate_grounded_facts(question, answer, extraction.facts)
    segments = answer_segments(answer)
    reviews = {s.segment_id: s.factual for s in extraction.segments}
    if len(reviews) != len(extraction.segments) or set(reviews) != set(range(len(segments))):
        raise ValueError("incomplete_segment_review")
    for s in segments:
        if reviews[s["segment_id"]] and not any(e.start < s["end"] and e.end > s["start"] for f in facts for e in f.evidence_spans):
            raise ValueError("uncovered_segment")
    return extraction.model_copy(update={"facts": facts})


_DIGITS = {c: n for n, c in enumerate("零一二三四五六七八九")} | {"〇": 0, "两": 2}
_SMALL = {"十": 10, "百": 100, "千": 1000}


def _chinese_integer(text):
    if all(c in _DIGITS for c in text):
        return int("".join(str(_DIGITS[c]) for c in text))
    for unit, factor in (("亿", 100000000), ("万", 10000)):
        if unit in text:
            left, right = text.split(unit, 1)
            return (_chinese_integer(left) if left else 1) * factor + (_chinese_integer(right) if right else 0)
    total = section = number = 0
    for c in text:
        if c in _DIGITS:
            number = _DIGITS[c]
        elif c in _SMALL:
            section += (number or 1) * _SMALL[c]
            number = 0
    return total + section + number


def literal_numbers(text: str) -> list[Decimal]:
    """No arithmetic inference: normalized numeric lexemes only, units kept apart."""
    text = text.replace("百分之", "").replace("千分之", "")
    values = [Decimal(n) for n in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text.replace(",", ""))]
    for token in re.findall(r"[负零〇一二两三四五六七八九十百千万亿]+(?:点[零〇一二三四五六七八九]+)?", text):
        negative = token.startswith("负")
        token = token.removeprefix("负")
        if not token or all(c in "万亿" for c in token):
            continue
        whole, _, fraction = token.partition("点")
        value = Decimal(_chinese_integer(whole))
        if fraction:
            value += Decimal("0." + "".join(str(_DIGITS[c]) for c in fraction))
        values.append(-value if negative else value)
    return values
