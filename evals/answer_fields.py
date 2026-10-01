"""Blind structured extraction; deterministic critical-field comparisons."""
from __future__ import annotations

from decimal import Decimal
import re

from pydantic import Field, field_validator

from evals.gold_v2_models import CriticalField, FrozenModel, Identifier, Text, finite_number
from evals.grounded_facts import SourceSpan, answer_segments, literal_numbers, resolve_span


class ObservedValue(FrozenModel):
    value: float
    unit: Text | None
    conditions: dict[str, str | int]
    quote: Text
    start: int = Field(strict=True, ge=0)
    end: int = Field(strict=True, ge=1)
    unit_evidence: list[SourceSpan] = Field(default_factory=list, max_length=8)
    condition_evidence: dict[str, SourceSpan] = Field(default_factory=dict)

    @field_validator("value", mode="before")
    @classmethod
    def finite(cls, value):
        return finite_number(value)


class ExtractedField(FrozenModel):
    name: Identifier
    values: list[ObservedValue] = Field(max_length=16)


class ExtractedFields(FrozenModel):
    fields: list[ExtractedField] = Field(max_length=128)


_DESCRIPTIONS = {
    "lodging_limit": "每人每晚住宿报销上限及实际适用对象/年份/城市",
    "increase": "相较上一年每人每晚住宿标准增加的金额（不是任何一年的上限）",
    "annual_availability": "按全年加总监测/中断分钟计算的年度可用率",
    "ending_inventory": "期初+入库-出库后的期末库存数量",
    "score": "问题所问测验的成绩",
}


def extraction_payload(question, response, expected: list[CriticalField]):
    # No gold value, tolerance, condition value, or reference answer reaches
    # this boundary. Unit and conditions must be observed, not copied from gold.
    return {"question": question, "response": response,
            "field_specs": [{"name": field.name, "description": _DESCRIPTIONS.get(field.name, field.name),
                             "condition_keys": sorted(field.conditions)} for field in expected]}


def _currency_expression_supported(observed, spans):
    """Bind the whole literal amount, including scale, to its own currency span."""
    digits = "零〇一二两三四五六七八九十百千万亿点负"
    pattern = rf"(?<![\d.{digits}])(?P<number>[+-]?(?:\d+(?:,\d{{3}})*(?:\.\d+)?\s*[十百千万亿]*|[{digits}]+))\s*(?P<currency>元(?:人民币)?|人民币|CNY)"
    expected = Decimal(str(observed.value)) * _unit(observed.unit)[1]
    for match in re.finditer(pattern, observed.quote):
        raw = match["number"].replace(",", "").replace(" ", "")
        arabic = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)([十百千万亿]*)", raw)
        if arabic:
            scale = literal_numbers("一" + arabic[2]) if arabic[2] else [Decimal(1)]
            values = [Decimal(arabic[1]) * scale[0]] if len(scale) == 1 else []
        else:
            values = literal_numbers(raw)
        start, end = observed.start + match.start("currency"), observed.start + match.end("currency")
        if expected in values and any(s.source == "response" and s.start < end and start < s.end for s in spans):
            return True
    return False


def validate_extraction(extracted: ExtractedFields, answer: str, expected, *, question="", require_grounding=False):
    names = [f.name for f in extracted.fields]
    if len(set(names)) != len(names) or set(names) != {f.name for f in expected}:
        raise ValueError("extraction omitted/duplicated/added field definition")
    for field in extracted.fields:
        for observed in field.values:
            if answer[observed.start:observed.end] != observed.quote:
                raise ValueError("extraction quote/span is not in final answer")
            # Ground the number in the actual answer. An LLM must not fill a
            # missing number from world knowledge or an expected gold value.
            currency = require_grounding and _unit(observed.unit)[0] in {"CNY", "CNY/person/night"}
            if not currency and Decimal(str(observed.value)) not in literal_numbers(observed.quote):
                raise ValueError("extracted number not supported by quote")
            if require_grounding:
                for key, value in observed.conditions.items():
                    if key not in observed.condition_evidence:
                        raise ValueError("condition_source_missing")
                    evidence = resolve_span(observed.condition_evidence[key], question, answer)
                    if (isinstance(value, int) and Decimal(value) not in literal_numbers(evidence.quote)) or (
                            isinstance(value, str) and value not in evidence.quote):
                        raise ValueError("condition_not_supported_by_quote")
                if observed.unit is not None:
                    quote_scope = SourceSpan(source="response", start=observed.start, end=observed.end, quote=observed.quote)
                    spans = [resolve_span(s, question, answer, [quote_scope]) for s in observed.unit_evidence]
                    number_segments = [s for s in answer_segments(answer)
                                       if s["start"] <= observed.start < observed.end <= s["end"]
                                       and Decimal(str(observed.value)) in literal_numbers(s["quote"])]
                    # Extra literal citations remain in the audit record, but
                    # cannot prove units or counting basis for this value.
                    # Equal numbers in an unrelated sentence are not anchors.
                    proof = [s for s in spans if s.source == "question" or any(
                        n["start"] <= s.start < s.end <= n["end"] for n in number_segments)]
                    response_units = " ".join(s.quote for s in proof if s.source == "response")
                    all_units = " ".join(s.quote for s in proof)
                    unit, _ = _unit(observed.unit)
                    supported = {
                        "CNY": any(s in response_units for s in ("元", "人民币", "CNY")),
                        "CNY/person/night": any(s in response_units for s in ("元", "人民币", "CNY"))
                            and any(s in all_units for s in ("每人", "/人", "person"))
                            and any(s in all_units for s in ("每晚", "/晚", "每夜", "/夜", "night")),
                        "percent": any(s in response_units for s in ("%", "％", "百分")),
                        "device": any(s in response_units for s in ("台", "device")),
                        "point": any(s in response_units for s in ("分", "point")),
                    }.get(unit, observed.unit in response_units)
                    if not supported or ("万" in observed.unit and "万" not in response_units):
                        raise ValueError("unit_not_supported_by_quote")
                    if currency and not _currency_expression_supported(observed, proof):
                        raise ValueError("unit_amount_expression_mismatch")


def _unit(unit):
    # Explicit conversions only; no fuzzy "close enough" currency or unit swap.
    aliases = {"%": ("percent", 1), "百分比": ("percent", 1), "percent": ("percent", 1),
               "ratio": ("percent", 100), "device": ("device", 1), "台": ("device", 1),
               "point": ("point", 1), "分": ("point", 1),
               "CNY/person/night": ("CNY/person/night", 1), "元/人/晚": ("CNY/person/night", 1),
               "元/人/夜": ("CNY/person/night", 1), "万元/人/晚": ("CNY/person/night", 10000),
               "CNY": ("CNY", 1), "元": ("CNY", 1), "万元": ("CNY", 10000)}
    return aliases.get(unit, (unit, 1))


def compare_fields(extracted: ExtractedFields, expected: list[CriticalField]) -> dict:
    by_name = {f.name: f.values for f in extracted.fields}
    rows = []
    for field in expected:
        values = by_name.get(field.name, [])
        status = "missing"
        if values:
            signatures = {(v.value * _unit(v.unit)[1], _unit(v.unit)[0], tuple(sorted(v.conditions.items()))) for v in values}
            if len(signatures) > 1:
                status = "contradiction"
            else:
                value = values[0]
                observed_unit, factor = _unit(value.unit)
                expected_unit, expected_factor = _unit(field.unit)
                if value.unit is None:
                    status = "missing_unit"
                elif observed_unit != expected_unit:
                    status = "incorrect_unit"
                elif any(value.conditions.get(k) != v for k, v in field.conditions.items()):
                    status = "incorrect_conditions"
                elif abs(Decimal(str(value.value)) * factor - Decimal(str(field.value)) * expected_factor) > Decimal(str(field.tolerance)) * expected_factor:
                    status = "incorrect_value"
                else:
                    status = "pass"
        rows.append({"name": field.name, "status": status})
    return {"status": "evaluated" if rows else "not_applicable", "fields": rows,
            "critical_field_accuracy": sum(r["status"] == "pass" for r in rows) / len(rows) if rows else None,
            "all_critical_fields_pass": float(all(r["status"] == "pass" for r in rows)) if rows else None,
            "contradiction_rate": sum(r["status"] == "contradiction" for r in rows) / len(rows) if rows else None}
