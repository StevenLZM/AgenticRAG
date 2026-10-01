"""Breaks caught: invented entities, ambiguous offsets, lost role provenance/coverage."""
import importlib.util

import pytest


def api():
    assert importlib.util.find_spec("evals.grounded_facts"), "grounded fact validator not implemented"
    from evals import grounded_facts
    return grounded_facts


def span(text, quote, source="response"):
    start = text.index(quote)
    return {"source": source, "start": start, "end": start + len(quote), "quote": quote}


def fact(text, **update):
    return {"fact_id": 0, "evidence_spans": [span(text, text)],
            "subject_spans": [], "predicate_spans": [], "value_spans": [], "unit_spans": [],
            "condition_spans": [], "polarity_spans": [], "coreference_links": [], **update}


def test_changed_entity_is_judge_error():
    mod = api()
    text = "虚构星河04公司上限500元。"
    bad = span(text, "虚构星河04公司") | {"quote": "虚构星尘04公司"}
    with pytest.raises(ValueError, match="source_quote"):
        mod.validate_grounded_facts("上限？", text, [mod.GroundedFact.model_validate(fact(text, subject_spans=[bad]))])


def test_wrong_entity_in_answer_is_not_repaired():
    mod = api()
    text = "虚构星尘04公司上限500元。"
    result = mod.validate_grounded_facts("虚构星河04公司上限？", text,
        [mod.GroundedFact.model_validate(fact(text, subject_spans=[span(text, "虚构星尘04公司")]))])
    assert result[0].subject_spans[0].quote == "虚构星尘04公司"


def test_unique_exact_quote_offset_relocated_but_ambiguous_quote_rejected():
    mod = api()
    text = "😀公司：500元；另一公司600元。"
    value = span(text, "500元") | {"start": 0, "end": 1}
    result = mod.validate_grounded_facts("q", text, [mod.GroundedFact.model_validate(fact(text, value_spans=[value]))])
    assert (result[0].value_spans[0].start, result[0].value_spans[0].end) == (4, 8)
    repeated = "500元和500元"
    with pytest.raises(ValueError, match="source_quote"):
        mod.validate_grounded_facts("q", repeated, [mod.GroundedFact.model_validate(fact(repeated,
            value_spans=[{"source": "response", "start": 3, "end": 4, "quote": "500元"}]))])


def test_repeated_role_quote_can_be_located_uniquely_inside_verified_evidence():
    mod = api()
    text = "A公司500元人民币。单位是元人民币。"
    unit = {"source": "response", "start": 0, "end": 1, "quote": "元人民币"}
    payload = fact(text, evidence_spans=[span(text, "A公司500元人民币。")], unit_spans=[unit])
    checked = mod.validate_grounded_facts("q", text, [mod.GroundedFact.model_validate(payload)])
    assert checked[0].unit_spans[0].start == 6


def test_saved_real_pilot_extraction_is_repaired_only_by_scoped_exact_positions():
    # Minimal frozen S004 pilot response; do not require local paid artifacts in CI.
    first = "虚构星河04公司2026年深圳住宿报销上限为每人每晚500元人民币。"
    second = "单位是元人民币（按每人每晚计）。"
    text = first + second
    raw = {"facts": [fact(text, evidence_spans=[span(text, first)],
        subject_spans=[{"source": "response", "start": 0, "end": 12, "quote": "虚构星河04公司"}],
        unit_spans=[{"source": "response", "start": 31, "end": 34, "quote": "元人民币"}]),
        fact(text, fact_id=1, evidence_spans=[span(text, second)],
            value_spans=[{"source": "response", "start": 37, "end": 41, "quote": "元人民币"}])],
        "segments": [{"segment_id": 0, "factual": True}, {"segment_id": 1, "factual": True}]}
    mod = api()
    extraction = mod.GroundedExtraction.model_validate(raw)
    checked = mod.validate_extraction_coverage("", text, extraction)
    assert [s.quote for f in checked.facts for s in f.evidence_spans] == [s.quote for f in extraction.facts for s in f.evidence_spans]


def test_cannot_fill_answer_value_from_question_or_detach_role_from_evidence():
    mod = api()
    text, question = "不知道。", "是500元吗？"
    with pytest.raises(ValueError, match="response_only"):
        mod.validate_grounded_facts(question, text, [mod.GroundedFact.model_validate(fact(text,
            value_spans=[span(question, "500元", "question")]))])
    text = "A公司500元。B公司600元。"
    with pytest.raises(ValueError, match="outside_evidence"):
        mod.validate_grounded_facts("q", text, [mod.GroundedFact.model_validate(fact(text,
            evidence_spans=[span(text, "A公司500元。")], value_spans=[span(text, "600元")]))])


def test_coreference_requires_literal_mention_and_antecedent_not_generated_name():
    mod = api()
    text = "A公司规定如下。该公司上限500元。"
    payload = fact(text, evidence_spans=[span(text, "该公司上限500元。")],
        subject_spans=[span(text, "A公司")], coreference_links=[{
            "mention": span(text, "该公司"), "antecedent": span(text, "A公司")}])
    assert mod.validate_grounded_facts("q", text, [mod.GroundedFact.model_validate(payload)])
    payload["coreference_links"][0]["antecedent"]["quote"] = "B公司"
    with pytest.raises(ValueError, match="source_quote"):
        mod.validate_grounded_facts("q", text, [mod.GroundedFact.model_validate(payload)])


def test_segment_coverage_cannot_silently_omit_factual_sentence():
    mod = api()
    text = "A公司500元。B公司600元。"
    extracted = mod.GroundedExtraction.model_validate({"facts": [fact(text, evidence_spans=[span(text, "A公司500元。")])],
        "segments": [{"segment_id": 0, "factual": True}, {"segment_id": 1, "factual": True}]})
    with pytest.raises(ValueError, match="uncovered_segment"):
        mod.validate_extraction_coverage("q", text, extracted)


@pytest.mark.parametrize("text,want", [("五百元", "500"), ("五百零五", "505"), ("百分之二十", "20"),
    ("二〇二六年", "2026"), ("0.05万元", "0.05"), ("五百点五", "500.5")])
def test_literal_number_normalization(text, want):
    from decimal import Decimal
    assert Decimal(want) in api().literal_numbers(text)


def test_negation_and_conditions_survive_payload_and_duplicate_ids_rejected():
    mod = api()
    text = "不是500元，仅深圳适用600元。"
    grounded = mod.GroundedFact.model_validate(fact(text, polarity_spans=[span(text, "不是")],
        condition_spans=[span(text, "仅深圳")]))
    facts = mod.validate_grounded_facts("q", text, [grounded])
    assert facts[0].polarity_spans[0].quote == "不是"
    assert facts[0].condition_spans[0].quote == "仅深圳"
    with pytest.raises(ValueError, match="duplicate_fact"):
        mod.validate_grounded_facts("q", text, [grounded, grounded])


def test_percent_prefix_is_not_an_extra_number():
    from decimal import Decimal
    assert api().literal_numbers("百分之二十") == [Decimal("20")]


def test_ambiguous_large_chinese_number_never_normalizes_to_wrong_value():
    from decimal import Decimal
    assert Decimal("10000") not in api().literal_numbers("一万亿")


def test_unit_cannot_be_borrowed_from_another_answer_sentence():
    from evals.answer_fields import ExtractedFields, validate_extraction
    from evals.gold_v2_models import CriticalField
    text = "住宿上限500。会议费100元。"
    raw = {"fields": [{"name": "lodging_limit", "values": [{"value": 500, "unit": "CNY", "conditions": {},
        "quote": "住宿上限500", "start": 0, "end": 7, "unit_evidence": [span(text, "元")], "condition_evidence": {}}]}]}
    with pytest.raises(ValueError, match="unit"):
        validate_extraction(ExtractedFields.model_validate(raw), text,
            [CriticalField(name="lodging_limit", value=500, unit="CNY", tolerance=0)], require_grounding=True)


def test_field_condition_and_unit_need_literal_evidence():
    from evals.answer_fields import ExtractedFields, validate_extraction
    import inspect
    assert "require_grounding" in inspect.signature(validate_extraction).parameters
    from evals.gold_v2_models import CriticalField
    text = "A公司五百元/人/晚。"
    expected = [CriticalField(name="lodging_limit", value=500, unit="CNY/person/night", tolerance=0, conditions={"company": "A公司"})]
    raw = {"fields": [{"name": "lodging_limit", "values": [{"value": 500, "unit": "CNY/person/night",
        "conditions": {"company": "B公司"}, "quote": text, "start": 0, "end": len(text),
        "unit_evidence": [span(text, "元/人/晚")], "condition_evidence": {"company": span(text, "A公司")}}]}]}
    with pytest.raises(ValueError, match="condition"):
        validate_extraction(ExtractedFields.model_validate(raw), text, expected, question="q", require_grounding=True)
    raw["fields"][0]["values"][0]["conditions"]["company"] = "A公司"
    validate_extraction(ExtractedFields.model_validate(raw), text, expected, question="q", require_grounding=True)
    raw["fields"][0]["values"][0]["unit"] = "percent"
    with pytest.raises(ValueError, match="unit"):
        validate_extraction(ExtractedFields.model_validate(raw), text, expected, question="q", require_grounding=True)


@pytest.mark.parametrize("text,value,unit", [
    ("住宿上限为500万元。", 500, "CNY"),
    ("住宿上限500，会议费500元。", 500, "CNY"),
    ("住宿上限为500亿元。", 500, "CNY"),
])
def test_currency_cannot_drop_scale_or_borrow_other_numeric_unit(text, value, unit):
    from evals.answer_fields import ExtractedFields, validate_extraction
    from evals.gold_v2_models import CriticalField
    quote = text.split("，")[0]
    raw = {"fields": [{"name": "lodging_limit", "values": [{"value": value, "unit": unit,
        "conditions": {}, "quote": quote, "start": 0, "end": len(quote),
        "unit_evidence": [span(text, "元")]}]}]}
    with pytest.raises(ValueError, match="unit"):
        validate_extraction(ExtractedFields.model_validate(raw), text,
            [CriticalField(name="lodging_limit", value=500, unit="CNY", tolerance=0)], require_grounding=True)


@pytest.mark.parametrize("text,value,unit", [("上限为0.05万元。", .05, "万元"),
    ("上限为五百元。", 500, "CNY"), ("上限为500元人民币。", 500, "CNY")])
def test_currency_complete_literal_expression_allows_explicit_conversion(text, value, unit):
    from evals.answer_fields import ExtractedFields, compare_fields, validate_extraction
    from evals.gold_v2_models import CriticalField
    raw = {"fields": [{"name": "lodging_limit", "values": [{"value": value, "unit": unit,
        "conditions": {}, "quote": text, "start": 0, "end": len(text),
        "unit_evidence": [span(text, "万元" if unit == "万元" else "元")]}]}]}
    expected = [CriticalField(name="lodging_limit", value=500, unit="CNY", tolerance=0)]
    extracted = ExtractedFields.model_validate(raw)
    validate_extraction(extracted, text, expected, require_grounding=True)
    assert compare_fields(extracted, expected)["all_critical_fields_pass"] == 1


def test_redundant_unit_explanation_does_not_invalidate_complete_local_proof():
    from evals.answer_fields import ExtractedFields, compare_fields, validate_extraction
    from evals.gold_v2_models import CriticalField
    quote = "每人每晚500元人民币。"
    text = quote + "单位是元人民币（按每人每晚计）。"
    raw = {"fields": [{"name": "lodging_limit", "values": [{"value": 500, "unit": "CNY/person/night",
        "conditions": {}, "quote": quote, "start": 0, "end": len(quote),
        "unit_evidence": [span(text, quote), span(text, "单位是元人民币（按每人每晚计）")]}]}]}
    extracted = ExtractedFields.model_validate(raw)
    expected = [CriticalField(name="lodging_limit", value=500, unit="CNY/person/night", tolerance=0)]
    validate_extraction(extracted, text, expected, require_grounding=True)
    assert compare_fields(extracted, expected)["all_critical_fields_pass"] == 1
    assert len(extracted.fields[0].values[0].unit_evidence) == 2  # source evidence is not deleted


@pytest.mark.parametrize("other", ["其他项目按每人每晚计。", "其他项目每人每晚500元。"])
def test_out_of_scope_unit_explanation_cannot_supply_missing_counting_basis(other):
    from evals.answer_fields import ExtractedFields, validate_extraction
    from evals.gold_v2_models import CriticalField
    quote = "住宿上限500元。"
    text = quote + other
    raw = {"fields": [{"name": "lodging_limit", "values": [{"value": 500, "unit": "CNY/person/night",
        "conditions": {}, "quote": quote, "start": 0, "end": len(quote),
        "unit_evidence": [span(text, "元"), span(text, "每人每晚")]}]}]}
    with pytest.raises(ValueError, match="unit"):
        validate_extraction(ExtractedFields.model_validate(raw), text,
            [CriticalField(name="lodging_limit", value=500, unit="CNY/person/night", tolerance=0)], require_grounding=True)
