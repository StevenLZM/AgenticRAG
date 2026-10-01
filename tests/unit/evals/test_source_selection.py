"""The model selects a numbered clause; only Python computes character offsets."""
import pytest


def selected_fact(evidence, **updates):
    return {"fact_id": 0, "evidence_segment_ids": evidence, "subject_spans": [], "predicate_spans": [],
            "value_spans": [], "unit_spans": [], "condition_spans": [], "polarity_spans": [],
            "coreference_links": [], **updates}


def ref(identity, quote):
    return {"span_id": identity, "quote": quote}


def test_numbered_clauses_disambiguate_repeated_units_without_model_offsets():
    from evals.source_selection import source_catalog, SpanSelection, resolve_selection
    answer = "😀旧值470元，新值530元，增加60元。"
    catalog = source_catalog("增加多少？", answer)
    got = resolve_selection(SpanSelection.model_validate(ref("r2", "元")), catalog)
    assert (got.source, got.start, got.end, got.quote) == ("response", 19, 20, "元")


@pytest.mark.parametrize("selection,reason", [
    (ref("r99", "元"), "unknown_span_id"), (ref("r0", "700元"), "selection_quote_missing"),
    (ref("r0", "元"), "selection_quote_ambiguous")])
def test_selection_never_guesses_id_or_quote_or_same_clause_occurrence(selection, reason):
    from evals.source_selection import source_catalog, SpanSelection, resolve_selection
    with pytest.raises(ValueError, match=reason):
        resolve_selection(SpanSelection.model_validate(selection), source_catalog("q", "500元和600元。"))


def test_catalog_preserves_thousands_decimals_whitespace_and_unicode():
    from evals.source_selection import source_catalog
    catalog = source_catalog("问？", "  😀1,500.25元，次项2元。\n")
    assert [(s.start, s.end, s.quote) for k, s in catalog.items() if k.startswith("r")] == [
        (0, 13, "  😀1,500.25元，"), (13, 18, "次项2元。")]


def test_overlapping_quotes_remain_ambiguous():
    from evals.source_selection import source_catalog, SpanSelection, resolve_selection
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_selection(SpanSelection.model_validate(ref("r0", "哈哈")), source_catalog("", "哈哈哈"))


def test_selected_roles_cannot_borrow_value_from_question_or_another_fact():
    from evals.source_selection import SelectedExtraction, source_catalog, resolve_facts
    answer = "A公司500元。B公司600元。"
    for selection in (ref("q0", "700元"), ref("r1", "600元")):
        raw = {"facts": [selected_fact([0], value_spans=[selection])],
               "segments": [{"segment_id": 0, "factual": True}, {"segment_id": 1, "factual": False}]}
        with pytest.raises(ValueError, match="response_only|outside_evidence"):
            resolve_facts("700元？", answer, SelectedExtraction.model_validate(raw), source_catalog("700元？", answer))


def test_selected_conditions_and_negation_survive_as_exact_originals():
    from evals.source_selection import SelectedExtraction, source_catalog, resolve_facts
    answer = "不是500元，仅深圳适用600元。"
    raw = {"facts": [selected_fact([0],
        polarity_spans=[ref("r0", "不是")], condition_spans=[ref("r1", "仅深圳")])],
        "segments": [{"segment_id": 0, "factual": True}]}
    result = resolve_facts("q", answer, SelectedExtraction.model_validate(raw), source_catalog("q", answer))
    assert result.facts[0].polarity_spans[0].quote == "不是"
    assert result.facts[0].condition_spans[0].quote == "仅深圳"


def test_shared_subject_across_clauses_uses_original_sentence_evidence():
    from evals.source_selection import SelectedExtraction, source_catalog, resolve_facts
    answer = "A公司2025年上限470元，2026年为530元。"
    raw = {"facts": [selected_fact([0], subject_spans=[ref("r0", "A公司")],
        value_spans=[ref("r1", "530元")], condition_spans=[ref("r1", "2026年")])],
        "segments": [{"segment_id": 0, "factual": True}]}
    result = resolve_facts("q", answer, SelectedExtraction.model_validate(raw), source_catalog("q", answer))
    assert result.facts[0].evidence_spans[0].quote == answer
    assert result.facts[0].value_spans[0].quote == "530元"


@pytest.mark.parametrize("identities", [[1], [0, 0], [-1], [True]])
def test_sentence_evidence_rejects_foreign_duplicate_and_invalid_ids(identities):
    from evals.source_selection import SelectedExtraction, source_catalog, resolve_facts
    answer = "500元。"
    with pytest.raises(ValueError):
        raw = SelectedExtraction.model_validate({"facts": [selected_fact(identities)],
            "segments": [{"segment_id": 0, "factual": True}]})
        resolve_facts("q", answer, raw, source_catalog("q", answer))


def test_selected_fields_bind_increase_and_mandatory_unit_provenance():
    from evals.source_selection import SelectedFields, source_catalog, resolve_fields
    from evals.answer_fields import validate_extraction, compare_fields
    from evals.gold_v2_models import CriticalField
    answer = "每人每晚旧470元，新530元，增加60元。"
    raw = {"fields": [{"name": "increase", "values": [{"value": 60, "unit": "CNY/person/night", "conditions": {},
        "evidence": ref("r2", "增加60元。"), "unit_evidence": [ref("r2", "元"), ref("r0", "每人每晚")],
        "condition_evidence": {}}]}]}
    expected = [CriticalField(name="increase", value=60, unit="CNY/person/night", tolerance=0)]
    result = resolve_fields(SelectedFields.model_validate(raw), source_catalog("q", answer))
    validate_extraction(result, answer, expected, question="q", require_grounding=True)
    assert compare_fields(result, expected)["all_critical_fields_pass"] == 1
    raw["fields"][0]["values"][0].pop("unit_evidence")
    with pytest.raises(ValueError):
        SelectedFields.model_validate(raw)
