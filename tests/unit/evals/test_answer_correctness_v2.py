import json

import pytest

from evals.answer_fields import ExtractedFields, compare_fields, extraction_payload, validate_extraction
from evals.gold_v2_models import CriticalField


def expected(**kwargs):
    return [CriticalField(name="lodging_limit", value=470, unit="CNY/person/night", tolerance=0,
                          conditions=kwargs)]


def extraction(answer, value=470, unit="CNY/person/night", conditions=None):
    return ExtractedFields.model_validate({"fields": [{"name": "lodging_limit", "values": [
        {"value": value, "unit": unit, "conditions": conditions or {}, "quote": answer, "start": 0, "end": len(answer)}]}]})


def test_blind_payload_contains_no_expected_value_or_reference():
    payload = extraction_payload("上海每人每晚上限？", "470元/人/晚", expected(year=2026))
    assert "470" not in json.dumps(payload["field_specs"])
    assert "reference" not in payload
    assert "value" not in payload["field_specs"][0]


@pytest.mark.parametrize("value,unit,conditions,status", [
    (410, "CNY/person/night", {"year": 2026}, "incorrect_value"),
    (470, "CNY/person/night", {"year": 2025}, "incorrect_conditions"),
    (470, None, {"year": 2026}, "missing_unit"),
    (470, "device", {"year": 2026}, "incorrect_unit"),
    (470, "CNY/person/night", {"year": 2026}, "pass"),
])
def test_amount_year_unit_checked_independently(value, unit, conditions, status):
    result = compare_fields(extraction(f"{value}元", value, unit, conditions), expected(year=2026))
    assert result["fields"][0]["status"] == status
    assert result["all_critical_fields_pass"] == float(status == "pass")


def test_omitted_and_contradictory_are_not_partial_task_success():
    empty = ExtractedFields.model_validate({"fields": [{"name": "lodging_limit", "values": []}]})
    assert compare_fields(empty, expected())["fields"][0]["status"] == "missing"
    first = extraction("470元").model_dump()
    first["fields"][0]["values"].append(extraction("410元", 410).model_dump()["fields"][0]["values"][0])
    assert compare_fields(ExtractedFields.model_validate(first), expected())["fields"][0]["status"] == "contradiction"


def test_conversion_and_grounding():
    ex = extraction("0.047万元/人/晚", .047, "万元/人/晚")
    validate_extraction(ex, "0.047万元/人/晚", expected())
    assert compare_fields(ex, expected())["critical_field_accuracy"] == 1
    with pytest.raises(ValueError, match="quote"):
        validate_extraction(ex, "410元", expected())
    with pytest.raises(ValueError, match="number"):
        validate_extraction(extraction("410元", 470), "410元", expected())


def test_nonfinite_and_unexpected_keys_rejected():
    with pytest.raises(ValueError):
        extraction("NaN", float("nan"))
    with pytest.raises(ValueError):
        ExtractedFields.model_validate({"fields": [], "reasoning": "private"})
