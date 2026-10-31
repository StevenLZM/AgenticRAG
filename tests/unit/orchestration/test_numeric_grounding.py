from agentic_rag.orchestration.numeric_grounding import unsupported_numbers


def test_new_difference_needs_a_verified_calculation_even_when_it_is_correct():
    assert unsupported_numbers("30% exceeds 20% by 10 percentage points", ["100 120 200 260"],
                               [{"value": 20, "expression": "(120-100)/100*100"}, {"value": 30, "expression": "(260-200)/200*100"}]) == ["10"]


def test_rounding_and_display_markers_do_not_create_new_numeric_claims():
    assert unsupported_numbers("1. Growth is 33.33% [cite:123456]", ["2025 revenue 3 and 4"],
                               [{"value": 100 / 3, "expression": "(4-3)/3*100"}]) == []


def test_half_up_rounding_and_numbered_markdown_headings_are_supported():
    assert unsupported_numbers("## 1. Result\nRounded value: 2.68", [],
                               [{"value": 2.675, "expression": "5.35/2"}]) == []


def test_decimal_lists_are_not_read_as_thousands_separators():
    assert unsupported_numbers("Inputs: 1,2", ["Inputs: 1 and 2"], []) == []


def test_decimal_literals_without_an_integer_part_still_need_support():
    assert unsupported_numbers("Increase .1 percentage points", [], []) == [".1"]
    assert unsupported_numbers("Change -.1", ["-0.1"], []) == []
