"""Tests for the AST-restricted numeric calculator."""

from __future__ import annotations

import pytest

from agentic_rag.query.calculator import SafeCalculator, UnsafeExpression


def test_calculator_evaluates_numeric_expression_without_eval() -> None:
    assert SafeCalculator().evaluate("(2 + 3) ** 2 // 5") == 5


@pytest.mark.parametrize(
    "expression", [
        "__import__('os').system('id')",
        "name + 1",
        "[1, 2]",
        "'text'",
        "2 ** 10000",
    ],
)
def test_calculator_rejects_unsafe_or_oversized_expressions(expression: str) -> None:
    with pytest.raises(UnsafeExpression):
        SafeCalculator().evaluate(expression)
