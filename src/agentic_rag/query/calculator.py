"""A deliberately small arithmetic evaluator for research observations."""

from __future__ import annotations

import ast
import math
from typing import cast


class UnsafeExpression(ValueError):
    """Raised for syntax or arithmetic outside the safe numeric subset."""


class SafeCalculator:
    """Evaluate bounded arithmetic without names, calls, attributes, or ``eval``."""

    def __init__(self, *, max_magnitude: float = 1_000_000_000_000) -> None:
        if max_magnitude <= 0:
            raise ValueError("max_magnitude must be positive")
        self._max_magnitude = max_magnitude

    def evaluate(self, expression: str) -> int | float:
        if not isinstance(expression, str) or not expression.strip() or len(expression) > 1_000:
            raise UnsafeExpression("expression must be a bounded non-empty string")
        try:
            parsed = ast.parse(expression, mode="eval")
        except SyntaxError as error:
            raise UnsafeExpression("invalid arithmetic syntax") from error
        return self._number(self._visit(parsed.body))

    def _visit(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return self._number(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = self._visit(node.operand)
            return self._number(operand if isinstance(node.op, ast.UAdd) else -operand)
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
        ):
            left, right = self._visit(node.left), self._visit(node.right)
            try:
                if isinstance(node.op, ast.Add):
                    result = left + right
                elif isinstance(node.op, ast.Sub):
                    result = left - right
                elif isinstance(node.op, ast.Mult):
                    result = left * right
                elif isinstance(node.op, ast.Div):
                    result = left / right
                elif isinstance(node.op, ast.FloorDiv):
                    result = left // right
                elif isinstance(node.op, ast.Mod):
                    result = left % right
                else:
                    if abs(right) > 100 or (abs(left) > 1 and right > 40):
                        raise UnsafeExpression("exponent exceeds calculator bound")
                    result = left**right
            except (ArithmeticError, OverflowError) as error:
                raise UnsafeExpression("arithmetic operation failed") from error
            return self._number(result)
        raise UnsafeExpression("only numeric literals and arithmetic operators are allowed")

    def _number(self, value: object) -> int | float:
        if type(value) not in (int, float):
            raise UnsafeExpression("numeric result exceeds calculator bound")
        number = cast(int | float, value)
        if not math.isfinite(number) or abs(number) > self._max_magnitude:
            raise UnsafeExpression("numeric result exceeds calculator bound")
        return number
