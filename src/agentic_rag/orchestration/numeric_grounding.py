"""Reject new decimal literals absent from sources or verified arithmetic.

This complements semantic assessment; it does not prove subject/unit attribution
or parse numbers written out as words.
"""
import re
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any

NUMBER = re.compile(r"(?<![0-9A-Za-z_.])-?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?![0-9A-Za-z_])")


def evidence_text(evidence: Mapping[str, Any]) -> str:
    """Use the same authorized heading context as operand verification."""
    return " ".join((evidence["content"], *evidence.get("heading_path", ())))


def rounded_value(value: Decimal, places: int) -> Decimal:
    with localcontext() as context:
        context.prec = max(28, len(value.as_tuple().digits) + abs(int(value.as_tuple().exponent)) + places + 1)
        return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def unsupported_numbers(text: str, sources: Sequence[str], calculations: Sequence[Mapping[str, Any]]) -> list[str]:
    source_texts = [*sources, *(str(c.get("expression", "")) for c in calculations)]
    known = {Decimal(n.replace(",", "")) for source in source_texts for n in NUMBER.findall(source)}
    values = [Decimal(str(c["value"])) for c in calculations if isinstance(c.get("value"), (int, float))]
    # Citation IDs and ordered-list markers are presentation, not claims.
    text = re.sub(r"\[cite:[^\]]+\]", "", text)
    text = re.sub(r"(?m)^([ \t]*(?:#{1,6}[ \t]+)?)\d+[.)][ \t]+", r"\1", text)
    unknown = []
    for literal in NUMBER.findall(text):
        number = Decimal(literal.replace(",", ""))
        places = max(0, -int(number.normalize().as_tuple().exponent))
        if number not in known and not any(number == rounded_value(value, places) for value in values):
            unknown.append(literal)
    return sorted(set(unknown))
