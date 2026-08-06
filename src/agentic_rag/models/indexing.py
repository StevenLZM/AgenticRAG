"""Shared validation for physical index-generation identifiers."""

from __future__ import annotations

import re


_INDEX_GENERATION = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def validate_index_generation(value: str) -> str:
    """Require the caller to provide the one canonical ES-safe identifier."""
    if _INDEX_GENERATION.fullmatch(value) is None:
        raise ValueError(
            "index_generation must already be a lowercase ES-safe version label"
        )
    return value
