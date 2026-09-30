"""Shared validation for physical index-generation identifiers."""

from __future__ import annotations

import re


_INDEX_GENERATION = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
DEFAULT_INDEX_GENERATION = "index-v3"
LEGACY_INDEX_GENERATIONS = frozenset({"index-v1"})
ACTIVE_CHILD_INDEX_ALIAS = "agenticrag-children-active"


def validate_index_generation(value: str) -> str:
    """Require the caller to provide the one canonical ES-safe identifier."""
    if _INDEX_GENERATION.fullmatch(value) is None:
        raise ValueError(
            "index_generation must already be a lowercase ES-safe version label"
        )
    return value


def validate_writable_index_generation(value: str) -> str:
    """Reject legacy generations that the current Child schema cannot create."""
    validate_index_generation(value)
    if value in LEGACY_INDEX_GENERATIONS:
        raise ValueError(
            f"index_generation {value!r} is legacy; use {DEFAULT_INDEX_GENERATION!r}"
        )
    return value
