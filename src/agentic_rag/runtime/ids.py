"""Identifier helpers for runtime and idempotency contracts."""

import hashlib

from uuid6 import uuid7


def new_id() -> str:
    """Return a time-sortable UUIDv7 string."""
    return str(uuid7())


def content_id(*parts: str) -> str:
    """Return a deterministic, unambiguous SHA-256 digest of string parts."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
