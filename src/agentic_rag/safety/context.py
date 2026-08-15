"""Serialization boundary for document-derived prompt context."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict


class DataEnvelope(BaseModel):
    """Explicitly mark a rendered document fragment as untrusted data."""

    model_config = ConfigDict(frozen=True)

    source_label: str
    evidence_id: str
    content: str

    def render(self) -> str:
        """Return data-only JSON, never instruction-shaped Markdown."""
        return json.dumps(
            {
                "trust": "untrusted_data",
                "source": self.source_label,
                "evidence_id": self.evidence_id,
                "content": self.content,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
