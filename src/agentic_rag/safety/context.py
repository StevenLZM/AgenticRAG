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
    heading_path: tuple[str, ...] = ()

    def render(self) -> str:
        """Return data-only JSON, never instruction-shaped Markdown."""
        payload: dict[str, object] = {
            "trust": "untrusted_data",
            "source": self.source_label,
            "evidence_id": self.evidence_id,
            "content": self.content,
        }
        if self.heading_path:
            payload["heading_path"] = list(self.heading_path)
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
