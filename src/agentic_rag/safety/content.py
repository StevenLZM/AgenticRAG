"""Safety scan over canonical extracted and OCR text without mutation."""

from __future__ import annotations

import re
import unicodedata

from agentic_rag.ingestion.assembler import CanonicalAst
from agentic_rag.safety.uploads import UploadDecision, UploadSafetyStatus


_INSTRUCTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|messages?)\b",
        r"\b(?:reveal|print|show|repeat)\s+(?:the\s+)?system\s+prompt\b",
        r"\b(?:system\s+prompt|developer\s+message)\b",
        r"\b(?:call|invoke|use)\s+(?:the\s+)?(?:tool|function)\b",
        r"\byou\s+are\s+(?:chatgpt|an?\s+assistant)\b",
    )
)


class ContentSafetyScanner:
    """Quarantine suspicious canonical content and preserve it for review."""

    def scan(self, canonical: CanonicalAst) -> UploadDecision:
        text = "\n".join(block.text for block in canonical.text_blocks)
        reasons: list[str] = []
        if not text.strip():
            reasons.append("no_retrievable_text")
        if any(unicodedata.category(character) == "Cf" for character in text):
            reasons.append("invisible_unicode")
        if any(pattern.search(text) for pattern in _INSTRUCTION_PATTERNS):
            reasons.append("instruction_like_content")
        return UploadDecision(
            status=(
                UploadSafetyStatus.QUARANTINED
                if reasons
                else UploadSafetyStatus.ACCEPTED
            ),
            detected_mime=_source_mime(canonical.envelope.source_type),
            content_hash=canonical.envelope.content_hash,
            reasons=tuple(reasons),
        )


def _source_mime(source_type: str) -> str:
    if source_type in {"pdf", "scanned_pdf"}:
        return "application/pdf"
    if source_type == "text":
        return "text/plain"
    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
