"""Deterministic safety gate for untrusted document uploads."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import StrEnum
from io import BytesIO
from pathlib import Path
from typing import Protocol
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel, ConfigDict


PDF_MIME = "application/pdf"
TEXT_MIME = "text/plain"
XLS_MIME = "application/vnd.ms-excel"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
ZIP_MIME = "application/zip"
OCTET_STREAM_MIME = "application/octet-stream"

_PDF_SIGNATURE = b"%PDF-"
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_MIME_EXTENSIONS = {
    PDF_MIME: {".pdf"},
    TEXT_MIME: {".txt"},
    XLS_MIME: {".xls"},
    XLSX_MIME: {".xlsx"},
}
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


class UploadSafetyStatus(StrEnum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


class UploadDecision(BaseModel):
    """Complete, immutable result of scanning one original byte sequence."""

    model_config = ConfigDict(frozen=True)

    status: UploadSafetyStatus
    detected_mime: str
    content_hash: str
    reasons: tuple[str, ...] = ()


class UploadSafetyScanner(Protocol):
    def scan(
        self, filename: str, declared_mime: str, content: bytes
    ) -> UploadDecision: ...


def normalize_upload_filename(filename: str) -> str:
    """Return one normalized basename or raise for any path-like client input."""
    normalized = unicodedata.normalize("NFC", filename).strip()
    if (
        not normalized
        or "\x00" in normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or any(unicodedata.category(character) == "Cc" for character in normalized)
    ):
        raise ValueError("upload filename must be one safe basename")
    return normalized


class DefaultUploadSafetyScanner:
    """Validate filename, declared MIME, signature, and plain-text heuristics."""

    def scan(
        self, filename: str, declared_mime: str, content: bytes
    ) -> UploadDecision:
        content_hash = hashlib.sha256(content).hexdigest()
        detected_mime, parsed_text = _detect_mime_and_text(content)
        reasons: list[str] = []

        try:
            normalized_filename = normalize_upload_filename(filename)
        except ValueError:
            normalized_filename = ""
            reasons.append("unsafe_filename")

        normalized_mime = declared_mime.partition(";")[0].strip().lower()
        extension = Path(normalized_filename).suffix.lower()
        allowed_extensions = _MIME_EXTENSIONS.get(normalized_mime)
        if allowed_extensions is None:
            reasons.append("unsupported_mime")
        elif extension not in allowed_extensions:
            reasons.append("mime_extension_mismatch")

        if not content:
            reasons.append("empty_content")
        if detected_mime == ZIP_MIME:
            reasons.append("unsupported_container")
        elif detected_mime == OCTET_STREAM_MIME:
            reasons.append("unsupported_signature")
        if normalized_mime in _MIME_EXTENSIONS and detected_mime != normalized_mime:
            reasons.append("mime_signature_mismatch")

        if reasons:
            return UploadDecision(
                status=UploadSafetyStatus.REJECTED,
                detected_mime=detected_mime,
                content_hash=content_hash,
                reasons=tuple(dict.fromkeys(reasons)),
            )

        quarantine_reasons: list[str] = []
        if parsed_text is not None:
            if any(unicodedata.category(character) == "Cf" for character in parsed_text):
                quarantine_reasons.append("invisible_unicode")
            if any(pattern.search(parsed_text) for pattern in _INSTRUCTION_PATTERNS):
                quarantine_reasons.append("instruction_like_content")

        return UploadDecision(
            status=(
                UploadSafetyStatus.QUARANTINED
                if quarantine_reasons
                else UploadSafetyStatus.ACCEPTED
            ),
            detected_mime=detected_mime,
            content_hash=content_hash,
            reasons=tuple(quarantine_reasons),
        )


def _detect_mime_and_text(content: bytes) -> tuple[str, str | None]:
    if content.startswith(_PDF_SIGNATURE):
        return PDF_MIME, None
    if content.startswith(_OLE_SIGNATURE):
        return XLS_MIME, None
    if content.startswith(_ZIP_SIGNATURES):
        return (
            (XLSX_MIME, None)
            if _is_safe_xlsx(content)
            else (ZIP_MIME, None)
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return OCTET_STREAM_MIME, None
    if any(
        unicodedata.category(character) == "Cc" and character not in "\t\n\r"
        for character in text
    ):
        return OCTET_STREAM_MIME, None
    return TEXT_MIME, text


def _is_safe_xlsx(content: bytes) -> bool:
    try:
        with ZipFile(BytesIO(content)) as workbook:
            entries = workbook.infolist()
            names = {entry.filename for entry in entries}
            if not {"[Content_Types].xml", "xl/workbook.xml"}.issubset(names):
                return False
            if any(
                entry.flag_bits & 0x1
                or entry.file_size > 100 * 1024 * 1024
                or ".." in Path(entry.filename).parts
                or entry.filename.startswith(("/", "\\"))
                or entry.filename.lower().endswith(
                    (".zip", ".7z", ".rar", ".tar", ".gz", ".xlsm", ".bin")
                )
                for entry in entries
            ):
                return False
            total_compressed = sum(max(entry.compress_size, 1) for entry in entries)
            total_uncompressed = sum(entry.file_size for entry in entries)
            return total_uncompressed <= 100 * 1024 * 1024 and (
                total_uncompressed / total_compressed <= 1000
            )
    except (BadZipFile, OSError, ValueError):
        return False
